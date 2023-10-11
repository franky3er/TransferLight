from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
import os.path
from pathlib import Path
import sys
from typing import Any, List, Dict, Tuple

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import orthogonal_
from torch_geometric.data import HeteroData, Batch
from torch_geometric.data.data import BaseData
from torch_geometric.utils import softmax

from src.data.replay_buffer import ReplayBuffer
from src.modules.distributions import GroupCategorical
from src.modules.networks import Network
from src.modules.utils import group_argmax
from src import params
from src.rl.environments import MarlEnvironment, MultiprocessingMarlEnvironment, MPMarlEnvMetaData, MarlEnvMetaData


@dataclass
class AgentRecord:
    scenario: str
    intersection: str
    action_seq: str
    mc_means_seq: str
    mc_stds_seq: str


class Agent(nn.Module):

    def __init__(self):
        super(Agent, self).__init__()
        self.records = defaultdict(lambda: dict())

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, Agent)
        return obj

    @abstractmethod
    @torch.no_grad()
    def act(self, state: Any) -> List[int]:
        pass

    @abstractmethod
    def fit(self, environment: MultiprocessingMarlEnvironment, steps: int = 1_000, skip_steps: int = 100,
            checkpoint_path: str = None):
        pass

    @abstractmethod
    def test(self, environment: MultiprocessingMarlEnvironment, checkpoint_path: str = None, stats_dir: str = None):
        pass

    @abstractmethod
    def demo(self, environment: MarlEnvironment, checkpoint_path: str = None):
        pass

    @staticmethod
    def _init_params(module):
        if isinstance(module, nn.Linear):
            orthogonal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.1)

    def _update_stats(self, actions: List[int], mc_means: List[float], mc_stds: List[float],
                      metadata: MPMarlEnvMetaData):
        agent_offsets = [0] + metadata.agent_offsets
        action_offsets = [0] + metadata.action_offsets
        for rank in range(metadata.n_workers):
            worker_mc_means, worker_mc_stds = None, None
            worker_actions = actions[agent_offsets[rank]:agent_offsets[rank + 1]]
            if mc_means is not None:
                worker_mc_means = mc_means[action_offsets[rank]:action_offsets[rank + 1]]
            if mc_stds is not None:
                worker_mc_stds = mc_stds[action_offsets[rank]:action_offsets[rank + 1]]
            worker_metadata = metadata.workers_metadata[rank]
            self._update_worker_stats(worker_actions, worker_mc_means, worker_mc_stds, worker_metadata)

    def _update_worker_stats(self, actions: List[int], mc_means: List[float], mc_stds: List[float],
                             metadata: MarlEnvMetaData):
        action_offsets = [0] + metadata.action_offsets
        scenario = metadata.scenario_name
        for rank in range(metadata.n_agents):
            intersection = metadata.agents_metadata[rank].name
            agent_mc_means = ""
            if mc_means is not None:
                agent_mc_means = mc_means[action_offsets[rank]:action_offsets[rank + 1]]
                agent_mc_means = "/".join([str(mean) for mean in agent_mc_means])
            agent_mc_stds = ""
            if mc_stds is not None:
                agent_mc_stds = mc_stds[action_offsets[rank]:action_offsets[rank + 1]]
                agent_mc_stds = "/".join([str(std) for std in agent_mc_stds])
            if intersection not in self.records[scenario].keys():
                self.records[scenario][intersection] = AgentRecord(
                    scenario=scenario,
                    intersection=intersection,
                    action_seq=str(actions[rank]),
                    mc_means_seq=agent_mc_means,
                    mc_stds_seq=agent_mc_stds
                )
            else:
                record = self.records[scenario][intersection]
                record.action_seq = "|".join([record.action_seq, str(actions[rank])])
                record.mc_means_seq = "|".join([record.mc_means_seq, agent_mc_means])
                record.mc_stds_seq = "|".join([record.mc_stds_seq, agent_mc_stds])
                self.records[scenario][intersection] = record

    def _store_stats(self, stats_dir: str):
        for scenario, records in self.records.items():
            stats_path = os.path.join(stats_dir, f"{scenario}.agent.csv")
            os.makedirs(str(Path(stats_path).parent), exist_ok=True)
            stats = pd.DataFrame(records.values())
            stats.to_csv(stats_path, index=False)


class DefaultAgent(Agent):

    def act(self, state: Any) -> List[int]:
        return []

    @torch.no_grad()
    def fit(self, environment: MultiprocessingMarlEnvironment, steps: int = 1_000, skip_steps: int = 100,
            checkpoint_path: str = None):
        _ = environment.reset()
        while environment.total_step < steps:
            while True:
                state = environment.state()
                actions = self.act(state)
                _, _, done = environment.step(actions)
                info = environment.info()
                if environment.total_step >= skip_steps:
                    print(info["progress"])
                if done:
                    break

    @torch.no_grad()
    def test(self, environment: MultiprocessingMarlEnvironment, checkpoint_path: str = None, stats_dir: str = None):
        state = environment.reset()
        while not environment.all_done():
            actions = self.act(state)
            state, _, done = environment.step(actions)
            print(environment.info()["progress"])
        environment.close()

    @torch.no_grad()
    def demo(self, environment: MarlEnvironment, checkpoint_path: str = None):
        state = environment.reset()
        while True:
            actions = self.act(state)
            state, _, done = environment.step(actions)
            if done:
                break
        environment.close()


class RandomAgent(Agent):

    def act(self, state: BaseData) -> List[int]:
        index = state["phase", "to", "intersection"].edge_index[1].to(params.DEVICE)
        logits = torch.zeros_like(state["phase"].x.to(params.DEVICE)).squeeze()
        actions = GroupCategorical(logits=logits, index=index).sample(return_indices=False)
        return actions.cpu().numpy().tolist()

    @torch.no_grad()
    def fit(self, environment: MultiprocessingMarlEnvironment, steps: int = 1_000, skip_steps: int = 100,
            checkpoint_path: str = None):
        _ = environment.reset()
        while environment.total_step < steps:
            while True:
                state = environment.state()
                actions = self.act(state)
                _, _, done = environment.step(actions)
                info = environment.info()
                if environment.total_step >= skip_steps:
                    print(info["progress"])
                if done:
                    break

    @torch.no_grad()
    def test(self, environment: MultiprocessingMarlEnvironment, checkpoint_path: str = None, stats_dir: str = None):
        _ = environment.reset()
        while not environment.all_done():
            state = environment.state()
            actions = self.act(state)
            _, _, _ = environment.step(actions)
            print(environment.info()["progress"])
        environment.close()

    @torch.no_grad()
    def demo(self, environment: MarlEnvironment, checkpoint_path: str = None):
        state = environment.reset()
        while True:
            actions = self.act(state)
            state, _, done = environment.step(actions)
            if done:
                break
        environment.close()


class MaxPressure(Agent):

    @torch.no_grad()
    def act(self, state: BaseData, act_random: bool = False) -> List[int]:
        state = state.clone().to(params.DEVICE)
        pressures = state["phase"].x#.squeeze()
        index = state["phase", "to", "intersection"].edge_index[1].to(params.DEVICE)
        if act_random:
            actions = GroupCategorical(logits=torch.zeros_like(pressures), index=index).sample(return_indices=False)
        else:
            actions = group_argmax(pressures, index)
        return actions.cpu().numpy().tolist()

    @torch.no_grad()
    def fit(self, environment: MultiprocessingMarlEnvironment, steps: int = 1_000, skip_steps: int = 100,
            checkpoint_dir: str = None):
        _ = environment.reset()
        while environment.total_step < steps:
            while True:
                state = environment.state()
                actions = self.act(state, act_random=environment.total_step < skip_steps)
                _, _, done = environment.step(actions)
                info = environment.info()
                if environment.total_step >= skip_steps:
                    print(info["progress"])
                if done:
                    break
        environment.close()

    @torch.no_grad()
    def test(self, environment: MultiprocessingMarlEnvironment, checkpoint_path: str = None, stats_dir: str = None):
        _ = environment.reset()
        while not environment.all_done():
            metadata = environment.metadata()
            state = environment.state()
            actions = self.act(state)
            _, _, _ = environment.step(actions)
            self._update_stats(actions, None, None, metadata)
            print(environment.info()["progress"])
        environment.close()
        self._store_stats(stats_dir)

    @torch.no_grad()
    def demo(self, environment: MarlEnvironment, checkpoint_path: str = None):
        state = environment.reset()
        while True:
            actions = self.act(state)
            state, _, done = environment.step(actions)
            if done:
                break
        environment.close()


class DQN(Agent):

    def __init__(
            self,
            network: Dict,
            discount_factor: float,
            learning_rate: float,
            mixing_factor: float,
            epsilon_greedy_prob: float,
            replay_buffer_size: int,
            batch_size: int,
            update_steps: int = 1
    ):
        super(DQN, self).__init__()
        self.online_q_network = Network.create(network["class_name"], network["init_args"])
        self.target_q_network = Network.create(network["class_name"], network["init_args"]).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online_q_network.parameters(), learning_rate)
        self.gamma = discount_factor
        self.tau = mixing_factor
        self.eps = epsilon_greedy_prob
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size, batch_size)
        self.update_steps = update_steps
        self.apply(self._init_params)
        self.n_workers = None
        self.worker_agent_offsets = None
        self.worker_action_offsets = None
        self.to(params.DEVICE)

    def forward(self, state: HeteroData) -> Tuple[torch.Tensor, torch.LongTensor]:
        action_values, index = self.online_q_network(state)
        return action_values, index

    def act(self, state: HeteroData, eps: float = 0.0, mc_samples: int = 1) -> \
            Tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_values_samples, agent_index = [], None
        for _ in range(mc_samples):
            action_values, agent_index = self.forward(state)
            action_values_samples.append(action_values.unsqueeze(-1))
        action_values_samples = torch.cat(action_values_samples, dim=-1)
        action_values_mean = torch.mean(action_values_samples, dim=-1)
        action_values_std = torch.std(action_values_samples, dim=-1) if mc_samples > 1 \
            else torch.zeros_like(action_values_mean)
        greedy_actions, greedy_action_indices = group_argmax(action_values_mean, agent_index, return_indices=True)
        random_actions, random_action_indices = \
            GroupCategorical(logits=torch.ones_like(action_values_mean), index=agent_index).sample(return_indices=True)
        act_greedy = torch.rand_like(greedy_actions.to(torch.float32)) > eps
        actions = act_greedy * greedy_actions + ~act_greedy * random_actions
        actions_index = act_greedy * greedy_action_indices + ~act_greedy * random_action_indices
        actions_bool_index = torch.zeros_like(action_values_mean)
        actions_bool_index[actions_index] = 1.0
        actions_bool_index = actions_bool_index.to(torch.bool)
        return actions.cpu().tolist(), actions_index.cpu(), actions_bool_index.cpu(), action_values_mean.cpu(), \
            action_values_std.cpu(), agent_index.cpu()

    def fit(self, environment: MultiprocessingMarlEnvironment, steps: int = 1_000, skip_steps: int = 100,
            checkpoint_dir: str = None):
        self.train()
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        highest_ema_reward = - sys.float_info.max
        _ = environment.reset()
        while environment.total_step <= steps + skip_steps:
            metadata = environment.metadata()
            self.n_workers = metadata.n_workers
            self.worker_agent_offsets = metadata.agent_offsets
            self.worker_action_offsets = metadata.action_offsets
            states = environment.state()
            actions, _, actions_bool_index, _, _, _ = self.act(states.to(params.DEVICE), eps=self.eps)

            next_states, rewards, done = environment.step(actions)
            if environment.total_step < skip_steps:
                continue

            self._update_replay_buffer(states, actions_bool_index, rewards, next_states)

            if len(self.replay_buffer) >= self.batch_size:
                self._train_step()

            info = environment.info()
            print(info["progress"])
            if checkpoint_dir is not None and environment.total_step % 100 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"{environment.total_step}.pt")
                torch.save(self.state_dict(), checkpoint_path)
            if checkpoint_dir is not None and highest_ema_reward < environment.ema_reward:
                print(f"New best..")
                highest_ema_reward = environment.ema_reward
                checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
                torch.save(self.state_dict(), checkpoint_path)
        environment.close()

    def _update_replay_buffer(self, states: Batch, actions_bool_index: torch.BoolTensor, rewards: torch.Tensor,
                              next_states: Batch):
        states = [states.get_example(i) for i in range(self.n_workers)]
        actions_bool_index = [actions_bool_index[start:end] for start, end in
                              [(0 if i == 0 else self.worker_action_offsets[i-1], self.worker_action_offsets[i])
                               for i in range(len(self.worker_action_offsets))]]
        rewards = [rewards[start:end] for start, end in
                   [(0 if i == 0 else self.worker_agent_offsets[i - 1], self.worker_agent_offsets[i])
                    for i in range(len(self.worker_agent_offsets))]]
        next_states = [next_states.get_example(i) for i in range(self.n_workers)]
        self.replay_buffer.add(states, actions_bool_index, rewards, next_states)

    def _train_step(self):
        for _ in range(self.update_steps):
            states, actions_bool_index, rewards, next_states = self.replay_buffer.sample()
            q_predictions = self._compute_q_predictions(states.to(params.DEVICE), actions_bool_index.to(params.DEVICE))
            q_targets = self._compute_q_targets(rewards.to(params.DEVICE), next_states.to(params.DEVICE))
            loss = F.mse_loss(q_predictions, q_targets)
            self._optimization_step(loss)
        self._update_target_q_network()

    def _compute_q_predictions(self, states: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
        action_values, _ = self.online_q_network(states)
        q_predictions = action_values[action_indices]
        return q_predictions

    @torch.no_grad()
    def _compute_q_targets(self, rewards: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        # Use Double Q-Learning Target
        action_values_local, index = self.online_q_network(next_states)
        action_values_target, _ = self.target_q_network(next_states)
        _, next_greedy_action_indices = group_argmax(action_values_local, index, return_indices=True)
        next_greedy_action_values = action_values_target[next_greedy_action_indices]
        q_targets = rewards + self.gamma * next_greedy_action_values
        return q_targets.detach()

    def _optimization_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _update_target_q_network(self):
        # Polyak averaging to update the parameters of the target network
        for target_param, local_param in zip(self.target_q_network.parameters(), self.online_q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    @torch.no_grad()
    def test(self, environment: MultiprocessingMarlEnvironment, checkpoint_path: str = None, stats_dir: str = None):
        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path, map_location=params.DEVICE))
        _ = environment.reset()
        while not environment.all_done():
            metadata = environment.metadata()
            state = environment.state().to(params.DEVICE)
            actions, _, _, action_value_means, action_value_stds, _ = self.act(state, mc_samples=100)
            _ = environment.step(actions)
            action_value_means = action_value_means.cpu().numpy().tolist()
            action_value_stds = action_value_stds.cpu().numpy().tolist()
            self._update_stats(actions, action_value_means, action_value_stds, metadata)
            print(environment.info()["progress"])
        environment.close()
        self._store_stats(stats_dir)

    @torch.no_grad()
    def demo(self, environment: MarlEnvironment, checkpoint_path: str = None):
        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path, map_location=params.DEVICE))
        state = environment.reset()
        while True:
            state = state.to(params.DEVICE)
            actions, _, _, _, _, _ = self.act(state, mc_samples=10)
            state, _, done = environment.step(actions)
            if done:
                break
        environment.close()


class A2C(Agent):

    def __init__(
            self,
            network: Dict,
            discount_factor: float = 0.9,
            learning_rate: float = 0.001,
            actor_loss_weight: float = 1.0,
            critic_loss_weight: float = 1.0,
            entropy_loss_weight: float = 0.0,
            gradient_clipping_max_norm: float = 1.0
    ):
        super(A2C, self).__init__()
        self.network = Network.create(network["class_name"], network["init_args"])
        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)

        self.gamma = discount_factor
        self.actor_loss_weight = actor_loss_weight
        self.critic_loss_weight = critic_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.gradient_clipping_max_norm = gradient_clipping_max_norm

        self.apply(self._init_params)
        self.to(params.DEVICE)

    def forward(self, state: BaseData) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        values, logits, index = self.network(state)
        return values, logits, index

    def full_path(self, state: BaseData) -> Tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor]:
        values, logits, index = self.forward(state)
        distribution = GroupCategorical(logits=logits, index=index)
        actions, action_indices = distribution.sample(return_indices=True)
        log_probs = distribution.log_prob(action_indices)
        entropy = distribution.entropy()
        return actions.cpu().numpy().tolist(), log_probs, values, entropy

    def actor_path(self, state: BaseData):
        actions, log_probs, _, entropy = self.full_path(state)
        return actions, log_probs, entropy

    def critic_path(self, state: BaseData) -> torch.Tensor:
        values = self.forward(state)[0]
        return values

    def act(self, state: BaseData, greedy: bool = False, mc_samples: int = 1) -> \
            Tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        probs_samples, agent_index = [], None
        for _ in range(mc_samples):
            _, logits, agent_index = self.forward(state)
            probs = softmax(logits, agent_index)
            probs_samples.append(probs.unsqueeze(-1))
        probs_samples = torch.cat(probs_samples, -1)
        probs_mean = torch.mean(probs_samples, -1)
        probs_std = torch.std(probs_samples, -1) if mc_samples > 1 else torch.zeros_like(probs_mean)
        distribution = GroupCategorical(probs=probs_mean, index=agent_index)
        actions, actions_index = group_argmax(probs_mean, agent_index, return_indices=True) if greedy \
            else distribution.sample(return_indices=True)
        actions = actions.cpu().numpy().tolist()
        return actions, actions_index, probs_mean, probs_std, agent_index

    def fit(self, environment: MultiprocessingMarlEnvironment, steps: int = 1_000, skip_steps: int = 100,
            checkpoint_dir: str = None):
        self.train()
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        _ = environment.reset()
        highest_ema_reward = - sys.float_info.max
        while environment.total_step <= steps + skip_steps:
            states = environment.state()
            states = states.to(params.DEVICE)
            actions, log_prob_actions, values, entropies = self.full_path(states)
            next_states, rewards, done = environment.step(actions)
            if environment.total_step < skip_steps:
                continue
            print("Entropy: ", torch.mean(entropies).item())
            next_states, rewards = next_states.to(params.DEVICE), rewards.to(params.DEVICE)
            next_values = self.critic_path(next_states)

            action_values = rewards + self.gamma * next_values
            advantages = action_values.detach() - values

            critic_loss = 0.5 * torch.mean(advantages ** 2)
            actor_loss = - torch.mean(advantages.detach() * log_prob_actions + self.entropy_loss_weight * entropies)

            self.optimization_step(actor_loss, critic_loss)

            info = environment.info()
            print(info["progress"])
            if checkpoint_dir is not None and environment.total_step % 100 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"{environment.total_step}.pt")
                torch.save(self.state_dict(), checkpoint_path)
            if checkpoint_dir is not None and highest_ema_reward < environment.ema_reward:
                print(f"New best..")
                highest_ema_reward = environment.ema_reward
                checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
                torch.save(self.state_dict(), checkpoint_path)

        environment.close()

    def optimization_step(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor):
        loss = self.actor_loss_weight * actor_loss + self.critic_loss_weight * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clipping_max_norm)
        self.optimizer.step()

    def _shared_optimization_step(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor):
        loss = self.actor_loss_weight * actor_loss + self.critic_loss_weight * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clipping_max_norm)
        self.optimizer.step()

    def _unshared_optimization_step(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor):
        actor_loss = self.actor_loss_weight * actor_loss
        critic_loss = self.critic_loss_weight * critic_loss

        self.optimizer_c.zero_grad()
        self.optimizer_a.zero_grad()
        critic_loss.backward()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.network_a.parameters()) + list(self.actor_head.parameters()),
                                       self.gradient_clipping_max_norm)
        torch.nn.utils.clip_grad_norm_(list(self.network_c.parameters()) + list(self.critic_head.parameters()),
                                       self.gradient_clipping_max_norm)
        self.optimizer_c.episode_time()
        self.optimizer_a.episode_time()

    @torch.no_grad()
    def test(self, environment: MultiprocessingMarlEnvironment, checkpoint_path: str = None, stats_dir: str = None):
        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path, map_location=params.DEVICE))
        state = environment.reset()
        while not environment.all_done():
            metadata = environment.metadata()
            state = environment.state().to(params.DEVICE)
            actions, _, probs_mean, probs_std, _ = self.act(state, greedy=True, mc_samples=100)
            _ = environment.step(actions)
            probs_mean = probs_mean.cpu().numpy().tolist()
            probs_std = probs_std.cpu().numpy().tolist()
            self._update_stats(actions, probs_mean, probs_std, metadata)
            print(environment.info()["progress"])
        environment.close()
        self._store_stats(stats_dir)

    @torch.no_grad()
    def demo(self, environment: MarlEnvironment, checkpoint_path: str = None):
        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path, map_location=params.DEVICE))
        state = environment.reset()
        while True:
            state = state.to(params.DEVICE)
            actions = self.act(state, greedy=True, mc_samples=10)[0]
            state, _, done = environment.step(actions)
            if done:
                break
        environment.close()
