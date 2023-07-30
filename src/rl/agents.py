from abc import abstractmethod
import sys
from typing import Any, List, Dict, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import orthogonal_
from torch_geometric.data import HeteroData, Batch
from torch_geometric.data.data import BaseData
from torch_geometric.utils import softmax

from src.data.replay_buffer import ReplayBuffer
from src.modules.distributions import GroupCategorical
from src.modules.network_bodies import NetworkBody
from src.modules.network_heads import NetworkHead
from src.modules.base_modules import MultiInputSequential
from src.modules.utils import group_argmax, group_sum
from src.params import ENV_ACTION_EXECUTION_TIME
from src.rl.environments import MarlEnvironment, MultiprocessingMarlEnvironment
from src.rl.exploration import ExpDecayEpsGreedyStrategy

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Use {device} device")


class IndependentAgents(nn.Module):

    @abstractmethod
    @torch.no_grad()
    def act(self, state: Any) -> List[int]:
        pass

    @abstractmethod
    def fit(self, environment: MarlEnvironment, episodes: int = 100, checkpoint_path: str = None):
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


class MaxPressure(IndependentAgents):

    def __init__(self, min_phase_duration: int):
        super(MaxPressure, self).__init__()
        self.min_phase_steps = min_phase_duration // ENV_ACTION_EXECUTION_TIME
        self.initialized = False
        self.actions = None
        self.action_durations = None

    def act(self, state: HeteroData) -> List[int]:
        x = state["phase"].x.squeeze().to(device)
        index = state["phase", "to", "intersection"].edge_index[1].to(device)
        max_pressure_actions = group_argmax(x, index)
        if self.initialized:
            action_change = torch.logical_and(self.action_durations >= self.min_phase_steps,
                                              self.actions != max_pressure_actions)
            self.actions = action_change * max_pressure_actions + ~action_change * self.actions
            self.action_durations = torch.zeros_like(self.actions) + ~action_change * (self.action_durations + 1)
        else:
            self.actions = max_pressure_actions
            self.action_durations = torch.zeros_like(self.actions)
            self.initialized = True
        return self.actions.cpu().numpy().tolist()

    def fit(self, environment: MultiprocessingMarlEnvironment, steps: int = 1_000, checkpoint_path: str = None):
        while environment.action_step < steps:
            state = environment.reset()
            episode_rewards = []
            while True:
                actions = self.act(state)
                next_state, rewards, done = environment.step(actions)
                state = next_state
                episode_rewards += rewards
                info = environment.info()
                print(info["progress"])
                if done:
                    self.phases = []
                    self.action_durations = []
                    self.initialized = False
                    break
        environment.close()

    def demo(self, environment: MarlEnvironment, checkpoint_path: str = None):
        state = environment.reset()
        while True:
            actions = self.act(state)
            state, _, done = environment.step(actions)
            if done:
                break
        environment.close()


class DQN(IndependentAgents):

    def __init__(
            self,
            network: Dict,
            discount_factor: float,
            learning_rate: float,
            mixing_factor: float,
            epsilon_greedy_prob: float,
            replay_buffer_size: int,
            batch_size: int,
            update_steps: int
    ):
        super(DQN, self).__init__()
        self.online_q_network = NetworkBody.create(network["class_name"], network["init_args"])
        self.target_q_network = NetworkBody.create(network["class_name"], network["init_args"]).requires_grad_(False)
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
        self.to(device)

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

    def fit(self, environment: MultiprocessingMarlEnvironment, steps: int = 1_000, checkpoint_path: str = None):
        self.train()
        highest_ema_reward = - sys.float_info.max
        _ = environment.reset()
        while environment.action_step <= steps:
            metadata = environment.metadata()
            self.n_workers = metadata["n_workers"]
            self.worker_agent_offsets = metadata["agent_offsets"]
            self.worker_action_offsets = metadata["action_offsets"]
            states = environment.state()
            actions, _, actions_bool_index, _, _, _ = self.act(states.to(device), eps=self.eps)

            next_states, rewards, done = environment.step(actions)
            if environment.episode == 0:
                continue

            self._update_replay_buffer(states, actions_bool_index, rewards, next_states)

            if len(self.replay_buffer) >= self.batch_size:
                self._train_step()

            info = environment.info()
            print(info["progress"])
            if checkpoint_path is not None and highest_ema_reward < environment.ema_reward:
                print(f"Save Checkpoint..")
                highest_ema_reward = environment.ema_reward
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
            q_predictions = self._compute_q_predictions(states.to(device), actions_bool_index.to(device))
            q_targets = self._compute_q_targets(rewards.to(device), next_states.to(device))
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
    def demo(self, environment: MarlEnvironment, checkpoint_path: str = None):
        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path, map_location=device))
        state = environment.reset()
        while True:
            state = state.to(device)
            actions, _, _, _, _, _ = self.act(state, mc_samples=10)
            state, _, done = environment.step(actions)
            if done:
                break
        environment.close()


class A2C(IndependentAgents):

    def __init__(
            self,
            network: Dict,
            discount_factor: float = 0.9,
            learning_rate: float = 0.001,
            actor_loss_weight: float = 1.0,
            critic_loss_weight: float = 1.0,
            entropy_loss_weight: float = 0.001,
            gradient_clipping_max_norm: float = 1.0
    ):
        super(A2C, self).__init__()
        self.network = NetworkBody.create(network["class_name"], network["init_args"])
        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)

        self.gamma = discount_factor
        self.actor_loss_weight = actor_loss_weight
        self.critic_loss_weight = critic_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.gradient_clipping_max_norm = gradient_clipping_max_norm

        self.apply(self._init_params)
        self.to(device)

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

    def fit(self, environment: MultiprocessingMarlEnvironment, steps: int = 1_000, checkpoint_path: str = None):
        self.train()
        _ = environment.reset()
        highest_ema_reward = - sys.float_info.max
        while environment.action_step <= steps:
            states = environment.state()
            states = states.to(device)
            actions, log_prob_actions, values, entropies = self.full_path(states)
            next_states, rewards, done = environment.step(actions)
            if environment.episode == 0:
                continue
            next_states, rewards = next_states.to(device), rewards.to(device)
            next_values = self.critic_path(next_states)

            action_values = rewards + self.gamma * next_values
            advantages = action_values.detach() - values

            critic_loss = 0.5 * torch.mean(advantages ** 2)
            actor_loss = - torch.mean(advantages.detach() * log_prob_actions + self.entropy_loss_weight * entropies)

            self.optimization_step(actor_loss, critic_loss)

            info = environment.info()
            print(info["progress"])
            if checkpoint_path is not None and highest_ema_reward < environment.ema_reward:
                print(f"Save Checkpoint..")
                highest_ema_reward = environment.ema_reward
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
        self.optimizer_c.step()
        self.optimizer_a.step()

    @torch.no_grad()
    def demo(self, environment: MarlEnvironment, checkpoint_path: str = None):
        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path, map_location=device))
        state = environment.reset()
        while True:
            state = state.to(device)
            actions = self.act(state, greedy=False, mc_samples=10)[0]
            state, _, done = environment.step(actions)
            if done:
                break
        environment.close()
