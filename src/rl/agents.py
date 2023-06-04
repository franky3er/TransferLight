from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any, List, Dict, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import orthogonal_
from torch_geometric.data import Data, HeteroData, Batch
from torch_geometric.nn.aggr import MaxAggregation

from src.data.replay_buffer import ReplayBuffer
from src.models.dqns import QNet
from src.modules.distributions import FlexibleCategorical
from src.networks.actor_heads import ActorHead
from src.networks.critic_heads import CriticHead
from src.networks.dqn_heads import DQNHead
from src.networks.shared import SharedNetwork
from src.modules.utils import MultiInputSequential, FlexibleArgmax, NumElementsAggregation
from src.params import ENV_ACTION_EXECUTION_TIME
from src.rl.environments import TscMarlEnvironment, MultiprocessingTscMarlEnvironment
from src.rl.exploration import ExpDecayEpsGreedyStrategy, ConstantEpsGreedyStrategy

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Use {device} device")


max_aggr = MaxAggregation()
argmax_aggr = FlexibleArgmax()
num_elements_aggr = NumElementsAggregation()


class IndependentAgents(nn.Module):

    @abstractmethod
    @torch.no_grad()
    def act(self, state: Any) -> List[int]:
        pass

    @abstractmethod
    def train_env(self, environment: TscMarlEnvironment, episodes: int = 100):
        pass

    @abstractmethod
    def demo_env(self, environment: TscMarlEnvironment):
        pass

    @staticmethod
    def _init_params(module):
        if isinstance(module, nn.Linear):
            orthogonal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.1)


class MaxPressureAgents(IndependentAgents):

    def __init__(self, min_phase_duration: int):
        super(MaxPressureAgents, self).__init__()
        self.min_phase_steps = min_phase_duration // ENV_ACTION_EXECUTION_TIME
        self.initialized = False
        self.actions = None
        self.action_durations = None
        self._argmax = FlexibleArgmax()

    def act(self, state: HeteroData) -> List[int]:
        x = state["phase"].x.squeeze().to(device)
        index = state["phase", "to", "intersection"].edge_index[1].to(device)
        max_pressure_actions = self._argmax(x, index)
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

    def training(self, environment: TscMarlEnvironment, episodes: int = 100):
        for i in range(episodes):
            state = environment.reset()
            episode_rewards = []
            while True:
                actions = self.act(state)
                next_state, rewards, done = environment.step(actions)
                state = next_state
                episode_rewards += rewards
                if done:
                    avg_episode_reward = torch.mean(torch.tensor(episode_rewards))
                    print(f"--- Epoch: {i}   Reward: {avg_episode_reward} ---")
                    self.phases = []
                    self.action_durations = []
                    self.initialized = False
                    break
        environment.close()

    def demo_env(self, environment: TscMarlEnvironment):
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
            dqn_head: Dict,
            discount_factor: float,
            learning_rate: float,
            tau: float,
            eps_greedy_start: float,
            eps_greedy_end: float,
            eps_greedy_steps: int,
            replay_buffer_size: int,
            batch_size: int,
            checkpoint_dir: str = None,
            save_checkpoint: bool = False,
            load_checkpoint: bool = False
    ):
        super(DQN, self).__init__()
        self.online_q_network = MultiInputSequential(
            SharedNetwork.create(network["class_name"], network["init_args"]),
            DQNHead.create(dqn_head["class_name"], dqn_head["init_args"])
        )
        self.target_q_network = MultiInputSequential(
            SharedNetwork.create(network["class_name"], network["init_args"]),
            DQNHead.create(dqn_head["class_name"], dqn_head["init_args"])
        ).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online_q_network.parameters(), learning_rate)

        self.discount_factor = discount_factor
        self.tau = tau
        self.eps_greedy_start = eps_greedy_start
        self.eps_greedy_end = eps_greedy_end
        self.eps_greedy_steps = eps_greedy_steps
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size, batch_size)
        self.apply(self._init_params)

        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            self.checkpoint_path = os.path.join(checkpoint_dir, "dqn-checkpoint.pt")
            self.save_checkpoint = save_checkpoint
            self.load_checkpoint = load_checkpoint
        else:
            self.save_checkpoint = False
            self.load_checkpoint = False
        if self.load_checkpoint:
            self.load_state_dict(torch.load(self.checkpoint_path, map_location=device))

        self.n_workers = None
        self.worker_agent_offsets = None
        self.worker_action_offsets = None

        self.to(device)

    def forward(self, state: HeteroData) -> Tuple[torch.Tensor, torch.LongTensor]:
        action_values, index = self.online_q_network(state)
        return action_values, index

    def act(self, state: HeteroData, eps: float = 0.0, training: bool = False) -> \
            Union[List[int], Tuple[List[int], torch.Tensor]]:
        action_values, index = self.forward(state)
        greedy_actions, greedy_action_indices = FlexibleArgmax()(action_values, index, return_argmax_indices=True)
        random_actions, random_action_indices = \
            FlexibleCategorical(logits=torch.ones_like(action_values), index=index).sample(return_sample_indices=True)
        act_greedy = torch.rand_like(greedy_actions.to(torch.float32)) > eps
        actions = act_greedy * greedy_actions + ~act_greedy * random_actions
        if not training:
            return actions.cpu().tolist()
        action_indices = act_greedy * greedy_action_indices + ~act_greedy * random_action_indices
        action_indices_bool = torch.zeros_like(action_values)
        action_indices_bool[action_indices] = 1.0
        action_indices_bool = action_indices_bool.to(torch.bool)
        return actions.cpu().tolist(), action_indices_bool.cpu()

    def train_env(self, environment: MultiprocessingTscMarlEnvironment, episodes: int = 100):
        eps_greedy = ExpDecayEpsGreedyStrategy(self.eps_greedy_start, self.eps_greedy_end, self.eps_greedy_steps)
        highest_episode_avg_reward = None
        for i in range(episodes):
            states, self.n_workers, self.worker_agent_offsets, self.worker_action_offsets = \
                environment.reset(return_n_workers=True, return_worker_offsets=True)
            episode_rewards = []
            while True:
                actions, action_indices = self.act(states.to(device), eps=eps_greedy.get_next_eps(), training=True)
                next_states, rewards, done = environment.step(actions)
                self._update_replay_buffer(states, action_indices, rewards, next_states)
                if len(self.replay_buffer) >= self.batch_size and i >= 1:
                    self._train_step()
                states = next_states
                episode_rewards += rewards
                if done:
                    avg_episode_reward = torch.mean(torch.tensor(episode_rewards))
                    print(f"--- Episode: {i}   Reward: {avg_episode_reward} ---")
                    if self.save_checkpoint and \
                            (highest_episode_avg_reward is None or highest_episode_avg_reward <= avg_episode_reward):
                        print(f"Save Checkpoint..")
                        highest_episode_avg_reward = avg_episode_reward
                        torch.save(self.state_dict(), self.checkpoint_path)
                    break
        environment.close()

    def _update_replay_buffer(self, states: Batch, action_indices: torch.BoolTensor, rewards: torch.Tensor,
                             next_states: Batch):
        states = [states.get_example(i) for i in range(self.n_workers)]
        action_indices = [action_indices[start:end] for start, end in
                          [(0 if i == 0 else self.worker_action_offsets[i-1], self.worker_action_offsets[i])
                           for i in range(len(self.worker_action_offsets))]]
        rewards = [rewards[start:end] for start, end in
                   [(0 if i == 0 else self.worker_agent_offsets[i - 1], self.worker_agent_offsets[i])
                    for i in range(len(self.worker_agent_offsets))]]
        next_states = [next_states.get_example(i) for i in range(self.n_workers)]
        self.replay_buffer.add(states, action_indices, rewards, next_states)

    def _train_step(self):
        states, action_indices, rewards, next_states = self.replay_buffer.sample()
        q_predictions = self._compute_q_predictions(states.to(device), action_indices.to(device))
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
        _, next_greedy_action_indices = FlexibleArgmax()(action_values_local, index, return_argmax_indices=True)
        next_greedy_action_values = action_values_target[next_greedy_action_indices]
        q_targets = rewards + self.discount_factor * next_greedy_action_values
        return q_targets.detach()

    def _optimization_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _update_target_q_network(self):
        # Polyak averaging to update the parameters of the target network
        for target_param, local_param in zip(self.target_q_network.parameters(), self.online_q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class IQLAgents(IndependentAgents):

    def __init__(self,
                 qnet_params: dict,
                 buffer_size: int,
                 batch_size: int,
                 learning_rate: float,
                 discount_factor: float,
                 tau: float,
                 eps_greedy_start: float,
                 eps_greedy_end: float,
                 eps_greedy_steps: int,
                 checkpoint_dir: str = None,
                 save_checkpoint: bool = False,
                 load_checkpoint: bool = False):
        self.qnet_local = QNet(**qnet_params).to(device)
        self.qnet_target = QNet(**qnet_params).to(device)
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.tau = tau
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.optimizer = torch.optim.Adam(self.qnet_local.parameters(), lr=learning_rate)
        self.eps_greedy = ExpDecayEpsGreedyStrategy(eps_greedy_start, eps_greedy_end, eps_greedy_steps)
        self.act_greedy = False
        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            self.qnet_local_checkpoint_path = os.path.join(checkpoint_dir, "qnet-local-checkpoint.pt")
            self.qnet_target_checkpoint_path = os.path.join(checkpoint_dir, "qnet-target-checkpoint.pt")
            self.save_checkpoint = save_checkpoint
            self.load_checkpoint = load_checkpoint
        else:
            self.save_checkpoint = False
            self.load_checkpoint = False
        if self.load_checkpoint:
            self.qnet_local.load_state_dict(torch.load(self.qnet_local_checkpoint_path, map_location=device))
            self.qnet_target.load_state_dict(torch.load(self.qnet_target_checkpoint_path, map_location=device))

    def act(self, state: Data) -> List[int]:
        x = state.x.to(device)
        with torch.no_grad():
            action_values = self.qnet_local(x)
            greedy_actions = torch.argmax(action_values, dim=-1)
            random_actions = torch.randint(low=0, high=action_values.size(-1), size=(greedy_actions.size(0),),
                                           device=device)
            eps = self.eps_greedy.get_next_eps() if not self.act_greedy else 0.0
            act_greedy = torch.rand(size=(greedy_actions.size(0),), device=device) > eps
            actions = act_greedy * greedy_actions + ~act_greedy * random_actions
        return actions.cpu().numpy().tolist()

    def training(self, environment: TscMarlEnvironment, episodes: int = 100):
        self.act_greedy = False
        highest_episode_avg_reward = None
        for i in range(episodes):
            state = environment.reset()
            episode_rewards = []
            while True:
                actions = self.act(state)
                next_state, rewards, done = environment.step(actions)
                self.train_step(state, actions, rewards, next_state, done)
                state = next_state
                episode_rewards += rewards
                if done:
                    avg_episode_reward = torch.mean(torch.tensor(episode_rewards))
                    print(f"--- Episode: {i}   Reward: {avg_episode_reward} ---")
                    if self.save_checkpoint and \
                            (highest_episode_avg_reward is None or highest_episode_avg_reward <= avg_episode_reward):
                        print(f"Save Checkpoint..")
                        highest_episode_avg_reward = avg_episode_reward
                        torch.save(self.qnet_local.state_dict(), self.qnet_local_checkpoint_path)
                        torch.save(self.qnet_target.state_dict(), self.qnet_target_checkpoint_path)
                    break
        environment.close()

    def demo_env(self, environment: TscMarlEnvironment):
        self.act_greedy = True
        state = environment.reset()
        while True:
            actions = self.act(state)
            state, _, done = environment.step(actions)
            if done:
                break
        environment.close()

    def train_step(self, state: Data, actions: List[int], rewards: torch.Tensor, next_state: Data, done: bool):
        self.replay_buffer.add(state, actions, rewards, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        states, actions, rewards, next_states, dones = states.to(device), actions.to(device), rewards.to(device), \
            next_states.to(device), dones.to(device)

        q_targets = self.compute_q_targets(rewards, next_states)
        q_predictions = self.qnet_local(states.x).gather(-1, actions.unsqueeze(-1)).squeeze()
        loss = F.mse_loss(q_predictions, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_target_dqn()

        states.cpu(), actions.cpu(), rewards.cpu(), next_states.cpu(), dones.cpu()

    def compute_q_targets(self, rewards: torch.Tensor, next_states: Batch):
        # Double DQN target computation
        with torch.no_grad():
            action_values_local = self.qnet_local(next_states.x)
            action_values_target = self.qnet_target(next_states.x)
            next_greedy_actions = torch.argmax(action_values_local, dim=-1, keepdim=True)
            next_greedy_action_values = torch.gather(action_values_target, -1, next_greedy_actions).squeeze()
            return rewards + self.discount_factor * next_greedy_action_values

    def _update_target_dqn(self):
        # Polyak averaging to update the parameters of the target network
        for target_param, local_param in zip(self.qnet_target.parameters(), self.qnet_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class A2C(IndependentAgents):

    def __init__(
            self,
            network: Dict,
            actor_head: Dict,
            critic_head: Dict,
            share_network: bool = False,
            discount_factor: float = 0.9,
            gae_discount_factor: float = 0.5,
            learning_rate: float = 0.001,
            n_steps: int = 10,
            actor_loss_weight: float = 1.0,
            critic_loss_weight: float = 1.0,
            entropy_loss_weight: float = 0.001,
            gradient_clipping_max_norm: float = 1.0,
            checkpoint_dir: str = None,
            save_checkpoint: bool = False,
            load_checkpoint: bool = False
    ):
        super(A2C, self).__init__()
        self.shared = share_network
        self.actor_head = ActorHead.create(actor_head["class_name"], actor_head["init_args"])
        self.critic_head = CriticHead.create(critic_head["class_name"], critic_head["init_args"])
        if self.shared:
            self.network = SharedNetwork.create(network["class_name"], network["init_args"])
            self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        else:
            self.network_a = SharedNetwork.create(network["class_name"], network["init_args"])
            self.network_c = SharedNetwork.create(network["class_name"], network["init_args"])
            self.optimizer_a = torch.optim.Adam(list(self.network_a.parameters()) + list(self.actor_head.parameters()),
                                                learning_rate)
            self.optimizer_c = torch.optim.Adam(list(self.network_c.parameters()) + list(self.critic_head.parameters()),
                                                learning_rate)

        self.gamma = discount_factor
        self.lamda = gae_discount_factor
        self.n_steps = n_steps
        self.actor_loss_weight = actor_loss_weight
        self.critic_loss_weight = critic_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.gradient_clipping_max_norm = gradient_clipping_max_norm

        self.register_buffer("trajectory_discount_weights", (self.gamma * self.lamda) ** torch.arange(n_steps))

        self.apply(self._init_params)
        self.to(device)

        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            self.checkpoint_path = os.path.join(checkpoint_dir, "actor-critic-checkpoint.pt")
            self.save_checkpoint = save_checkpoint
            self.load_checkpoint = load_checkpoint
        else:
            self.save_checkpoint = False
            self.load_checkpoint = False
        if self.load_checkpoint:
            self.load_state_dict(torch.load(self.checkpoint_path, map_location=device))

    def forward(self, state: HeteroData, actor_out: bool = True, critic_out: bool = True) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        if self.shared:
            action_embedding, agent_index = self.network(state)
            actor_action_embedding, critic_action_embedding = action_embedding, action_embedding
        else:
            actor_action_embedding, agent_index = self.network_a(state)
            critic_action_embedding, _ = self.network_c(state)
        logits = self.actor_head(actor_action_embedding, agent_index) if actor_out else None
        values = self.critic_head(critic_action_embedding, agent_index) if critic_out else None
        return logits, values, agent_index

    def full_path(self, state: HeteroData) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, index = self.forward(state)
        distribution = FlexibleCategorical(logits, index)
        actions, action_indices = distribution.sample(return_sample_indices=True)
        log_probs = distribution.log_prob(action_indices)
        entropy = distribution.entropy()
        return actions, log_probs, values, entropy

    def actor_path(self, state: HeteroData):
        actions, log_probs, _, entropy = self.full_path(state)
        return actions, log_probs, entropy

    def critic_path(self, state: HeteroData) -> torch.Tensor:
        values = self.forward(state, actor_out=False)[1]
        return values

    def act(self, state: HeteroData) -> List[int]:
        state = state.to(device)
        actions = self.full_path(state)[0]
        return actions.detach().cpu().numpy().tolist()

    def train_env(self, environment: MultiprocessingTscMarlEnvironment, episodes: int = 100):
        highest_episode_avg_reward = None
        for episode in range(episodes):
            episode_rewards = None
            episode_entropies = None
            states = environment.reset()
            states.to(device)
            done_episode = False

            while not done_episode:
                torch.cuda.empty_cache()
                trajectory_rewards, trajectory_values, trajectory_next_values = [], [], []
                log_prob_actions, entropy = None, None
                for t in range(self.n_steps):
                    actions, log_prob_actions_, values, entropy_ = self.full_path(states)
                    next_states, rewards, done = environment.step(actions.cpu().numpy().tolist())
                    next_states, rewards, entropy_ = next_states.to(device), rewards.to(device), entropy_.to(device)
                    next_values = self.critic_path(next_states)

                    trajectory_rewards.append(rewards.unsqueeze(1))
                    trajectory_values.append(values.unsqueeze(1))
                    trajectory_next_values.append(next_values.unsqueeze(1))

                    states = next_states

                    episode_rewards = rewards.detach().cpu() if episode_rewards is None \
                        else torch.cat([episode_rewards, rewards.detach().cpu()], dim=0)
                    episode_entropies = entropy_.detach().cpu() if episode_entropies is None \
                        else torch.cat([episode_entropies, entropy_.detach().cpu()], dim=0)

                    if t == 0:
                        log_prob_actions = log_prob_actions_
                        entropy = entropy_
                    if done:
                        done_episode = True
                        break

                trajectory_rewards = torch.cat(trajectory_rewards, dim=1)
                trajectory_values = torch.cat(trajectory_values, dim=1)
                trajectory_next_values = torch.cat(trajectory_next_values, dim=1)
                trajectory_bootstrap_returns = (trajectory_rewards + self.gamma * trajectory_next_values).detach()
                trajectory_td_advantages = trajectory_bootstrap_returns - trajectory_values
                trajectory_discount_weights = self.trajectory_discount_weights[:trajectory_rewards.size(1)].unsqueeze(0)

                gae_advantages = torch.sum(trajectory_discount_weights * trajectory_td_advantages, dim=1)

                # Only train critic in the first episode to account for high bias esp. in the beginning of training
                actor_loss = - torch.mean(gae_advantages.detach() * log_prob_actions) if episode >= 1 else 0
                critic_loss = torch.mean((trajectory_td_advantages ** 2) * 0.5)
                entropy_loss = - torch.mean(entropy)

                self.optimization_step(actor_loss, critic_loss, entropy_loss)

                if done_episode:
                    avg_episode_reward = torch.mean(episode_rewards).item()
                    avg_episode_entropy = torch.mean(episode_entropies).item()
                    print(f"--- Episode: {episode}   Reward: {avg_episode_reward}   Entropy: {avg_episode_entropy}---")
                    if self.save_checkpoint and \
                            (highest_episode_avg_reward is None or highest_episode_avg_reward <= avg_episode_reward):
                        print(f"Save Checkpoint..")
                        highest_episode_avg_reward = avg_episode_reward
                        torch.save(self.state_dict(), self.checkpoint_path)
                    break

        environment.close()

    def optimization_step(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor, entropy_loss: torch.Tensor):
        if self.shared:
            self._shared_optimization_step(actor_loss, critic_loss, entropy_loss)
        else:
            self._unshared_optimization_step(actor_loss, critic_loss, entropy_loss)

    def _shared_optimization_step(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor,
                                  entropy_loss: torch.Tensor):
        loss = self.actor_loss_weight * actor_loss + self.critic_loss_weight * critic_loss + \
               self.entropy_loss_weight * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clipping_max_norm)
        self.optimizer.step()

    def _unshared_optimization_step(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor,
                                    entropy_loss: torch.Tensor):

        #print(f"Actor Loss: {actor_loss}  --  Critic Loss: {critic_loss}  --  Entropy Loss: {entropy_loss}")
        actor_loss = self.actor_loss_weight * actor_loss + self.entropy_loss_weight * entropy_loss
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

    def demo_env(self, environment: TscMarlEnvironment):
        state = environment.reset()
        while True:
            actions = self.act(state)
            state, _, done = environment.step(actions)
            if done:
                break
        environment.close()
