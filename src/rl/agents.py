from abc import abstractmethod
import os
from pathlib import Path
from typing import Any, List, Dict, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import orthogonal_
from torch_geometric.data import HeteroData, Batch

from src.data.replay_buffer import ReplayBuffer
from src.modules.distributions import GroupCategorical
from src.modules.network_bodies import NetworkBody
from src.modules.network_heads import NetworkHead
from src.modules.base_modules import MultiInputSequential
from src.modules.utils import group_argmax
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
    def train_env(self, environment: MarlEnvironment, episodes: int = 100, checkpoint_path: str = None):
        pass

    @abstractmethod
    def demo_env(self, environment: MarlEnvironment, checkpoint_path: str = None):
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

    def train_env(self, environment: MultiprocessingMarlEnvironment, episodes: int = 100, checkpoint_path: str = None):
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

    def demo_env(self, environment: MarlEnvironment, checkpoint_path: str = None):
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
            batch_size: int
    ):
        super(DQN, self).__init__()
        self.online_q_network = MultiInputSequential(
            NetworkBody.create(network["class_name"], network["init_args"]),
            NetworkHead.create(dqn_head["class_name"], dqn_head["init_args"])
        )
        self.target_q_network = MultiInputSequential(
            NetworkBody.create(network["class_name"], network["init_args"]),
            NetworkHead.create(dqn_head["class_name"], dqn_head["init_args"])
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
        greedy_actions, greedy_action_indices = group_argmax(action_values, index, return_indices=True)
        random_actions, random_action_indices = \
            GroupCategorical(logits=torch.ones_like(action_values), index=index).sample(return_indices=True)
        act_greedy = torch.rand_like(greedy_actions.to(torch.float32)) > eps
        actions = act_greedy * greedy_actions + ~act_greedy * random_actions
        if not training:
            return actions.cpu().tolist()
        action_indices = act_greedy * greedy_action_indices + ~act_greedy * random_action_indices
        action_indices_bool = torch.zeros_like(action_values)
        action_indices_bool[action_indices] = 1.0
        action_indices_bool = action_indices_bool.to(torch.bool)
        return actions.cpu().tolist(), action_indices_bool.cpu()

    def train_env(self, environment: MultiprocessingMarlEnvironment, episodes: int = 100, checkpoint_path: str = None):
        eps_greedy = ExpDecayEpsGreedyStrategy(self.eps_greedy_start, self.eps_greedy_end, self.eps_greedy_steps)
        highest_avg_reward = None
        for episode in range(episodes):
            states, self.n_workers, self.worker_agent_offsets, self.worker_action_offsets = \
                environment.reset(return_n_workers=True, return_worker_offsets=True)
            episode_rewards = []
            while True:
                actions, action_indices = self.act(states.to(device),
                                                   eps=eps_greedy.get_next_eps() if episode > 0 else 1.0,
                                                   training=True)
                next_states, rewards, done = environment.step(actions)
                if episode > 0:
                    self._update_replay_buffer(states, action_indices, rewards, next_states)
                if len(self.replay_buffer) >= self.batch_size and episode > 0:
                    self._train_step()
                states = next_states
                episode_rewards += rewards
                if done:
                    break

            avg_reward = torch.mean(torch.tensor(episode_rewards))
            print(f"--- Episode: {episode}   Reward: {avg_reward}   Epsilon: {eps_greedy.get_current_eps()}   "
                  f"Steps: {environment.n_steps} ---")
            if checkpoint_path is not None and \
                    (highest_avg_reward is None or highest_avg_reward <= avg_reward):
                print(f"Save Checkpoint..")
                highest_avg_reward = avg_reward
                torch.save(self.state_dict(), checkpoint_path)
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
        _, next_greedy_action_indices = group_argmax(action_values_local, index, return_indices=True)
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

    def demo_env(self, environment: MarlEnvironment, checkpoint_path: str = None):
        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path, map_location=device))
        state = environment.reset()
        while True:
            state = state.to(device)
            actions = self.act(state)
            state, _, done = environment.step(actions)
            if done:
                break
        environment.close()


class A2C(IndependentAgents):

    def __init__(
            self,
            network: Dict,
            actor_head: Dict,
            critic_head: Dict,
            share_network: bool = False,
            discount_factor: float = 0.9,
            learning_rate: float = 0.001,
            actor_loss_weight: float = 1.0,
            critic_loss_weight: float = 1.0,
            entropy_loss_weight: float = 0.001,
            gradient_clipping_max_norm: float = 1.0
    ):
        super(A2C, self).__init__()
        self.shared = share_network
        self.actor_head = NetworkHead.create(actor_head["class_name"], actor_head["init_args"])
        self.critic_head = NetworkHead.create(critic_head["class_name"], critic_head["init_args"])
        if self.shared:
            self.network = NetworkBody.create(network["class_name"], network["init_args"])
            self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        else:
            self.network_a = NetworkBody.create(network["class_name"], network["init_args"])
            self.network_c = NetworkBody.create(network["class_name"], network["init_args"])
            self.optimizer_a = torch.optim.Adam(list(self.network_a.parameters()) + list(self.actor_head.parameters()),
                                                learning_rate)
            self.optimizer_c = torch.optim.Adam(list(self.network_c.parameters()) + list(self.critic_head.parameters()),
                                                learning_rate)

        self.gamma = discount_factor
        self.actor_loss_weight = actor_loss_weight
        self.critic_loss_weight = critic_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.gradient_clipping_max_norm = gradient_clipping_max_norm

        self.apply(self._init_params)
        self.to(device)

    def forward(self, state: HeteroData, actor_out: bool = True, critic_out: bool = True) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        if self.shared:
            out = self.network(state)
            input_actor_head, input_critic_head = out, out
        else:
            input_actor_head = self.network_a(state)
            input_critic_head = self.network_c(state)
        logits, values, index = None, None, None
        if actor_out:
            logits, index = self.actor_head(*input_actor_head)
        if critic_out:
            values, index = self.critic_head(*input_critic_head) if critic_out else None
        return logits, values, index

    def full_path(self, state: HeteroData) -> Tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, index = self.forward(state)
        distribution = GroupCategorical(logits, index)
        actions, action_indices = distribution.sample(return_indices=True)
        log_probs = distribution.log_prob(action_indices)
        entropy = distribution.entropy()
        return actions.cpu().numpy().tolist(), log_probs, values, entropy

    def actor_path(self, state: HeteroData):
        actions, log_probs, _, entropy = self.full_path(state)
        return actions, log_probs, entropy

    def critic_path(self, state: HeteroData) -> torch.Tensor:
        values = self.forward(state, actor_out=False)[1]
        return values

    def act(self, state: HeteroData, greedy: bool = False) -> List[int]:
        if greedy:
            logits, _, index = self.forward(state, critic_out=False)
            actions = group_argmax(logits, index)
            return actions.cpu().numpy().tolist()
        state = state
        actions = self.full_path(state)[0]
        return actions

    def train_env(self, environment: MultiprocessingMarlEnvironment, episodes: int = 100, checkpoint_path: str = None):
        highest_avg_reward = None
        for episode in range(episodes):
            episode_rewards = []
            episode_entropies = []
            states = environment.reset()
            states.to(device)

            while True:
                actions, log_prob_actions, values, entropies = self.full_path(states)
                next_states, rewards, done = environment.step(actions)
                if done:
                    break
                episode_rewards.append(rewards.detach().cpu())
                episode_entropies.append(entropies.detach().cpu())
                next_states, rewards = next_states.to(device), rewards.to(device)
                if episode == 0:
                    continue
                next_values = self.critic_path(next_states)
                states = next_states

                action_values = rewards + self.gamma * next_values
                advantages = action_values.detach() - values

                critic_loss = 0.5 * torch.mean(advantages ** 2)
                actor_loss = - torch.mean(advantages.detach() * log_prob_actions + self.entropy_loss_weight * entropies)

                self.optimization_step(actor_loss, critic_loss)

            avg_reward = torch.mean(torch.cat(episode_rewards, dim=0)).item()
            avg_entropy = torch.mean(torch.cat(episode_entropies, dim=0)).item()
            print(f"--- Episode: {episode}   Reward: {avg_reward}   Entropy: {avg_entropy}   "
                  f"Steps: {environment.n_steps} ---")
            if checkpoint_path is not None and (highest_avg_reward is None or highest_avg_reward <= avg_reward):
                print(f"Save Checkpoint..")
                highest_avg_reward = avg_reward
                torch.save(self.state_dict(), checkpoint_path)

        environment.close()

    def optimization_step(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor):
        if self.shared:
            self._shared_optimization_step(actor_loss, critic_loss)
        else:
            self._unshared_optimization_step(actor_loss, critic_loss)

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

    def demo_env(self, environment: MarlEnvironment, checkpoint_path: str = None):
        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path, map_location=device))
        state = environment.reset()
        while True:
            state = state.to(device)
            actions = self.act(state, greedy=False  )
            state, _, done = environment.step(actions)
            if done:
                break
        environment.close()
