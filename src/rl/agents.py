from abc import ABC, abstractmethod
import os
from pathlib import Path
import random
from typing import Any, List, Dict

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Data, HeteroData, Batch
from torch_geometric.nn.aggr import MaxAggregation

from src.data.replay_buffer import ReplayBuffer
from src.models.dqns import QNet, HieraGLightDQN
from src.models.actor_critic import ActorNetwork, CriticNetwork
from src.models.modules import ArgmaxAggregation, NumElementsAggregation
from src.params import ENV_ACTION_EXECUTION_TIME
from src.rl.environments import TscMarlEnvironment, MultiprocessingTscMarlEnvironment
from src.rl.exploration import ExpDecayEpsGreedyStrategy, ConstantEpsGreedyStrategy

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Use {device} device")


max_aggr = MaxAggregation()
argmax_aggr = ArgmaxAggregation()
num_elements_aggr = NumElementsAggregation()


class TscMarlAgent(ABC):

    @abstractmethod
    def act(self, state: Any) -> List[int]:
        pass

    @abstractmethod
    def train(self, environment: TscMarlEnvironment, episodes: int = 100):
        pass

    @abstractmethod
    def demo(self, environment: TscMarlEnvironment):
        pass


class HieraGLightAgent(TscMarlAgent):

    def __init__(self,
                 movement_dim: int,
                 hidden_dim: int,
                 buffer_size: int,
                 batch_size: int,
                 learning_rate: float,
                 discount_factor: float,
                 tau: float,
                 eps_greedy_start: float,
                 eps_greedy_end: float,
                 eps_greedy_steps: int,
                 act_greedy: bool = False):
        self.dqn_local = HieraGLightDQN(movement_dim, hidden_dim).to(device)
        self.dqn_target = HieraGLightDQN(movement_dim, hidden_dim).to(device)
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.dqn_local.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.tau = tau
        self.eps_greedy = ConstantEpsGreedyStrategy(0.0) if act_greedy \
            else ExpDecayEpsGreedyStrategy(eps_greedy_start, eps_greedy_end, eps_greedy_steps)
        self.qnet_local_checkpoint_name = "qnet-local-checkpoint.pt"
        self.qnet_target_checkpoint_name = "qnet-target-checkpoint.pt"

    def act(self, state: HeteroData) -> List[int]:
        with torch.no_grad():
            state = state.to(device)
            tls_index = state.edge_index_dict[("junction", "to", "phase")][0]
            action_values = self.dqn_local(state.x_dict, state.edge_index_dict)
            greedy_actions = argmax_aggr(action_values, tls_index)
            random_actions = torch.cat([torch.randint(high=n, size=(1,), device=device)
                                        for n in num_elements_aggr(tls_index)], dim=0)
            eps = self.eps_greedy.get_next_eps()
            act_greedy = torch.rand(size=(greedy_actions.size(0),), device=device) > eps
            actions = act_greedy * greedy_actions + ~act_greedy * random_actions
            state.cpu()
            return actions.cpu().numpy().tolist()

    def train_step(self, state: Any, actions: List[int], rewards: torch.Tensor, next_state: Any, done: bool):
        self.replay_buffer.add(state, actions, rewards, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        states, actions, rewards, next_states, dones = states.to(device), actions.to(device), rewards.to(device), \
            next_states.to(device), dones.to(device)

        q_targets = self._compute_q_targets(rewards, next_states)
        q_predictions = self._compute_q_predictions(states, actions)

        loss = F.mse_loss(q_predictions, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_target_dqn()

        states.cpu()
        next_states.cpu()

    def _compute_q_targets(self, rewards: torch.Tensor, next_states: HeteroData) -> torch.Tensor:
        with torch.no_grad():
            tls_index = next_states.edge_index_dict[("junction", "to", "phase")][0]
            action_values_local = self.dqn_local(next_states.x_dict, next_states.edge_index_dict)
            action_values_target = self.dqn_target(next_states.x_dict, next_states.edge_index_dict)
            next_greedy_action_indices = argmax_aggr(action_values_local, tls_index, return_indices=True)
            next_greedy_action_values = action_values_target[next_greedy_action_indices]
            return rewards.unsqueeze(dim=1) + self.discount_factor * next_greedy_action_values

    def _compute_q_predictions(self, states: HeteroData, actions: torch.Tensor) -> torch.Tensor:
        tls_index = states.edge_index_dict[("junction", "to", "phase")][0]
        possible_actions = num_elements_aggr(tls_index)
        possible_actions_summed = torch.tensor([0 if i == 0 else torch.sum(possible_actions[:i]).item()
                                                for i in range(possible_actions.size(0))], device=possible_actions.device)
        action_indices = actions + possible_actions_summed
        return self.dqn_local(states.x_dict, states.edge_index_dict)[action_indices]

    def _update_target_dqn(self):
        # Polyak averaging to update the parameters of the target network
        for target_param, local_param in zip(self.dqn_target.parameters(), self.dqn_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class RandomAgents(TscMarlAgent):

    def act(self, state: List[List[int]]) -> List[int]:
        actions = [random.randint(0, len(a)-1) for a in state]
        return actions

    def train(self, environment: TscMarlEnvironment, episodes: int = 100):
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
                    break
        environment.close()

    def demo(self, environment: TscMarlEnvironment):
        state = environment.reset()
        while True:
            actions = self.act(state)
            state, _, done = environment.step(actions)
            if done:
                break
        environment.close()


class MaxPressureAgents(TscMarlAgent):

    def __init__(self, min_phase_duration: int):
        super(MaxPressureAgents, self).__init__()
        self.min_phase_steps = min_phase_duration // ENV_ACTION_EXECUTION_TIME
        self.initialized = False
        self.phases = []
        self.phase_durations = []

    def act(self, state: List[List[int]]) -> List[int]:
        actions = []
        for i, pressures in enumerate(state):
            max_pressure_phase = np.argmax(pressures)
            if not self.initialized:
                actions.append(max_pressure_phase)
                self.phases.append(max_pressure_phase)
                self.phase_durations.append(0)
            else:
                if self.phase_durations[i] < self.min_phase_steps or self.phases[i] == max_pressure_phase:
                    actions.append(self.phases[i])
                    self.phase_durations[i] += 1
                else:
                    actions.append(max_pressure_phase)
                    self.phases[i] = max_pressure_phase
                    self.phase_durations[i] = 0
        self.initialized = True
        return actions

    def train(self, environment: TscMarlEnvironment, episodes: int = 100):
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
                    break
        environment.close()

    def demo(self, environment: TscMarlEnvironment):
        state = environment.reset()
        while True:
            actions = self.act(state)
            state, _, done = environment.step(actions)
            if done:
                break
        environment.close()


class IQLAgents(TscMarlAgent):

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

    def train(self, environment: TscMarlEnvironment, episodes: int = 100):
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

    def demo(self, environment: TscMarlEnvironment):
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


class IA2CAgents(TscMarlAgent):

    def __init__(self,
                 actor: Dict,
                 critic: Dict,
                 learning_rate_actor: float,
                 learning_rate_critic: float,
                 discount_factor: float,
                 entropy_weight: float,
                 checkpoint_dir: str = None,
                 save_checkpoint: bool = False,
                 load_checkpoint: bool = False):
        self.actor = ActorNetwork.create(actor["class_name"], actor["init_args"]).to(device)
        self.critic = CriticNetwork.create(critic["class_name"], critic["init_args"]).to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=learning_rate_critic)
        self.discount_factor = discount_factor
        self.entropy_weight = entropy_weight

        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            self.actor_checkpoint_path = os.path.join(checkpoint_dir, "actor-checkpoint.pt")
            self.critic_checkpoint_path = os.path.join(checkpoint_dir, "critic-checkpoint.pt")
            self.save_checkpoint = save_checkpoint
            self.load_checkpoint = load_checkpoint
        else:
            self.save_checkpoint = False
            self.load_checkpoint = False
        if self.load_checkpoint:
            self.actor.load_state_dict(torch.load(self.actor_checkpoint_path, map_location=device))
            self.critic.load_state_dict(torch.load(self.critic_checkpoint_path, map_location=device))

    def act(self, state: Any) -> List[int]:
        state = state.to(device)
        actions, _, _ = self.actor.full_path(state)
        return actions.detach().cpu().numpy().tolist()

    def train(self, environment: MultiprocessingTscMarlEnvironment, episodes: int = 100):
        highest_episode_avg_reward = None
        for episode in range(episodes):
            episode_rewards = None
            episode_entropies = None
            states = environment.reset()
            states.to(device)
            while True:
                actions, log_prob_actions, entropy = self.actor.full_path(states)
                next_states, rewards, done = environment.step(actions.cpu().numpy().tolist())
                next_states, rewards, entropy = next_states.to(device), rewards.to(device), entropy.to(device)
                values = self.critic(states)
                next_values = self.critic(next_states)

                advantages = rewards + self.discount_factor * next_values - values
                #print(f"rewards {rewards.shape},  advantages {advantages.shape},  entropy {entropy.shape},  values {values.shape}")

                actor_loss = - torch.mean(log_prob_actions * advantages.detach() + self.entropy_weight * entropy)
                #print(f"actor loss {actor_loss}")
                critic_loss = torch.mean(advantages ** 2)
                #print(f"critic loss {critic_loss}")

                self.critic_optim.zero_grad()
                self.actor_optim.zero_grad()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                critic_loss.backward()
                # Only train critic in the first episode to account for high bias esp. in the beginning of training
                if episode >= 1:
                    actor_loss.backward()
                self.critic_optim.step()
                self.actor_optim.step()

                states = next_states

                episode_rewards = rewards if episode_rewards is None else torch.cat([episode_rewards, rewards], dim=0)
                episode_entropy = entropy if episode_entropies is None else torch.cat([episode_entropy, entropy], dim=0)

                if done:
                    avg_episode_reward = torch.mean(episode_rewards).item()
                    avg_episode_entropy = torch.mean(episode_entropy).item()
                    print(f"--- Episode: {episode}   Reward: {avg_episode_reward}   Entropy: {avg_episode_entropy}---")
                    if self.save_checkpoint and \
                            (highest_episode_avg_reward is None or highest_episode_avg_reward <= avg_episode_reward):
                        print(f"Save Checkpoint..")
                        highest_episode_avg_reward = avg_episode_reward
                        torch.save(self.actor.state_dict(), self.actor_checkpoint_path)
                        torch.save(self.critic.state_dict(), self.critic_checkpoint_path)
                    break

        environment.close()

    def demo(self, environment: TscMarlEnvironment):
        state = environment.reset()
        while True:
            actions = self.act(state)
            state, _, done = environment.step(actions)
            if done:
                break
        environment.close()

    def _batch(self, states: Any, rewards: List[torch.Tensor], dones: List[bool]):
        states = self._batch_states(states)
        rewards = self._batch_rewards(rewards)
        done = self._batch_dones(dones)
        return states, rewards, done

    @staticmethod
    def _batch_states(states: Any) -> Any:
        return Batch.from_data_list(states).to(device)

    @staticmethod
    def _batch_rewards(rewards: List[torch.Tensor]):
        return torch.cat(rewards, dim=0).to(device)

    @staticmethod
    def _batch_dones(dones: List[bool]):
        return all(dones)
