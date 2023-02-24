from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Data, HeteroData, Batch

from src.data.replay_buffer import ReplayBuffer
from src.models.dqns import LitDQN, HieraGLightDQN
from src.params import ENV_ACTION_EXECUTION_TIME
from src.rl.exploration import ExpDecayEpsGreedyStrategy, ConstantEpsGreedyStrategy

device = "cuda" if torch.cuda.is_available() else "cpu"


class TscMarlAgent(ABC):

    @abstractmethod
    def act(self, state: Any) -> List[int]:
        pass

    @abstractmethod
    def train_step(self, state: Any, actions: List[int], rewards: torch.Tensor, next_state: Any, done: bool):
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint_dir: str):
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_dir: str):
        pass


class HieraGLightAgent(TscMarlAgent):

    def __init__(self, movement_dim: int, phase_dim: int, hidden_dim: int):
        self.dqn_local = HieraGLightDQN(movement_dim, phase_dim, hidden_dim)
        self.dqn_target = HieraGLightDQN(movement_dim, phase_dim, hidden_dim)

    def act(self, state: HeteroData) -> List[int]:
        q_locals = self.dqn_local(state.x_dict, state.edge_index_dict)

    def train_step(self, state: Any, actions: List[int], rewards: torch.Tensor, next_state: Any, done: bool):
        pass

    def save_checkpoint(self, checkpoint_dir: str):
        pass

    def load_checkpoint(self, checkpoint_dir: str):
        pass


class MaxPressureAgent(TscMarlAgent):

    def __init__(self, min_phase_duration: int):
        super(MaxPressureAgent, self).__init__()
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

    def train_step(self, state: Any, actions: List[int], rewards: torch.Tensor, next_state: Any, done: bool):
        pass  # not applicable

    def save_checkpoint(self, checkpoint_dir: str):
        pass  # not applicable

    def load_checkpoint(self, checkpoint_dir: str):
        pass  # not applicable


class LitAgent(TscMarlAgent):

    def __init__(self,
                 state_size: int,
                 hidden_size: int,
                 action_size: int,
                 buffer_size: int,
                 batch_size: int,
                 learning_rate: float,
                 discount_factor: float,
                 tau: float,
                 eps_greedy_start: float,
                 eps_greedy_end: float,
                 eps_greedy_steps: int,
                 act_greedy: bool = False):
        self.qnet_local = LitDQN(state_size, hidden_size, action_size).to(device)
        self.qnet_target = LitDQN(state_size, hidden_size, action_size).to(device)
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.tau = tau
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.optimizer = torch.optim.Adam(self.qnet_local.parameters(), lr=learning_rate)
        self.eps_greedy = ConstantEpsGreedyStrategy(0.0) if act_greedy \
            else ExpDecayEpsGreedyStrategy(eps_greedy_start, eps_greedy_end, eps_greedy_steps)
        self.qnet_local_checkpoint_name = "qnet-local-checkpoint.pt"
        self.qnet_target_checkpoint_name = "qnet-target-checkpoint.pt"

    def act(self, state: Data) -> List[int]:
        x = state.x.to(device)
        with torch.no_grad():
            action_values = self.qnet_local(x)
            greedy_actions = torch.argmax(action_values, dim=-1)
            random_actions = torch.randint(low=0, high=action_values.size(-1), size=(greedy_actions.size(0),),
                                           device=device)
            eps = self.eps_greedy.get_next_eps()
            act_greedy = torch.rand(size=(greedy_actions.size(0),), device=device) > eps
            actions = act_greedy * greedy_actions + ~act_greedy * random_actions
        return actions.cpu().numpy().tolist()

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

    def save_checkpoint(self, checkpoint_dir: str):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        qnet_local_checkpoint_path = os.path.join(checkpoint_dir, self.qnet_local_checkpoint_name)
        qnet_target_checkpoint_path = os.path.join(checkpoint_dir, self.qnet_target_checkpoint_name)
        torch.save(self.qnet_local.state_dict(), qnet_local_checkpoint_path)
        torch.save(self.qnet_target.state_dict(), qnet_target_checkpoint_path)

    def load_checkpoint(self, checkpoint_dir: str):
        qnet_local_checkpoint_path = os.path.join(checkpoint_dir, self.qnet_local_checkpoint_name)
        qnet_target_checkpoint_path = os.path.join(checkpoint_dir, self.qnet_target_checkpoint_name)
        self.qnet_local.load_state_dict(torch.load(qnet_local_checkpoint_path, map_location=device))
        self.qnet_target.load_state_dict(torch.load(qnet_target_checkpoint_path, map_location=device))
