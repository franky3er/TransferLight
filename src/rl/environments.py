from abc import ABC, abstractmethod
import itertools
import os
import random
from typing import Any, List, Tuple, Union

import libsumo as traci
import sumolib.net
import torch
from torch import multiprocessing as mp
from torch_geometric.data import Batch, Data, HeteroData

from src.params import ENV_ACTION_EXECUTION_TIME, ENV_YELLOW_TIME, ENV_RED_TIME
from src.traffic.traffic_representation import TrafficRepresentation


class MarlEnvironment(ABC):

    @abstractmethod
    def reset(self) -> Any:
        pass

    @abstractmethod
    def step(self, actions: List[int]) -> Tuple[Any, torch.Tensor, bool]:
        pass

    @abstractmethod
    def close(self):
        pass


class TscMarlEnvironment(MarlEnvironment):

    def __init__(self, scenarios_dir: str, max_steps: int, traffic_representation: str, use_default: bool = False,
                 demo: bool = False, skip_scenarios: List[bool] = None):
        if not skip_scenarios:
            skip_scenarios = [False for _ in range(len(os.listdir(scenarios_dir)))]
        scenarios = []
        for scenario_dir, skip_scenario in zip(os.listdir(scenarios_dir), skip_scenarios):
            if skip_scenario:
                continue
            scenario_dir = os.path.join(scenarios_dir, scenario_dir)
            net_xml_path = os.path.join(scenario_dir, "network.net.xml")
            rou_xml_path = os.path.join(scenario_dir, "routes.rou.xml")
            scenarios.append((net_xml_path, rou_xml_path))
        self.scenarios = itertools.cycle(scenarios)
        self.traffic_representation_name = traffic_representation
        self.traffic_representation = None
        self.net = None
        self.sumo = "sumo-gui" if demo else "sumo"
        self.max_steps = max_steps
        self.use_default = use_default
        self.current_step = 0
        self.current_episode = 0

    def reset(self, return_n_agents: bool = False) -> Any:
        self.close()
        self.current_step = 0
        net_xml_path, rou_xml_path = next(self.scenarios)
        traci.start([self.sumo, "-n", net_xml_path, "-r", rou_xml_path, "--time-to-teleport", str(-1), "--no-warnings"])
        self.net = sumolib.net.readNet(net_xml_path)
        if not self.use_default:
            for tls in self.net.getTrafficLights():
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls.getID())[0]
                new_phases = []
                for phase in logic.phases:
                    if "y" in phase.state or len([c for c in [*phase.state] if c != "r"]) == 0:
                        continue
                    new_phases.append(traci.trafficlight.Phase(9999999, phase.state))
                random_phase_idx = random.randrange(len(new_phases))
                new_logic = traci.trafficlight.Logic(f"{logic.programID}-new", logic.type, random_phase_idx, new_phases)
                traci.trafficlight.setCompleteRedYellowGreenDefinition(tls.getID(), new_logic)
        self.traffic_representation = TrafficRepresentation.create(self.traffic_representation_name, self.net)
        n_agents = len(self.traffic_representation.get_signalized_intersections())
        state = self.traffic_representation.get_state()
        if return_n_agents:
            return state, n_agents
        return state

    def close(self):
        traci.close()
        self.traffic_representation = None
        self.net = None
        self.current_step = 0

    def step(self, actions: List[int]) -> Tuple[Any, torch.Tensor, bool]:
        self._apply_actions(actions)
        state = self.traffic_representation.get_state()
        rewards = self.traffic_representation.get_rewards()
        done = True if traci.simulation.getMinExpectedNumber() == 0 or self.current_step >= self.max_steps else False
        return state, rewards, done

    def _apply_actions(self, actions: List[int]):
        if self.use_default:
            self._apply_default_actions()
        else:
            self._apply_controlled_actions(actions)
        self.current_step += 1

    @staticmethod
    def _apply_default_actions():
        for _ in range(ENV_ACTION_EXECUTION_TIME):
            traci.simulationStep()

    def _apply_controlled_actions(self, actions: List[int]):
        previous_actions = self.traffic_representation.get_current_phases()
        tls_junctions = self.traffic_representation.get_signalized_intersections()
        transition_signals = [self._get_transition_signals(junction_id, prev_action, action)
                              for junction_id, prev_action, action in zip(tls_junctions, previous_actions, actions)]
        for t in range(ENV_ACTION_EXECUTION_TIME):
            for tls_junction_id, tls_transition_signals in zip(tls_junctions, transition_signals):
                traci.trafficlight.setRedYellowGreenState(tls_junction_id, tls_transition_signals[t])
            traci.simulationStep()

    def _get_transition_signals(self, junction_id: str, current_phase: int, next_phase: int) -> List[str]:
        current_green_signal = self._get_signal(junction_id, current_phase)
        next_green_signal = self._get_signal(junction_id, next_phase)
        next_yellow_signal, next_red_signal = [], []
        for current_s, next_s in zip([*current_green_signal], [*next_green_signal]):
            if (current_s == "g" or current_s == "G") and next_s == "r":
                next_yellow_signal.append("y")
                next_red_signal.append("r")
            else:
                next_yellow_signal.append(current_s)
                next_red_signal.append(current_s)
        next_yellow_signal, next_red_signal = "".join(next_yellow_signal), "".join(next_red_signal)
        transition_signals = []
        for _ in range(ENV_YELLOW_TIME):
            transition_signals.append(next_yellow_signal)
        for _ in range(ENV_RED_TIME):
            transition_signals.append(next_red_signal)
        for _ in range(ENV_ACTION_EXECUTION_TIME - ENV_YELLOW_TIME - ENV_RED_TIME):
            transition_signals.append(next_green_signal)
        return transition_signals

    @staticmethod
    def _get_signal(junction_id: str, phase: int):
        return traci.trafficlight.getCompleteRedYellowGreenDefinition(junction_id)[1].phases[phase].state


class MultiprocessingTscMarlEnvironment(MarlEnvironment):

    def __init__(self, scenarios_dir: str, max_steps: int, traffic_representation: str, n_workers: int,
                 use_default: bool = False):
        self.scenarios_dir = scenarios_dir
        self.n_scenarios = len(os.listdir(scenarios_dir))
        self.traffic_representation_name = traffic_representation
        self.max_steps = max_steps
        self.n_workers = n_workers
        self.use_default = use_default
        self.demo = False
        self.pipes = [mp.Pipe() for _ in range(self.n_workers)]
        self.workers = [mp.Process(target=self.work, args=self._get_work_args(rank))
                        for rank in range(self.n_workers)]
        [worker.start() for worker in self.workers]
        self.n_agents = None
        self.agent_worker_assignment = None

    def _get_work_args(self, rank: int):
        skip_scenarios = [(s-rank) % self.n_workers != 0 for s in range(self.n_scenarios)]
        return rank, self.pipes[rank][1], self.scenarios_dir, self.traffic_representation_name, self.max_steps, \
            self.use_default, skip_scenarios

    def reset(self) -> Any:
        self.broadcast_msg(("reset", {"return_n_agents": True}))
        states = []
        self.n_agents = []
        self.agent_worker_assignment = []
        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            state, n_agents = parent_end.recv()
            states.append(state)
            self.n_agents.append(n_agents)
        self.agent_worker_assignment = list(itertools.chain(*([rank for _ in range(n_agents)]
                                                              for rank, n_agents in enumerate(self.n_agents))))
        return self._batch_states(states)

    def step(self, actions: List[int]) -> Tuple[Batch, torch.Tensor, bool]:
        actions = self._distribute_actions(actions)
        [self.send_msg(("step", {"actions": actions[rank]}), rank) for rank in range(self.n_workers)]
        states, rewards, dones = [], [], []
        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            state, rewards_, done = parent_end.recv()
            states.append(state)
            rewards.append(rewards_)
            dones.append(done)
        return self._batch_states(states), self._batch_rewards(rewards), self._batch_dones(dones)

    def _distribute_actions(self, actions: List[int]) -> List[List[int]]:
        actions_ = [[] for _ in range(self.n_workers)]
        for assignment, action in zip(self.agent_worker_assignment, actions):
            actions_[assignment].append(action)
        return actions_

    def close(self):
        self.broadcast_msg(("close", {}))
        [worker.join() for worker in self.workers]

    @staticmethod
    def work(rank, worker_end, scenarios_dir, traffic_representation, max_steps, use_default, skip_scenarios):
        env = TscMarlEnvironment(scenarios_dir, max_steps, traffic_representation, use_default,
                                 skip_scenarios=skip_scenarios)
        print(f"Worker {rank} started")
        while True:
            cmd, kwargs = worker_end.recv()
            if cmd == "reset":
                worker_end.send(env.reset(**kwargs))
            elif cmd == "step":
                worker_end.send(env.step(**kwargs))
            else:
                env.close()
                del env
                worker_end.close()
                break

    def send_msg(self, msg, rank):
        parent_end, _ = self.pipes[rank]
        parent_end.send(msg)

    def broadcast_msg(self, msg):
        [parent_end.send(msg) for parent_end, _ in self.pipes]

    @staticmethod
    def _batch_states(states: List[Union[Data, HeteroData]]) -> Batch:
        return Batch.from_data_list(states)

    @staticmethod
    def _batch_rewards(rewards: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(rewards, dim=0)

    @staticmethod
    def _batch_dones(dones: List[bool]) -> bool:
        return all(dones)
