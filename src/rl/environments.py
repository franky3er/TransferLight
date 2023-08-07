from abc import ABC, abstractmethod
import itertools
import os
import random
import sys
from typing import Any, Dict, List, Tuple, Union
import xml.etree.ElementTree as ET

import libsumo as traci
import numpy as np
import sumolib.net
import torch
from torch import multiprocessing as mp
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.data.data import BaseData

from src.callbacks.environment_callbacks import VehicleStats, IntersectionStats
from src.params import ENV_ACTION_EXECUTION_TIME, ENV_YELLOW_TIME, ENV_RED_TIME
from src.rl.problem_formulations import ProblemFormulation
from src.sumo.net import readNetState


class Environment(ABC):

    @abstractmethod
    def reset(self) -> Any:
        pass

    @abstractmethod
    def step(self, actions: List[int]) -> Tuple[Any, torch.Tensor, bool]:
        pass

    @abstractmethod
    def state(self) -> BaseData:
        pass

    @abstractmethod
    def metadata(self) -> Dict:
        pass

    @abstractmethod
    def close(self):
        pass


class MarlEnvironment(Environment):

    def __init__(self, name: str = None, scenario_path: str = None, scenarios_dir: str = None,
                 max_patience: int = sys.maxsize, problem_formulation: str = None, use_default: bool = False,
                 demo: bool = False):
        self.name = "1" if name is None else name
        self.scenarios, self.scenarios_dir, self.scenario = None, None, None
        self.setup_scenarios(scenario_path, scenarios_dir)
        self.problem_formulation_name = problem_formulation
        self.problem_formulation = None
        self.net_xml_path = None
        self.net = None
        self.sumo = "sumo-gui" if demo else "sumo"
        self.max_patience = max_patience
        self.use_default = use_default
        self.episode = -1
        self.total_step, self.total_time, self.episode_step, self.episode_time = 0, 0, 0, 0
        self.rewards = None
        self.callbacks = [VehicleStats("results/vehicle"), IntersectionStats("results/intersection")]

    def setup_scenarios(self, scenario_path: str = None, scenarios_dir: str = None):
        assert (scenario_path is not None) != (scenarios_dir is not None)
        if scenario_path is not None:
            assert scenario_path.endswith(".sumocfg")
            scenarios = [scenario_path]
            self.scenarios_dir = os.path.dirname(scenario_path)
        else:
            scenarios = [os.path.join(scenarios_dir, scenario) for scenario in os.listdir(scenarios_dir)
                         if os.path.isfile(os.path.join(scenarios_dir, scenario))
                         and scenario.endswith(".sumocfg")]
            self.scenarios_dir = scenarios_dir
        self.scenarios = itertools.cycle(scenarios)
        self.scenario = None

    def reset(self) -> BaseData:
        self.close()
        self.episode += 1
        self.episode_step = 0
        self.episode_time = 0
        self.scenario = next(self.scenarios)
        net_xml_path = os.path.join(self.scenarios_dir,
                                    ET.parse(self.scenario).getroot().find("input").find("net-file").attrib["value"])
        self.net = readNetState(net_xml_path)
        sumo_cmd = [self.sumo, "-c", self.scenario, "--time-to-teleport", str(-1), "--no-warnings"]
        traci.start(sumo_cmd)
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
        self.problem_formulation = ProblemFormulation.create(self.problem_formulation_name, self.net)
        state = self.problem_formulation.get_state()
        [callback.on_episode_start(self) for callback in self.callbacks]
        return state

    def close(self):
        [callback.on_close(self) for callback in self.callbacks]
        traci.close()
        self.problem_formulation = None
        self.net = None
        self.episode_step = 0
        self.episode_time = 0

    def step(self, actions: List[int], return_info: bool = False) \
            -> Union[Tuple[BaseData, torch.Tensor, bool], Tuple[BaseData, torch.Tensor, bool, Dict]]:
        self._apply_actions(actions)
        state = self.problem_formulation.get_state()
        rewards = self.problem_formulation.get_rewards()
        self.rewards = rewards
        max_waiting_time = self.problem_formulation.get_max_vehicle_waiting_time()
        if max_waiting_time > self.max_patience:
            print(f"max_waiting_time > max_patience   ({self.scenario})")
        done = (True if traci.simulation.getMinExpectedNumber() == 0 or max_waiting_time > self.max_patience
                else False)
        [callback.on_episode_end(self) for callback in self.callbacks if done]
        if return_info:
            return state, rewards, done, self.info()
        return state, rewards, done

    def state(self) -> Batch:
        return self.problem_formulation.get_state()

    def metadata(self) -> Dict:
        signalized_intersections = self.problem_formulation.get_signalized_intersections()
        metadata = {
            "n_agents": len(signalized_intersections),
            "n_actions": sum([len(self.problem_formulation.get_phases(intersection_id))
                              for intersection_id in signalized_intersections])
        }
        return metadata

    def info(self) -> Dict:
        info = {
            "episode": self.episode,
            "total_step": self.total_step,
            "episode_step": self.episode_step,
            "ema_reward": None
        }
        return info

    def _apply_actions(self, actions: List[int]):
        if self.use_default:
            self._apply_default_actions()
        else:
            self._apply_controlled_actions(actions)
        self.episode_step += 1
        self.total_step += 1

    def _apply_default_actions(self):
        for _ in range(ENV_ACTION_EXECUTION_TIME):
            [callback.on_step_start(self) for callback in self.callbacks]
            traci.simulationStep()
            self.episode_time += 1
            self.total_time += 1
            [callback.on_step_end(self) for callback in self.callbacks]

    def _apply_controlled_actions(self, actions: List[int]):
        actions = [actions] if isinstance(actions, int) else actions
        previous_actions = self.problem_formulation.get_current_phases()
        signalized_intersections = self.problem_formulation.get_signalized_intersections()
        transition_signals = [self._get_transition_signals(intersection, prev_action, action)
                              for intersection, prev_action, action
                              in zip(signalized_intersections, previous_actions, actions)]
        green_signals = [self._get_signal(intersection, action)
                         for intersection, action in zip(signalized_intersections, actions)]
        for t in range(ENV_ACTION_EXECUTION_TIME):
            [callback.on_step_start(self) for callback in self.callbacks]
            for signalized_intersection, transition_signals_ in zip(signalized_intersections, transition_signals):
                traci.trafficlight.setRedYellowGreenState(signalized_intersection, transition_signals_[t])
            traci.simulationStep()
            self.episode_time += 1
            self.total_time += 1
            [callback.on_step_end(self) for callback in self.callbacks]
        for signalized_intersection, green_signal in zip(signalized_intersections, green_signals):
            traci.trafficlight.setRedYellowGreenState(signalized_intersection, green_signal)

    def _get_transition_signals(self, intersection: str, current_phase: int, next_phase: int) -> List[str]:
        current_green_signal = self._get_signal(intersection, current_phase)
        next_green_signal = self._get_signal(intersection, next_phase)
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


class MultiprocessingMarlEnvironment(Environment):

    def __init__(self, scenario_path: str = None, scenarios_dir: str = None, max_patience: int = sys.maxsize,
                 problem_formulation: str = None, n_workers: int = 1, use_default: bool = False):
        self.scenarios, self.scenarios_dir, self.scenario = None, None, None
        self.setup_scenarios(scenario_path, scenarios_dir)
        self.problem_formulation_name = problem_formulation
        self.max_patience = max_patience
        self.n_workers = n_workers
        self.use_default = use_default
        self.demo = False
        self.pipes = [mp.Pipe() for _ in range(self.n_workers)]
        self.workers = [mp.Process(target=self.work, args=self._get_work_args(rank))
                        for rank in range(self.n_workers)]
        [worker.start() for worker in self.workers]
        self.worker_n_agents = None
        self.worker_n_actions = None
        self.agent_worker_assignment = None
        self.rewards = None
        self.ema_reward = 0.0
        self.ema_weight = 0.1
        self.episode = [-1 for _ in range(self.n_workers)]
        self.episode_step = [0 for _ in range(self.n_workers)]
        self.total_step = 0

    def setup_scenarios(self, scenario_path: str = None, scenarios_dir: str = None):
        assert (scenario_path is not None) != (scenarios_dir is not None)
        if scenario_path is not None:
            assert scenario_path.endswith(".sumocfg")
            scenarios = [scenario_path]
            self.scenarios_dir = os.path.dirname(scenario_path)
        else:
            scenarios = [os.path.join(scenarios_dir, scenario) for scenario in os.listdir(scenarios_dir)
                         if os.path.isfile(os.path.join(scenarios_dir, scenario))
                         and scenario.endswith(".sumocfg")]
            self.scenarios_dir = scenarios_dir
        self.scenarios = mp.Queue()
        [self.scenarios.put(scenario) for scenario in scenarios]

    def _get_work_args(self, rank: int):
        return rank, self.pipes[rank][1], self.scenarios, self.problem_formulation_name, self.max_patience, \
            self.use_default

    @staticmethod
    def work(rank, worker_end, scenarios, problem_formulation, max_patience, use_default):
        print(f"Worker {rank} started")
        env = None
        while True:
            cmd, kwargs = worker_end.recv()
            if cmd == "reset":
                scenario = scenarios.get()
                scenarios.put(scenario)
                if env is None:
                    env = MarlEnvironment(name=rank, scenario_path=scenario, max_patience=max_patience,
                                          problem_formulation=problem_formulation, use_default=use_default)
                else:
                    env.setup_scenarios(scenario_path=scenario)
                worker_end.send(env.reset(**kwargs))
            elif cmd == "step":
                worker_end.send(env.step(**kwargs))
            elif cmd == "state":
                worker_end.send(env.state())
            elif cmd == "metadata":
                worker_end.send(env.metadata())
            elif cmd == "info":
                worker_end.send(env.info())
            else:
                env.close()
                del env
                worker_end.close()
                break

    def reset(self) -> Batch:
        self.broadcast_msg(("reset", {}))
        states = []
        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            state = parent_end.recv()
            states.append(state)
            self.episode[rank] += 1
            self.episode_step[rank] = 0
        self._update_internal_params()
        states = self._batch_states(states)
        return states

    def step(self, actions: List[int]) -> Tuple[Batch, torch.Tensor, bool]:
        self.total_step += 1
        actions = self._distribute_actions(actions)
        [self.send_msg(("step", {"actions": actions[rank], "return_info": True}), rank)
         for rank in range(self.n_workers)]
        states, rewards, dones = [], [], []
        for rank in range(self.n_workers):
            self.episode_step[rank] += 1
            parent_end, _ = self.pipes[rank]
            state, reward, done, info = parent_end.recv()
            if done:
                self.episode[rank] += 1
                self.episode_step[rank] = 0
                self.send_msg(("reset", {}), rank)
                _ = parent_end.recv()
            states.append(state)
            rewards.append(reward)
            dones.append(done)
        if any(dones):
            self._update_internal_params()
        states, rewards, done = self._batch_states(states), self._batch_rewards(rewards), self._batch_dones(dones)
        self.ema_reward = self.ema_weight * torch.mean(rewards).item() + (1.0 - self.ema_weight) * self.ema_reward
        return states, rewards, done

    def state(self) -> Batch:
        self.broadcast_msg(("state", {}))
        states = []
        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            state = parent_end.recv()
            states.append(state)
        states = self._batch_states(states)
        return states

    def metadata(self) -> Dict:
        self.broadcast_msg(("metadata", {}))
        metadata = {"n_agents": [], "n_actions": []}
        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            md = parent_end.recv()
            metadata["n_agents"].append(md["n_agents"])
            metadata["n_actions"].append(md["n_actions"])
        metadata["agent_offsets"] = np.cumsum(metadata["n_agents"]).tolist()
        metadata["action_offsets"] = np.cumsum(metadata["n_actions"]).tolist()
        metadata["n_workers"] = self.n_workers
        return metadata

    def info(self) -> Dict:
        info = dict()
        info["episode"] = self.episode
        info["episode_step"] = self.episode_step
        info["total_step"] = self.total_step
        info["ema_reward"] = self.ema_reward
        info["progress"] = (f'--- Episode: {np.min(self.episode)}  '
                            f'Total Step: {info["total_step"]}  EMA Reward: {info["ema_reward"]} ---')
        return info

    def _update_internal_params(self):
        self.worker_n_agents = []
        self.worker_n_actions = []
        self.agent_worker_assignment = []
        self.broadcast_msg(("metadata", {}))
        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            metadata = parent_end.recv()
            self.worker_n_agents.append(metadata["n_agents"])
            self.worker_n_actions.append(metadata["n_actions"])
        self.agent_worker_assignment = list(itertools.chain(*([rank for _ in range(n_agents)]
                                                              for rank, n_agents in enumerate(self.worker_n_agents))))

    def _distribute_actions(self, actions: List[int]) -> List[List[int]]:
        actions_ = [[] for _ in range(self.n_workers)]
        for assignment, action in zip(self.agent_worker_assignment, actions):
            actions_[assignment].append(action)
        return actions_

    def close(self):
        self.broadcast_msg(("close", {}))
        [worker.join() for worker in self.workers]

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
        return any(dones)
