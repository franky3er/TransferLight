from abc import ABC, abstractmethod
import itertools
import os
import random
from typing import Any, Dict, List, Tuple, Union

import libsumo as traci
import numpy as np
import pandas as pd
import sumolib.net
import torch
from torch import multiprocessing as mp
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.data.data import BaseData

from src.params import ENV_ACTION_EXECUTION_TIME, ENV_YELLOW_TIME, ENV_RED_TIME
from src.rl.problem_formulations import ProblemFormulation


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
    def statistics(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def close(self):
        pass


class MarlEnvironment(Environment):

    def __init__(self, scenarios_dir: str, max_patience: int, problem_formulation: str, use_default: bool = False,
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
        self.net_xml_path = None
        self.scenarios = itertools.cycle(scenarios)
        self.problem_formulation_name = problem_formulation
        self.problem_formulation = None
        self.net = None
        self.sumo = "sumo-gui" if demo else "sumo"
        self.max_patience = max_patience
        self.use_default = use_default
        self.episode = None
        self.action_step = 0
        self.time_step = 0
        self.episode_action_step = 0
        self.episode_time_step = 0
        self.trip_statistics = None
        self.intersection_statistics = None
        self.rewards = None

    def reset(self) -> BaseData:
        self.close()
        self.episode = 0 if self.episode is None else self.episode + 1
        self.episode_action_step = 0
        self.episode_time_step = 0
        net_xml_path, rou_xml_path = next(self.scenarios)
        self.net_xml_path = net_xml_path
        sumo_cmd = [self.sumo, "-n", net_xml_path, "-r", rou_xml_path, "--time-to-teleport", str(-1), "--no-warnings"]
        traci.start(sumo_cmd)
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
        self.problem_formulation = ProblemFormulation.create(self.problem_formulation_name, self.net)
        state = self.problem_formulation.get_state()
        return state

    def close(self):
        traci.close()
        self.problem_formulation = None
        self.net = None
        self.episode_action_step = 0
        self.episode_time_step = 0

    def step(self, actions: List[int], return_info: bool = False) \
            -> Union[Tuple[BaseData, torch.Tensor, bool], Tuple[BaseData, torch.Tensor, bool, Dict]]:
        self._apply_actions(actions)
        state = self.problem_formulation.get_state()
        rewards = self.problem_formulation.get_rewards()
        self.rewards = rewards
        max_waiting_time = self.problem_formulation.get_max_vehicle_waiting_time()
        if max_waiting_time > self.max_patience:
            print(f"max_waiting_time > max_patience   ({self.net_xml_path})")
        done = (True if traci.simulation.getMinExpectedNumber() == 0 or max_waiting_time > self.max_patience
                else False)
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
            "step": self.action_step,
            "episode_step": self.episode_action_step,
            "ema_reward": None
        }
        return info

    def statistics(self) -> pd.DataFrame:
        pass

    def _apply_actions(self, actions: List[int]):
        if self.use_default:
            self._apply_default_actions()
        else:
            self._apply_controlled_actions(actions)
        self.episode_action_step += 1
        self.action_step += 1

    def _update_statistics(self):
        self._update_trip_statistics()
        self._update_intersection_statistics()

    def _update_trip_statistics(self):
        trip_statistics = []
        for vehicle in traci.vehicle.getIDList():
            trip_statistics.append({
                "net_xml_path": self.net_xml_path,
                "episode": self.episode,
                "time_step": self.time_step,
                "action_step": self.action_step,
                "episode_time_step": self.episode_time_step,
                "episode_action_step": self.episode_action_step,
                "vehicle": vehicle,
                "distance": traci.vehicle.getDistance(vehicle),
                "lane": traci.vehicle.getLaneID(vehicle),
                "speed": traci.vehicle.getSpeed(vehicle),
                "waiting_time": traci.vehicle.getWaitingTime(vehicle)
            })
        trip_statistics = pd.DataFrame(data=trip_statistics)
        self.trip_statistics = trip_statistics if self.trip_statistics is None \
            else pd.concat([self.trip_statistics, trip_statistics])

    def _update_intersection_statistics(self):
        intersection_statistics = []
        for intersection in traci.trafficlight.getIDList():
            intersection_statistics.append({
                "net_xml_path": self.net_xml_path,
                "episode": self.episode,
                "time_step": self.time_step,
                "action_step": self.action_step,
                "episode_time_step": self.episode_time_step,
                "episode_action_step": self.episode_action_step,
                "intersection": intersection,
                "pressure": self.problem_formulation.get_intersection_normalized_pressure(intersection),
                "queue_length": self.problem_formulation.get_intersection_queue_length(intersection)
            })
        intersection_statistics = pd.DataFrame(data=intersection_statistics)
        self.intersection_statistics = intersection_statistics if self.intersection_statistics is None \
            else pd.concat([self.intersection_statistics, intersection_statistics])

    def _apply_default_actions(self):
        for _ in range(ENV_ACTION_EXECUTION_TIME):
            traci.simulationStep()
            self.episode_time_step += 1
            self.time_step += 1
            self._update_statistics()

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
            for signalized_intersection, transition_signals_ in zip(signalized_intersections, transition_signals):
                traci.trafficlight.setRedYellowGreenState(signalized_intersection, transition_signals_[t])
            traci.simulationStep()
            self.episode_time_step += 1
            self.time_step += 1
            self._update_statistics()
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
        self.worker_n_agents = None
        self.worker_n_actions = None
        self.agent_worker_assignment = None
        self.n_steps = None
        self.rewards = None
        self.ema_reward = 0.0
        self.ema_weight = 0.1
        self.action_step = 0
        self.episode = 0

    def _get_work_args(self, rank: int):
        skip_scenarios = [(s-rank) % self.n_workers != 0 for s in range(self.n_scenarios)]
        return rank, self.pipes[rank][1], self.scenarios_dir, self.traffic_representation_name, self.max_steps, \
            self.use_default, skip_scenarios

    def reset(self) -> Batch:
        self.n_steps = 0
        self.broadcast_msg(("reset", {}))
        states = []
        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            state = parent_end.recv()
            states.append(state)
        self._update_internal_params()
        states = self._batch_states(states)
        return states

    def step(self, actions: List[int]) -> Tuple[Batch, torch.Tensor, bool]:
        self.n_steps += 1
        actions = self._distribute_actions(actions)
        [self.send_msg(("step", {"actions": actions[rank], "return_info": True}), rank)
         for rank in range(self.n_workers)]
        states, rewards, dones = [], [], []
        episodes = []
        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            state, reward, done, info = parent_end.recv()
            episodes.append(info["episode"])
            if done:
                self.send_msg(("reset", {}), rank)
                _ = parent_end.recv()
            states.append(state)
            rewards.append(reward)
            dones.append(done)
        if any(dones):
            self._update_internal_params()
        states, rewards, done = self._batch_states(states), self._batch_rewards(rewards), self._batch_dones(dones)
        self.episode = np.array(episodes).min()
        self.action_step += 1 if self.episode > 0 else 0
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
        metadata = {"n_agents": [], "n_actions": [], "episode": [], "episode_action_step": [], "episode_time_step": []}
        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            md = parent_end.recv()
            metadata["n_agents"].append(md["n_agents"])
            metadata["n_actions"].append(md["n_actions"])
        #metadata["episode"] = np.array(metadata["episode"]).min()
        metadata["agent_offsets"] = np.cumsum(metadata["n_agents"]).tolist()
        metadata["action_offsets"] = np.cumsum(metadata["n_actions"]).tolist()
        metadata["n_workers"] = self.n_workers
        return metadata

    def info(self) -> Dict:
        info = dict()
        info["episode"] = self.episode
        info["step"] = self.action_step
        info["ema_reward"] = self.ema_reward
        info["progress"] = f'--- Episode: {info["episode"]}  Step: {info["step"]}  EMA Reward: {info["ema_reward"]} ---'
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

    def statistics(self) -> pd.DataFrame:
        pass

    @staticmethod
    def work(rank, worker_end, scenarios_dir, traffic_representation, max_steps, use_default, skip_scenarios):
        env = MarlEnvironment(scenarios_dir, max_steps, traffic_representation, use_default,
                              skip_scenarios=skip_scenarios)
        print(f"Worker {rank} started")
        while True:
            cmd, kwargs = worker_end.recv()
            if cmd == "reset":
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
