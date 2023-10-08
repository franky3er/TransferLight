from abc import ABC, abstractmethod
import itertools
import math
import os
import random
import sys
from typing import Any, Dict, List, Tuple, Union, NamedTuple
import xml.etree.ElementTree as ET

import libsumo as traci
import numpy as np
import torch
from torch import multiprocessing as mp
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.data.data import BaseData

from src.callbacks.environment_callbacks import VehicleStatsCallback, IntersectionStatsCallback
from src.params import ACTION_TIME, YELLOW_CHANGE_TIME, ALL_RED_TIME
from src.rl.problem_formulations import ProblemFormulation
from src.sumo.net import read_traffic_net


class AgentMetadata(NamedTuple):
    name: str
    n_actions: int
    action_names: List[str]


class MarlEnvMetaData(NamedTuple):
    name: str
    scenario_path: str
    scenario_name: str
    episode: int
    episode_step: int
    total_step: int
    episode_time: int
    total_time: int
    n_agents: int
    n_actions: int
    action_offsets: List[int]
    agents_metadata: List[AgentMetadata]


class MPMarlEnvMetaData(NamedTuple):
    n_workers: int
    agent_offsets: List[int]
    action_offsets: List[int]
    workers_metadata: List[MarlEnvMetaData]


class Environment(ABC):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, Environment)
        return obj

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
    def close(self):
        pass

    @abstractmethod
    def metadata(self) -> Dict:
        pass

    @abstractmethod
    def setup_scenarios(self, scenario_path: str = None, scenarios_dir: str = None):
        pass


class MarlEnvironment(Environment):

    def __init__(self, name: str = None, scenario_path: str = None, scenarios_dirs: Union[str, List[str]] = None,
                 max_patience: int = sys.maxsize, max_time: int = sys.maxsize, problem_formulation: str = None,
                 use_default: bool = False, action_time: int = ACTION_TIME,
                 yellow_change_time: int = YELLOW_CHANGE_TIME, all_red_time: int = ALL_RED_TIME,
                 demo: bool = False, stats_dir: str = None):
        self.name = "1" if name is None else name
        self.scenarios, self.scenario = None, None
        self.setup_scenarios(scenario_path, scenarios_dirs)
        self.problem_formulation_name = problem_formulation
        self.problem_formulation = None
        self.net_xml_path = None
        self.net = None
        self.sumo = "sumo-gui" if demo else "sumo"
        self.max_patience = max_patience
        self.max_time = max_time
        self.use_default = use_default
        self.episode = -1
        self.total_step, self.total_time, self.episode_step, self.episode_time = 0, 0, 0, 0
        self.rewards = None
        self.action_time, self.yellow_change_time, self.all_red_time = action_time, yellow_change_time, all_red_time
        self.callbacks = []
        if stats_dir is not None:
            self.callbacks.append(VehicleStatsCallback(stats_dir))
            self.callbacks.append(IntersectionStatsCallback(stats_dir))

    def setup_scenarios(self, scenario_path: str = None, scenarios_dirs: str = None):
        assert (scenario_path is not None) != (scenarios_dirs is not None)
        if scenario_path is not None:
            assert scenario_path.endswith(".sumocfg")
            scenarios = [scenario_path]
        else:
            scenarios_dirs = [scenarios_dirs] if isinstance(scenarios_dirs, str) else scenarios_dirs
            scenarios = [os.path.join(scenarios_dirs, scenario)
                         for scenarios_dir in scenarios_dirs
                         for scenario in os.listdir(scenarios_dir)
                         if os.path.isfile(os.path.join(scenarios_dir, scenario))
                         and scenario.endswith(".sumocfg")]
        self.scenarios = itertools.cycle(scenarios)
        self.scenario = None

    def reset(self) -> BaseData:
        self.close()
        self.episode += 1
        self.episode_step = 0
        self.episode_time = 0
        self.scenario = next(self.scenarios)
        scenario_dir = os.path.dirname(self.scenario)
        net_xml_path = os.path.join(scenario_dir,
                                    ET.parse(self.scenario).getroot().find("input").find("net-file").attrib["value"])
        sumo_cmd = [self.sumo, "-c", self.scenario, "--time-to-teleport", str(-1), "--no-warnings"]
        traci.start(sumo_cmd)
        if not self.use_default:
            for intersection in traci.trafficlight.getIDList():
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(intersection)[0]
                new_phases = []
                for phase in logic.phases:
                    if "y" in phase.state or len([c for c in [*phase.state] if c != "r"]) == 0:
                        continue
                    new_phases.append(traci.trafficlight.Phase(9999999, phase.state))
                random_phase_idx = random.randrange(len(new_phases))
                new_logic = traci.trafficlight.Logic(f"{logic.programID}-new", logic.type, random_phase_idx, new_phases)
                traci.trafficlight.setCompleteRedYellowGreenDefinition(intersection, new_logic)
        self.net = read_traffic_net(net_xml_path, withPrograms=True)
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
        max_waiting_time = 0.0 if len(traci.vehicle.getIDList()) == 0 \
            else np.max([traci.vehicle.getWaitingTime(vehID=veh_id) for veh_id in traci.vehicle.getIDList()])
        max_patience_exceeded = max_waiting_time > self.max_patience
        max_time_exceeded = self.episode_time > self.max_time
        if max_patience_exceeded:
            print(f"Max patience exceeded: ({self.scenario})")
        done = (True if traci.simulation.getMinExpectedNumber() == 0 or max_patience_exceeded or max_time_exceeded
                else False)
        [callback.on_episode_end(self) for callback in self.callbacks if done]
        if return_info:
            return state, rewards, done, self.info()
        return state, rewards, done

    def state(self) -> Batch:
        return self.problem_formulation.get_state()

    def metadata(self) -> MarlEnvMetaData:
        agents_metadata = []
        for intersection in self.net.signalized_intersections:
            phases = self.net.get_phases(intersection)
            agent_metadata = AgentMetadata(
                name=intersection,
                n_actions=len(phases),
                action_names=[signal for _, signal in phases]
            )
            agents_metadata.append(agent_metadata)

        metadata = MarlEnvMetaData(
            name=self.name,
            scenario_path=self.scenario,
            scenario_name=os.path.join(*os.path.normpath(self.scenario).split(os.path.sep)[-2:]).split(".")[0],
            episode=self.episode,
            episode_step=self.episode_step,
            total_step=self.total_step,
            episode_time=self.episode_time,
            total_time=self.total_time,
            n_agents=len(self.net.signalized_intersections),
            n_actions=sum([len(self.net.get_phases(intersection))
                           for intersection in self.net.signalized_intersections]),
            action_offsets=np.cumsum([len(self.net.get_phases(intersection))
                                      for intersection in self.net.signalized_intersections]).tolist(),
            agents_metadata=agents_metadata
        )
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
        [callback.on_action_step_start(self) for callback in self.callbacks]
        if self.use_default:
            self._apply_default_actions()
        else:
            self._apply_controlled_actions(actions)
        self.episode_step += 1
        self.total_step += 1
        [callback.on_action_step_end(self) for callback in self.callbacks]

    def _apply_default_actions(self):
        for _ in range(self.action_time):
            [callback.on_time_step_start(self) for callback in self.callbacks]
            traci.simulationStep()
            self.episode_time += 1
            self.total_time += 1
            [callback.on_time_step_end(self) for callback in self.callbacks]

    def _apply_controlled_actions(self, actions: List[int]):
        actions = [actions] if isinstance(actions, int) else actions
        signalized_intersections = self.net.signalized_intersections
        current_phases = [self.net.get_current_phase(intersection) for intersection in signalized_intersections]
        next_phases = [self.net.get_phases(intersection)[action]
                       for intersection, action in zip(signalized_intersections, actions)]
        transition_states = [self._get_transition_states(current_phase[1], next_phase[1])
                             for current_phase, next_phase in zip(current_phases, next_phases)]
        green_states = [next_phase[1] for next_phase in next_phases]
        for t in range(self.action_time):
            [callback.on_time_step_start(self) for callback in self.callbacks]
            for signalized_intersection, transition_state_ in zip(signalized_intersections, transition_states):
                traci.trafficlight.setRedYellowGreenState(signalized_intersection, transition_state_[t])
            traci.simulationStep()
            self.episode_time += 1
            self.total_time += 1
            [callback.on_time_step_end(self) for callback in self.callbacks]
        for signalized_intersection, green_state in zip(signalized_intersections, green_states):
            traci.trafficlight.setRedYellowGreenState(signalized_intersection, green_state)

    def _get_transition_states(self, current_state: str, next_state: str) -> List[str]:
        next_yellow_state, next_red_state, next_green_state = [], [], next_state
        for current_signal, next_signal in zip([*current_state], [*next_state]):
            if (current_signal == "g" or current_signal == "G") and next_signal == "r":
                next_yellow_state.append("y")
                next_red_state.append("r")
            else:
                next_yellow_state.append(current_signal)
                next_red_state.append(current_signal)
        next_yellow_state, next_red_state = "".join(next_yellow_state), "".join(next_red_state)
        transition_signals = []
        for _ in range(self.yellow_change_time):
            transition_signals.append(next_yellow_state)
        for _ in range(self.all_red_time):
            transition_signals.append(next_red_state)
        for _ in range(self.action_time - self.yellow_change_time - self.all_red_time):
            transition_signals.append(next_green_state)
        return transition_signals


class MultiprocessingMarlEnvironment(Environment):

    def __init__(self, scenario_path: str = None, scenarios_dirs: str = None, cycle_scenarios: bool = True,
                 max_patience: int = sys.maxsize, max_time: int = sys.maxsize, problem_formulation: str = None,
                 n_workers: int = 1, use_default: bool = False, action_time: int = ACTION_TIME,
                 yellow_change_time: int = YELLOW_CHANGE_TIME, all_red_time: int = ALL_RED_TIME,
                 stats_dir: str = None):
        self.n_workers = n_workers
        self.scenarios, self.scenario, self.n_scenarios = None, None, None
        self.cycle_scenarios = cycle_scenarios
        self.setup_scenarios(scenario_path, scenarios_dirs)
        self.problem_formulation_name = problem_formulation
        self.max_patience = max_patience
        self.max_time = max_time
        self.use_default = use_default
        self.demo = False
        self.stats_dir = stats_dir
        self.pipes = [mp.Pipe() for _ in range(self.n_workers)]
        self.action_time, self.yellow_change_time, self.all_red_time = action_time, yellow_change_time, all_red_time
        self.workers = [mp.Process(target=self.work, args=self._get_work_args(rank))
                        for rank in range(self.n_workers)]
        [worker.start() for worker in self.workers]
        self.worker_done = [False for _ in range(self.n_workers)]
        self.worker_n_agents = None
        self.worker_n_actions = None
        self.agent_worker_assignment = None
        self.rewards = None
        self.ema_reward = 0.0
        self.ema_weight = 0.1
        self.episode = [-1 for _ in range(self.n_workers)]
        self.episode_step = [0 for _ in range(self.n_workers)]
        self.total_step = 0

    def setup_scenarios(self, scenario_path: str = None, scenarios_dirs: Union[str, List[str]] = None):
        assert (scenario_path is not None) != (scenarios_dirs is not None)
        if scenario_path is not None:
            assert scenario_path.endswith(".sumocfg")
            scenarios = [scenario_path]
        else:
            scenarios_dirs = [scenarios_dirs] if isinstance(scenarios_dirs, str) else scenarios_dirs
            scenarios = [os.path.join(scenarios_dir, scenario)
                         for scenarios_dir in scenarios_dirs
                         for scenario in os.listdir(scenarios_dir)
                         if os.path.isfile(os.path.join(scenarios_dir, scenario))
                         and scenario.endswith(".sumocfg")]
            print(f"found {len(scenarios)} scenarios")
        if self.scenarios is None:
            self.scenarios = mp.Queue()
        else:
            # Flush scenarios queue
            while self.scenarios.full():
                _ = self.scenarios.get()
        if self.cycle_scenarios and self.n_workers > len(scenarios):
            repeats = math.ceil(self.n_workers / len(scenarios))
            scenarios = scenarios * repeats
        [self.scenarios.put(scenario) for scenario in scenarios]
        if not self.cycle_scenarios:
            self.scenarios.put("ALL DONE")

    def _get_work_args(self, rank: int):
        return (rank, self.pipes[rank][1], self.scenarios, self.cycle_scenarios, self.problem_formulation_name,
                self.max_patience, self.max_time, self.use_default, self.action_time, self.yellow_change_time,
                self.all_red_time, self.stats_dir)

    @staticmethod
    def work(rank, worker_end, scenarios, cycle_scenarios, problem_formulation, max_patience, max_time, use_default,
             action_time, yellow_change_time, all_red_time, stats_dir):
        print(f"Worker {rank} started")
        env = None
        while True:
            cmd, kwargs = worker_end.recv()
            if cmd == "reset":
                scenario = scenarios.get()
                if cycle_scenarios:
                    scenarios.put(scenario)
                if scenario == "ALL DONE":
                    scenarios.put(scenario)
                    worker_end.send("ALL DONE")
                    continue
                print(f"Worker {rank} reset: {scenario}")
                if env is None:
                    env = MarlEnvironment(name=rank, scenario_path=scenario, max_patience=max_patience,
                                          max_time=max_time, problem_formulation=problem_formulation,
                                          use_default=use_default, action_time=action_time,
                                          yellow_change_time=yellow_change_time, all_red_time=all_red_time,
                                          stats_dir=stats_dir)
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
                if not env is None:
                    env.close()
                    del env
                worker_end.close()
                del worker_end
                while True:
                    scenario = scenarios.get()
                    if scenario == "ALL DONE":
                        scenarios.put(scenario)
                        break
                break
        print(f"Worker {rank} closed")

    def reset(self) -> BaseData:
        self.broadcast_msg(("reset", {}))
        states = []
        for rank in range(self.n_workers):
            if self.worker_done[rank]:
                continue
            parent_end, _ = self.pipes[rank]
            state = parent_end.recv()
            if state == "ALL DONE":
                self.worker_done[rank] = True
                self.send_msg(("close", {}), rank)
                continue
            states.append(state)
            self.episode[rank] += 1
            self.episode_step[rank] = 0
        if all(self.worker_done):
            return None
        self._update_internal_params()
        states = self._batch_states(states)
        return states

    def step(self, actions: List[int]) -> Tuple[BaseData, torch.Tensor, bool]:
        self.total_step += 1
        actions = self._distribute_actions(actions)
        [self.send_msg(("step", {"actions": actions[rank], "return_info": True}), rank)
         for rank in range(self.n_workers) if not self.worker_done[rank]]
        worker_done = False
        states, rewards, dones = [], [], []
        for rank in range(self.n_workers):
            if self.worker_done[rank]:
                continue
            self.episode_step[rank] += 1
            parent_end, _ = self.pipes[rank]
            state, reward, done, info = parent_end.recv()
            if done:
                self.episode[rank] += 1
                self.episode_step[rank] = 0
                self.send_msg(("reset", {}), rank)
                msg = parent_end.recv()
                if msg == "ALL DONE":
                    self.worker_done[rank] = True
                    self.send_msg(("close", {}), rank)
                    worker_done = True
                    continue
            states.append(state)
            rewards.append(reward)
            dones.append(done)
        if any(dones) or worker_done:
            self._update_internal_params()
        if all(self.worker_done):
            return None, None, None
        states, rewards, done = self._batch_states(states), self._batch_rewards(rewards), self._batch_dones(dones)
        self.ema_reward = self.ema_weight * torch.mean(rewards).item() + (1.0 - self.ema_weight) * self.ema_reward
        return states, rewards, done

    def state(self) -> BaseData:
        self.broadcast_msg(("state", {}))
        states = []
        for rank in range(self.n_workers):
            if self.worker_done[rank]:
                continue
            parent_end, _ = self.pipes[rank]
            state = parent_end.recv()
            states.append(state)
        if all(self.worker_done):
            return None
        states = self._batch_states(states)
        return states

    def metadata(self) -> MPMarlEnvMetaData:
        self.broadcast_msg(("metadata", {}))
        n_agents, n_actions, workers_metadata = [], [], []
        for rank in range(self.n_workers):
            if self.worker_done[rank]:
                continue
            parent_end, _ = self.pipes[rank]
            md = parent_end.recv()
            n_agents.append(md.n_agents)
            n_actions.append(md.n_actions)
            workers_metadata.append(md)
        metadata = MPMarlEnvMetaData(
            n_workers=sum([not done for done in self.worker_done]),
            agent_offsets=np.cumsum(n_agents).tolist(),
            action_offsets=np.cumsum(n_actions).tolist(),
            workers_metadata=workers_metadata
        )
        return metadata

    def info(self) -> Dict:
        info = dict()
        info["episode"] = self.episode
        info["episode_step"] = self.episode_step
        info["total_step"] = self.total_step
        info["ema_reward"] = self.ema_reward
        episode = np.min([self.episode[rank] for rank in range(self.n_workers)
                          if not self.worker_done[rank]]) if not self.all_done() else None
        info["progress"] = (f'--- Episode: {episode}  '
                            f'Total Step: {info["total_step"]}  EMA Reward: {info["ema_reward"]} ---')
        return info

    def all_done(self) -> bool:
        return all(self.worker_done)

    def _update_internal_params(self):
        self.worker_n_agents = []
        self.worker_n_actions = []
        self.agent_worker_assignment = []
        self.broadcast_msg(("metadata", {}))
        for rank in range(self.n_workers):
            if self.worker_done[rank]:
                self.worker_n_agents.append(0)
                self.worker_n_actions.append(0)
                continue
            parent_end, _ = self.pipes[rank]
            metadata = parent_end.recv()
            self.worker_n_agents.append(metadata.n_agents)
            self.worker_n_actions.append(metadata.n_actions)
        self.agent_worker_assignment = list(itertools.chain(*([rank for _ in range(n_agents)]
                                                              for rank, n_agents in enumerate(self.worker_n_agents)
                                                              if not self.worker_done[rank])))

    def _distribute_actions(self, actions: List[int]) -> List[List[int]]:
        actions_ = [[] for _ in range(self.n_workers)]
        for assignment, action in zip(self.agent_worker_assignment, actions):
            actions_[assignment].append(action)
        return actions_

    def close(self):
        if self.cycle_scenarios:
            self.scenarios.put("ALL DONE")
        self.broadcast_msg(("close", {}))
        self.scenarios.close()
        self.scenarios.join_thread()
        for worker in self.workers:
            worker.join()
            worker.close()
            del worker
        for pipe in self.pipes:
            parent_end, _ = pipe
            parent_end.close()
            del parent_end

    def send_msg(self, msg, rank):
        parent_end, _ = self.pipes[rank]
        parent_end.send(msg)

    def broadcast_msg(self, msg):
        [self.pipes[rank][0].send(msg) for rank in range(self.n_workers) if not self.worker_done[rank]]

    @staticmethod
    def _batch_states(states: List[Union[Data, HeteroData]]) -> Batch:
        return Batch.from_data_list(states)

    @staticmethod
    def _batch_rewards(rewards: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(rewards, dim=0)

    @staticmethod
    def _batch_dones(dones: List[bool]) -> bool:
        return any(dones)
