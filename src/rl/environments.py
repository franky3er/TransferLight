from abc import ABC, abstractmethod
import itertools
import os
import random
from typing import Any, List, Tuple

import libsumo as traci
import sumolib.net
import torch
from torch_geometric.data import HeteroData

from src.params import ENV_ACTION_EXECUTION_TIME, ENV_YELLOW_TIME, ENV_RED_TIME
from src.traffic.traffic_representation import TrafficRepresentation


class MarlEnvironment(ABC):

    @abstractmethod
    def reset(self) -> HeteroData:
        pass

    @abstractmethod
    def step(self, actions: List[int]) -> Tuple[Any, torch.Tensor, bool]:
        pass

    @abstractmethod
    def close(self):
        pass


class TscMarlEnvironment(MarlEnvironment):

    def __init__(self, scenarios_dir: str, max_steps: int, traffic_representation: str, use_default: bool = False,
                 demo: bool = False):
        scenarios = []
        for scenario_dir in os.listdir(scenarios_dir):
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

    def reset(self) -> HeteroData:
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
        state = self.traffic_representation.get_state()
        return state

    def close(self):
        traci.close()
        self.traffic_representation = None
        self.net = None
        self.current_step = 0

    def step(self, actions: List[int]) -> Tuple[Any, torch.Tensor, bool]:
        self._apply_actions(actions)
        state = self.traffic_representation.get_state()
        rewards = - torch.tensor(self.traffic_representation.get_total_queue_lengths())
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
        tls_junctions = self.traffic_representation.get_tls_junctions()
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
