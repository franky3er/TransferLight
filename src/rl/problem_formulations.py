from abc import ABC, abstractmethod
from collections import defaultdict, deque
import itertools
import sys
from typing import Any, List, Tuple

import numpy as np
import sumolib.net.lane
import torch
from sumolib.net import Net
import libsumo as traci
from torch_geometric.data import Data, HeteroData

from src.rl.utils import sort_approaches_clockwise, angle_approach, angle_between_approaches
from src.sumo.net import TrafficNet


def to_edge_index(index: List[Tuple], src: List, dst: List) -> torch.Tensor:
    return torch.tensor([[src.index(i), dst.index(j)] for (i, j) in index], dtype=torch.int64).t().contiguous()


class ProblemFormulation(ABC):

    @classmethod
    def create(cls, class_name: str, net: Net):
        return getattr(sys.modules[__name__], class_name)(net)

    def __init__(self, net: TrafficNet):
        # Node Types
        self.net = net

    @abstractmethod
    def get_state(self) -> Any:
        pass

    @abstractmethod
    def get_rewards(self) -> torch.Tensor:
        pass

    @classmethod
    def get_metadata(cls):
        pass


class MaxPressureProblemFormulation(ProblemFormulation):

    def get_state(self) -> HeteroData:
        data = HeteroData()
        phase_pressures = defaultdict(lambda: 0.0)
        for movement, phase in self.net.index_movement_to_phase:
            signal = self.net.get_movement_signals(phase)[movement]
            if signal not in ["G", "g"]:
                continue
            pressure = self.net.get_pressure(movement=movement, method="density")
            phase_pressures[phase] += pressure
        x = [[phase_pressures[phase]] for phase in self.net.phases]
        data["phase"].x = torch.tensor(x)
        data["intersection"].x = torch.zeros(len(self.net.signalized_intersections), 1)
        data["phase", "to", "intersection"].edge_index = to_edge_index(
            self.net.index_phase_to_intersection, self.net.phases, self.net.signalized_intersections)
        return data

    def get_rewards(self) -> torch.Tensor:
        return - torch.tensor([self.net.get_pressure(intersection=intersection, method="density")
                               for intersection in self.net.signalized_intersections], dtype=torch.float32)


class SimpleTransferLightProblemFormulation(ProblemFormulation):

    def __init__(self, *args, segment_length: float = 25.0):
        super(SimpleTransferLightProblemFormulation, self).__init__(*args)
        self.net.init_segments(segment_length=segment_length)

    @classmethod
    def get_metadata(cls):
        metadata = dict()

        node_dim = defaultdict(lambda: 0)
        node_dim["segment"] = 1
        node_dim["movement"] = 3
        node_dim["phase"] = 1
        metadata["node_dim"] = node_dim

        edge_dim = defaultdict(lambda: 0)
        edge_dim[("movement", "to", "phase")] = 3
        edge_dim[("movement", "to", "movement")] = 2
        edge_dim[("phase", "to", "phase")] = 1
        metadata["edge_dim"] = edge_dim

        pos = defaultdict(lambda: False)
        metadata["pos"] = pos

        return metadata

    def get_rewards(self) -> torch.Tensor:
        return - torch.tensor([self.net.get_pressure(intersection=intersection, method="density")
                               for intersection in self.net.signalized_intersections], dtype=torch.float32)

    def get_state(self) -> HeteroData:
        data = HeteroData()
        data["segment"].x = self.get_segment_features()
        data["movement"].x = self.get_movement_features()
        data["phase"].x = self.get_phase_features()
        data["intersection"].x = self.get_intersection_features()

        data["segment", "to_down", "movement"].edge_index = to_edge_index(
            self.net.index_segment_to_down_movement, self.net.segments, self.net.movements)
        data["segment", "to_up", "movement"].edge_index = to_edge_index(
            self.net.index_segment_to_up_movement, self.net.segments, self.net.movements)
        data["movement", "to", "movement"].edge_index = to_edge_index(
            self.net.index_movement_to_movement, self.net.movements, self.net.movements)
        data["movement", "to_up", "movement"].edge_index = to_edge_index(
            self.net.index_movement_to_up_movement, self.net.movements, self.net.movements)
        data["movement", "to_down", "movement"].edge_index = to_edge_index(
            self.net.index_movement_to_down_movement, self.net.movements, self.net.movements)
        data["movement", "to", "phase"].edge_index = to_edge_index(
            self.net.index_movement_to_phase, self.net.movements, self.net.phases)
        data["phase", "to", "phase"].edge_index = to_edge_index(
            self.net.index_phase_to_phase, self.net.phases, self.net.phases)
        data["phase", "to", "intersection"].edge_index = to_edge_index(
            self.net.index_phase_to_intersection, self.net.phases, self.net.signalized_intersections)

        data["movement", "to", "movement"].edge_attr = self.get_movement_to_movement_edge_attr()
        data["movement", "to", "phase"].edge_attr = self.get_movement_to_phase_edge_attr()
        data["phase", "to", "phase"].edge_attr = self.get_phase_to_phase_edge_attr()

        return data

    def get_segment_features(self):
        x = []
        for segment in self.net.segments:
            veh_density = self.net.get_vehicle_density(segment=segment)
            x.append([veh_density])
        return torch.tensor(x, dtype=torch.float32)

    def get_movement_features(self):
        x = []
        for movement in self.net.movements:
            signal = self.net.get_current_signal(movement)
            signal = [int(signal == "G"), int(signal == "g"), int(signal == "r")]
            in_lane, out_lane = movement[0], movement[2]
            in_segments = sorted(self.net.get_segments(in_lane), key=lambda k: k[1])
            in_segment_veh_densities = [self.net.get_vehicle_density(segment=segment) for segment in in_segments]
            out_lane_veh_density = self.net.get_vehicle_density(lane=out_lane)
            in_lane_len = float(self.net.getLane(in_lane).getLength()) / 1_000
            out_lane_len = float(self.net.getLane(out_lane).getLength()) / 1_000
            x.append(signal) #+ in_segment_veh_densities + [out_lane_veh_density] + [in_lane_len] + [out_lane_len])
        return torch.tensor(x, dtype=torch.float32)

    def get_phase_features(self):
        return torch.tensor([[int(phase == self.net.get_current_phase(phase[0]))] for phase in self.net.phases],
                            dtype=torch.float32)

    def get_intersection_features(self):
        return torch.tensor([[1.0] for _ in self.net.signalized_intersections], dtype=torch.float32)

    def get_movement_to_movement_edge_attr(self):
        x = []
        for movement_1, movement_2 in self.net.index_movement_to_movement:
            in_approach_1, in_approach_2 = (self.net.getLane(movement_1[0]).getEdge().getID(),
                                            self.net.getLane(movement_2[0]).getEdge().getID())
            out_approach_1, out_approach_2 = (self.net.getLane(movement_1[2]).getEdge().getID(),
                                              self.net.getLane(movement_2[2]).getEdge().getID())
            x.append([float(in_approach_1 == in_approach_2), float(out_approach_1 == out_approach_2)])
        return torch.tensor(x, dtype=torch.float32)

    def get_movement_to_phase_edge_attr(self):
        x = []
        for movement, phase in self.net.index_movement_to_phase:
            movement_signals = self.net.get_movement_signals(phase)
            signal = movement_signals[movement]
            signal = [int(signal == "G"), int(signal == "g"), int(signal == "r")]
            x.append(signal)
        return torch.tensor(x, dtype=torch.float32)

    def get_phase_to_phase_edge_attr(self):
        x = []
        for phase_a, phase_b in self.net.index_phase_to_phase:
            green_movements_a = set([movement for movement, signal in self.net.get_movement_signals(phase_a).items()
                                     if signal in ["G", "g"]])
            green_movements_b = set([movement for movement, signal in self.net.get_movement_signals(phase_b).items()
                                     if signal in ["G", "g"]])
            overlap = (len(green_movements_a.intersection(green_movements_b)) /
                       len(green_movements_a.union(green_movements_b)))
            x.append([int(overlap > 0.0)])
        return torch.tensor(x, dtype=torch.float32)


class TransferLightProblemFormulation(ProblemFormulation):

    def __init__(self, *args, segment_length: float = 25.0):
        super(TransferLightProblemFormulation, self).__init__(*args)
        self.net.init_segments(segment_length)

    @classmethod
    def get_metadata(cls):
        metadata = dict()

        node_dim = defaultdict(lambda: 0)
        node_dim["segment"] = 1
        node_dim["movement"] = 3
        node_dim["phase"] = 1
        metadata["node_dim"] = node_dim

        edge_dim = defaultdict(lambda: 0)
        edge_dim[("movement", "to", "phase")] = 3
        edge_dim[("phase", "to", "phase")] = 1
        metadata["edge_dim"] = edge_dim

        pos = defaultdict(lambda: False)
        pos[("segment", "to_up", "segment")] = True
        pos[("segment", "to_down", "segment")] = True
        metadata["pos"] = pos

        return metadata

    def get_rewards(self) -> torch.Tensor:
        return - torch.tensor([self.net.get_pressure(intersection=intersection, method="density")
                               for intersection in self.net.signalized_intersections], dtype=torch.float32)

    def get_state(self) -> HeteroData:
        data = HeteroData()
        data["segment"].x = self.get_segment_features()
        data["lane"].x = self.get_lane_features()
        data["movement"].x = self.get_movement_features()
        data["phase"].x = self.get_phase_features()
        data["intersection"].x = self.get_intersection_features()

        data["segment", "to_up", "segment"].edge_index = to_edge_index(
            self.net.index_segment_to_up_segment, self.net.segments, self.net.segments)
        data["segment", "to_down", "segment"].edge_index = to_edge_index(
            self.net.index_segment_to_down_segment, self.net.segments, self.net.segments)
        data["segment", "to", "lane"].edge_index = to_edge_index(
            self.net.index_segment_to_lane, self.net.segments, self.net.lanes)
        data["lane", "to_down", "movement"].edge_index = to_edge_index(
            self.net.index_lane_to_down_movement, self.net.lanes, self.net.movements)
        data["lane", "to_up", "movement"].edge_index = to_edge_index(
            self.net.index_lane_to_up_movement, self.net.lanes, self.net.movements)
        data["movement", "to_up", "movement"].edge_index = to_edge_index(
            self.net.index_movement_to_up_movement, self.net.movements, self.net.movements)
        data["movement", "to_down", "movement"].edge_index = to_edge_index(
            self.net.index_movement_to_down_movement, self.net.movements, self.net.movements)
        data["movement", "to", "phase"].edge_index = to_edge_index(
            self.net.index_movement_to_phase, self.net.movements, self.net.phases)
        data["phase", "to", "phase"].edge_index = to_edge_index(
            self.net.index_phase_to_phase, self.net.phases, self.net.phases)
        data["movement", "to", "intersection"].edge_index = to_edge_index(
            self.net.index_movement_to_intersection, self.net.movements, self.net.signalized_intersections)
        data["phase", "to", "intersection"].edge_index = to_edge_index(
            self.net.index_phase_to_intersection, self.net.phases, self.net.signalized_intersections)

        data["movement", "to", "phase"].edge_attr = self.get_movement_to_phase_edge_attr()
        data["phase", "to", "phase"].edge_attr = self.get_phase_to_phase_edge_attr()

        data["segment", "to_up", "segment"].pos = torch.tensor(self.net.pos_segment_to_up_segment)
        data["segment", "to_down", "segment"].pos = torch.tensor(self.net.pos_segment_to_down_segment)

        return data

    def get_segment_features(self):
        return torch.tensor([[self.net.get_vehicle_density(segment=segment)] for segment in self.net.segments],
                            dtype=torch.float32)

    def get_lane_features(self):
        return torch.tensor([[1.0] for _ in self.signalized_intersections], dtype=torch.float32)

    def get_movement_features(self):
        x = []
        for movement in self.net.movements:
            signal = self.net.get_current_signal(movement)
            signal = [int(signal == "G"), int(signal == "g"), int(signal == "r")]
            x.append(signal)
        return torch.tensor(x, dtype=torch.float32)

    def get_phase_features(self):
        return torch.tensor([[int(phase == self.net.get_current_phase(phase[0]))] for phase in self.phases],
                            dtype=torch.float32)

    def get_intersection_features(self):
        return torch.tensor([[1.0] for _ in self.signalized_intersections], dtype=torch.float32)

    def get_movement_to_phase_edge_attr(self):
        x = []
        for movement, phase in self.net.index_movement_to_phase:
            movement_signals = self.net.get_movement_signals(phase)
            signal = movement_signals[movement]
            signal = [int(signal == "G"), int(signal == "g"), int(signal == "r")]
            x.append(signal)
        return torch.tensor(x, dtype=torch.float32)

    def get_phase_to_phase_edge_attr(self):
        x = []
        for phase_a, phase_b in self.net.index_phase_to_phase:
            green_movements_a = set([movement for movement, signal in self.net.get_movement_signals(phase_a).items()
                                     if signal in ["G", "g"]])
            green_movements_b = set([movement for movement, signal in self.net.get_movement_signals(phase_b).items()
                                     if signal in ["G", "g"]])
            overlap = (len(green_movements_a.intersection(green_movements_b)) /
                       len(green_movements_a.union(green_movements_b)))
            x.append([int(overlap > 0.0)])
        return torch.tensor(x, dtype=torch.float32)
