from abc import ABC, abstractmethod
from collections import defaultdict
import sys
from typing import Any, List, Tuple

import torch
from sumolib.net import Net
from torch_geometric.data import HeteroData

from src.modules.utils import sinusoidal_positional_encoding
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
        for movement, phase in self.net.index[("movement", "to", "phase")]:
            signal = self.net.get_movement_signals(phase)[movement]
            if signal not in ["G", "g"]:
                continue
            pressure = self.net.get_pressure(movement=movement, method="count")
            phase_pressures[phase] += pressure
        x = [[phase_pressures[phase]] for phase in self.net.phases]
        data["phase"].x = torch.tensor(x)
        data["intersection"].x = torch.zeros(len(self.net.signalized_intersections), 1)
        data["phase", "to", "intersection"].edge_index = to_edge_index(
            self.net.index[("phase", "to", "intersection")], self.net.phases, self.net.signalized_intersections)
        return data

    def get_rewards(self) -> torch.Tensor:
        return - torch.tensor([self.net.get_pressure(intersection=intersection, method="density")
                               for intersection in self.net.signalized_intersections], dtype=torch.float32)


class TransferLightProblemFormulation(ProblemFormulation):

    def __init__(self, *args, segment_length: float = 10.0):
        super(TransferLightProblemFormulation, self).__init__(*args)
        self.net.init_segments(segment_length=segment_length)

    @classmethod
    def get_metadata(cls):
        metadata = dict()

        node_dim = defaultdict(lambda: 0)
        node_dim["segment"] = 9
        node_dim["movement"] = 3
        node_dim["phase"] = 1
        node_dim["intersection"] = 1
        metadata["node_dim"] = node_dim

        edge_dim = defaultdict(lambda: 0)
        edge_dim[("movement", "to", "phase")] = 3
        edge_dim[("movement", "to", "movement")] = 4
        edge_dim[("phase", "to", "phase")] = 1
        metadata["edge_dim"] = edge_dim

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

        data["segment", "to", "segment"].edge_index = to_edge_index(
            self.net.index[("segment", "to", "segment")], self.net.segments, self.net.segments)
        data["segment", "to_down", "movement"].edge_index = to_edge_index(
            self.net.index[("segment", "to_down", "movement")], self.net.segments, self.net.movements)
        data["segment", "to_up", "movement"].edge_index = to_edge_index(
            self.net.index[("segment", "to_up", "movement")], self.net.segments, self.net.movements)
        data["movement", "to", "movement"].edge_index = to_edge_index(
            self.net.index[("movement", "to", "movement")], self.net.movements, self.net.movements)
        data["movement", "to", "phase"].edge_index = to_edge_index(
            self.net.index[("movement", "to", "phase")], self.net.movements, self.net.phases)
        data["phase", "to", "phase"].edge_index = to_edge_index(
            self.net.index[("phase", "to", "phase")], self.net.phases, self.net.phases)
        data["phase", "to", "intersection"].edge_index = to_edge_index(
            self.net.index[("phase", "to", "intersection")], self.net.phases, self.net.signalized_intersections)

        data["movement", "to", "movement"].edge_attr = self.get_movement_to_movement_edge_attr()
        data["movement", "to", "phase"].edge_attr = self.get_movement_to_phase_edge_attr()
        data["phase", "to", "phase"].edge_attr = self.get_phase_to_phase_edge_attr()

        data["segment"].pos = torch.tensor(self.net.pos["segment"])

        return data

    def get_segment_features(self):
        x = []
        for segment in self.net.segments:
            veh_density = self.net.get_vehicle_density(segment=segment)
            x.append([veh_density])
        pos = torch.tensor(self.net.pos["segment"])
        pe = sinusoidal_positional_encoding(pos, 8)
        return torch.cat([torch.tensor(x, dtype=torch.float32), pe], dim=1)

    def get_movement_features(self):
        x = []
        for movement in self.net.movements:
            signal = self.net.get_current_signal(movement)
            signal = [int(signal == "G"), int(signal == "g"), int(signal == "r")]
            x.append(signal)
        return torch.tensor(x, dtype=torch.float32)

    def get_phase_features(self):
        return torch.tensor([[int(phase == self.net.get_current_phase(phase[0]))] for phase in self.net.phases],
                            dtype=torch.float32)

    def get_intersection_features(self):
        return torch.tensor([[0.0] for _ in self.net.signalized_intersections], dtype=torch.float32)

    def get_movement_to_movement_edge_attr(self):
        x = []
        for movement_1, movement_2 in self.net.index[("movement", "to", "movement")]:
            in_lane_1, in_lane_2 = movement_1[0], movement_2[0]
            out_lane_1, out_lane_2 = movement_1[2], movement_2[2]
            in_approach_1, in_approach_2 = (self.net.getLane(in_lane_1).getEdge().getID(),
                                            self.net.getLane(in_lane_2).getEdge().getID())
            out_approach_1, out_approach_2 = (self.net.getLane(out_lane_1).getEdge().getID(),
                                              self.net.getLane(out_lane_2).getEdge().getID())
            x.append([float(in_lane_1 == in_lane_2), float(out_lane_1 == out_lane_2),
                      float(in_approach_1 == in_approach_2), float(out_approach_1 == out_approach_2)])
        return torch.tensor(x, dtype=torch.float32)

    def get_movement_to_phase_edge_attr(self):
        x = []
        for movement, phase in self.net.index[("movement", "to", "phase")]:
            movement_signals = self.net.get_movement_signals(phase)
            signal = movement_signals[movement]
            signal = [int(signal == "G"), int(signal == "g"), int(signal == "r")]
            x.append(signal)
        return torch.tensor(x, dtype=torch.float32)

    def get_phase_to_phase_edge_attr(self):
        x = []
        for phase_a, phase_b in self.net.index[("phase", "to", "phase")]:
            green_movements_a = set([movement for movement, signal in self.net.get_movement_signals(phase_a).items()
                                     if signal in ["G", "g"]])
            green_movements_b = set([movement for movement, signal in self.net.get_movement_signals(phase_b).items()
                                     if signal in ["G", "g"]])
            overlap = (len(green_movements_a.intersection(green_movements_b)) /
                       len(green_movements_a.union(green_movements_b)))
            x.append([float(overlap)])
        return torch.tensor(x, dtype=torch.float32)
