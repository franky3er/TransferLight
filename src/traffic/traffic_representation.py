from abc import ABC, abstractmethod
import itertools
import sys
from typing import Any, List, Tuple

import torch
from sumolib.net import Net
import libsumo as traci
from torch_geometric.data import Data, HeteroData

from src.traffic.utils import sort_roads_clockwise


class TrafficRepresentation(ABC):

    @classmethod
    def create(cls, class_name: str, net: Net):
        return getattr(sys.modules[__name__], class_name)(net)

    def __init__(self, net: Net):
        # Node Types
        self.net = net
        self.junctions = [junction.getID() for junction in net.getNodes()]
        self.tls_junctions = [junction_id for junction_id in self.junctions
                              if net.getNode(junction_id).getType() == "traffic_light"]
        self.roads = [road.getID() for road in net.getEdges()]
        self.lanes = [lane.getID() for road_id in self.roads for lane in net.getEdge(road_id).getLanes()]
        self.movements = [
            (connection.getFromLane().getID(), junction_id, connection.getToLane().getID())
            for junction_id in self.junctions for connection in net.getNode(junction_id).getConnections()]
        self.phases = []
        for junction_id in self.tls_junctions:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(junction_id)[1]
            phases = logic.phases
            for phase_idx in range(len(phases)):
                self.phases.append((junction_id, phase_idx))

        # Edge Indices
        self.edge_index_junction_to_junction = [
            (self.junctions.index(edge.getFromNode().getID()), self.junctions.index(edge.getToNode().getID()))
            for edge in [net.getEdge(edge_id) for edge_id in self.roads]]
        self.edge_index_movement_to_phase = []
        for phase in self.phases:
            junction_id, phase_idx = phase
            state = traci.trafficlight.getCompleteRedYellowGreenDefinition(junction_id)[1].phases[phase_idx].state
            movement_ids = [i for i, s in enumerate([*state]) if s == "G" or s == "g"]
            movements = [traci.trafficlight.getControlledLinks(junction_id)[movement_id][0][:-1]
                         for movement_id in movement_ids]
            for movement in movements:
                movement = (movement[0], junction_id, movement[1])
                self.edge_index_movement_to_phase.append((self.movements.index(movement),
                                                          self.phases.index(phase)))
        self.edge_index_phase_to_movement = self.reverse_edge_index(self.edge_index_movement_to_phase)
        self.edge_index_phase_to_junction = [
            (self.phases.index((junction_id, phase)), self.tls_junctions.index(junction_id))
            for junction_id, phase in self.phases]
        self.edge_index_junction_to_phase = self.reverse_edge_index(self.edge_index_phase_to_junction)
        self.edge_index_lane_to_downstream_movement = [
            (self.lanes.index(from_lane), self.movements.index((from_lane, junction_id, to_lane)))
            for from_lane, junction_id, to_lane in self.movements
        ]
        self.edge_index_movement_to_upstream_lane = self.reverse_edge_index(self.edge_index_lane_to_downstream_movement)
        self.edge_index_lane_to_upstream_movement = [
            (self.lanes.index(to_lane), self.movements.index((from_lane, junction_id, to_lane)))
            for from_lane, junction_id, to_lane in self.movements
        ]
        self.edge_index_movement_to_downstream_lane = self.reverse_edge_index(self.edge_index_lane_to_upstream_movement)
        self.edge_index_phase_to_phase = []
        for junction_id in self.tls_junctions:
            phase_indices = [phase_idx for junction_idx, phase_idx in self.edge_index_junction_to_phase
                             if self.tls_junctions[junction_idx] == junction_id]
            edge_index_phase_to_phase = list(itertools.combinations(phase_indices, 2))
            edge_index_phase_to_phase += self.reverse_edge_index(edge_index_phase_to_phase)
            self.edge_index_phase_to_phase += edge_index_phase_to_phase

    @staticmethod
    def reverse_edge_index(edge_index: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return [(to_idx, from_idx) for from_idx, to_idx in edge_index]

    @staticmethod
    def edge_index_to_tensor(edge_index: List[Tuple[int, int]]) -> torch.Tensor:
        return torch.tensor([[node_i, node_j] for node_i, node_j in edge_index], dtype=torch.int64).t().contiguous()

    @abstractmethod
    def get_state(self) -> Any:
        pass

    def get_tls_junctions(self) -> List[str]:
        return self.tls_junctions

    @staticmethod
    def get_current_phase(tls_junction_id: str):
        signal = traci.trafficlight.getRedYellowGreenState(tls_junction_id)
        phase_signals = [phase.state
                         for phase in traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_junction_id)[1].phases]
        return phase_signals.index(signal)

    def get_current_phases(self) -> List[int]:
        phases = []
        for junction_id in self.tls_junctions:
            phases.append(self.get_current_phase(junction_id))
        return phases

    @staticmethod
    def get_phases(tls_junction_id: str):
        return range(len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_junction_id)[1].phases))

    def get_total_queue_length(self, tls_junction_id: str):
        total_queue_length = float(sum([traci.edge.getLastStepHaltingNumber(edge.getID())
                                        for edge in self.net.getEdges()
                                        if edge.getToNode().getID() == tls_junction_id]))
        return total_queue_length

    def get_total_queue_lengths(self):
        total_queue_lengths = []
        for junction_id in self.tls_junctions:
            total_queue_lengths.append(self.get_total_queue_length(junction_id))
        return total_queue_lengths


class MaxPressureTrafficRepresentation(TrafficRepresentation):

    def get_state(self) -> List[List[int]]:
        pressures = []
        for junction_id in self.tls_junctions:
            junction_idx = self.junctions.index(junction_id)
            tls_pressures = []
            for phase_idx in [phase_idx for phase_idx, _junction_idx in self.edge_index_phase_to_junction
                              if _junction_idx == junction_idx]:
                phase_pressure = 0
                for movement_idx in [movement_idx for movement_idx, _phase_idx in self.edge_index_movement_to_phase
                                     if _phase_idx == phase_idx]:
                    from_lane_id, to_lane_id = self.movements[movement_idx]
                    upstream_veh = traci.lane.getLastStepVehicleNumber(from_lane_id)
                    downstream_veh = traci.lane.getLastStepVehicleNumber(to_lane_id)
                    phase_pressure += upstream_veh - downstream_veh
                tls_pressures.append(phase_pressure)
            pressures.append(tls_pressures)
        return pressures


class LitTrafficRepresentation(TrafficRepresentation):

    def get_state(self) -> Data:
        x = []
        for junction_id in self.tls_junctions:
            x_junction = [0.0 for _ in range(len(self.get_phases(junction_id)))]
            x_junction[self.get_current_phase(junction_id)] = 1.0
            roads = sort_roads_clockwise(
                [edge for edge in self.net.getEdges() if edge.getToNode().getID() == junction_id])
            lanes = [lane for road in roads for lane in road.getLanes()]
            for lane in lanes:
                x_junction.append(float(traci.lane.getLastStepVehicleNumber(lane.getID())))
            x.append(x_junction)
        x = torch.tensor(x)
        return Data(x=x)


class HieraGLightTrafficRepresentation(TrafficRepresentation):

    def get_state(self) -> HeteroData:
        data = HeteroData()
        data["movement"].x = self._get_movement_features()
        data["phase"].x = self._get_phase_features()
        data["movement", "to", "phase"].edge_index = self.edge_index_to_tensor(self.edge_index_movement_to_phase)
        data["phase", "to", "phase"].edge_index = self.edge_index_to_tensor(self.edge_index_phase_to_phase)
        data["junction", "to", "phase"].edge_index = self.edge_index_to_tensor(self.edge_index_junction_to_phase)
        return data

    def _get_lane_features(self):
        x = []
        for lane_id in self.lanes:
            n_vehicles = float(traci.lane.getLastStepVehicleNumber(lane_id))
            x.append([n_vehicles])
        return torch.tensor(x)

    def _get_movement_features(self) -> torch.Tensor:
        x = []
        for movement_idx, (from_lane_id, junction_id, to_lane_id) in enumerate(self.movements):
            features = []
            if junction_id in self.tls_junctions:
                current_phase = self.get_current_phase(junction_id)
                movement_phase = self.phases[[idx_j for idx_i, idx_j in self.edge_index_movement_to_phase
                                              if idx_i == movement_idx][0]][1]
                active = float(current_phase == movement_phase)
                features.append(active)
            else:
                features.append(1.0)
            num_vehicles = traci.lane.getLastStepVehicleNumber(from_lane_id)
            features.append(num_vehicles)
            x.append(features)
        return torch.tensor(x)

    def _get_phase_features(self) -> torch.Tensor:
        x = []
        for junction_id, phase_idx in self.phases:
            if phase_idx is None:
                x.append([1.0])
            else:
                current_phase_idx = self.get_current_phase(junction_id)
                x.append([float(phase_idx == current_phase_idx)])
        return torch.tensor(x)

    def _get_junction_features(self) -> torch.Tensor:
        x = []
        for _ in self.tls_junctions:
            x.append([1.0])
        return torch.tensor(x)
