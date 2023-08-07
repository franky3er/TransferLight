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


class ProblemFormulation(ABC):

    @classmethod
    def create(cls, class_name: str, net: Net):
        return getattr(sys.modules[__name__], class_name)(net)

    def __init__(self, net: Net):
        # Node Types
        self.net = net
        self.intersections = [intersection_id.getID() for intersection_id in net.getNodes()]
        self.signalized_intersections = [intersection_id for intersection_id in self.intersections
                                         if net.getNode(intersection_id).getType() == "traffic_light"]
        self.roads = [road.getID() for road in net.getEdges()]
        self.lanes = [lane.getID() for road_id in self.roads for lane in net.getEdge(road_id).getLanes()]
        self.movements = defaultdict(lambda: [])
        for intersection_id in self.intersections:
            connections = sorted(net.getNode(intersection_id).getConnections(),
                                 key=lambda con: (angle_approach(con.getFromLane().getEdge()),
                                                  angle_between_approaches(con.getFromLane().getEdge(),
                                                                           con.getToLane().getEdge())))
            for connection in connections:
                incoming_lane, outgoing_lane = connection.getFromLane(), connection.getToLane()
                incoming_approach, outgoing_approach = incoming_lane.getEdge(), outgoing_lane.getEdge()
                incoming_lane_id, outgoing_lane_id = incoming_lane.getID(), outgoing_lane.getID()
                incoming_approach_id, outgoing_approach_id = incoming_approach.getID(), outgoing_approach.getID()
                self.movements[(incoming_approach_id, intersection_id, outgoing_approach_id)].append(
                    (incoming_lane_id, intersection_id, outgoing_lane_id))
        self.phases = []
        for intersection_id in self.signalized_intersections:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(intersection_id)[1]
            phases = logic.phases
            for phase_idx in range(len(phases)):
                self.phases.append((intersection_id, phase_idx))

        # Convenience Data Structures
        self.phase_movements = defaultdict(lambda: [])
        self.phase_movement_params = {}
        for intersection_id, phase_idx in self.phases:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(intersection_id)[1]
            phase = logic.phases[phase_idx]
            state = phase.state
            for i, s in [(i, s) for i, s in enumerate([*state])]:
                controlled_links = traci.trafficlight.getControlledLinks(intersection_id)
                controlled_links = controlled_links[i]
                incoming_lane_id, outgoing_lane_id, _ = controlled_links[0]
                incoming_lane, outgoing_lane = net.getLane(incoming_lane_id), net.getLane(outgoing_lane_id)
                incoming_approach_id, outgoing_approach_id = incoming_lane.getEdge().getID(), \
                    outgoing_lane.getEdge().getID()
                phase = (intersection_id, phase_idx)
                movement = (incoming_approach_id, intersection_id, outgoing_approach_id)
                prohibited = True if s == "r" else False
                permitted = True if s == "g" else False
                protected = True if s == "G" else False
                params = {"prohibited": prohibited, "permitted": permitted, "protected": protected}
                if movement not in self.phase_movements[phase]:
                    self.phase_movements[phase].append(movement)
                    self.phase_movement_params[(phase, movement)] = params

        # Edge Indices
        self.edge_index_lane_to_downstream_movement, self.edge_index_lane_to_upstream_movement = [], []
        for movement in self.movements.keys():
            incoming_lanes = set([lane_id for lane_id, _, _ in self.movements[movement]])
            outgoing_lanes = set([lane_id for _, _, lane_id in self.movements[movement]])
            for lane_id in incoming_lanes:
                self.edge_index_lane_to_downstream_movement.append((self.lanes.index(lane_id),
                                                                    list(self.movements.keys()).index(movement)))
            for lane_id in outgoing_lanes:
                self.edge_index_lane_to_upstream_movement.append((self.lanes.index(lane_id),
                                                                  list(self.movements.keys()).index(movement)))
        self.edge_index_movement_to_upstream_lane = self.reverse_edge_index(self.edge_index_lane_to_downstream_movement)
        self.edge_index_movement_to_downstream_lane = self.reverse_edge_index(self.edge_index_lane_to_upstream_movement)

        self.edge_index_movement_to_downstream_movement = [
            (list(self.movements.keys()).index(movement), list(self.movements.keys()).index(downstream_movement))
            for movement in self.movements.keys()
            for downstream_movement in self.movements.keys()
            if movement[2] == downstream_movement[0]
        ]
        self.edge_index_movement_to_upstream_movement = [
            (list(self.movements.keys()).index(movement), list(self.movements.keys()).index(upstream_movement))
            for movement in self.movements.keys()
            for upstream_movement in self.movements.keys()
            if movement[0] == upstream_movement[2]
        ]

        self.edge_index_movement_to_phase = []
        for phase, movements in self.phase_movements.items():
            for movement in movements:
                self.edge_index_movement_to_phase.append((list(self.movements.keys()).index(movement),
                                                          self.phases.index(phase)))
        self.edge_index_phase_to_movement = self.reverse_edge_index(self.edge_index_movement_to_phase)

        self.edge_index_phase_to_intersection = [
            (self.phases.index((intersection_id, phase)), self.signalized_intersections.index(intersection_id))
            for intersection_id, phase in self.phases]
        self.edge_index_intersection_to_phase = self.reverse_edge_index(self.edge_index_phase_to_intersection)
        self.edge_index_phase_to_phase = []
        for intersection_id in self.signalized_intersections:
            phase_indices = [phase_idx for junction_idx, phase_idx in self.edge_index_intersection_to_phase
                             if self.signalized_intersections[junction_idx] == intersection_id]
            edge_index_phase_to_phase = list(itertools.combinations(phase_indices, 2))
            edge_index_phase_to_phase += self.reverse_edge_index(edge_index_phase_to_phase)
            self.edge_index_phase_to_phase += edge_index_phase_to_phase
        self.edge_index_movement_to_intersection = [
            (list(self.movements.keys()).index(movement), self.signalized_intersections.index(intersection))
            for intersection in self.signalized_intersections for movement in self.movements
            if movement[1] == intersection
        ]
        self.edge_index_intersection_to_intersection = [
            (self.intersections.index(edge.getFromNode().getID()), self.intersections.index(edge.getToNode().getID()))
            for edge in [net.getEdge(edge_id) for edge_id in self.roads]]

    @staticmethod
    def reverse_edge_index(edge_index: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return [(to_idx, from_idx) for from_idx, to_idx in edge_index]

    @staticmethod
    def edge_index_to_tensor(edge_index: List[Tuple[int, int]]) -> torch.Tensor:
        return torch.tensor([[node_i, node_j] for node_i, node_j in edge_index], dtype=torch.int64).t().contiguous()

    @abstractmethod
    def get_state(self) -> Any:
        pass

    def get_signalized_intersections(self) -> List[str]:
        return self.signalized_intersections

    @staticmethod
    def get_current_phase(signalized_intersection_id: str):
        signal = traci.trafficlight.getRedYellowGreenState(signalized_intersection_id)
        phase_signals = \
            [phase.state for phase
             in traci.trafficlight.getCompleteRedYellowGreenDefinition(signalized_intersection_id)[1].phases]
        return phase_signals.index(signal)

    def get_current_phases(self) -> List[int]:
        phases = []
        for junction_id in self.signalized_intersections:
            phases.append(self.get_current_phase(junction_id))
        return phases

    @staticmethod
    def get_phases(signalized_intersection_id: str):
        return range(len(traci.trafficlight.getCompleteRedYellowGreenDefinition(signalized_intersection_id)[1].phases))

    def get_intersection_queue_length(self, signalized_intersection_id: str):
        queue_length = float(sum([traci.edge.getLastStepHaltingNumber(edge.getID()) for edge in self.net.getEdges()
                                  if edge.getToNode().getID() == signalized_intersection_id]))
        return queue_length

    def get_intersection_waiting_time(self, signalized_intersection_id: str):
        waiting_time = sum([traci.edge.getWaitingTime(edge.getID())
                            for edge in self.net.getEdges()
                            if edge.getToNode().getID() == signalized_intersection_id])
        return waiting_time

    def get_intersection_pressure(self, signalized_intersection_id: str):
        pressure = 0
        for movement in [m for m in self.movements.keys() if m[1] == signalized_intersection_id]:
            incoming_lanes = list({incoming_lane_id for incoming_lane_id, _, _ in self.movements[movement]})
            outgoing_lanes = list({outgoing_lane_id for _, _, outgoing_lane_id in self.movements[movement]})
            n_vehicles_incoming_approach = sum([traci.lane.getLastStepVehicleNumber(lane_id)
                                                for lane_id in incoming_lanes])
            n_vehicles_outgoing_approach = sum([traci.lane.getLastStepVehicleNumber(lane_id)
                                                for lane_id in outgoing_lanes])
            movement_pressure = n_vehicles_incoming_approach - n_vehicles_outgoing_approach
            pressure += movement_pressure
        return abs(float(pressure))

    def get_intersection_normalized_pressure(self, signalized_intersection_id: str):
        normalized_pressure = 0
        for movement in [m for m in self.movements.keys() if m[1] == signalized_intersection_id]:
            incoming_lanes = set([lane_id for lane_id, _, _ in self.movements[movement]])
            outgoing_lanes = set([lane_id for _, _, lane_id in self.movements[movement]])
            total_incoming_lane_len = sum([self.net.getLane(lane_id).getLength() for lane_id in incoming_lanes])
            total_outgoing_lane_len = sum([self.net.getLane(lane_id).getLength() for lane_id in outgoing_lanes])
            incoming_vehicles = [veh_id for lane_id in incoming_lanes
                                 for veh_id in traci.lane.getLastStepVehicleIDs(lane_id)]
            outgoing_lanes = [veh_id for lane_id in outgoing_lanes
                              for veh_id in traci.lane.getLastStepVehicleIDs(lane_id)]
            total_incoming_veh_len = sum([traci.vehicle.getLength(veh_id) for veh_id in incoming_vehicles])
            total_outgoing_veh_len = sum([traci.vehicle.getLength(veh_id) for veh_id in outgoing_lanes])
            normalized_pressure += total_incoming_veh_len / total_incoming_lane_len - \
                                   total_outgoing_veh_len / total_outgoing_lane_len
        return abs(normalized_pressure)

    def get_intersection_level_metrics(self, metric: str = None):
        if metric == "queue_length":
            metric_fn = self.get_intersection_queue_length
        elif metric == "pressure":
            metric_fn = self.get_intersection_pressure
        elif metric == "normalized_pressure":
            metric_fn = self.get_intersection_normalized_pressure
        else:
            return None
        metric_values = []
        for junction_id in self.signalized_intersections:
            metric_values.append(metric_fn(junction_id))
        return metric_values

    @abstractmethod
    def get_rewards(self) -> torch.Tensor:
        pass

    @staticmethod
    def get_max_vehicle_waiting_time() -> float:
        if len(traci.vehicle.getIDList()) == 0:
            return 0.0
        return np.max([traci.vehicle.getWaitingTime(vehID=veh_id) for veh_id in traci.vehicle.getIDList()])

    @classmethod
    def get_metadata(cls):
        pass


class MaxPressureProblemFormulation(ProblemFormulation):

    def get_state(self) -> HeteroData:
        data = HeteroData()
        x = []
        for phase, movements in self.phase_movements.items():
            relevant_movements = [m for m in movements if not self.phase_movement_params[(phase, m)]["prohibited"]]
            incoming_lanes = set(incoming_lane for movement in relevant_movements
                                 for incoming_lane, _, _ in self.movements[movement])
            outgoing_lanes = set(outgoing_lane for movement in relevant_movements
                                 for _, _, outgoing_lane in self.movements[movement])
            incoming_lanes_veh = sum([traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in incoming_lanes])
            outgoing_lanes_veh = sum([traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in outgoing_lanes])
            phase_pressure = float(incoming_lanes_veh) - float(outgoing_lanes_veh)
            x.append([phase_pressure])
        data["phase"].x = torch.tensor(x)
        data["intersection"].x = torch.zeros(len(self.signalized_intersections), 1)
        data["phase", "to", "intersection"].edge_index = self.edge_index_to_tensor(
            self.edge_index_phase_to_intersection)
        return data

    def get_rewards(self) -> torch.Tensor:
        return - torch.tensor(self.get_intersection_level_metrics("normalized_pressure"))


class PressLightProblemFormulation(ProblemFormulation):

    def __init__(self, *args, n_segments: int = 3):
        super(PressLightProblemFormulation, self).__init__(*args)
        self.n_segments = n_segments

    def get_state(self) -> Data:
        x = []
        for intersection_id in self.signalized_intersections:
            x_intersection = [0.0 for _ in range(len(self.get_phases(intersection_id)))]
            x_intersection[self.get_current_phase(intersection_id)] = 1.0
            incoming_approaches = sort_approaches_clockwise([edge for edge in self.net.getEdges()
                                                             if edge.getToNode().getID() == intersection_id])
            outgoing_approaches = sort_approaches_clockwise([edge for edge in self.net.getEdges()
                                                             if edge.getFromNode().getID() == intersection_id],
                                                            switch_src_dest=True)
            incoming_lanes = [lane for approach in incoming_approaches for lane in approach.getLanes()]
            outgoing_lanes = [lane for approach in outgoing_approaches for lane in approach.getLanes()]
            for lane in incoming_lanes:
                x_intersection += self._get_segment_data(lane)
            for lane in outgoing_lanes:
                x_intersection.append(float(traci.lane.getLastStepVehicleNumber(lane.getID())))
            x.append(x_intersection)
        x = torch.tensor(x, dtype=torch.float32)
        return Data(x=x)

    def _get_segment_data(self, lane: sumolib.net.lane.Lane):
        lane_id = lane.getID()
        lane_length = lane.getLength()
        segment_length = lane_length / self.n_segments
        segment_bins = [(segment+1) * segment_length for segment in range(self.n_segments - 1)]
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
        if not vehicle_ids:
            return [0.0 for _ in range(self.n_segments)]
        vehicle_positions = [traci.vehicle.getLanePosition(vehicle_id) for vehicle_id in vehicle_ids]
        vehicle_segment_positions = np.digitize(vehicle_positions, segment_bins)
        vehicle_segment_positions_one_hot = np.eye(self.n_segments)[vehicle_segment_positions]
        segment_vehicle_counts = np.sum(vehicle_segment_positions_one_hot, axis=0)
        return list(segment_vehicle_counts)

    def get_rewards(self) -> torch.Tensor:
        return - torch.tensor(self.get_intersection_level_metrics("pressure"))


class GeneraLightProblemFormulation(ProblemFormulation):

    def __init__(self, *args, segment_length: float = 10.0, history_length: int = 1):
        super(GeneraLightProblemFormulation, self).__init__(*args)
        self.segment_length = segment_length
        self.lane_segments = []
        self.lane_segment_bins = {}
        for lane_id in self.lanes:
            lane = self.net.getLane(lane_id)
            lane_length = lane.getLength()
            n_segments = int(lane_length // self.segment_length)
            segment_bins = [(i+1) * segment_length for i in range(n_segments)]
            self.lane_segment_bins[lane_id] = segment_bins
            segments = [(lane_id, i) for i in range(n_segments)]
            self.lane_segments += segments

        self.edge_index_lane_segment_to_lane = []
        self.edge_attr_lane_segment_to_lane = []
        for lane_id in self.lanes:
            n_segments = len(self.lane_segment_bins[lane_id])
            self.edge_index_lane_segment_to_lane += [
                (self.lane_segments.index((lane_id, i)), self.lanes.index(lane_id))
                for i in range(n_segments)
            ]
            self.edge_attr_lane_segment_to_lane += [
                [float(i)] for i in range(n_segments)
            ]
        self.edge_attr_lane_segment_to_lane = torch.tensor(
            self.edge_attr_lane_segment_to_lane)

        self.edge_attr_movement_to_phase = []
        for movement, phase in [(list(self.movements.keys())[m], self.phases[p])
                                for m, p in self.edge_index_movement_to_phase]:
            params = self.phase_movement_params[(phase, movement)]
            edge_attr = [float(params["prohibited"]), float(params["permitted"]), float(params["protected"])]
            self.edge_attr_movement_to_phase.append(edge_attr)
        self.edge_attr_movement_to_phase = torch.tensor(self.edge_attr_movement_to_phase)

        self.edge_attr_phase_to_phase = []
        for phase_j, phase_i in [(self.phases[j], self.phases[i]) for j, i in self.edge_index_phase_to_phase]:
            phase_j_movements = self.phase_movements[phase_j]
            phase_i_movements = self.phase_movements[phase_i]
            intersection_movements = set(phase_j_movements).intersection(set(phase_i_movements))
            union_movements = set(phase_j_movements).union(set(phase_i_movements))
            overlap = (len(intersection_movements) / len(union_movements)) > 0.0
            self.edge_attr_phase_to_phase.append([float(overlap)])
        self.edge_attr_phase_to_phase = torch.tensor(self.edge_attr_phase_to_phase)

        self.history_length = history_length
        self.history_lane_segment_x = deque(maxlen=history_length)
        self.history_movement_x = deque(maxlen=history_length)
        self.history_phase_x = deque(maxlen=history_length)

    @classmethod
    def get_metadata(cls):
        metadata = dict()

        node_dim = defaultdict(lambda: 0)
        node_dim["lane_segment"] = 1
        node_dim["movement"] = 3
        node_dim["phase"] = 1
        metadata["node_dim"] = node_dim

        edge_dim = defaultdict(lambda: 0)
        #edge_dim[("lane_segment", "to", "lane")] = 1
        #edge_dim[("lane", "to_downstream", "movement")] = 1
        #edge_dim[("lane", "to_upstream", "movement")] = 1
        edge_dim[("movement", "to", "phase")] = 3
        edge_dim[("phase", "to", "phase")] = 1
        metadata["edge_dim"] = edge_dim

        return metadata

    def get_state(self) -> HeteroData:
        data = HeteroData()

        # Node Features
        data["lane_segment"].x = self._get_lane_segment_features()
        data["movement"].x = self._get_movement_features()
        data["lane"].x = self._get_lane_features()
        data["phase"].x = self._get_phase_features()
        data["intersection"].x = self._get_signalized_intersection_features()

        # Edge Indices
        data["lane_segment", "to", "lane"].edge_index = self.edge_index_to_tensor(
            self.edge_index_lane_segment_to_lane)
        data["lane", "to_downstream", "movement"].edge_index = self.edge_index_to_tensor(
            self.edge_index_lane_to_downstream_movement)
        data["lane", "to_upstream", "movement"].edge_index = self.edge_index_to_tensor(
            self.edge_index_lane_to_upstream_movement
        )
        data["movement", "to", "movement"].edge_index = self.edge_index_to_tensor(
            self.edge_index_movement_to_downstream_movement)
        data["movement", "to", "phase"].edge_index = self.edge_index_to_tensor(self.edge_index_movement_to_phase)
        data["phase", "to", "phase"].edge_index = self.edge_index_to_tensor(self.edge_index_phase_to_phase)
        data["phase", "to", "intersection"].edge_index = self.edge_index_to_tensor(
            self.edge_index_phase_to_intersection)
        data["movement", "to", "intersection"].edge_index = self.edge_index_to_tensor(
            self.edge_index_movement_to_intersection)

        # Edge Features
        #data["lane_segment", "to", "lane"].edge_attr = self.edge_attr_lane_segment_to_lane
        #data["lane", "to_downstream", "movement"].edge_attr = self.edge_attr_lane_to_downstream_movement
        #data["lane", "to_upstream", "movement"].edge_attr = self.edge_attr_lane_to_upstream_movement
        data["movement", "to", "phase"].edge_attr = self.edge_attr_movement_to_phase
        data["phase", "to", "phase"].edge_attr = self.edge_attr_phase_to_phase

        return data

    def _get_lane_segment_features(self) -> torch.Tensor:
        x = []
        for lane_id in self.lanes:
            lane_length = self.net.getLane(lane_id).getLength()
            segment_veh_len = [0.0 for _ in range(len(self.lane_segment_bins[lane_id]))]
            max_bin = self.lane_segment_bins[lane_id][-1]
            for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                veh_len = traci.vehicle.getLength(veh_id)
                veh_pos = lane_length - traci.vehicle.getLanePosition(veh_id)
                if veh_pos >= max_bin:
                    continue
                veh_pos = np.digitize(veh_pos, self.lane_segment_bins[lane_id])
                segment_veh_len[veh_pos] += veh_len
            segment_veh_densities = np.clip(np.array(segment_veh_len) / self.segment_length, 0.0, 1.0)
            x += [[density] for density in segment_veh_densities]
        x = torch.tensor(x, dtype=torch.float32)
        if len(self.history_lane_segment_x) < self.history_lane_segment_x.maxlen:
            for _ in range(self.history_lane_segment_x.maxlen):
                self.history_lane_segment_x.append(x)
        else:
            self.history_lane_segment_x.append(x)
        return torch.cat(list(self.history_lane_segment_x), dim=1)

    def _get_lane_features(self) -> torch.Tensor:
        x = []
        for _ in self.lanes:
            x.append([1.0])  # dummy entries
        return torch.tensor(x, dtype=torch.float32)

    def _get_movement_features(self) -> torch.Tensor:
        x = []
        for movement in self.movements.keys():
            incoming_approach_id, intersection_id, outgoing_approach_id = movement
            movement_prohibited = True
            movement_permitted = False
            movement_protected = False
            if intersection_id in self.signalized_intersections:
                phase = (intersection_id, self.get_current_phase(intersection_id))
                if movement in self.phase_movements[phase]:
                    movement_params = self.phase_movement_params[(phase, movement)]
                    movement_prohibited = False
                    movement_permitted = not movement_params["protected"]
                    movement_protected = movement_params["protected"]
            else:
                movement_prohibited = False
                movement_permitted = True
                movement_protected = False
            x_movement = [float(movement_prohibited), float(movement_permitted), float(movement_protected)]
            x.append(x_movement)
        x = torch.tensor(x, dtype=torch.float32)
        if len(self.history_movement_x) < self.history_movement_x.maxlen:
            for _ in range(self.history_movement_x.maxlen):
                self.history_movement_x.append(x)
        else:
            self.history_movement_x.append(x)
        return torch.cat(list(self.history_movement_x), dim=1)

    def _get_phase_features(self) -> torch.Tensor:
        x = []
        for phase in self.phases:
            intersection_id, phase_idx = phase
            phase_active = True if self.get_current_phase(intersection_id) == phase_idx else False
            x_phase = [float(phase_active)]
            x.append(x_phase)
        x = torch.tensor(x, dtype=torch.float32)
        if len(self.history_phase_x) < self.history_phase_x.maxlen:
            for _ in range(self.history_phase_x.maxlen):
                self.history_phase_x.append(x)
        else:
            self.history_phase_x.append(x)
        return torch.cat(list(self.history_phase_x), dim=1)

    def _get_signalized_intersection_features(self) -> torch.Tensor:
        x = []
        for _ in self.signalized_intersections:
            x.append([1.0])  # dummy entries
        return torch.tensor(x, dtype=torch.float32)

    def get_rewards(self) -> torch.Tensor:
        return - torch.tensor(self.get_intersection_level_metrics("normalized_pressure"))
