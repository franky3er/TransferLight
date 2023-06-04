import math
from abc import ABC, abstractmethod
from collections import defaultdict
import itertools
import sys
from typing import Any, List, Tuple, Union

import numpy as np
import sumolib.net.lane
import torch
from sumolib.net import Net
import libsumo as traci
from torch_geometric.data import Data, HeteroData

from src.traffic.utils import sort_approaches_clockwise, angle_approach, angle_between_approaches


class TrafficRepresentation(ABC):

    @classmethod
    def create(cls, class_name: str, net: Net):
        return getattr(sys.modules[__name__], class_name)(net)

    def __init__(self, net: Net):
        # Node Types
        self.net = net
        self.intersections = [junction.getID() for junction in net.getNodes()]
        self.signalized_intersections = [junction_id for junction_id in self.intersections
                                         if net.getNode(junction_id).getType() == "traffic_light"]
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
            for i, s in [(i, s) for i, s in enumerate([*state]) if s == "G" or s == "g"]:
                controlled_links = traci.trafficlight.getControlledLinks(intersection_id)
                controlled_links = controlled_links[i]
                incoming_lane_id, outgoing_lane_id, _ = controlled_links[0]
                incoming_lane, outgoing_lane = net.getLane(incoming_lane_id), net.getLane(outgoing_lane_id)
                incoming_approach_id, outgoing_approach_id = incoming_lane.getEdge().getID(), \
                    outgoing_lane.getEdge().getID()
                phase = (intersection_id, phase_idx)
                movement = (incoming_approach_id, intersection_id, outgoing_approach_id)
                protected = True if s == "G" else False
                params = {"protected": protected}
                if movement not in self.phase_movements[phase]:
                    self.phase_movements[phase].append(movement)
                    self.phase_movement_params[(phase, movement)] = params

        # Edge Indices
        self.edge_index_intersection_to_intersection = [
            (self.intersections.index(edge.getFromNode().getID()), self.intersections.index(edge.getToNode().getID()))
            for edge in [net.getEdge(edge_id) for edge_id in self.roads]]
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

    def get_queue_length_junction(self, signalized_intersection_id: str):
        queue_length = float(sum([traci.edge.getLastStepHaltingNumber(edge.getID()) for edge in self.net.getEdges()
                                  if edge.getToNode().getID() == signalized_intersection_id]))
        return queue_length

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

    def get_intersection_scaled_pressure(self, signalized_intersection_id: str):
        pressure = 0
        for movement in [m for m in self.movements.keys() if m[1] == signalized_intersection_id]:
            incoming_lanes = list({incoming_lane_id for incoming_lane_id, _, _ in self.movements[movement]})
            outgoing_lanes = list({outgoing_lane_id for _, _, outgoing_lane_id in self.movements[movement]})
            incoming_occupancy = self._get_vehicle_density(incoming_lanes)
            outgoing_occupancy = self._get_vehicle_density(outgoing_lanes)
            movement_pressure = incoming_occupancy - outgoing_occupancy
            pressure += movement_pressure
        return abs(pressure)

    @staticmethod
    def _get_vehicle_density(lanes: Union[str, List[str]], n_segments: int = 1):
        if isinstance(lanes, str):
            lanes = [lanes]
        total_segments_length = [0.0 for _ in range(n_segments)]
        total_segments_vehicle_length = [0.0 for _ in range(n_segments)]
        for lane_id in lanes:
            lane_length = traci.lane.getLength(lane_id)
            lane_segment_length = lane_length / n_segments
            total_segments_length = [total_segment_length + lane_segment_length
                                     for total_segment_length in total_segments_length]
            lane_segment_bins = [(segment+1) * lane_segment_length for segment in range(n_segments)]
            lane_vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            for vehicle_id in lane_vehicles:
                vehicle_length = traci.vehicle.getLength(vehicle_id)
                vehicle_lane_pos = traci.vehicle.getLanePosition(vehicle_id)
                for segment, segment_threshold in enumerate(lane_segment_bins):
                    if vehicle_lane_pos <= segment_threshold:
                        total_segments_vehicle_length[segment] += vehicle_length
                        break
        segments_vehicle_density = (np.array(total_segments_vehicle_length) / np.array(total_segments_length)).tolist()
        return segments_vehicle_density if n_segments > 1 else segments_vehicle_density[0]

    def get_intersection_level_metrics(self, metric: str = None):
        if metric == "queue_length":
            metric_fn = self.get_queue_length_junction
        elif metric == "pressure":
            metric_fn = self.get_intersection_pressure
        elif metric == "normalized_pressure":
            metric_fn = self.get_intersection_scaled_pressure
        else:
            return None
        metric_values = []
        for junction_id in self.signalized_intersections:
            metric_values.append(metric_fn(junction_id))
        return metric_values

    @abstractmethod
    def get_rewards(self) -> torch.Tensor:
        pass


class MaxPressureTrafficRepresentation(TrafficRepresentation):

    def get_state(self) -> HeteroData:
        data = HeteroData()
        x = []
        for phase, movements in self.phase_movements.items():
            incoming_lanes = set(incoming_lane for movement in movements
                                 for incoming_lane, _, _ in self.movements[movement])
            outgoing_lanes = set(outgoing_lane for movement in movements
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


class LitTrafficRepresentation(TrafficRepresentation):

    def get_state(self) -> Data:
        x = []
        for intersection_id in self.signalized_intersections:
            x_intersection= [0.0 for _ in range(len(self.get_phases(intersection_id)))]
            x_intersection[self.get_current_phase(intersection_id)] = 1.0
            in_approaches = sort_approaches_clockwise([edge for edge in self.net.getEdges()
                                                       if edge.getToNode().getID() == intersection_id])
            lanes = [lane for road in in_approaches for lane in road.getLanes()]
            for lane in lanes:
                x_intersection.append(float(traci.lane.getLastStepVehicleNumber(lane.getID())))
            x.append(x_intersection)
        x = torch.tensor(x)
        return Data(x=x)

    def get_rewards(self) -> torch.Tensor:
        return - torch.tensor(self.get_intersection_level_metrics("queue_length"))


class PressLightTrafficRepresentation(TrafficRepresentation):

    def __init__(self, *args, n_segments: int = 3):
        super(PressLightTrafficRepresentation, self).__init__(*args)
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


class GeneraLightTrafficRepresentation(TrafficRepresentation):

    def __init__(self, *args, segment_length: float = 20.0):
        super(GeneraLightTrafficRepresentation, self).__init__(*args)
        self.segment_length = segment_length
        self.lane_segments = []
        self.lane_segment_bins = {}
        self.lane_segment_proportional_length = {}
        for lane_id in self.lanes:
            lane = self.net.getLane(lane_id)
            lane_length = lane.getLength()
            n_segments = math.ceil(lane_length / self.segment_length)
            segment_bins = [((i+1) * self.segment_length if i < n_segments - 1 else lane_length)
                            for i in range(n_segments)]
            segment_length = [(segment_bins[i] if i == 0 else segment_bins[i] - segment_bins[i-1]) / self.segment_length
                              for i in range(n_segments)]
            self.lane_segment_bins[lane_id] = segment_bins
            self.lane_segment_proportional_length[lane_id] = segment_length
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

        self.edge_index_lane_to_downstream_movement = []
        self.edge_index_lane_to_upstream_movement = []
        self.edge_attr_lane_to_downstream_movement = []
        self.edge_attr_lane_to_upstream_movement = []
        lane_movements = defaultdict(lambda: {})
        for lane_id in self.lanes:
            lane_movements[lane_id] = set([movement[2] for movement in self.movements
                                           for incoming_lane_id, _, _ in self.movements[movement]
                                           if incoming_lane_id == lane_id])
        for movement in self.movements.keys():
            movement_incoming_lanes = set([incoming_lane for incoming_lane, _, _ in self.movements[movement]])
            incoming_lanes = [lane.getID() for lane in self.net.getEdge(movement[0]).getLanes()]
            incoming_involved = [1 if lane_id in movement_incoming_lanes else 0 for lane_id in incoming_lanes]
            incoming_involved_start = incoming_involved.index(1)
            incoming_involved_end = len(incoming_lanes) - list(reversed(incoming_involved)).index(1) - 1
            n_incoming = len(incoming_lanes)
            n_left_incoming_uninvolved = incoming_involved_start
            n_incoming_involved = incoming_involved_end - incoming_involved_start + 1
            n_right_incoming_uninvolved = n_incoming - n_left_incoming_uninvolved - n_incoming_involved
            incoming_lanes_relative_pos = \
                list(reversed(range(1, n_left_incoming_uninvolved + 1))) + \
                [0 for _ in range(n_incoming_involved)] + \
                list(range(1, n_right_incoming_uninvolved + 1))
            for lane_id, rel_pos in zip(incoming_lanes, incoming_lanes_relative_pos):
                self.edge_index_lane_to_downstream_movement.append((self.lanes.index(lane_id),
                                                                    list(self.movements.keys()).index(movement)))
                self.edge_attr_lane_to_downstream_movement.append([rel_pos])

            movement_outgoing_lanes = set([outgoing_lane for _, _, outgoing_lane in self.movements[movement]])
            outgoing_lanes = [lane.getID() for lane in self.net.getEdge(movement[2]).getLanes()]
            outgoing_involved = [1 if lane_id in movement_outgoing_lanes else 0 for lane_id in outgoing_lanes]
            outgoing_involved_start = outgoing_involved.index(1)
            outgoing_involved_end = len(outgoing_lanes) - list(reversed(outgoing_involved)).index(1) - 1
            n_outgoing = len(outgoing_lanes)
            n_left_outgoing_uninvolved = outgoing_involved_start
            n_outgoing_involved = outgoing_involved_end - incoming_involved_start + 1
            n_right_outgoing_uninvolved = n_outgoing - n_left_outgoing_uninvolved - n_outgoing_involved
            outgoing_lanes_relative_pos = \
                list(reversed(range(1, n_left_outgoing_uninvolved + 1))) + \
                [0 for _ in range(n_outgoing_involved)] + \
                list(range(1, n_right_outgoing_uninvolved + 1))
            for lane_id, rel_pos in zip(outgoing_lanes, outgoing_lanes_relative_pos):
                self.edge_index_lane_to_upstream_movement.append((self.lanes.index(lane_id),
                                                                  list(self.movements.keys()).index(movement)))
                self.edge_attr_lane_to_upstream_movement.append([rel_pos])

        self.edge_attr_lane_to_downstream_movement = torch.tensor(self.edge_attr_lane_to_downstream_movement)
        self.edge_attr_lane_to_upstream_movement = torch.tensor(self.edge_attr_lane_to_upstream_movement)

        self.edge_attr_movement_to_phase = []
        for movement, phase in [(list(self.movements.keys())[m], self.phases[p])
                                for m, p in self.edge_index_movement_to_phase]:
            params = self.phase_movement_params[(phase, movement)]
            protected = params["protected"]
            permitted = not protected
            self.edge_attr_movement_to_phase.append([float(permitted), float(protected)])
        self.edge_attr_movement_to_phase = torch.tensor(self.edge_attr_movement_to_phase)

        self.edge_attr_phase_to_phase = []
        for phase_j, phase_i in [(self.phases[j], self.phases[i]) for j, i in self.edge_index_phase_to_phase]:
            phase_j_movements = self.phase_movements[phase_j]
            phase_i_movements = self.phase_movements[phase_i]
            intersecting_movements = set(phase_j_movements).intersection(set(phase_i_movements))
            partial_competing = False
            if len(intersecting_movements) > 0:
                partial_competing = True
            self.edge_attr_phase_to_phase.append([int(partial_competing)])
        self.edge_attr_phase_to_phase = torch.tensor(self.edge_attr_phase_to_phase)

    def get_state(self) -> HeteroData:
        data = HeteroData()
        data["lane_segment"].x = self._get_lane_segment_features()
        data["lane"].x = self._get_lane_features()
        data["movement"].x = self._get_movement_features()
        data["phase"].x = self._get_phase_features()
        data["intersection"].x = self._get_signalized_intersection_features()

        data["lane_segment", "to", "lane"].edge_index = self.edge_index_to_tensor(
            self.edge_index_lane_segment_to_lane)
        data["lane", "to_downstream", "movement"].edge_index = self.edge_index_to_tensor(
            self.edge_index_lane_to_downstream_movement)
        data["lane", "to_upstream", "movement"].edge_index = self.edge_index_to_tensor(
            self.edge_index_lane_to_upstream_movement
        )
        data["movement", "to", "phase"].edge_index = self.edge_index_to_tensor(self.edge_index_movement_to_phase)
        data["phase", "to", "phase"].edge_index = self.edge_index_to_tensor(self.edge_index_phase_to_phase)
        data["phase", "to", "intersection"].edge_index = self.edge_index_to_tensor(
            self.edge_index_phase_to_intersection)

        data["lane_segment", "to", "lane"].edge_attr = self.edge_attr_lane_segment_to_lane
        data["lane", "to_downstream", "movement"].edge_attr  =self.edge_attr_lane_to_downstream_movement
        data["lane", "to_upstream", "movement"].edge_attr = self.edge_attr_lane_to_upstream_movement
        data["movement", "to", "phase"].edge_attr = self.edge_attr_movement_to_phase
        data["phase", "to", "phase"].edge_attr = self.edge_attr_phase_to_phase

        return data

    def _get_lane_segment_features(self) -> torch.Tensor:
        x = []
        for lane_id in self.lanes:
            segment_veh_len = [0.0 for _ in range(len(self.lane_segment_bins[lane_id]))]
            for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                veh_len = traci.vehicle.getLength(veh_id)
                veh_pos = np.digitize(traci.vehicle.getLanePosition(veh_id), self.lane_segment_bins[lane_id])
                segment_veh_len[veh_pos] += veh_len
            segment_veh_density = np.clip(
                np.array(segment_veh_len) /
                (np.array(self.lane_segment_proportional_length[lane_id]) * self.segment_length),
                0.0, 1.0
            )
            for segment_data in zip(segment_veh_density,
                                    self.lane_segment_proportional_length[lane_id]):
                x.append(list(segment_data))
        return torch.tensor(x, dtype=torch.float32)

    def _get_lane_features(self) -> torch.Tensor:
        x = []
        for _ in self.lanes:
            x.append([1.0])
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
        return torch.tensor(x, dtype=torch.float32)

    def _get_phase_features(self) -> torch.Tensor:
        x = []
        for phase in self.phases:
            intersection_id, phase_idx = phase
            phase_active = True if self.get_current_phase(intersection_id) == phase_idx else False
            x_phase = [float(phase_active)]
            x.append(x_phase)
        return torch.tensor(x, dtype=torch.float32)

    def _get_signalized_intersection_features(self) -> torch.Tensor:
        x = []
        for _ in self.signalized_intersections:
            x.append([1.0])
        return torch.tensor(x, dtype=torch.float32)

    def get_rewards(self) -> torch.Tensor:
        return - torch.tensor(self.get_intersection_level_metrics("normalized_pressure"))


class TestTrafficRepresentation(TrafficRepresentation):

    def __init__(self, *args, n_segments: int = 3):
        super(TestTrafficRepresentation, self).__init__(*args)
        self.n_segments = n_segments

    def get_state(self) -> HeteroData:
        data = HeteroData()
        data["movement"].x = self._get_movement_features()
        data["phase"].x = self._get_phase_features()
        data["intersection"].x = self._get_signalized_intersection_features()
        data["movement", "to", "phase"].edge_index = self.edge_index_to_tensor(self.edge_index_movement_to_phase)
        data["phase", "to", "phase"].edge_index = self.edge_index_to_tensor(self.edge_index_phase_to_phase)
        data["phase", "to", "intersection"].edge_index = self.edge_index_to_tensor(self.edge_index_phase_to_intersection)
        return data

    def _get_movement_features(self) -> torch.Tensor:
        x = []
        for movement in self.movements.keys():
            incoming_approach_id, intersection_id, outgoing_approach_id = movement
            movement_allowed = False
            movement_protected = False
            if intersection_id in self.signalized_intersections:
                phase = (intersection_id, self.get_current_phase(intersection_id))
                if movement in self.phase_movements[phase]:
                    movement_params = self.phase_movement_params[(phase, movement)]
                    movement_allowed = True
                    movement_protected = movement_params["protected"]
            else:
                movement_allowed = True
                movement_protected = True
            len_incoming_approach = self.net.getEdge(incoming_approach_id).getLength() / 1_000
            len_outgoing_approach = self.net.getEdge(outgoing_approach_id).getLength() / 1_000
            incoming_lanes = list({incoming_lane_id for incoming_lane_id, _, _ in self.movements[movement]})
            outgoing_lanes = list({outgoing_lane_id for _, _, outgoing_lane_id in self.movements[movement]})
            n_incoming_lanes = len(incoming_lanes) / 4
            n_outgoing_lanes = len(outgoing_lanes) / 4
            incoming_segments_vehicle_density = self._get_vehicle_density(incoming_lanes, self.n_segments)
            outgoing_approach_vehicle_density = self._get_vehicle_density(outgoing_lanes)
            x_movement = [float(movement_allowed)] + [float(movement_protected)] + incoming_segments_vehicle_density + \
                         [outgoing_approach_vehicle_density] + [len_incoming_approach] + [len_outgoing_approach] + \
                         [n_incoming_lanes] + [n_outgoing_lanes]
            x.append(x_movement)
        return torch.tensor(x)

    def _get_phase_features(self) -> torch.Tensor:
        x = []
        for phase in self.phases:
            intersection_id, phase_idx = phase
            phase_active = True if self.get_current_phase(intersection_id) == phase_idx else False
            x_phase = [float(phase_active)]
            x.append(x_phase)
        return torch.tensor(x)

    def _get_signalized_intersection_features(self) -> torch.Tensor:
        x = []
        for _ in self.signalized_intersections:
            x.append([1.0])
        return torch.tensor(x)

    def get_rewards(self) -> torch.Tensor:
        return - torch.tensor(self.get_intersection_level_metrics("normalized_pressure"))
