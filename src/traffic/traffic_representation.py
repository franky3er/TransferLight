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


class TestTrafficRepresentation(TrafficRepresentation):

    def __init__(self, *args, n_segments: int = 3):
        super(TestTrafficRepresentation, self).__init__(*args)
        self.n_segments = n_segments

    def get_state(self) -> Any:
        x = []
        for intersection_id in self.signalized_intersections:
            x_intersection = [0.0 for _ in range(len(self.get_phases(intersection_id)))]
            x_intersection[self.get_current_phase(intersection_id)] = 1.0

            for movement in [m for m in self.movements.keys() if m[1] == intersection_id]:
                x_movement = []
                incoming_approach, _, outgoing_approach = movement
                lane_movements = self.movements[movement]
                incoming_lanes = {lane_id for lane_id, _, _ in lane_movements}
                outgoing_lanes = {lane_id for _, _, lane_id in lane_movements}
                incoming_approach_data = self._get_incoming_approach_data(incoming_lanes)
                outgoing_approach_data = self._get_outgoing_approach_data(outgoing_lanes)
                x_movement += incoming_approach_data
                x_movement += outgoing_approach_data
                x_intersection += x_movement

            x.append(x_intersection)
        x = torch.tensor(x, dtype=torch.float32)
        return Data(x=x)

    def _get_incoming_approach_data(self, lanes: List[str]):
        segment_lengths = [0 for _ in range(self.n_segments)]
        segment_lengths_vehicles = [0 for _ in range(self.n_segments)]
        for lane_id in lanes:
            lane_segment_length = traci.lane.getLength(lane_id) / self.n_segments
            segment_lengths = [segment_length + lane_segment_length for segment_length in segment_lengths]
            lane_segment_bins = [(segment+1) * lane_segment_length for segment in range(self.n_segments)]
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            for vehicle_id in vehicle_ids:
                vehicle_len = traci.vehicle.getLength(vehicle_id)
                vehicle_pos = traci.vehicle.getLanePosition(vehicle_id)
                vehicle_seg = None
                for bin, border_val in enumerate(lane_segment_bins):
                    if vehicle_pos <= border_val:
                        vehicle_seg = bin
                        break
                segment_lengths_vehicles[vehicle_seg] += vehicle_len
        segment_lengths = np.array(segment_lengths)
        segment_lengths_vehicles = np.array(segment_lengths_vehicles)
        segment_data = (segment_lengths_vehicles / segment_lengths)
        return segment_data.tolist()

    def _get_outgoing_approach_data(self, lanes: List[str]):
        approach_length = 0
        approach_length_vehicles = 0
        for lane_id in lanes:
            approach_length += traci.lane.getLength(lane_id)
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            for vehicle_id in vehicle_ids:
                approach_length_vehicles += traci.vehicle.getLength(vehicle_id)
        return [approach_length_vehicles / approach_length]

    def get_rewards(self) -> torch.Tensor:
        return - torch.tensor(self.get_intersection_level_metrics("normalized_pressure"))


class HieraGLightTrafficRepresentation(TrafficRepresentation):

    def __init__(self, *args, n_segments: int = 3):
        super(HieraGLightTrafficRepresentation, self).__init__(*args)
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
