import gzip
import itertools
from sumolib import net
from typing import Tuple, List, Dict
from xml.sax import parse

import libsumo as traci
import numpy as np


class TrafficNet(net.Net):

    def __init__(self):
        super(TrafficNet, self).__init__()
        self.intersections, self.signalized_intersections, self.approaches, self.lanes, self.movements, self.phases = (
            None, None, None, None, None, None)
        self.index_in_lane_to_movement, self.index_out_lane_to_movement = None, None
        self.index_segment_to_down_movement, self.index_segment_to_up_movement = None, None
        self.index_movement_to_phase = None
        self.index_movement_to_intersection, self.index_phase_to_intersection = None, None
        self.index_movement_to_down_movement, self.index_movement_to_up_movement = None, None
        self.index_phase_to_phase = None

        self.segment_length = None
        self.segments = None
        self.include_last_segment = None
        self.index_segment_to_lane = None

    def init(self):
        self.intersections = [intersection_id.getID() for intersection_id in self.getNodes()]
        self.signalized_intersections = [intersection_id for intersection_id in self.intersections
                                         if self.getNode(intersection_id).getType() == "traffic_light"]
        self.approaches = [road.getID() for road in self.getEdges()]
        self.lanes = [lane.getID() for road_id in self.approaches for lane in self.getEdge(road_id).getLanes()]
        self.movements = [(con.getFromLane().getID(), intersection, con.getToLane().getID())
                          for intersection in self.intersections
                          for con in self.getNode(intersection).getConnections()]
        self.phases = []
        for intersection in self.signalized_intersections:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(intersection)[1]
            states = [p.state for p in logic.phases]
            [self.phases.append((intersection, state)) for state in states]

        self.index_in_lane_to_movement = [(movement[0], movement) for movement in self.movements]
        self.index_out_lane_to_movement = [(movement[2], movement) for movement in self.movements]
        self.index_movement_to_phase = [(movement, phase)
                                        for phase in self.phases
                                        for movement in self.get_movement_signals(phase).keys()]
        self.index_phase_to_intersection = [(phase, phase[0]) for phase in self.phases]
        self.index_movement_to_intersection = [(movement, movement[1]) for movement in self.movements
                                               if movement[1] in self.signalized_intersections]
        self.index_movement_to_down_movement = [
            (movement, down_movement)
            for movement in self.movements for down_movement in self.movements
            if self.getLane(movement[2]).getEdge().getID() == self.getLane(down_movement[0]).getEdge().getID()
        ]
        self.index_movement_to_up_movement = [
            (movement, up_movement)
            for movement in self.movements for up_movement in self.movements
            if self.getLane(movement[0]).getEdge().getID() == self.getLane(up_movement[2]).getEdge().getID()
        ]
        self.index_phase_to_phase = []
        for intersection in self.signalized_intersections:
            phases = self.get_phases(intersection)
            index_phase_to_phase = list(itertools.combinations(phases, 2))
            index_phase_to_phase += [(phase_b, phase_a) for phase_a, phase_b in index_phase_to_phase]
            self.index_phase_to_phase += index_phase_to_phase

    def init_segments(self, segment_length: float, include_last: bool = False):
        self.segment_length = segment_length
        self.segments = []
        self.include_last_segment = include_last
        for lane in self.lanes:
            lane_length = self.getLane(lane).getLength()
            n_segments = int(lane_length // self.segment_length)
            n_segments += 1 if lane_length % self.segment_length > 0 and include_last else 0
            self.segments += [(lane, i) for i in range(n_segments)]
        self.index_segment_to_lane = [(segment, segment[0]) for segment in self.segments]
        self.index_segment_to_down_movement = [(segment, movement)
                                               for movement in self.movements
                                               for segment in self.get_segments(movement[0])]
        self.index_segment_to_up_movement = [(segment, movement)
                                             for movement in self.movements
                                             for segment in self.get_segments(movement[2])]

    def get_length(self, segment: Tuple[str, int] = None, lane: str = None, approach: str = None) -> float:
        assert bool(segment is not None) + bool(lane is not None) + bool(approach is not None) == 1.0
        if lane is not None:
            return self.getLane(lane).getLength()
        elif approach is not None:
            return self.getEdge(lane).getLength()
        else:
            start, end = self.get_bins(segment)
            return end - start

    def get_segments(self, lane: str) -> List[Tuple[str, int]]:
        assert self.segments is not None
        return [segment for segment in self.segments if segment[0] == lane]

    def get_bins(self, segment: Tuple[str, int]) -> Tuple[float, float]:
        lane, seg_idx = segment
        lane_length = self.getLane(lane).getLength()
        n_segments = int(lane_length // self.segment_length)
        start = seg_idx * self.segment_length
        end = (seg_idx + 1) * self.segment_length if seg_idx < n_segments else lane_length
        return start, end

    def get_vehicles(self, segment: Tuple[str, int] = None, lane: str = None, approach: str = None) -> List[str]:
        assert bool(segment is not None) + bool(lane is not None) + bool(approach is not None) == 1.0
        if lane is not None:
            return traci.lane.getLastStepVehicleIDs(lane)
        elif approach is not None:
            return traci.edge.getLastStepVehicleIDs(approach)
        else:
            assert segment in self.segments
            lane, seg_idx = segment
            start, end = self.get_bins(segment)
            lane_length = self.getLane(lane).getLength()
            return [veh for veh in traci.lane.getLastStepVehicleIDs(lane)
                    if start <= lane_length - traci.vehicle.getLanePosition(veh) < end]

    def get_vehicle_length(self, segment: Tuple[str, int] = None, lane: str = None, approach: str = None) -> float:
        assert bool(segment is not None) + bool(lane is not None) + bool(approach is not None) == 1.0
        vehicles = self.get_vehicles(segment=segment, lane=lane, approach=approach)
        return float(sum([traci.vehicle.getLength(veh) for veh in vehicles]))

    def get_vehicle_number(self, segment: Tuple[str, int] = None, lane: str = None, approach: str = None) -> int:
        assert bool(segment is not None) + bool(lane is not None) + bool(approach is not None) == 1.0
        return len(self.get_vehicles(segment=segment, lane=lane, approach=approach))

    def get_vehicle_density(self, segment: Tuple[str, int] = None, lane: str = None, approach: str = None) -> float:
        assert bool(segment is not None) + bool(lane is not None) + bool(approach is not None) == 1.0
        if approach is not None:
            length = sum([self.get_length(lane=lane.getID()) for lane in self.getEdge(approach).getLanes()])
        else:
            length = self.get_length(segment=segment, lane=lane)
        vehicle_length = self.get_vehicle_length(segment=segment, lane=lane, approach=approach)
        return float(np.clip(np.array(vehicle_length) / length, 0.0, 1.0))

    def get_current_signal(self, movement: Tuple[str, str, str]) -> str:
        incoming_lane, intersection, outgoing_lane = movement
        if intersection not in self.signalized_intersections:
            return "G"
        movement_signals = self.get_movement_signals(self.get_current_phase(intersection))
        return movement_signals[movement]

    @staticmethod
    def get_movement_signals(phase: Tuple[str, str]) -> Dict[str, str]:
        intersection, state = phase
        movements = [(con[0][0], intersection, con[0][1])
                     for con in traci.trafficlight.getControlledLinks(intersection)]
        signals = [*state]
        return {movement: signal for movement, signal in zip(movements, signals)}

    def get_current_movement_signals(self, intersection: str) -> Dict[str, str]:
        current_phase = self.get_current_phase(intersection)
        return self.get_movement_signals(current_phase)

    def get_current_phase(self, intersection: str) -> Tuple[str, str]:
        assert intersection in self.signalized_intersections
        return intersection, traci.trafficlight.getRedYellowGreenState(intersection)

    def get_phases(self, intersection: str) -> List[Tuple[str, str]]:
        assert intersection in self.signalized_intersections
        return [(intersection, phase.state)
                for phase in traci.trafficlight.getCompleteRedYellowGreenDefinition(intersection)[1].phases]

    def get_current_phase_idx(self, intersection: str) -> int:
        assert intersection in self.signalized_intersections
        current_state = traci.trafficlight.getRedYellowGreenState(intersection)
        states = [phase.state
                  for phase in traci.trafficlight.getCompleteRedYellowGreenDefinition(intersection)[1].phases]
        return states.index(current_state)

    def get_pressure(self, intersection: str = None, movement: Tuple[str, str, str] = None,
                     phase: Tuple[str, str] = None, method: str = "count") -> float:
        assert bool(intersection is not None) + bool(movement is not None) + bool(phase is not None) == 1.0
        if intersection is not None:
            assert intersection in self.signalized_intersections
            movements = [(con[0][0], intersection, con[0][1])
                         for con in traci.trafficlight.getControlledLinks(intersection)]
            return abs(sum([self.get_pressure(movement=movement, method=method) for movement in movements]))
        elif movement is not None:
            if method == "count":
                return self.get_pressure_count(movement)
            elif method == "density":
                return self.get_vehicle_density(movement)
            else:
                raise Exception(f"Pressure Method \"{method}\" not implemented")
        else:
            movements = list(self.get_movement_signals(phase).keys())
            return sum([self.get_pressure(movement=movement, method=method) for movement in movements])

    def get_pressure_count(self, movement: Tuple[str, str, str]):
        veh_number_in = self.get_vehicle_number(lane=movement[0])
        veh_number_out = self.get_vehicle_number(lane=movement[2])
        return veh_number_in - veh_number_out

    def get_pressure_density(self, movement: Tuple[str, str, str]):
        veh_density_in = self.get_vehicle_density(lane=movement[0])
        veh_density_out = self.get_vehicle_density(lane=movement[2])
        return veh_density_in - veh_density_out

    def get_queue_length(self, intersection: str = None, lane: str = None, approach: str = None) -> float:
        assert bool(intersection is not None) + bool(lane is not None) + bool(approach is not None) == 1.0
        if intersection is not None:
            assert intersection in self.signalized_intersections
            lanes = {con[0][0] for con in traci.trafficlight.getControlledLinks(intersection)}
            return sum([self.get_queue_length(lane=lane) for lane in lanes])
        elif lane is not None:
            return traci.lane.getLastStepHaltingNumber(lane)
        else:
            return traci.edge.getLastStepHaltingNumber(approach)


class TrafficNetReader(net.NetReader):

    def __init__(self, **others):
        super(TrafficNetReader, self).__init__(**others)
        self._net = others.get("net", TrafficNet())


def read_traffic_net(filename, **others) -> TrafficNet:
    netreader = TrafficNetReader(**others)
    try:
        parse(gzip.open(filename), netreader)
    except IOError:
        parse(filename, netreader)
    net = netreader.getNet()
    net.init()
    return net
