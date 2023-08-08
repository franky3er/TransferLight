import gzip
import itertools
from sumolib import net
from typing import Tuple, List, Dict
from xml.sax import parse

import libsumo as traci


class NetState(net.Net):

    def __init__(self):
        super(NetState, self).__init__()
        self.intersections, self.signalized_intersections, self.approaches, self.lanes, self.movements, self.phases = (
            None, None, None, None, None, None)
        self.index_incoming_lane_to_movement, self.index_outgoing_lane_to_movement = None, None
        self.index_movement_to_phase = None
        self.index_movement_to_intersection, self.index_phase_to_intersection = None, None
        self.index_movement_to_down_movement, self.index_movement_to_up_movement = None, None
        self.index_phase_to_phase = None

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

        self.index_incoming_lane_to_movement = [(movement[0], movement) for movement in self.movements]
        self.index_outgoing_lane_to_movement = [(movement[2], movement) for movement in self.movements]
        self.index_movement_to_phase = [(movement, phase)
                                        for phase in self.phases
                                        for movement in self.get_movement_signals(phase).keys()]
        self.index_phase_to_intersection = [(phase, phase[0]) for phase in self.phases]
        self.index_movement_to_intersection = [(movement, movement[0]) for movement in self.movements]
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
                     phase: Tuple[str, str] = None) -> float:
        assert (intersection is not None) != (movement is not None) != (phase is not None)
        if intersection is not None:
            assert intersection in self.signalized_intersections
            movements = [(con[0][0], intersection, con[0][1])
                         for con in traci.trafficlight.getControlledLinks(intersection)]
            return abs(sum([self.get_pressure(movement=movement) for movement in movements]))
        elif movement is not None:
            n_in_vehicles = float(traci.lane.getLastStepVehicleNumber(movement[0]))
            n_out_vehicles = float(traci.lane.getLastStepVehicleNumber(movement[2]))
            return n_in_vehicles - n_out_vehicles
        else:
            movements = list(self.get_movement_signals(phase).keys())
            return sum([self.get_pressure(movement=movement) for movement in movements])

    def get_queue_length(self, intersection: str = None, lane: str = None, approach: str = None) -> float:
        assert (intersection is not None) != (lane is not None) != (approach is not None)
        if intersection is not None:
            assert intersection in self.signalized_intersections
            lanes = {con[0][0] for con in traci.trafficlight.getControlledLinks(intersection)}
            return sum([self.get_queue_length(lane=lane) for lane in lanes])
        elif lane is not None:
            return traci.lane.getLastStepHaltingNumber(lane)
        else:
            return traci.edge.getLastStepHaltingNumber(approach)


class NetStateReader(net.NetReader):

    def __init__(self, **others):
        super(NetStateReader, self).__init__(**others)
        self._net = others.get("net", NetState())


def readNetState(filename, **others) -> NetState:
    netreader = NetStateReader(**others)
    try:
        parse(gzip.open(filename), netreader)
    except IOError:
        parse(filename, netreader)
    net = netreader.getNet()
    net.init()
    return net
