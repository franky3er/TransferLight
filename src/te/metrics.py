import libsumo as traci
from sumolib.net import Net


class Metric:

    def __init__(self, net: Net):
        self.net = net

    def intersection_queue_length(self, intersection_id: str):
        queue_length = float(sum([traci.edge.getLastStepHaltingNumber(edge.getID()) for edge in self.net.getEdges()
                                  if edge.getToNode().getID() == intersection_id]))
        return queue_length

    def intersection_waiting_time(self, intersection_id: str):
        waiting_time = sum([traci.edge.getWaitingTime(edge.getID()) for edge in self.net.getEdges()
                            if edge.getToNode().getID() == intersection_id])
        return waiting_time

    def intersection_pressure(self, intersection_id: str, method: str = "density"):
        pass
