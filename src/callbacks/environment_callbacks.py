from abc import ABC, abstractmethod
import ntpath
import os.path
from collections import defaultdict

import libsumo as traci
import pandas as pd


class EnvironmentCallback(ABC):

    def __init__(self):
        self.episode = 0
        self.episode_step = 0
        self.total_step = 0

    def on_step_start(self, environment):
        pass

    def on_step_end(self, environment):
        self.episode_step += 1
        self.total_step += 1

    def on_episode_start(self, environment):
        self.episode_step = 0

    def on_episode_end(self, environment):
        self.episode += 1

    def on_close(self, environment):
        pass


class StatsCallback(EnvironmentCallback):

    def __init__(self, results_dir: str):
        super(StatsCallback, self).__init__()
        self.results_file_extension = None
        self.results_dir = results_dir
        self.df = None

    def on_episode_end(self, environment):
        self._store_results(environment)
        super().on_episode_end(environment)

    def on_close(self, environment):
        self._store_results(environment)
        super().on_close(environment)

    def _store_results(self, environment):
        if self.df is not None:
            results_name = f"{ntpath.split(environment.scenario)[1].split('.')[0]}.{self.results_file_extension}"
            results_path = os.path.join(self.results_dir, results_name)
            os.makedirs(self.results_dir, exist_ok=True)
            self.df.to_csv(results_path, index=False)
            self.df = None


class VehicleStatsCallback(StatsCallback):

    def __init__(self, results_dir: str):
        super(VehicleStatsCallback, self).__init__(results_dir)
        self.results_file_extension = "vehicle.csv"

    def on_step_end(self, environment):
        vehicles = traci.vehicle.getIDList()
        records = defaultdict(lambda: [])
        for vehicle in vehicles:
            records["scenario"].append(environment.scenario)
            records["episode"].append(self.episode)
            records["episode_step"].append(self.episode_step)
            records["total_step"].append(self.total_step)
            records["vehicle"].append(vehicle)
            records["speed"].append(traci.vehicle.getSpeed(vehicle))
            records["distance"].append(traci.vehicle.getDistance(vehicle))
            records["position_x"].append(traci.vehicle.getPosition(vehicle)[0])
            records["position_y"].append(traci.vehicle.getPosition(vehicle)[1])
            records["position_lane"].append(traci.vehicle.getLanePosition(vehicle))
            records["road"].append(traci.vehicle.getRoadID(vehicle))
            records["lane"].append(traci.vehicle.getLaneID(vehicle))
        records = pd.DataFrame(data=records)
        self.df = records if self.df is None else pd.concat([self.df, records])

        super().on_step_end(environment)


class IntersectionStatsCallback(StatsCallback):

    def __init__(self, results_dir: str):
        super(IntersectionStatsCallback, self).__init__(results_dir)
        self.results_file_extension = "intersection.csv"

    def on_step_end(self, environment):
        intersections = traci.trafficlight.getIDList()
        records = defaultdict(lambda: [])
        for intersection in intersections:
            records["scenario"].append(environment.scenario)
            records["intersection"].append(intersection)
            records["queue_length"].append(environment.net.get_queue_length(intersection=intersection))
            records["pressure"].append(environment.net.get_pressure(intersection=intersection))
            records["normalized_pressure"].append(environment.net.get_pressure(
                intersection=intersection, method="density"))
            records["waiting_time"] = environment.net.get_waiting_time(intersection=intersection)
        records = pd.DataFrame(data=records)
        self.df = records if self.df is None else pd.concat([self.df, records])

        super().on_step_end(environment)
