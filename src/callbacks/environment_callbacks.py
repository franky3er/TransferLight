from abc import ABC
import os.path
from pathlib import Path
from collections import defaultdict

import libsumo as traci
import pandas as pd


class EnvironmentCallback(ABC):

    def __init__(self):
        self.episode = 0
        self.episode_time = 0
        self.total_time = 0
        self.episode_step = 0
        self.total_step = 0

    def on_time_step_start(self, environment):
        pass

    def on_time_step_end(self, environment):
        self.episode_time += 1
        self.total_time += 1

    def on_action_step_start(self, environment):
        pass

    def on_action_step_end(self, environment):
        self.episode_step += 1
        self.total_step += 1

    def on_episode_start(self, environment):
        self.episode_time = 0
        self.episode_step = 0

    def on_episode_end(self, environment):
        self.episode += 1

    def on_close(self, environment):
        pass


class StatsCallback(EnvironmentCallback):

    def __init__(self, stats_dir: str):
        super(StatsCallback, self).__init__()
        self.stats_file_extension = None
        self.stats_dir = stats_dir
        self.stats = defaultdict(lambda: pd.DataFrame())

    def on_episode_end(self, environment):
        self._store_results()
        super().on_episode_end(environment)

    def on_close(self, environment):
        self._store_results()
        super().on_close(environment)

    def _store_results(self):
        for scenario, records in self.stats.items():
            stats_path = os.path.join(self.stats_dir, f"{scenario}.{self.stats_file_extension}")
            os.makedirs(str(Path(stats_path).parent), exist_ok=True)
            records.to_csv(stats_path, index=False)
            self.stats = defaultdict(lambda: pd.DataFrame())


class VehicleStatsCallback(StatsCallback):

    def __init__(self, stats_dir: str):
        super(VehicleStatsCallback, self).__init__(stats_dir)
        self.stats_file_extension = "vehicle.csv"

    def on_time_step_end(self, environment):
        vehicles = traci.vehicle.getIDList()
        records = defaultdict(lambda: [])
        scenario = os.path.join(*os.path.normpath(environment.scenario).split(os.path.sep)[-2:]).split(".")[0]
        for vehicle in vehicles:
            records["scenario"].append(scenario)
            records["episode"].append(self.episode)
            records["episode_step"].append(self.episode_time)
            records["total_step"].append(self.total_time)
            records["episode_time"].append(self.episode_time)
            records["total_time"].append(self.total_time)
            records["vehicle"].append(vehicle)
            records["speed"].append(traci.vehicle.getSpeed(vehicle))
            records["distance"].append(traci.vehicle.getDistance(vehicle))
            records["position_x"].append(traci.vehicle.getPosition(vehicle)[0])
            records["position_y"].append(traci.vehicle.getPosition(vehicle)[1])
            records["position_lane"].append(traci.vehicle.getLanePosition(vehicle))
            records["road"].append(traci.vehicle.getRoadID(vehicle))
            records["lane"].append(traci.vehicle.getLaneID(vehicle))
        self.stats[scenario] = pd.concat([self.stats[scenario], pd.DataFrame(records)])

        super().on_time_step_end(environment)


class IntersectionStatsCallback(StatsCallback):

    def __init__(self, stats_dir: str):
        super(IntersectionStatsCallback, self).__init__(stats_dir)
        self.stats_file_extension = "intersection.csv"

    def on_time_step_end(self, environment):
        intersections = traci.trafficlight.getIDList()
        records = defaultdict(lambda: [])
        scenario = os.path.join(*os.path.normpath(environment.scenario).split(os.path.sep)[-2:]).split(".")[0]
        for intersection in intersections:
            records["scenario"].append(scenario)
            records["episode"].append(self.episode)
            records["episode_step"].append(self.episode_time)
            records["total_step"].append(self.total_time)
            records["episode_time"].append(self.episode_time)
            records["total_time"].append(self.total_time)
            records["intersection"].append(intersection)
            records["signal"].append(environment.net.get_current_phase(intersection)[1])
            records["queue_length"].append(environment.net.get_queue_length(intersection=intersection))
            records["pressure"].append(environment.net.get_pressure(intersection=intersection))
            records["normalized_pressure"].append(environment.net.get_pressure(
                intersection=intersection, method="density"))
            records["waiting_time"] = environment.net.get_waiting_time(intersection=intersection)
        self.stats[scenario] = pd.concat([self.stats[scenario], pd.DataFrame(records)])

        super().on_time_step_end(environment)
