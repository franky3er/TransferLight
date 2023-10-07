from abc import ABC, abstractmethod
from dataclasses import dataclass
import os.path
from pathlib import Path
from typing import Optional

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
        self.scenario = None
        self.stats = None
        self.previous_records = None
        self.current_records = None

    def on_episode_start(self, environment):
        self.scenario = os.path.join(*os.path.normpath(environment.scenario).split(os.path.sep)[-2:]).split(".")[0]
        self.stats = pd.DataFrame()
        self.previous_records = None
        self.current_records = None
        super().on_episode_start(environment)

    def on_time_step_end(self, environment):
        self._update_stats(environment)
        super().on_time_step_end(environment)

    def _update_stats(self, environment):
        self._update_current_records(environment)
        if self.previous_records is not None:
            self._update_previous_records(environment)
            self.stats = pd.concat([self.stats, pd.DataFrame(self.previous_records)])
        self.previous_records = self.current_records

    @abstractmethod
    def _update_current_records(self, environment):
        pass

    @abstractmethod
    def _update_previous_records(self, environment):
        pass

    def _store_stats(self):
        stats_path = os.path.join(self.stats_dir, f"{self.scenario}.{self.stats_file_extension}")
        os.makedirs(str(Path(stats_path).parent), exist_ok=True)
        self.stats.to_csv(stats_path, index=False)

    def on_episode_end(self, environment):
        if self.stats is not None:
            self._update_stats(environment)
            self._store_stats()
        super().on_episode_end(environment)

    def on_close(self, environment):
        if self.stats is not None:
            self._update_stats(environment)
            self._store_stats()
        super().on_close(environment)


@dataclass
class VehicleRecord:
    scenario: str
    step: int
    time: int
    vehicle: str
    speed: float
    distance: float
    lane_position: float
    approach: str
    lane: str
    departed: bool
    arrived: Optional[bool] = None


class VehicleStatsCallback(StatsCallback):

    def __init__(self, stats_dir: str):
        super(VehicleStatsCallback, self).__init__(stats_dir)
        self.stats_file_extension = "vehicle.csv"
        self.current_vehicles = None
        self.previous_vehicles = None

    def _update_current_records(self, environment):
        self.current_vehicles = set(traci.vehicle.getIDList())
        self.previous_vehicles = {record.vehicle for record in self.previous_records} \
            if self.previous_vehicles is not None else set()
        departed_vehicles = self.current_vehicles - self.previous_vehicles
        self.current_records = []
        for vehicle in self.current_vehicles:
            record = VehicleRecord(
                scenario=self.scenario,
                step=self.episode_step,
                time=self.episode_time,
                vehicle=vehicle,
                speed=traci.vehicle.getSpeed(vehicle),
                approach=traci.vehicle.getRoadID(vehicle),
                lane=traci.vehicle.getLaneID(vehicle),
                distance=traci.vehicle.getDistance(vehicle),
                lane_position=traci.vehicle.getLanePosition(vehicle),
                departed=vehicle in departed_vehicles,
            )
            self.current_records.append(record)

    def _update_previous_records(self, environment):
        arrived_vehicles = self.previous_vehicles - self.current_vehicles
        previous_records = []
        for prev_record in self.previous_records:
            prev_record.arrived = prev_record.vehicle in arrived_vehicles
            previous_records.append(prev_record)
        self.previous_records = previous_records


@dataclass
class IntersectionRecord:
    scenario: str
    step: int
    time: int
    intersection: int
    signal: str
    queue_length: int
    pressure: int
    normalized_pressure: float
    waiting_time: float


class IntersectionStatsCallback(StatsCallback):

    def __init__(self, stats_dir: str):
        super(IntersectionStatsCallback, self).__init__(stats_dir)
        self.stats_file_extension = "intersection.csv"
        self.current_intersections = None

    def _update_current_records(self, environment):
        self.current_intersections = traci.trafficlight.getIDList()
        self.current_records = []
        for intersection in self.current_intersections:
            record = IntersectionRecord(
                scenario=self.scenario,
                step=self.episode_step,
                time=self.episode_time,
                intersection=intersection,
                signal=environment.net.get_current_phase(intersection)[1],
                queue_length=environment.net.get_queue_length(intersection=intersection),
                pressure=environment.net.get_pressure(intersection=intersection),
                normalized_pressure=environment.net.get_pressure(intersection=intersection, method="density"),
                waiting_time=environment.net.get_waiting_time(intersection=intersection)
            )
            self.current_records.append(record)

    def _update_previous_records(self, environment):
        pass
