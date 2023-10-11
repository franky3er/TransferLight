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
        self.records = None

    def on_episode_start(self, environment):
        self.scenario = os.path.join(*os.path.normpath(environment.scenario).split(os.path.sep)[-2:]).split(".")[0]
        self.records = dict()
        super().on_episode_start(environment)

    def on_time_step_end(self, environment):
        self._update_stats(environment)
        super().on_time_step_end(environment)

    @abstractmethod
    def _update_stats(self, environment):
        pass

    def _store_stats(self):
        stats_path = os.path.join(self.stats_dir, f"{self.scenario}.{self.stats_file_extension}")
        os.makedirs(str(Path(stats_path).parent), exist_ok=True)
        stats = pd.DataFrame(self.records.values())
        stats.to_csv(stats_path, index=False)

    def on_episode_end(self, environment):
        if self.records is not None:
            self._store_stats()
        self.records = None
        self._clear()
        super().on_episode_end(environment)

    def on_close(self, environment):
        if self.records is not None:
            self._store_stats()
        self.records = None
        self._clear()
        super().on_close(environment)

    @abstractmethod
    def _clear(self):
        pass


@dataclass
class VehicleRecord:
    scenario: str
    vehicle: str
    departure_time: int
    arrival_time: int
    distance_seq: str
    lane_seq: str


class VehicleStatsCallback(StatsCallback):

    def __init__(self, stats_dir: str):
        super(VehicleStatsCallback, self).__init__(stats_dir)
        self.stats_file_extension = "vehicle.csv"
        self.current_vehicles = None
        self.previous_vehicles = None

    def _update_stats(self, environment):
        self.current_vehicles = set(traci.vehicle.getIDList())
        self._create_new_records()
        self._update_existing_records()
        self.previous_vehicles = self.current_vehicles

    def _create_new_records(self):
        departed_vehicles = self.current_vehicles - self.previous_vehicles \
            if self.previous_vehicles is not None else self.current_vehicles
        for vehicle in departed_vehicles:
            self.records[vehicle] = VehicleRecord(
                scenario=self.scenario,
                vehicle=vehicle,
                departure_time=self.episode_time,
                arrival_time=-1,
                distance_seq=str(traci.vehicle.getDistance(vehicle)),
                lane_seq=str(traci.vehicle.getLaneID(vehicle))
            )

    def _update_existing_records(self):
        self.previous_vehicles = set() if self.previous_vehicles is None else self.previous_vehicles
        arrived_vehicles = self.previous_vehicles - self.current_vehicles
        for vehicle in self.previous_vehicles:
            if vehicle in arrived_vehicles:
                self.records[vehicle].arrival_time = self.episode_time - 1
                continue
            self.records[vehicle].distance_seq = "|".join(
                [self.records[vehicle].distance_seq, str(traci.vehicle.getDistance(vehicle))])
            self.records[vehicle].lane_seq = "|".join(
                [self.records[vehicle].lane_seq, str(traci.vehicle.getLaneID(vehicle))])

    def _clear(self):
        self.previous_vehicles, self.current_vehicles = None, None


@dataclass
class IntersectionRecord:
    scenario: str
    intersection: int
    states: str
    queue_length_seq: str
    pressure_seq: str
    normalized_pressure_seq: str
    waiting_time_seq: str


class IntersectionStatsCallback(StatsCallback):

    def __init__(self, stats_dir: str):
        super(IntersectionStatsCallback, self).__init__(stats_dir)
        self.stats_file_extension = "intersection.csv"

    def _update_stats(self, environment):
        for intersection in traci.trafficlight.getIDList():
            if intersection not in list(self.records.keys()):
                self.records[intersection] = IntersectionRecord(
                    scenario=self.scenario,
                    intersection=intersection,
                    states=environment.net.get_current_phase(intersection)[1],
                    queue_length_seq=str(environment.net.get_queue_length(intersection=intersection)),
                    pressure_seq=str(environment.net.get_pressure(intersection=intersection)),
                    normalized_pressure_seq=str(
                        environment.net.get_pressure(intersection=intersection, method="density")),
                    waiting_time_seq=str(environment.net.get_waiting_time(intersection=intersection))
                )
            else:
                self.records[intersection].states = "|".join(
                    [self.records[intersection].states, environment.net.get_current_phase(intersection)[1]])
                self.records[intersection].queue_length_seq = "|".join(
                    [self.records[intersection].queue_length_seq,
                     str(environment.net.get_queue_length(intersection=intersection))])
                self.records[intersection].pressure_seq = "|".join(
                    [self.records[intersection].pressure_seq,
                     str(environment.net.get_pressure(intersection=intersection))])
                self.records[intersection].normalized_pressure_seq = "|".join(
                    [self.records[intersection].normalized_pressure_seq,
                     str(environment.net.get_pressure(intersection=intersection, method="density"))])
                self.records[intersection].waiting_time_seq = "|".join(
                    [self.records[intersection].waiting_time_seq,
                     str(environment.net.get_waiting_time(intersection=intersection))])

    def _clear(self):
        pass
