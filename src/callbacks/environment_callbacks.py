import os.path
from abc import ABC
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


class VehicleStats(EnvironmentCallback):

    def __init__(self, results_dir: str):
        super(VehicleStats, self).__init__()
        self.results_dir = results_dir
        self.df = None

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
            records["acceleration"].append(traci.vehicle.getAcceleration(vehicle))
            records["distance"].append(traci.vehicle.getDistance(vehicle))
            records["position_x"].append(traci.vehicle.getPosition(vehicle)[0])
            records["position_y"].append(traci.vehicle.getPosition(vehicle)[1])
            records["position_lane"].append(traci.vehicle.getLanePosition(vehicle))
            records["road"].append(traci.vehicle.getRoadID(vehicle))
            records["lane"].append(traci.vehicle.getLaneID(vehicle))
            records["CO2_emission"].append(traci.vehicle.getCOEmission(vehicle))
            records["CO_emission"].append(traci.vehicle.getCOEmission(vehicle))
            records["HC_emission"].append(traci.vehicle.getHCEmission(vehicle))
            records["PMx_emission"].append(traci.vehicle.getPMxEmission(vehicle))
            records["NOx_emission"].append(traci.vehicle.getNOxEmission(vehicle))
            records["noise_emission"].append(traci.vehicle.getNoiseEmission(vehicle))
            records["fuel_consumption"].append(traci.vehicle.getFuelConsumption(vehicle))
        records = pd.DataFrame(data=records)
        self.df = records if self.df is None else pd.concat([self.df, records])

        super().on_step_end(environment)

    def on_episode_end(self, environment):
        self._store_results(environment)
        super().on_episode_end(environment)

    def on_close(self, environment):
        self._store_results(environment)
        super().on_close(environment)

    def _store_results(self, environment):
        if self.df is not None:
            results_name = f"worker{environment.name}-episode{self.episode}.csv"
            results_path = os.path.join(self.results_dir, results_name)
            os.makedirs(self.results_dir, exist_ok=True)
            self.df.to_csv(results_path, index=False)
            self.df = None


class IntersectionStats(EnvironmentCallback):

    def __init__(self, results_dir: str):
        super(IntersectionStats, self).__init__()
        self.results_dir = results_dir
        self.df = None

    def on_step_end(self, environment):
        intersections = traci.trafficlight.getIDList()
        records = defaultdict(lambda: [])
        for intersection in intersections:
            records["scenario"].append(environment.scenario)
            records["intersection"].append(intersection)
            records["queue_length"].append(environment.problem_formulation.get_intersection_queue_length(intersection))
            records["pressure"].append(environment.problem_formulation.get_intersection_pressure(intersection))
            records["normalized_pressure"].append(
                environment.problem_formulation.get_intersection_normalized_pressure(intersection))
            records["waiting_time"] = environment.problem_formulation.get_intersection_waiting_time(intersection)
            print(environment.problem_formulation.get_intersection_waiting_time(intersection))
