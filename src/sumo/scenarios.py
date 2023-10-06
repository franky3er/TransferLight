from abc import ABC, abstractmethod
import os
import sys
import random
from typing import List, Tuple

import networkx as nx
import numpy as np
from scipy.stats import beta
import sumolib
from tqdm import tqdm
import xml.etree.ElementTree as ET

from src.sumo.utils import netgenerate, randomTrips
from src.params import ScenarioSpec, YELLOW_CHANGE_TIME, ALL_RED_TIME, SATURATION_FLOW_RATE, STARTUP_TIME


RAND_ITERATIONS = 15
RAND_MIN_DISTANCE = 100
RAND_MAX_DISTANCE = 200
RAND_MIN_ANGLE = 45
RAND_NUM_TRIES = 100
MAX_NUM_LANES = 4
RANDOM_LANE_NUMBER = True
SEED = 42

VEHICLE_INSERTION_BEGIN = 0
VEHICLE_INSERTION_END = 1_800
VEHICLE_DEPARTURE_RATE = 1_800
VEHICLE_DEPARTURE_ALPHA = 1.0
VEHICLE_DEPARTURE_BETA = 1.0
VEHICLE_DEPARTURE_ALPHA_MIN = 1.0
VEHICLE_DEPARTURE_ALPHA_MAX = 5.0
VEHICLE_DEPARTURE_BETA_MIN = 1.0
VEHICLE_DEPARTURE_BETA_MAX = 5.0

N_TRAIN_SCENARIOS = 1_000
N_TEST_SCENARIOS = 10


class Counter:

    def __init__(self):
        self.cnt = 0

    def __call__(self) -> int:
        self.cnt += 1
        return self.cnt


counter = Counter()
linspace = np.linspace(0, 1, 100)
nl = "\n"
np.random.seed(SEED)
random.seed(SEED)


def create_rand_netgenerate_cmd(
        net_xml_path: str,
        rand_iterations: int = RAND_ITERATIONS,
        rand_num_tries: int = RAND_NUM_TRIES,
        rand_min_distance: int = RAND_MIN_DISTANCE,
        rand_max_distance: int = RAND_MAX_DISTANCE,
        rand_min_angle: int = RAND_MIN_ANGLE,
        max_num_lanes: int = MAX_NUM_LANES,
        random_lane_number: bool = RANDOM_LANE_NUMBER,
        seed: int = SEED
) -> List[str]:
    netgenerate_cmd = [
        "-r", "--rand.iterations", rand_iterations, "--rand.max-distance", rand_max_distance,
        "--rand.min-distance", rand_min_distance, "-j", "traffic_light", "--tls.discard-simple",
        "--rand.min-angle", rand_min_angle, "--rand.num-tries", rand_num_tries,
        "--rand.neighbor-dist5", 10, "--rand.neighbor-dist6", 0, "-L", max_num_lanes,
        "--random-lanenumber" if random_lane_number else "", "--no-turnarounds", "-o", net_xml_path,
        "--seed", seed
    ]
    return [str(arg) for arg in netgenerate_cmd]


def create_random_trips_cmd(
        net_xml_path: str,
        trips_xml_path: str,
        rou_xml_path: str,
        vehicle_insertion_begin: int = VEHICLE_INSERTION_BEGIN,
        vehicle_insertion_end: int = VEHICLE_INSERTION_END,
        vehicle_departure_rate: int = VEHICLE_DEPARTURE_RATE,
        vehicle_departure_alpha: float = VEHICLE_DEPARTURE_ALPHA,
        vehicle_departure_beta: float = VEHICLE_DEPARTURE_BETA,
        weight_prefix: str = None
) -> List[str]:
    departure_rates = beta.pdf(linspace, vehicle_departure_alpha, vehicle_departure_beta) \
        * vehicle_departure_rate + 1e-10
    random_trips_cmd = [
        "-n", net_xml_path, "-o", trips_xml_path, "-r", rou_xml_path, "-b", vehicle_insertion_begin,
        "-e", vehicle_insertion_end, "--random-depart", "--random", "-L",
        "-l", "--seed", counter()
    ]
    random_trips_cmd += ["--insertion-rate"] + departure_rates.tolist()
    if weight_prefix is not None:
        random_trips_cmd += ["--weights-prefix"] + [weight_prefix]
    return [str(arg) for arg in random_trips_cmd]


def create_weight_files(net_xml_path: str, random_weights: bool = False, src_edge: str = None, dst_edge: str = None,
                        begin: int = VEHICLE_INSERTION_BEGIN, end: int = VEHICLE_INSERTION_END) \
        -> str:
    parent_dir = os.path.dirname(net_xml_path)
    prefix = os.path.join(parent_dir, "edge-weights")
    net = sumolib.net.readNet(net_xml_path)
    graph = to_graph(net)
    if src_edge is None:
        src_node_probas = [(n, np.random.uniform(low=0.0, high=1.0, size=1)[0] if random_weights else 1.0)
                           for n in graph.nodes]
    else:
        src_node_probas = [(n, 0.0 if src_edge != n else sys.float_info.max) for n in graph.nodes]
    if dst_edge is None:
        dst_node_probas = [(n, np.random.uniform(low=0.0, high=1.0, size=1)[0] if random_weights else 1.0)
                           for n in graph.nodes]
    else:
        dst_node_probas = [(n, 0.0 if dst_edge != n else sys.float_info.max) for n in graph.nodes]
    trips_src_xml_path = f"{prefix}.src.xml"
    trips_dst_xml_path = f"{prefix}.dst.xml"
    trips_src_xml_content = f"""
    <edgedata>
        <interval begin="{begin}" end="{end}">
            {"".join([f'<edge id="{n}" value="{p}"/>{nl}' for n, p in src_node_probas])}
        </interval>
    </edgedata>
    """
    trips_dst_xml_content = f"""
    <edgedata>
        <interval begin="{begin}" end="{end}">
            {"".join([f'<edge id="{n}" value="{p}"/>{nl}' for n, p in dst_node_probas])}
        </interval>
    </edgedata>
    """
    with open(trips_src_xml_path, "w") as f:
        f.write(trips_src_xml_content)
    with open(trips_dst_xml_path, "w") as f:
        f.write(trips_dst_xml_content)
    return prefix


def create_tls_cycle_adaptation_cmd(
        net_xml_path: str,
        rou_xml_path: str,
        tls_add_xml_path: str,
        yellow_change_time: int = YELLOW_CHANGE_TIME,
        all_red_time: int = ALL_RED_TIME
):
    cmd = ["-n", net_xml_path, "-r", rou_xml_path, "-o", tls_add_xml_path,
           "-b", 900, "-y", yellow_change_time, "-a", all_red_time]
    return [str(arg) for arg in cmd]


def create_tls_coordinator_cmd(
        net_xml_path: str,
        rou_xml_path: str,
        tls_add_xml_path: str,
):
    cmd = ["-n", net_xml_path, "-r", rou_xml_path, "-a", tls_add_xml_path, "-o", tls_add_xml_path]
    return [str(arg) for arg in cmd]


def create_sumocfg_file(sumocfg_path: str, net_xml: str, rou_xml: str, add_xml: str = None):
    content = f"""
        <configuration>
            <input>
    """
    content += f"""
                <net-file value="{net_xml}"/>
                <route-files value="{rou_xml}"/>
    """
    if add_xml is not None:
        content += f"""
                <additional-files value="{add_xml}"/>
        """
    content += """
            </input>
        </configuration>
    """
    with open(sumocfg_path, "w") as f:
        f.write(content)


def get_scenario_name(total: int, index: int) -> str:
    return "".join(["0" for _ in range(len(str(total)) - len(str(index)))]) + str(index)


def get_scenario_paths(dir_scenario: str, scenario_name: str) -> Tuple[str, str, str, str]:
    net_xml_path = os.path.join(dir_scenario, f"{scenario_name}.net.xml")
    trips_xml_path = os.path.join(dir_scenario, f"{scenario_name}.trips.xml")
    rou_xml_path = os.path.join(dir_scenario, f"{scenario_name}.rou.xml")
    sumocfg_path = os.path.join(dir_scenario, f"{scenario_name}.sumocfg")
    return net_xml_path, trips_xml_path, rou_xml_path, sumocfg_path


def to_graph(net: sumolib.net.Net) -> nx.Graph:
    intersections = [intersection_id.getID() for intersection_id in net.getNodes()]
    nodes, edges = set(), set()
    for intersection_id in intersections:
        connections = net.getNode(intersection_id).getConnections()
        for connection in connections:
            incoming_lane, outgoing_lane = connection.getFromLane(), connection.getToLane()
            incoming_approach, outgoing_approach = incoming_lane.getEdge(), outgoing_lane.getEdge()
            incoming_approach_id, outgoing_approach_id = incoming_approach.getID(), outgoing_approach.getID()
            nodes.add(incoming_approach_id)
            nodes.add(outgoing_approach_id)
            edges.add((incoming_approach_id, outgoing_approach_id))
    graph = nx.Graph()
    graph.add_nodes_from(list(nodes))
    graph.add_edges_from(list(edges))
    return graph


class ScenariosGenerator(ABC):

    @classmethod
    def create(cls, scenario_spec: ScenarioSpec):
        class_name = scenario_spec.generator
        obj = getattr(sys.modules[__name__], class_name)(scenario_spec)
        assert isinstance(obj, ScenariosGenerator)
        return obj

    def __init__(self, scenario_spec: ScenarioSpec):
        self.scenario_spec = scenario_spec
        self.name = scenario_spec.name
        self.train_dir = scenario_spec.train_dir
        self.test_dir = scenario_spec.test_dir
        self._init(**scenario_spec.generator_args)

    @abstractmethod
    def _init(self, **kwargs):
        pass

    @abstractmethod
    def generate_scenarios(self):
        pass


class TransferLightScenariosGenerator(ScenariosGenerator):

    def _init(self, random_network: bool = False, random_rate: bool = False, random_location: bool = False):
        self.random_network = random_network
        self.random_rate = random_rate
        self.random_location = random_location

    def generate_scenarios(self):
        for i in tqdm(range(N_TRAIN_SCENARIOS + N_TEST_SCENARIOS)):
            if i < N_TRAIN_SCENARIOS:
                name_scenario = get_scenario_name(N_TRAIN_SCENARIOS, i)
                dir_scenario = self.train_dir
            else:
                name_scenario = get_scenario_name(N_TEST_SCENARIOS, i - N_TRAIN_SCENARIOS + 1)
                dir_scenario = self.test_dir
            os.makedirs(dir_scenario, exist_ok=True)

            net_xml_path, trips_xml_path, rou_xml_path, sumocfg_path = get_scenario_paths(dir_scenario, name_scenario)
            netgenerate_cmd = create_rand_netgenerate_cmd(net_xml_path, seed=counter() if self.random_network else SEED)
            netgenerate(*netgenerate_cmd)

            alpha = np.random.uniform(VEHICLE_DEPARTURE_ALPHA_MIN, VEHICLE_DEPARTURE_ALPHA_MAX) \
                if self.random_rate else VEHICLE_DEPARTURE_ALPHA
            beta = np.random.uniform(VEHICLE_DEPARTURE_BETA_MIN, VEHICLE_DEPARTURE_BETA_MAX) \
                if self.random_rate else VEHICLE_DEPARTURE_BETA

            weights_prefix = create_weight_files(net_xml_path, random_weights=self.random_location)

            random_trips_cmd = create_random_trips_cmd(net_xml_path, trips_xml_path, rou_xml_path,
                                                       vehicle_departure_alpha=alpha, vehicle_departure_beta=beta,
                                                       weight_prefix=weights_prefix)
            randomTrips(*random_trips_cmd)

            net_xml_name, rou_xml_name = f"{name_scenario}.net.xml", f"{name_scenario}.rou.xml"
            create_sumocfg_file(sumocfg_path, net_xml_name, rou_xml_name)


class ArterialScenariosGenerator(ScenariosGenerator):

    def _init(
            self,
            n_intersections: int = 6,
            lane_length: float = 300.0,
            allowed_speed: float = 10.0,
            arterial_flow_rate: float = VEHICLE_DEPARTURE_RATE,
            side_street_flow_rate: float = VEHICLE_DEPARTURE_RATE,
            saturation_flow_rate: float = SATURATION_FLOW_RATE,
            yellow_change_time: float = YELLOW_CHANGE_TIME,
            all_red_time: float = ALL_RED_TIME,
            startup_time: float = STARTUP_TIME
    ):
        self.n_intersections = n_intersections
        self.lane_length = lane_length
        self.allowed_speed = allowed_speed
        self.arterial_demand, self.side_street_demand = arterial_flow_rate, side_street_flow_rate
        l = startup_time
        R = 2 * (yellow_change_time + all_red_time)
        L = 2 * l + R
        y_arterial = arterial_flow_rate / saturation_flow_rate
        y_side = side_street_flow_rate / saturation_flow_rate
        Y = y_arterial + y_side
        c_0 = (1.5 * L + 5) / (1 - Y)
        self.all_red_time, self.yellow_change_time = all_red_time, yellow_change_time
        self.green_time_arterial = (y_arterial / Y) * (c_0 - L) + l
        self.green_time_side_street = (y_side / Y) * (c_0 - L) + l
        self.green_wave_offset = lane_length / allowed_speed

    def generate_scenarios(self):
        n_train_scenarios = 100
        for i in tqdm(range(n_train_scenarios + N_TEST_SCENARIOS)):
            if i < n_train_scenarios:
                self.name_scenario = get_scenario_name(n_train_scenarios, i)
                self.dir_scenario = self.train_dir
            else:
                self.name_scenario = get_scenario_name(N_TEST_SCENARIOS, i - n_train_scenarios + 1)
                self.dir_scenario = self.test_dir
            os.makedirs(self.dir_scenario, exist_ok=True)

            self._generate_network()
            self._generate_traffic()
            self._generate_tls_settings()
            self._generate_sumocfg()

    def _generate_network(self):
        self.net_xml_name = f"{self.name_scenario}.net.xml"
        self.net_xml_path = os.path.join(self.dir_scenario, self.net_xml_name)
        netgenerate_cmd = ["-g", "--grid.length", self.lane_length, "--grid.attach-length", self.lane_length,
                           "--grid.x-number", 1, "--grid.y-number", self.n_intersections, "--no-turnarounds",
                           "--no-left-connections", "-j", "traffic_light", "-S", self.allowed_speed,
                           "--tls.allred.time", ALL_RED_TIME, "--tls.yellow.time", YELLOW_CHANGE_TIME,
                           "-o", self.net_xml_path]
        netgenerate_cmd = [str(arg) for arg in netgenerate_cmd]
        netgenerate(*netgenerate_cmd)

    def _generate_traffic(self):
        trips, self.rou_xml_paths, self.rou_xml_names = [], [], []
        trips.append(("bottom0A0", f"A{self.n_intersections-1}top0", self.arterial_demand))
        for i in range(self.n_intersections):
            trips.append((f"left{i}A{i}", f"A{i}right{i}", self.side_street_demand))
        for i, (src_edge, dst_edge, demand) in enumerate(trips):
            trips_xml_path = os.path.join(self.dir_scenario, f"{self.name_scenario}.trips.xml")
            rou_xml_path = os.path.join(self.dir_scenario, f"{self.name_scenario}.{i}.rou.xml")
            self.rou_xml_paths.append(rou_xml_path)
            self.rou_xml_names.append(f"{self.name_scenario}.{i}.rou.xml")
            weight_prefix = create_weight_files(self.net_xml_path, src_edge=src_edge, dst_edge=dst_edge)
            random_trips_cmd = create_random_trips_cmd(self.net_xml_path, trips_xml_path, rou_xml_path,
                                                       vehicle_departure_rate=demand,
                                                       weight_prefix=weight_prefix)
            randomTrips(*random_trips_cmd)
        id_cnt = Counter()
        for rou_xml_path in self.rou_xml_paths:
            tree = ET.parse(rou_xml_path)
            root = tree.getroot()
            [vehicle.set("id", str(id_cnt())) for vehicle in root.iter("vehicle")]
            tree.write(rou_xml_path)

    def _generate_tls_settings(self):
        self.tls_add_xml_name = f"{self.name_scenario}.tls.add.xml"
        self.tls_add_xml_path = os.path.join(self.dir_scenario, self.tls_add_xml_name)
        additional_content = []
        for i in range(self.n_intersections):
            tl_logic = f"""
                <tlLogic id="A{i}" type="static" programID="a" offset="{i*self.green_wave_offset}">
                    <phase duration="{self.green_time_arterial}" state="GGrrGGrr"/>
                    <phase duration="{self.yellow_change_time}" state="yyrryyrr"/>
                    <phase duration="{self.all_red_time}" state="rrrrrrrr"/>
                    <phase duration="{self.green_time_side_street}" state="rrGGrrGG"/>
                    <phase duration="{self.yellow_change_time}" state="rryyrryy"/>
                    <phase duration="{self.all_red_time}" state="rrrrrrrr"/>
                </tlLogic>
            """
            additional_content.append(tl_logic)
        "\n".join(additional_content)
        content = f"""
        <additional>
        {additional_content}
        </additional>
        """
        with open(self.tls_add_xml_path, "w") as f:
            f.write(content)

    def _generate_sumocfg(self):
        sumocfg_path = os.path.join(self.dir_scenario, f"{self.name_scenario}.sumocfg")
        create_sumocfg_file(sumocfg_path, net_xml=self.net_xml_name, rou_xml=",".join(self.rou_xml_names),
                            add_xml=self.tls_add_xml_name)
