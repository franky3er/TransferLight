import os
import sys
import random
from typing import List, Tuple

import networkx as nx
import numpy as np
from scipy.stats import beta
import sumolib
from tqdm import tqdm

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from src.sumo.utils import netgenerate, randomTrips
from src.params import ScenarioSpec, scenario_specs


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
        "-e", vehicle_insertion_end, "--random-depart", "--vehicle-class", "passenger", "--random", "-L",
        "-l", "--seed", counter()
    ]
    random_trips_cmd += ["--insertion-rate"] + departure_rates.tolist()
    if weight_prefix is not None:
        random_trips_cmd += ["--weights-prefix"] + [weight_prefix]
    return [str(arg) for arg in random_trips_cmd]


def create_weight_files(net_xml_path: str, random_weights: bool = False) -> str:
    parent_dir = os.path.dirname(net_xml_path)
    prefix = os.path.join(parent_dir, "edge-weights")
    net = sumolib.net.readNet(net_xml_path)
    graph = to_graph(net)
    src_node_probas = [(n, np.random.uniform(low=0.0, high=1.0, size=1)[0] if random_weights else 1.0)
                       for n in graph.nodes]
    dst_node_probas = [(n, np.random.uniform(low=0.0, high=1.0, size=1)[0] if random_weights else 1.0)
                       for n in graph.nodes]
    trips_src_xml_path = f"{prefix}.src.xml"
    trips_dst_xml_path = f"{prefix}.dst.xml"
    trips_src_xml_content = f"""
    <edgedata>
        <interval begin="{VEHICLE_INSERTION_BEGIN}" end="{VEHICLE_INSERTION_END}">
            {"".join([f'<edge id="{n}" value="{p}"/>{nl}' for n, p in src_node_probas])}
        </interval>
    </edgedata>
    """
    trips_dst_xml_content = f"""
    <edgedata>
        <interval begin="{VEHICLE_INSERTION_BEGIN}" end="{VEHICLE_INSERTION_END}">
            {"".join([f'<edge id="{n}" value="{p}"/>{nl}' for n, p in dst_node_probas])}
        </interval>
    </edgedata>
    """
    with open(trips_src_xml_path, "w") as f:
        f.write(trips_src_xml_content)
    with open(trips_dst_xml_path, "w") as f:
        f.write(trips_dst_xml_content)
    return prefix


def create_sumocfg_file(sumocfg_path: str, net_xml_name: str, rou_xml_name: str):
    content = f"""
    <configuration>
        <input>
            <net-file value="{net_xml_name}"/>
            <route-files value="{rou_xml_name}"/>
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


def get_random_nodes(graph: nx.Graph, min_hops: int = 4) -> Tuple[str, str]:
    for _ in range(100):
        src_node, dst_node = random.sample(list(graph.nodes), 2)
        hops = nx.shortest_path_length(graph, src_node, dst_node)
        if hops >= min_hops:
            return src_node, dst_node
    raise Exception(f"No nodes found that are {min_hops} hops apart.")


def generate_scenarios(scenario_spec: ScenarioSpec):
    for i in tqdm(range(N_TRAIN_SCENARIOS + N_TEST_SCENARIOS)):
        if i < N_TRAIN_SCENARIOS:
            name_scenario = get_scenario_name(N_TRAIN_SCENARIOS, i)
            dir_scenario = scenario_spec.train_dir
        else:
            name_scenario = get_scenario_name(N_TEST_SCENARIOS, i - N_TRAIN_SCENARIOS + 1)
            dir_scenario = scenario_spec.test_dir
        os.makedirs(dir_scenario, exist_ok=True)

        net_xml_path, trips_xml_path, rou_xml_path, sumocfg_path = get_scenario_paths(dir_scenario, name_scenario)
        netgenerate_cmd = create_rand_netgenerate_cmd(net_xml_path,
                                                      seed=counter() if scenario_spec.random_network else SEED)
        netgenerate(*netgenerate_cmd)

        alpha = np.random.uniform(VEHICLE_DEPARTURE_ALPHA_MIN, VEHICLE_DEPARTURE_ALPHA_MAX) \
            if scenario_spec.random_rate else VEHICLE_DEPARTURE_ALPHA
        beta = np.random.uniform(VEHICLE_DEPARTURE_BETA_MIN, VEHICLE_DEPARTURE_BETA_MAX) \
            if scenario_spec.random_rate else VEHICLE_DEPARTURE_BETA

        weights_prefix = create_weight_files(net_xml_path, random_weights=scenario_spec.random_location)

        random_trips_cmd = create_random_trips_cmd(net_xml_path, trips_xml_path, rou_xml_path,
                                                   vehicle_departure_alpha=alpha, vehicle_departure_beta=beta,
                                                   weight_prefix=weights_prefix)
        randomTrips(*random_trips_cmd)

        net_xml_name, rou_xml_name = f"{name_scenario}.net.xml", f"{name_scenario}.rou.xml"
        create_sumocfg_file(sumocfg_path, net_xml_name, rou_xml_name)


if __name__ == "__main__":
    for scenario_spec in scenario_specs.values():
        print(f"Generate {scenario_spec.name} scenarios")
        generate_scenarios(scenario_spec)
