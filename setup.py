import os
import random
from typing import List, Tuple

import networkx as nx
import numpy as np
from scipy.stats import beta
import sumolib
from tqdm import tqdm

from src.sumo.utils import netgenerate, randomTrips
from src.params import TRAIN_SCENARIOS_ROOT, DEMO_SCENARIOS_ROOT


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
N_DEMO_SCENARIOS = 1


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


def create_weight_files(net_xml_path: str) -> str:
    parent_dir = os.path.dirname(net_xml_path)
    prefix = os.path.join(parent_dir, "edge-weights")
    net = sumolib.net.readNet(net_xml_path)
    graph = to_graph(net)
    src_node, dst_node = get_random_nodes(graph, min_hops=3)
    src_proba, dst_proba = np.random.uniform(low=0.0, high=1.0, size=2)
    other_src_nodes, other_dst_nodes = list(graph.nodes - {src_node}), list(graph.nodes - {dst_node})
    other_src_probas = [(1 - src_proba) / len(other_src_nodes) for _ in range(len(other_src_nodes))]
    other_dst_probas = [(1 - dst_proba) / len(other_dst_nodes) for _ in range(len(other_dst_nodes))]
    src_node_probas = [(src_node, src_proba)] + [(n, p) for n, p in zip(other_src_nodes, other_src_probas)]
    dst_node_probas = [(dst_node, dst_proba)] + [(n, p) for n, p in zip(other_dst_nodes, other_dst_probas)]
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


def get_scenario_name(total: int, index: int) -> str:
    return "".join(["0" for _ in range(len(str(total)) - len(str(index)))]) + str(index)


def get_scenario_paths(dir_scenario: str) -> Tuple[str, str, str]:
    net_xml_path = os.path.join(dir_scenario, "network.net.xml")
    trips_xml_path = os.path.join(dir_scenario, "trips.trips.xml")
    rou_xml_path = os.path.join(dir_scenario, "routes.rou.xml")
    return net_xml_path, trips_xml_path, rou_xml_path


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


def fixed_all_scenarios():
    name_environment = "fixed-all"
    for i in tqdm(range(N_TRAIN_SCENARIOS + N_DEMO_SCENARIOS)):
        if i < N_TRAIN_SCENARIOS:
            name_scenario = get_scenario_name(N_TRAIN_SCENARIOS, i)
            dir_scenario = os.path.join(TRAIN_SCENARIOS_ROOT, name_environment, name_scenario)
        else:
            name_scenario = get_scenario_name(N_DEMO_SCENARIOS, i - N_TRAIN_SCENARIOS + 1)
            dir_scenario = os.path.join(DEMO_SCENARIOS_ROOT, name_environment, name_scenario)
        os.makedirs(dir_scenario, exist_ok=True)
        net_xml_path, trips_xml_path, rou_xml_path = get_scenario_paths(dir_scenario)
        netgenerate_cmd = create_rand_netgenerate_cmd(net_xml_path)
        netgenerate(*netgenerate_cmd)
        random_trips_cmd = create_random_trips_cmd(net_xml_path, trips_xml_path, rou_xml_path)
        randomTrips(*random_trips_cmd)


def random_topology_scenarios():
    name_environment = "random-topology"
    for i in tqdm(range(N_TRAIN_SCENARIOS + N_DEMO_SCENARIOS)):
        if i < N_TRAIN_SCENARIOS:
            name_scenario = get_scenario_name(N_TRAIN_SCENARIOS, i)
            dir_scenario = os.path.join(TRAIN_SCENARIOS_ROOT, name_environment, name_scenario)
        else:
            name_scenario = get_scenario_name(N_DEMO_SCENARIOS, i - N_TRAIN_SCENARIOS + 1)
            dir_scenario = os.path.join(DEMO_SCENARIOS_ROOT, name_environment, name_scenario)
        os.makedirs(dir_scenario, exist_ok=True)
        net_xml_path, trips_xml_path, rou_xml_path = get_scenario_paths(dir_scenario)
        netgenerate_cmd = create_rand_netgenerate_cmd(net_xml_path, seed=counter())
        netgenerate(*netgenerate_cmd)
        random_trips_cmd = create_random_trips_cmd(net_xml_path, trips_xml_path, rou_xml_path)
        randomTrips(*random_trips_cmd)


def random_rate_scenarios():
    name_environment = "random-rate"
    for i in tqdm(range(N_TRAIN_SCENARIOS + N_DEMO_SCENARIOS)):
        if i < N_TRAIN_SCENARIOS:
            name_scenario = get_scenario_name(N_TRAIN_SCENARIOS, i)
            dir_scenario = os.path.join(TRAIN_SCENARIOS_ROOT, name_environment, name_scenario)
        else:
            name_scenario = get_scenario_name(N_DEMO_SCENARIOS, i - N_TRAIN_SCENARIOS + 1)
            dir_scenario = os.path.join(DEMO_SCENARIOS_ROOT, name_environment, name_scenario)
        os.makedirs(dir_scenario, exist_ok=True)
        net_xml_path, trips_xml_path, rou_xml_path = get_scenario_paths(dir_scenario)
        netgenerate_cmd = create_rand_netgenerate_cmd(net_xml_path)
        netgenerate(*netgenerate_cmd)
        alpha = np.random.uniform(VEHICLE_DEPARTURE_ALPHA_MIN, VEHICLE_DEPARTURE_ALPHA_MAX)
        beta = np.random.uniform(VEHICLE_DEPARTURE_BETA_MIN, VEHICLE_DEPARTURE_BETA_MAX)
        random_trips_cmd = create_random_trips_cmd(net_xml_path, trips_xml_path, rou_xml_path,
                                                   vehicle_departure_alpha=alpha, vehicle_departure_beta=beta)
        randomTrips(*random_trips_cmd)


def random_location_scenarios():
    name_environment = "random-location"
    for i in tqdm(range(N_TRAIN_SCENARIOS + N_DEMO_SCENARIOS)):
        if i < N_TRAIN_SCENARIOS:
            name_scenario = get_scenario_name(N_TRAIN_SCENARIOS, i)
            dir_scenario = os.path.join(TRAIN_SCENARIOS_ROOT, name_environment, name_scenario)
        else:
            name_scenario = get_scenario_name(N_DEMO_SCENARIOS, i - N_TRAIN_SCENARIOS + 1)
            dir_scenario = os.path.join(DEMO_SCENARIOS_ROOT, name_environment, name_scenario)
        os.makedirs(dir_scenario, exist_ok=True)
        net_xml_path, trips_xml_path, rou_xml_path = get_scenario_paths(dir_scenario)
        netgenerate_cmd = create_rand_netgenerate_cmd(net_xml_path)
        netgenerate(*netgenerate_cmd)
        weights_prefix = create_weight_files(net_xml_path)
        random_trips_cmd = create_random_trips_cmd(net_xml_path, trips_xml_path, rou_xml_path,
                                                   weight_prefix=weights_prefix)
        randomTrips(*random_trips_cmd)


def random_all_scenarios():
    name_environment = "random-all"
    for i in tqdm(range(N_TRAIN_SCENARIOS + N_DEMO_SCENARIOS)):
        if i < N_TRAIN_SCENARIOS:
            name_scenario = get_scenario_name(N_TRAIN_SCENARIOS, i)
            dir_scenario = os.path.join(TRAIN_SCENARIOS_ROOT, name_environment, name_scenario)
        else:
            name_scenario = get_scenario_name(N_DEMO_SCENARIOS, i - N_TRAIN_SCENARIOS + 1)
            dir_scenario = os.path.join(DEMO_SCENARIOS_ROOT, name_environment, name_scenario)
        os.makedirs(dir_scenario, exist_ok=True)
        net_xml_path, trips_xml_path, rou_xml_path = get_scenario_paths(dir_scenario)
        netgenerate_cmd = create_rand_netgenerate_cmd(net_xml_path, seed=counter())
        netgenerate(*netgenerate_cmd)
        alpha = np.random.uniform(VEHICLE_DEPARTURE_ALPHA_MIN, VEHICLE_DEPARTURE_ALPHA_MAX)
        beta = np.random.uniform(VEHICLE_DEPARTURE_BETA_MIN, VEHICLE_DEPARTURE_BETA_MAX)
        weights_prefix = create_weight_files(net_xml_path)
        random_trips_cmd = create_random_trips_cmd(net_xml_path, trips_xml_path, rou_xml_path,
                                                   vehicle_departure_alpha=alpha, vehicle_departure_beta=beta,
                                                   weight_prefix=weights_prefix)
        randomTrips(*random_trips_cmd)


if __name__ == "__main__":
    fixed_all_scenarios()
    random_topology_scenarios()
    random_rate_scenarios()
    random_location_scenarios()
    random_all_scenarios()
