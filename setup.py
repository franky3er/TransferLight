import os

import numpy as np
import sumolib
from tqdm import tqdm

from src.sumo.utils import netgenerate, netconvert, randomTrips
from src.params import SCENARIOS_ROOT


N_TRAIN_SCENARIOS = 1_000
N_TEST_SCENARIOS = 1_000
N_DEMO_SCENARIOS = 1


def gen_isolated_intersection_scenarios():
    dir_scenarios = os.path.join(SCENARIOS_ROOT, "isolated")
    dir_train_scenarios = os.path.join(dir_scenarios, "train")

    # Train Scenarios
    for i in tqdm(range(N_TRAIN_SCENARIOS)):
        name_scenario = "".join(["0" for _ in range(len(str(N_TRAIN_SCENARIOS)) - len(str(i)))]) + str(i)
        dir_scenario = os.path.join(dir_train_scenarios, name_scenario)
        os.makedirs(dir_scenario, exist_ok=True)
        net_xml_path = os.path.join(dir_scenario, "network.net.xml")
        trips_xml_path = os.path.join(dir_scenario, "trips.trips.xml")
        rou_xml_path = os.path.join(dir_scenario, "routes.rou.xml")
        netgenerate_cmd = [
            "-r", "--rand.iterations", 15, "--rand.max-distance", 200, "--rand.min-distance", 100,
            "-j", "traffic_light", "--tls.discard-simple", "--rand.min-angle", 45, "--rand.num-tries", 10000,
            "--rand.neighbor-dist1", 0, "--rand.neighbor-dist2", 0,
            "--rand.neighbor-dist3", 10, "--rand.neighbor-dist4", 10,  "--rand.neighbor-dist5", 10,
            "--rand.neighbor-dist6", 10, "--bidi-probability", 0.9, "-L", 4, "--random-lanenumber",
            "--no-turnarounds", "-o", net_xml_path, "--seed", i
        ]
        netgenerate(*netgenerate_cmd)
        net = sumolib.net.readNet(net_xml_path)
        intersections = [intersection for intersection in net.getNodes()
                         if len(intersection.getIncoming()) >= 3 and len(intersection.getOutgoing()) >= 3]
        rand_intersection = np.random.choice(intersections)
        rand_intersection_approaches = rand_intersection.getIncoming() + rand_intersection.getOutgoing()
        rand_intersection_approaches = [approach.getID() for approach in rand_intersection_approaches]
        netconvert_cmd = [
            "-s", net_xml_path, "-o", net_xml_path, "--no-turnarounds", "--tls.discard-simple",
            "--keep-edges.explicit", ",".join(rand_intersection_approaches)
        ]
        netconvert(*netconvert_cmd)
        random_trips_cmd = [
            "-n", net_xml_path, "-o", trips_xml_path, "-r", rou_xml_path, "-b", str(0),
            "-e", str(900), "--insertion-rate", str(3_000), "--random-depart", "--vehicle-class", "passenger",
            "--random", "-L", "-l", "--seed", str(i)
        ]
        randomTrips(*random_trips_cmd)


if __name__ == "__main__":
    gen_isolated_intersection_scenarios()
