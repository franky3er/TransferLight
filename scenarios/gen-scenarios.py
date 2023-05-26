import os
from pathlib import Path
from typing import List

from tqdm import tqdm

from src.params import SCENARIOS_ROOT, PYTHON, RANDOM_TRIPS_SCRIPT
from src.sumo.netgenerate import NetGenerate


netgenerate = NetGenerate()


def get_gen_commands() -> List[str]:
    commands = []
    commands += get_gen_grid_commands()
    return commands


def get_gen_grid_commands() -> List[str]:
    commands = []
    # Train 5x5-150m
    for i in range(100):
        out_dir = os.path.join(SCENARIOS_ROOT, "grid", "3x3-150m", "train", f"{i}-scenario")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        net_xml_path = os.path.join(out_dir, "network.net.xml")
        trips_xml_path = os.path.join(out_dir, "trips.trips.xml")
        rou_xml_path = os.path.join(out_dir, "routes.rou.xml")
        c = [
            "-g", "--grid.number", str(3), "--grid.length", str(150), "--grid.attach-length", str(150),
            "-j", "traffic_light", "--tls.discard-simple", "-L", str(2), "--no-turnarounds", "--turn-lanes", str(1),
            "--turn-lanes.length", str(70), "-o", net_xml_path
        ]
        path = netgenerate(*c)
        #commands.append(" ".join([
        #    "netgenerate", "-g", "--grid.number", str(3), "--grid.length", str(150), "--grid.attach-length", str(150),
        #    "-j", "traffic_light", "--tls.discard-simple", "-L", str(2), "--no-turnarounds", "--turn-lanes", str(1),
        #    "--turn-lanes.length", str(70), "-o", net_xml_path
        #]))
        #commands.append(" ".join([
        #    PYTHON, RANDOM_TRIPS_SCRIPT, "-n", net_xml_path, "-o", trips_xml_path, "-r", rou_xml_path, "-b", str(0),
#       #     "-e", str(900), "--insertion-rate", str(12_500), "--random-depart", "--vehicle-class", "passenger",
        #    "-e", str(900), "--insertion-rate", str(6_000), "--random-depart", "--vehicle-class", "passenger",#

        #    "--random", "--seed", str(i)
        #]))


def get_gen_rand_command():
    commands = []
    for i in range(1000):
        out_dir = os.path.join(SCENARIOS_ROOT, "rand", "rand-15", "train", f"{i}-scenario")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        net_xml_path = os.path.join(out_dir, "network.net.xml")
        trips_xml_path = os.path.join(out_dir, "trips.trips.xml")
        rou_xml_path = os.path.join(out_dir, "routes.rou.xml")
        commands.append(" ".join([
            "netgenerate", "-r", "--rand.iterations", str(15), "-j", "traffic_light",
            "--tls.discard-simple", "-L", str(4), "--random-lanenumber", "--no-turnarounds","-o", net_xml_path,
            "--seed", str(i)
        ]))
        commands.append(" ".join([
            PYTHON, RANDOM_TRIPS_SCRIPT, "-n", net_xml_path, "-o", trips_xml_path, "-r", rou_xml_path, "-b", str(0),
            "-e", str(900), "--insertion-rate", str(12_000), "--random-depart", "--vehicle-class", "passenger",
            "--random", "--seed", str(i)
        ]))

    return commands


if __name__ == "__main__":
    gen_commands = get_gen_commands()
    for command in tqdm(gen_commands):
        os.system(command)
