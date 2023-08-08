import os
import sys
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
SCENARIOS_ROOT = os.path.join(PROJECT_ROOT, "scenarios")
TRAIN_SCENARIOS_ROOT = os.path.join(SCENARIOS_ROOT, "train")
DEMO_SCENARIOS_ROOT = os.path.join(SCENARIOS_ROOT, "demo")
TEST_SCENARIOS_ROOT = os.path.join(SCENARIOS_ROOT, "test")
TUNE_SCENARIOS_ROOT = os.path.join(SCENARIOS_ROOT, "tune")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if "SUMO_HOME" not in os.environ:
    print("For SUMO scripts and demo purposes, please declare environment variable \"SUMO_HOME\"")
else:
    SUMO_HOME = os.environ["SUMO_HOME"]
    SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")
    sys.path.append(SUMO_TOOLS)
    RANDOM_TRIPS_SCRIPT = os.path.join(SUMO_TOOLS, "randomTrips.py")
PYTHON = "python3"

ACTION_TIME = 10
YELLOW_CHANGE_TIME = 3
ALL_RED_TIME = 2

N_TRAIN_EPISODES = 50
N_WORKERS = 64
