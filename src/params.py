import os
import sys

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
SCENARIOS_ROOT = os.path.join(PROJECT_ROOT, "scenarios")
TMP_DIR = os.path.join(PROJECT_ROOT, "tmp")

if not "SUMO_HOME" in os.environ:
    sys.exit("Please declare environment variable \"SUMO_HOME\"")
SUMO_HOME = os.environ["SUMO_HOME"]
SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")
sys.path.append(SUMO_TOOLS)
RANDOM_TRIPS_SCRIPT = os.path.join(SUMO_TOOLS, "randomTrips.py")
PYTHON = "python3"

ENV_ACTION_EXECUTION_TIME = 5
ENV_YELLOW_TIME = 3
ENV_RED_TIME = 1
