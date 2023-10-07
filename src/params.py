from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
import os
import sys
from typing import Optional, Dict

import torch


class ConfigEnum(Enum):
    def __get__(self, instance, owner):
        return self.value


PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
SCENARIOS_ROOT = os.path.join(PROJECT_ROOT, "scenarios")
TRAIN_SCENARIOS_ROOT = os.path.join(SCENARIOS_ROOT, "train")
DEMO_SCENARIOS_ROOT = os.path.join(SCENARIOS_ROOT, "demo")
TEST_SCENARIOS_ROOT = os.path.join(SCENARIOS_ROOT, "test")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
SCRIPTS_ROOT = os.path.join(PROJECT_ROOT, "scripts")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SUMO_HOME = None
SUMO_TOOLS = None
RANDOM_TRIPS_SCRIPT = None
TLS_CYCLE_ADAPTATION_SCRIPT = None
TLS_COORDINATOR_SCRIPT = None
if "SUMO_HOME" not in os.environ:
    print("For SUMO scripts and demo purposes, please declare environment variable \"SUMO_HOME\"")
else:
    SUMO_HOME = os.environ["SUMO_HOME"]
    SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")
    sys.path.append(SUMO_TOOLS)
    RANDOM_TRIPS_SCRIPT = os.path.join(SUMO_TOOLS, "randomTrips.py")
    TLS_CYCLE_ADAPTATION_SCRIPT = os.path.join(SUMO_TOOLS, "tlsCycleAdaptation.py")
    TLS_COORDINATOR_SCRIPT = os.path.join(SUMO_TOOLS, "tlsCoordinator.py")
PYTHON = "python3"

MAX_PATIENCE = 600
ACTION_TIME = 10
N_WORKERS = 64

YELLOW_CHANGE_TIME = 3
ALL_RED_TIME = 2
SATURATION_FLOW_RATE = 1_700
STARTUP_TIME = 2

TRAIN_STEPS = 2_000
TRAIN_SKIP_STEPS = 100

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


class ScenarioNames(ConfigEnum):
    FIXED_ALL = "fixed-all"
    FIXED_NETWORK = "fixed-network"
    FIXED_LOCATION = "fixed-location"
    FIXED_RATE = "fixed-rate"
    RANDOM_ALL = "random-all"
    RANDOM_NETWORK = "random-network"
    RANDOM_LOCATION = "random-location"
    RANDOM_RATE = "random-rate"
    ARTERIAL_LIGHT = "arterial-light"
    ARTERIAL_HEAVY = "arterial-heavy"


class TrainScenariosDirs(ConfigEnum):
    FIXED_ALL = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.FIXED_ALL)
    FIXED_NETWORK = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.FIXED_NETWORK)
    FIXED_LOCATION = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.FIXED_LOCATION)
    FIXED_RATE = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.FIXED_RATE)
    RANDOM_ALL = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.RANDOM_ALL)
    RANDOM_NETWORK = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.RANDOM_NETWORK)
    RANDOM_LOCATION = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.RANDOM_LOCATION)
    RANDOM_RATE = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.RANDOM_RATE)
    ARTERIAL_LIGHT = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.ARTERIAL_LIGHT)
    ARTERIAL_HEAVY = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.ARTERIAL_HEAVY)


class TestScenarioDirs(ConfigEnum):
    FIXED_ALL = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.FIXED_ALL)
    FIXED_NETWORK = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.FIXED_NETWORK)
    FIXED_LOCATION = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.FIXED_LOCATION)
    FIXED_RATE = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.FIXED_RATE)
    RANDOM_ALL = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.RANDOM_ALL)
    RANDOM_NETWORK = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.RANDOM_NETWORK)
    RANDOM_LOCATION = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.RANDOM_LOCATION)
    RANDOM_RATE = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.RANDOM_RATE)
    ARTERIAL_LIGHT = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.ARTERIAL_LIGHT)
    ARTERIAL_HEAVY = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.ARTERIAL_HEAVY)


@dataclass
class ScenarioSpec:
    name: str
    generator: str
    generator_args: Dict
    test_dir: str
    test_max_time: int = VEHICLE_INSERTION_END
    test_max_patience: int = sys.maxsize
    train_dir: Optional[str] = None
    train_max_time: int = sys.maxsize
    train_max_patience: int = MAX_PATIENCE


scenario_specs = {
    ScenarioNames.FIXED_ALL: ScenarioSpec(
        name=ScenarioNames.FIXED_ALL,
        train_dir=TrainScenariosDirs.FIXED_ALL,
        test_dir=TestScenarioDirs.FIXED_ALL,
        generator="TransferLightScenariosGenerator",
        generator_args={"random_network": False, "random_rate": False, "random_location": False}
    ),
    ScenarioNames.RANDOM_NETWORK: ScenarioSpec(
        name=ScenarioNames.RANDOM_NETWORK,
        train_dir=TrainScenariosDirs.RANDOM_NETWORK,
        test_dir=TestScenarioDirs.RANDOM_NETWORK,
        generator="TransferLightScenariosGenerator",
        generator_args={"random_network": True, "random_rate": False, "random_location": False}
    ),
    ScenarioNames.RANDOM_LOCATION: ScenarioSpec(
        name=ScenarioNames.RANDOM_LOCATION,
        train_dir=TrainScenariosDirs.RANDOM_LOCATION,
        test_dir=TestScenarioDirs.RANDOM_LOCATION,
        generator="TransferLightScenariosGenerator",
        generator_args={"random_network": False, "random_rate": False, "random_location": True}
    ),
    ScenarioNames.RANDOM_RATE: ScenarioSpec(
        name=ScenarioNames.RANDOM_RATE,
        train_dir=TrainScenariosDirs.RANDOM_RATE,
        test_dir=TestScenarioDirs.RANDOM_RATE,
        generator="TransferLightScenariosGenerator",
        generator_args={"random_network": False, "random_rate": True, "random_location": False}
    ),
    ScenarioNames.FIXED_NETWORK: ScenarioSpec(
        name=ScenarioNames.FIXED_NETWORK,
        train_dir=TrainScenariosDirs.FIXED_NETWORK,
        test_dir=TestScenarioDirs.FIXED_NETWORK,
        generator="TransferLightScenariosGenerator",
        generator_args={"random_network": False, "random_rate": True, "random_location": True}
    ),
    ScenarioNames.FIXED_LOCATION: ScenarioSpec(
        name=ScenarioNames.FIXED_LOCATION,
        train_dir=TrainScenariosDirs.FIXED_LOCATION,
        test_dir=TestScenarioDirs.FIXED_LOCATION,
        generator="TransferLightScenariosGenerator",
        generator_args={"random_network": True, "random_rate": True, "random_location": False}
    ),
    ScenarioNames.FIXED_RATE: ScenarioSpec(
        name=ScenarioNames.FIXED_RATE,
        train_dir=TrainScenariosDirs.FIXED_RATE,
        test_dir=TestScenarioDirs.FIXED_RATE,
        generator="TransferLightScenariosGenerator",
        generator_args={"random_network": True, "random_rate": False, "random_location": True}
    ),
    ScenarioNames.RANDOM_ALL: ScenarioSpec(
        name=ScenarioNames.RANDOM_ALL,
        train_dir=TrainScenariosDirs.RANDOM_ALL,
        test_dir=TestScenarioDirs.RANDOM_ALL,
        generator="TransferLightScenariosGenerator",
        generator_args={"random_network": True, "random_rate": True, "random_location": True}
    ),
    ScenarioNames.ARTERIAL_HEAVY: ScenarioSpec(
        name=ScenarioNames.ARTERIAL_HEAVY,
        train_dir=TrainScenariosDirs.ARTERIAL_HEAVY,
        test_dir=TestScenarioDirs.ARTERIAL_HEAVY,
        generator="ArterialScenariosGenerator",
        generator_args={"n_intersections": 5, "lane_length": 200.0, "allowed_speed": 13.89, "arterial_flow_rate": 700.0,
                        "side_street_flow_rate": 420.0}
    ),
    ScenarioNames.ARTERIAL_LIGHT: ScenarioSpec(
        name=ScenarioNames.ARTERIAL_LIGHT,
        train_dir=TrainScenariosDirs.ARTERIAL_LIGHT,
        test_dir=TestScenarioDirs.ARTERIAL_LIGHT,
        generator="ArterialScenariosGenerator",
        generator_args={"n_intersections": 5, "lane_length": 200.0, "allowed_speed": 13.89, "arterial_flow_rate": 300.0,
                        "side_street_flow_rate": 180.0}
    )
}


class AgentNames(ConfigEnum):
    TRANSFERLIGHT = "TransferLight"
    TRANSFERLIGHT_A2C = f"{TRANSFERLIGHT}-A2C"
    TRANSFERLIGHT_DQN = f"{TRANSFERLIGHT}-DQN"
    TRANSFERLIGHT_A2C_FIXED_ALL = f"{TRANSFERLIGHT_A2C}-{ScenarioNames.FIXED_ALL}"
    TRANSFERLIGHT_A2C_FIXED_NETWORK = f"{TRANSFERLIGHT_A2C}-{ScenarioNames.FIXED_NETWORK}"
    TRANSFERLIGHT_A2C_FIXED_LOCATION = f"{TRANSFERLIGHT_A2C}-{ScenarioNames.FIXED_LOCATION}"
    TRANSFERLIGHT_A2C_FIXED_RATE = f"{TRANSFERLIGHT_A2C}-{ScenarioNames.FIXED_RATE}"
    TRANSFERLIGHT_A2C_RANDOM_ALL = f"{TRANSFERLIGHT_A2C}-{ScenarioNames.RANDOM_ALL}"
    TRANSFERLIGHT_A2C_RANDOM_NETWORK = f"{TRANSFERLIGHT_A2C}-{ScenarioNames.RANDOM_NETWORK}"
    TRANSFERLIGHT_A2C_RANDOM_LOCATION = f"{TRANSFERLIGHT_A2C}-{ScenarioNames.RANDOM_LOCATION}"
    TRANSFERLIGHT_A2C_RANDOM_RATE = f"{TRANSFERLIGHT_A2C}-{ScenarioNames.RANDOM_RATE}"
    TRANSFERLIGHT_DQN_FIXED_ALL = f"{TRANSFERLIGHT_DQN}-{ScenarioNames.FIXED_ALL}"
    TRANSFERLIGHT_DQN_FIXED_NETWORK = f"{TRANSFERLIGHT_DQN}-{ScenarioNames.FIXED_NETWORK}"
    TRANSFERLIGHT_DQN_FIXED_LOCATION = f"{TRANSFERLIGHT_DQN}-{ScenarioNames.FIXED_LOCATION}"
    TRANSFERLIGHT_DQN_FIXED_RATE = f"{TRANSFERLIGHT_DQN}-{ScenarioNames.FIXED_RATE}"
    TRANSFERLIGHT_DQN_RANDOM_ALL = f"{TRANSFERLIGHT_DQN}-{ScenarioNames.RANDOM_ALL}"
    TRANSFERLIGHT_DQN_RANDOM_NETWORK = f"{TRANSFERLIGHT_DQN}-{ScenarioNames.RANDOM_NETWORK}"
    TRANSFERLIGHT_DQN_RANDOM_LOCATION = f"{TRANSFERLIGHT_DQN}-{ScenarioNames.RANDOM_LOCATION}"
    TRANSFERLIGHT_DQN_RANDOM_RATE = f"{TRANSFERLIGHT_DQN}-{ScenarioNames.RANDOM_RATE}"
    TRANSFERLIGHT_DQN_ARTERIAL_HEAVY = f"{TRANSFERLIGHT_DQN}-{ScenarioNames.ARTERIAL_HEAVY}"
    TRANSFERLIGHT_DQN_ARTERIAL_LIGHT = f"{TRANSFERLIGHT_DQN}-{ScenarioNames.ARTERIAL_LIGHT}"
    PRESSLIGHT = "PressLight"
    PRESSLIGHT_ARTERIAL_LIGHT = f"{PRESSLIGHT}-{ScenarioNames.ARTERIAL_LIGHT}"
    PRESSLIGHT_ARTERIAL_HEAVY = f"{PRESSLIGHT}-{ScenarioNames.ARTERIAL_HEAVY}"
    MAX_PRESSURE = "MaxPressure"
    FIXED_TIME = "FixedTime"
    RANDOM = "Random"


class AgentDirs(ConfigEnum):
    TRANSFERLIGHT_A2C_FIXED_ALL = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_A2C_FIXED_ALL)
    TRANSFERLIGHT_A2C_FIXED_NETWORK = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_A2C_FIXED_NETWORK)
    TRANSFERLIGHT_A2C_FIXED_LOCATION = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_A2C_FIXED_LOCATION)
    TRANSFERLIGHT_A2C_FIXED_RATE = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_A2C_FIXED_RATE)
    TRANSFERLIGHT_A2C_RANDOM_ALL = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_A2C_RANDOM_ALL)
    TRANSFERLIGHT_A2C_RANDOM_NETWORK = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_A2C_RANDOM_NETWORK)
    TRANSFERLIGHT_A2C_RANDOM_LOCATION = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_A2C_RANDOM_LOCATION)
    TRANSFERLIGHT_A2C_RANDOM_RATE = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_A2C_RANDOM_RATE)
    TRANSFERLIGHT_DQN_FIXED_ALL = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_DQN_FIXED_ALL)
    TRANSFERLIGHT_DQN_FIXED_NETWORK = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_DQN_FIXED_NETWORK)
    TRANSFERLIGHT_DQN_FIXED_LOCATION = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_DQN_FIXED_LOCATION)
    TRANSFERLIGHT_DQN_FIXED_RATE = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_DQN_FIXED_RATE)
    TRANSFERLIGHT_DQN_RANDOM_ALL = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_DQN_RANDOM_ALL)
    TRANSFERLIGHT_DQN_RANDOM_NETWORK = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_DQN_RANDOM_NETWORK)
    TRANSFERLIGHT_DQN_RANDOM_LOCATION = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_DQN_RANDOM_LOCATION)
    TRANSFERLIGHT_DQN_RANDOM_RATE = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_DQN_RANDOM_RATE)
    TRANSFERLIGHT_DQN_ARTERIAL_HEAVY = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_DQN_ARTERIAL_HEAVY)
    TRANSFERLIGHT_DQN_ARTERIAL_LIGHT = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_DQN_ARTERIAL_LIGHT)
    PRESSLIGHT_ARTERIAL_LIGHT = os.path.join(RESULTS_ROOT, AgentNames.PRESSLIGHT_ARTERIAL_LIGHT)
    PRESSLIGHT_ARTERIAL_HEAVY = os.path.join(RESULTS_ROOT, AgentNames.PRESSLIGHT_ARTERIAL_HEAVY)
    MAX_PRESSURE = os.path.join(RESULTS_ROOT, AgentNames.MAX_PRESSURE)
    FIXED_TIME = os.path.join(RESULTS_ROOT, AgentNames.FIXED_TIME)
    RANDOM = os.path.join(RESULTS_ROOT, AgentNames.RANDOM)


class TransferLightConfig(ConfigEnum):
    HIDDEN_DIM = 64
    N_ATTENTION_HEADS = 8
    DROPOUT_PROB = 0.1


class PressLightConfig(ConfigEnum):
    STATE_DIM = 18
    HIDDEN_DIM = 64
    N_ACTIONS = 2
    N_LAYERS = 3
    DROPOUT_PROB = 0.1


class NetworkConfig(ConfigEnum):
    TRANSFERLIGHT_DQN = {
        "class_name": "TransferLightNetwork",
        "init_args": {
            "network_type": "DQN",
            "hidden_dim": TransferLightConfig.HIDDEN_DIM,
            "n_attention_heads": TransferLightConfig.N_ATTENTION_HEADS,
            "dropout_prob": TransferLightConfig.DROPOUT_PROB
        }
    }
    TRANSFERLIGHT_A2C = {
        "class_name": "TransferLightNetwork",
        "init_args": {
            "network_type": "A2C",
            "hidden_dim": TransferLightConfig.HIDDEN_DIM,
            "n_attention_heads": TransferLightConfig.N_ATTENTION_HEADS,
            "dropout_prob": TransferLightConfig.DROPOUT_PROB
        }
    }
    PRESSLIGHT = {
        "class_name": "PressLightNetwork",
        "init_args": {
            "network_type": "DQN",
            "state_dim": PressLightConfig.STATE_DIM,
            "hidden_dim": PressLightConfig.HIDDEN_DIM,
            "n_actions": PressLightConfig.N_ACTIONS,
            "n_layers": PressLightConfig.N_LAYERS,
            "dropout_prob": PressLightConfig.DROPOUT_PROB
        }
    }


class AgentConfigs(ConfigEnum):
    TRANSFERLIGHT_DQN = {
        "class_name": "DQN",
        "init_args": {
            "network": NetworkConfig.TRANSFERLIGHT_DQN,
            "discount_factor": 0.9,
            "batch_size": 64,
            "replay_buffer_size": 10_000,
            "learning_rate": 0.001,
            "epsilon_greedy_prob": 0.0,
            "mixing_factor": 0.01
        }
    }
    TRANSFERLIGHT_A2C = {
        "class_name": "A2C",
        "init_args": {
            "network": NetworkConfig.TRANSFERLIGHT_A2C,
            "discount_factor": 0.9,
            "learning_rate": 0.001,
            "actor_loss_weight": 1.0,
            "critic_loss_weight": 1.0,
            "entropy_loss_weight": 0.0,
            "gradient_clipping_max_norm": 1.0
        }
    }
    PRESSLIGHT = {
        "class_name": "DQN",
        "init_args": {
            "network": NetworkConfig.PRESSLIGHT,
            "discount_factor": 0.9,
            "batch_size": 64,
            "replay_buffer_size": 10_000,
            "learning_rate": 0.001,
            "epsilon_greedy_prob": 0.0,
            "mixing_factor": 0.01
        }
    }
    MAX_PRESSURE = {
        "class_name": "MaxPressure",
        "init_args": {}
    }
    DEFAULT = {
        "class_name": "DefaultAgent",
        "init_args": {}
    }
    RANDOM = {
        "class_name": "RandomAgent",
        "init_args": {}
    }


class ProblemFormulationConfig(ConfigEnum):
    DUMMY = "DummyProblemFormulation"
    TRANSFERLIGHT = "TransferLightProblemFormulation"
    PRESSLIGHT = "PressLightProblemFormulation"
    MAX_PRESSURE = "MaxPressureProblemFormulation"


MARL_ENV_INIT_ARGS = {
    "max_patience": MAX_PATIENCE,
    "yellow_change_time": YELLOW_CHANGE_TIME,
    "all_red_time": ALL_RED_TIME,
    "action_time": ACTION_TIME
}

MP_MARL_ENV_INIT_ARGS = dict(MARL_ENV_INIT_ARGS, n_workers=N_WORKERS)


class EnvironmentConfig(ConfigEnum):
    MARL = {
        "class_name": "MarlEnvironment",
        "init_args": MARL_ENV_INIT_ARGS
    }
    MARL_DEMO = {
        "class_name": "MarlEnvironment",
        "init_args": dict(MARL_ENV_INIT_ARGS, demo=True)
    }
    MP_MARL = {
        "class_name": "MultiprocessingMarlEnvironment",
        "init_args": dict(MP_MARL_ENV_INIT_ARGS)
    }


@dataclass
class AgentSpec:
    agent_name: str
    agent_config: str
    agent_dir: str
    scenario_name: Optional[str]
    problem_formulation: str = ProblemFormulationConfig.DUMMY
    is_default: bool = False


agent_specs = {

    # TransferLight-A2C
    AgentNames.TRANSFERLIGHT_A2C_FIXED_ALL: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_A2C_FIXED_ALL,
        agent_config=AgentConfigs.TRANSFERLIGHT_A2C,
        agent_dir=AgentDirs.TRANSFERLIGHT_A2C_FIXED_ALL,
        scenario_name=ScenarioNames.FIXED_ALL,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_A2C_FIXED_NETWORK: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_A2C_FIXED_NETWORK,
        agent_config=AgentConfigs.TRANSFERLIGHT_A2C,
        agent_dir=AgentDirs.TRANSFERLIGHT_A2C_FIXED_NETWORK,
        scenario_name=ScenarioNames.FIXED_NETWORK,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_A2C_FIXED_LOCATION: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_A2C_FIXED_LOCATION,
        agent_config=AgentConfigs.TRANSFERLIGHT_A2C,
        agent_dir=AgentDirs.TRANSFERLIGHT_A2C_FIXED_LOCATION,
        scenario_name=ScenarioNames.FIXED_LOCATION,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_A2C_FIXED_RATE: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_A2C_FIXED_RATE,
        agent_config=AgentConfigs.TRANSFERLIGHT_A2C,
        agent_dir=AgentDirs.TRANSFERLIGHT_A2C_FIXED_RATE,
        scenario_name=ScenarioNames.FIXED_RATE,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_A2C_RANDOM_ALL: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_A2C_RANDOM_ALL,
        agent_config=AgentConfigs.TRANSFERLIGHT_A2C,
        agent_dir=AgentDirs.TRANSFERLIGHT_A2C_RANDOM_ALL,
        scenario_name=ScenarioNames.RANDOM_ALL,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_A2C_RANDOM_NETWORK: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_A2C_RANDOM_NETWORK,
        agent_config=AgentConfigs.TRANSFERLIGHT_A2C,
        agent_dir=AgentDirs.TRANSFERLIGHT_A2C_RANDOM_NETWORK,
        scenario_name=ScenarioNames.RANDOM_NETWORK,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_A2C_RANDOM_LOCATION: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_A2C_RANDOM_LOCATION,
        agent_config=AgentConfigs.TRANSFERLIGHT_A2C,
        agent_dir=AgentDirs.TRANSFERLIGHT_A2C_RANDOM_LOCATION,
        scenario_name=ScenarioNames.RANDOM_LOCATION,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_A2C_RANDOM_RATE: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_A2C_RANDOM_RATE,
        agent_config=AgentConfigs.TRANSFERLIGHT_A2C,
        agent_dir=AgentDirs.TRANSFERLIGHT_A2C_RANDOM_RATE,
        scenario_name=ScenarioNames.RANDOM_RATE,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),

    # TransferLight-DQN
    AgentNames.TRANSFERLIGHT_DQN_FIXED_ALL: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_DQN_FIXED_ALL,
        agent_config=AgentConfigs.TRANSFERLIGHT_DQN,
        agent_dir=AgentDirs.TRANSFERLIGHT_DQN_FIXED_ALL,
        scenario_name=ScenarioNames.FIXED_ALL,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_DQN_FIXED_NETWORK: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_DQN_FIXED_NETWORK,
        agent_config=AgentConfigs.TRANSFERLIGHT_DQN,
        agent_dir=AgentDirs.TRANSFERLIGHT_DQN_FIXED_NETWORK,
        scenario_name=ScenarioNames.FIXED_NETWORK,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_DQN_FIXED_LOCATION: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_DQN_FIXED_LOCATION,
        agent_config=AgentConfigs.TRANSFERLIGHT_DQN,
        agent_dir=AgentDirs.TRANSFERLIGHT_DQN_FIXED_LOCATION,
        scenario_name=ScenarioNames.FIXED_LOCATION,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_DQN_FIXED_RATE: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_DQN_FIXED_RATE,
        agent_config=AgentConfigs.TRANSFERLIGHT_DQN,
        agent_dir=AgentDirs.TRANSFERLIGHT_DQN_FIXED_RATE,
        scenario_name=ScenarioNames.FIXED_RATE,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_DQN_RANDOM_ALL: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_DQN_RANDOM_ALL,
        agent_config=AgentConfigs.TRANSFERLIGHT_DQN,
        agent_dir=AgentDirs.TRANSFERLIGHT_DQN_RANDOM_ALL,
        scenario_name=ScenarioNames.RANDOM_ALL,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_DQN_RANDOM_NETWORK: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_DQN_RANDOM_NETWORK,
        agent_config=AgentConfigs.TRANSFERLIGHT_DQN,
        agent_dir=AgentDirs.TRANSFERLIGHT_DQN_RANDOM_NETWORK,
        scenario_name=ScenarioNames.RANDOM_NETWORK,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_DQN_RANDOM_LOCATION: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_DQN_RANDOM_LOCATION,
        agent_config=AgentConfigs.TRANSFERLIGHT_DQN,
        agent_dir=AgentDirs.TRANSFERLIGHT_DQN_RANDOM_LOCATION,
        scenario_name=ScenarioNames.RANDOM_LOCATION,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_DQN_RANDOM_RATE: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_DQN_RANDOM_RATE,
        agent_config=AgentConfigs.TRANSFERLIGHT_DQN,
        agent_dir=AgentDirs.TRANSFERLIGHT_DQN_RANDOM_RATE,
        scenario_name=ScenarioNames.RANDOM_RATE,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_DQN_ARTERIAL_HEAVY: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_DQN_ARTERIAL_HEAVY,
        agent_config=AgentConfigs.TRANSFERLIGHT_DQN,
        agent_dir=AgentDirs.TRANSFERLIGHT_DQN_ARTERIAL_HEAVY,
        scenario_name=ScenarioNames.ARTERIAL_HEAVY,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_DQN_ARTERIAL_LIGHT: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_DQN_ARTERIAL_LIGHT,
        agent_config=AgentConfigs.TRANSFERLIGHT_DQN,
        agent_dir=AgentDirs.TRANSFERLIGHT_DQN_ARTERIAL_LIGHT,
        scenario_name=ScenarioNames.ARTERIAL_LIGHT,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),

    # PressLight
    AgentNames.PRESSLIGHT_ARTERIAL_LIGHT: AgentSpec(
        agent_name=AgentNames.PRESSLIGHT_ARTERIAL_LIGHT,
        agent_config=AgentConfigs.PRESSLIGHT,
        agent_dir=AgentDirs.PRESSLIGHT_ARTERIAL_LIGHT,
        scenario_name=ScenarioNames.ARTERIAL_LIGHT,
        problem_formulation=ProblemFormulationConfig.PRESSLIGHT
    ),
    AgentNames.PRESSLIGHT_ARTERIAL_HEAVY: AgentSpec(
        agent_name=AgentNames.PRESSLIGHT_ARTERIAL_HEAVY,
        agent_config=AgentConfigs.PRESSLIGHT,
        agent_dir=AgentDirs.PRESSLIGHT_ARTERIAL_HEAVY,
        scenario_name=ScenarioNames.ARTERIAL_HEAVY,
        problem_formulation=ProblemFormulationConfig.PRESSLIGHT
    ),

    # Others
    AgentNames.MAX_PRESSURE: AgentSpec(
        agent_name=AgentNames.MAX_PRESSURE,
        agent_config=AgentConfigs.MAX_PRESSURE,
        agent_dir=AgentDirs.MAX_PRESSURE,
        scenario_name=None,
        problem_formulation=ProblemFormulationConfig.MAX_PRESSURE
    ),
    AgentNames.FIXED_TIME: AgentSpec(
        agent_name=AgentNames.FIXED_TIME,
        agent_config=AgentConfigs.DEFAULT,
        agent_dir=AgentDirs.FIXED_TIME,
        scenario_name=None,
        is_default=True
    ),
    AgentNames.RANDOM: AgentSpec(
        agent_name=AgentNames.RANDOM,
        agent_config=AgentConfigs.RANDOM,
        agent_dir=AgentDirs.RANDOM,
        scenario_name=None,
        problem_formulation=ProblemFormulationConfig.DUMMY
    )
}
