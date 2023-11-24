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

RESCO_GITHUB_LINK = "https://github.com/Pi-Star-Lab/RESCO.git"
RESCO_ROOT = os.path.join(PROJECT_ROOT, "tmp", "RESCO")
RESCO_SCENARIOS_DIR = os.path.join(RESCO_ROOT, "resco_benchmark", "environments")
RESCO_COLOGNE3_DIR = os.path.join(RESCO_SCENARIOS_DIR, "cologne3")
RESCO_COLOGNE8_DIR = os.path.join(RESCO_SCENARIOS_DIR, "cologne8")
RESCO_INGOLSTADT7_DIR = os.path.join(RESCO_SCENARIOS_DIR, "ingolstadt7")

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
VEHICLE_DEPARTURE_ALPHA_MAX = 10.0
VEHICLE_DEPARTURE_BETA_MIN = 1.0
VEHICLE_DEPARTURE_BETA_MAX = 10.0

N_VEHICLES = 300
N_FLOWS = 25
FLOW_ALPHA_RANGE = (1.0, 10.0)
FLOW_BETA_RANGE = (1.0, 10.0)
DURATION = 900

N_TRAIN_SCENARIOS = 1_000
N_TEST_SCENARIOS = 10


class ScenarioNames(ConfigEnum):
    FIXED = "fixed"
    RANDOM = "random"
    RANDOM_NETWORK = "random-network"
    RANDOM_TRAFFIC = "random-traffic"

    RANDOM_LIGHT = "random-light"
    RANDOM_HEAVY = "random-heavy"
    ARTERIAL = "arterial"

    COLOGNE3 = "cologne3"
    COLOGNE8 = "cologne8"
    INGOLSTADT7 = "ingolstadt7"


class TrainScenariosDirs(ConfigEnum):
    FIXED = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.FIXED)
    RANDOM = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.RANDOM)
    RANDOM_NETWORK = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.RANDOM_NETWORK)
    RANDOM_TRAFFIC = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.RANDOM_TRAFFIC)

    RANDOM_LIGHT = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.RANDOM_LIGHT)
    RANDOM_HEAVY = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.RANDOM_HEAVY)
    ARTERIAL = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.ARTERIAL)

    COLOGNE3 = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.COLOGNE3)
    COLOGNE8 = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.COLOGNE8)
    INGOLSTADT7 = os.path.join(TRAIN_SCENARIOS_ROOT, ScenarioNames.INGOLSTADT7)


class TestScenarioDirs(ConfigEnum):
    FIXED = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.FIXED)
    RANDOM = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.RANDOM)
    RANDOM_NETWORK = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.RANDOM_NETWORK)
    RANDOM_TRAFFIC = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.RANDOM_TRAFFIC)

    RANDOM_LIGHT = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.RANDOM_LIGHT)
    RANDOM_HEAVY = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.RANDOM_HEAVY)
    ARTERIAL = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.ARTERIAL)

    COLOGNE3 = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.COLOGNE3)
    COLOGNE8 = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.COLOGNE8)
    INGOLSTADT7 = os.path.join(TEST_SCENARIOS_ROOT, ScenarioNames.INGOLSTADT7)


@dataclass
class ScenarioSpec:
    name: str
    generator: str
    generator_args: Dict
    test_dir: str
    test_max_time: int = DURATION
    test_max_patience: int = sys.maxsize
    n_test_scenarios: int = N_TEST_SCENARIOS
    train_dir: Optional[str] = None
    train_max_time: int = sys.maxsize
    train_max_patience: int = MAX_PATIENCE
    n_train_scenarios: int = N_TRAIN_SCENARIOS


scenario_specs = {
    ScenarioNames.FIXED: ScenarioSpec(
        name=ScenarioNames.FIXED,
        train_dir=TrainScenariosDirs.FIXED,
        test_dir=TestScenarioDirs.FIXED,
        n_train_scenarios=1,
        n_test_scenarios=1,
        generator="DomainRandomizationScenariosGenerator",
        generator_args={"random_network": False, "random_traffic": False}
    ),
    ScenarioNames.RANDOM: ScenarioSpec(
        name=ScenarioNames.RANDOM,
        train_dir=TrainScenariosDirs.RANDOM,
        test_dir=TestScenarioDirs.RANDOM,
        n_train_scenarios=N_TRAIN_SCENARIOS,
        n_test_scenarios=N_TEST_SCENARIOS,
        generator="DomainRandomizationScenariosGenerator",
        generator_args={"random_network": True, "random_traffic": True, "seed_network": 0, "seed_traffic": 0}
    ),
    ScenarioNames.RANDOM_NETWORK: ScenarioSpec(
        name=ScenarioNames.RANDOM_NETWORK,
        train_dir=TrainScenariosDirs.RANDOM_NETWORK,
        test_dir=TestScenarioDirs.RANDOM_NETWORK,
        generator="DomainRandomizationScenariosGenerator",
        generator_args={"random_network": True, "random_traffic": False, "seed_network": 3030, "seed_traffic": SEED}
    ),
    ScenarioNames.RANDOM_TRAFFIC: ScenarioSpec(
        name=ScenarioNames.RANDOM_TRAFFIC,
        train_dir=TrainScenariosDirs.RANDOM_TRAFFIC,
        test_dir=TestScenarioDirs.RANDOM_TRAFFIC,
        generator="DomainRandomizationScenariosGenerator",
        generator_args={"random_network": False, "random_traffic": True, "seed_network": SEED, "seed_traffic": 4040}
    ),

    ScenarioNames.RANDOM_LIGHT: ScenarioSpec(
        name=ScenarioNames.RANDOM_LIGHT,
        train_dir=TrainScenariosDirs.RANDOM_LIGHT,
        test_dir=TestScenarioDirs.RANDOM_LIGHT,
        n_train_scenarios=0,
        n_test_scenarios=1,
        test_max_time=3600,
        generator="DomainRandomizationScenariosGenerator",
        generator_args={"random_network": True, "random_traffic": True, "n_flows": 100,
                        "n_veh": int(((N_VEHICLES - (1/3) * N_VEHICLES) / DURATION) * 3_600), "duration": 3_600,
                        "seed_network": 6062, "seed_traffic": 6062}
    ),
    ScenarioNames.RANDOM_HEAVY: ScenarioSpec(
        name=ScenarioNames.RANDOM_HEAVY,
        train_dir=TrainScenariosDirs.RANDOM_HEAVY,
        test_dir=TestScenarioDirs.RANDOM_HEAVY,
        n_train_scenarios=0,
        n_test_scenarios=1,
        test_max_time=3600,
        generator="DomainRandomizationScenariosGenerator",
        generator_args={"random_network": True, "random_traffic": True, "n_flows": 100,
                        "n_veh": int(((N_VEHICLES + (1/3) * N_VEHICLES) / DURATION) * 3_600), "duration": 3_600,
                        "seed_network": 6062, "seed_traffic": 6062}
    ),
    ScenarioNames.ARTERIAL: ScenarioSpec(
        name=ScenarioNames.ARTERIAL,
        train_dir=TrainScenariosDirs.ARTERIAL,
        test_dir=TestScenarioDirs.ARTERIAL,
        generator="ArterialScenariosGenerator",
        n_train_scenarios=1,
        n_test_scenarios=1,
        test_max_time=3600,
        generator_args={"n_intersections": 5, "lane_length": 200.0, "allowed_speed": 13.89, "arterial_flow_rate": 700.0,
                        "side_street_flow_rate": 420.0, "duration": 3_600}
    ),

    ScenarioNames.COLOGNE3: ScenarioSpec(
        name=ScenarioNames.COLOGNE3,
        train_dir=TestScenarioDirs.COLOGNE3,
        test_dir=TestScenarioDirs.COLOGNE3,
        test_max_time=3600,
        generator="RESCOScenariosGenerator",
        generator_args={"name": ScenarioNames.COLOGNE3}
    ),
    ScenarioNames.COLOGNE8: ScenarioSpec(
        name=ScenarioNames.COLOGNE8,
        train_dir=TestScenarioDirs.COLOGNE8,
        test_dir=TestScenarioDirs.COLOGNE8,
        test_max_time=3600,
        generator="RESCOScenariosGenerator",
        generator_args={"name": ScenarioNames.COLOGNE8}
    ),
    ScenarioNames.INGOLSTADT7: ScenarioSpec(
        name=ScenarioNames.INGOLSTADT7,
        train_dir=TestScenarioDirs.INGOLSTADT7,
        test_dir=TestScenarioDirs.INGOLSTADT7,
        test_max_time=3600,
        generator="RESCOScenariosGenerator",
        generator_args={"name": ScenarioNames.INGOLSTADT7}
    ),
}


class AgentNames(ConfigEnum):
    TRANSFERLIGHT = "TransferLight"
    TRANSFERLIGHT_A2C = f"{TRANSFERLIGHT}-A2C"
    TRANSFERLIGHT_DQN = f"{TRANSFERLIGHT}-DQN"

    TRANSFERLIGHT_A2C_FIXED = f"{TRANSFERLIGHT_A2C}-{ScenarioNames.FIXED}"
    TRANSFERLIGHT_A2C_RANDOM = f"{TRANSFERLIGHT_A2C}-{ScenarioNames.RANDOM}"
    TRANSFERLIGHT_A2C_RANDOM_NETWORK = f"{TRANSFERLIGHT_A2C}-{ScenarioNames.RANDOM_NETWORK}"
    TRANSFERLIGHT_A2C_RANDOM_TRAFFIC = f"{TRANSFERLIGHT_A2C}-{ScenarioNames.RANDOM_TRAFFIC}"
    TRANSFERLIGHT_DQN_FIXED = f"{TRANSFERLIGHT_DQN}-{ScenarioNames.FIXED}"
    TRANSFERLIGHT_DQN_RANDOM = f"{TRANSFERLIGHT_DQN}-{ScenarioNames.RANDOM}"
    TRANSFERLIGHT_DQN_RANDOM_NETWORK = f"{TRANSFERLIGHT_DQN}-{ScenarioNames.RANDOM_NETWORK}"
    TRANSFERLIGHT_DQN_RANDOM_TRAFFIC = f"{TRANSFERLIGHT_DQN}-{ScenarioNames.RANDOM_TRAFFIC}"
    TRANSFERLIGHT_DQN_ARTERIAL = f"{TRANSFERLIGHT_DQN}-{ScenarioNames.ARTERIAL}"

    PRESSLIGHT = "PressLight"
    MAX_PRESSURE = "MaxPressure"
    FIXED_TIME = "FixedTime"
    RANDOM = "Random"


class AgentDirs(ConfigEnum):
    TRANSFERLIGHT_A2C_FIXED = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_A2C_FIXED)
    TRANSFERLIGHT_A2C_RANDOM = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_A2C_RANDOM)
    TRANSFERLIGHT_A2C_RANDOM_NETWORK = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_A2C_RANDOM_NETWORK)
    TRANSFERLIGHT_A2C_RANDOM_TRAFFIC = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_A2C_RANDOM_TRAFFIC)
    TRANSFERLIGHT_DQN_FIXED = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_DQN_FIXED)
    TRANSFERLIGHT_DQN_RANDOM = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_DQN_RANDOM)
    TRANSFERLIGHT_DQN_RANDOM_NETWORK = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_DQN_RANDOM_NETWORK)
    TRANSFERLIGHT_DQN_RANDOM_TRAFFIC = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_DQN_RANDOM_TRAFFIC)
    TRANSFERLIGHT_DQN_ARTERIAL = os.path.join(RESULTS_ROOT, AgentNames.TRANSFERLIGHT_DQN_ARTERIAL)

    PRESSLIGHT = os.path.join(RESULTS_ROOT, AgentNames.PRESSLIGHT)
    MAX_PRESSURE = os.path.join(RESULTS_ROOT, AgentNames.MAX_PRESSURE)
    FIXED_TIME = os.path.join(RESULTS_ROOT, AgentNames.FIXED_TIME)
    RANDOM = os.path.join(RESULTS_ROOT, AgentNames.RANDOM)


class TransferLightConfig(ConfigEnum):
    HIDDEN_DIM = 128
    N_ATTENTION_HEADS = 8
    DROPOUT_PROB = 0.1


class PressLightConfig(ConfigEnum):
    STATE_DIM = 18
    HIDDEN_DIM = 128
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
            "batch_size": 128,
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
    AgentNames.TRANSFERLIGHT_A2C_FIXED: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_A2C_FIXED,
        agent_config=AgentConfigs.TRANSFERLIGHT_A2C,
        agent_dir=AgentDirs.TRANSFERLIGHT_A2C_FIXED,
        scenario_name=ScenarioNames.FIXED,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_A2C_RANDOM: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_A2C_RANDOM,
        agent_config=AgentConfigs.TRANSFERLIGHT_A2C,
        agent_dir=AgentDirs.TRANSFERLIGHT_A2C_RANDOM,
        scenario_name=ScenarioNames.RANDOM,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_A2C_RANDOM_NETWORK: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_A2C_RANDOM_NETWORK,
        agent_config=AgentConfigs.TRANSFERLIGHT_A2C,
        agent_dir=AgentDirs.TRANSFERLIGHT_A2C_RANDOM_NETWORK,
        scenario_name=ScenarioNames.RANDOM_NETWORK,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_A2C_RANDOM_TRAFFIC: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_A2C_RANDOM_TRAFFIC,
        agent_config=AgentConfigs.TRANSFERLIGHT_A2C,
        agent_dir=AgentDirs.TRANSFERLIGHT_A2C_RANDOM_TRAFFIC,
        scenario_name=ScenarioNames.RANDOM_TRAFFIC,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_DQN_FIXED: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_DQN_FIXED,
        agent_config=AgentConfigs.TRANSFERLIGHT_DQN,
        agent_dir=AgentDirs.TRANSFERLIGHT_DQN_FIXED,
        scenario_name=ScenarioNames.FIXED,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_DQN_RANDOM: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_DQN_RANDOM,
        agent_config=AgentConfigs.TRANSFERLIGHT_DQN,
        agent_dir=AgentDirs.TRANSFERLIGHT_DQN_RANDOM,
        scenario_name=ScenarioNames.RANDOM,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_DQN_RANDOM_NETWORK: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_DQN_RANDOM_NETWORK,
        agent_config=AgentConfigs.TRANSFERLIGHT_DQN,
        agent_dir=AgentDirs.TRANSFERLIGHT_DQN_RANDOM_NETWORK,
        scenario_name=ScenarioNames.RANDOM_NETWORK,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_DQN_RANDOM_TRAFFIC: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_DQN_RANDOM_TRAFFIC,
        agent_config=AgentConfigs.TRANSFERLIGHT_DQN,
        agent_dir=AgentDirs.TRANSFERLIGHT_DQN_RANDOM_TRAFFIC,
        scenario_name=ScenarioNames.RANDOM_TRAFFIC,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),
    AgentNames.TRANSFERLIGHT_DQN_ARTERIAL: AgentSpec(
        agent_name=AgentNames.TRANSFERLIGHT_DQN_ARTERIAL,
        agent_config=AgentConfigs.TRANSFERLIGHT_DQN,
        agent_dir=AgentDirs.TRANSFERLIGHT_DQN_ARTERIAL,
        scenario_name=ScenarioNames.ARTERIAL,
        problem_formulation=ProblemFormulationConfig.TRANSFERLIGHT
    ),

    # PressLight
    AgentNames.PRESSLIGHT: AgentSpec(
        agent_name=AgentNames.PRESSLIGHT,
        agent_config=AgentConfigs.PRESSLIGHT,
        agent_dir=AgentDirs.PRESSLIGHT,
        scenario_name=ScenarioNames.ARTERIAL,
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
