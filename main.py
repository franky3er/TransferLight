import argparse
from collections import defaultdict
import ntpath
import os
import subprocess
import time

from src import params
from src.params import (agent_specs, EnvironmentConfig, PROJECT_ROOT, TRAIN_STEPS,
                        TRAIN_SKIP_STEPS, ScenarioNames, scenario_specs, TRAIN_SCENARIOS_ROOT)
from src.rl.environments import Environment
from src.rl.agents import Agent, DefaultAgent
from src.sumo.scenarios import ScenariosGenerator


trainable_agent_specs = {agent_name: agent_spec for agent_name, agent_spec in agent_specs.items()
                         if agent_spec.scenario_name is not None}


def setup_handler(args):
    scenarios = args.scenario if args.scenario else scenario_specs.keys()
    for scenario in scenarios:
        scenario_spec = scenario_specs[scenario]
        generator = ScenariosGenerator.create(scenario_spec)
        generator.generate_scenarios()


def train_handler(args):
    device = args.device
    params.DEVICE = device
    single_agent = len(args.agent) == 1 and args.agent[0] != "ALL"
    if single_agent:
        print(f"train agent {args.agent[0]}")
        agent_spec = agent_specs[args.agent[0]]
        results_dir = agent_spec.agent_dir
        checkpoint_dir = os.path.join(results_dir, "checkpoints")
        scenario_spec = scenario_specs[agent_spec.scenario_name]
        train_dir = scenario_spec.train_dir
        max_time = scenario_spec.train_max_time
        max_patience = scenario_spec.train_max_patience
        problem_formulation = agent_spec.problem_formulation
        use_default = agent_spec.is_default
        environment = Environment.create(EnvironmentConfig.MP_MARL["class_name"],
                                         dict(EnvironmentConfig.MP_MARL["init_args"],
                                              scenarios_dirs=train_dir, max_time=max_time, max_patience=max_patience,
                                              problem_formulation=problem_formulation, use_default=use_default))
        agent = Agent.create(agent_spec.agent_config["class_name"], agent_spec.agent_config["init_args"])
        agent.fit(environment, steps=TRAIN_STEPS, skip_steps=TRAIN_SKIP_STEPS, checkpoint_dir=checkpoint_dir)
    else:
        if args.agent[0] == "ALL":
            agents = list(trainable_agent_specs.keys())
        else:
            agents = args.agent
        for agent in agents:
            cmd = f"python {os.path.join(PROJECT_ROOT, 'main.py')} train -a {agent} -d {device}"
            process = subprocess.Popen(cmd, shell=True)
            process.wait()


def test_handler(args):
    device = args.device
    params.DEVICE = device
    agent_constraints_satisfied = len(args.agent) == 1 and "ALL" not in args.agent
    checkpoint_constraints_satisfied = len(args.checkpoint) == 1 and "ALL" not in args.checkpoint
    if "ALL" in args.scenario or "AGENT" in args.scenario:
        scenario_constraints_satisfied = False
    elif len({(scenario_specs[scenario].test_max_time, scenario_specs[scenario].test_max_patience)
              for scenario in args.scenario}) > 1:
        scenario_constraints_satisfied = False
    else:
        scenario_constraints_satisfied = True
    if agent_constraints_satisfied and checkpoint_constraints_satisfied and scenario_constraints_satisfied:
        print(f"Test agent '{args.agent[0]}' on scenarios {args.scenario} using checkpoint {args.checkpoint[0]}]")
        agent = args.agent[0]
        checkpoint_path = args.checkpoint[0]
        agent_spec = agent_specs[agent]
        agent_config = agent_spec.agent_config
        scenarios_dirs = [scenario_specs[scenario].test_dir for scenario in args.scenario]

        env_config = EnvironmentConfig.MP_MARL
        stats_dir = os.path.join(agent_spec.agent_dir, "test",
                                 "best" if checkpoint_path is None else ntpath.split(checkpoint_path)[1].split(".")[0])
        max_time = scenario_specs[args.scenario[0]].test_max_time
        max_patience = scenario_specs[args.scenario[0]].test_max_patience
        environment = Environment.create(env_config["class_name"],
                                         dict(env_config["init_args"],
                                              scenarios_dirs=scenarios_dirs, stats_dir=stats_dir, cycle_scenarios=False,
                                              max_time=max_time, max_patience=max_patience,
                                              problem_formulation=agent_spec.problem_formulation,
                                              use_default=agent_spec.is_default))
        agent = Agent.create(agent_config["class_name"], agent_config["init_args"])
        agent.test(environment, checkpoint_path=checkpoint_path, stats_dir=stats_dir)
        exit(0)

    test_cases = []
    if "ALL" in args.agent:
        agents = list(agent_specs.keys())
    elif "TRAINABLE" in args.agent:
        agents = list(trainable_agent_specs.keys())
    else:
        agents = args.agent

    for agent in agents:
        agent_spec = agent_specs[agent]

        if "AGENT" in args.scenario:
            if agent_spec.scenario_name is None:
                print(f"No scenario for agent {agent} specified")
                continue
            scenario_specs_ = [scenario_specs[agent_spec.scenario_name]]
        elif "ALL" in args.scenario:
            scenario_specs_ = scenario_specs.values()
        else:
            scenario_specs_ = [scenario_specs[scenario] for scenario in args.scenario]
        scenarios = defaultdict(lambda: [])
        for scenario_spec in scenario_specs_:
            scenarios[(scenario_spec.test_max_time, scenario_spec.test_max_patience)].append(scenario_spec.name)

        if "BEST" in args.checkpoint:
            checkpoint_paths = [os.path.join(agent_spec.agent_dir, "checkpoints", "best.pt")]
            if os.path.exists(checkpoint_paths[0]):
                print(f"No best checkpoint for agent {agent} specified")
                continue
        elif "ALL" in args.checkpoint:
            checkpoint_dir = os.path.join(agent_spec.agent_dir, "checkpoints")
            checkpoint_paths = [os.path.join(checkpoint_dir, file) for file in os.listdir(checkpoint_dir)]
        else:
            checkpoint_paths = args.checkpoint

        for scenarios in scenarios.values():
            test_cases += [(agent, scenarios, checkpoint_path) for checkpoint_path in checkpoint_paths]

    for agent, scenarios, checkpoint_path in test_cases:
        scenarios = " ".join(scenarios)
        cmd = (f"python {os.path.join(PROJECT_ROOT, 'main.py')} test -a {agent} -s {scenarios} -c {checkpoint_path}"
               f" -d {device}")
        process = subprocess.Popen(cmd, shell=True)
        process.wait()


def demo_handler(args):
    agent_name = args.agent
    agent_spec = agent_specs[agent_name]
    agent_config = agent_spec.agent_config
    checkpoint_path = args.checkpoint
    scenario_path = args.scenario
    params.DEVICE = args.device
    env_config = EnvironmentConfig.MARL_DEMO
    agent = Agent.create(agent_config["class_name"], agent_config["init_args"])
    use_default = DefaultAgent is agent.__class__
    environment = Environment.create(env_config["class_name"],
                                     dict(env_config["init_args"],
                                          scenario_path=scenario_path, use_default=use_default,
                                          problem_formulation=agent_spec.problem_formulation))
    agent.demo(environment, checkpoint_path=checkpoint_path)


def list_handler(args):
    if args.agent:
        if not args.exclude_testable:
            print("Testable agents: ")
            print(f"{args.separator}".join(agent_specs.keys()))
        if not args.exclude_trainable:
            print("\nTrainable agents: ")
            print(f"{args.separator}".join(trainable_agent_specs.keys()))
    elif args.scenario:
        raise NotImplementedError("Functionality not implemented yet")


parser = argparse.ArgumentParser(
    prog="Main Program",
    description="Train / Test / Demonstrate TSC Agents"
)
subparsers = parser.add_subparsers()

setup_parser = subparsers.add_parser("setup", help="setup train and test scenarios")
setup_parser.add_argument("-s", "--scenario", choices=list(scenario_specs.keys()), required=False, nargs="*",
                          default=[])
setup_parser.set_defaults(func=setup_handler)

train_parser = subparsers.add_parser("train", help="train agent(s)")
train_parser.add_argument("-a", "--agent", choices=list(agent_specs.keys()) + ["ALL"], nargs="+",
                          help="agent(s) to be trained ('ALL' is default and trains all available agents)",
                          default=["ALL"])
train_parser.add_argument("-d", "--device", help="device name (cpu/cuda)", default=params.DEVICE,
                          required=False)
train_parser.set_defaults(func=train_handler)

test_parser = subparsers.add_parser("test", help="test agent(s) on scenario(s)")
test_parser.add_argument("-a", "--agent", choices=list(agent_specs.keys()) + ["ALL", "TRAINABLE"], nargs="+",
                         help="agent(s) to be tested ('ALL' is default and tests all available agents, "
                              "'TransferLight' selects all TransferLight agents)",
                         default=["ALL"])
test_parser.add_argument("-c", "--checkpoint", required=False, nargs="*", default=["BEST"],
                         help="checkpoint(s) to test if available ('BEST' is default and selects the best checkpoint, "
                              "'ALL' selects all available checkpoints)")
test_parser.add_argument("-s", "--scenario", required=False, nargs="+", default=["ALL"],
                         choices=[s.value for s in ScenarioNames] + ["ALL"] + ["AGENT"],
                         help="scenario(s) to test agent on ('ALL' is default and selects all available scenarios, "
                              "'AGENT' selects the test scenarios specified for each individual agent)")
test_parser.add_argument("-d", "--device", help="device name (cpu/cuda)", default=params.DEVICE,
                         required=False)
test_parser.set_defaults(func=test_handler)

demo_parser = subparsers.add_parser("demo", help="demonstrate agent on scenario")
demo_parser.add_argument("-a", "--agent", choices=list(agent_specs.keys()), help="agent",
                         required=True)
demo_parser.add_argument("-s", "--scenario", help="path to scenario (.sumocfg) file",
                         required=True)
demo_parser.add_argument("-c", "--checkpoint", help="path to checkpoint", required=False)
demo_parser.add_argument("-d", "--device", help="device name (cpu/cuda)", default="cpu", required=False)
demo_parser.set_defaults(func=demo_handler)

list_parser = subparsers.add_parser("list", help="list available agents / scenarios")
list_parser_group_1 = list_parser.add_mutually_exclusive_group(required=True)
list_parser_group_1.add_argument("-a", "--agent", help="list agents", action="store_true")
list_parser_group_1.add_argument("-s", "--scenario", help="list scenarios", action="store_true")
list_parser.add_argument("--exclude-trainable", help="exclude trainable", action="store_true")
list_parser.add_argument("--exclude-testable", help="exclude testable", action="store_true")
list_parser.add_argument("--separator", default="\n", required=False)
list_parser.set_defaults(func=list_handler)


if __name__ == "__main__":
    args = parser.parse_args()
    args.func(args)
