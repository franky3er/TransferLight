import argparse
import os
import subprocess

from src import params
from src.params import (agent_specs, TrainScenariosDirs, TestScenarioDirs, EnvironmentConfig, PROJECT_ROOT, TRAIN_STEPS,
                        TRAIN_SKIP_STEPS)
from src.rl.environments import Environment
from src.rl.agents import Agent


trainable_agent_specs = {agent_name: agent_spec for agent_name, agent_spec in agent_specs.items()
                         if agent_spec.train_scenarios_dir is not None}


def train_handler(args):
    device = args.device
    params.DEVICE = device
    single = len(args.agent) == 1 and args.agent[0] != "*"
    if single:
        agent_spec = trainable_agent_specs[args.agent[0]]
        results_dir = agent_spec.agent_dir
        checkpoint_dir = os.path.join(results_dir, "checkpoints")
        stats_dir = os.path.join(results_dir, "train-stats")
        vehicle_stats_dir = os.path.join(stats_dir, "vehicle")
        intersection_stats_dir = os.path.join(stats_dir, "intersection")
        environment = Environment.create(EnvironmentConfig.MP_MARL["class_name"],
                                         dict(EnvironmentConfig.MP_MARL["init_args"],
                                              scenarios_dir=agent_spec.train_scenarios_dir,
                                              problem_formulation=agent_spec.problem_formulation,
                                              vehicle_stats_dir=vehicle_stats_dir,
                                              intersection_stats_dir=intersection_stats_dir))
        agent = Agent.create(agent_spec.agent_config["class_name"], agent_spec.agent_config["init_args"])
        agent.fit(environment, steps=TRAIN_STEPS, skip_steps=TRAIN_SKIP_STEPS, checkpoint_dir=checkpoint_dir)
    else:
        agents = args.agent if args.agent[0] != "*" else list(trainable_agent_specs.keys())
        for agent in agents:
            cmd = f"python {os.path.join(PROJECT_ROOT, 'main.py')} train -a {agent} -d {device}"
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
    environment = Environment.create(env_config["class_name"],
                                     dict(env_config["init_args"],
                                          scenario_path=scenario_path,
                                          problem_formulation=agent_spec.problem_formulation))
    agent.demo(environment, checkpoint_path=checkpoint_path)


def test_handler(args):
    pass


def list_handler(args):
    if args.agent:
        if not args.exclude_testable:
            print("Testable agents: ")
            print("\n".join(agent_specs.keys()))
        if not args.exclude_trainable:
            print("\nTrainable agents: ")
            print("\n".join(trainable_agent_specs.keys()))
    elif args.scenario:
        raise NotImplementedError("Functionality not implemented yet")




parser = argparse.ArgumentParser(
    prog="Main Program",
    description="Train / Test / Demonstrate TSC Agents"
)
subparsers = parser.add_subparsers()

train_parser = subparsers.add_parser("train", help="train agent(s)")
train_parser.add_argument("-a", "--agent", choices=list(trainable_agent_specs.keys()) + ["*"], nargs="+",
                          help="agent(s) to be trained ('*' is a wildcard to train all agents)", required=True)
train_parser.add_argument("-d", "device", help="device name (cpu/cuda)", default=params.DEVICE,
                          required=False)
train_parser.set_defaults(func=train_handler)

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
list_parser.set_defaults(func=list_handler)


if __name__ == "__main__":
    args = parser.parse_args()
    args.func(args)
