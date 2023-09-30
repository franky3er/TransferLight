import argparse
import os
import subprocess
from typing import List

from src import params
from src.params import agent_specs, EnvironmentConfig, TRAIN_STEPS, TRAIN_SKIP_STEPS, PROJECT_ROOT
from src.rl.agents import Agent
from src.rl.environments import Environment


trainable_agent_specs = {name: spec for name, spec in agent_specs.items() if spec.train_scenarios_dir is not None}

parser = argparse.ArgumentParser(
    prog="Training Program",
    description="Trains an agent specified by name"
)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-r", "--run", help="train agent(s)", choices=list(trainable_agent_specs.keys()), nargs="+")
group.add_argument("-a", "--all", help="train all available agents", action="store_true")
group.add_argument("-l", "--list", help="list available agents", action="store_true")
parser.add_argument("-d", "--device", help="device name (cpu/cuda)", required=False, default=params.DEVICE)


def execute_train_job(agent_name: str):
    print(f"run train job {agent_name}")
    agent_spec = trainable_agent_specs[agent_name]
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


def execute_train_jobs(job_names: List[str]):
    for job_name in job_names:
        command = f"python {PROJECT_ROOT}/train.py -r {job_name} -d {params.DEVICE}"
        process = subprocess.Popen(command, shell=True)
        process.wait()


if __name__ == "__main__":
    args = parser.parse_args()
    params.DEVICE = args.device
    if args.list:
        print("\n".join(list(trainable_agent_specs.keys())))
    elif args.run is not None:
        if len(args.run) == 1:
            execute_train_job(args.run[0])
        else:
            execute_train_jobs(args.run)
    elif args.all:
        execute_train_jobs(list(trainable_agent_specs.keys()))
