import os

import torch

from src.params import SCENARIOS_ROOT
from src.rl.environments import TscMarlEnvironment
from src.rl.agents import MaxPressureAgents, IQLAgents, A2C

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    #scenarios_dir = os.path.join(SCENARIOS_ROOT, "grid", "3x3-150m", "train")
    scenarios_dir = os.path.join(SCENARIOS_ROOT, "isolated", "train")

    #environment = TscMarlEnvironment(scenarios_dir, 180, "MaxPressureTrafficRepresentation", use_default=False, demo=True)
    environment = TscMarlEnvironment(scenarios_dir, 500, "GeneraLightTrafficRepresentation", use_default=False,
                                     demo=True)

    input_dim = 10
    hidden_dim = 64

    network = {
        "class_name": "GeneraLightNetwork",
        "init_args": {
            "hidden_dim": hidden_dim
        }
    }
    actor_head = {
        "class_name": "LinearActorHead",
        "init_args": {
            "input_dim": hidden_dim,
            "hidden_dim": hidden_dim,
            "residual_stack_size": 2
        }
    }
    critic_head = {
        "class_name": "LinearCriticHead",
        "init_args": {
            "input_dim": hidden_dim,
            "hidden_dim": hidden_dim,
            "residual_stack_sizes": (2, 2),
            "aggr_fn": "sum"
        }
    }


    #agents = MaxPressureAgents(20)
    qnet_params = {"state_size": 16, "hidden_size": 64, "action_size": 4}
    #agents = IQLAgents(qnet_params, 1000, 128, 0.01, 0.9, 0.01, 1.0, 0.1, 10_000, checkpoint_dir=os.path.join("agents", "IQL"), load_checkpoint=True)
    agents = A2C(network, actor_head, critic_head, share_network=True, n_steps=1, gae_discount_factor=0.0, checkpoint_dir=os.path.join("agents", "A2C"), load_checkpoint=True)

    agents.demo_env(environment)

