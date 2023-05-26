import os

import torch
import torch_geometric

from src.params import SCENARIOS_ROOT
from src.rl.environments import TscMarlEnvironment
from src.rl.agents import MaxPressureAgents, IQLAgents, HieraGLightAgent, IA2CAgents

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    #scenarios_dir = os.path.join(SCENARIOS_ROOT, "grid", "3x3-150m", "train")
    scenarios_dir = os.path.join(SCENARIOS_ROOT, "isolated", "train")

    #environment = TscMarlEnvironment(scenarios_dir, 180, "MaxPressureTrafficRepresentation", use_default=False, demo=True)
    environment = TscMarlEnvironment(scenarios_dir, 500, "MaxPressureTrafficRepresentation", use_default=False, demo=True)

    input_dim = 10
    hidden_dim = 32

    traffic_embedding_homogeneous = {
        "class_name": "HomogeneousTrafficEmbedding",
        "init_args": {
            "intersection_embedding": {
                "class_name": "LinearIntersectionEmbedding",
                "init_args": {
                    "input_dim": 16,
                    "hidden_dim": hidden_dim,
                    "output_dim": hidden_dim,
                    "n_layer": 2
                }
            }
        }
    }

    traffic_embedding_heterogeneous = {
        "class_name": "HeterogeneousTrafficEmbedding",
        "init_args": {
            "movement_embedding": {
                "class_name": "LinearMovementEmbedding",
                "init_args": {
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "output_dim": hidden_dim,
                    "n_layer": 2
                }
            },
            "movement_to_phase_aggregation": {
                "class_name": "SimpleMovementToPhaseAggregation",
                "init_args": {
                    "aggr_fn": "mean"
                }
            },
            "phase_embedding": {
                "class_name": "CompetitionPhaseEmbedding",
                "init_args": {
                    "input_dim": hidden_dim,
                    "hidden_dim": hidden_dim // 2,
                    "output_dim": hidden_dim
                }
            }
        }
    }

    actor_homogeneous = {
        "class_name": "ActorNetwork",
        "init_args": {
            "traffic_embedding": traffic_embedding_homogeneous,
            "actor_head": {
                "class_name": "FixedActorHead",
                "init_args": {
                    "input_dim": hidden_dim,
                    "hidden_dim": hidden_dim,
                    "actions": 4,
                    "n_layer": 2
                }
            }
        }
    }

    actor_heterogeneous = {
        "class_name": "ActorNetwork",
        "init_args": {
            "traffic_embedding": traffic_embedding_heterogeneous,
            "actor_head": {
                "class_name": "FlexibleActorHead",
                "init_args": {
                    "input_dim": hidden_dim,
                    "hidden_dim": hidden_dim,
                    "n_layer": 2
                }
            }
        }
    }

    critit_homogeneous = {
        "class_name": "CriticNetwork",
        "init_args": {
            "traffic_embedding": traffic_embedding_homogeneous,
            "critic_head": {
                "class_name": "FixedCriticHead",
                "init_args": {
                    "input_dim": hidden_dim,
                    "hidden_dim": hidden_dim,
                    "n_layer": 2
                }
            }
        }
    }
    critic_heterogeneous = {
        "class_name": "CriticNetwork",
        "init_args": {
            "traffic_embedding": traffic_embedding_heterogeneous,
            "critic_head": {
                "class_name": "FlexibleCriticHead",
                "init_args": {
                    "input_dim": hidden_dim,
                    "hidden_dim": hidden_dim,
                    "n_layer": (2, 2),
                    "aggregation": {
                        "class_name": "SimpleNodeAggregation",
                        "init_args": {
                            "aggr": "sum"
                        }
                    }
                }
            }
        }
    }


    agents = MaxPressureAgents(20)
    qnet_params = {"state_size": 16, "hidden_size": 64, "action_size": 4}
    #agents = IQLAgents(qnet_params, 1000, 128, 0.01, 0.9, 0.01, 1.0, 0.1, 10_000, checkpoint_dir=os.path.join("agents", "IQL"), load_checkpoint=True)
    #agents = IA2CAgents(actor_heterogeneous, critic_heterogeneous, 0.001, 0.001, 0.9, 0.5, checkpoint_dir=os.path.join("agents", "IA2C"), load_checkpoint=True)

    agents.demo(environment)

