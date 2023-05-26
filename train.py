import os

from src.params import SCENARIOS_ROOT
from src.rl.environments import MultiprocessingTscMarlEnvironment, TscMarlEnvironment
from src.rl.agents import IA2CAgents, MaxPressureAgents, IQLAgents

if __name__ == "__main__":

    scenarios_dir = os.path.join(SCENARIOS_ROOT, "isolated", "train")
    traffic_representation = "MaxPressureTrafficRepresentation"
    #environment = TscMarlEnvironment(scenarios_dir, 180, "MaxPressureTrafficRepresentation", use_default=False, demo=False)
    environment = MultiprocessingTscMarlEnvironment(scenarios_dir, 180, traffic_representation, 2)

    input_dim = 10
    hidden_dim = 32

    model_params = {"state_size": 16, "hidden_size": 64, "action_size": 4}
    train_params = {"buffer_size": 1_000, "batch_size": 128, "learning_rate": 0.01, "discount_factor": 0.9,
                    "tau": 0.01, "eps_greedy_start": 1.0, "eps_greedy_end": 0.1, "eps_greedy_steps": 10_000}

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
                    "hidden_dim": hidden_dim//2,
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

    #agents = RandomAgents()
    agents = MaxPressureAgents(20)
    #agents = IQLAgents(model_params, 1000, 128, 0.01, 0.9, 0.01, 1.0, 0.1, 10_000, checkpoint_dir=os.path.join("agents", "IQL"), save_checkpoint=True)

    #agents = IA2CAgents(actor_heterogeneous, critic_heterogeneous, 0.001, 0.001, 0.9, 0.001, checkpoint_dir=os.path.join("agents", "IA2C"), save_checkpoint=True)
    #agent = LitAgent(16, 64, 4, 1000, 128, 0.01, 0.99, 0.01, 1.0, 0.1, 10_000)
    #agent = HieraGLightAgent(2, 32, 1000, 128, 0.01, 0.9, 0.01, 1.0, 0.1, 10_000)
    agents.train(environment, episodes=100)
