import os

from src.params import SCENARIOS_ROOT
from src.rl.environments import MultiprocessingTscMarlEnvironment, TscMarlEnvironment
from src.rl.agents import MaxPressureAgents, IQLAgents, A2C, DQN

if __name__ == "__main__":

    scenarios_dir = os.path.join(SCENARIOS_ROOT, "isolated", "train")
    traffic_representation = "GeneraLightTrafficRepresentation"
    #environment = TscMarlEnvironment(scenarios_dir, 180, "MaxPressureTrafficRepresentation", use_default=False, demo=False)
    environment = MultiprocessingTscMarlEnvironment(scenarios_dir, 180, traffic_representation, 64)

    input_dim = 10
    hidden_dim = 64

    model_params = {"state_size": 16, "hidden_size": 64, "action_size": 4}
    train_params = {"buffer_size": 1_000, "batch_size": 128, "learning_rate": 0.01, "discount_factor": 0.9,
                    "tau": 0.01, "eps_greedy_start": 1.0, "eps_greedy_end": 0.1, "eps_greedy_steps": 10_000}

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
    dqn_head = {
        "class_name": "LinearDuelingHead",
        "init_args": {
            "input_dim": hidden_dim,
            "hidden_dim": hidden_dim,
            "value_residual_stack_sizes": (2, 2),
            "advantage_residual_stack_size": 2,
            "aggr_fn": "sum"
        }
    }

    #agents = RandomAgents()
    #agents = MaxPressureAgents(20)
    #agents = IQLAgents(model_params, 1000, 128, 0.01, 0.9, 0.01, 1.0, 0.1, 10_000, checkpoint_dir=os.path.join("agents", "IQL"), save_checkpoint=True)
    #agents = A2C(network, actor_head, critic_head, share_network=True, n_steps=1, gae_discount_factor=0.0, checkpoint_dir=os.path.join("agents", "A2C"), save_checkpoint=True)
    agents = DQN(network, dqn_head, discount_factor=0.9, batch_size=64, replay_buffer_size=10_000, learning_rate=0.01, eps_greedy_start=1.0, eps_greedy_end=0.1, eps_greedy_steps=10_000, tau=0.01, checkpoint_dir=os.path.join("agents", "DQN"))
    #agents = IA2CAgents(actor_heterogeneous, critic_heterogeneous, 0.001, 0.001, 0.9, 0.01, checkpoint_dir=os.path.join("agents", "IA2C"), save_checkpoint=True)
    #agent = LitAgent(16, 64, 4, 1000, 128, 0.01, 0.99, 0.01, 1.0, 0.1, 10_000)
    #agent = HieraGLightAgent(2, 32, 1000, 128, 0.01, 0.9, 0.01, 1.0, 0.1, 10_000)
    agents.train_env(environment, episodes=100)
