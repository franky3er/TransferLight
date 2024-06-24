# TransferLight: Domain Randomization of Deep Reinforcement Learning Environments for Zero-Shot Traffic Signal Control
This repository comprises the code and evaluation results associated with my Master Thesis: 
***Domain Randomization of Deep Reinforcement Learning Environments for Zero-Shot Traffic Signal Control***.

| Fixed Time (conventional)      | TransferLight (ours) |
|--------------------------------| ---------- |
| <img src="gifs/FixedTime.gif"> | <img src="gifs/TransferLight.gif"> |

***Abstract***

The escalating traffic demand in urban areas necessitates intelligent transportation engineering solutions that can regulate traffic more efficiently. The optimization of traffic signal control is considered a crucial factor in this regard. In recent years, deep reinforcement learning has gained traction as a potential solution concept to learn an adaptive signal controller, with promising initial results demonstrating its superiority over conventional solutions. However, the practical utility of such solutions is often constrained by a rigid model representation for the signal controller and a limited representation of traffic scenarios during the training phase. As a result, these methods severely overfit to the idiosyncrasies of the traffic environment used during training and fail to adapt to changing traffic conditions. To enhance adaptability, this thesis introduces TransferLight, a model architecture that employs neural message passing on a graph-structured state representation of an intersection to predict its next phase. This approach facilitates adaptability to arbitrarily structured intersections and is implemented for both a Q-learning and an advantage actor-critic algorithm. Additionally, a domain randomization mechanism is proposed, perturbing specific parameters associated with the road network and traffic within the environment during each training episode. This deliberate randomization aims to enhance the variability of traffic conditions encountered during the training phase. The efficacy of the proposed techniques is assessed by comparing them to established baseline solutions derived from conventional transportation engineering theories and reinforcement learning on both synthetically generated and real-world test scenarios. The findings reveal that TransferLight exhibits pronounced overfitting when trained on identical scenarios in each episode. Conversely, when the proposed domain randomization mechanism is leveraged to train TransferLight, generalization markedly improves and enables a zero-shot transfer to the test scenarios with a more consistent and robust performance, often being superior to the benchmark solutions, with respect to average travel time and throughput. However, upon closer examination within the context of an arterial road network scenario, the behavior of the trained model does not consistently align well with  solutions specifically calibrated for this scenario. The latter exhibit improved signal progression, necessitating fewer stops for vehicles to traverse the arterial roadway. An estimation of model uncertainty using Monte Carlo dropout further indicates that despite the application of domain randomization during training, TransferLight's confidence in its predictions is not consistently high across traffic conditions, as the uncertainty distribution for some scenarios show a noticeable shift or spread towards higher uncertainties. These observations imply that although the proposed domain randomization mechanism enhances the transfer performance of TransferLight to a wide range of traffic conditions, it does not implicitly equip the model with the necessary knowledge to act optimally and with a high degree of certainty on a consistent basis. 

## Setup
To set up the environment follow the following steps: 
1. [Install SUMO](https://sumo.dlr.de/docs/Installing/index.html)
2. [Set environment variable `SUMO_HOME`](https://sumo.dlr.de/docs/Basics/Basic_Computer_Skills.html#sumo_home)
3. [Install anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)
4. Clone repository: <br>`git clone https://github.com/franky3er/TransferLight.git`
5. Change directory: <br>`cd TransferLight`
6. Create conda environment: <br>`conda env create -f environment.yml`
7. [Fix libstdc++ problem](https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris): <br>`cd <PATH_TO_ANACONDA>/envs/TransferLight/lib`<br>`mkdir backup`<br>`mv libstd* backup`<br>`cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6  ./`<br>`ln -s libstdc++.so.6 libstdc++.so`<br>`ln -s libstdc++.so.6 libstdc++.so.6.0.19`<br> where `<PATH_TO_ANACONDA>` is the path to the directory where anaconda is installed
8. Change to the directory where the repository was cloned
9. Activate environment: <br>`conda activate TransferLight`
10. Set up training and test scenarios: <br>`python main.py setup`

## Training
Models can be trained using the following command:<br> 
`python main.py train -a <AGENT> -d <DEVICE>`<br>
- `<AGENT>` is a placeholder for the name of the agent(s) to train (multiple agents need to be separated by spaces)
- `<DEVICE>` is a placeholder for the device name that shall we used for training (e.g., `cuda:0` / `cpu`).<br>

This command will store a model checkpoint after every 100th training step under `results/<AGENT>/checkpoints/`.

For more information run the following help command:<br>
`python main.py train -h`<br>
This command will also display a list of possible agents that can be trained.

**Note**: All relevant models are already trained and the corresponding checkpoints are available in this repository.

## Testing

Models can be tested with the following command: <br>
`python main.py test -a <AGENT> -c <CHECKPOINT> -s <SCENARIO> -d <DEVICE>`
- `<AGENT` is a placeholder for the name of the agent(s) to test (multiple agents need to be separated by spaces). `ALL` will select all available agents.
- `<CHECKPOINT>` is a placeholder for the path(s) to the model checkpoint(s) that shall be used for testing (multiple checkpoints need to be separated by spaces). `ALL` will select all available checkpoints for the specified agent(s). `BEST` will select the checkpoint file 'best.pt' in the agent's checkpoint directory (file needs to be created in advance).
- `<SCENARIO>` is a placeholder for the scenario(s) to test the specified agent(s) on (multiple scenarios need to be separated by spaces). `ALL` will select all available scenarios. `AGENT` will only select the scenarios the agent was trained on. 
- `<DEVICE>` is a placeholder for the device name that shall we used for testing (e.g., `cuda:0` / `cpu`).<br>

This command will generate csv files for each `<AGENT>`-`<checkpoint>`-`<scenario>` combination which are stored under directory `results/<agent>/<checkpoint>/<SCENARIO>/`. These csv files comprise statistics about individual intersections (ending `intersection.csv`), individual vehicles (ending `vehicle.csv`) and model uncertainties (ending `agent.csv`).

For more information run the following help command:<br>
`python main.py test -h`<br>
This command will also display a list of available agents and scenarios.

**Note**: All relevant csv files were already generated and are available in this repository. A jupyter notebook comprising plots demonstrating the evaluation results is available under `notebooks/evaluation-results.ipynb`.

## Demo

A demo showcasing the effectiveness of a particular model can be run with the following command: <br>
`python main.py demo -a <AGENT> -c <CHECKPOINT> -s <SCENARIO> -d <DEVICE>`
- `<AGENT` is a placeholder for the name of the agent.
- `<CHECKPOINT>` is a placeholder for the path to the model checkpoint.
- `<SCENARIO>` is a placeholder for the path to the scenario (i.e., the `.sumocfg` file) to run the demo on 
- `<DEVICE>` is a placeholder for the device name that shall we used for the demo (e.g., `cuda:0` / `cpu`).<br>

This command will open the sumo-gui. 

For more information run the following help command:<br>
`python main.py demo -h`<br>
This command will also display a list of available agents.

