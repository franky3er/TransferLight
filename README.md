# TransferLight: Domain Randomization of Deep Reinforcement Learning Environments for Zero-Shot Traffic Signal Control
This repository comprises the code and evaluation results associated with my Master Thesis: 
***Domain Randomization of Deep Reinforcement Learning Environments for Zero-Shot Traffic Signal Control***.

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

## Testing

## Demo