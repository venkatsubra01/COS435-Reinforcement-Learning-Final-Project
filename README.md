# Goal-Conditioned Multi-Agent Cooperation with Contrastive RL
Codebase for Goal-Conditioned Multi-Agent Cooperation with Contrastive RL project.

## Setup Instructions
Install conda environment with ``conda env create -f environment.yml``
Refer to [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL) for instructions on running the single-agent algorithms and refer to [JaxMARL](https://github.com/FLAIROx/JaxMARL/tree/main) for instructions on running the multi-agent algorithms

## Our Contributions
Refer to the "submission" branch to see our contributions more clearly

## Code
Uses code from [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL) and [JaxMARL](https://github.com/FLAIROx/JaxMARL/tree/main). We extend these by creating the AntSoccerV1 and AntSoccerV2 environments. We also define a dense reward function for PPO to work on these environments, as well as create multi-agent wrappers for multi-agent RL algorithms like ICRL to work on these environments.

## Results and Further Information
Our paper and results from this project are available, titled "Benchmarking Goal-Conditioned Reinforcement Learning with Soccer Paper.pdf". The data analysis and charts shown in the paper are available under the "data_analysis" directory in the submission branch.
