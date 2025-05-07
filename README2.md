## COS 435 Final Project Soccer: Anthony, Sai, Venkat

## Setup Instructions
Install conda environment with ``conda env create -f environment.yml``

## Code
Built on top of code from https://anonymous.4open.science/r/gcrl_marl/README.md, which uses code from [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL) and [JaxMARL](https://github.com/FLAIROx/JaxMARL/tree/main).

## New/Edited Files (Our Technical Contributions)

Extensions of jaxgcrl/envs/assets/ant_ball.xml
- brax/envs/assets/ant_soccer.xml
- brax/envs/assets/ant_soccer_v2.xml
    - These contain the obstacle, boundaries, etc

Extensions of jaxgcrl/envs/ant_ball.py
- brax/envs/antsoccer
- brax/envs/antsoccerv2
    - These were added to a local copy of brax to allow for multi agent wrappers to work on our new environments
    - These two new files are identical except for xml assets

brax/envs/__init__.py
- added our new envs to a local clone of brax

Single agent environments that we used to train PPO, CRL - extension of jaxgcrl/envs/ant_ball.py
- envs/ant_soccer.py (with our new xml assets and some slight modifications to target location generation)

Multi agent wrappers that we used to train ICRL - significant extension of envs/mabrax_ant.py
- envs/mabrax_ant_soccer.py
- envs/mabrax_ant_soccerv2.py
    - These 2 files wrap brax/envs/antsoccer and brax/envs/antsoccerv2 to allow for training multi agent algorithms on our new environments

More multi agent wrapping functionality
- jaxmarl/environments/mabrax/mabrax_env.py
- jaxmarl/environments/mabrax/mappings.py
    - Extended multi agent mappings and environment initialization from baseline Ant environment to also work for AntSoccerV1 and AntSoccerV2

Small Bug Fixes
- evaluator.py
    - Minor fix with training error due to incorrect number of agents