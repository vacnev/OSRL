<h1 align="center">
<br>
PDT: Pareto-regularized Decision Transformer for Offline Safe Reinforcement Learning :racehorse:
</h1>

<!-- <p align="center">
Repo for "<a href="https://openreview.net/forum?id=nrRkAAAufl" target="_blank">Constraint-Conditioned Actor-Critic for Offline Safe Reinforcement Learning</a>" [ICLR 2025]
</p> -->

<!-- The official implementation of OASIS, a **Data-centric** approach for offline safe RL. -->

## Method

**PDT** is an adaptive offline safe RL method that learns the Pareto Frontier directly from offline datasets. It combines the Decision Transformer architecture with dynamic-programming critics to stitch together high-value behaviors while explicitly reasoning about the remaining cost budget at each step. PDT provides zero-shot adaptation to arbitrary constraint thresholds, avoids fragile generative components, and consistently produces safer, higher-reward policies than existing baselines across diverse benchmarks.

## Installation
We provide a docker-compose container environment for a convenient, isolated instalation. Please, first build and run the container, then install the packages inside it:

``` bash
docker-compose build
docker-compose up
bash ./install.sh
```

## Training (optional)

To train a PDT agent, simply run:
``` bash
cd OSRL
python3 -m examples.train.train_pdt --task <env_name> --param1 <args1> ... 
```
By default, the config file and the logs during training will be written to `logs\` folder and the training plots can be viewed online using Wandb.
The default parameters can be found in `OSRL/examples/configs/pdt_configs.py`.

## Evaluation

<!-- **The pre-trained models are available [here](https://drive.google.com/drive/folders/1cM7tr5My-wkzl0uxepyb99G_XESTm3TJ?usp=sharing).** -->
To evaluate a trained PDT agent, simply run:
``` bash
cd OSRL/examples/eval
python eval_pdt.py --path <path_to_model> --eval_episodes <number_of_episodes> --costs <list_of_target_cost_thresholds> --returns <list_of_target_returns>
```
It will load config file from `path_to_model/config.yaml` and model file from `path_to_model/checkpoints/model.pt`, run the number of episodes for each target return and cost threshold, and print the average normalized reward and cost.

## Github Reference
- OSRL: https://github.com/liuzuxin/osrl
- DSRL: https://github.com/liuzuxin/dsrl

<!-- ## Bibtex

If you find our code and paper can help, please consider citing our paper as:
```
@inproceedings{guoconstraint,
  title={Constraint-Conditioned Actor-Critic for Offline Safe Reinforcement Learning},
  author={Guo, Zijian and Zhou, Weichao and Wang, Shengao and Li, Wenchao},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
``` -->