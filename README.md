<h1 align="center">
<br>
PDT: Pareto Decision Transformer for Offline Safe Reinforcement Learning :racehorse:
</h1>

## Method

**PDT** is an adaptive offline safe RL method that learns the cost-reward Pareto Frontier directly from offline datasets. It combines the Decision Transformer architecture with dynamic-programming critics to stitch together high-value behaviors while explicitly reasoning about the remaining cost budget at each step. PDT provides zero-shot adaptation to arbitrary constraint thresholds, avoids fragile generative components, and consistently produces safer, higher-reward policies than existing baselines across diverse benchmarks.

## Installation
We provide a docker-compose container environment for a convenient, isolated instalation. Please, first build and run the container, then install the packages inside it:

``` bash
docker-compose up -d --build
docker exec -it OSRL bash
bash ./install.sh
```

## Training

To train a PDT agent, simply run:
``` bash
cd OSRL
python3 -m examples.train.train_pdt --device cuda --task <env_name> --param1 <args1> ... 
```
By default, the config file and the logs during training will be written to `logs/` folder and the training plots can be viewed online using Wandb.
The default parameters can be found in `OSRL/examples/configs/pdt_configs.py`.

## Evaluation

To evaluate a trained PDT agent, simply run:
``` bash
cd OSRL/examples/eval
python3 eval_pdt.py --path <path_to_model> --eval_episodes <number_of_episodes> --costs <list_of_target_cost_thresholds> --returns <list_of_lists_of_target_returns>
```
It will load config file from `path_to_model/config.yaml` and model file from `path_to_model/checkpoints/model.pt`, run the number of episodes for each target return and cost threshold, and print the average normalized reward and cost.

## Reproducibility

To train all PDT models that were evaluated in the paper you can run Wandb sweeps, one for BulletGym and one for SafetyGymnasium:
``` bash
cd OSRL
wandb sweep examples/train/pdt_bullet_sweep.yaml
wandb sweep examples/train/pdt_gymnasium_sweep.yaml
```

The trained model will be saved into the `logs/` folder. To evaluate all of the trained models you can simply run:
``` bash
python3 examples/eval/eval_adap.py --algo_name pdt --envs all
```
The results from the evaluation will be saved into the `results/` folder and will correspond to the results reported in the paper.

## Github Reference
- OSRL: https://github.com/liuzuxin/osrl
- DSRL: https://github.com/liuzuxin/dsrl
