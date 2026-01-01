from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple


@dataclass
class PDTTrainConfig:
    # wandb params
    project: str = "Benchmark"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "PDT"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float], ...] = None
    epsilon: float = None
    density: float = 1.0
    # model params
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    action_head_layers: int = 1
    seq_len: int = 10
    episode_len: int = 300
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    time_emb: bool = True
    num_qr: int = 2
    num_qc: int = 2
    c_hidden_sizes: Tuple[int, ...] = (256, 256)
    use_verification: bool = True
    infer_q: bool = True
    # training params
    task: str = "OfflineCarCircle-v0"
    max_cost: int = 100
    dataset: str = None
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    clip_grad_critic: Optional[float] = 1.0
    batch_size: int = 2048
    update_steps: int = 200_000
    lr_warmup_steps: int = 500
    reward_scale: float = 0.1
    cost_scale: float = 1
    num_workers: int = 6
    tau: float = 0.005
    gamma: float = 0.99
    cost_gamma: float = 0.99
    n_step: bool = False
    # evaluation params
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((450.0, 400.0, 350.0), 10), ((500.0, 450.0, 400.0), 20), ((550.0, 500.0, 450.0), 40))  # reward, cost
    cost_limit: int = 5
    eval_episodes: int = 10
    eval_every: int = 10_000
    # general params
    seed: int = 0
    device: str = "cuda:2"
    threads: int = 6
    # augmentation param
    deg: int = 4
    pf_sample: bool = False
    beta: float = 1.0
    augment_percent: float = 0.2
    # maximum absolute value of reward for the augmented trajs
    max_reward: float = 600.0
    # minimum reward above the PF curve
    min_reward: float = 1.0
    # the max drecrease of ret between the associated traj
    # w.r.t the nearest pf traj
    max_rew_decrease: float = 100.0
    # model mode params
    use_rew: bool = True
    use_cost: bool = True
    cost_transform: bool = True
    add_cost_feat: bool = False
    mul_cost_feat: bool = False
    cat_cost_feat: bool = False
    loss_cost_weight: float = 0.02
    loss_state_weight: float = 0
    cost_reverse: bool = False
    eta: float = 5.0
    max_lag: float = 5.0
    min_lag: float = -20.0
    # pf only mode param
    pf_only: bool = False
    rmin: float = 300
    cost_bins: int = 60
    npb: int = 5
    cost_sample: bool = True
    linear: bool = True  # linear or inverse
    start_sampling: bool = False
    prob: float = 0.2
    stochastic: bool = True
    init_temperature: float = 0.1
    no_entropy: bool = False
    # random augmentation
    random_aug: float = 0
    aug_rmin: float = 400
    aug_rmax: float = 500
    aug_cmin: float = -2
    aug_cmax: float = 25
    cgap: float = 5
    rstd: float = 1
    cstd: float = 0.2


@dataclass
class PDTCarCircleConfig(PDTTrainConfig):
    pass


@dataclass
class PDTAntRunConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 200
    # training params
    task: str = "OfflineAntRun-v0"
    max_cost: int = 150
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((700.0, 650.0, 600.0), 10), ((750.0, 700.0, 650.0), 20), ((800.0, 750.0, 700.0), 40))
    # augmentation param
    deg: int = 3
    max_reward: float = 1000.0
    max_rew_decrease: float = 150
    device: str = "cuda:2"


@dataclass
class PDTDroneRunConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 200
    # training params
    task: str = "OfflineDroneRun-v0"
    max_cost: int = 140
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((400.0, 350.0, 300.0), 10), ((500.0, 450.0, 400.0), 20), ((600.0, 550.0, 500.0), 40))
    # augmentation param
    deg: int = 1
    max_reward: float = 700.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class PDTDroneCircleConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 300
    eta: float = 3.0
    # training params
    task: str = "OfflineDroneCircle-v0"
    max_cost: int = 100
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((700.0, 650.0, 600.0), 10), ((750.0, 700.0, 650.0), 20), ((800.0, 750.0, 700.0), 40))
    update_steps: int = 300_000
    # augmentation param
    deg: int = 1
    max_reward: float = 1000.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class PDTCarRunConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 200
    # training params
    task: str = "OfflineCarRun-v0"
    max_cost: int = 40
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((575.0,), 10), ((575.0,), 20), ((575.0,), 40))
    update_steps: int = 100_000
    # augmentation param
    deg: int = 0
    max_reward: float = 600.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class PDTAntCircleConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 500
    eta: float = 3.0
    # training params
    task: str = "OfflineAntCircle-v0"
    max_cost: int = 200
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((300.0, 250.0, 200.0), 10), ((350.0, 300.0, 250.0), 20), ((400.0, 350.0, 300.0), 40))
    # augmentation param
    deg: int = 2
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class PDTBallRunConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 100
    # training params
    task: str = "OfflineBallRun-v0"
    max_cost: int = 80
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((500.0,), 10), ((500.0,), 20), ((700.0,), 40))
    # augmentation param
    deg: int = 2
    max_reward: float = 1400.0
    max_rew_decrease: float = 200
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class PDTBallCircleConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 200
    eta: float = 5.0
    # training params
    task: str = "OfflineBallCircle-v0"
    max_cost: int = 80
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((700.0, 650.0, 600.0), 10), ((750.0, 700.0, 650.0), 20), ((800.0, 750.0, 700.0), 40))
    # augmentation param
    deg: int = 2
    max_reward: float = 1000.0
    max_rew_decrease: float = 200
    min_reward: float = 1
    device: str = "cuda:1"


@dataclass
class PDTCarButton1Config(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    eta: float = 1.0
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    max_cost: int = 250
    target_returns: Tuple[Tuple[Tuple[float, ...], float], ...] = (((20.0, 15.0, 10.0), 20), ((20.0, 15.0, 10.0), 40), ((20.0, 15.0, 10.0), 80), ((20.0, 15.0, 10.0), 120))
    # augmentation param
    deg: int = 0
    max_reward: float = 45.0
    max_rew_decrease: float = 10
    min_reward: float = 1
    device: str = "cuda:0"


@dataclass
class PDTCarButton2Config(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    eta: float = 8.0
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    max_cost: int = 300
    target_returns: Tuple[Tuple[Tuple[float, ...], float], ...] = (((20.0, 15.0, 10.0), 20), ((20.0, 15.0, 10.0), 40), ((20.0, 15.0, 10.0), 80), ((20.0, 15.0, 10.0), 120))
    # augmentation param
    deg: int = 0
    max_reward: float = 50.0
    max_rew_decrease: float = 10
    min_reward: float = 1
    device: str = "cuda:0"


@dataclass
class PDTCarCircle1Config(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 500
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    max_cost: int = 250
    target_returns: Tuple[Tuple[Tuple[float, ...], float], ...] = (((20.0,), 20), ((22.5,), 40), ((25.0,), 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 30.0
    max_rew_decrease: float = 10
    min_reward: float = 1
    device: str = "cuda:0"


@dataclass
class PDTCarCircle2Config(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 500
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    max_cost: int = 400
    target_returns: Tuple[Tuple[Tuple[float, ...], float], ...] = (((20.0,), 20), ((21.0,), 40), ((22.0,), 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 30.0
    max_rew_decrease: float = 10
    min_reward: float = 1
    device: str = "cuda:0"


@dataclass
class PDTCarGoal1Config(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    eta: float = 1.0
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    max_cost: int = 120
    target_returns: Tuple[Tuple[Tuple[float, ...], float], ...] = (((40.0, 35.0, 25.0), 20), ((40.0, 35.0, 25.0), 40), ((40.0, 35.0, 25.0), 80), ((40.0, 35.0, 25.0), 120))
    # augmentation param
    deg: int = 1
    max_reward: float = 50.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:1"


@dataclass
class PDTCarGoal2Config(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    eta: float = 1.0
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    max_cost: int = 200
    target_returns: Tuple[Tuple[Tuple[float, ...], float], ...] = (((30.0, 25.0, 20.0), 20), ((30.0, 25.0, 20.0), 40), ((30.0, 25.0, 20.0), 80), ((30.0, 25.0, 20.0), 120))
    # augmentation param
    deg: int = 1
    max_reward: float = 35.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:1"


@dataclass
class PDTCarPush1Config(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    eta: float = 1.0
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    max_cost: int = 200
    target_returns: Tuple[Tuple[Tuple[float, ...], float], ...] = (((15.0, 12.0, 10.0), 20), ((15.0, 12.0, 10.0), 40), ((15.0, 12.0, 10.0), 80), ((15.0, 12.0, 10.0), 120))
    # augmentation param
    deg: int = 0
    max_reward: float = 20.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:1"


@dataclass
class PDTCarPush2Config(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    eta: float = 8.0
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    max_cost: int = 250
    target_returns: Tuple[Tuple[Tuple[float, ...], float], ...] = (((12.0, 10.0, 8.0), 20), ((12.0, 10.0, 8.0), 40), ((12.0, 10.0, 8.0), 80), ((12.0, 10.0, 8.0), 120))
    # augmentation param
    deg: int = 0
    max_reward: float = 15.0
    max_rew_decrease: float = 3
    min_reward: float = 1
    device: str = "cuda:1"


@dataclass
class PDTPointButton1Config(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    max_cost: int = 200
    target_returns: Tuple[Tuple[Tuple[float, ...], float], ...] = (((40.0,), 20), ((40.0,), 40), ((40.0,), 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 45.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class PDTPointButton2Config(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    max_cost: int = 250
    target_returns: Tuple[Tuple[Tuple[float, ...], float], ...] = (((40.0,), 20), ((40.0,), 40), ((40.0,), 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 50.0
    max_rew_decrease: float = 10
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class PDTPointCircle1Config(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 500
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    max_cost: int = 200
    target_returns: Tuple[Tuple[Tuple[float, ...], float], ...] = (((50.0,), 20), ((52.5,), 40), ((55.0,), 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 65.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class PDTPointCircle2Config(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 500
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    max_cost: int = 300
    target_returns: Tuple[Tuple[Tuple[float, ...], float], ...] = (((45.0,), 20), ((47.5,), 40), ((50.0,), 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 55.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class PDTPointGoal1Config(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    max_cost: int = 100
    target_returns: Tuple[Tuple[Tuple[float, ...], float], ...] = (((30.0,), 20), ((30.0,), 40), ((30.0,), 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 35.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class PDTPointGoal2Config(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    max_cost: int = 200
    target_returns: Tuple[Tuple[Tuple[float, ...], float], ...] = (((30.0,), 20), ((30.0,), 40), ((30.0,), 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 35.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class PDTPointPush1Config(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    max_cost: int = 150
    target_returns: Tuple[Tuple[Tuple[float, ...], float], ...] = (((15.0,), 20), ((15.0,), 40), ((15.0,), 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 20.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class PDTPointPush2Config(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    max_cost: int = 200
    target_returns: Tuple[Tuple[Tuple[float, ...], float], ...] = (((12.0,), 20), ((12.0,), 40), ((12.0,), 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 15.0
    max_rew_decrease: float = 3
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class PDTAntVelocityConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    max_cost: int = 250
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((2800.0,), 10), ((2800.0,), 20))
    # augmentation param
    deg: int = 1
    max_reward: float = 3000.0
    max_rew_decrease: float = 500
    min_reward: float = 1
    device: str = "cuda:1"


@dataclass
class PDTHalfCheetahVelocityConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    eta: float = 5.0
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    max_cost: int = 250
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((3000.0, 2800.0, 2600.0), 20), ((3000.0, 2800.0, 2600.0), 40), ((3000.0, 2800.0, 2600.0), 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 3000.0
    max_rew_decrease: float = 500
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class PDTHopperVelocityConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    eta: float = 1.0
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    max_cost: int = 250
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((2000.0, 1750.0, 1500.0), 20), ((2000.0, 1750.0, 1500.0), 40), ((2000.0, 1750.0, 1500.0), 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 2000.0
    max_rew_decrease: float = 300
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class PDTSwimmerVelocityConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    eta: float = 1.0
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    max_cost: int = 200
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((200.0, 180.0, 160.0), 20), ((200.0, 180.0, 160.0), 40), ((200.0, 180.0, 160.0), 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 250.0
    max_rew_decrease: float = 50
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class PDTWalker2dVelocityConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    max_cost: int = 300
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((2800.0,), 10), ((2800.0,), 20))
    # augmentation param
    deg: int = 1
    max_reward: float = 3600.0
    max_rew_decrease: float = 800
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class PDTEasySparseConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    max_cost: int = 85
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((300.0,), 10), ((350.0,), 20), ((400.0,), 40))
    # augmentation param
    deg: int = 2
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class PDTEasyMeanConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    max_cost: int = 85
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((300.0,), 10), ((350.0,), 20), ((400.0,), 40))
    # augmentation param
    deg: int = 2
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class PDTEasyDenseConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    max_cost: int = 85
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((300.0,), 10), ((350.0,), 20), ((400.0,), 40))
    # augmentation param
    deg: int = 2
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class PDTMediumSparseConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    max_cost: int = 50
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((300.0,), 10), ((300.0,), 20), ((300.0,), 40))
    # augmentation param
    deg: int = 0
    max_reward: float = 300.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class PDTMediumMeanConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    max_cost: int = 50
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((300.0,), 10), ((300.0,), 20), ((300.0,), 40))
    # augmentation param
    deg: int = 0
    max_reward: float = 300.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class PDTMediumDenseConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    max_cost: int = 50
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((300.0,), 10), ((300.0,), 20), ((300.0,), 40))
    # augmentation param
    deg: int = 0
    max_reward: float = 300.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class PDTHardSparseConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    max_cost: int = 85
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((300.0,), 10), ((350.0,), 20), ((400.0,), 40))
    # augmentation param
    deg: int = 1
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class PDTHardMeanConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    max_cost: int = 85
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((300.0,), 10), ((350.0,), 20), ((400.0,), 40))
    # augmentation param
    deg: int = 1
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class PDTHardDenseConfig(PDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    max_cost: int = 85
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[Tuple[float, ...], float],
                          ...] = (((300.0,), 10), ((350.0,), 20), ((400.0,), 40))
    # augmentation param
    deg: int = 1
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


PDT_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": PDTCarCircleConfig,
    "OfflineAntRun-v0": PDTAntRunConfig,
    "OfflineDroneRun-v0": PDTDroneRunConfig,
    "OfflineDroneCircle-v0": PDTDroneCircleConfig,
    "OfflineCarRun-v0": PDTCarRunConfig,
    "OfflineAntCircle-v0": PDTAntCircleConfig,
    "OfflineBallCircle-v0": PDTBallCircleConfig,
    "OfflineBallRun-v0": PDTBallRunConfig,
    # safety_gymnasium
    "OfflineCarButton1Gymnasium-v0": PDTCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": PDTCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": PDTCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": PDTCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": PDTCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": PDTCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": PDTCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": PDTCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": PDTPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": PDTPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": PDTPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": PDTPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": PDTPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": PDTPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": PDTPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": PDTPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": PDTAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": PDTHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": PDTHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": PDTSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": PDTWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": PDTEasySparseConfig,
    "OfflineMetadrive-easymean-v0": PDTEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": PDTEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": PDTMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": PDTMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": PDTMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": PDTHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": PDTHardMeanConfig,
    "OfflineMetadrive-harddense-v0": PDTHardDenseConfig
}