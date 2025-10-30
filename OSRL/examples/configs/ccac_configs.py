from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field


@dataclass
class CCACTrainConfig:
    # wandb params
    project: str = "Benchmark"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "CCAC"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float, float, float], ...] = None
    epsilon: float = None
    density: float = 1.0
    # training params
    task: str = "OfflineCarCircle-v0"
    dataset: str = None
    seed: int = 0
    device: str = "cpu"
    threads: int = 4
    reward_scale: float = 0.1
    cost_scale: float = 1
    actor_lr: float = 0.0001
    critic_lr: float = 0.001
    vae_lr: float = 0.001
    cost_conditioned: bool = True
    cost_limit: int = 10
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 200_000
    num_workers: int = 8
    # model params
    a_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    c_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    vae_hidden_sizes: int = 512
    vae_latent_sizes: int = 64
    gamma: float = 0.99
    tau: float = 0.005
    beta: float = 0.5
    num_q: int = 4
    num_qc: int = 4
    ood_threshold: float = 0.5
    max_lag: float = 5.0
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 10_000


@dataclass
class CCACCarCircleConfig(CCACTrainConfig):
    pass


@dataclass
class CCACAntRunConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class CCACDroneRunConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class CCACDroneCircleConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300
    update_steps: int = 300_000


@dataclass
class CCACCarRunConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200
    batch_size: int = 1024
    update_steps: int = 100_000


@dataclass
class CCACAntCircleConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class CCACBallRunConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100
    

@dataclass
class CCACBallCircleConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200


@dataclass
class CCACCarButton1Config(CCACTrainConfig):
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CCACCarButton2Config(CCACTrainConfig):
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CCACCarCircle1Config(CCACTrainConfig):
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class CCACCarCircle2Config(CCACTrainConfig):
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class CCACCarGoal1Config(CCACTrainConfig):
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CCACCarGoal2Config(CCACTrainConfig):
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CCACCarPush1Config(CCACTrainConfig):
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CCACCarPush2Config(CCACTrainConfig):
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CCACPointButton1Config(CCACTrainConfig):
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CCACPointButton2Config(CCACTrainConfig):
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CCACPointCircle1Config(CCACTrainConfig):
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class CCACPointCircle2Config(CCACTrainConfig):
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class CCACPointGoal1Config(CCACTrainConfig):
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CCACPointGoal2Config(CCACTrainConfig):
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CCACPointPush1Config(CCACTrainConfig):
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CCACPointPush2Config(CCACTrainConfig):
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CCACAntVelocityConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CCACHalfCheetahVelocityConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CCACHopperVelocityConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CCACSwimmerVelocityConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000
    reward_scale: float = 0.01


@dataclass
class CCACWalker2dVelocityConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CCACEasySparseConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CCACEasyMeanConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CCACEasyDenseConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CCACMediumSparseConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CCACMediumMeanConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CCACMediumDenseConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CCACHardSparseConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CCACHardMeanConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CCACHardDenseConfig(CCACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


CCAC_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": CCACCarCircleConfig,
    "OfflineAntRun-v0": CCACAntRunConfig,
    "OfflineDroneRun-v0": CCACDroneRunConfig,
    "OfflineDroneCircle-v0": CCACDroneCircleConfig,
    "OfflineCarRun-v0": CCACCarRunConfig,
    "OfflineAntCircle-v0": CCACAntCircleConfig,
    "OfflineBallCircle-v0": CCACBallCircleConfig,
    "OfflineBallRun-v0": CCACBallRunConfig,
    # safety_gymnasium
    "OfflineCarButton1Gymnasium-v0": CCACCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": CCACCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": CCACCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": CCACCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": CCACCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": CCACCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": CCACCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": CCACCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": CCACPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": CCACPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": CCACPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": CCACPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": CCACPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": CCACPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": CCACPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": CCACPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": CCACAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": CCACHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": CCACHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": CCACSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": CCACWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": CCACEasySparseConfig,
    "OfflineMetadrive-easymean-v0": CCACEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": CCACEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": CCACMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": CCACMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": CCACMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": CCACHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": CCACHardMeanConfig,
    "OfflineMetadrive-harddense-v0": CCACHardDenseConfig
}