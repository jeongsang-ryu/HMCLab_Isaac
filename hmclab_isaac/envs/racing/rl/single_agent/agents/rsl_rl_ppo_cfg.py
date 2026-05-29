"""A2 — Asymmetric CNN-PPO config for rsl_rl >= 4.0.

Actor : ``CNNModel``  consumes ``["policy", "images"]`` → CNN(image) + MLP(proprio)
Critic: ``CNNModel``  consumes ``["policy", "images", "privileged"]`` → adds Frenet

Both encoders use the Champion-paper conv stack (32@8x8/s4 → 64@4x4/s2 →
64@3x3/s2 → 64@3x3/s1) adapted for 128x128 input. We deviate from the
champion paper's GRU module because rsl_rl 4.x has no CNN+RNN combined model
class; recurrent variant lives in the custom A1 (sync A2C) for comparison.

Algorithm hyperparameters track rsl_rl defaults with Champion-paper-inspired
adjustments (entropy 0.005, lr 3e-4, KL target 0.01).
"""
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


# Champion-paper conv stack adjusted for 128x128 → ~6x6x64 feature map
_CHAMPION_CNN_CFG = RslRlCNNModelCfg.CNNCfg(
    output_channels=[32, 64, 64, 64],
    kernel_size=[8, 4, 3, 3],
    stride=[4, 2, 2, 1],
    padding="none",
    activation="relu",
    flatten=True,
)

_GAUSSIAN_DIST = RslRlMLPModelCfg.GaussianDistributionCfg(
    init_std=0.5,
    std_type="log",
)


@configclass
class UnicornRacingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Asymmetric CNN-PPO runner for the single-agent vision racing env."""

    seed = 0
    device = "cuda:0"

    num_steps_per_env = 32        # rollout: 32 envs × 32 steps = 1024 transitions/iter
    max_iterations = 4096
    save_interval = 250
    experiment_name = "racing_ppo_asym_cnn"
    empirical_normalization = False

    # Map env obs groups → actor/critic sets (rsl_rl ≥ 4.0)
    obs_groups = {
        "actor": ["policy", "images"],
        "critic": ["policy", "images", "privileged"],
    }

    # rsl_rl ≥ 4.0 deprecated `stochastic` / `init_noise_std` / `noise_std_type`
    # / `state_dependent_std` in favour of `distribution_cfg`, BUT the IsaacLab
    # config dataclass still requires them as MISSING fields, and the runtime
    # CNNModel/MLPModel reject them as unexpected kwargs. We supply dummy
    # values here and strip them in `train_racing_ppo.py` before handing the
    # dict to `OnPolicyRunner`.
    actor = RslRlCNNModelCfg(
        class_name="CNNModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=_GAUSSIAN_DIST,
        cnn_cfg=_CHAMPION_CNN_CFG,
        stochastic=True,
        init_noise_std=0.5,
        noise_std_type="scalar",
        state_dependent_std=False,
    )

    critic = RslRlCNNModelCfg(
        class_name="CNNModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,       # critic outputs a scalar value
        cnn_cfg=_CHAMPION_CNN_CFG,
        stochastic=False,
        init_noise_std=0.0,
        noise_std_type="scalar",
        state_dependent_std=False,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        share_cnn_encoders=False,    # actor and critic each have their own CNN
        normalize_advantage_per_mini_batch=False,
    )
