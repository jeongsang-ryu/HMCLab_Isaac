"""A1 — Sync A2C training (A3C paper §5.2 / §8 faithful).

Single-machine synchronous descendant of A3C:
* On-policy n-step advantage actor-critic
* Shared conv encoder + optional LSTM-256 (symmetric — A3C paper spirit)
* Gaussian continuous-action policy with state-independent ``log_std``
  (paper uses (μ, σ²) via SoftPlus, but log-std parameterization is more
  numerically stable and equivalent in expectation)
* RMSProp, γ=0.99, entropy β=0.01, gradient norm clip 40

Parallelism comes from Isaac Lab's N parallel envs in lockstep rather than
A3C's async threads — the decorrelation effect is the same.

The env returns a dict; A2C reads only ``{"policy", "images"}`` (no privileged
critic obs, faithful to the symmetric A3C paper design).

Usage (from a training script after AppLauncher boot)::

    from hmclab_isaac.algos.sync_a2c import A2CAgent, A2CConfig, train
    agent = A2CAgent(env, A2CConfig())
    train(env, agent, A2CConfig())
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class A2CConfig:
    """Hyperparameters mirroring A3C paper §5.2 / §8 with sync adaptation."""

    # rollout / optim
    t_max: int = 5              # n-step bootstrap horizon
    max_iterations: int = 4096
    gamma: float = 0.99
    entropy_coef: float = 0.001     # lower than paper's 0.01 — our reward
                                    # magnitude is large (terminal ±50) so a
                                    # 0.01 entropy bonus dominates and lets
                                    # log_std balloon. 0.001 keeps exploration
                                    # alive without competing with reward.
    value_coef: float = 0.5
    max_grad_norm: float = 40.0
    lr: float = 3e-4                # 7e-4 from paper destabilizes here; image
                                    # obs needs more conservative steps
    rmsprop_alpha: float = 0.99
    rmsprop_eps: float = 1e-1

    # architecture
    use_lstm: bool = True           # paper's "A3C, LSTM" variant
    lstm_hidden: int = 256
    cnn_out_dim: int = 256
    init_log_std: float = -0.5
    log_std_min: float = -2.0       # clamp learned log_std (σ ∈ [0.135, 2.72])
    log_std_max: float = 1.0

    # logging / checkpoints
    log_interval: int = 10
    save_interval: int = 200
    experiment_name: str = "racing_a2c"
    log_dir: str = "/tmp/hmclab_runs"
    seed: int = 0


DEFAULT_A2C_CONFIG = A2CConfig()


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class A3CActorCritic(nn.Module):
    """A3C paper TORCS arch: Conv 16@8x8/s4 → Conv 32@4x4/s2 → FC256 → LSTM → heads.

    Backbone is shared between actor and critic (symmetric — A3C paper). The
    LSTM is optional; without it the model is the "A3C, FF" variant from
    paper Table 1.
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        proprio_dim: int,
        action_dim: int,
        cfg: A2CConfig,
    ):
        super().__init__()
        c, h, w = image_shape  # channels-first (3, 128, 128)
        # Paper §8 says 16ch@8x8/s4 → 32ch@4x4/s2; for 128x128 inputs this
        # leaves a 14x14 spatial map (Atari 84x84 left 9x9).
        self.conv1 = nn.Conv2d(c, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # Compute conv output spatial size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            feat = F.relu(self.conv1(dummy))
            feat = F.relu(self.conv2(feat))
            self.conv_flat_dim = int(feat.flatten(1).shape[1])
        # FC 256
        self.fc = nn.Linear(self.conv_flat_dim + proprio_dim, cfg.cnn_out_dim)

        # Recurrent
        self.use_lstm = cfg.use_lstm
        if self.use_lstm:
            self.lstm = nn.LSTMCell(cfg.cnn_out_dim, cfg.lstm_hidden)
            head_in = cfg.lstm_hidden
        else:
            head_in = cfg.cnn_out_dim

        # Heads
        self.policy_mean = nn.Linear(head_in, action_dim)
        self.value = nn.Linear(head_in, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), cfg.init_log_std))
        self.log_std_min = cfg.log_std_min
        self.log_std_max = cfg.log_std_max

        # Init: small policy weights so initial action is near 0
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.zeros_(self.policy_mean.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def encode(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        x = torch.cat([x, proprio], dim=-1)
        x = F.relu(self.fc(x))
        return x

    def forward(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """Return (mean, std, value, new_hidden)."""
        feat = self.encode(image, proprio)
        if self.use_lstm:
            if hidden is None:
                h = torch.zeros(image.shape[0], self.lstm.hidden_size, device=image.device)
                c = torch.zeros_like(h)
            else:
                h, c = hidden
            h, c = self.lstm(feat, (h, c))
            head = h
            new_hidden = (h, c)
        else:
            head = feat
            new_hidden = None
        mean = self.policy_mean(head)
        # Clamp learned log_std so a free-entropy-bonus exploit cannot let it
        # diverge to ~14 (which we observed in early training: H ≈ 30,
        # σ ≈ exp(14) ≈ 1M, policy becomes pure noise even after clipping).
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max).expand_as(mean)
        std = log_std.exp()
        value = self.value(head).squeeze(-1)
        return mean, std, value, new_hidden


def _gaussian_log_prob(mean: torch.Tensor, std: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    var = std * std
    log_density = -((x - mean) ** 2) / (2.0 * var) - std.log() - 0.5 * float(torch.log(torch.tensor(2 * 3.14159265)))
    return log_density.sum(dim=-1)


def _gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
    return (std.log() + 0.5 * float(torch.log(torch.tensor(2 * 3.14159265 * 2.71828)))).sum(dim=-1)


# ---------------------------------------------------------------------------
# Agent wrapper
# ---------------------------------------------------------------------------
class A2CAgent:
    """Stateful actor-critic + RMSProp optimizer + LSTM hidden state."""

    def __init__(self, env, cfg: A2CConfig):
        self.cfg = cfg
        self.env = env
        self.device = env.device
        # Probe a sample obs to discover shapes
        obs, _ = env.reset()
        img_shape = tuple(obs["images"].shape[1:])    # (C, H, W) for channel-first
        proprio_dim = int(obs["policy"].shape[1])
        # env.action_space is BATCHED (shape (num_envs, action_dim)) in
        # DirectRLEnv; single_action_space is per-env.
        action_dim = int(env.single_action_space.shape[0])
        self.net = A3CActorCritic(img_shape, proprio_dim, action_dim, cfg).to(self.device)
        self.opt = torch.optim.RMSprop(
            self.net.parameters(),
            lr=cfg.lr,
            alpha=cfg.rmsprop_alpha,
            eps=cfg.rmsprop_eps,
        )
        E = env.num_envs
        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        if cfg.use_lstm:
            h = torch.zeros(E, cfg.lstm_hidden, device=self.device)
            c = torch.zeros_like(h)
            self.hidden = (h, c)

    def reset_hidden(self, dones: torch.Tensor):
        if self.hidden is None:
            return
        mask = (~dones).float().unsqueeze(-1)
        h, c = self.hidden
        self.hidden = (h * mask, c * mask)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(env, agent: A2CAgent, cfg: A2CConfig, start_iter: int = 0) -> str:
    """Run synchronous A2C on the given Isaac Lab vec-env. Returns log dir.

    ``start_iter`` lets a resumed run continue iteration numbering from where
    the loaded checkpoint left off; it does NOT skip rollout / training.
    """
    log_dir = os.path.join(cfg.log_dir, cfg.experiment_name, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[a2c] log_dir = {log_dir}", flush=True)
    if start_iter > 0:
        print(f"[a2c] resuming from iter {start_iter}", flush=True)

    torch.manual_seed(cfg.seed)
    obs, _ = env.reset()
    ep_returns = torch.zeros(env.num_envs, device=agent.device)
    ep_lengths = torch.zeros(env.num_envs, device=agent.device)
    finished_returns: list[float] = []

    for it in range(start_iter, cfg.max_iterations):
        # ---- Rollout (t_max steps) ----
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

        for t in range(cfg.t_max):
            image = obs["images"]
            proprio = obs["policy"]
            mean, std, value, agent.hidden = agent.net(image, proprio, agent.hidden)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            # Step env — clip to [-1, 1] is also done env-side
            obs, reward, terminated, truncated, _ = env.step(action.clamp(-1.0, 1.0))
            done = terminated | truncated

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done.float())
            entropies.append(entropy)

            ep_returns += reward
            ep_lengths += 1.0
            for env_idx in done.nonzero(as_tuple=False).flatten().tolist():
                finished_returns.append(float(ep_returns[env_idx].item()))
                ep_returns[env_idx] = 0.0
                ep_lengths[env_idx] = 0.0
            agent.reset_hidden(done)

        # ---- Bootstrap from final state ----
        with torch.no_grad():
            _, _, last_value, _ = agent.net(obs["images"], obs["policy"], agent.hidden)
        returns: list[torch.Tensor] = [None] * cfg.t_max  # type: ignore
        R = last_value
        for t in reversed(range(cfg.t_max)):
            R = rewards[t] + cfg.gamma * R * (1.0 - dones[t])
            returns[t] = R

        # Stack to (T, E)
        returns_t = torch.stack(returns)               # (T, E)
        values_t = torch.stack(values)                 # (T, E)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)
        advantages = (returns_t - values_t).detach()

        # ---- Losses ----
        policy_loss = -(log_probs_t * advantages).mean()
        value_loss = F.mse_loss(values_t, returns_t)
        entropy_loss = -entropies_t.mean()

        loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

        agent.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.net.parameters(), cfg.max_grad_norm)
        agent.opt.step()
        # detach LSTM state to break the graph between iterations
        if agent.hidden is not None:
            agent.hidden = (agent.hidden[0].detach(), agent.hidden[1].detach())

        # ---- Logging ----
        if it % cfg.log_interval == 0:
            mean_ret = (
                sum(finished_returns[-50:]) / max(1, len(finished_returns[-50:]))
                if finished_returns else 0.0
            )
            print(
                f"[a2c] iter {it:5d}  "
                f"loss={float(loss.item()):+.3f}  "
                f"pi={float(policy_loss.item()):+.3f}  "
                f"v={float(value_loss.item()):+.3f}  "
                f"H={float(-entropy_loss.item()):+.3f}  "
                f"ret(50)={mean_ret:+.2f}  "
                f"n_ep={len(finished_returns)}",
                flush=True,
            )
        if it > 0 and it % cfg.save_interval == 0:
            torch.save(
                {"net": agent.net.state_dict(), "opt": agent.opt.state_dict(), "iter": it},
                os.path.join(log_dir, f"a2c_iter{it:06d}.pt"),
            )

    # Final checkpoint
    torch.save(
        {"net": agent.net.state_dict(), "opt": agent.opt.state_dict(), "iter": cfg.max_iterations},
        os.path.join(log_dir, "a2c_final.pt"),
    )
    return log_dir
