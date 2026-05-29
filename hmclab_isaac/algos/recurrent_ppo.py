"""A3 — Recurrent CNN-PPO whose networks mirror the asymmetric SAC exactly.

Why this exists
---------------
rsl_rl 4.x has no combined CNN+RNN model class, so the stock PPO (A2,
``rsl_rl_ppo_cfg.py``) is forced to be memoryless. To compare on-policy PPO
against off-policy SAC with the *same* architecture, this module re-implements
PPO in-house on top of the same recurrent rollout machinery used by the A2C
trainer (``sync_a2c.py``), but with the **identical SAC network**:

    Actor  : CNN(Champion 32/64/64/64) → proj-512 → cat(proprio) → GRUCell-256
             → MLP(256,256) → (mean, log_std)   [tanh-squashed Gaussian]
    Critic : CNN → proj-512 → cat(proprio + privileged) → MLP(256,256) → V(s)
             (NO GRU — matches SAC's critic; privileged Frenet ≈ full state)

The only structural differences left vs SAC are the ones that *define*
on-policy vs off-policy and cannot be removed:
  * critic estimates V(s) (no action input) instead of twin Q(s,a)
  * update is clipped-surrogate PPO + GAE over fresh on-policy rollouts
    instead of replayed off-policy transitions.

Actor-head parity with SAC is full by default (``use_tanh=True`` +
``state_dependent_std=True``); set them False for the more stable PPO-standard
plain-Gaussian head if training destabilizes.

Recurrence in the PPO update uses truncated BPTT: each iteration replays the
actor GRU forward through the whole ``num_steps_per_env`` rollout from a stored
per-env initial hidden state, masking the hidden to zero at episode boundaries
— the same convention the rollout uses.

Usage (after AppLauncher boot)::

    from hmclab_isaac.algos.recurrent_ppo import RecurrentPPO, RPPOConfig, train
    agent = RecurrentPPO(env, RPPOConfig())
    train(env, agent, RPPOConfig())
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class RPPOConfig:
    """Recurrent CNN-PPO hyperparameters.

    Algorithm fields track the stock PPO (``rsl_rl_ppo_cfg.py``); network fields
    track SAC (``asym_sac.py``) so the actor/critic are architecturally equal.
    """

    # --- rollout / PPO ---
    num_steps_per_env: int = 32      # T: 32 envs × 32 steps = 1024 transitions/iter
    max_iterations: int = 4096
    gamma: float = 0.99
    lam: float = 0.95               # GAE λ
    clip_param: float = 0.2
    num_learning_epochs: int = 5
    num_mini_batches: int = 4        # split the ENV dim (recurrence needs whole sequences)
    entropy_coef: float = 0.008      # modest (0.015 caused std→clamp blow-up; the log_std_max cap below now makes exploration safe)
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    learning_rate: float = 3.0e-4
    desired_kl: float = 0.01         # reverted (0.02 + high entropy let the std walk to the clamp ceiling)
    adaptive_lr: bool = True
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    max_grad_norm: float = 1.0

    # --- network (mirror SAC asym_sac.SACConfig) ---
    cnn_channels: tuple[int, ...] = (32, 64, 64, 64)
    cnn_kernels: tuple[int, ...] = (8, 4, 3, 3)
    cnn_strides: tuple[int, ...] = (4, 2, 2, 1)
    cnn_out_dim: int = 512
    mlp_hidden: tuple[int, ...] = (256, 256)
    use_actor_gru: bool = True
    gru_hidden: int = 256
    # actor head: full SAC parity by default
    use_tanh: bool = True                 # tanh-squashed Gaussian (SAC) vs plain Gaussian
    state_dependent_std: bool = True      # log_std = Linear(x) (SAC) vs a single Parameter
    init_log_std: float = -0.5            # reverted (std 0.6 start); used only when state_dependent_std=False
    log_std_min: float = -20.0
    log_std_max: float = 0.0              # CAP std ≤ 1.0 — structurally prevents the entropy-driven std→bang-bang blow-up

    # --- logging / checkpoints ---
    log_interval: int = 10
    save_interval: int = 200
    experiment_name: str = "racing_rppo"
    log_dir: str = "/tmp/hmclab_runs"
    seed: int = 0


DEFAULT_RPPO_CONFIG = RPPOConfig()


# ---------------------------------------------------------------------------
# Networks  (CNN copied verbatim from asym_sac._make_cnn so weights/shapes match)
# ---------------------------------------------------------------------------
def _make_cnn(in_ch: int, cfg: RPPOConfig, h: int, w: int) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    c = in_ch
    for out_ch, k, s in zip(cfg.cnn_channels, cfg.cnn_kernels, cfg.cnn_strides):
        layers.append(nn.Conv2d(c, out_ch, kernel_size=k, stride=s))
        layers.append(nn.ReLU(inplace=True))
        c = out_ch
    layers.append(nn.Flatten())
    cnn = nn.Sequential(*layers)
    with torch.no_grad():
        flat = cnn(torch.zeros(1, in_ch, h, w)).shape[1]
    return cnn, int(flat)


class RPPOActor(nn.Module):
    """Same network as ``asym_sac.SACActor``: CNN → proj → cat(proprio) → GRU → MLP → (μ, log_std)."""

    def __init__(self, image_shape: tuple[int, int, int], proprio_dim: int,
                 action_dim: int, cfg: RPPOConfig):
        super().__init__()
        c, h, w = image_shape
        self.cfg = cfg
        self.action_dim = action_dim
        self.cnn, flat = _make_cnn(c, cfg, h, w)
        self.proj = nn.Linear(flat, cfg.cnn_out_dim)
        gru_in = cfg.cnn_out_dim + proprio_dim
        self.use_gru = bool(cfg.use_actor_gru)
        self.gru_hidden = int(cfg.gru_hidden) if self.use_gru else 0
        if self.use_gru:
            self.gru = nn.GRUCell(gru_in, cfg.gru_hidden)
            head_in = cfg.gru_hidden
        else:
            head_in = gru_in
        layers: list[nn.Module] = []
        d = head_in
        for hd in cfg.mlp_hidden:
            layers += [nn.Linear(d, hd), nn.ReLU(inplace=True)]
            d = hd
        self.trunk = nn.Sequential(*layers)
        self.head_mean = nn.Linear(d, action_dim)
        self.use_tanh = bool(cfg.use_tanh)
        self.state_dependent_std = bool(cfg.state_dependent_std)
        if self.state_dependent_std:
            self.head_log_std = nn.Linear(d, action_dim)
        else:
            self.log_std_param = nn.Parameter(torch.full((action_dim,), cfg.init_log_std))
        self.log_std_min = cfg.log_std_min
        self.log_std_max = cfg.log_std_max

    def forward(self, image: torch.Tensor, proprio: torch.Tensor,
                hidden: torch.Tensor | None = None):
        feat = self.cnn(image)
        feat = F.relu(self.proj(feat))
        x = torch.cat([feat, proprio], dim=-1)
        if self.use_gru:
            if hidden is None:
                hidden = torch.zeros(x.shape[0], self.gru_hidden, device=x.device)
            new_hidden = self.gru(x, hidden)
            x = new_hidden
        else:
            new_hidden = None
        x = self.trunk(x)
        mean = self.head_mean(x)
        if self.state_dependent_std:
            log_std = self.head_log_std(x)
        else:
            log_std = self.log_std_param.expand_as(mean)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mean, log_std, new_hidden

    # -- log-prob of a pre-squash Gaussian sample z (tanh correction cancels in
    #    the PPO ratio, but we keep it so the stored value is a true log π(a)) --
    def _log_prob(self, dist: torch.distributions.Normal, z: torch.Tensor) -> torch.Tensor:
        lp = dist.log_prob(z).sum(dim=-1)
        if self.use_tanh:
            a = torch.tanh(z)
            lp = lp - torch.log(1.0 - a.pow(2) + 1e-6).sum(dim=-1)
        return lp

    def act(self, image, proprio, hidden=None):
        """Sample for rollout. Returns (action, z, log_prob, mean, log_std, new_hidden)."""
        mean, log_std, new_hidden = self(image, proprio, hidden)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        z = dist.sample()
        action = torch.tanh(z) if self.use_tanh else z
        log_prob = self._log_prob(dist, z)
        return action, z, log_prob, mean, log_std, new_hidden

    def evaluate(self, image, proprio, z, hidden=None):
        """Re-evaluate stored z under current params. Returns (log_prob, entropy, mean, log_std, new_hidden)."""
        mean, log_std, new_hidden = self(image, proprio, hidden)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_prob = self._log_prob(dist, z)
        entropy = dist.entropy().sum(dim=-1)   # analytic Gaussian entropy (pre-tanh)
        return log_prob, entropy, mean, log_std, new_hidden

    def act_deterministic(self, image, proprio, hidden=None):
        """Greedy action for evaluation/play (mean, optionally tanh-squashed)."""
        mean, _, new_hidden = self(image, proprio, hidden)
        action = torch.tanh(mean) if self.use_tanh else mean
        return action, new_hidden


class RPPOCritic(nn.Module):
    """Asymmetric value network V(s): CNN → proj → cat(proprio + privileged) → MLP → scalar.

    Same shape as ``asym_sac.SACCritic`` minus the action input and twin head
    (PPO needs V(s), not Q(s,a)). No GRU — matches SAC's critic.
    """

    def __init__(self, image_shape: tuple[int, int, int], proprio_dim: int,
                 privileged_dim: int, cfg: RPPOConfig):
        super().__init__()
        c, h, w = image_shape
        self.cnn, flat = _make_cnn(c, cfg, h, w)
        self.proj = nn.Linear(flat, cfg.cnn_out_dim)
        in_dim = cfg.cnn_out_dim + proprio_dim + privileged_dim
        layers: list[nn.Module] = []
        d = in_dim
        for hd in cfg.mlp_hidden:
            layers += [nn.Linear(d, hd), nn.ReLU(inplace=True)]
            d = hd
        layers.append(nn.Linear(d, 1))
        self.v = nn.Sequential(*layers)

    def forward(self, image, proprio, privileged):
        feat = self.cnn(image)
        feat = F.relu(self.proj(feat))
        x = torch.cat([feat, proprio, privileged], dim=-1)
        return self.v(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Agent wrapper
# ---------------------------------------------------------------------------
class RecurrentPPO:
    """Recurrent actor + asymmetric V-critic + Adam, with per-env actor hidden state."""

    def __init__(self, env, cfg: RPPOConfig):
        self.cfg = cfg
        self.env = env
        self.device = env.device
        obs, _ = env.reset()
        img_shape = tuple(obs["images"].shape[1:])         # (C, H, W)
        proprio_dim = int(obs["policy"].shape[1])
        privileged_dim = int(obs["privileged"].shape[1])
        action_dim = int(env.single_action_space.shape[0])
        self.actor = RPPOActor(img_shape, proprio_dim, action_dim, cfg).to(self.device)
        self.critic = RPPOCritic(img_shape, proprio_dim, privileged_dim, cfg).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.learning_rate,
        )
        self.lr = float(cfg.learning_rate)
        E = env.num_envs
        self.hidden = (
            torch.zeros(E, cfg.gru_hidden, device=self.device)
            if self.actor.use_gru else None
        )

    def reset_hidden(self, dones: torch.Tensor):
        if self.hidden is None:
            return
        self.hidden = self.hidden * (~dones).float().unsqueeze(-1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(env, agent: RecurrentPPO, cfg: RPPOConfig, start_iter: int = 0) -> str:
    log_dir = os.path.join(cfg.log_dir, cfg.experiment_name, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[rppo] log_dir = {log_dir}", flush=True)
    print(f"[rppo] actor: tanh={cfg.use_tanh} state_std={cfg.state_dependent_std} "
          f"gru={cfg.use_actor_gru}({cfg.gru_hidden}) | T={cfg.num_steps_per_env} "
          f"epochs={cfg.num_learning_epochs} mb={cfg.num_mini_batches}", flush=True)
    if start_iter > 0:
        print(f"[rppo] resuming from iter {start_iter}", flush=True)

    # TensorBoard — same tag schema as asym_sac so SAC/PPO overlay on one chart
    # (x-axis = env_step). Never let logging kill training.
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
    except Exception as _tb_exc:
        writer = None
        print(f"[rppo] tensorboard unavailable ({_tb_exc}); console only", flush=True)

    torch.manual_seed(cfg.seed)
    device = agent.device
    actor, critic = agent.actor, agent.critic
    T = cfg.num_steps_per_env
    E = env.num_envs

    obs, _ = env.reset()
    ep_returns = torch.zeros(E, device=device)
    ep_steps = torch.zeros(E, device=device)
    finished_returns: list[float] = []
    finished_lengths: list[float] = []
    lap_times: list[float] = []
    _step_dt = float(env.cfg.decimation) * float(env.cfg.sim.dt)
    _lap_thr = 0.5 * float(env.cfg.r_terminal_lap)   # +50 lap vs −20 bad termination
    _coll_prev = (
        int(getattr(env, "_coll_steps", 0)),
        int(getattr(env, "_coll_n_wall", 0)),
    )
    _t0 = _t_prev = time.time()
    _step_prev = start_iter * T * E

    for it in range(start_iter, cfg.max_iterations):
        # ---- Rollout buffers (T, E, ...) ----
        b_img, b_pro, b_priv = [], [], []
        b_z, b_logp, b_val = [], [], []
        b_rew, b_done, b_term = [], [], []
        b_mean, b_logstd = [], []
        # hidden entering this rollout (per env) — replay starts from here
        h0 = agent.hidden.detach().clone() if agent.hidden is not None else None

        for t in range(T):
            image, proprio, priv = obs["images"], obs["policy"], obs["privileged"]
            with torch.no_grad():
                action, z, logp, mean, log_std, new_h = actor.act(image, proprio, agent.hidden)
                value = critic(image, proprio, priv)
            agent.hidden = new_h

            b_img.append(image); b_pro.append(proprio); b_priv.append(priv)
            b_z.append(z); b_logp.append(logp); b_val.append(value)
            b_mean.append(mean); b_logstd.append(log_std)

            obs, reward, terminated, truncated, _ = env.step(action.clamp(-1.0, 1.0))
            done = terminated | truncated
            b_rew.append(reward); b_done.append(done.float()); b_term.append(terminated.float())

            ep_returns += reward
            ep_steps += 1.0
            for ei in done.nonzero(as_tuple=False).flatten().tolist():
                finished_returns.append(float(ep_returns[ei].item()))
                finished_lengths.append(float(ep_steps[ei].item()))
                if float(reward[ei].item()) >= _lap_thr:
                    lap_times.append(float(ep_steps[ei].item()) * _step_dt)
                ep_returns[ei] = 0.0
                ep_steps[ei] = 0.0
            agent.reset_hidden(done)

        # ---- Bootstrap + GAE ----
        with torch.no_grad():
            last_value = critic(obs["images"], obs["policy"], obs["privileged"])
        img = torch.stack(b_img); pro = torch.stack(b_pro); priv = torch.stack(b_priv)   # (T,E,...)
        z = torch.stack(b_z); logp_old = torch.stack(b_logp); values = torch.stack(b_val)
        rew = torch.stack(b_rew); done = torch.stack(b_done); term = torch.stack(b_term)
        mean_old = torch.stack(b_mean); logstd_old = torch.stack(b_logstd)

        advantages = torch.zeros(T, E, device=device)
        gae = torch.zeros(E, device=device)
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            next_nonterminal = 1.0 - term[t]            # bootstrap on timeout, not on true terminal
            delta = rew[t] + cfg.gamma * next_value * next_nonterminal - values[t]
            gae = delta + cfg.gamma * cfg.lam * (1.0 - done[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        # explained variance of V (1 = perfect fit, ≤0 = no better than mean)
        with torch.no_grad():
            r_flat, v_flat = returns.reshape(-1), values.reshape(-1)
            var_r = float(r_flat.var())
            explained_var = (1.0 - float((r_flat - v_flat).var()) / var_r) if var_r > 1e-8 else 0.0
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- PPO update (recurrent BPTT, minibatch over the ENV dim) ----
        mb_size = max(1, E // cfg.num_mini_batches)
        std_old = logstd_old.exp()
        last_kl = 0.0
        pi_loss_acc = v_loss_acc = ent_acc = clip_frac_acc = 0.0
        n_upd = 0
        for _epoch in range(cfg.num_learning_epochs):
            perm = torch.randperm(E, device=device)
            for mb in range(cfg.num_mini_batches):
                ids = perm[mb * mb_size:(mb + 1) * mb_size]
                if ids.numel() == 0:
                    continue
                # -- replay actor GRU through the rollout for these envs --
                h = h0[ids] if h0 is not None else None
                lp_list, ent_list, mean_list, logstd_list = [], [], [], []
                for t in range(T):
                    lp, ent, m, ls, h_next = actor.evaluate(img[t][ids], pro[t][ids], z[t][ids], h)
                    lp_list.append(lp); ent_list.append(ent)
                    mean_list.append(m); logstd_list.append(ls)
                    if h_next is not None:
                        h = h_next * (1.0 - done[t][ids]).unsqueeze(-1)
                logp_new = torch.stack(lp_list)            # (T, mb)
                entropy = torch.stack(ent_list)
                mean_new = torch.stack(mean_list)
                logstd_new = torch.stack(logstd_list)

                # -- critic value (no recurrence → flatten T×mb) --
                Tm = T * ids.numel()
                v_new = critic(
                    img[:, ids].reshape(Tm, *img.shape[2:]),
                    pro[:, ids].reshape(Tm, pro.shape[2]),
                    priv[:, ids].reshape(Tm, priv.shape[2]),
                ).reshape(T, ids.numel())

                adv_mb = advantages[:, ids]
                ret_mb = returns[:, ids]
                logp_old_mb = logp_old[:, ids]
                val_old_mb = values[:, ids]

                # -- KL (analytic Gaussian) for adaptive lr --
                with torch.no_grad():
                    std_new = logstd_new.exp()
                    kl = (
                        logstd_new - logstd_old[:, ids]
                        + (std_old[:, ids].pow(2) + (mean_old[:, ids] - mean_new).pow(2))
                        / (2.0 * std_new.pow(2)) - 0.5
                    ).sum(dim=-1).mean()
                    last_kl = float(kl.item())
                if cfg.adaptive_lr:
                    if last_kl > 2.0 * cfg.desired_kl:
                        agent.lr = max(cfg.lr_min, agent.lr / 1.5)
                    elif 0.0 < last_kl < 0.5 * cfg.desired_kl:
                        agent.lr = min(cfg.lr_max, agent.lr * 1.5)
                    for g in agent.opt.param_groups:
                        g["lr"] = agent.lr

                # -- PPO clipped surrogate --
                ratio = torch.exp(logp_new - logp_old_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * adv_mb
                pi_loss = -torch.min(surr1, surr2).mean()

                if cfg.use_clipped_value_loss:
                    v_clipped = val_old_mb + (v_new - val_old_mb).clamp(-cfg.clip_param, cfg.clip_param)
                    v_loss = 0.5 * torch.max((v_new - ret_mb).pow(2), (v_clipped - ret_mb).pow(2)).mean()
                else:
                    v_loss = 0.5 * (v_new - ret_mb).pow(2).mean()

                ent_loss = -entropy.mean()
                loss = pi_loss + cfg.value_loss_coef * v_loss + cfg.entropy_coef * ent_loss

                agent.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), cfg.max_grad_norm
                )
                agent.opt.step()

                pi_loss_acc += float(pi_loss.item())
                v_loss_acc += float(v_loss.item())
                ent_acc += float((-ent_loss).item())
                with torch.no_grad():
                    clip_frac_acc += float(((ratio - 1.0).abs() > cfg.clip_param).float().mean().item())
                n_upd += 1

        # detach carried hidden so next iteration starts a fresh graph
        if agent.hidden is not None:
            agent.hidden = agent.hidden.detach()

        # ---- Logging (TensorBoard every iter; console every log_interval) ----
        env_step = (it + 1) * T * E
        n = max(1, n_upd)
        recent = finished_returns[-50:]
        mean_ret = sum(recent) / max(1, len(recent)) if recent else 0.0
        if writer is not None:
            now = time.time()
            sps = (env_step - _step_prev) / max(1e-6, now - _t_prev)
            _t_prev, _step_prev = now, env_step
            el = finished_lengths[-50:]
            cw = int(getattr(env, "_coll_n_wall", 0))
            cs = int(getattr(env, "_coll_steps", 0))
            d_s = max(1, cs - _coll_prev[0])
            wall_rate = 100.0 * (cw - _coll_prev[1]) / d_s
            _coll_prev = (cs, cw)
            writer.add_scalar("perf/return_mean", mean_ret, env_step)
            writer.add_scalar("perf/episode_count", len(finished_returns), env_step)
            if el:
                writer.add_scalar("perf/episode_len_mean", sum(el) / len(el), env_step)
            writer.add_scalar("perf/lap_count", len(lap_times), env_step)
            if lap_times:
                _lt = lap_times[-50:]
                writer.add_scalar("perf/lap_time_mean_s", sum(_lt) / len(_lt), env_step)
                writer.add_scalar("perf/lap_time_best_s", min(lap_times), env_step)
            writer.add_scalar("behavior/wall_collision_rate", wall_rate, env_step)
            writer.add_scalar("time/steps_per_sec", sps, env_step)
            writer.add_scalar("time/wall_minutes", (now - _t0) / 60.0, env_step)
            writer.add_scalar("ppo/policy_loss", pi_loss_acc / n, env_step)
            writer.add_scalar("ppo/value_loss", v_loss_acc / n, env_step)
            writer.add_scalar("ppo/entropy", ent_acc / n, env_step)
            writer.add_scalar("ppo/approx_kl", last_kl, env_step)
            writer.add_scalar("ppo/clip_fraction", clip_frac_acc / n, env_step)
            writer.add_scalar("ppo/learning_rate", agent.lr, env_step)
            writer.add_scalar("ppo/explained_variance", explained_var, env_step)
            writer.add_scalar("env/max_wheel_rps", float(getattr(env, "_cur_max_rps", 0.0)), env_step)
        if it % cfg.log_interval == 0:
            print(
                f"[rppo] iter {it:5d}  "
                f"pi={pi_loss_acc / n:+.3f}  v={v_loss_acc / n:+.3f}  "
                f"H={ent_acc / n:+.3f}  kl={last_kl:.4f}  lr={agent.lr:.1e}  "
                f"ev={explained_var:+.2f}  "
                f"ret(50)={mean_ret:+.2f}  n_ep={len(finished_returns)}",
                flush=True,
            )
        if it > 0 and it % cfg.save_interval == 0:
            torch.save(
                {"actor": actor.state_dict(), "critic": critic.state_dict(),
                 "opt": agent.opt.state_dict(), "iter": it, "cfg": vars(cfg)},
                os.path.join(log_dir, f"rppo_iter{it:06d}.pt"),
            )

    torch.save(
        {"actor": actor.state_dict(), "critic": critic.state_dict(),
         "opt": agent.opt.state_dict(), "iter": cfg.max_iterations, "cfg": vars(cfg)},
        os.path.join(log_dir, "rppo_final.pt"),
    )
    if writer is not None:
        writer.close()
    print(f"[rppo] DONE — checkpoints in {log_dir}", flush=True)
    return log_dir
