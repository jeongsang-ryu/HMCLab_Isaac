"""A3 — Asymmetric SAC training (Champion-paper inspired, recurrent → frame-stack).

Off-policy soft actor-critic with:
* CNN(image) + MLP(proprio frame-stack) actor — local features only
* CNN(image) + MLP(proprio + privileged Frenet + action) critic — privileged
* Twin Q with Polyak-averaged targets (τ=0.005)
* Auto-tuned entropy coefficient α (target = −action_dim)
* Replay buffer holds raw uint8 images on CPU and proprio/privileged on GPU
  to keep VRAM finite; we move uint8 → float32 only at mini-batch sampling

Distributional QR-SAC head (32 quantiles) from the Champion paper is omitted
in this first cut — plain twin-Q is the well-established baseline. The
network and storage are designed so swapping in a quantile critic is a
focused 30-line change.
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
class SACConfig:
    """Plain (twin-Q) SAC hyperparameters, Champion-paper-inspired."""

    # rollout / optim
    total_env_steps: int = 4_000_000     # ~ 4090, 8h
    start_random_steps: int = 1000       # random actions for buffer warmup
    update_every: int = 1                # 1 gradient step per env step (Champion paper)
    updates_per_step: int = 1
    batch_size: int = 256
    gamma: float = 0.99                  # aligned with recurrent_ppo for fair PPO/SAC comparison (was 0.9896, Champion value)
    tau: float = 0.005                   # Polyak avg
    lr: float = 2.5e-5                   # Champion paper value
    target_entropy_scale: float = 1.0    # target = -action_dim * scale

    # buffer
    buffer_capacity: int = 200_000       # ~12 GB at 128x128 uint8 + proprio/priv

    # network
    cnn_channels: tuple[int, ...] = (32, 64, 64, 64)
    cnn_kernels: tuple[int, ...] = (8, 4, 3, 3)
    cnn_strides: tuple[int, ...] = (4, 2, 2, 1)
    cnn_out_dim: int = 512
    mlp_hidden: tuple[int, ...] = (256, 256)
    init_log_std_min: float = -20.0
    init_log_std_max: float = 2.0
    # GRU for actor (Champion-paper spirit): adds short-term memory so the
    # policy can integrate motion across time. Critic stays MLP (privileged
    # info is enough). Hidden state is stored per-env between env steps and
    # also in the replay buffer per transition.
    use_actor_gru: bool = True
    gru_hidden: int = 256

    # logging
    log_interval_steps: int = 1000
    save_interval_steps: int = 25_000     # ~ 2–3 min on 4090 with 16 envs
    experiment_name: str = "racing_sac_asym"
    log_dir: str = "/tmp/hmclab_runs"
    seed: int = 0


SKRL_SAC_CONFIG = SACConfig()   # exported under this name so __init__ entry_point resolves


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
def _make_cnn(in_ch: int, cfg: SACConfig, h: int, w: int) -> tuple[nn.Sequential, int]:
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


class SACActor(nn.Module):
    """Tanh-Gaussian actor: CNN(image) → concat(proprio) → GRUCell → MLP → (μ, log_std).

    The GRU adds short-term memory so the policy can infer velocity and
    track shape from a window of frames + proprio. Hidden state is fed in /
    out for each forward call and tracked per-env by the SACAgent.
    """

    def __init__(self, image_shape: tuple[int, int, int], proprio_dim: int,
                 action_dim: int, cfg: SACConfig):
        super().__init__()
        c, h, w = image_shape
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
        self.head_log_std = nn.Linear(d, action_dim)
        self.log_std_min = cfg.init_log_std_min
        self.log_std_max = cfg.init_log_std_max

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
        log_std = self.head_log_std(x).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std, new_hidden

    def sample(self, image: torch.Tensor, proprio: torch.Tensor,
               hidden: torch.Tensor | None = None):
        mean, log_std, new_hidden = self(image, proprio, hidden)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z).sum(dim=-1)
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob, torch.tanh(mean), new_hidden


class SACCritic(nn.Module):
    """Twin Q-network with asymmetric privileged input.

    Input: image + (proprio + privileged + action). Image is encoded via CNN
    shared between the two Qs (saves params; champion paper uses a separate
    critic CNN, which we can opt into via cfg if needed).
    """

    def __init__(self, image_shape: tuple[int, int, int], proprio_dim: int,
                 privileged_dim: int, action_dim: int, cfg: SACConfig):
        super().__init__()
        c, h, w = image_shape
        self.cnn, flat = _make_cnn(c, cfg, h, w)
        self.proj = nn.Linear(flat, cfg.cnn_out_dim)
        in_dim = cfg.cnn_out_dim + proprio_dim + privileged_dim + action_dim
        def _q():
            layers: list[nn.Module] = []
            d = in_dim
            for hd in cfg.mlp_hidden:
                layers += [nn.Linear(d, hd), nn.ReLU(inplace=True)]
                d = hd
            layers.append(nn.Linear(d, 1))
            return nn.Sequential(*layers)
        self.q1 = _q()
        self.q2 = _q()

    def forward(self, image: torch.Tensor, proprio: torch.Tensor,
                privileged: torch.Tensor, action: torch.Tensor):
        feat = self.cnn(image)
        feat = F.relu(self.proj(feat))
        x = torch.cat([feat, proprio, privileged, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Replay buffer (image uint8 on CPU, rest on GPU)
# ---------------------------------------------------------------------------
class SACReplayBuffer:
    """Ring buffer. Images stored as uint8 to save 4× VRAM."""

    def __init__(self, capacity: int, image_shape: tuple[int, int, int],
                 proprio_dim: int, privileged_dim: int, action_dim: int,
                 gru_hidden_dim: int, device: str):
        self.capacity = int(capacity)
        self.device = device
        c, h, w = image_shape
        # images live on CPU (pin_memory for fast transfer)
        self.images = torch.zeros((self.capacity, c, h, w), dtype=torch.uint8)
        self.next_images = torch.zeros_like(self.images)
        # 1D tensors on GPU
        self.proprio = torch.zeros((self.capacity, proprio_dim), device=device)
        self.next_proprio = torch.zeros_like(self.proprio)
        self.privileged = torch.zeros((self.capacity, privileged_dim), device=device)
        self.next_privileged = torch.zeros_like(self.privileged)
        self.actions = torch.zeros((self.capacity, action_dim), device=device)
        self.rewards = torch.zeros((self.capacity,), device=device)
        self.dones = torch.zeros((self.capacity,), device=device)
        # Per-transition GRU hidden (h_prev = pre-step, h_curr = post-step).
        # 0-dim if GRU disabled (acts as a no-op placeholder).
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.hidden_prev = torch.zeros((self.capacity, self.gru_hidden_dim), device=device)
        self.hidden_next = torch.zeros_like(self.hidden_prev)
        self.size = 0
        self.idx = 0

    def add_batch(self, img, prop, priv, act, rew, done, next_img, next_prop, next_priv,
                  h_prev=None, h_next=None):
        """Add a batch of transitions (env-step worth). All inputs are (E, …).

        ``h_prev`` and ``h_next`` are optional per-env GRU hidden states.
        Pass ``None`` if the actor has no GRU (placeholder zero rows are stored).
        """
        E = img.shape[0]
        # image: (E, C, H, W) — cast to uint8 if input is float in [0,1]
        if img.dtype != torch.uint8:
            img_u8 = (img.clamp(0, 1) * 255.0).to(torch.uint8).cpu()
            next_img_u8 = (next_img.clamp(0, 1) * 255.0).to(torch.uint8).cpu()
        else:
            img_u8 = img.cpu()
            next_img_u8 = next_img.cpu()
        if h_prev is None:
            h_prev = torch.zeros((E, self.gru_hidden_dim), device=self.device)
        if h_next is None:
            h_next = torch.zeros((E, self.gru_hidden_dim), device=self.device)

        end = self.idx + E
        if end <= self.capacity:
            self.images[self.idx:end] = img_u8
            self.next_images[self.idx:end] = next_img_u8
            self.proprio[self.idx:end] = prop
            self.next_proprio[self.idx:end] = next_prop
            self.privileged[self.idx:end] = priv
            self.next_privileged[self.idx:end] = next_priv
            self.actions[self.idx:end] = act
            self.rewards[self.idx:end] = rew
            self.dones[self.idx:end] = done.float()
            self.hidden_prev[self.idx:end] = h_prev
            self.hidden_next[self.idx:end] = h_next
        else:
            # wrap
            first = self.capacity - self.idx
            second = E - first
            self.images[self.idx:] = img_u8[:first]
            self.images[:second] = img_u8[first:]
            self.next_images[self.idx:] = next_img_u8[:first]
            self.next_images[:second] = next_img_u8[first:]
            self.proprio[self.idx:] = prop[:first]
            self.proprio[:second] = prop[first:]
            self.next_proprio[self.idx:] = next_prop[:first]
            self.next_proprio[:second] = next_prop[first:]
            self.privileged[self.idx:] = priv[:first]
            self.privileged[:second] = priv[first:]
            self.next_privileged[self.idx:] = next_priv[:first]
            self.next_privileged[:second] = next_priv[first:]
            self.actions[self.idx:] = act[:first]
            self.actions[:second] = act[first:]
            self.rewards[self.idx:] = rew[:first]
            self.rewards[:second] = rew[first:]
            self.dones[self.idx:] = done[:first].float()
            self.dones[:second] = done[first:].float()
            self.hidden_prev[self.idx:] = h_prev[:first]
            self.hidden_prev[:second] = h_prev[first:]
            self.hidden_next[self.idx:] = h_next[:first]
            self.hidden_next[:second] = h_next[first:]
        self.idx = (self.idx + E) % self.capacity
        self.size = min(self.size + E, self.capacity)

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,))
        img = self.images[idx].to(self.device, non_blocking=True).float() / 255.0
        next_img = self.next_images[idx].to(self.device, non_blocking=True).float() / 255.0
        prop = self.proprio[idx]
        next_prop = self.next_proprio[idx]
        priv = self.privileged[idx]
        next_priv = self.next_privileged[idx]
        act = self.actions[idx]
        rew = self.rewards[idx]
        done = self.dones[idx]
        h_prev = self.hidden_prev[idx]
        h_next = self.hidden_next[idx]
        return (img, prop, priv, act, rew, done,
                next_img, next_prop, next_priv, h_prev, h_next)


# ---------------------------------------------------------------------------
# Agent + training loop
# ---------------------------------------------------------------------------
class SACAgent:
    def __init__(self, env, cfg: SACConfig):
        self.env = env
        self.cfg = cfg
        self.device = env.device
        obs, _ = env.reset()
        img_shape = tuple(obs["images"].shape[1:])
        proprio_dim = int(obs["policy"].shape[1])
        privileged_dim = int(obs["privileged"].shape[1])
        # See sync_a2c.A2CAgent.__init__ — use single_action_space for the
        # per-env action dim (env.action_space is the batched vector space).
        action_dim = int(env.single_action_space.shape[0])
        self.action_dim = action_dim

        self.actor = SACActor(img_shape, proprio_dim, action_dim, cfg).to(self.device)
        self.critic = SACCritic(img_shape, proprio_dim, privileged_dim, action_dim, cfg).to(self.device)
        self.target_critic = SACCritic(img_shape, proprio_dim, privileged_dim, action_dim, cfg).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        # Auto-tune α
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        self.target_entropy = -float(action_dim) * cfg.target_entropy_scale

        # Per-env actor GRU hidden state (zeros at boot; reset on done).
        self.gru_hidden_dim = self.actor.gru_hidden
        self.actor_hidden = torch.zeros(
            (env.num_envs, max(self.gru_hidden_dim, 1)), device=self.device
        )

        self.buffer = SACReplayBuffer(
            cfg.buffer_capacity, img_shape, proprio_dim, privileged_dim, action_dim,
            self.gru_hidden_dim, self.device,
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()


def train(env, agent: SACAgent, cfg: SACConfig, start_step: int = 0) -> str:
    """Run asymmetric SAC. ``start_step`` continues env-step numbering from a
    loaded checkpoint (used only for logging/save-interval bookkeeping —
    replay buffer is rebuilt from scratch)."""
    log_dir = os.path.join(cfg.log_dir, cfg.experiment_name, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[sac] log_dir = {log_dir}", flush=True)
    if start_step > 0:
        print(f"[sac] resuming from step {start_step} (replay buffer fresh)", flush=True)

    # TensorBoard — shared tag schema with recurrent_ppo so SAC/PPO overlay on
    # the same charts (x-axis = env_step). Never let logging kill training.
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
    except Exception as _tb_exc:
        writer = None
        print(f"[sac] tensorboard unavailable ({_tb_exc}); console only", flush=True)

    torch.manual_seed(cfg.seed)
    obs, _ = env.reset()
    env_step = int(start_step)
    finished_returns: list[float] = []
    ep_returns = torch.zeros(env.num_envs, device=agent.device)
    # Lap-time tracking: per-env step counter (reset on episode end) and a
    # rolling list of completed-lap times. A lap is "completed" when the
    # terminal reward on the done step is the lap bonus (r_terminal_lap):
    # episode reward >= 0.5*r_terminal_lap cleanly separates a lap (+50)
    # from a bad termination (-20) / timeout (0). lap_time = steps * dt.
    ep_steps = torch.zeros(env.num_envs, device=agent.device)
    lap_times: list[float] = []
    _step_dt = float(env.cfg.decimation) * float(env.cfg.sim.dt)
    _lap_thr = 0.5 * float(env.cfg.r_terminal_lap)
    # Snapshot of the env's cumulative collision counters; diffed at each
    # log step to report the interval collision rate.
    _coll_prev = (
        int(getattr(env, "_coll_steps", 0)),
        int(getattr(env, "_coll_n_wall", 0)),
        int(getattr(env, "_coll_n_opp", 0)),
        int(getattr(env, "_coll_n_any", 0)),
    )
    finished_lengths: list[float] = []          # per-episode step counts (for TB)
    last_critic_loss = last_actor_loss = last_alpha_loss = 0.0
    last_q_mean = last_logp = 0.0
    _t0 = _t_prev = time.time()                 # wall-clock / throughput
    _step_prev = env_step
    while env_step < cfg.total_env_steps:
        # ---- Action ----
        h_prev_save = agent.actor_hidden.detach().clone()
        if env_step < cfg.start_random_steps:
            action = torch.empty(
                (env.num_envs, agent.action_dim), device=agent.device
            ).uniform_(-1.0, 1.0)
            # Still forward through GRU so its hidden state advances coherently
            with torch.no_grad():
                _, _, _, new_hidden = agent.actor.sample(
                    obs["images"], obs["policy"],
                    agent.actor_hidden if agent.actor.use_gru else None,
                )
        else:
            with torch.no_grad():
                action, _, _, new_hidden = agent.actor.sample(
                    obs["images"], obs["policy"],
                    agent.actor_hidden if agent.actor.use_gru else None,
                )
        if new_hidden is None:
            new_hidden = agent.actor_hidden
        else:
            agent.actor_hidden = new_hidden.detach()

        # Save current obs for buffer addition AFTER step
        cur_img = obs["images"].detach()
        cur_prop = obs["policy"].detach()
        cur_priv = obs["privileged"].detach()

        next_obs, reward, terminated, truncated, _ = env.step(action.detach().clamp(-1.0, 1.0))
        done = terminated | truncated

        agent.buffer.add_batch(
            cur_img, cur_prop, cur_priv,
            action.detach(), reward.detach(), done.detach(),
            next_obs["images"].detach(),
            next_obs["policy"].detach(),
            next_obs["privileged"].detach(),
            h_prev=h_prev_save if agent.actor.use_gru else None,
            h_next=new_hidden.detach() if agent.actor.use_gru else None,
        )

        # Reset GRU hidden for envs that terminated/truncated.
        if agent.actor.use_gru:
            keep_mask = (~done).float().unsqueeze(-1)
            agent.actor_hidden = agent.actor_hidden * keep_mask
        ep_returns += reward
        ep_steps += 1.0
        for env_idx in done.nonzero(as_tuple=False).flatten().tolist():
            finished_returns.append(float(ep_returns[env_idx].item()))
            finished_lengths.append(float(ep_steps[env_idx].item()))
            if float(reward[env_idx].item()) >= _lap_thr:
                lap_times.append(float(ep_steps[env_idx].item()) * _step_dt)
            ep_returns[env_idx] = 0.0
            ep_steps[env_idx] = 0.0

        obs = next_obs
        env_step += env.num_envs

        # ---- Updates ----
        if env_step >= cfg.start_random_steps and agent.buffer.size >= cfg.batch_size:
            for _ in range(cfg.updates_per_step):
                (img_b, prop_b, priv_b, act_b, rew_b, done_b,
                 nimg_b, nprop_b, npriv_b, h_prev_b, h_next_b) = agent.buffer.sample(cfg.batch_size)

                h_in  = h_prev_b if agent.actor.use_gru else None
                h_nxt = h_next_b if agent.actor.use_gru else None

                # ---- critic loss ----
                with torch.no_grad():
                    next_act, next_logp, _, _ = agent.actor.sample(nimg_b, nprop_b, h_nxt)
                    q1_t, q2_t = agent.target_critic(nimg_b, nprop_b, npriv_b, next_act)
                    q_t = torch.min(q1_t, q2_t) - agent.alpha.detach() * next_logp
                    target = rew_b + cfg.gamma * (1.0 - done_b) * q_t

                q1, q2 = agent.critic(img_b, prop_b, priv_b, act_b)
                critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
                # Skip non-finite updates and clip gradient norm. Without
                # this the policy diverged to NaN in the hard randomized
                # env (~12.7 M steps) and the run died. Clipping bounds
                # update magnitude only — it does not bias the objective.
                if torch.isfinite(critic_loss):
                    agent.critic_opt.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.critic.parameters(), 1.0
                    )
                    agent.critic_opt.step()

                # ---- actor loss ----
                new_act, new_logp, _, _ = agent.actor.sample(img_b, prop_b, h_in)
                q1_n, q2_n = agent.critic(img_b, prop_b, priv_b, new_act)
                q_n = torch.min(q1_n, q2_n)
                actor_loss = (agent.alpha.detach() * new_logp - q_n).mean()
                if torch.isfinite(actor_loss):
                    agent.actor_opt.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.actor.parameters(), 1.0
                    )
                    agent.actor_opt.step()

                # ---- alpha (entropy temperature) ----
                alpha_loss = -(agent.log_alpha * (new_logp + agent.target_entropy).detach()).mean()
                if torch.isfinite(alpha_loss):
                    agent.alpha_opt.zero_grad()
                    alpha_loss.backward()
                    agent.alpha_opt.step()

                # capture last-update scalars for TensorBoard
                last_critic_loss = float(critic_loss.item())
                last_actor_loss = float(actor_loss.item())
                last_alpha_loss = float(alpha_loss.item())
                last_q_mean = float(q_n.mean().item())
                last_logp = float(new_logp.mean().item())

                # ---- target soft update ----
                with torch.no_grad():
                    for tp, p in zip(agent.target_critic.parameters(), agent.critic.parameters()):
                        tp.data.mul_(1.0 - cfg.tau)
                        tp.data.add_(cfg.tau * p.data)

        # ---- Log / save ----
        if env_step // cfg.log_interval_steps != (env_step - env.num_envs) // cfg.log_interval_steps:
            ret50 = (
                sum(finished_returns[-50:]) / max(1, len(finished_returns[-50:]))
                if finished_returns else 0.0
            )
            cs = int(getattr(env, "_coll_steps", 0))
            cw = int(getattr(env, "_coll_n_wall", 0))
            co = int(getattr(env, "_coll_n_opp", 0))
            ca = int(getattr(env, "_coll_n_any", 0))
            d_s = max(1, cs - _coll_prev[0])
            wall_rate = 100.0 * (cw - _coll_prev[1]) / d_s
            opp_rate = 100.0 * (co - _coll_prev[2]) / d_s
            any_rate = 100.0 * (ca - _coll_prev[3]) / d_s
            _coll_prev = (cs, cw, co, ca)
            if lap_times:
                _lt = lap_times[-50:]
                lap_str = (f"lap(50) mean={sum(_lt)/len(_lt):5.2f}s "
                           f"best={min(lap_times):5.2f}s "
                           f"n={len(lap_times)}")
            else:
                lap_str = "lap(50) —(no completed laps yet)"
            print(
                f"[sac] step={env_step:8d}  buf={agent.buffer.size:7d}  "
                f"α={float(agent.alpha.item()):.3f}  "
                f"ret(50)={ret50:+.2f}  n_ep={len(finished_returns)}  "
                f"coll%(wall/opp/any)="
                f"{wall_rate:.1f}/{opp_rate:.1f}/{any_rate:.1f}  "
                f"{lap_str}",
                flush=True,
            )
            if writer is not None:
                now = time.time()
                sps = (env_step - _step_prev) / max(1e-6, now - _t_prev)
                _t_prev, _step_prev = now, env_step
                el = finished_lengths[-50:]
                writer.add_scalar("perf/return_mean", ret50, env_step)
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
                writer.add_scalar("sac/alpha", float(agent.alpha.item()), env_step)
                writer.add_scalar("sac/critic_loss", last_critic_loss, env_step)
                writer.add_scalar("sac/actor_loss", last_actor_loss, env_step)
                writer.add_scalar("sac/alpha_loss", last_alpha_loss, env_step)
                writer.add_scalar("sac/q_mean", last_q_mean, env_step)
                writer.add_scalar("sac/policy_entropy", -last_logp, env_step)
                writer.add_scalar("sac/buffer_size", agent.buffer.size, env_step)
        if env_step // cfg.save_interval_steps != (env_step - env.num_envs) // cfg.save_interval_steps:
            torch.save(
                {
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "target_critic": agent.target_critic.state_dict(),
                    "log_alpha": agent.log_alpha.detach(),
                    "step": env_step,
                },
                os.path.join(log_dir, f"sac_step{env_step:08d}.pt"),
            )

    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "target_critic": agent.target_critic.state_dict(),
            "log_alpha": agent.log_alpha.detach(),
            "step": env_step,
        },
        os.path.join(log_dir, "sac_final.pt"),
    )
    if writer is not None:
        writer.close()
    return log_dir
