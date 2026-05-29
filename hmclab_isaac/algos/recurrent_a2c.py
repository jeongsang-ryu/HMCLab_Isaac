"""A4 — Recurrent A2C with the SAME network & hyperparameters as recurrent PPO.

For the on/off-policy study we want a SECOND on-policy baseline whose model is
*identical* to SAC/PPO so that only the algorithm differs. This module reuses
the exact network (``RPPOActor`` + asymmetric V-critic ``RPPOCritic``), agent
wrapper (``RecurrentPPO``) and config (``RPPOConfig``) from ``recurrent_ppo``,
plus the same synchronous rollout + GAE. The ONLY change is the update rule:

    A2C : ONE on-policy update per rollout — vanilla policy gradient
          (loss = -logp·advantage), no importance ratio, no clipping, no
          multiple epochs, fixed learning rate.
    PPO : clipped surrogate, multiple epochs/minibatches, KL-adaptive lr.

So A2C vs PPO isolates "vanilla PG vs trust-region clipped PG", and both vs SAC
isolates on- vs off-policy — all on the identical CNN+GRU actor / privileged
critic. The gradient is accumulated over env-chunks (to bound memory) and then
applied in a single optimizer step, which is exactly full-batch A2C.

Note: the rollout / GAE / TensorBoard logging below intentionally mirror
``recurrent_ppo.train`` so the two trainers are byte-comparable; only the
``# ---- A2C update ----`` block differs.

Usage (after AppLauncher boot)::

    from hmclab_isaac.algos.recurrent_ppo import RecurrentPPO, RPPOConfig
    from hmclab_isaac.algos.recurrent_a2c import train_a2c
    cfg = RPPOConfig(experiment_name="racing_ra2c")
    agent = RecurrentPPO(env, cfg)
    train_a2c(env, agent, cfg)
"""
from __future__ import annotations

import os
import time

import torch
import torch.nn as nn

# Re-use the EXACT same network / agent / config as recurrent PPO so the two
# on-policy methods are architecturally identical.
from hmclab_isaac.algos.recurrent_ppo import RecurrentPPO, RPPOConfig  # noqa: F401


def train_a2c(env, agent: RecurrentPPO, cfg: RPPOConfig, start_iter: int = 0) -> str:
    log_dir = os.path.join(cfg.log_dir, cfg.experiment_name, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[ra2c] log_dir = {log_dir}", flush=True)
    print(f"[ra2c] actor: tanh={cfg.use_tanh} state_std={cfg.state_dependent_std} "
          f"gru={cfg.use_actor_gru}({cfg.gru_hidden}) | T={cfg.num_steps_per_env} "
          f"update=single-pass vanilla-PG (no clip/epochs), fixed lr={cfg.learning_rate:.1e}", flush=True)
    if start_iter > 0:
        print(f"[ra2c] resuming from iter {start_iter}", flush=True)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
    except Exception as _tb_exc:
        writer = None
        print(f"[ra2c] tensorboard unavailable ({_tb_exc}); console only", flush=True)

    torch.manual_seed(cfg.seed)
    device = agent.device
    actor, critic = agent.actor, agent.critic
    T = cfg.num_steps_per_env
    E = env.num_envs
    agent.lr = float(cfg.learning_rate)          # A2C: fixed lr (no KL-adaptive)
    for g in agent.opt.param_groups:
        g["lr"] = agent.lr

    obs, _ = env.reset()
    ep_returns = torch.zeros(E, device=device)
    ep_steps = torch.zeros(E, device=device)
    finished_returns: list[float] = []
    finished_lengths: list[float] = []
    lap_times: list[float] = []
    _step_dt = float(env.cfg.decimation) * float(env.cfg.sim.dt)
    _lap_thr = 0.5 * float(env.cfg.r_terminal_lap)
    _coll_prev = (int(getattr(env, "_coll_steps", 0)), int(getattr(env, "_coll_n_wall", 0)))
    _t0 = _t_prev = time.time()
    _step_prev = start_iter * T * E

    for it in range(start_iter, cfg.max_iterations):
        # ---- Rollout (identical to recurrent_ppo) ----
        b_img, b_pro, b_priv = [], [], []
        b_z, b_logp, b_val = [], [], []
        b_rew, b_done, b_term = [], [], []
        h0 = agent.hidden.detach().clone() if agent.hidden is not None else None

        for t in range(T):
            image, proprio, priv = obs["images"], obs["policy"], obs["privileged"]
            with torch.no_grad():
                action, z, logp, _mean, _ls, new_h = actor.act(image, proprio, agent.hidden)
                value = critic(image, proprio, priv)
            agent.hidden = new_h

            b_img.append(image); b_pro.append(proprio); b_priv.append(priv)
            b_z.append(z); b_logp.append(logp); b_val.append(value)

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

        # ---- Bootstrap + GAE (identical) ----
        with torch.no_grad():
            last_value = critic(obs["images"], obs["policy"], obs["privileged"])
        img = torch.stack(b_img); pro = torch.stack(b_pro); priv = torch.stack(b_priv)
        z = torch.stack(b_z); values = torch.stack(b_val)
        rew = torch.stack(b_rew); done = torch.stack(b_done); term = torch.stack(b_term)

        advantages = torch.zeros(T, E, device=device)
        gae = torch.zeros(E, device=device)
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            next_nonterminal = 1.0 - term[t]
            delta = rew[t] + cfg.gamma * next_value * next_nonterminal - values[t]
            gae = delta + cfg.gamma * cfg.lam * (1.0 - done[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        with torch.no_grad():
            r_flat, v_flat = returns.reshape(-1), values.reshape(-1)
            var_r = float(r_flat.var())
            explained_var = (1.0 - float((r_flat - v_flat).var()) / var_r) if var_r > 1e-8 else 0.0
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- A2C update: ONE vanilla-PG step (grad accumulated over env-chunks) ----
        mb_size = max(1, E // cfg.num_mini_batches)
        pi_loss_acc = v_loss_acc = ent_acc = 0.0
        agent.opt.zero_grad()
        for c in range(cfg.num_mini_batches):
            ids = torch.arange(c * mb_size, min((c + 1) * mb_size, E), device=device)
            if ids.numel() == 0:
                continue
            h = h0[ids] if h0 is not None else None
            lp_list, ent_list = [], []
            for t in range(T):
                lp, ent, _m, _ls, h_next = actor.evaluate(img[t][ids], pro[t][ids], z[t][ids], h)
                lp_list.append(lp); ent_list.append(ent)
                if h_next is not None:
                    h = h_next * (1.0 - done[t][ids]).unsqueeze(-1)
            logp = torch.stack(lp_list)
            entropy = torch.stack(ent_list)
            Tm = T * ids.numel()
            v_new = critic(
                img[:, ids].reshape(Tm, *img.shape[2:]),
                pro[:, ids].reshape(Tm, pro.shape[2]),
                priv[:, ids].reshape(Tm, priv.shape[2]),
            ).reshape(T, ids.numel())
            adv_mb = advantages[:, ids]
            ret_mb = returns[:, ids]
            pi_loss = -(logp * adv_mb).mean()                 # vanilla policy gradient (no ratio/clip)
            v_loss = 0.5 * (v_new - ret_mb).pow(2).mean()
            ent_loss = -entropy.mean()
            w = ids.numel() / E                               # chunk weight → full-batch mean
            (w * (pi_loss + cfg.value_loss_coef * v_loss + cfg.entropy_coef * ent_loss)).backward()
            pi_loss_acc += float(pi_loss.item()) * w
            v_loss_acc += float(v_loss.item()) * w
            ent_acc += float(entropy.mean().item()) * w
        nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), cfg.max_grad_norm)
        agent.opt.step()

        if agent.hidden is not None:
            agent.hidden = agent.hidden.detach()

        # ---- Logging (same shared schema as SAC/PPO; algo-internal under a2c/) ----
        env_step = (it + 1) * T * E
        recent = finished_returns[-50:]
        mean_ret = sum(recent) / max(1, len(recent)) if recent else 0.0
        if writer is not None:
            now = time.time()
            sps = (env_step - _step_prev) / max(1e-6, now - _t_prev)
            _t_prev, _step_prev = now, env_step
            el = finished_lengths[-50:]
            cw = int(getattr(env, "_coll_n_wall", 0)); cs = int(getattr(env, "_coll_steps", 0))
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
            writer.add_scalar("a2c/policy_loss", pi_loss_acc, env_step)
            writer.add_scalar("a2c/value_loss", v_loss_acc, env_step)
            writer.add_scalar("a2c/entropy", ent_acc, env_step)
            writer.add_scalar("a2c/learning_rate", agent.lr, env_step)
            writer.add_scalar("a2c/explained_variance", explained_var, env_step)
            writer.add_scalar("env/max_wheel_rps", float(getattr(env, "_cur_max_rps", 0.0)), env_step)
        if it % cfg.log_interval == 0:
            print(
                f"[ra2c] iter {it:5d}  pi={pi_loss_acc:+.3f}  v={v_loss_acc:+.3f}  "
                f"H={ent_acc:+.3f}  lr={agent.lr:.1e}  ev={explained_var:+.2f}  "
                f"ret(50)={mean_ret:+.2f}  n_ep={len(finished_returns)}",
                flush=True,
            )
        if it > 0 and it % cfg.save_interval == 0:
            torch.save(
                {"actor": actor.state_dict(), "critic": critic.state_dict(),
                 "opt": agent.opt.state_dict(), "iter": it, "cfg": vars(cfg)},
                os.path.join(log_dir, f"ra2c_iter{it:06d}.pt"),
            )

    torch.save(
        {"actor": actor.state_dict(), "critic": critic.state_dict(),
         "opt": agent.opt.state_dict(), "iter": cfg.max_iterations, "cfg": vars(cfg)},
        os.path.join(log_dir, "ra2c_final.pt"),
    )
    if writer is not None:
        writer.close()
    print(f"[ra2c] DONE — checkpoints in {log_dir}", flush=True)
    return log_dir
