"""
Batch evaluation for the trained SAC beta-correction agent.

The 10-row table we've been eyeballing in chat is a fixed, tiny sample (one
seed, 10 scenarios) -- good for spot-checking, not for trusting the result.
This script runs the trained model over a large number of RANDOM scenarios
(default 500), spread across the full torque and magnet-temperature ranges,
and reports:
  - excess loss vs. the TRUE oracle (the number that actually matters --
    Is^2 is proportional to copper loss), as mean/median/p90/p99/max
  - beta angle error vs. the oracle, same statistics
  - a breakdown by torque tertile (low/mid/high) and temperature tertile
    (cool/mid/hot), so a weak region (e.g. low-torque, hot) doesn't hide
    inside a good-looking overall average
  - a CSV of every episode's raw numbers, for your own slicing
  - a scatter plot: excess loss (%) vs. commanded torque, colored by
    magnet temperature, so any structure (e.g. "still bad at low torque")
    is visible at a glance rather than read off a table

The oracle here uses the TRUE psi_f (privileged/training-only info), same
as r_oracle in rl_kickoff_sac.py -- this script is for evaluation, not for
anything the agent itself gets to see.

Usage:
    python batch_eval_sac.py --model sac_pmsm_kickoff_beta_v3 --episodes 500
"""
import argparse
import csv

import numpy as np

from rl_kickoff_sacV2 import (
    PMSMEfficiencyEnv, psi_of_T, solve_true_optimum, I_RATED,
)

# palette (dataviz skill reference palette, sequential blue + status colors)
COLOR_COOL = "#86b6ef"     # step 250, coolest magnet temp
COLOR_HOT = "#0d366b"      # step 700, hottest magnet temp
COLOR_WARN = "#fab219"
COLOR_CRIT = "#d03b3b"
COLOR_GOOD = "#0ca30c"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def run_episode(raw_env, predict_fn):
    """One episode: reset, predict, step. Returns a dict of everything
    needed for the agent-vs-oracle comparison."""
    obs, _ = raw_env.reset()
    action = predict_fn(obs)
    obs2, reward, term, trunc, info = raw_env.step(action)

    Te_target = raw_env._Te_target
    T_true = raw_env._T_true
    psi_true = psi_of_T(T_true)

    id_star, iq_star = solve_true_optimum(Te_target, psi_true)
    Is_star = float(np.hypot(id_star, iq_star))
    beta_star = float(np.degrees(np.arctan2(-id_star, iq_star)))

    id_agent, iq_agent = info["id_cmd"], info["iq_cmd"]
    Is_agent = info["Is"]
    beta_agent = info["beta_deg"]

    excess_loss_pct = 100.0 * (Is_agent ** 2 - Is_star ** 2) / (Is_star ** 2 + 1e-12)

    return dict(
        Te_target=Te_target, T_true=T_true,
        beta_agent=beta_agent, beta_star=beta_star, dbeta=beta_agent - beta_star,
        id_agent=id_agent, id_star=id_star,
        iq_agent=iq_agent, iq_star=iq_star,
        Is_agent=Is_agent, Is_star=Is_star,
        excess_loss_pct=excess_loss_pct, reward=reward,
    )


def summarize(rows, key, label):
    vals = np.array([r[key] for r in rows], dtype=float)
    print(f"{label:>28}: mean={np.mean(vals):8.4f}  median={np.median(vals):8.4f}  "
          f"p90={np.percentile(vals, 90):8.4f}  p99={np.percentile(vals, 99):8.4f}  "
          f"max={np.max(vals):8.4f}")
    return vals


def tertile_breakdown(rows, split_key, split_label, metric_key="excess_loss_pct"):
    vals = np.array([r[split_key] for r in rows])
    edges = np.percentile(vals, [33.3, 66.7])
    bins = np.digitize(vals, edges)
    names = ["low", "mid", "high"]
    print(f"\n  by {split_label} tertile (mean excess loss %):")
    for b, name in enumerate(names):
        mask = bins == b
        if mask.sum() == 0:
            continue
        m = np.array([r[metric_key] for r in rows])[mask]
        lo = vals[mask].min()
        hi = vals[mask].max()
        print(f"    {name:>4} ({lo:6.2f}-{hi:6.2f}): n={mask.sum():4d}  "
              f"mean excess loss={np.mean(m):7.4f}%  max={np.max(m):7.4f}%")


def make_plot(rows, out_path="batch_eval_scatter.png"):
    import matplotlib.pyplot as plt

    Te = np.array([r["Te_target"] for r in rows])
    T = np.array([r["T_true"] for r in rows])
    exc = np.array([r["excess_loss_pct"] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    sc = ax.scatter(Te, exc, c=T, cmap="Blues", s=18, alpha=0.75,
                     edgecolors="none", vmin=25, vmax=120)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("magnet temperature (°C)", color=INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_SECONDARY)

    ax.axhline(0, color=INK_MUTED, linewidth=1, linestyle="--", zorder=0)
    ax.set_xlabel("commanded torque $T_e$ (Nm)", color=INK_SECONDARY)
    ax.set_ylabel("excess loss vs. oracle (%)", color=INK_SECONDARY)
    ax.set_title(f"Batch eval: {len(rows)} random scenarios, agent vs. true oracle\n"
                 "excess loss = (Is_agent² - Is_oracle²) / Is_oracle² × 100",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nsaved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sac_pmsm_kickoff_beta_v3",
                         help="path to the saved SB3 model (without .zip)")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", default="batch_eval_results.csv")
    parser.add_argument("--plot", default="batch_eval_scatter.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    from stable_baselines3 import SAC
    model = SAC.load(args.model)

    def predict_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    raw_env = PMSMEfficiencyEnv(seed=args.seed)
    rows = [run_episode(raw_env, predict_fn) for _ in range(args.episodes)]

    print(f"\n=== batch eval: {args.episodes} random scenarios, model={args.model} ===\n")
    summarize(rows, "excess_loss_pct", "excess loss vs oracle (%)")
    summarize(rows, "dbeta", "beta error vs oracle (deg)")
    tertile_breakdown(rows, "Te_target", "commanded torque")
    tertile_breakdown(rows, "T_true", "magnet temperature")

    # flag the worst offenders directly, not just their aggregate stats
    worst = sorted(rows, key=lambda r: -r["excess_loss_pct"])[:5]
    print("\n  5 worst-case scenarios (highest excess loss):")
    print(f"  {'Te':>6} {'T':>6} {'dbeta':>7} {'excess_loss%':>13}")
    for r in worst:
        print(f"  {r['Te_target']:6.2f} {r['T_true']:6.1f} {r['dbeta']:7.2f} "
              f"{r['excess_loss_pct']:13.4f}")

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {args.csv}  ({len(rows)} rows)")

    if not args.no_plot:
        make_plot(rows, args.plot)


if __name__ == "__main__":
    main()