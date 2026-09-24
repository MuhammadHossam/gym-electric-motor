"""
Diagnostic run for the GEM closed loop, with the RL correction FROZEN AT
ZERO (delta_beta = 0 always -- pure nominal-table MTPA angle, no RL noise
at all).

Why: the smoke test showed Te_actual staying far below Te_target, and
getting WORSE over 300ms rather than settling -- that's not a startup
transient, something is genuinely broken. But the smoke test used a
random, untrained policy taking wild +/-10 degree jumps every 1ms, which
mixes two possible causes together:
  (a) the underlying physical loop itself (speed PI -> torque-to-current
      block -> current PI -> real plant) doesn't track correctly, or
  (b) the loop tracks fine on its own, but an untrained agent hopping the
      angle by up to 10 degrees every 1ms never lets the current PI settle
      onto any one reference.

This script isolates (a): it holds delta_beta at exactly 0 for the whole
run (same nominal MTPA angle your existing "analytical" torque_control
would produce), and records EVERY tau step (not just every 10th, like the
RL-facing env does) so we can see the actual transient shape, not just
sparse snapshots.

Plots, same 4-panel style as your closed_loop_foc_temperature_episodes.py:
  1. speed: actual vs. reference [RPM]
  2. i_sq: actual vs. commanded [A]
  3. i_sd: actual vs. commanded [A]
  4. torque: actual vs. the live reference from the speed loop [Nm]

Run this, look at the plot (and the printed summary), and send it back --
what we're looking for:
  - Does omega ever reach anywhere near omega_ref, or does it stay near 0?
  - Does i_sq actual ever catch up to i_sq commanded, or is there a
    persistent gap/opposite sign/wrong scale?
  - Does torque actual grow when i_sq commanded grows, or stay flat/drop?
"""
import numpy as np
import matplotlib.pyplot as plt

from gem_closed_loop_env import PMSMGemClosedLoopEnv, TAU, VDC


def main():
    env = PMSMGemClosedLoopEnv(seed=0)
    obs, info = env.reset(seed=1)
    print(f"T_true = {env._T_true:.1f} C")
    print(f"omega_ref (per-unit) = {env._omega_ref_gen._reference_value:.3f}")

    # Freeze the RL correction at exactly 0 for the whole run.
    env._pending_action = np.array([0.0], dtype=np.float32)

    N_STEPS = 50000   # 5000 * 100us = 500ms simulated
    t = np.arange(N_STEPS) * TAU

    ps = env.env.unwrapped.physical_system
    limits = ps.limits
    state_names = ps.state_names
    omega_idx = state_names.index("omega")
    i_sd_idx = state_names.index("i_sd")
    i_sq_idx = state_names.index("i_sq")
    torque_idx = state_names.index("torque")
    u_sd_idx = state_names.index("u_sd")
    u_sq_idx = state_names.index("u_sq")

    omega_act = np.empty(N_STEPS)
    omega_ref_arr = np.empty(N_STEPS)
    i_sd_act = np.empty(N_STEPS)
    i_sq_act = np.empty(N_STEPS)
    i_sd_cmd = np.empty(N_STEPS)
    i_sq_cmd = np.empty(N_STEPS)
    torque_act = np.empty(N_STEPS)
    torque_ref = np.empty(N_STEPS)
    beta_arr = np.empty(N_STEPS)
    modulation_a = np.empty(N_STEPS)   # 2*sqrt(vd^2+vq^2)/VDC -- voltage utilization,
                                        # a_max = 2/sqrt(3) ~= 1.1547 is the hard ceiling

    a_max = 2.0 / np.sqrt(3)

    state, reference = env._state, env._reference
    for k in range(N_STEPS):
        action = env.controller.control(state, reference)
        (state, reference), _, terminated, truncated, _ = env.env.step(action)
        if terminated or truncated:
            (state, reference), _ = env.env.reset()
            env.controller.reset()

        omega_act[k] = state[omega_idx] * limits[omega_idx]
        omega_ref_arr[k] = reference[0] * limits[omega_idx]
        i_sd_act[k] = state[i_sd_idx] * limits[i_sd_idx]
        i_sq_act[k] = state[i_sq_idx] * limits[i_sq_idx]
        torque_act[k] = state[torque_idx] * limits[torque_idx]
        i_sd_cmd[k] = env.torque_block.last_id_cmd
        i_sq_cmd[k] = env.torque_block.last_iq_cmd
        torque_ref[k] = env.torque_block.last_torque_target
        beta_arr[k] = env.torque_block.last_beta_deg

        vd = state[u_sd_idx] * limits[u_sd_idx]
        vq = state[u_sq_idx] * limits[u_sq_idx]
        modulation_a[k] = 2 * np.sqrt(vd ** 2 + vq ** 2) / VDC

    env._state, env._reference = state, reference

    rads_to_rpm = 30.0 / np.pi
    fig, axes = plt.subplots(5, 1, figsize=(11, 16), sharex=True)
    fig.suptitle(f"GEM closed-loop diagnostic -- RL correction FROZEN at 0 "
                 f"(nominal MTPA angle only), T_true={env._T_true:.1f}C",
                 fontsize=12, fontweight="bold")

    axes[0].plot(t, omega_act * rads_to_rpm, color="#2a78d6", label="omega actual [RPM]")
    axes[0].plot(t, omega_ref_arr * rads_to_rpm, "k--", label="omega ref [RPM]")
    axes[0].set_ylabel("Speed [RPM]")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.7)

    axes[1].plot(t, i_sq_act, color="#eb6834", label="i_sq actual [A]")
    axes[1].plot(t, i_sq_cmd, "k--", label="i_sq commanded [A]")
    axes[1].set_ylabel("i_sq [A]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.7)

    axes[2].plot(t, i_sd_act, color="#1baf7a", label="i_sd actual [A]")
    axes[2].plot(t, i_sd_cmd, "k--", label="i_sd commanded [A]")
    axes[2].set_ylabel("i_sd [A]")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle=":", alpha=0.7)

    axes[3].plot(t, torque_act, color="#4a3aa7", label="torque actual [Nm]")
    axes[3].plot(t, torque_ref, "k--", label="torque ref (speed loop output) [Nm]")
    axes[3].set_ylabel("Torque [Nm]")
    axes[3].set_xlabel("time [s]")
    axes[3].legend(loc="upper left")
    axes[3].grid(True, linestyle=":", alpha=0.7)

    axes[4].plot(t, modulation_a, color="#c0392b", label="modulation index a")
    axes[4].axhline(a_max, color="k", linestyle="--", label="a_max = 2/sqrt(3)")
    axes[4].set_ylabel("Voltage utilization")
    axes[4].set_xlabel("time [s]")
    axes[4].legend(loc="upper left")
    axes[4].grid(True, linestyle=":", alpha=0.7)

    fig.tight_layout()
    fig.savefig("diagnose_gem_closed_loop.png", dpi=150, bbox_inches="tight")
    print("\nsaved diagnose_gem_closed_loop.png")

    # =====================================================================
    # Interactive Plotly dashboard, same pattern as your
    # closed_loop_foc_temperature_episodes.py -- lets you zoom into the
    # exact ms where things go wrong instead of eyeballing a static PNG.
    # =====================================================================
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plotly_fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "1. Speed: actual vs. reference [RPM]",
            "2. i_sq: actual vs. commanded [A]",
            "3. i_sd: actual vs. commanded [A]",
            "4. Torque: actual vs. reference (speed loop output) [Nm]",
            "5. Voltage utilization: modulation index a vs. a_max (saturation ceiling)",
        ],
    )

    plotly_fig.add_trace(
        go.Scattergl(x=t, y=omega_act * rads_to_rpm, mode="lines", name="omega actual",
                     line=dict(color="#2a78d6", width=1.6)),
        row=1, col=1,
    )
    plotly_fig.add_trace(
        go.Scattergl(x=t, y=omega_ref_arr * rads_to_rpm, mode="lines", name="omega ref",
                     line=dict(color="black", dash="dash", width=1.6)),
        row=1, col=1,
    )

    plotly_fig.add_trace(
        go.Scattergl(x=t, y=i_sq_act, mode="lines", name="i_sq actual",
                     line=dict(color="#eb6834", width=1.6)),
        row=2, col=1,
    )
    plotly_fig.add_trace(
        go.Scattergl(x=t, y=i_sq_cmd, mode="lines", name="i_sq commanded",
                     line=dict(color="black", dash="dash", width=1.6)),
        row=2, col=1,
    )

    plotly_fig.add_trace(
        go.Scattergl(x=t, y=i_sd_act, mode="lines", name="i_sd actual",
                     line=dict(color="#1baf7a", width=1.6)),
        row=3, col=1,
    )
    plotly_fig.add_trace(
        go.Scattergl(x=t, y=i_sd_cmd, mode="lines", name="i_sd commanded",
                     line=dict(color="black", dash="dash", width=1.6)),
        row=3, col=1,
    )

    plotly_fig.add_trace(
        go.Scattergl(x=t, y=torque_act, mode="lines", name="torque actual",
                     line=dict(color="#4a3aa7", width=1.6)),
        row=4, col=1,
    )
    plotly_fig.add_trace(
        go.Scattergl(x=t, y=torque_ref, mode="lines", name="torque ref",
                     line=dict(color="black", dash="dash", width=1.6)),
        row=4, col=1,
    )

    plotly_fig.add_trace(
        go.Scattergl(x=t, y=modulation_a, mode="lines", name="modulation index a",
                     line=dict(color="#c0392b", width=1.6)),
        row=5, col=1,
    )
    plotly_fig.add_trace(
        go.Scattergl(x=t, y=np.full(N_STEPS, a_max), mode="lines", name="a_max (saturation)",
                     line=dict(color="black", dash="dash", width=1.6)),
        row=5, col=1,
    )

    plotly_fig.update_layout(
        title=dict(
            text=f"<b>GEM Closed-Loop Diagnostic (RL correction frozen at 0, "
                 f"T_true={env._T_true:.1f}°C)</b>",
            x=0.5, xanchor="center", font=dict(size=18),
        ),
        height=1350,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=60, r=40, t=110, b=50),
    )
    plotly_fig.update_xaxes(title_text="Time [s]", row=5, col=1, rangeslider=dict(visible=True, thickness=0.04))
    plotly_fig.update_yaxes(title_text="Speed [RPM]", row=1, col=1)
    plotly_fig.update_yaxes(title_text="i_sq [A]", row=2, col=1)
    plotly_fig.update_yaxes(title_text="i_sd [A]", row=3, col=1)
    plotly_fig.update_yaxes(title_text="Torque [Nm]", row=4, col=1)
    plotly_fig.update_yaxes(title_text="Modulation index a", row=5, col=1)

    html_path = "diagnose_gem_closed_loop_dashboard.html"
    plotly_fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"saved {html_path}")

    # printed summary too, in case the image is hard to eyeball precisely
    last = slice(-500, None)   # last 50ms
    print(f"\nlast 50ms summary:")
    print(f"  omega:  actual={np.mean(omega_act[last])*rads_to_rpm:8.1f} RPM   "
          f"ref={np.mean(omega_ref_arr[last])*rads_to_rpm:8.1f} RPM")
    print(f"  i_sq:   actual={np.mean(i_sq_act[last]):7.3f} A   cmd={np.mean(i_sq_cmd[last]):7.3f} A")
    print(f"  i_sd:   actual={np.mean(i_sd_act[last]):7.3f} A   cmd={np.mean(i_sd_cmd[last]):7.3f} A")
    print(f"  torque: actual={np.mean(torque_act[last]):7.3f} Nm  ref={np.mean(torque_ref[last]):7.3f} Nm")
    print(f"  modulation index a: mean={np.mean(modulation_a[last]):.4f}  "
          f"max={np.max(modulation_a[last]):.4f}  (a_max={a_max:.4f} -- at/near this means "
          f"voltage-saturated, i.e. flux-weakening territory)")
    print(f"  beta (should be ~constant, RL frozen at 0): "
          f"mean={np.mean(beta_arr[last]):.2f} deg  std={np.std(beta_arr[last]):.4f} deg")


if __name__ == "__main__":
    main()
