# GEM Closed-Loop RL Training Env — Session Summary

Context handoff doc. Paste this into a new chat (or point Claude at it) to
resume this work without re-explaining everything.

## Goal

Mohamed has an RL agent (`rl_kickoff_sacV2.py`, SAC via stable-baselines3)
that learns a **beta-angle correction on top of a nominal-table MTPA
baseline**, for a TI SPRACF3 ferrite-magnet PMSM (HVAC/appliance-class).
It was already trained and validated on a closed-form analytic physics
stand-in (near-zero excess loss over 500 random batch-eval scenarios).

Next step (this session): retrain from a **real gym-electric-motor (GEM)
closed-loop simulation** instead of the analytic stand-in, since that's
closer to the real hardware case.

## Architecture

- Keep Mohamed's existing, working speed-control + current-control PI
  cascade (`classic_controllers.Controller`, `ControlType.SpeedControl`)
  fully intact. RL does **not** touch the speed loop or current loops.
- RL replaces **only** the torque-to-current (MTPA) conversion block, via
  the same runtime-attribute-swap pattern Mohamed's own
  `closed_loop_foc_adaptive_mtpa_temperature_episodes.py` already uses:
  `controller.torque_controller = RLBetaTorqueToCurrent(...)`.
- RL acts once every 10 tau cycles (tau=100us -> 1ms), holding
  `delta_beta` fixed between decisions while the real current PI + plant
  continue running every tau cycle underneath it.
- `RLBetaTorqueToCurrent.control(state, torque)`: computes a nominal
  (25°C, `PSI_NOM`) MTPA baseline angle `beta0` via closed-form + brentq,
  adds the RL's `delta_beta` correction, then solves for the current
  magnitude `Is` needed to hit the torque target at that angle
  (`solve_Is_for_beta`, closed-form quadratic), decomposes back to
  `(id, iq)` (`id_iq_from_beta_Is`), and returns them as the `(iq_ref,
  id_ref)` the current PI tracks. Deliberately uses `PSI_NOM`, not the
  true (temperature-drifted) `psi_p` — it's meant to be a sensor-only
  controller; correcting for the unknown true flux via `delta_beta` is
  the RL agent's whole job.

## Key files

All in `C:\Hossam\Repositories\gym-electric-motor\examples\userdefinedExamples\MTPA\`:

- `rl_kickoff_sacV2.py` — canonical analytic-env RL script (already
  validated). Exports the shared physics helpers
  (`P, LD, LQ, RS, PSI_NOM, I_RATED, VDC, S, psi_of_T, torque, copper_loss,
  id_iq_from_beta_Is, solve_Is_for_beta, solve_true_optimum`, etc.) that
  `gem_closed_loop_env.py` imports and reuses.
- `gem_closed_loop_env.py` — **the new file this session built**: the
  real-GEM closed-loop `gym.Env` (`PMSMGemClosedLoopEnv`) wrapping GEM +
  the PI cascade + `RLBetaTorqueToCurrent`. Not yet wired into actual SAC
  training — environment-building and diagnostics only so far.
- `diagnose_gem_closed_loop.py` — diagnostic script: runs the closed loop
  with RL frozen at `delta_beta=0` (pure nominal MTPA, no RL noise) and
  dumps a 5-panel Plotly dashboard (speed, i_sq, i_sd, torque, voltage
  utilization) + PNG + console summary. This has been the main tool for
  debugging.
- `smoke_test_gem_env.py` — quick sanity script with random actions.
  Written early, not touched recently; should be re-run now that the
  bugs below are fixed (see Next Steps).
- `batch_eval_sac.py` — the analytic-env batch evaluation script (500
  random scenarios, near-zero excess loss). Not part of the GEM work.

## Bugs found and fixed this session (chronological)

1. **Import path bug**: `gem_closed_loop_env.py` computed the
   `classic_controllers` path one level too shallow (copied from scripts
   living one directory up). Fixed.
2. **`we` observation-space lower bound**: was `WE_RANGE[0]=100` (carried
   over from the analytic env), but the real rotor starts at rest
   (`we≈0`). Fixed to `-10.0`.
3. **Stale import**: was importing `rl_kickoff_sac` instead of
   `rl_kickoff_sacV2`. Fixed (explicit user correction).
4. **Zero-headroom current/torque limit → repeated env resets (the big
   one)**: `nominal_values["i"]` and `limit_values["i"]` were both set to
   `I_RATED` — zero headroom between "rated" and GEM's hard termination
   ceiling. `RLBetaTorqueToCurrent` commands current right up to
   `I_RATED` too, so any normal PI overshoot crossed the ceiling,
   terminating the episode; the diagnostic script's own loop immediately
   reset state+controller, over and over (~every 2-3ms) — this produced
   a very distinctive sawtooth pattern in every channel (visually
   confirmed via the dashboard) and was the reason `omega` never got
   past ~2000 RPM before snapping back to 0. **Fixed**: added headroom —
   `I_LIMIT = 1.2 * I_RATED` for current, `T_base = 1.2 * T_rated` for
   torque, used only in `limit_values` (the `nominal_values` /
   `RLBetaTorqueToCurrent`'s own clip stayed at `I_RATED`).
5. **Voltage saturation / flux-weakening gap**: after fixing #4, a
   different run showed `omega` plateauing well below `omega_ref` with
   the current PI unable to track a still-climbing commanded current —
   confirmed via an added "modulation index a vs a_max" diagnostic panel
   (`a` was pinned above `a_max=1.1547`). Root cause: `omega_ref_range`
   was `(0.3, 0.9)` pu, which pushes into flux-weakening territory, but
   `RLBetaTorqueToCurrent` has **no flux-weakening logic** (unlike GEM's
   real `TorqueToCurrentConversion`, which has `modulation_control()`).
   **Fixed for now** by narrowing `omega_ref_range` to `(0.15, 0.35)` —
   stays inside the MTPA/constant-torque region. (Adding real
   flux-weakening to `RLBetaTorqueToCurrent` was explicitly deferred by
   Mohamed: "I don't need to work in field weakening region now, decrease
   the speed request".)
6. **`torque_ref` > `torque_act` steady-state gap** — investigated and
   confirmed **expected, not a bug**: `RLBetaTorqueToCurrent` computes
   `id`/`iq` assuming `PSI_NOM`, but the real plant's flux is
   `psi_of_T(T_true)` (lower at high temp, ferrite magnet). The speed
   loop's integral action naturally settles with `torque_ref` sitting
   above `torque_act` by exactly the amount needed to compensate — this
   is healthy PI behavior under a persistent, one-directional model bias,
   not windup or instability. It's exactly the gap RL's `delta_beta` is
   meant to close once trained.
7. **Load coupling (physical constraint)**: load torque was previously
   fixed at `t_load=0.3` regardless of speed. Mohamed asked to vary load
   across episodes for training generalization, but flagged that torque
   and speed can't both be high without flux-weakening (which isn't
   implemented — see #5). **Fixed**: added
   `max_voltage_limited_torque(we, psi_f, margin)` — bisection solve for
   the largest MTPA torque achievable at a given electrical speed without
   exceeding the voltage limit (includes the Rs drop term). `reset()` now
   samples `omega_ref` first, then samples `t_load` uniformly from
   `[0, LOAD_MARGIN * ceiling]` with `LOAD_MARGIN = 0.8` (Mohamed's
   choice), using the **true** (temperature-drifted) flux for the
   ceiling calc since that's what the real plant experiences.
   `ConfigurableLoad.t_load` is now mutated per-episode in `reset()`
   rather than fixed at construction.
8. **Near-miss termination at an extreme (T_true=114.6°C, near-ceiling
   load) seed**: diagnostic showed a sawtooth pattern again, visually
   identical to bug #4. Added per-reset instrumentation to
   `diagnose_gem_closed_loop.py` (prints which state variable is at/over
   its limit on each termination). Root cause confirmed: **not** a new
   bug — `i_b` (a phase current, ABC frame) hit `12.010 A` against the
   `12.000 A` limit (`1.2 * I_RATED`, confirming `I_RATED=10A`) — a
   0.08% overshoot from normal PI transient response right after
   `reset()`, in the single hardest case in the sampling range (hottest
   temperature + near-ceiling load compounding). **Left as-is per
   Mohamed** ("It is working now") — a further headroom increase (e.g.
   1.2x -> 1.3x) was proposed but not applied; worth revisiting if this
   recurs during actual training.
9. **Episode settling gate**: `EPISODE_DURATION_S=0.5s` is *shorter*
   than the ~1s physical settling time Mohamed observed empirically —
   meaning most training transitions would have been transient response
   (governed by the PI cascade, not RL) rather than steady-state
   operation (what beta-correction actually addresses). **Fixed**: added
   `_burn_in_until_settled()` — `reset()` now runs the loop tau-by-tau
   with `delta_beta` frozen at 0 until `omega` stays within
   `SETTLE_OMEGA_TOL=0.05` (5%) of `omega_ref` continuously for
   `SETTLE_HOLD_S=0.5` (500ms), or gives up after
   `SETTLE_TIMEOUT_S=2.0` (2s) and starts the episode anyway (so
   hard-to-settle sampled combinations aren't silently excluded from
   training). Added `reset(options={"skip_burn_in": True})` so
   `diagnose_gem_closed_loop.py` can still see the raw startup transient
   (it now passes this flag).

## Known placeholders / open items (not yet resolved)

- **`ROTOR_J = 5e-5 kg*m^2`** — still a guess, never independently
  verified against a real datasheet/measurement. Behaves consistently
  across scripts so not blocking, but should be replaced before trusting
  final results.
- **Current/speed PI gains** — computed via the same symmetric-optimum
  formula Mohamed's other scripts use, re-evaluated at the TI SPRACF3
  Ld/Lq/Rs numbers, but not independently tuned/validated for damping
  quality across the full operating range.
- **Rs temperature drift** — not modeled; only `psi_p` (magnet flux)
  drifts with `T_true`. Same simplification as the analytic env.
- **No flux-weakening in `RLBetaTorqueToCurrent`** — deliberately scoped
  out for now (see bug #5); `omega_ref_range` was narrowed instead of
  implementing it. Revisit if higher-speed operation is needed later.
- **1.2x current/torque headroom margin** — held up in one near-miss
  case (bug #8) but wasn't widened. Watch for recurrence during actual
  training with many more random seeds.
- **Burn-in wall-clock cost** — `reset()` can now take up to 2s of
  simulated burn-in per episode before the "real" episode starts. Not
  yet measured how this affects training throughput.

## Tooling note

The `mcp__remote-devices__device_commit_files` tool has repeatedly
reported success (`"written": [...]`) while silently landing a **stale**
version of the file on Mohamed's machine (missing the most recent edits).
Established workaround, used consistently from partway through this
session onward: after every commit, re-stage the file
(`device_stage_files`) and `diff` it against the local sandbox copy to
confirm they're byte-identical; retry the commit if not. **Keep doing
this for every future file commit in this project.**

## Next steps (where this session left off)

1. Re-run `smoke_test_gem_env.py` (random, unfrozen RL actions) now that
   bugs #4, #5, #7, #9 are all fixed — this was the original plan before
   the reset-bug investigation took over, and hasn't been revisited
   since.
2. Measure typical burn-in time (`_burn_in_until_settled`) across a
   handful of seeds to understand the wall-clock cost added to training.
3. Wire `PMSMGemClosedLoopEnv` into actual SAC training in
   `rl_kickoff_sacV2.py` (swap it in for the analytic `PMSMEfficiencyEnv`)
   — this has **not been done yet**; everything so far has been
   environment-building and diagnostics only.
4. Once trained, validate the same way the analytic-env agent was
   validated (large-N random batch eval, `batch_eval_sac.py`-style) but
   against the real GEM closed loop.
5. Resolve the open placeholders above (`ROTOR_J`, PI gain validation) as
   time allows — not blocking but flagged for before trusting final
   results.
