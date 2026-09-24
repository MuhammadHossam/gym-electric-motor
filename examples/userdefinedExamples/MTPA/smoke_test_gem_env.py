"""
Smoke test for PMSMGemClosedLoopEnv -- RUN THIS FIRST, before wiring SAC in.

gem_closed_loop_env.py was written without access to gym-electric-motor or
your classic_controllers package (they're not installed in the sandbox this
was built in), so it has only been checked for Python syntax, never
executed. This script just resets the env and takes a handful of random
actions, printing shapes and values along the way, so any wiring problem
(index mismatch, unit mismatch, controller API mismatch) shows up here --
cheap and fast -- instead of after a training run.

If this throws, paste the full traceback back and we'll fix it the same way
we've fixed everything else in this project.
"""
import numpy as np

from gem_closed_loop_env import PMSMGemClosedLoopEnv


def main():
    print("building env + controller...")
    env = PMSMGemClosedLoopEnv(seed=0)
    print("observation_space:", env.observation_space.shape)
    print("action_space:", env.action_space.shape)

    print("\nreset()...")
    obs, info = env.reset(seed=1)
    print("obs shape:", obs.shape, " (expected:", env.observation_space.shape, ")")
    print("obs[:6]:", obs[:6])
    print("T_true (privileged, not in obs):", env._T_true)

    # 300 decisions * 1ms/decision = 300ms simulated time -- long enough for
    # the current PI (~0.8ms time constant at 200Hz) and the mechanical
    # dynamics to clearly settle past the cold-start transient (rotor at
    # rest -> ramping up), so a persistent tracking gap late in the run
    # means a real tuning/wiring problem, not just "hasn't settled yet".
    N_STEPS = 300
    print(f"\nstepping {N_STEPS} times with random actions "
          f"({N_STEPS}ms simulated, well past the cold-start transient)...")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(N_STEPS):
        action = np.array([rng.uniform(-10, 10)], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        rows.append(dict(i=i, reward=reward, **info))
        if i < 10 or i % 50 == 0:
            print(f"  step {i:3d}: reward={reward:8.4f}  "
                  f"Te_target={info['Te_target']:6.3f}  Te_actual={info['Te_actual']:6.3f}  "
                  f"beta={info['beta_deg']:7.2f}  id_cmd={info['id_cmd']:7.3f}  "
                  f"iq_cmd={info['iq_cmd']:7.3f}  T_true={info['T_true']:6.1f}")
        if term or trunc:
            obs, info = env.reset()

    print("\nOK -- no exception raised.")
    print("\ntracking-error trend (mean |Te_actual - Te_target| / Te_target):")
    for label, chunk in [("first 20 steps (cold-start transient)", rows[:20]),
                         ("last 50 steps (should be settled by now)", rows[-50:])]:
        err_pct = [100 * abs(r["Te_actual"] - r["Te_target"]) / max(r["Te_target"], 1e-6) for r in chunk]
        print(f"  {label}: mean={np.mean(err_pct):6.1f}%  max={np.max(err_pct):6.1f}%")
    print("\nIf the 'last 50 steps' error is still large (tens of percent), that's a")
    print("real gain/wiring problem, not just startup transient -- paste this back and")
    print("we'll dig into the current-PI gains / decoupling terms next.")
    id_cmds = [r["id_cmd"] for r in rows]
    print(f"\nid_cmd sign check: {sum(1 for x in id_cmds if x < 0)}/{len(id_cmds)} negative "
          f"(should be nearly all -- id_cmd>0 is the wrong MTPA region)")


if __name__ == "__main__":
    main()
