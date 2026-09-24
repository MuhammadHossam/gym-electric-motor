"""
Step 1 kickoff: a minimal, self-contained SAC training script for the
PMSM efficiency-correction RL problem.

STATE FIX (v3): earlier versions of this script put a value derived
directly from the TRUE magnet temperature (T_true) into the observation,
mislabeled in comments as "T_wind_proxy". That is not available on real
hardware -- there is no magnet-temperature sensor -- and it let the agent
"cheat" by reading the answer instead of inferring it. The state now
matches Step 0's original design (generate_dataset.py): the agent gets
[Te_target, we, Vdc] plus a short STACKED WINDOW of noisy (vd, vq, id, iq)
measurements at the baseline (nominal-table) operating point. Any
temperature information the agent uses has to come from the pattern in
those noisy voltage/current numbers, exactly like a real drive would have
to infer it.

ACTION SPACE v2: the agent now outputs a single number, delta_beta (a
correction, in degrees, to the current ANGLE) instead of (delta_id,
delta_iq). Given beta, Is is recovered from the torque equation directly
as a CLOSED-FORM QUADRATIC -- no root-find, no fsolve:

    Te = 1.5*p*psi_f*cos(beta)*Is - 1.5*p*s*sin(beta)*cos(beta)*Is^2
       = A*Is^2 + B*Is   (with A = -1.5*p*s*sin(beta)*cos(beta),
                                B =  1.5*p*psi_f*cos(beta))
    Is = (-B + sqrt(B^2 + 4*A*Te_target)) / (2*A)   [or Te_target/B if A~0]

This makes torque tracking EXACT by construction whenever the required Is
is feasible (<= I_RATED) -- the agent never has to learn to hit torque,
only to pick a good angle. When the required Is would exceed I_RATED (or
no real solution exists), current is clipped to I_RATED at that same
angle -- exactly the "torque saturation" failure mode we characterized
earlier (speed droops, current never exceeds the rating) -- and the
reward penalizes both the shortfall this causes AND the raw (unclipped)
Is that would have been needed, so the agent learns to steer away from
that region, not just to survive being clipped there.

WHY THIS SHAPE: don't debug RL and the environment at the same time. This
script uses a FAST, built-in physics environment (same equations as
pmsm_model.py / mtpa_generic_solver.py -- TI SPRACF3 motor, ferrite magnet)
instead of your real GEM setup, so you can confirm the SAC + reward +
action-space pipeline actually works BEFORE plugging in the slower, more
complex GEM environment. Once this trains and the reward curve climbs,
swap PMSMEfficiencyEnv's step() internals for a call into your GEM env --
everything else (SAC config, action/state spec, reward shaping) carries
over unchanged.

Requires: pip install stable-baselines3 gymnasium torch
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------
# Motor physics (TI SPRACF3, ferrite magnet) -- same as mtpa_generic_solver.py
# ---------------------------------------------------------------------
P = 3
LD = 1.532e-3
LQ = 7.324e-3
RS = 0.130185
PSI_NOM = 0.2084
T_REF = 25.0
I_RATED = 10.0
VDC = 200.0
ALPHA_PSI_FERRITE = -0.0020
S = LD - LQ

# how far (in degrees) the agent may correct the baseline angle. The real
# temperature-driven shift we measured earlier tops out around 2.5-3 deg
# across the full 25-120C range, so this gives roughly 3x headroom for
# exploration without letting the agent wander into nonphysical angles.
MAX_DELTA_BETA_DEG = 10.0

# electrical speed range the state is sampled over (rad/s) -- needed now
# because the state includes measured vd, vq, which depend on speed via
# the back-EMF / cross-coupling terms.
WE_RANGE = (100.0, 550.0)

# noisy sensor window -- same shape as Step 0's generate_dataset.py, so the
# agent is solving exactly the "infer drift from noisy measurements" problem
# the rest of the project is built around, not an easier one.
N_WINDOW = 5
CURRENT_NOISE_STD = 0.03          # A
VOLTAGE_NOISE_STD_FRAC = 0.01     # fraction of Vdc


def steady_state_voltage(id_, iq, we, psi_f):
    vd = RS * id_ - we * LQ * iq
    vq = RS * iq + we * LD * id_ + we * psi_f
    return vd, vq


def psi_of_T(T):
    return PSI_NOM * (1.0 + ALPHA_PSI_FERRITE * (T - T_REF))


def torque(id_, iq, psi_f):
    return 1.5 * P * (psi_f * iq + S * id_ * iq)


def copper_loss(id_, iq):
    return 1.5 * RS * (id_ ** 2 + iq ** 2)


def id_iq_from_beta_Is(beta_rad, Is):
    return -Is * np.sin(beta_rad), Is * np.cos(beta_rad)


def _mtpa_id_closed_form(iq, psi_f):
    """Closed-form MTPA id(iq) -- same formula as pmsm_model.py's mtpa_id /
    mtpa_vs_temperature.py's mtpa_id / the env's own _mtpa_id_at. Valid for
    Ld < Lq (this motor)."""
    A = psi_f / (2.0 * S)
    return -A - np.sqrt(A ** 2 + iq ** 2)


def solve_true_optimum(Te_target, psi_f, guess=None):
    """The TRUE oracle (id*, iq*) for Te_target at the given psi_f.

    NOTE (fixed): this used to be a 2-equation/2-unknown fsolve (torque eq +
    MTPA eq together). That's fine for a handful of hand-picked eval points
    with a good initial guess, but it started throwing RuntimeError once
    r_oracle made this run inside every training episode's reset() (tens of
    thousands of random Te_target/psi_f draws) -- fsolve occasionally can't
    converge on some of them.

    Fix: solve it the same robust way the rest of this project already does
    (pmsm_model.py's find_optimal_id_iq, mtpa_vs_temperature.py's
    id_at_torque) -- substitute the closed-form id(iq) into the torque
    equation, which makes Te a MONOTONIC function of iq alone. That turns
    this into a 1-D root-find (brentq), which doesn't fail the way a 2-D
    Newton solve can. `guess` is accepted for backward compatibility but is
    no longer needed.
    """
    from scipy.optimize import brentq

    def Te_of_iq(iq):
        return torque(_mtpa_id_closed_form(iq, psi_f), iq, psi_f)

    iq_lo = 1e-6
    iq_hi = 5.0 * I_RATED  # generous headroom: Te_target is sampled against
    # the NOMINAL (25C) current-limited max, so at a hotter/weaker psi_f the
    # unconstrained oracle can need more than I_RATED -- that's fine here,
    # this is the theoretical optimum, not a feasibility check.
    while Te_of_iq(iq_hi) < Te_target and iq_hi < 100.0 * I_RATED:
        iq_hi *= 2.0  # widen further in the rare/extreme case

    iq = brentq(lambda x: Te_of_iq(x) - Te_target, iq_lo, iq_hi, xtol=1e-10)
    id_ = _mtpa_id_closed_form(iq, psi_f)
    return id_, iq


def solve_Is_for_beta(beta_rad, Te_target, psi_f):
    """Closed-form quadratic solve for Is given a FIXED angle beta and a
    torque target -- one equation, one unknown (derived in the chat: beta
    is the agent's choice, Is is what's left to solve for). Returns
    (Is, feasible) where feasible=False means either no real solution, or
    the solution is negative/nonsensical -- caller decides what to do."""
    A = -1.5 * P * S * np.sin(beta_rad) * np.cos(beta_rad)
    B = 1.5 * P * psi_f * np.cos(beta_rad)

    if abs(A) < 1e-9:
        if abs(B) < 1e-9:
            return 0.0, False
        Is = Te_target / B
        return Is, Is > 0
    disc = B ** 2 + 4 * A * Te_target
    if disc < 0:
        return np.nan, False
    Is = (-B + np.sqrt(disc)) / (2 * A)
    return Is, Is > 0


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
class PMSMEfficiencyEnv(gym.Env):
    """
    One episode = one randomly sampled (torque command, magnet temperature)
    operating point, matching Step 0's framing.

    Action:  [delta_beta] (degrees), a SINGLE number -- a correction to the
             25C-nominal MTPA baseline angle beta0 for the commanded torque.
             Is is then recovered in closed form (see solve_Is_for_beta),
             which makes torque tracking exact whenever feasible.
    State:   [Te_target, we, Vdc] + N_WINDOW stacked noisy (vd, vq, id, iq)
             measurements at the BASELINE (nominal-table) operating point --
             same shape as Step 0's generate_dataset.py. NO temperature of
             any kind is in this vector, true or proxy: a real drive has no
             magnet-temperature sensor, so the agent must infer flux drift
             (if it uses it at all) from the pattern in these noisy
             voltage/current numbers, exactly like the plant it will
             eventually run on.
    Reward:
        - r_torque: penalizes torque SHORTFALL, which now only happens
          when the requested angle needs more current than I_RATED allows
          (the "torque saturation" case) -- zero whenever feasible, since
          torque is then exact by construction.
        - r_current: penalizes the RAW (unclipped) Is that beta would have
          required, even when we end up clipping it -- this is what
          teaches the agent to avoid angles that eat into current margin,
          not just to tolerate being clipped there.
        - r_loss: small term, efficiency is the fine-tuning objective, not
          the dominant one (see the earlier current-margin discussion).
        - r_id_sign: penalizes id > 0. Copper loss depends on id^2, so it
          can't tell a wrong-sign id from a correct-sign one of the same
          magnitude -- verified in chat that a mirrored (correct-sign) id
          gives IDENTICAL loss to a wrong-sign id of equal |id|. Without
          this term the agent can (and did, at low torque where |id*| is
          small) land in the physically-wrong MTPA region while still
          getting a good loss score. This term makes the sign matter.
        - r_oracle: direct squared-error penalty against the TRUE optimal
          (id*, iq*), computed once per episode with the TRUE psi_f
          (privileged/training-only info -- never added to the observation).
          Added because r_loss alone was too weak a signal: loss is locally
          FLAT near the MTPA optimum, so a several-degree beta error costs
          almost nothing in loss even though it's a poor angle match. This
          term stays informative exactly where r_loss doesn't.
    """

    metadata = {"render_modes": []}

    def __init__(self, T_range=(25.0, 120.0), Te_frac_range=(0.15, 0.9),
                 we_range=WE_RANGE, seed=None):
        super().__init__()
        self.T_range = T_range
        self.Te_frac_range = Te_frac_range
        self.we_range = we_range
        self.rng = np.random.default_rng(seed)

        # action: delta_beta in degrees, a single continuous number
        self.action_space = spaces.Box(
            low=np.array([-MAX_DELTA_BETA_DEG], dtype=np.float32),
            high=np.array([MAX_DELTA_BETA_DEG], dtype=np.float32),
        )
        # state: [Te_target, we, Vdc] + N_WINDOW * (vd, vq, id, iq) -- no
        # temperature term, true or proxy (see class docstring).
        v_bound = VDC  # generous bound on measured vd/vq; true values stay well inside
        i_bound = I_RATED + 1.0  # a little headroom over I_rated for sensor noise
        low = [0.0, we_range[0], VDC] + [-v_bound, -v_bound, -i_bound, -i_bound] * N_WINDOW
        high = [12.0, we_range[1], VDC] + [v_bound, v_bound, i_bound, i_bound] * N_WINDOW
        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
        )

        self._Te_target = None
        self._T_true = None
        self._we = None
        self._id0 = None
        self._iq0 = None
        self._beta0_deg = None
        self._obs = None

    def _mtpa_id_at(self, iq, psi_f):
        A = psi_f / (2.0 * S)
        return -A - np.sqrt(A ** 2 + iq ** 2)

    def _baseline_25C(self, Te_target):
        """25C-nominal MTPA baseline (id0, iq0) for Te_target -- same
        closed-form solve as pmsm_model.py, root-find on iq."""
        from scipy.optimize import brentq
        psi_nom = PSI_NOM

        def Te_of_iq(iq):
            return torque(self._mtpa_id_at(iq, psi_nom), iq, psi_nom)

        iq = brentq(lambda x: Te_of_iq(x) - Te_target, 1e-6, I_RATED, xtol=1e-8)
        id_ = self._mtpa_id_at(iq, psi_nom)
        return id_, iq

    def _noisy_window(self, id0, iq0, we, psi_true):
        """N_WINDOW noisy (vd, vq, id, iq) samples at the baseline
        (nominal-table) operating point, on the TRUE plant -- same
        construction as generate_dataset.py. This is the only place any
        temperature-related information can enter the state, and only
        implicitly, through how vd/vq/id/iq happen to look given the true
        (drifted) psi_f -- never as a direct temperature value."""
        vd_true, vq_true = steady_state_voltage(id0, iq0, we, psi_true)
        v_noise_std = VOLTAGE_NOISE_STD_FRAC * VDC
        window = []
        for _ in range(N_WINDOW):
            vd_n = vd_true + self.rng.normal(0, v_noise_std)
            vq_n = vq_true + self.rng.normal(0, v_noise_std)
            id_n = id0 + self.rng.normal(0, CURRENT_NOISE_STD)
            iq_n = iq0 + self.rng.normal(0, CURRENT_NOISE_STD)
            window.extend([vd_n, vq_n, id_n, iq_n])
        return window

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        Te_max_nom = torque(0.0, I_RATED, PSI_NOM)
        self._Te_target = self.rng.uniform(*self.Te_frac_range) * Te_max_nom
        self._T_true = self.rng.uniform(*self.T_range)
        self._we = self.rng.uniform(*self.we_range)

        # baseline (nominal-table) angle AND magnitude -- what the current
        # table-based controller would command, before any RL correction.
        self._id0, self._iq0 = self._baseline_25C(self._Te_target)
        self._beta0_deg = np.degrees(np.arctan2(-self._id0, self._iq0))

        psi_true = psi_of_T(self._T_true)
        window = self._noisy_window(self._id0, self._iq0, self._we, psi_true)

        # TRAINING-ONLY privileged target: the true oracle (id*, iq*) under
        # the true psi_f, used ONLY inside the reward (see r_oracle in
        # step()). This never enters self._obs, so the agent still never
        # sees temperature -- it just gets scored against the right answer,
        # the same way generate_dataset.py's labels used the true psi_f
        # while the network's *input* stayed sensor-only.
        self._id_star, self._iq_star = solve_true_optimum(
            self._Te_target, psi_true, guess=(self._id0, self._iq0))

        self._obs = np.array([self._Te_target, self._we, VDC] + window, dtype=np.float32)
        return self._obs, {}

    def step(self, action):
        delta_beta_deg = float(np.clip(action, -MAX_DELTA_BETA_DEG, MAX_DELTA_BETA_DEG)[0])
        beta_deg = self._beta0_deg + delta_beta_deg
        beta_rad = np.radians(beta_deg)

        psi_true = psi_of_T(self._T_true)
        Is_req, feasible_solve = solve_Is_for_beta(beta_rad, self._Te_target, psi_true)

        if feasible_solve and Is_req <= I_RATED:
            # torque is EXACT by construction -- no root-find, closed form
            Is = Is_req
            id_cmd, iq_cmd = id_iq_from_beta_Is(beta_rad, Is)
            Te_actual = torque(id_cmd, iq_cmd, psi_true)   # == Te_target, up to fp error
        else:
            # infeasible: either no real solution, or it needs more than
            # I_RATED -- a real drive clips current here, torque falls
            # short (the "torque saturation" failure mode from earlier)
            Is = I_RATED
            id_cmd, iq_cmd = id_iq_from_beta_Is(beta_rad, Is)
            Te_actual = torque(id_cmd, iq_cmd, psi_true)
            if not feasible_solve or np.isnan(Is_req):
                Is_req = 2.0 * I_RATED   # sentinel: "far" infeasible, for the penalty below

        loss = copper_loss(id_cmd, iq_cmd)

        # --- reward ---
        torque_err = Te_actual - self._Te_target
        r_torque = -50.0 * (torque_err ** 2)                       # ~0 whenever feasible
        r_current = -20.0 * max(0.0, Is_req - I_RATED) ** 2         # penalizes the RAW need, not just the clip
        r_loss = -0.05 * loss
        # id > 0 is the wrong MTPA region: copper loss depends on id^2, so it's
        # loss-BLIND to the sign of id (a wrong-sign id costs the same loss as
        # the correct-sign one of equal magnitude -- verified in chat). Nothing
        # above tells the agent to prefer id<=0, so add it explicitly. Weight
        # chosen to matter at the low-torque/near-zero-id points where this was
        # observed, without swamping r_torque/r_current at high torque.
        r_id_sign = -10.0 * max(0.0, id_cmd) ** 2

        # r_oracle: direct penalty for landing away from the TRUE optimal
        # (id*, iq*) at this episode's true psi_f (computed once in reset(),
        # privileged/training-only -- never in self._obs). This is the
        # missing piece: r_current is zero everywhere under I_RATED, and
        # r_loss is a very weak, indirect pull toward the minimum-current
        # point (loss is locally FLAT near the MTPA optimum, same flatness
        # that made beta drift cheap in the first place). r_oracle penalizes
        # distance from the actual optimum directly, so it stays informative
        # exactly where r_loss goes flat.
        r_oracle = -20.0 * ((id_cmd - self._id_star) ** 2 + (iq_cmd - self._iq_star) ** 2)

        reward = r_torque + r_current + r_loss + r_id_sign + r_oracle

        terminated = True
        truncated = False

        # single-step episodes: the next observation is never bootstrapped
        # from (terminated=True), so we simply hand back the same state the
        # agent acted on. It carries no temperature leak either way.
        info = {"Te_actual": Te_actual, "Is": Is, "Is_req": Is_req,
                "loss": loss, "T_true": self._T_true, "beta_deg": beta_deg,
                "id_cmd": id_cmd, "iq_cmd": iq_cmd}
        return self._obs, float(reward), terminated, truncated, info


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def main():
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    raw_env = PMSMEfficiencyEnv(seed=0)
    env = Monitor(raw_env)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log="./sac_pmsm_tb",
        seed=0,
        # deployment target is a C2000 MCU, not a workstation -- SB3's default
        # net_arch is [256,256] (~72k actor params, ~282KB fp32), way more
        # than this 23-in/1-out problem needs. [64,64] cuts the deployed
        # actor to ~5.8k params (~22KB fp32, ~5.6KB int8). Shrink further
        # (e.g. [32,32]) once you know the exact flash/RAM budget.
        policy_kwargs=dict(net_arch=[64, 64]),
    )

    # kickoff scale: enough to see whether the reward curve climbs, not a
    # final training run. Watch the reward in tensorboard
    # (tensorboard --logdir ./sac_pmsm_tb) -- if it's flat after this many
    # steps, something in reward/env is wrong before you scale up.
    model.learn(total_timesteps=50_000, log_interval=10)
    model.save("sac_pmsm_kickoff_beta_v3")
    print("saved sac_pmsm_kickoff_beta_v3.zip")

    # quick sanity eval -- agent's (beta, id, iq) alongside the TRUE oracle's
    # (beta, id, iq) for the same 10 scenarios, so the comparison we've been
    # doing by hand in chat is built into the script itself.
    obs, _ = env.reset(seed=123)
    print(f"\n{'Te_t':>6} {'T_true':>7} | {'agent_beta':>10} {'oracle_beta':>11} | "
          f"{'agent_id':>9} {'oracle_id':>9} | {'agent_iq':>9} {'oracle_iq':>9} | "
          f"{'agent_Is':>9} {'oracle_Is':>9} | {'reward':>8}")
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        Te_target, T_true = raw_env._Te_target, raw_env._T_true
        id0_guess, iq0_guess = raw_env._id0, raw_env._iq0

        obs, reward, term, trunc, info = env.step(action)
        id_agent, iq_agent = float(info["id_cmd"]), float(info["iq_cmd"])
        beta_agent = info["beta_deg"]

        psi_true = psi_of_T(T_true)
        id_star, iq_star = solve_true_optimum(Te_target, psi_true, guess=(id0_guess, iq0_guess))
        beta_star = np.degrees(np.arctan2(-id_star, iq_star))
        Is_star = np.hypot(id_star, iq_star)

        print(f"{Te_target:6.2f} {T_true:7.1f} | {beta_agent:10.2f} {beta_star:11.2f} | "
              f"{id_agent:9.4f} {id_star:9.4f} | {iq_agent:9.4f} {iq_star:9.4f} | "
              f"{info['Is']:9.4f} {Is_star:9.4f} | {reward:8.3f}")

        obs, _ = env.reset()


if __name__ == "__main__":
    main()