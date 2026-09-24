"""
Real-GEM closed-loop training environment for the beta-correction RL agent.

This REPLACES the analytic PMSMEfficiencyEnv from rl_kickoff_sac.py with the
real gym-electric-motor (GEM) plant + your existing tuned speed/current PI
cascade (classic_controllers.Controller), following the architecture we
agreed on:

  - Keep your FULL working cascade exactly as in
    closed_loop_foc_temperature_episodes.py: ControlType.SpeedControl,
    the real mechanical load, and the real speed PI producing a live torque
    reference every control cycle. RL does NOT touch this -- omega evolves
    from real rotor dynamics, not a hand-picked constant.
  - RL replaces ONLY the torque-to-current (MTPA) block. Your
    CascadedFieldOrientedController calls
        self.ref[0], self.ref[1] = self.torque_controller.control(state, torque)
    once per control cycle (cascaded_foc_controller.py line ~208).
    `self.torque_controller` is a public, swappable attribute -- your own
    adaptive-MTPA script already replaces it at runtime
    (`controller.torque_controller = TorqueToCurrentConversion(...)`), so
    dropping in an RL-driven replacement with the same `.control(state,
    torque) -> (iq_ref, id_ref)` interface is using the library the way it's
    designed to be extended, not a hack.
  - RL acts once every 10 tau cycles (tau=100us -> 1ms), not every cycle --
    the beta correction is held fixed for those 10 cycles while the real
    current PI + plant continue running every cycle underneath it.

WHAT'S STILL A PLACEHOLDER -- FIX BEFORE TRUSTING RESULTS:
  - ROTOR_J: no rotor inertia value has been given for the TI SPRACF3 motor
    anywhere in this project (only electrical parameters have been agreed).
    5e-5 kg*m^2 below is a rough small-appliance-motor guess, not a real
    number. Replace it, and re-tune LOAD_VISCOUS_FRICTION / LOAD_TORQUE_*
    to whatever your actual HVAC/compressor load looks like.
  - Current/speed PI gains are computed with the SAME symmetric-optimum
    formulas your existing scripts use (fc_curr=200Hz, fc_spd=10Hz,
    phi=80deg), just re-evaluated at the TI SPRACF3 Ld/Lq/Rs numbers. These
    are a reasonable starting tune, not a validated one -- watch the speed/
    current response before trusting the closed loop is well-damped.
  - THIS FILE HAS NOT BEEN EXECUTED. gym-electric-motor and your
    classic_controllers package aren't available in the sandbox this was
    written in, so this has only been checked for Python syntax
    (py_compile), not run. Run smoke_test_gem_env.py FIRST, on your machine,
    before wiring this into SAC training, and send back whatever it prints
    (including any traceback) the same way we've been debugging everything
    else in this project.

Everything else -- the beta action space, solve_Is_for_beta, the reward
shape (r_torque, r_current, r_loss, r_id_sign, r_oracle), the oracle solve,
and the [64,64] MCU-sized policy network -- carries over unchanged from
rl_kickoff_sac.py.
"""
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------
# Make your classic_controllers package importable, same as your own scripts.
# NOTE: this file lives in .../examples/userdefinedExamples/MTPA/, one level
# DEEPER than closed_loop_foc_temperature_episodes.py etc. (which live
# directly in .../examples/userdefinedExamples/), so it needs one more ".."
# than those scripts use to reach .../examples/classic_controllers/.
# ---------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CLASSIC_CONTROLLERS_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, "..", "..", "classic_controllers"))
if _CLASSIC_CONTROLLERS_DIR not in sys.path:
    sys.path.append(_CLASSIC_CONTROLLERS_DIR)

import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.envs.motors import ActionType, ControlType, Motor, MotorType
from gym_electric_motor.physical_systems.mechanical_loads import MechanicalLoad
from classic_controllers import Controller

from rl_kickoff_sacV2 import (
    P, LD, LQ, RS, PSI_NOM, T_REF, I_RATED, VDC, ALPHA_PSI_FERRITE, S,
    MAX_DELTA_BETA_DEG, N_WINDOW, CURRENT_NOISE_STD,
    psi_of_T, torque as torque_eq, copper_loss, id_iq_from_beta_Is,
    solve_Is_for_beta, solve_true_optimum,
)

# =========================================================================
# TI SPRACF3 motor parameters, mapped into GEM's motor_parameter dict
# (same Ld/Lq/Rs/psi_f/I_rated/Vdc used everywhere else in this project)
# =========================================================================
ROTOR_J = 5e-5   # kg*m^2 -- PLACEHOLDER, see module docstring
WE_RANGE = (100.0, 550.0)   # electrical speed range, rad/s (same as rl_kickoff_sac.py)

RL_DECISION_EVERY_N_TAU = 10   # RL acts once per 10 tau cycles (100us*10=1ms)
TAU = 1e-4                      # 100 microseconds, same as your scripts
EPISODE_DURATION_S = 10        # simulated seconds per RL "episode"
N_TAU_STEPS_PER_EPISODE = int(EPISODE_DURATION_S / TAU)
N_RL_DECISIONS_PER_EPISODE = N_TAU_STEPS_PER_EPISODE // RL_DECISION_EVERY_N_TAU


class ConfigurableLoad(MechanicalLoad):
    """Minimal constant-torque + viscous-friction load -- same shape as
    your ConfigurableStepLoad, without the step (we don't need a
    disturbance step for RL training, just a plausible steady load)."""

    def __init__(self, j_load=0.0, t_load=0.3, viscous_friction=1e-4, **kwargs):
        super().__init__(j_load=j_load, **kwargs)
        self.t_load = t_load
        self.viscous_friction = viscous_friction

    def mechanical_ode(self, t, mechanical_state, torque):
        omega = mechanical_state[self.OMEGA_IDX]
        total_load = self.t_load + self.viscous_friction * omega
        d_omega = (torque - total_load) / self._j_total
        return np.array([d_omega])


def _pi_gains_symmetric_optimum(fc_hz, phi_deg, L, I_base, v_base):
    """Same symmetric-optimum PI tuning formula your scripts already use,
    factored out so it can be re-evaluated at the TI SPRACF3 Ld/Lq."""
    phi = phi_deg * np.pi / 180
    wc = 2 * np.pi * fc_hz
    kp_phy = L * np.sin(phi) * wc
    ki_phy = (wc * (1 / np.tan(phi))) * kp_phy
    kp_pu = kp_phy * I_base / v_base
    ki_pu = ki_phy * I_base / v_base
    return kp_pu, ki_pu


class RLBetaTorqueToCurrent:
    """Drop-in replacement for classic_controllers'
    TorqueToCurrentConversion, matching its `.control(state, torque) ->
    (iq_ref, id_ref)` interface (both normalized/per-unit, same as the
    original -- see cascaded_foc_controller.py line ~208).

    RL only updates its delta_beta every RL_DECISION_EVERY_N_TAU calls;
    the other calls reuse the last decision, so the actual current PI +
    plant still run every tau cycle underneath a beta correction that's
    only 1ms-granular -- matches the agreed design.
    """

    def __init__(self, env, predict_fn):
        ps = env.unwrapped.physical_system
        self.state_names = ps.state_names
        self.limits = ps.limits
        self.i_sd_idx = self.state_names.index("i_sd")
        self.i_sq_idx = self.state_names.index("i_sq")
        self.u_sd_idx = self.state_names.index("u_sd")
        self.u_sq_idx = self.state_names.index("u_sq")
        self.omega_idx = self.state_names.index("omega")

        self.predict_fn = predict_fn
        self._call_count = 0
        self._delta_beta_deg = 0.0

        # rolling window of the last N_WINDOW (vd, vq, id, iq) measurements
        # -- REAL plant readings, not synthetic noise, so this is a more
        # honest sensor-only state than the analytic env's synthetic one.
        self._window = []

        # exposed so the outer gym.Env can read the current torque target
        # and the last commanded (id_cmd, iq_cmd) for the reward.
        self.last_torque_target = 0.0
        self.last_id_cmd = 0.0
        self.last_iq_cmd = 0.0
        self.last_beta_deg = 0.0

    def _baseline_beta0_deg(self, Te_target):
        """25C-nominal MTPA baseline angle for Te_target -- same closed-form
        as PMSMEfficiencyEnv._baseline_25C in rl_kickoff_sac.py."""
        from scipy.optimize import brentq

        def mtpa_id_nom(iq):
            A = PSI_NOM / (2.0 * S)
            return -A - np.sqrt(A ** 2 + iq ** 2)

        def Te_of_iq(iq):
            return torque_eq(mtpa_id_nom(iq), iq, PSI_NOM)

        iq = brentq(lambda x: Te_of_iq(x) - Te_target, 1e-6, I_RATED, xtol=1e-8)
        id_ = mtpa_id_nom(iq)
        return np.degrees(np.arctan2(-id_, iq))

    def control(self, state, torque):
        """torque: physical (un-normalized) torque reference [Nm], from the
        speed loop. Returns (iq_ref, id_ref), BOTH per-unit (normalized),
        matching what CascadedFieldOrientedController expects back."""
        vd = state[self.u_sd_idx] * self.limits[self.u_sd_idx]
        vq = state[self.u_sq_idx] * self.limits[self.u_sq_idx]
        id_meas = state[self.i_sd_idx] * self.limits[self.i_sd_idx]
        iq_meas = state[self.i_sq_idx] * self.limits[self.i_sq_idx]
        self._window.append((vd, vq, id_meas, iq_meas))
        if len(self._window) > N_WINDOW:
            self._window.pop(0)

        self.last_torque_target = torque

        if self._call_count % RL_DECISION_EVERY_N_TAU == 0:
            beta0_deg = self._baseline_beta0_deg(torque)
            window_flat = []
            w = self._window if len(self._window) == N_WINDOW else \
                self._window + [self._window[-1]] * (N_WINDOW - len(self._window))
            for vd_n, vq_n, id_n, iq_n in w:
                window_flat.extend([vd_n, vq_n, id_n, iq_n])
            we = state[self.omega_idx] * self.limits[self.omega_idx] * P
            obs = np.array([torque, we, VDC] + window_flat, dtype=np.float32)

            action = self.predict_fn(obs)
            self._delta_beta_deg = float(
                np.clip(action, -MAX_DELTA_BETA_DEG, MAX_DELTA_BETA_DEG)[0])
            self.last_beta_deg = beta0_deg + self._delta_beta_deg

        self._call_count += 1

        # Use the NOMINAL psi for the Is-solve -- this is the sensor-only
        # controller, it does NOT know the true (drifted) psi_f, same as
        # the analytic env's design. The TRUE plant (with true psi_f) is
        # what actually responds to this current command.
        Is_req, feasible = solve_Is_for_beta(
            np.radians(self.last_beta_deg), torque, PSI_NOM)
        if not feasible or not np.isfinite(Is_req) or Is_req > I_RATED:
            Is_req = min(abs(Is_req) if np.isfinite(Is_req) else I_RATED, I_RATED)
        id_cmd, iq_cmd = id_iq_from_beta_Is(np.radians(self.last_beta_deg), Is_req)
        self.last_id_cmd, self.last_iq_cmd = id_cmd, iq_cmd

        iq_ref_pu = iq_cmd / self.limits[self.i_sq_idx]
        id_ref_pu = id_cmd / self.limits[self.i_sd_idx]
        return iq_ref_pu, id_ref_pu

    def reset(self):
        self._call_count = 0
        self._window = []


class PMSMGemClosedLoopEnv(gym.Env):
    """gym.Env wrapping the real GEM closed loop, matching the SB3-facing
    interface of PMSMEfficiencyEnv in rl_kickoff_sac.py (same action space,
    same reward terms) but driven by real plant dynamics instead of a
    closed-form single-step model. See module docstring for the
    architecture and the PLACEHOLDER values that still need real numbers.
    """

    metadata = {"render_modes": []}

    def __init__(self, T_range=(25.0, 120.0), omega_ref_range=(0.15, 0.35), seed=None):
        super().__init__()
        # omega_ref_range was (0.3, 0.9) -- the frozen-RL diagnostic showed
        # the modulation index (voltage utilization) already pinned above
        # a_max at 0.865 pu (mean 1.230 vs. a_max=1.155), i.e. the real
        # corner speed for this motor/VDC is well below what wm_base's
        # f_base=200Hz assumption implied. RLBetaTorqueToCurrent has no
        # flux-weakening logic, so for now we keep the training envelope
        # inside the MTPA/constant-torque region instead of adding it.
        # Re-check with diagnose_gem_closed_loop.py before widening this.
        self.T_range = T_range
        self.omega_ref_range = omega_ref_range
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(
            low=np.array([-MAX_DELTA_BETA_DEG], dtype=np.float32),
            high=np.array([MAX_DELTA_BETA_DEG], dtype=np.float32),
        )
        v_bound = VDC
        i_bound = I_RATED + 1.0
        # we's lower bound must allow the real cold-start transient (rotor
        # at rest -> we=0, ramping up), not WE_RANGE[0] -- that bound only
        # made sense in the old analytic env, where we was an independent
        # random sample that never started at 0. A little negative headroom
        # covers any small reverse-direction wobble during settling too.
        low = [0.0, -10.0, VDC] + [-v_bound, -v_bound, -i_bound, -i_bound] * N_WINDOW
        high = [12.0, WE_RANGE[1], VDC] + [v_bound, v_bound, i_bound, i_bound] * N_WINDOW
        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32), high=np.array(high, dtype=np.float32))

        self._build_env_and_controller()

        self._pending_action = None   # set by SB3 via step(), read by predict_fn
        self._decision_count = 0
        self._T_true = None
        self._state = None
        self._reference = None

    def _predict_fn(self, obs):
        # called from inside RLBetaTorqueToCurrent.control() -- returns
        # whatever action was set by the most recent gym step() call.
        return self._pending_action

    def _build_env_and_controller(self):
        p = P
        Rs_20 = RS
        Ld, Lq = LD, LQ
        psi_p_20 = PSI_NOM
        j = ROTOR_J
        u_dc_link = VDC

        f_rated = 120
        w_rated = 2 * np.pi * f_rated
        wm_rated = w_rated / p
        v_rated = u_dc_link / np.sqrt(3)
        T_rated = 1.5 * p * psi_p_20 * I_RATED

        f_base = 200
        w_base = 2 * np.pi * f_base
        wm_base = w_base / p
        v_base = v_rated
        # Same zero-headroom issue as current: T_rated assumes id=0, but
        # MTPA's negative id (reluctance torque boost) can genuinely produce
        # more torque than that at the same current -- give the limit some
        # margin so a normal MTPA operating point doesn't trip termination.
        T_base = 1.2 * T_rated

        # speed loop gain formula uses j (inertia), not an inductance, so it
        # doesn't fit _pi_gains_symmetric_optimum's signature -- computed
        # directly here, same formula as your scripts.
        phi_spd = 80 * np.pi / 180
        wc_spd = 2 * np.pi * 10
        kp_spd_phy = j * np.sin(phi_spd) * wc_spd
        ki_spd_phy = (wc_spd * (1 / np.tan(phi_spd))) * kp_spd_phy
        kp_spd_pu = kp_spd_phy * wm_base / T_base
        ki_spd_pu = ki_spd_phy * wm_base / T_base

        kpd_pu, kid_pu = _pi_gains_symmetric_optimum(200, 80, Ld, I_RATED, v_base)
        kpq_pu, kiq_pu = _pi_gains_symmetric_optimum(200, 80, Lq, I_RATED, v_base)

        current_d_controller = {"controller_type": "pi_controller", "p_gain": kpd_pu, "i_gain": kid_pu}
        current_q_controller = {"controller_type": "pi_controller", "p_gain": kpq_pu, "i_gain": kiq_pu}
        speed_controller = {"controller_type": "pi_controller", "p_gain": kp_spd_pu, "i_gain": ki_spd_pu}

        # Current's nominal and limit must NOT be the same value: nominal is
        # the continuous rating RLBetaTorqueToCurrent commands up to
        # (I_RATED); limit is GEM's hard ceiling -- cross it and GEM
        # terminates the episode on the spot. With both set to I_RATED there
        # was zero headroom, so any normal current-loop overshoot tripped
        # termination every ~1-2ms: env.reset() zeroed omega/currents, they
        # ramped back up, tripped again -- the sawtooth pattern that showed
        # up in the frozen-RL diagnostic (omega never getting past ~2000
        # RPM before snapping back to 0). 20% headroom matches the margin
        # the real TorqueToCurrentConversion relies on (it clips its own
        # output to nominal_values, not limit_values, for the same reason).
        I_LIMIT = 1.2 * I_RATED

        motor_parameter = {"p": p, "r_s": Rs_20, "l_d": Ld, "l_q": Lq, "psi_p": psi_p_20, "j_rotor": j}
        nominal_values = {"omega": wm_rated, "torque": T_rated, "i": I_RATED, "u": v_rated, "epsilon": 2 * np.pi}
        limit_values = {"omega": wm_base, "torque": T_base, "i": I_LIMIT, "u": v_base, "epsilon": 2 * np.pi}

        # omega reference is re-set per episode in reset() via _reference_value
        self._omega_ref_gen = rg.ConstReferenceGenerator(reference_state="omega", reference_value=0.5)

        self._load = ConfigurableLoad(j_load=0.0, t_load=0.3, viscous_friction=1e-4)

        motor = Motor(MotorType.PermanentMagnetSynchronousMotor, ControlType.SpeedControl, ActionType.Continuous)
        self.env = gem.make(
            motor.env_id(),
            supply=dict(u_nominal=u_dc_link),
            load=self._load,
            reference_generator=self._omega_ref_gen,
            motor=dict(motor_parameter=motor_parameter, nominal_values=nominal_values, limit_values=limit_values),
            tau=TAU,
        )

        self.controller = Controller.make(
            self.env,
            stages=[[current_d_controller, current_q_controller], [speed_controller]],
            torque_control="analytical",   # built with the analytical block first, then swapped below
            decoupling=False,
            plot_torque=False,
        )
        # Swap in the RL-driven torque-to-current block -- the same
        # runtime-swap pattern your adaptive-MTPA script already uses.
        self.torque_block = RLBetaTorqueToCurrent(self.env, self._predict_fn)
        self.controller.torque_controller = self.torque_block

        ps = self.env.unwrapped.physical_system
        self.motor_instance = ps.electrical_motor
        self.state_names = ps.state_names
        self.limits = ps.limits
        self.omega_idx = self.state_names.index("omega")
        self.torque_idx = self.state_names.index("torque")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._T_true = self.rng.uniform(*self.T_range)
        psi_true = psi_of_T(self._T_true)
        self.motor_instance.motor_parameter["psi_p"] = psi_true
        # NOTE: Rs winding-temperature drift is still not modeled here,
        # same simplification as rl_kickoff_sac.py -- Rs stays fixed.

        omega_ref = self.rng.uniform(*self.omega_ref_range)
        self._omega_ref_gen._reference_value = omega_ref

        (state, reference), _ = self.env.reset()
        self.controller.reset()
        self.torque_block.reset()
        self._state, self._reference = state, reference
        self._decision_count = 0

        # run one RL-decision worth of cycles with a neutral (zero) action
        # just to populate the sensor window before the first real action
        self._pending_action = np.array([0.0], dtype=np.float32)
        obs = self._run_cycles(RL_DECISION_EVERY_N_TAU)
        return obs, {}

    def _run_cycles(self, n_cycles):
        obs = None
        for _ in range(n_cycles):
            action = self.controller.control(self._state, self._reference)
            (self._state, self._reference), _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                (self._state, self._reference), _ = self.env.reset()
                self.controller.reset()
        # rebuild the observation the same way RLBetaTorqueToCurrent does,
        # from whatever window it has accumulated
        we = self._state[self.omega_idx] * self.limits[self.omega_idx] * P
        window_flat = []
        w = self.torque_block._window
        if len(w) < N_WINDOW:
            w = w + [w[-1]] * (N_WINDOW - len(w)) if w else [(0.0, 0.0, 0.0, 0.0)] * N_WINDOW
        for vd_n, vq_n, id_n, iq_n in w[-N_WINDOW:]:
            window_flat.extend([vd_n, vq_n, id_n, iq_n])
        Te_target = self.torque_block.last_torque_target
        obs = np.array([Te_target, we, VDC] + window_flat, dtype=np.float32)
        return obs

    def step(self, action):
        self._pending_action = action
        obs = self._run_cycles(RL_DECISION_EVERY_N_TAU)

        Te_target = self.torque_block.last_torque_target
        Te_actual = self._state[self.torque_idx] * self.limits[self.torque_idx]
        id_cmd = self.torque_block.last_id_cmd
        iq_cmd = self.torque_block.last_iq_cmd
        Is_cmd = float(np.hypot(id_cmd, iq_cmd))
        loss = copper_loss(id_cmd, iq_cmd)

        psi_true = psi_of_T(self._T_true)
        id_star, iq_star = solve_true_optimum(Te_target, psi_true)

        torque_err = Te_actual - Te_target
        r_torque = -50.0 * (torque_err ** 2)
        r_current = -20.0 * max(0.0, Is_cmd - I_RATED) ** 2
        r_loss = -0.05 * loss
        r_id_sign = -10.0 * max(0.0, id_cmd) ** 2
        r_oracle = -20.0 * ((id_cmd - id_star) ** 2 + (iq_cmd - iq_star) ** 2)
        reward = r_torque + r_current + r_loss + r_id_sign + r_oracle

        self._decision_count += 1
        terminated = False
        truncated = self._decision_count >= N_RL_DECISIONS_PER_EPISODE

        info = {"Te_actual": Te_actual, "Te_target": Te_target, "Is": Is_cmd,
                "loss": loss, "T_true": self._T_true, "beta_deg": self.torque_block.last_beta_deg,
                "id_cmd": id_cmd, "iq_cmd": iq_cmd}
        return obs, float(reward), terminated, truncated, info
