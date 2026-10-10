import math

import numpy as np

from .synchronous_motor import SynchronousMotor


class SaturatedThermalPMSM(SynchronousMotor):
    """
    IPMSM with magnetic saturation / cross-coupling and temperature-dependent magnet flux and resistance.

    The flux linkages derive from one scalar co-energy potential, so the differential inductance matrix
    is symmetric (:math:`L_{dq} = L_{qd}`) by construction:

    .. math::
        \\psi_d &= \\lambda_{pm}(T_m) + L_{d0} i_d - \\tfrac{a_d}{2} i_d^2 - \\tfrac{k_d}{3} i_d^3
                  - \\tfrac{k_c}{2} i_d i_q^2 - \\tfrac{a_{dq}}{2} i_q^2 \\\\
        \\psi_q &= L_{q0} i_q - \\tfrac{b_q}{2} i_q^2 - \\tfrac{k_q}{3} i_q^3 - \\tfrac{k_c}{2} i_d^2 i_q
                  - a_{dq} i_d i_q

    and the dynamics are :math:`\\mathbf{L}(i)\\, di/dt = u - R_s(T_w) i - \\omega_e J \\psi(i)`.

    **Iron loss.** A conductance :math:`g(\\omega_e)` sits in parallel with the speed voltage
    :math:`e = \\omega_e J \\psi`, so the terminal current splits into a core-loss part and a magnetising part:

    .. math::
        i = i_o + i_c, \\qquad i_c = g(\\omega_e)\\, e, \\qquad \\psi = \\psi(i_o)

    The ODE state ``i_sd, i_sq`` is the MAGNETISING current :math:`i_o` (saturation and torque act on it); the
    TERMINAL current :math:`i` is what the converter and the sensors (controllers, observations) see, see
    :py:meth:`terminal_currents`. :math:`R_s` carries :math:`i`, so the core loss shows up in the input power.
    ``g`` is split into an eddy part (:math:`\\propto \\omega_e^2` loss) and a hysteresis part
    (:math:`\\propto \\omega_e` loss), fixed by the iron loss :math:`P_0` at the rated electrical speed
    :math:`\\omega_0` and flux :math:`\\psi_0` (nominal magnet flux), with hysteresis share ``a``:

    .. math::
        g = \\frac{1-a}{r_{iron}} + \\frac{a\\,\\omega_0}{r_{iron}\\,\\max(|\\omega_e|, \\omega_{min})},
        \\quad P_0 = \\frac{1.5\\,\\omega_0^2 \\psi_0^2}{r_{iron}}, \\quad \\omega_{min} = 0.05\\,\\omega_0

    ``motor_parameter`` keeps the NOMINAL (25 degC, unsaturated) ``r_s``, ``l_d``, ``l_q`` and ``psi_p``.
    Controllers built by ``gem_controllers`` read exactly these values, so they stay blind to saturation and
    drift while the plant does not. The true values depend on the plant state set through
    :py:meth:`set_temperatures`.

    =====================  ==========  ============================================================
    Motor Parameter        Unit        Description
    =====================  ==========  ============================================================
    r_s, l_d, l_q, psi_p   see PMSM    Nominal values at ``t_ref``
    a_d, k_d, b_q, k_q     H/A, H/A^2  Self-saturation coefficients (d and q axis)
    k_c, a_dq              H/A^2, H/A  Cross-coupling coefficients
    alpha_cu               1/degC      Copper resistance temperature coefficient
    beta_pm                1/degC      Magnet flux temperature coefficient
    t_ref                  degC        Reference temperature of the nominal values
    r_iron                 Ohm         Iron-loss resistance at rated speed and nominal magnet flux
    hysteresis_fraction    1           Share ``a`` of the rated-speed iron loss that is hysteresis (0 = pure eddy)
    iron_loss_dynamic      bool        True: core-loss current acts on the plant; False: loss reporting only
    =====================  ==========  ============================================================
    """

    #### Defaults: TI SPRACF3 ferrite IPMSM (motor_params.yaml of MotorCtrl-AI, the single source of truth)
    _default_motor_parameter = {
        "p": 3,
        "l_d": 1.532e-3,
        "l_q": 7.324e-3,
        "j_rotor": 0.0026,
        "r_s": 0.130185,
        "psi_p": 0.2084,
        "a_d": 0.0,
        "k_d": 3.0e-6,
        "b_q": 0.0,
        "k_q": 1.8e-5,
        "k_c": 4.0e-6,
        "a_dq": 0.0,
        "alpha_cu": 0.00393,
        "beta_pm": -0.0020,
        "t_ref": 25.0,
        "r_iron": 450.0,
        "hysteresis_fraction": 0.0,
        "iron_loss_dynamic": True,
    }
    HAS_JACOBIAN = False
    _default_limits = dict(omega=3e3 * np.pi / 30, torque=0.0, i=12.0, epsilon=math.pi, u=230.0)
    _default_nominal_values = dict(omega=2.4e3 * np.pi / 30, torque=0.0, i=10.0, epsilon=math.pi, u=230.0)
    _default_initializer = {
        "states": {"i_sq": 0.0, "i_sd": 0.0, "epsilon": 0.0},
        "interval": None,
        "random_init": None,
        "random_params": (None, None),
    }

    IO_VOLTAGES = ["u_a", "u_b", "u_c", "u_sd", "u_sq"]
    IO_CURRENTS = ["i_a", "i_b", "i_c", "i_sd", "i_sq"]

    def __init__(self, *args, temperature_winding=None, temperature_magnet=None, **kwargs):
        # None means "at t_ref"; set before super().__init__ because it evaluates the torque limit
        self.temperature_winding = temperature_winding
        self.temperature_magnet = temperature_magnet
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ thermal state
    def set_temperatures(self, temperature_winding=None, temperature_magnet=None):
        """Set the TRUE plant temperatures [degC]; ``None`` leaves a temperature unchanged."""
        if temperature_winding is not None:
            self.temperature_winding = float(temperature_winding)
        if temperature_magnet is not None:
            self.temperature_magnet = float(temperature_magnet)

    def r_s_true(self, temperature_winding=None):
        mp = self._motor_parameter
        t = temperature_winding if temperature_winding is not None else self.temperature_winding
        t = mp["t_ref"] if t is None else t
        return mp["r_s"] * (1.0 + mp["alpha_cu"] * (t - mp["t_ref"]))

    def psi_pm_true(self, temperature_magnet=None):
        mp = self._motor_parameter
        t = temperature_magnet if temperature_magnet is not None else self.temperature_magnet
        t = mp["t_ref"] if t is None else t
        return mp["psi_p"] * (1.0 + mp["beta_pm"] * (t - mp["t_ref"]))

    # ------------------------------------------------------------------ magnetics (vectorised)
    def flux_linkages(self, i_d, i_q, temperature_magnet=None):
        """(psi_d, psi_q) of the true plant."""
        mp = self._motor_parameter
        lam = self.psi_pm_true(temperature_magnet)
        psi_d = (
            lam
            + mp["l_d"] * i_d
            - 0.5 * mp["a_d"] * i_d**2
            - mp["k_d"] / 3.0 * i_d**3
            - 0.5 * mp["k_c"] * i_d * i_q**2
            - 0.5 * mp["a_dq"] * i_q**2
        )
        psi_q = (
            mp["l_q"] * i_q
            - 0.5 * mp["b_q"] * i_q**2
            - mp["k_q"] / 3.0 * i_q**3
            - 0.5 * mp["k_c"] * i_d**2 * i_q
            - mp["a_dq"] * i_d * i_q
        )
        return psi_d, psi_q

    def differential_inductances(self, i_d, i_q):
        """(L_dd, L_qq, L_dq) with L_dq = L_qd."""
        mp = self._motor_parameter
        l_dd = mp["l_d"] - mp["a_d"] * i_d - mp["k_d"] * i_d**2 - 0.5 * mp["k_c"] * i_q**2
        l_qq = mp["l_q"] - mp["b_q"] * i_q - mp["k_q"] * i_q**2 - 0.5 * mp["k_c"] * i_d**2 - mp["a_dq"] * i_d
        l_dq = -i_q * (mp["k_c"] * i_d + mp["a_dq"])
        return l_dd, l_qq, l_dq

    def torque_dq(self, i_d, i_q, temperature_magnet=None):
        """Electromagnetic torque 1.5 p (psi_d i_oq - psi_q i_od) [Nm] of the MAGNETISING current i_o; vectorised."""
        psi_d, psi_q = self.flux_linkages(i_d, i_q, temperature_magnet)
        return 1.5 * self._motor_parameter["p"] * (psi_d * i_q - psi_q * i_d)

    # ------------------------------------------------------------------ iron loss (core-loss branch)
    OMEGA_MIN_FRACTION = 0.05  # omega_min / omega_0: below this speed the hysteresis conductance stops growing

    def iron_conductance(self, omega_el):
        """g(omega_e) [S] of the core-loss branch; eddy part constant, hysteresis part ~ 1/|omega_e|."""
        mp = self._motor_parameter
        omega_0 = self.nominal_values["omega"] * mp["p"]
        a = mp["hysteresis_fraction"]
        omega_floor = np.maximum(np.abs(omega_el), self.OMEGA_MIN_FRACTION * omega_0)
        return ((1.0 - a) + a * omega_0 / omega_floor) / mp["r_iron"]

    def core_currents(self, psi_d, psi_q, omega_el):
        """(i_c,d, i_c,q) = g e with e = omega_e (-psi_q, psi_d); zero if the loss is not dynamic."""
        if not self._motor_parameter["iron_loss_dynamic"]:
            return np.zeros_like(psi_d * omega_el), np.zeros_like(psi_q * omega_el)
        g = self.iron_conductance(omega_el)
        return -g * omega_el * psi_q, g * omega_el * psi_d

    def terminal_from_magnetizing(self, i_od, i_oq, omega_el, temperature_magnet=None):
        """Terminal current i = i_o + i_c (explicit); vectorised."""
        psi_d, psi_q = self.flux_linkages(i_od, i_oq, temperature_magnet)
        i_cd, i_cq = self.core_currents(psi_d, psi_q, omega_el)
        return i_od + i_cd, i_oq + i_cq

    def magnetizing_from_terminal(self, i_d, i_q, omega_el, temperature_magnet=None, iterations=12):
        """
        Magnetising current i_o for a given terminal current: fixed point of i_o = i - i_c(psi(i_o)). The map is a
        contraction (factor ~ g omega_e L ~ 1e-2), so a dozen iterations reach machine precision. Vectorised.
        """
        i_d, i_q = np.asarray(i_d, dtype=float), np.asarray(i_q, dtype=float)
        i_od, i_oq = i_d, i_q
        if not self._motor_parameter["iron_loss_dynamic"]:
            return i_od, i_oq
        for _ in range(iterations):
            psi_d, psi_q = self.flux_linkages(i_od, i_oq, temperature_magnet)
            i_cd, i_cq = self.core_currents(psi_d, psi_q, omega_el)
            i_od, i_oq = i_d - i_cd, i_q - i_cq
        return i_od, i_oq

    def torque_from_terminal(self, i_d, i_q, omega_el, temperature_magnet=None):
        """Electromagnetic torque [Nm] for a TERMINAL current at electrical speed ``omega_el``."""
        i_od, i_oq = self.magnetizing_from_terminal(i_d, i_q, omega_el, temperature_magnet)
        return self.torque_dq(i_od, i_oq, temperature_magnet)

    def steady_state_voltages(self, i_d, i_q, omega_el, temperature_winding=None, temperature_magnet=None):
        """(u_d, u_q) at steady state for the TERMINAL current ``i`` and electrical speed ``omega_el`` [rad/s]."""
        r_s = self.r_s_true(temperature_winding)
        i_od, i_oq = self.magnetizing_from_terminal(i_d, i_q, omega_el, temperature_magnet)
        psi_d, psi_q = self.flux_linkages(i_od, i_oq, temperature_magnet)
        return r_s * i_d - omega_el * psi_q, r_s * i_q + omega_el * psi_d

    def losses(self, i_d, i_q, omega_el, temperature_winding=None, temperature_magnet=None):
        """
        (P_copper, P_iron, P_total) [W] for the TERMINAL current ``i`` (what sensors and observations report).
        P_cu = 1.5 R_s |i|^2 with the terminal current; P_iron = 1.5 g omega_e^2 |psi(i_o)|^2 with the plant flux.
        """
        i_od, i_oq = self.magnetizing_from_terminal(i_d, i_q, omega_el, temperature_magnet)
        return self.losses_from_magnetizing(i_od, i_oq, omega_el, temperature_winding, temperature_magnet)

    def losses_from_magnetizing(self, i_od, i_oq, omega_el, temperature_winding=None, temperature_magnet=None):
        """Same as :py:meth:`losses` but for the magnetising current ``i_o`` (no fixed-point solve)."""
        i_d, i_q = self.terminal_from_magnetizing(i_od, i_oq, omega_el, temperature_magnet)
        p_cu = 1.5 * self.r_s_true(temperature_winding) * (i_d**2 + i_q**2)
        psi_d, psi_q = self.flux_linkages(i_od, i_oq, temperature_magnet)
        p_iron = 1.5 * self.iron_conductance(omega_el) * omega_el**2 * (psi_d**2 + psi_q**2)
        return p_cu, p_iron, p_cu + p_iron

    # ------------------------------------------------------------------ GEM interface
    def _update_model(self):
        # Docstring of superclass: everything is evaluated on the fly (state-dependent inductances)
        pass

    def torque(self, currents):
        # Docstring of superclass; the ODE state is the magnetising current
        return self.torque_dq(currents[self.I_SD_IDX], currents[self.I_SQ_IDX])

    def terminal_currents(self, currents, omega):
        # Docstring of superclass; i = i_o + g e
        omega_el = omega * self._motor_parameter["p"]
        i_d, i_q = self.terminal_from_magnetizing(currents[self.I_SD_IDX], currents[self.I_SQ_IDX], omega_el)
        out = np.array(currents, dtype=float)
        out[self.I_SD_IDX], out[self.I_SQ_IDX] = i_d, i_q
        return out

    def _torque_limit(self):
        # Docstring of superclass: largest torque on the nominal-current circle at the current plant state
        i_n = self.nominal_values["i"]
        beta = np.linspace(0.0, math.pi / 2, 181)
        return float(np.max(self.torque_dq(-i_n * np.sin(beta), i_n * np.cos(beta))))

    def electrical_ode(self, state, u_dq, omega, *_):
        # Docstring of superclass
        mp = self._motor_parameter
        i_d, i_q = state[self.I_SD_IDX], state[self.I_SQ_IDX]
        omega_el = omega * mp["p"]
        psi_d, psi_q = self.flux_linkages(i_d, i_q)
        l_dd, l_qq, l_dq = self.differential_inductances(i_d, i_q)
        r_s = self.r_s_true()
        i_cd, i_cq = self.core_currents(psi_d, psi_q, omega_el)  # R_s carries the terminal current i_o + i_c
        rhs_d = u_dq[0] - r_s * (i_d + i_cd) + omega_el * psi_q
        rhs_q = u_dq[1] - r_s * (i_q + i_cq) - omega_el * psi_d
        det = l_dd * l_qq - l_dq**2
        return np.array(
            [
                (l_qq * rhs_d - l_dq * rhs_q) / det,
                (l_dd * rhs_q - l_dq * rhs_d) / det,
                omega_el,
            ]
        )
