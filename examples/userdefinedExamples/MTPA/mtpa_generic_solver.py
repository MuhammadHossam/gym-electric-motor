"""
Generic MTPA solver: solves the 2-equation / 2-unknown system

    (1) Torque eq:  Te_target = 1.5*p*(psi_f*iq + (Ld-Lq)*id*iq)
    (2) MTPA eq:    id*psi_f + s*id^2 - s*iq^2 = 0,   s = Ld - Lq

simultaneously for (id, iq) -- exactly the two equations discussed: the
torque command from the speed PI is one equation, the MTPA condition
(loss-minimizing point on that torque curve) is the other. This solves
them together with a numeric nonlinear solver (fsolve), NOT the closed
-form shortcut -- so it works unchanged for ANY (p, Ld, Lq, psi_f), which
is what "generic" means here. The closed-form id*(iq) from pmsm_model.py
is used only as a cross-check, not as part of the solve.

Then sweeps Te_target from 0 up to the current-limited maximum, at each
of several magnet temperatures (ferrite magnet, matching the motor we
agreed on), and plots:
  - the MTPA curve (id, iq) at each temperature
  - the stator-current-limit circle Is = I_rated, over the FULL id range
    (-I_rated to +I_rated), not just the negative-id quadrant
  - a constant-torque trajectory (iso-torque hyperbola): the locus of
    every (id, iq) that delivers the SAME torque. By construction it is
    tangent to the nominal-temperature MTPA curve exactly at the point
    where MTPA meets the current limit -- that tangency is the textbook
    definition of MTPA (minimum current on a given torque curve), so
    plotting it alongside MTPA is a direct visual proof, not just a
    decoration.

Motor: TI SPRACF3 reference parameters.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ---- TI SPRACF3 motor parameters (the motor we agreed on) ----
p = 2# 3
Ld = 18.5e-3 #1.532e-3       # H
Lq = 38e-3 #7.324e-3        # H
Rs = 1.15#0.130185        # ohm (not needed for the MTPA angle itself)
psi_nom = 0.175#0.2084     # Wb, at T_ref
T_ref = 25.0
I_rated = 10.0       # A

# ferrite magnet, agreed temperature coefficient
alpha_psi_ferrite = -0.0020   # 1/degC  (-0.20 %/C)

s = Ld - Lq   # < 0 for this motor (Lq > Ld)


def psi_of_T(T):
    return psi_nom * (1.0 + alpha_psi_ferrite * (T - T_ref))


# ---------------------------------------------------------------------
# The generic solve: two equations, two unknowns, solved together.
# ---------------------------------------------------------------------
def equations(x, Te_target, psi_f):
    id_, iq = x
    torque_eq = 1.5 * p * (psi_f * iq + s * id_ * iq) - Te_target
    mtpa_eq = id_ * psi_f + s * id_ ** 2 - s * iq ** 2
    return [torque_eq, mtpa_eq]


def solve_id_iq(Te_target, psi_f, guess):
    """fsolve on the 2x2 nonlinear system above. `guess` should be the
    previous point on the curve (continuation) -- MTPA is smooth, so the
    previous solution is an excellent starting point for the next."""
    sol, info, ier, msg = fsolve(equations, guess, args=(Te_target, psi_f),
                                  full_output=True, xtol=1e-12)
    if ier != 1:
        raise RuntimeError(f"fsolve did not converge at Te={Te_target}: {msg}")
    return sol[0], sol[1]


def boundary_equations(x, psi_f):
    """Where the MTPA curve crosses the current limit: MTPA condition +
    id^2+iq^2 = I_rated^2, solved the same way (2 eq, 2 unknown)."""
    id_, iq = x
    mtpa_eq = id_ * psi_f + s * id_ ** 2 - s * iq ** 2
    current_eq = id_ ** 2 + iq ** 2 - I_rated ** 2
    return [mtpa_eq, current_eq]


def mtpa_current_limit_point(psi_f, guess=(-2.0, 9.5)):
    sol, info, ier, msg = fsolve(boundary_equations, guess, args=(psi_f,),
                                  full_output=True, xtol=1e-12)
    if ier != 1:
        raise RuntimeError(f"fsolve did not converge on boundary: {msg}")
    return sol[0], sol[1]


def torque(id_, iq, psi_f):
    return 1.5 * p * (psi_f * iq + s * id_ * iq)


def const_torque_trajectory(Te_target, psi_f, id_range=None, n=400):
    """Constant-torque trajectory (iso-torque hyperbola): solve the SAME
    torque equation for iq as a function of id, at a fixed Te_target --

        Te_target = 1.5*p*(psi_f*iq + s*id*iq) = 1.5*p*iq*(psi_f + s*id)
        => iq(id) = Te_target / (1.5*p*(psi_f + s*id))

    This is a direct algebraic solve (one equation, one unknown, no
    fsolve needed) -- id is free, iq is determined. It blows up at
    id = -psi_f/s (where psi_f + s*id = 0); points near that asymptote
    and outside a sane current window are masked out (NaN) so the plot
    doesn't get dominated by the asymptote.
    """
    if id_range is None:
        id_range = (-I_rated, I_rated)
    id_ = np.linspace(id_range[0], id_range[1], n)
    denom = psi_f + s * id_
    with np.errstate(divide="ignore", invalid="ignore"):
        iq = Te_target / (1.5 * p * denom)
    # mask: keep only the physically sane branch (iq > 0, within ~1.5x I_rated)
    bad = (iq <= 0) | (iq > 1.5 * I_rated) | (np.abs(denom) < 1e-6)
    iq[bad] = np.nan
    return id_, iq


def beta_angle_deg(id_, iq):
    """Current angle beta, measured from the q-axis (standard convention):
    id = -Is*sin(beta), iq = Is*cos(beta)  =>  beta = atan2(-id, iq).
    beta = 0 at id=0 (pure q-axis current); beta grows as id becomes more
    negative (current vector rotates away from the q-axis, toward -d)."""
    return np.degrees(np.arctan2(-id_, iq))


def mtpa_curve(psi_f, n=150):
    """Sweep Te_target from ~0 up to the current-limited max, solving the
    2x2 system at each step with continuation."""
    id_b, iq_b = mtpa_current_limit_point(psi_f)
    Te_max = torque(id_b, iq_b, psi_f)

    Te_sweep = np.linspace(0.02 * Te_max, Te_max, n)
    ids, iqs = np.empty(n), np.empty(n)
    guess = (0.0, 1e-3)   # start near the origin
    for k, Te_t in enumerate(Te_sweep):
        id_, iq = solve_id_iq(Te_t, psi_f, guess)
        ids[k], iqs[k] = id_, iq
        guess = (id_, iq)   # continuation: next guess = this solution
    return ids, iqs, Te_sweep, Te_max


def main():
    temps = [25, 45, 65, 85, 105, 120]
    colors = {
        25:  "#86b6ef", 45:  "#5598e7", 65:  "#2a78d6",
        85:  "#256abf", 105: "#184f95", 120: "#0d366b",
    }

    fig, (ax, ax_beta) = plt.subplots(
        1, 2, figsize=(14, 7), dpi=150, gridspec_kw={"wspace": 0.3})

    print(f"{'T (C)':>7} {'psi_f (Wb)':>12} {'Te_max_MTPA (Nm)':>18} {'beta @ Te_max (deg)':>20}")
    Te_max_nom = None
    for T in temps:
        psi_f = psi_of_T(T)
        ids, iqs, Te_sweep, Te_max = mtpa_curve(psi_f)
        ax.plot(ids, iqs, color=colors[T], linewidth=2, label=f"{T:.0f} °C")

        # beta (MTPA current angle) vs. torque, at this temperature --
        # this is the curve that shows the angle shifting with temperature.
        beta_deg = beta_angle_deg(ids, iqs)
        ax_beta.plot(Te_sweep, beta_deg, color=colors[T], linewidth=2,
                     label=f"{T:.0f} °C")

        print(f"{T:7.0f} {psi_f:12.5f} {Te_max:18.3f} {beta_deg[-1]:20.2f}")
        if T == T_ref:
            Te_max_nom = Te_max   # nominal-temperature current-limited max torque

    # stator current limit circle -- full range, +I_rated to -I_rated on id
    theta = np.linspace(0, np.pi, 200)
    ax.plot(I_rated * np.cos(theta), I_rated * np.sin(theta),
            color="#c3c2b7", linewidth=1.5, linestyle="--", zorder=0,
            label=f"$I_s$ = {I_rated:.0f} A limit")

    # constant-torque trajectory, at the nominal (25C) current-limited max
    # torque -- this hyperbola is tangent to the 25C MTPA curve exactly at
    # the point where that curve meets the current limit (the textbook
    # definition of MTPA: minimum current for a given torque).
    id_ct, iq_ct = const_torque_trajectory(Te_max_nom, psi_of_T(T_ref))
    ax.plot(id_ct, iq_ct, color="#eb6834", linewidth=1.5, linestyle=":",
            label=f"const. torque = {Te_max_nom:.2f} Nm\n(@ {T_ref:.0f}°C, tangent to MTPA)")

    ax.set_xlabel("$i_d$ (A)", color="#52514e")
    ax.set_ylabel("$i_q$ (A)", color="#52514e")
    ax.set_title(
        "MTPA curves vs. magnet temperature (ferrite, α$_m$ = -0.20%/°C)\n"
        "each point solved from Te_target + MTPA eq., 2 equations / 2 unknowns "
        "(TI SPRACF3 motor)", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#52514e")
    ax.set_xlim(-I_rated * 1.05, I_rated * 1.05)
    ax.set_ylim(0, I_rated * 1.05)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", title="magnet temp",
               title_fontsize=8)

    # ---- right panel: MTPA current angle (beta) vs torque, per temperature ----
    ax_beta.set_xlabel("$T_e$ (Nm)", color="#52514e")
    ax_beta.set_ylabel(r"MTPA current angle $\beta$ (deg, from $q$-axis)", color="#52514e")
    ax_beta.set_title(
        "MTPA angle $\\beta$ shifts with magnet temperature\n"
        r"$\beta = \mathrm{atan2}(-i_d, i_q)$ along each temperature's MTPA curve",
        fontsize=9.5)
    ax_beta.spines[["top", "right"]].set_visible(False)
    ax_beta.tick_params(colors="#52514e")
    ax_beta.legend(frameon=False, fontsize=8, loc="upper left", title="magnet temp",
                   title_fontsize=8)

    fig.tight_layout()
    fig.savefig("mtpa_generic_solver.png", bbox_inches="tight")
    print("\nsaved mtpa_generic_solver.png")


if __name__ == "__main__":
    main()