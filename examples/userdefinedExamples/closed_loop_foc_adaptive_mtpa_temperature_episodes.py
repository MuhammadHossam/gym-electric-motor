import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure classic_controllers path is accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
classic_controllers_dir = os.path.abspath(os.path.join(current_dir, "..", "classic_controllers"))
if classic_controllers_dir not in sys.path:
    sys.path.append(classic_controllers_dir)

import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.envs.motors import ActionType, ControlType, Motor, MotorType
from gym_electric_motor.physical_systems.mechanical_loads import MechanicalLoad
from classic_controllers import Controller
from controllers.torque_to_current_conversion import TorqueToCurrentConversion


class ConfigurableStepLoad(MechanicalLoad):
    """Mechanical load with configurable step load torque disturbance."""

    def __init__(
        self,
        j_load=1e-4,
        t_load_initial=0.0,
        t_load_step=1.0,
        step_time=2.0,
        viscous_friction=1e-4,
        **kwargs,
    ):
        super().__init__(j_load=j_load, **kwargs)
        self.t_load_initial = t_load_initial
        self.t_load_step = t_load_step
        self.step_time = step_time
        self.viscous_friction = viscous_friction
        self.current_load_torque = t_load_initial

    def mechanical_ode(self, t, mechanical_state, torque):
        omega = mechanical_state[self.OMEGA_IDX]
        base_torque = self.t_load_step if t >= self.step_time else self.t_load_initial
        total_load = base_torque + self.viscous_friction * omega
        self.current_load_torque = total_load
        d_omega = (torque - total_load) / self._j_total
        return np.array([d_omega])


def main():
    # =========================================================================
    # 1. BASE MOTOR PARAMETERS (at Reference Temperature T_0 = 20 deg C)
    # =========================================================================
    p = 2                   # Pole pair number
    Rs_20 = 1.15            # Stator phase resistance at 20 deg C [Ohm]
    Ld = 18.5e-3            # Direct-axis inductance L_d [H]
    Lq = 38.0e-3            # Quadrature-axis inductance L_q [H]
    psi_p_20 = 0.175        # Permanent magnet flux linkage at 20 deg C [Vs]
    j = 2.6e-3              # Rotor moment of inertia [kg*m^2]
    u_dc_link = 400.0       # DC-link voltage [V]

    # Thermal coefficients
    T_0 = 20.0              # Reference baseline temperature [deg C]
    alpha_mag = -0.0011     # NdFeB reversible temp coeff of flux: -0.11 % / deg C
    alpha_cu = 0.00393      # Copper temp coeff of resistance: +0.393 % / deg C

    # Operational bases & ratings
    f_rated = 120           # Rated frequency [Hz]
    I_rated = 9.2           # Rated current [A]
    w_rated = 2 * np.pi * f_rated
    wm_rated = w_rated / p
    v_rated = u_dc_link / np.sqrt(3)
    T_rated = 1.5 * p * psi_p_20 * I_rated

    f_base = 200            # Base frequency [Hz]
    w_base = 2 * np.pi * f_base
    wm_base = w_base / p
    I_base = I_rated
    v_base = v_rated
    T_base = T_rated

    # =========================================================================
    # 2. TEMPERATURE EPISODES CONFIGURATION
    # =========================================================================
    # Define temperatures for each episode in degrees Celsius
    temperatures = [20.0, 60.0, 100.0, 140.0]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]  # Blue, Green, Orange, Red

    # =========================================================================
    # 3. CONTROLLER TUNING (Tuned at baseline 20 deg C)
    # =========================================================================
    fc_spd = 10
    phi_spd = 80 * np.pi / 180
    wc_spd = 2 * np.pi * fc_spd
    kp_spd_phy = j * np.sin(phi_spd) * wc_spd
    ki_spd_phy = (wc_spd * (1 / np.tan(phi_spd))) * kp_spd_phy
    kp_spd_pu = kp_spd_phy * wm_base / T_base
    ki_spd_pu = ki_spd_phy * wm_base / T_base

    fc_curr = 200
    phi_curr = 80 * np.pi / 180
    wc_curr = 2 * np.pi * fc_curr
    kpd_curr_phy = Ld * np.sin(phi_curr) * wc_curr
    kid_curr_phy = (wc_curr * (1 / np.tan(phi_curr))) * kpd_curr_phy
    kpd_curr_pu = kpd_curr_phy * I_base / v_base
    kid_curr_pu = kid_curr_phy * I_base / v_base

    kpq_curr_phy = Lq * np.sin(phi_curr) * wc_curr
    kiq_curr_phy = (wc_curr * (1 / np.tan(phi_curr))) * kpq_curr_phy
    kpq_curr_pu = kpq_curr_phy * I_base / v_base
    kiq_curr_pu = kiq_curr_phy * I_base / v_base

    current_d_controller = {"controller_type": "pi_controller", "p_gain": kpd_curr_pu, "i_gain": kid_curr_pu}
    current_q_controller = {"controller_type": "pi_controller", "p_gain": kpq_curr_pu, "i_gain": kiq_curr_pu}
    speed_controller = {"controller_type": "pi_controller", "p_gain": kp_spd_pu, "i_gain": ki_spd_pu}

    motor_parameter = {
        "p": p,
        "r_s": Rs_20,
        "l_d": Ld,
        "l_q": Lq,
        "psi_p": psi_p_20,
        "j_rotor": j,
    }
    nominal_values = {"omega": wm_rated, "torque": T_rated, "i": I_rated, "u": v_rated, "epsilon": 2 * np.pi}
    limit_values = {"omega": wm_base, "torque": T_base, "i": I_base, "u": v_base, "epsilon": 2 * np.pi}

    # Speed step reference (0.3 PU = 1800 RPM)
    step_reference_generator = rg.ConstReferenceGenerator(reference_state="omega", reference_value=0.3)

    # Mechanical load with step torque at t = 2.0s
    step_load_time = 2.0
    mechanical_load = ConfigurableStepLoad(
        j_load=0,
        t_load_initial=0.0,
        t_load_step=1.0,
        step_time=step_load_time,
        viscous_friction=1e-4,
    )

    # Simulation timing
    tau = 1e-4                  # 100 microseconds
    simulation_duration = 4.0   # 4 seconds per episode
    n_simulation_steps = int(simulation_duration / tau)

    # =========================================================================
    # 4. ENVIRONMENT & CONTROLLER INITIALIZATION
    # =========================================================================
    motor = Motor(MotorType.PermanentMagnetSynchronousMotor, ControlType.SpeedControl, ActionType.Continuous)
    env = gem.make(
        motor.env_id(),
        supply=dict(u_nominal=u_dc_link),
        load=mechanical_load,
        reference_generator=step_reference_generator,
        motor=dict(
            motor_parameter=motor_parameter,
            nominal_values=nominal_values,
            limit_values=limit_values,
        ),
        tau=tau,
    )

    current_controllers = [current_d_controller, current_q_controller]
    speed_controllers = [speed_controller]
    stages = [current_controllers, speed_controllers]

    controller = Controller.make(
        env,
        stages=stages,
        torque_control="analytical",
        decoupling=False,
        plot_torque=False,
    )

    ps = env.unwrapped.physical_system
    motor_instance = ps.electrical_motor
    state_names = ps.state_names
    limits = ps.limits

    omega_idx = state_names.index("omega")
    torque_idx = state_names.index("torque")
    i_sd_idx = state_names.index("i_sd")
    i_sq_idx = state_names.index("i_sq")
    rads_to_rpm = 30.0 / np.pi

    time_arr = np.linspace(0, simulation_duration, n_simulation_steps, endpoint=False)

    # Store episode results
    episode_results = []

    print("================================================================================")
    print(f"Running Multi-Episode ADAPTIVE MTPA Temperature Simulation ({len(temperatures)} Episodes)")
    print(" (Both Physical Plant AND Controller MTPA Maps are updated per episode)")
    print("================================================================================")

    for ep_idx, T_celsius in enumerate(temperatures):
        # Calculate temperature-dependent parameters
        psi_p_T = psi_p_20 * (1.0 + alpha_mag * (T_celsius - T_0))
        Rs_T = Rs_20 * (1.0 + alpha_cu * (T_celsius - T_0))

        # 1. Update physical plant parameters
        motor_instance.motor_parameter["psi_p"] = psi_p_T
        motor_instance.motor_parameter["r_s"] = Rs_T

        # 2. Update the CONTROLLER's MTPA model & recalculate analytical LUT for this temperature
        controller.psi_p = psi_p_T
        controller.mp["r_s"] = Rs_T
        controller.mp["psi_p"] = psi_p_T
        controller.torque_controller = TorqueToCurrentConversion(
            env,
            torque_control="analytical",
            plot_torque=False,
        )

        delta_psi_pct = (psi_p_T - psi_p_20) / psi_p_20 * 100.0
        delta_rs_pct = (Rs_T - Rs_20) / Rs_20 * 100.0

        print(f"\n[Episode {ep_idx + 1}/{len(temperatures)}] Temperature: {T_celsius:.1f} °C")
        print(f"  -> Physical & Controller PM Flux (psi_p): {psi_p_T:.4f} Vs ({delta_psi_pct:+.1f}%)")
        print(f"  -> Physical & Controller Stator Rs: {Rs_T:.3f} Ohm ({delta_rs_pct:+.1f}%)")
        print(f"  -> Controller MTPA Lookup Table successfully re-computed for {T_celsius:.1f} °C.")

        # Reset environment and controller
        (state, reference), _ = env.reset()
        controller.reset()

        # Pre-allocate arrays for this episode
        omega_act_rpm = np.empty(n_simulation_steps)
        omega_ref_rpm = np.empty(n_simulation_steps)
        i_sq_act = np.empty(n_simulation_steps)
        i_sd_act = np.empty(n_simulation_steps)
        torque_act = np.empty(n_simulation_steps)

        for step in range(n_simulation_steps):
            action = controller.control(state, reference)

            w_act = state[omega_idx] * limits[omega_idx]
            w_ref = reference[0] * limits[omega_idx]

            omega_act_rpm[step] = w_act * rads_to_rpm
            omega_ref_rpm[step] = w_ref * rads_to_rpm
            i_sd_act[step] = state[i_sd_idx] * limits[i_sd_idx]
            i_sq_act[step] = state[i_sq_idx] * limits[i_sq_idx]
            torque_act[step] = state[torque_idx] * limits[torque_idx]

            (state, reference), _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        episode_results.append({
            "temp": T_celsius,
            "psi_p": psi_p_T,
            "r_s": Rs_T,
            "omega_act_rpm": omega_act_rpm,
            "omega_ref_rpm": omega_ref_rpm,
            "i_sq_act": i_sq_act,
            "i_sd_act": i_sd_act,
            "torque_act": torque_act,
        })

    env.close()

    # =========================================================================
    # 5. GENERATE COMPARATIVE STATIC FIGURE
    # =========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle("Adaptive MTPA FOC Temperature Sensitivity Analysis (Updated MTPA per Episode)", fontsize=13, fontweight="bold")

    # 1. Speed Tracking
    for idx, ep in enumerate(episode_results):
        lbl = f"T = {ep['temp']:.0f}°C (Adapted MTPA: ψ={ep['psi_p']:.3f} Vs, Rs={ep['r_s']:.2f} Ω)"
        axes[0].plot(time_arr, ep["omega_act_rpm"], color=colors[idx], label=lbl, linewidth=1.5)
    axes[0].plot(time_arr, episode_results[0]["omega_ref_rpm"], "k--", label="ω* Reference", linewidth=1.8)
    axes[0].set_ylabel("Speed [RPM]", fontweight="bold")
    axes[0].set_title("1. Speed Response vs. Operating Temperature (with Adaptive MTPA)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.7)
    axes[0].legend(loc="lower right", fontsize=9)

    # 2. Quadrature Current i_sq
    for idx, ep in enumerate(episode_results):
        lbl = f"i_sq @ {ep['temp']:.0f}°C"
        axes[1].plot(time_arr, ep["i_sq_act"], color=colors[idx], label=lbl, linewidth=1.5)
    axes[1].set_ylabel("i_sq Current [A]", fontweight="bold")
    axes[1].set_title("2. Optimal Quadrature Current (i_sq) from Temperature-Adapted MTPA", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.7)
    axes[1].legend(loc="upper right", fontsize=9)

    # 3. Direct Current i_sd
    for idx, ep in enumerate(episode_results):
        lbl = f"i_sd @ {ep['temp']:.0f}°C"
        axes[2].plot(time_arr, ep["i_sd_act"], color=colors[idx], label=lbl, linewidth=1.5)
    axes[2].set_ylabel("i_sd Current [A]", fontweight="bold")
    axes[2].set_xlabel("Time [s]", fontweight="bold")
    axes[2].set_title("3. Optimal Direct Current (i_sd) Reluctance Injection (Adapted MTPA)", fontsize=11)
    axes[2].grid(True, linestyle=":", alpha=0.7)
    axes[2].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    png_path = os.path.join(current_dir, "adaptive_mtpa_temperature_comparison.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nStatic comparison figure saved to: {png_path}")

    # =========================================================================
    # 6. GENERATE INTERACTIVE PLOTLY COMPARISON DASHBOARD
    # =========================================================================
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plotly_fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "1. Rotor Speed Response (Adaptive MTPA) [RPM]",
            "2. Quadrature Current (i_sq) under Temperature-Aware MTPA [A]",
            "3. Direct Current (i_sd) Reluctance Injection [A]",
        ],
    )

    # Speed reference
    plotly_fig.add_trace(
        go.Scattergl(x=time_arr, y=episode_results[0]["omega_ref_rpm"], mode="lines", name="ω* Ref [RPM]", line=dict(color="black", dash="dash", width=2)),
        row=1, col=1,
    )

    for idx, ep in enumerate(episode_results):
        t_label = f"{ep['temp']:.0f}°C (Adapted MTPA: ψ={ep['psi_p']:.3f}Vs, R={ep['r_s']:.2f}Ω)"
        c = colors[idx]
        # Speed
        plotly_fig.add_trace(
            go.Scattergl(x=time_arr, y=ep["omega_act_rpm"], mode="lines", name=f"Speed @ {t_label}", line=dict(color=c, width=1.8)),
            row=1, col=1,
        )
        # i_sq
        plotly_fig.add_trace(
            go.Scattergl(x=time_arr, y=ep["i_sq_act"], mode="lines", name=f"i_sq @ {ep['temp']:.0f}°C", line=dict(color=c, width=1.8)),
            row=2, col=1,
        )
        # i_sd
        plotly_fig.add_trace(
            go.Scattergl(x=time_arr, y=ep["i_sd_act"], mode="lines", name=f"i_sd @ {ep['temp']:.0f}°C", line=dict(color=c, width=1.8)),
            row=3, col=1,
        )

    plotly_fig.update_layout(
        title=dict(
            text="<b>Adaptive MTPA Multi-Episode Temperature Dashboard (Controller MTPA Updated Every Episode)</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        height=900,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=60, r=40, t=100, b=50),
    )

    plotly_fig.update_xaxes(title_text="Time [s]", row=3, col=1, rangeslider=dict(visible=True, thickness=0.04))
    plotly_fig.update_yaxes(title_text="Speed [RPM]", row=1, col=1)
    plotly_fig.update_yaxes(title_text="i_sq Current [A]", row=2, col=1)
    plotly_fig.update_yaxes(title_text="i_sd Current [A]", row=3, col=1)

    html_path = os.path.join(current_dir, "adaptive_mtpa_temperature_dashboard.html")
    plotly_fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Interactive dashboard generated at: {html_path}")


if __name__ == "__main__":
    main()
