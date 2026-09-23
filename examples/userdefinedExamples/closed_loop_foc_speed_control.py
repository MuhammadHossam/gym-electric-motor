import os
import sys
import numpy as np

# Ensure classic_controllers path is accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
classic_controllers_dir = os.path.abspath(os.path.join(current_dir, "..", "classic_controllers"))
if classic_controllers_dir not in sys.path:
    sys.path.append(classic_controllers_dir)

import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.envs.motors import ActionType, ControlType, Motor, MotorType
from gym_electric_motor.physical_systems.mechanical_loads import MechanicalLoad
from gym_electric_motor.visualization import MotorDashboard
from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot


class ConfigurableStepLoad(MechanicalLoad):
    """Mechanical load with configurable step load torque disturbance."""

    def __init__(
        self,
        j_load=1e-4,
        t_load_initial=0.0,
        t_load_step=1.0,
        step_time=0.5,
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
        
        # Step load torque profile
        base_torque = self.t_load_step if t >= self.step_time else self.t_load_initial
        
        # Total load = Step load + Viscous damping (B * omega)
        total_load = base_torque + self.viscous_friction * omega
        self.current_load_torque = total_load

        # J_total * domega/dt = T_motor - T_load
        d_omega = (torque - total_load) / self._j_total
        return np.array([d_omega])


def main():
    

    # =========================================================================
    # 1. DC-LINK VOLTAGE / POWER SUPPLY SETTINGS / LIMITs 
    # =========================================================================
    # Nominal DC-link voltage U_dc [V]
    p = 2               # Number of pole pairs 
    Rs = 1.15           # Stator phase resistance [Ohm]
    Ld = 18.5e-3         # Direct-axis inductance L_d [H]
    Lq = 38e-3         # Quadrature-axis inductance L_q [H]
    psi_p = 0.175       # Permanent magnet flux linkage [Vs] (0 for SynRM)
    j = 2.6e-3          # Rotor moment of inertia [kg*m^2]
    u_dc_link = 200.0  # Set your desired DC-link voltage in Volts
    


    f_rated = 120      # The rated frequency 
    I_rated = 9.2       # The rated current 
    w_rated = 2*np.pi*f_rated # Rated electrical angular speed.
    wm_rated = w_rated/p # Rated mechanical angular speed.
    v_rated = u_dc_link/np.sqrt(3) # Rated phase voltage
    T_rated = 1.5 * p * psi_p * I_rated # Rated Torque. 

    v_base = v_rated    # The Maximum Limited voltage 
    I_base = I_rated        # The Maximum limited current
    f_base = 200       # The base frequency (Max Limit)
    w_base = 2*np.pi*f_base   # The base angular electrical speed. 
    wm_base = w_base/p # base mechanical angular speed.
    T_base = T_rated  #p_base/(w_base/p) # Base Torque. 
    

    
    # =========================================================================
    # 2. PI Controllers Parameters  
    # =========================================================================
    
    # Speed Controller 
    fc_spd = 10          
    phi_spd = 80*np.pi/180       # The phase margin in radian
    wc_spd = 2*np.pi*fc_spd          # The cross over frequency
    kp_spd_phy = j * np.sin(phi_spd) * wc_spd # The physical Kp parameter
    ki_spd_phy = (wc_spd * (1 / np.tan(phi_spd))) * kp_spd_phy  # The physical Ki (Parallel PI)
    kp_spd_pu = kp_spd_phy * wm_base/T_base # The normalized Kp
    ki_spd_pu = ki_spd_phy * wm_base/T_base # The normalized Kp

    # D-axis Current Controller 
    fc_curr = 200          
    phi_curr = 80*np.pi/180       # The phase margin in radian
    wc_curr = 2*np.pi*fc_curr          # The cross over frequency


    kpd_curr_phy = Ld * np.sin(phi_curr) * wc_curr # The physical Kp parameter
    kid_curr_phy = (wc_curr * (1 / np.tan(phi_curr))) * kpd_curr_phy  # The physical Ki (Parallel PI)
    kpd_curr_pu = kpd_curr_phy * I_base/v_base # The normalized Kp
    kid_curr_pu = kid_curr_phy * I_base/v_base # The normalized Kp

    kpq_curr_phy = Lq * np.sin(phi_curr) * wc_curr # The physical Kp parameter
    kiq_curr_phy = (wc_curr * (1 / np.tan(phi_curr))) * kpq_curr_phy  # The physical Ki (Parallel PI)
    kpq_curr_pu = kpq_curr_phy * I_base/v_base # The normalized Kp
    kiq_curr_pu = kiq_curr_phy * I_base/v_base # The normalized Kp

    # =========================================================================
    # 2. MOTOR PARAMETERS (Fill with your motor data)
    # =========================================================================
    # Motor type selection:
    #   MotorType.PermanentMagnetSynchronousMotor ('PMSM')
    #   MotorType.SynchronousReluctanceMotor ('SynRM')
    selected_motor_type = MotorType.PermanentMagnetSynchronousMotor
    # Equivalent circuit & physical parameters
    motor_parameter = {
        "p": p,              # Pole pair number [-]
        "r_s": Rs,        # Stator phase resistance [Ohm]
        "l_d": Ld,      # Direct-axis inductance L_d [H]
        "l_q": Lq,       # Quadrature-axis inductance L_q [H]
        "psi_p": psi_p,      # Permanent magnet flux linkage [Vs] (0 for SynRM)
        "j_rotor": j,  # Rotor moment of inertia [kg*m^2]
    }

    # Nominal (rated) operational values
    nominal_values = {
        "omega": wm_rated,  # Nominal mechanical speed [rad/s] (e.g., 10,000 RPM)
        "torque": T_rated,                # Nominal torque [Nm]
        "i": I_rated,                       # Nominal phase current amplitude [A]
        "u": v_rated,    # Nominal phase voltage amplitude [V]
        "epsilon": 2*np.pi,           # Angle normalization [rad]
    }

    # Absolute physical limits (safety boundaries)
    limit_values = {
        "omega": wm_base,  # Max speed limit [rad/s]
        "torque": T_base,             # Max torque limit [Nm]
        "i": I_base,                  # Max peak current limit [A]
        "u": v_base,                  # Max inverter voltage limit [V]
        "epsilon": 2*np.pi,            # Max angle [rad]
    }

    # =========================================================================
    # 3. SPEED REFERENCE GENERATOR CONFIGURATION (Step to 0.5 PU)
    # =========================================================================
    # ConstReferenceGenerator maintains a constant 0.5 PU speed reference (50% base speed)
    # without any other step ups or downs throughout the simulation.
    step_reference_generator = rg.ConstReferenceGenerator(
        reference_state="omega",
        reference_value=0.15,
    )

    # =========================================================================
    # 4. CONTROLLER CONFIGURATION & PI GAINS (Fill with your tuning parameters)
    # =========================================================================
    # Set to False to use your custom PI gains below, or True for auto-tuning via Symmetrical Optimum
    use_auto_tuning = False
    a_symmetrical_optimum = 4  # Symmetrical optimum tuning factor (if use_auto_tuning=True)

    # --- Inner Current Loop: Direct-axis (d-axis) PI Controller ---
    current_d_controller = {
        "controller_type": "pi_controller",
        "p_gain": kpd_curr_pu,      # Proportional gain K_p,d [V/A]
        "i_gain": kid_curr_pu,    # Integral gain K_i,d [V/(A*s)]
    }

    # --- Inner Current Loop: Quadrature-axis (q-axis) PI Controller ---
    current_q_controller = {
        "controller_type": "pi_controller",
        "p_gain": kpq_curr_pu,      # Proportional gain K_p,q [V/A]
        "i_gain": kiq_curr_pu,   # Integral gain K_i,q [V/(A*s)]
    }

    # --- Outer Speed Loop: Speed PI Controller ---
    speed_controller = {
        "controller_type": "pi_controller",
        "p_gain": kp_spd_pu,     # Proportional gain K_p,w [Nm/(rad/s)]
        "i_gain": ki_spd_pu,   # Integral gain K_i,w [Nm/(rad)]
    }

    # Torque-to-current setpoint calculation mode: 'analytical', 'interpolate', or 'online'
    torque_control_mode = "analytical"

    # Cross-coupling & back-EMF feedforward decoupling in current loop (True/False)
    use_decoupling = False

   

    # =========================================================================
    # 5. MECHANICAL LOAD & STEP LOAD SETTINGS (Fill with your load parameters)
    # =========================================================================
    j_load = 0            # Coupled load moment of inertia [kg*m^2]
    t_load_initial = 0.0      # Initial load torque before step [Nm]
    t_load_step = 1.0         # Step load torque applied at step time [Nm]
    step_load_time = 5.0      # Time in seconds when the load step is applied [s]
    viscous_friction = 1e-4   # Viscous damping coefficient B [Nm*s/rad]

    mechanical_load = ConfigurableStepLoad(
        j_load=j_load,
        t_load_initial=t_load_initial,
        t_load_step=t_load_step,
        step_time=step_load_time,
        viscous_friction=viscous_friction,
    )

    # =========================================================================
    # 6. SIMULATION & TIMING SETTINGS
    # =========================================================================
    tau = 1e-4                  # Sampling time / step size [s] (100 microseconds / 10 kHz)
    simulation_duration = 10.0  # Total simulation time in seconds [s]
    n_simulation_steps = int(simulation_duration / tau)  # 100,000 simulation steps (10.0 seconds)

    # =========================================================================
    # 7. ENVIRONMENT INITIALIZATION
    # =========================================================================
    motor = Motor(
        selected_motor_type,
        ControlType.SpeedControl,
        ActionType.Continuous
    )

    # Variables to record and display on the live dashboard
    external_ref_plots = [
        ExternallyReferencedStatePlot(state)
        for state in ["omega", "torque", "i_sd", "i_sq", "u_sd", "u_sq"]
    ]
    motor_dashboard = MotorDashboard(additional_plots=external_ref_plots)

    # Create the GEM environment with user-defined motor, supply, load & step reference generator
    env = gem.make(
        motor.env_id(),
        supply=dict(u_nominal=u_dc_link),
        load=mechanical_load,
        reference_generator=step_reference_generator,
        visualization=motor_dashboard,
        motor=dict(
            motor_parameter=motor_parameter,
            nominal_values=nominal_values,
            limit_values=limit_values,
        ),
        tau=tau,
    )

    # =========================================================================
    # 8. CONTROLLER INITIALIZATION
    # =========================================================================
    if use_auto_tuning:
        stages = None  # Auto-tune based on Symmetrical Optimum
    else:
        # Structure: [[d_current, q_current], [speed]]
        current_controllers = [current_d_controller, current_q_controller]
        speed_controllers = [speed_controller]
        stages = [current_controllers, speed_controllers]

    controller = Controller.make(
        env,
        stages=stages,
        external_ref_plots=external_ref_plots,
        torque_control=torque_control_mode,
        decoupling=use_decoupling,
        automated_gain=use_auto_tuning,
        a=a_symmetrical_optimum,
    )

    # =========================================================================
    # 9. CLOSED-LOOP SIMULATION RUN & DATA LOGGING
    # =========================================================================
    import matplotlib.pyplot as plt

    print(f"Starting closed-loop FOC simulation for {motor.env_id()} with StepReferenceGenerator & StepLoad...")
    (state, reference), _ = env.reset()
    controller.reset()

    # State indices and physical limit scaling
    ps = env.unwrapped.physical_system
    state_names = ps.state_names
    limits = ps.limits

    omega_idx = state_names.index("omega")
    torque_idx = state_names.index("torque")
    i_sd_idx = state_names.index("i_sd")
    i_sq_idx = state_names.index("i_sq")
    i_a_idx = state_names.index("i_a")
    i_b_idx = state_names.index("i_b")
    i_c_idx = state_names.index("i_c")

    rads_to_rpm = 30.0 / np.pi

    # Pre-allocated arrays for fast logging
    time_arr = np.linspace(0, simulation_duration, n_simulation_steps, endpoint=False)
    omega_ref_rpm = np.empty(n_simulation_steps)
    omega_act_rpm = np.empty(n_simulation_steps)
    i_sd_ref = np.empty(n_simulation_steps)
    i_sd_act = np.empty(n_simulation_steps)
    i_sq_ref = np.empty(n_simulation_steps)
    i_sq_act = np.empty(n_simulation_steps)
    i_a = np.empty(n_simulation_steps)
    i_b = np.empty(n_simulation_steps)
    i_c = np.empty(n_simulation_steps)
    torque_ref = np.empty(n_simulation_steps)
    torque_act = np.empty(n_simulation_steps)
    torque_load = np.empty(n_simulation_steps)

    for step in range(n_simulation_steps):
        # Calculate continuous voltage action from cascaded PI FOC controller
        action = controller.control(state, reference)

        # Log physical quantities (de-normalized using limits)
        w_act = state[omega_idx] * limits[omega_idx]
        w_ref = reference[0] * limits[omega_idx]

        omega_ref_rpm[step] = w_ref * rads_to_rpm
        omega_act_rpm[step] = w_act * rads_to_rpm
        i_sq_ref[step] = controller.ref[0] * limits[i_sq_idx]
        i_sd_ref[step] = controller.ref[1] * limits[i_sd_idx]
        torque_ref[step] = controller.ref[2] * limits[torque_idx]
        torque_act[step] = state[torque_idx] * limits[torque_idx]
        torque_load[step] = mechanical_load.current_load_torque
        i_sd_act[step] = state[i_sd_idx] * limits[i_sd_idx]
        i_sq_act[step] = state[i_sq_idx] * limits[i_sq_idx]
        i_a[step] = state[i_a_idx] * limits[i_a_idx]
        i_b[step] = state[i_b_idx] * limits[i_b_idx]
        i_c[step] = state[i_c_idx] * limits[i_c_idx]

        # Advance environment by one step
        (state, reference), reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            (state, reference), _ = env.reset()
            controller.reset()

    print("Simulation completed successfully. Generating high-resolution interactive dashboard...")
    env.close()

    # =========================================================================
    # 10. PLOT SPEED FEEDBACK VS REFERENCE & MOTOR CURRENTS (STATIC & INTERACTIVE)
    # =========================================================================
    # --- 10A. Interactive WebGL Plotly Dashboard (Zoom, Pan, Hover Enabled) ---
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plotly_fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "1. Speed Control: Reference vs. Actual Feedback [RPM]",
            "2. Direct & Quadrature Currents (i_sd, i_sq) [A]",
            "3. Three-Phase Stator Currents (i_a, i_b, i_c) [A]",
            "4. Electromagnetic Torque & Load Disturbance [Nm]",
        ],
    )

    # Subplot 1: Speed
    plotly_fig.add_trace(
        go.Scattergl(x=time_arr, y=omega_ref_rpm, mode="lines", name="ω* Ref [RPM]", line=dict(color="red", dash="dash", width=2)),
        row=1, col=1,
    )
    plotly_fig.add_trace(
        go.Scattergl(x=time_arr, y=omega_act_rpm, mode="lines", name="ω Actual [RPM]", line=dict(color="#1f77b4", width=2)),
        row=1, col=1,
    )

    # Subplot 2: d-q Currents
    plotly_fig.add_trace(
        go.Scattergl(x=time_arr, y=i_sd_ref, mode="lines", name="i_sd* Ref [A]", line=dict(color="red", dash="dash", width=1.5)),
        row=2, col=1,
    )
    plotly_fig.add_trace(
        go.Scattergl(x=time_arr, y=i_sd_act, mode="lines", name="i_sd Actual [A]", line=dict(color="#1f77b4", width=1.5)),
        row=2, col=1,
    )
    plotly_fig.add_trace(
        go.Scattergl(x=time_arr, y=i_sq_ref, mode="lines", name="i_sq* Ref [A]", line=dict(color="#d62728", dash="dash", width=1.5)),
        row=2, col=1,
    )
    plotly_fig.add_trace(
        go.Scattergl(x=time_arr, y=i_sq_act, mode="lines", name="i_sq Actual [A]", line=dict(color="#2ca02c", width=1.5)),
        row=2, col=1,
    )

    # Subplot 3: 3-Phase Currents
    plotly_fig.add_trace(
        go.Scattergl(x=time_arr, y=i_a, mode="lines", name="i_a [A]", line=dict(color="#1f77b4", width=1.2)),
        row=3, col=1,
    )
    plotly_fig.add_trace(
        go.Scattergl(x=time_arr, y=i_b, mode="lines", name="i_b [A]", line=dict(color="#ff7f0e", width=1.2)),
        row=3, col=1,
    )
    plotly_fig.add_trace(
        go.Scattergl(x=time_arr, y=i_c, mode="lines", name="i_c [A]", line=dict(color="#2ca02c", width=1.2)),
        row=3, col=1,
    )

    # Subplot 4: Torque
    plotly_fig.add_trace(
        go.Scattergl(x=time_arr, y=torque_ref, mode="lines", name="T* Ref [Nm]", line=dict(color="red", dash="dash", width=1.5)),
        row=4, col=1,
    )
    plotly_fig.add_trace(
        go.Scattergl(x=time_arr, y=torque_act, mode="lines", name="T Actual [Nm]", line=dict(color="black", width=1.5)),
        row=4, col=1,
    )
    plotly_fig.add_trace(
        go.Scattergl(x=time_arr, y=torque_load, mode="lines", name="T_L Load [Nm]", line=dict(color="#2ca02c", dash="dot", width=2)),
        row=4, col=1,
    )

    plotly_fig.update_layout(
        title=dict(
            text="<b>Closed-Loop FOC Speed Control Simulation Dashboard (Interactive)</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        height=1000,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=60, r=40, t=100, b=50),
    )

    plotly_fig.update_xaxes(title_text="Time [s]", row=4, col=1, rangeslider=dict(visible=True, thickness=0.04))
    plotly_fig.update_yaxes(title_text="Speed [RPM]", row=1, col=1)
    plotly_fig.update_yaxes(title_text="Current [A]", row=2, col=1)
    plotly_fig.update_yaxes(title_text="Current [A]", row=3, col=1)
    plotly_fig.update_yaxes(title_text="Torque [Nm]", row=4, col=1)

    html_path = os.path.join(current_dir, "foc_simulation_dashboard.html")
    plotly_fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Interactive dashboard successfully generated at: {html_path}")

    # --- 10B. Static Matplotlib Figure ---
    try:
        fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
        fig.suptitle("Closed-Loop Field-Oriented Control (FOC) Simulation Results", fontsize=14, fontweight="bold")

        axes[0].plot(time_arr, omega_ref_rpm, "r--", linewidth=2.0, label=r"$\omega^*$ Reference")
        axes[0].plot(time_arr, omega_act_rpm, "b-", linewidth=1.5, label=r"$\omega$ Actual Feedback")
        axes[0].set_ylabel("Speed [RPM]", fontsize=11, fontweight="bold")
        axes[0].set_title("1. Speed Control: Reference vs. Actual Feedback", fontsize=11)
        axes[0].grid(True, linestyle=":", alpha=0.7)
        axes[0].legend(loc="upper right", framealpha=0.9)

        axes[1].plot(time_arr, i_sd_ref, "r--", linewidth=1.5, label=r"$i_{sd}^*$ Reference")
        axes[1].plot(time_arr, i_sd_act, "b-", linewidth=1.2, label=r"$i_{sd}$ Actual")
        axes[1].plot(time_arr, i_sq_ref, "m--", linewidth=1.5, label=r"$i_{sq}^*$ Reference")
        axes[1].plot(time_arr, i_sq_act, "g-", linewidth=1.2, label=r"$i_{sq}$ Actual")
        axes[1].set_ylabel("Current [A]", fontsize=11, fontweight="bold")
        axes[1].set_title(r"2. $d-q$ Frame Currents: Direct ($i_d$) & Quadrature ($i_q$)", fontsize=11)
        axes[1].grid(True, linestyle=":", alpha=0.7)
        axes[1].legend(loc="upper right", framealpha=0.9, ncol=2)

        axes[2].plot(time_arr, i_a, label=r"$i_a$", linewidth=1.0, color="#1f77b4")
        axes[2].plot(time_arr, i_b, label=r"$i_b$", linewidth=1.0, color="#ff7f0e")
        axes[2].plot(time_arr, i_c, label=r"$i_c$", linewidth=1.0, color="#2ca02c")
        axes[2].set_ylabel("Current [A]", fontsize=11, fontweight="bold")
        axes[2].set_title(r"3. Three-Phase Stator Currents ($i_a, i_b, i_c$)", fontsize=11)
        axes[2].grid(True, linestyle=":", alpha=0.7)
        axes[2].legend(loc="upper right", framealpha=0.9, ncol=3)

        axes[3].plot(time_arr, torque_ref, "r--", linewidth=1.5, label=r"$T^*$ Reference")
        axes[3].plot(time_arr, torque_act, "k-", linewidth=1.2, label=r"$T$ Actual Motor Torque")
        axes[3].plot(time_arr, torque_load, "g:", linewidth=2.0, label=r"$T_L$ Load Torque Disturbance")
        axes[3].set_ylabel("Torque [Nm]", fontsize=11, fontweight="bold")
        axes[3].set_xlabel("Time [s]", fontsize=11, fontweight="bold")
        axes[3].set_title("4. Electromagnetic Torque & Load Disturbance Response", fontsize=11)
        axes[3].grid(True, linestyle=":", alpha=0.7)
        axes[3].legend(loc="upper right", framealpha=0.9, ncol=3)

        plt.tight_layout()
        output_fig_path = os.path.join(current_dir, "foc_speed_current_response.png")
        plt.savefig(output_fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Static graph successfully saved to: {output_fig_path}")
    except Exception as e:
        print(f"Notice: Static PNG save skipped ({e}). Interactive HTML dashboard is ready.")


if __name__ == "__main__":
    main()
