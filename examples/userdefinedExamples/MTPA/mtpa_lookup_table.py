"""
MTPA lookup-table generator.

Builds an (Te -> id, iq) table at any configured magnet temperature(s), by
sweeping Te and solving the same 2-equation/2-unknown system
(torque eq + MTPA eq) as mtpa_generic_solver.py -- imported from there
directly, so there is exactly one place the physics equations live.

Configure TEMPERATURES_C and N_POINTS below and run. Writes one CSV per
temperature (Te, id, iq, Is columns) plus a combined CSV with all
temperatures stacked (adds a T_degC column) -- the combined file is what
you'd interpolate on-chip if the controller needs to pick a temperature
that isn't exactly one of the configured points.

Motor: TI SPRACF3 reference parameters, ferrite magnet (same as
mtpa_generic_solver.py -- change parameters there, not here).
"""
import csv
import numpy as np

from mtpa_generic_solver import (
    psi_of_T, mtpa_curve, torque, I_rated, T_ref, alpha_psi_ferrite,
)

# ---- configure this ----
TEMPERATURES_C = [25, 45, 65, 85, 105, 120]   # any list of magnet temps, degC
N_POINTS = 50                                  # Te points per temperature
OUT_PREFIX = "mtpa_lut"
# -------------------------


def build_table(T_degC, n_points=N_POINTS):
    """Returns a dict of arrays: Te (Nm), id (A), iq (A), Is (A) at a
    single magnet temperature, evenly spaced in Te from ~0 to the
    current-limited max at that temperature."""
    psi_f = psi_of_T(T_degC)
    ids, iqs, Te_sweep, Te_max = mtpa_curve(psi_f, n=n_points)
    Is = np.hypot(ids, iqs)
    return {"Te": Te_sweep, "id": ids, "iq": iqs, "Is": Is, "psi_f": psi_f, "Te_max": Te_max}


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        w.writerows(rows)


def main():
    print(f"ferrite magnet, alpha_psi = {alpha_psi_ferrite:+.4f} /degC, T_ref = {T_ref:.0f} C\n")
    print(f"{'T (C)':>7} {'psi_f (Wb)':>12} {'Te_max (Nm)':>12}   file")

    combined_rows = []
    for T_degC in TEMPERATURES_C:
        table = build_table(T_degC)
        fname = f"{OUT_PREFIX}_{T_degC:.0f}C.csv"
        rows = list(zip(table["Te"], table["id"], table["iq"], table["Is"]))
        write_csv(fname, rows, ["Te_Nm", "id_A", "iq_A", "Is_A"])
        print(f"{T_degC:7.0f} {table['psi_f']:12.5f} {table['Te_max']:12.3f}   {fname}")

        for Te, id_, iq, Is in rows:
            combined_rows.append((T_degC, Te, id_, iq, Is))

    combined_fname = f"{OUT_PREFIX}_all_temperatures.csv"
    write_csv(combined_fname, combined_rows, ["T_degC", "Te_Nm", "id_A", "iq_A", "Is_A"])
    print(f"\nsaved {combined_fname}  ({len(combined_rows)} rows total)")


if __name__ == "__main__":
    main()
