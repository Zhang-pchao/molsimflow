"""Plot auditable descriptive CH3 acid/base ion-water diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from molsimflow.postprocess.nanobubble_ion_mediation_review import read_tsv


def render(review: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    flow = read_tsv(review / "LOCAL-FLOW-AND-TOPOLOGY.tsv")
    ions = read_tsv(review / "FIXED-GRID-ION-WATER.tsv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    labels = ["HCl-X", "HCl-Y", "NaOH-X", "NaOH-Y"]
    order = ["hcl_63pairs__waterO_f8e-6_x", "hcl_63pairs__waterO_f8e-6_y",
             "naoh_ph13p4__waterO_f8e-6_x", "naoh_ph13p4__waterO_f8e-6_y"]
    index = {row["label"]: row for row in flow}
    x = np.arange(4)
    width = 0.24
    for offset, field, name, color in (
        (-width, "bubble_velocity_mps", "bubble", "#243b53"),
        (0, "front_water_velocity_mps", "front water", "#1f9d8a"),
        (width, "rear_water_velocity_mps", "rear water", "#d58337"),
    ):
        axes[0].bar(x + offset, [float(index[item][field]) for item in order],
                    width, label=name, color=color)
    axes[0].axhline(0, color="#333", linewidth=0.7)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Mean velocity (m s$^{-1}$)")
    axes[0].set_title("A  Local flow is heterogeneous")
    axes[0].legend(frameon=False, fontsize=8)
    y_labels = ["HCl / H$_3$O$^+$", "HCl / Cl$^-$", "NaOH / OH$^-$", "NaOH / Na$^+$"]
    y_cases = [("hcl_63pairs__waterO_f8e-6_y", "H3O_plus_candidate"),
               ("hcl_63pairs__waterO_f8e-6_y", "Cl_minus"),
               ("naoh_ph13p4__waterO_f8e-6_y", "OH_minus_candidate"),
               ("naoh_ph13p4__waterO_f8e-6_y", "Na_plus")]
    lookup = {(row["label"], row["species"], row["region"]): row for row in ions}
    for region, offset, color in (("front", -0.17, "#1f9d8a"), ("rear", 0.17, "#d58337")):
        ratios = []
        for case, sp in y_cases:
            near = float(lookup[(case, sp, region)]["ion_density_A3_inv"])
            far = float(lookup[(case, sp, "far_wall")]["ion_density_A3_inv"])
            ratios.append(near / far if far > 0 else np.nan)
        axes[1].bar(np.arange(4) + offset, ratios, 0.32, color=color, label=region)
    axes[1].axhline(1, color="#333", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(range(4), y_labels, rotation=30, ha="right")
    axes[1].set_ylabel("Ion density / far-wall density")
    axes[1].set_title("B  Liquid-accessible edge")
    axes[1].legend(frameon=False, fontsize=8)
    for region, offset, color in (("front", -0.17, "#1f9d8a"), ("rear", 0.17, "#d58337")):
        effect = [float(lookup[(case, sp, region)]["exposed_minus_unexposed_turnover"])
                  for case, sp in y_cases]
        axes[2].bar(np.arange(4) + offset, effect, 0.32, color=color, label=region)
    axes[2].axhline(0, color="#333", linewidth=0.8)
    axes[2].set_xticks(range(4), y_labels, rotation=30, ha="right")
    axes[2].set_ylabel("Exposed minus unexposed turnover")
    axes[2].set_title("C  Sampled oxygen renewal")
    fig.suptitle("CH$_3$ nanobubble: 0.2–4 ns, 10 ps observers; single trajectory per condition", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.review, args.output)
