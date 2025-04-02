import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
import matplotlib as mpl

# Settings for the plot appearance using LaTeX for font rendering
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = 'Palatino'
# mpl.rcParams['text.usetex'] = 'true'
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{newtxmath}'
mpl.rcParams['font.size'] = 22
#mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set3.colors)
fontsize = 75
hatch_style = ['xx', 'OO', 'o', '///', '\\\\\\', '/', '\\', '||', 'x']

depths = [28, 16, 10]
widths = [10, 8, 4]
for model_idx in range(len(depths)):
    results_txt_path = os.path.join("MIMOConv/plots/results_txt/latency", f"WRN-{depths[model_idx]}-{widths[model_idx]}.txt")
    data = np.loadtxt(results_txt_path)
    MD_Time = data[:, 0]
    COMM_time = data[:, 1]
    ES_time = data[:, 2]

    # Approach names
    xlabels = ["DEC/M", "BF-9/S", "BF-3/S", "DEC/M", "BF-9/S", "BF-3/S", "MCR2/S", "D-MIMO"]

    # Create a figure with a single subplot
    fig, ax1 = plt.subplots(figsize=(20, 15))

    # Bar width for accuracy (left axis) and times (right axis)
    bar_width = 0.15

    time_bars = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.25, 2.75]  # adjusted positions
    ax1.bar(time_bars, MD_Time, width=bar_width, color='C1', hatch=hatch_style[5], edgecolor="black", linewidth=2, alpha=0.8, label='Mobile Device')
    ax1.bar(time_bars, ES_time, width=bar_width, bottom=MD_Time, color='C0', hatch=hatch_style[6], edgecolor="black", linewidth=2, alpha=0.9, label='Edge Server')
    ax1.bar(time_bars, COMM_time, width=bar_width, bottom=MD_Time + ES_time, color='C2', hatch=hatch_style[2], edgecolor="black", linewidth=2, alpha=0.7, label='Comm.')

    # ax1.set_yscale('log')

    # ax1.set_ylim(6, 14000)

    # Set the x-axis ticks and labels
    ax1.set_xticks(time_bars)
    ax1.set_xticklabels(xlabels, fontsize=fontsize-10, rotation=45)
    for label in ax1.get_xticklabels():
        if label.get_text() == "D-MIMO":
            label.set_fontweight('bold')

    # ax1.set_yticks([100, 200, 300], [ r'$100$', r'$200$', r'$300$'], fontsize=fontsize-10)

    ax1.set_xlabel('Approaches', fontsize=fontsize, labelpad=60)
    ax1.set_ylabel('End-to-end\nlatency (ms)', fontsize=fontsize)

    # Add another x-axis label
    ax1.text(0.15, -0.35, 'LoS', fontsize=fontsize - 10, ha='center', va='top', transform=ax1.transAxes)
    ax1.text(0.5, -0.35, 'NLoS', fontsize=fontsize - 10, ha='center', va='top', transform=ax1.transAxes)

    # Increase y-axis tick font size
    ax1.tick_params(axis='both', which='both', labelsize=fontsize-10)

    # Add grid and legend
    ax1.grid(True, which="both", axis="y", linestyle='--', linewidth=2)  # grid for log scale

    # Adding legends
    lines, labels = ax1.get_legend_handles_labels()

    ax1.legend(lines, labels, loc='upper center', ncol=2, fontsize=fontsize-15, framealpha=0.7, bbox_to_anchor=(0.55, 1), bbox_transform=plt.gcf().transFigure, columnspacing=0.5, handlelength=1)

    # Save and show plot
    figs_path = results_txt_path.replace("results_txt", "figs").replace(".txt", ".pdf")
    plt.tight_layout()
    plt.savefig(figs_path, dpi=300, format='pdf')
    plt.show()
