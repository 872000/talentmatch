#!/usr/bin/env python3
"""Generate real output visuals for the README: score heatmap + ranking bar chart.

Run:  python3 scripts/make_charts.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from talentmatch.cli import _all_resumes, _jd_path, _read  # noqa: E402
from talentmatch.engine import match  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

RESUMES = ["backend_java", "backend_node", "frontend_react", "frontend_vue",
           "data_scientist", "data_engineer", "mobile_flutter", "mobile_ios"]
JDS = ["backend_java_jd", "frontend_react_jd", "data_scientist_jd"]
JD_LABELS = ["Backend\n(Java)", "Frontend\n(React)", "Data\nScientist"]

resumes = _all_resumes()
scores = np.zeros((len(RESUMES), len(JDS)))
for i, r in enumerate(RESUMES):
    for j, jd in enumerate(JDS):
        scores[i, j] = match(resumes[r], _read(_jd_path(jd))).score

# --- heatmap ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 6.5))
im = ax.imshow(scores, cmap="YlGn", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(JDS)), JD_LABELS)
ax.set_yticks(range(len(RESUMES)), RESUMES)
ax.set_xlabel("Job description", fontsize=11)
ax.set_ylabel("Candidate resume", fontsize=11)
ax.set_title("TalentMatch — candidate × JD match scores (0–100)", fontsize=13, pad=12)
for i in range(len(RESUMES)):
    for j in range(len(JDS)):
        v = scores[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if v > 55 else "black")
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Match score", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "heatmap.png"), dpi=150)
plt.close(fig)

# --- bar chart: ranking for the React frontend JD --------------------------
jd_name = "frontend_react_jd"
ranked = sorted(
    ((r, match(resumes[r], _read(_jd_path(jd_name))).score) for r in RESUMES),
    key=lambda t: t[1],
    reverse=True,
)
names = [r for r, _ in ranked]
vals = [s for _, s in ranked]
colors = ["#2ca02c" if v >= 60 else "#ff7f0e" if v >= 30 else "#7f7f7f" for v in vals]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(names, vals, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xlim(0, 100)
ax.set_xlabel("Match score (0–100)", fontsize=11)
ax.set_title("Ranking — all candidates vs Frontend (React) JD", fontsize=13, pad=12)
ax.invert_yaxis()
for bar, v in zip(bars, vals):
    ax.text(v + 1, bar.get_y() + bar.get_height() / 2, f"{v:.1f}",
            va="center", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "ranking.png"), dpi=150)
plt.close(fig)

print("Wrote assets/heatmap.png and assets/ranking.png")
print("Score matrix:")
print(np.round(scores, 1))
