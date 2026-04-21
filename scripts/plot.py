#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
ASSETS_DIR = REPO_ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# Load and concatenate all CSVs
csv_files = sorted(RESULTS_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RESULTS_DIR}")

df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

# Convert ns -> µs, drop warmup frames
df["dispatch_us"] = df["dispatch_ns"] / 1000.0
df = df[df["frame"] >= 5]

# Aggregate stats per (particle_count, workgroup_size)
agg = (
    df.groupby(["particle_count", "workgroup_size"])["dispatch_us"]
    .agg(
        median="median",
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
        p5=lambda x: x.quantile(0.05),
        p95=lambda x: x.quantile(0.95),
        std="std",
    )
    .reset_index()
)

datestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ── Plot 1: Latency vs particle count — IQR + p5/p95 shaded bands ─────────────
fig, ax = plt.subplots(figsize=(10, 6))
for wg, g in agg.groupby("workgroup_size"):
    g = g.sort_values("particle_count")
    x = g["particle_count"]
    ax.plot(x, g["median"], marker="o", label=f"wg={wg}")

ax.set_xscale("log", base=2)
pc_ticks = sorted(agg["particle_count"].unique())
ax.set_xticks(pc_ticks)
ax.set_xticklabels([f"{v:,}" for v in pc_ticks], rotation=45, ha="right")
ax.set_xlabel("Particle count")
ax.set_ylabel("Dispatch time (µs)")
ax.set_title("GPU dispatch latency vs particle count")
ax.legend(title="Workgroup size")
ax.grid(True, which="both", linestyle="--", alpha=0.5)
fig.tight_layout()
path1 = ASSETS_DIR / f"{datestamp}_latency_vs_particles.png"
fig.savefig(path1, dpi=150)
print(f"Saved: {path1}")


# ── Plot 2: Latency vs workgroup size — ±1 std dev error bars ─────────────────
fig, ax = plt.subplots(figsize=(10, 6))
agg_no4m = agg[agg["particle_count"] != 4_194_304]
for pc, g in agg_no4m.groupby("particle_count"):
    g = g.sort_values("workgroup_size")
    ax.errorbar(
        g["workgroup_size"], g["median"],
        yerr=g["std"],
        marker="o", capsize=4, label=f"p={pc:,}",
    )

ax.set_xscale("log", base=2)
wg_ticks = sorted(agg_no4m["workgroup_size"].unique())
ax.set_xticks(wg_ticks)
ax.set_xticklabels([str(v) for v in wg_ticks])
ax.set_xlabel("Workgroup size")
ax.set_ylabel("Dispatch time (µs)")
ax.set_title("GPU dispatch latency vs workgroup size (≤2M particles)")
ax.legend(title="Particle count", fontsize=8)
ax.grid(True, which="both", linestyle="--", alpha=0.5)
fig.tight_layout()
path2 = ASSETS_DIR / f"{datestamp}_latency_vs_workgroup.png"
fig.savefig(path2, dpi=150)
print(f"Saved: {path2}")


# ── Plot 3: Frame-by-frame latency, wg=256, 2×2 grid ─────────────────────────
PANEL_PARTICLES = [131072, 262144, 2097152, 4194304]
PANEL_LABELS    = ["131k — fits in L2", "262k — spills out of L2", "2M — fits in Infinity Cache", "4M — spills to GDDR6"]
WG_FOCUS = 256

raw = df[(df["workgroup_size"] == WG_FOCUS) & (df["particle_count"].isin(PANEL_PARTICLES))]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 1].sharey(axes[0, 0])  # top row shares its own scale
fig.suptitle(f"Frame-by-frame dispatch latency  (workgroup size = {WG_FOCUS})", fontsize=13)

for ax, pc, label in zip(axes.flat, PANEL_PARTICLES, PANEL_LABELS):
    subset = raw[raw["particle_count"] == pc].sort_values("frame")
    ax.plot(subset["frame"], subset["dispatch_us"], linewidth=0.8, color="steelblue")
    ax.set_title(f"{pc:,}  —  {label}", fontsize=9)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Dispatch time (µs)")
    ax.grid(True, linestyle="--", alpha=0.5)

axes[1, 1].set_ylim(500, 600)

fig.tight_layout()
path3 = ASSETS_DIR / f"{datestamp}_frame_latency_wg{WG_FOCUS}.png"
fig.savefig(path3, dpi=150)
print(f"Saved: {path3}")


# ── Plot 4: ns per particle vs particle count ─────────────────────────────────
agg4 = agg.copy()
agg4["ns_per_particle"] = (agg4["median"] * 1000) / agg4["particle_count"]

fig, ax = plt.subplots(figsize=(10, 6))
for wg, g in agg4.groupby("workgroup_size"):
    g = g.sort_values("particle_count")
    ax.plot(g["particle_count"], g["ns_per_particle"], marker="o", label=f"wg={wg}")

ax.set_xscale("log", base=2)
pc_ticks = sorted(agg4["particle_count"].unique())
ax.set_xticks(pc_ticks)
ax.set_xticklabels([f"{v:,}" for v in pc_ticks], rotation=45, ha="right")
ax.set_xlabel("Particle count")
ax.set_ylabel("ns per particle")
ax.set_title("Compute throughput: ns per particle vs particle count")
ax.legend(title="Workgroup size")
ax.grid(True, which="both", linestyle="--", alpha=0.5)
fig.tight_layout()
path4 = ASSETS_DIR / f"{datestamp}_ns_per_particle.png"
fig.savefig(path4, dpi=150)
print(f"Saved: {path4}")
