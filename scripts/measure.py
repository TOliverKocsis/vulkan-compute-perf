#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXECUTABLE = REPO_ROOT / "build" / "VulkanComputePerf"
DURATION = 3  # seconds per run

# Threads: 32 .. 1024
thread_counts = [32 * (2 ** i) for i in range(6)]

# Particles count
# 8192 .. 4,194,304
particle_counts = []
p = 8192
while p < 5_000_000:
    particle_counts.append(p)
    p *= 2

total = len(particle_counts) * len(thread_counts)
run = 0

for particles in particle_counts:
    for threads in thread_counts:
        if particles % threads != 0:
            continue
        run += 1
        print(f"[{run}/{total}] particles={particles:>8}  threads={threads:>4}", flush=True)
        result = subprocess.run(
            [
                str(EXECUTABLE),
                "--particle-count", str(particles),
                "--workgroup-size", str(threads),
                "--duration", str(DURATION),
            ],
            cwd=REPO_ROOT / "build",
        )
        if result.returncode != 0:
            print(f"  ERROR: exited with code {result.returncode}", file=sys.stderr)

print("Done.")
