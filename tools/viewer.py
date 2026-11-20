"""Replays a trajectory CSV written by Env::set_trajectory_log (simulator v1).

Usage:
  python3 viewer.py logs/traj_HandCoded.csv                 # watch the first episode in the file
  python3 viewer.py logs/traj_HandCoded.csv --save hc.gif   # save an animation instead

Blue = keepers, red = takers, black = ball. The dashed ring marks the keeper that
had the ball on that step (its kickable radius).
"""
import argparse
import csv

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle


def load(path, episode):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if episode is None:
                episode = int(r["episode"])
            if int(r["episode"]) == episode:
                rows.append({k: float(v) for k, v in r.items()})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--field", type=float, default=20.0)
    p.add_argument("--radius", type=float, default=1.0, help="kickable radius in metres")
    p.add_argument("--step", type=float, default=0.1, help="seconds per simulation step")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--save", default=None, help="write a .gif instead of opening a window")
    a = p.parse_args()

    rows = load(a.csv, a.episode)
    if not rows:
        raise SystemExit("no rows for that episode")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(Rectangle((0, 0), a.field, a.field, fill=False, lw=2))
    ax.set_xlim(-1, a.field + 1)
    ax.set_ylim(-1, a.field + 1)
    ax.set_aspect("equal")
    keepers = [ax.plot([], [], "o", color="tab:blue", ms=12)[0] for _ in range(3)]
    labels = [ax.text(0, 0, str(k), color="white", ha="center", va="center", fontsize=8)
              for k in range(3)]
    takers = [ax.plot([], [], "o", color="tab:red", ms=12)[0] for _ in range(2)]
    ball, = ax.plot([], [], "o", color="black", ms=6)
    ring = Circle((0, 0), a.radius, fill=False, ls="--", color="tab:blue")
    ax.add_patch(ring)
    title = ax.set_title("")

    def update(i):
        r = rows[i]
        for k in range(3):
            keepers[k].set_data([r[f"k{k}x"]], [r[f"k{k}y"]])
            labels[k].set_position((r[f"k{k}x"], r[f"k{k}y"]))
        for d in range(2):
            takers[d].set_data([r[f"d{d}x"]], [r[f"d{d}y"]])
        ball.set_data([r["bx"]], [r["by"]])
        h = int(r["holder"])
        ring.set_visible(h >= 0)
        if h >= 0:
            ring.center = (r[f"k{h}x"], r[f"k{h}y"])
        title.set_text(f"t = {r['t'] * a.step:.1f} s    steal count = {int(r['steal'])}")
        return keepers + labels + takers + [ball, ring, title]

    anim = FuncAnimation(fig, update, frames=len(rows), interval=1000 / a.fps, blit=False)
    if a.save:
        anim.save(a.save, writer=PillowWriter(fps=a.fps))
        print(f"saved {a.save} ({len(rows)} frames)")
    else:
        plt.show()


if __name__ == "__main__":
    main()
