import matplotlib.pyplot as plt
plt.style.use('dark_background')
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from typing import List, Dict, Tuple
import numpy as np
from utils import Note, Scale

# Define chromatic note circle
def get_coords(note_labels: List[str], radius: float = 1.0) -> Dict[str, tuple]:
    angles = np.linspace(0, 2 * np.pi, len(note_labels), endpoint=False)
    angle_step = - (2 * np.pi / 12)
    phase = 0 * angle_step
    coords = {note: (radius * np.cos(a + phase), radius * np.sin(a + phase))
              for note, a in zip(note_labels, angles)}
    return coords

def make_outward_arc(
    start: Tuple[float, float],
    end: Tuple[float, float],
    outward_scale: float = 1.3,
    color: str = 'white',
    linewidth: float = 2.0,
    alpha: float = 0.7
) -> PathPatch:
    """
    Returns a PathPatch representing a quadratic Bezier arc from start to end,
    with a control point pushed outward from the center of the unit circle.
    """
    # Midpoint
    mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2

    # Direction from origin to midpoint
    norm = (mx**2 + my**2)**0.5
    if norm == 0:
        ctrl = (0, 0)
    else:
        ctrl = (mx / norm * outward_scale, my / norm * outward_scale)

    # Bezier curve
    verts = [start, ctrl, end]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    path = Path(verts, codes)
    return PathPatch(path, edgecolor=color, facecolor='none', linewidth=linewidth, alpha=alpha)

def plot_tuning(tuning: List[Note], coords: Dict[str, tuple], ax: plt.Axes, color: str = 'white') -> None:
    tuning_coords = [coords[str(n)] for n in tuning]
    tuning_coords.append(tuning_coords[0])  # close the loop

    for i in range(len(tuning_coords) - 1):
        start = tuning_coords[i]
        end = tuning_coords[i + 1]
        patch = make_outward_arc(start, end, outward_scale=4.0, color=color)
        ax.add_patch(patch)
    # Draw tuning nodes
    for (x, y) in tuning_coords[:-1]:
        ax.plot(x, y, 'o', color=color, markersize=6, alpha=0.6)


# Plot a scale as a dashed loop
def plot_scale(scale: Scale, coords: Dict[str, tuple], ax: plt.Axes, linestyle: str = '--', color: str = 'cyan') -> None:
    scale_coords = [coords[str(n)] for n in scale.notes]
    for i in range(len(scale_coords) - 1):
        x_vals = [scale_coords[i][0], scale_coords[i+1][0]]
        y_vals = [scale_coords[i][1], scale_coords[i+1][1]]
        ax.plot(x_vals, y_vals, linestyle=linestyle, color=color, linewidth=1.5)

    for (x, y) in scale_coords:
        ax.plot(x, y, 'o', color=color, markersize=5, alpha=0.8)

# Main plotting function
def plot_structure(tuning: List[Note], scales: List[Scale]) -> None:
    notes = [
        'C', 'C#', 'D', 'D#', 'E', 'F', 
        'F#', 'G', 'G#', 'A', 'A#', 'B'
    ]
    notes = notes[::-1]
    coords = get_coords(notes)

    fig, ax = plt.subplots(figsize=(6, 6))
    for note, (x, y) in coords.items():
        ax.text(x * 1.1, y * 1.1, note, ha='center', va='center', fontsize=10)
        ax.plot(x, y, 'ko', markersize=4)

    plot_tuning(tuning, coords, ax)
    scale_colors_cmap = plt.get_cmap('viridis', len(scales))
    for idx, scale in enumerate(scales):
        plot_scale(scale, coords, ax, linestyle='--', color=scale_colors_cmap(idx))

    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(
        'Tuning and Scale Visualization'
        '\nTuning: ' + ', '.join(str(n) for n in tuning)
    )
    plt.show()

if __name__ == "__main__":
    D = Note('D')
    tuning = [D + i * 5 for i in range(5)]
    # scale1 = Scale(D, [2, 2, 2, 2, 2, 2, 2])
    # scale2 = Scale(D + 3, [2, 2, 2, 2, 2, 2, 2])


    scales = [
                Scale(D, [2, 2, 2, 2, 2, 2, 2])

                # Scale(D, [2, 1, 3, 2, 3, 1, 2]),
                # Scale(D, [3, 2, 1, 1, 2, 1, 2, 2]),

                # Scale(D, [2, 1, 3, 2, 2, 2, 2]),
                # Scale(D, [2, 2, 2, 2, 3, 1, 2]),
        ]

    plot_structure(tuning, scales)
