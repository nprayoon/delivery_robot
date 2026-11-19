"""Checks that A* returns valid, consistent paths on the delivery problem.

From the delivery_robot folder:

    python planners/test_planners.py

A valid path starts at problem.start_xy, ends at problem.goal_xy, and every
straight segment between consecutive points is allowed (finite
problem.segment_cost): on the sidewalk, no stairs, no slope over the limit,
and clear of obstacles.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import campus_grid as cg                       # noqa: E402
from planners.astar import astar               # noqa: E402
from planners.problem import PlanningProblem   # noqa: E402

DELIVERIES = [('Cohon University Center', 'Doherty Hall'),
              ('Hunt Library', 'Wean Hall'),
              ('Scaife Hall', 'Purnell Center for the Arts')]


def check_path(problem, path):
    """Assert that path is a valid solution of problem; return its evaluation."""
    path = np.asarray(path, dtype=float)
    assert path.ndim == 2 and path.shape[1] == 2, 'path must be an (N, 2) array'
    assert np.linalg.norm(path[0] - problem.start_xy) < 0.5, 'does not start at start'
    assert np.linalg.norm(path[-1] - problem.goal_xy) < 0.5, 'does not end at goal'
    for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
        assert np.isfinite(problem.segment_cost(a, b)), \
            f'segment {i} {a.round(1)} -> {b.round(1)} is not drivable'
    return problem.evaluate(path)


def main():
    grid, _ = cg.load_campus_grid()
    elev, er = cg.load_elevation()
    for pickup, dropoff in DELIVERIES:
        problem = PlanningProblem.between_buildings(pickup, dropoff)
        t0 = time.perf_counter()
        path = astar(problem)
        dt = time.perf_counter() - t0
        e = check_path(problem, path)
        # the cost A* minimized equals the problem's own cost of the path
        assert abs(problem.path_cost(path) - e['cost']) < 1e-6
        print(f"{pickup} -> {dropoff}: {e['length_m']:.0f} m, cost {e['cost']:.0f}, "
              f"steepest {e['max_grade'] * 100:.1f}% ({dt:.1f} s)")

    # Obstacles: a person standing on the route must be avoided
    problem = PlanningProblem.between_buildings(*DELIVERIES[0])
    route = astar(problem)
    blocker = route[len(route) // 2]
    problem = PlanningProblem(grid, problem.requested_start_xy, problem.requested_goal_xy,
                              elevation=elev, elev_res=er, obstacles=[blocker],
                              obstacle_radius=0.9)
    path = astar(problem)
    check_path(problem, path)
    dense = np.vstack([np.linspace(a, b, 20) for a, b in zip(path[:-1], path[1:])])
    gap = np.min(np.linalg.norm(dense - blocker, axis=1))
    assert gap >= 0.9 - problem.res, f'passes {gap:.2f} m from the person'
    print(f'With a person standing on the route: avoided ({gap:.2f} m away)')
    print('All checks passed')


if __name__ == '__main__':
    main()
