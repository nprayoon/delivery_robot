"""A* search for the delivery robot.

A* explores the grid of cells outward from the start, always expanding the
cell with the smallest f = g + h:

    g = cost of the best path found so far from the start to the cell
    h = problem.heuristic(cell), an estimate of the cost left to the goal

Because h never overestimates the real remaining cost (it is "admissible"),
the first time the goal is taken off the queue its path is the cheapest one
on the grid. Each cell connects to its 8 neighbors; a move's cost is
problem.move_cost (distance, slope and climbing; inf if not allowed).
"""

import heapq

import numpy as np


def astar(problem):
    """Cheapest path from problem.start to problem.goal on the problem's grid.

    Returns:
        path: (N, 2) array of (x, y) points in meters, one per cell (0.5 m apart,
            0.71 m on diagonals), from problem.start_xy to problem.goal_xy

    Raises:
        RuntimeError: no path exists (for example, obstacles close every way through)
    """
    start, goal = problem.start, problem.goal
    H, W = problem.drivable.shape
    # Local copies, so the inner loop below stays fast
    free = problem.drivable & ~problem.blocked
    z, edge, res = problem.z, problem.edge, problem.res
    climb_w, descent_w, max_grade = problem.climb_weight, problem.descent_weight, problem.max_grade
    from campus_grid import RAMP_GRADE, STEEP_WEIGHT
    steep_w = STEEP_WEIGHT if problem.uses_elevation else 0.0
    sqrt2 = np.sqrt(2.0)

    cost = {start: 0.0}          # g: best known cost from the start
    came = {start: None}         # parent of each cell on its best path
    queue = [(problem.heuristic(start), start)]
    closed = set()               # cells already expanded
    while queue:
        _, cur = heapq.heappop(queue)
        if cur == goal:
            break
        if cur in closed:
            continue
        closed.add(cur)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nb = (cur[0] + dr, cur[1] + dc)
                if not ((dr or dc) and 0 <= nb[0] < H and 0 <= nb[1] < W and free[nb]):
                    continue
                # Same formula as problem.move_cost(cur, nb), written out for speed
                L = (sqrt2 if dr and dc else 1.0) * res
                dz = z[nb] - z[cur]
                grade = abs(dz) / L
                if grade > max_grade:
                    continue
                step = (L * (1.0 + edge[nb]) + L * steep_w * max(0.0, grade - RAMP_GRADE)
                        + climb_w * max(dz, 0.0) + descent_w * max(-dz, 0.0))
                g = cost[cur] + step
                if g < cost.get(nb, np.inf):
                    cost[nb] = g
                    came[nb] = cur
                    heapq.heappush(queue, (g + problem.heuristic(nb), nb))
    if goal not in came:
        raise RuntimeError('No sidewalk route between start and goal')

    cells = [goal]
    while came[cells[-1]] is not None:
        cells.append(came[cells[-1]])
    return np.array([problem.xy_of(c) for c in reversed(cells)])
