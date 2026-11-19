"""The route-planning problem: map, start, goal, rules and costs.

The robot must drive from a start point to a goal point on CMU's sidewalks:

- only on drivable ground: sidewalk cells at least 30 cm from the edge, no
  stairs, and in the one big connected piece of sidewalk the robot can reach;
- never up or down a slope steeper than MAX_GRADE (20%);
- keeping obstacle_radius meters (center to center) from any obstacle, such
  as pedestrians the robot has seen.

Among all such paths, a cheaper one is better. The cost of driving straight
from point a to point b (horizontal length L meters, height change dz) is

    L * (1 + edge penalty)                         distance; keep off the edge
    + L * STEEP_WEIGHT * (grade - RAMP_GRADE)      only for grades over 8.3%
    + CLIMB_WEIGHT * (meters climbed)
    + DESCENT_WEIGHT * (meters descended)

(the constants are in campus_grid.py). Without elevation only the first line
counts.

Everything is in map coordinates: meters from the map's south-west corner,
x east, y north. Internally the problem is a grid of res-meter cells
(res = 0.5 m), indexed (row, col) with row 0 at the south edge.
"""


import numpy as np

import campus_grid as cg


class PlanningProblem:
    """A start, a goal, the map and the rules. Planners read it; they never change it.

    Attributes:
        res: cell size in meters (0.5)
        drivable: (H, W) bool, cells the robot may drive on (ignoring obstacles)
        blocked: (H, W) bool, cells too close to an obstacle (all False if none)
        z: (H, W) ground elevation of each cell in meters (smoothed; zeros if none)
        edge: (H, W) extra cost factor near the sidewalk edge (0 to CENTER_WEIGHT)
        start, goal: (row, col) cells; start_xy, goal_xy: their centers in meters
        requested_start_xy, requested_goal_xy: the points asked for (doors); start
            and goal are the nearest cells the robot can actually drive to
        uses_elevation: whether slopes and climbing count
    """

    def __init__(self, grid, start_xy, goal_xy, elevation=None, elev_res=1.0,
                 obstacles=None, obstacle_radius=0.8, step_m=0.5,
                 climb_weight=cg.CLIMB_WEIGHT, descent_weight=cg.DESCENT_WEIGHT,
                 max_grade=cg.MAX_GRADE, smooth_m=cg.SMOOTH_M):
        prep = cg._prepare(grid, step_m, elevation, elev_res, smooth_m, max_grade)
        self.grid = grid
        self.res = prep['res_c']
        self.drivable = prep['main']
        self.z = prep['z']
        self.edge = prep['edge']
        self.uses_elevation = elevation is not None
        self.climb_weight = climb_weight if self.uses_elevation else 0.0
        self.descent_weight = descent_weight if self.uses_elevation else 0.0
        self.max_grade = max_grade
        self.obstacles = np.empty((0, 2)) if obstacles is None else np.atleast_2d(obstacles)
        self.obstacle_radius = obstacle_radius
        self.requested_start_xy = np.asarray(start_xy, dtype=float)
        self.requested_goal_xy = np.asarray(goal_xy, dtype=float)

        self.start = self._nearest_drivable(start_xy)
        self.goal = self._nearest_drivable(goal_xy)
        self.blocked = self._block(self.obstacles, obstacle_radius)
        self.blocked[self.start] = self.blocked[self.goal] = False   # never block the ends
        self.start_xy = self.xy_of(self.start)
        self.goal_xy = self.xy_of(self.goal)

    # --- Building a problem from building names --------------------------------

    @classmethod
    def between_buildings(cls, pickup, dropoff, floor=None, from_floor=None,
                          png_path=cg.DEFAULT_MAP, obstacles=None, obstacle_radius=0.8):
        """The delivery problem between two buildings (doors chosen as in campus_grid).

        The chosen doors are kept as problem.pickup and problem.dropoff.
        """
        grid, _ = cg.load_campus_grid(png_path)
        entrances = cg.load_entrances(png_path)
        elev, elev_res = cg.load_elevation(png_path)
        chosen = cg.plan_delivery(grid, entrances, elev, elev_res, pickup, dropoff,
                                  floor=floor, from_floor=from_floor, verbose=False)
        problem = cls(grid, chosen['start']['delivery_xy'], chosen['goal']['delivery_xy'],
                      elevation=elev, elev_res=elev_res, obstacles=obstacles,
                      obstacle_radius=obstacle_radius)
        problem.pickup, problem.dropoff = chosen['start'], chosen['goal']
        problem.elevation, problem.elev_res = elev, elev_res
        return problem

    # --- Cells and points ------------------------------------------------------

    def cell_of(self, xy):
        """The (row, col) cell containing point xy (meters)."""
        return int(xy[1] // self.res), int(xy[0] // self.res)

    def xy_of(self, cell):
        """Center of a cell, in meters."""
        return np.array([(cell[1] + 0.5) * self.res, (cell[0] + 0.5) * self.res])

    def in_bounds(self, cell):
        return 0 <= cell[0] < self.drivable.shape[0] and 0 <= cell[1] < self.drivable.shape[1]

    def is_free(self, cell):
        """True if the robot may be in this cell (drivable and away from obstacles)."""
        return self.in_bounds(cell) and self.drivable[cell] and not self.blocked[cell]

    def random_free_xy(self, rng):
        """A random point the robot may be at, anywhere on the map (for sampling)."""
        if not hasattr(self, '_free_cells'):
            self._free_cells = np.argwhere(self.drivable & ~self.blocked)
        r, c = self._free_cells[rng.integers(len(self._free_cells))]
        return self.xy_of((r, c)) + rng.uniform(-0.5, 0.5, 2) * self.res

    # --- Costs ---------------------------------------------------------------

    def move_cost(self, a, b):
        """Cost of driving straight between the centers of cells a and b
        (usually neighbors), or inf if that move is not allowed."""
        if not (self.is_free(a) and self.is_free(b)):
            return np.inf
        L = np.hypot(a[0] - b[0], a[1] - b[1]) * self.res
        if L == 0:
            return 0.0
        dz = self.z[b] - self.z[a]
        grade = abs(dz) / L
        if grade > self.max_grade:
            return np.inf
        steep = L * cg.STEEP_WEIGHT * max(0.0, grade - cg.RAMP_GRADE) if self.uses_elevation else 0.0
        return (L * (1.0 + self.edge[b]) + steep
                + self.climb_weight * max(dz, 0.0) + self.descent_weight * max(-dz, 0.0))

    def segment_cost(self, a_xy, b_xy):
        """Cost of driving in a straight line from point a to point b (meters),
        or inf if the line crosses anything not allowed (off the sidewalk, stairs,
        too steep, too close to an obstacle).

        The line is followed cell by cell, adding move_cost between consecutive
        cells, so a path built from straight segments costs the same as the
        cell-by-cell A* path along it. Use it to check that a straight segment is drivable.
        """
        a_xy, b_xy = np.asarray(a_xy, float), np.asarray(b_xy, float)
        n = max(2, int(np.ceil(np.linalg.norm(b_xy - a_xy) / (self.res / 4))) + 1)
        cells = []
        for p in np.linspace(a_xy, b_xy, n):
            c = self.cell_of(p)
            if not cells or c != cells[-1]:
                # A line through a cell corner touches a side cell for an
                # instant; drop it and take the diagonal move, as A* does.
                if (len(cells) >= 2 and abs(c[0] - cells[-2][0]) == 1
                        and abs(c[1] - cells[-2][1]) == 1):
                    cells.pop()
                cells.append(c)
        if len(cells) == 1:
            return 0.0 if self.is_free(cells[0]) else np.inf
        total = 0.0
        for c1, c2 in zip(cells[:-1], cells[1:]):
            total += self.move_cost(c1, c2)
            if total == np.inf:
                return np.inf
        return total

    def heuristic(self, cell):
        """A*'s estimate of the cost from cell to the goal; never more than the
        real cost: octile distance + CLIMB_WEIGHT * net height still to climb."""
        dr, dc = abs(self.goal[0] - cell[0]), abs(self.goal[1] - cell[1])
        octile = (max(dr, dc) + (np.sqrt(2.0) - 1.0) * min(dr, dc)) * self.res
        return octile + self.climb_weight * max(0.0, self.z[self.goal] - self.z[cell])

    # --- Scoring a path ----------------------------------------------------

    def path_cost(self, path):
        """Total cost of a path of points (sum of segment_cost); inf if invalid."""
        path = np.asarray(path, dtype=float)
        return float(sum(self.segment_cost(a, b) for a, b in zip(path[:-1], path[1:])))

    def evaluate(self, path):
        """Numbers to compare planners: validity, length, cost, climbing, steepest grade."""
        path = np.asarray(path, dtype=float)
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cost = self.path_cost(path)
        z = np.array([self.z[self.cell_of(p)] if self.in_bounds(self.cell_of(p)) else np.nan
                      for p in path])
        dz = np.diff(z)
        grades = np.abs(dz) / np.maximum(seg, 1e-9)
        return {
            'valid': bool(np.isfinite(cost)),
            'length_m': float(seg.sum()),
            'cost': cost,
            'climb_m': float(dz[dz > 0].sum()),
            'descent_m': float(-dz[dz < 0].sum()),
            'max_grade': float(grades[seg > 0.2].max()) if (seg > 0.2).any() else 0.0,
            'points': len(path),
            'start_error_m': float(np.linalg.norm(path[0] - self.start_xy)),
            'goal_error_m': float(np.linalg.norm(path[-1] - self.goal_xy)),
        }

    # --- Internals ---------------------------------------------------------

    def _nearest_drivable(self, xy):
        rr, cc = np.nonzero(self.drivable)
        d = ((cc + 0.5) * self.res - xy[0]) ** 2 + ((rr + 0.5) * self.res - xy[1]) ** 2
        i = int(np.argmin(d))
        return int(rr[i]), int(cc[i])

    def _block(self, obstacles, radius):
        blocked = np.zeros(self.drivable.shape, dtype=bool)
        H, W = blocked.shape
        reach = int(np.ceil(radius / self.res))
        for ox, oy in obstacles:
            c0, r0 = int(ox / self.res), int(oy / self.res)
            for r in range(max(r0 - reach, 0), min(r0 + reach + 1, H)):
                for c in range(max(c0 - reach, 0), min(c0 + reach + 1, W)):
                    if np.hypot((c + 0.5) * self.res - ox, (r + 0.5) * self.res - oy) < radius:
                        blocked[r, c] = True
        return blocked


