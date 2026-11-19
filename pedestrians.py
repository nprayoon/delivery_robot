"""Simulated pedestrians on CMU's sidewalks.

Pedestrians are placed on sidewalks near the robot's route. With speed 0 (the
default for now) they stand still. With a speed they walk trips between random
sidewalk points along A* routes and start a new trip when they arrive.

Unlike the robot, pedestrians may use stairs, and they do not react to the
robot (scripted walkers; a social-force model is a later stretch goal).
"""

import cv2
import numpy as np

import campus_grid as cg

PED_RADIUS = 0.25       # meters (a person seen from above)


def walking_grid(grid, png_path=cg.DEFAULT_MAP):
    """A copy of the campus grid where stairs are walkable too (for people)."""
    walk = cg.empty_like(grid)
    walk.data = np.asarray(grid.data).copy()
    stairs = cg.load_layer(png_path, 'stairs')
    if stairs is not None:
        walk.data[stairs.ravel()] = grid.min_clamp
    return walk


class Crowd:
    """A group of pedestrians near a route.

    Attributes:
        pos: (n, 2) positions in meters (map coordinates)
        vel: (n, 2) velocities in m/s
        speed: (n,) preferred walking speed of each pedestrian, m/s (0 = standing)
    """

    def __init__(self, grid, route, n, speed=(0.0, 0.0), spread_m=20.0,
                 keep_clear_m=4.0, trip_m=(30.0, 150.0), seed=0, png_path=cg.DEFAULT_MAP):
        """Place n pedestrians on sidewalks within spread_m of the route.

        Args:
            grid: the robot's campus Grid2D
            route: (N, 2) robot route; pedestrians spawn near it so they meet
            n: number of pedestrians
            speed: (low, high) walking speed range in m/s; (0, 0) = all standing
            keep_clear_m: no pedestrian starts this close to the route's ends
            trip_m: (shortest, longest) straight-line length of a walking trip
            seed: random seed, so a run can be repeated exactly
        """
        self.rng = np.random.default_rng(seed)
        self.grid = walking_grid(grid, png_path)
        self.trip_m = trip_m

        # Walkable coarse cells near the route (the same cells plan_route uses)
        prep = cg._prepare(self.grid, 0.5, None, 1.0, cg.SMOOTH_M, cg.MAX_GRADE)
        main, res_c = prep['main'], prep['res_c']
        near = np.zeros(main.shape, dtype=np.uint8)
        cells = (np.asarray(route) / res_c).astype(int)
        cv2.polylines(near, [cells.reshape(-1, 1, 2)], False, 1, 1)
        k = int(spread_m / res_c)
        near = cv2.dilate(near, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2))
        rr, cc = np.nonzero(main & (near > 0))
        xy = np.c_[cc * res_c + res_c / 2, rr * res_c + res_c / 2]
        ends = np.asarray(route)[[0, -1]]
        far_from_ends = np.min(np.linalg.norm(xy[:, None] - ends[None], axis=2), axis=1)
        self.candidates = xy[far_from_ends > keep_clear_m]
        if len(self.candidates) == 0:
            raise RuntimeError('No sidewalk near the route to place pedestrians on')

        pick = self.rng.choice(len(self.candidates), size=n, replace=len(self.candidates) < n)
        self.pos = self.candidates[pick] + self.rng.uniform(-0.15, 0.15, (n, 2))
        self.vel = np.zeros((n, 2))
        self.speed = self.rng.uniform(speed[0], speed[1], n)
        self.paths = [None] * n          # current trip of each walking pedestrian
        self.along = np.zeros(n)         # meters walked along the current trip

    def __len__(self):
        return len(self.pos)

    def _new_trip(self, i):
        """Send pedestrian i toward a random sidewalk point trip_m away."""
        d = np.linalg.norm(self.candidates - self.pos[i], axis=1)
        options = np.flatnonzero((d >= self.trip_m[0]) & (d <= self.trip_m[1]))
        if len(options) == 0:
            options = np.arange(len(self.candidates))
        goal = self.candidates[self.rng.choice(options)]
        path = cg.plan_route(self.grid, self.pos[i], goal)
        self.paths[i] = np.vstack([self.pos[i], path])
        self.along[i] = 0.0

    def step(self, dt):
        """Move every walking pedestrian dt seconds along its trip."""
        for i in np.flatnonzero(self.speed > 0):
            if self.paths[i] is None:
                self._new_trip(i)
            path = self.paths[i]
            seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            self.along[i] += self.speed[i] * dt
            if self.along[i] >= seg[-1]:          # arrived: start another trip
                self.pos[i] = path[-1]
                self.vel[i] = 0.0
                self._new_trip(i)
                continue
            new = np.array([np.interp(self.along[i], seg, path[:, 0]),
                            np.interp(self.along[i], seg, path[:, 1])])
            self.vel[i] = (new - self.pos[i]) / dt
            self.pos[i] = new
