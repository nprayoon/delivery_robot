"""Checks for the pedestrian simulation.

Usage:
    python test_simulate.py
"""

import numpy as np

import campus_grid as cg
import simulate
from pedestrians import PED_RADIUS, Crowd


def main():
    gt, _ = cg.load_campus_grid()
    entrances = cg.load_entrances()
    elev, er = cg.load_elevation()
    plan = cg.plan_delivery(gt, entrances, elev, er, 'Cohon University Center',
                            'Doherty Hall', floor='1', verbose=False)
    route, goal = plan['route'], np.array(plan['goal']['delivery_xy'])

    # Standing pedestrians are placed on walkable ground near the route and stay put
    crowd = Crowd(gt, route, 30, seed=1)
    start = crowd.pos.copy()
    walk = crowd.grid
    assert all(cg.is_free(walk, x, y) for x, y in start), 'a pedestrian is off the sidewalk'
    d = np.min(np.linalg.norm(start[:, None] - route[None], axis=2), axis=1)
    assert d.max() < 25.0, 'a pedestrian is far from the route'
    crowd.step(1.0)
    assert np.allclose(crowd.pos, start), 'standing pedestrians moved'
    print(f'30 standing pedestrians placed on sidewalks, within {d.max():.1f} m of the route')

    # The robot plans around a pedestrian standing right on its route
    blocker = route[len(route) // 2]
    detour = cg.plan_route(gt, route[0], goal, elevation=elev, elev_res=er,
                           obstacles=[blocker], obstacle_radius=simulate.PASS_CLEARANCE)
    gap = np.min(np.linalg.norm(detour - blocker, axis=1))
    assert gap >= simulate.PASS_CLEARANCE - 0.36, f'detour passes {gap:.2f} m from the person'
    print(f'Detour keeps {gap:.2f} m (center to center) from a person on the route')

    # A full run with standing pedestrians: arrives, stays on free cells, never touches anyone
    result = simulate.run(gt, elev, er, route, goal, Crowd(gt, route, 30, seed=1))
    assert result['arrived'], 'the robot did not arrive'
    assert result['contacts'] == 0, f"{result['contacts']} contacts with standing people"
    assert all(cg.is_free(gt, x, y) for x, y in result['trail'][::10]), 'robot left the sidewalk'
    assert result['min_gap_m'] > 0, 'robot touched a person'
    print(f"Delivered in {result['time_s']:.0f} s with {result['replans']} replans; "
          f"closest approach {result['min_gap_m']:.2f} m (robot radius {simulate.ROBOT_RADIUS} m, "
          f"person radius {PED_RADIUS} m)")
    print('All checks passed')


if __name__ == '__main__':
    main()
