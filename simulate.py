"""Delivery simulation: the robot drives its route among pedestrians.

Every DT seconds the pedestrians move (or stand still, with speed 0), the
robot looks for pedestrians within its sensor range, and the baseline
controller decides what to do:

- If a pedestrian it can see is on its path ahead, it replans the A* route
  around them (pedestrians become temporary obstacles).
- If someone is right in front of it anyway, it stops and waits.
- Otherwise it drives at up to V_MAX, with acceleration limited to A_MAX.

Metrics are printed and the run is saved as an animation that follows the
robot (docs/simulation_follow.gif), one of the whole campus map
(docs/simulation_map.gif), and an overview picture (docs/simulation.png).

    python simulate.py                       (asks for buildings, floors, crowd size)
    python simulate.py --from "Cohon University Center" --to "Doherty Hall" --pedestrians 30
    python simulate.py --from "Hunt Library" --to "Wean Hall" --ped-speed 1.2
"""

import argparse
import time
from pathlib import Path

import numpy as np

import campus_grid as cg
from pedestrians import PED_RADIUS, Crowd

HERE = Path(__file__).resolve().parent

DT = 0.1                 # simulation step (s)
V_MAX = 1.2              # robot top speed (m/s), a brisk walk
A_MAX = 0.5              # robot acceleration / braking limit (m/s^2)
ROBOT_RADIUS = 0.35      # m (0.7 m across, about a sidewalk delivery robot)
SENSOR_RANGE = 8.0       # robot sees pedestrians this far away (m)
PASS_CLEARANCE = ROBOT_RADIUS + PED_RADIUS + 0.3   # center-to-center when passing (m)
STOP_DISTANCE = ROBOT_RADIUS + PED_RADIUS + 0.4    # stop if someone this close ahead (m)
REPLAN_EVERY = 1.0       # replan at most this often (s)
NEAR_MISS = 0.2          # gap between robot and person counted as a near-miss (m)


def _arc(path):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]


def _at(path, arc, s):
    return np.array([np.interp(s, arc, path[:, 0]), np.interp(s, arc, path[:, 1])])


def run(gt, elev, elev_res, route, goal_xy, crowd, max_time=None, record_every=2.0):
    """Simulate one delivery.

    Returns a dict with the metrics and 'frames': snapshots every
    record_every seconds (robot position, heading, speed, path ahead, status,
    pedestrian positions and which ones the robot sees).
    """
    path = np.asarray(route, dtype=float)
    arc = _arc(path)
    s, v, t = 0.0, 0.0, 0.0
    max_time = max_time or 3.0 * arc[-1] / V_MAX + 120.0
    last_replan = -1e9
    m = {'stopped_s': 0.0, 'replans': 0, 'replan_failures': 0, 'min_gap_m': np.inf,
         'near_misses': 0, 'contacts': 0, 'driven_m': 0.0}
    close_now = np.zeros(len(crowd), dtype=bool)     # within NEAR_MISS last step
    touching_now = np.zeros(len(crowd), dtype=bool)  # overlapping the robot last step
    frames, trail = [], [path[0].copy()]
    next_frame = 0.0
    status = 'driving'

    while t < max_time:
        crowd.step(DT)
        pos = _at(path, arc, s)
        ahead_end = min(s + SENSOR_RANGE, arc[-1])
        heading_pt = _at(path, arc, min(s + 0.5, arc[-1]))
        heading = heading_pt - pos
        heading = heading / (np.linalg.norm(heading) + 1e-9)

        rel = crowd.pos - pos
        dist = np.linalg.norm(rel, axis=1)
        seen = dist < SENSOR_RANGE

        # Is a seen pedestrian on the path ahead? Then plan around them.
        ahead = path[(arc >= s) & (arc <= ahead_end)]
        conflict = False
        if seen.any() and len(ahead):
            d_path = np.min(np.linalg.norm(ahead[:, None] - crowd.pos[seen][None], axis=2), axis=0)
            conflict = bool((d_path < PASS_CLEARANCE).any())
        if conflict and t - last_replan >= REPLAN_EVERY:
            last_replan = t
            try:
                new = cg.plan_route(gt, pos, goal_xy, elevation=elev, elev_res=elev_res,
                                    obstacles=crowd.pos[seen], obstacle_radius=PASS_CLEARANCE)
                path = np.vstack([pos, new])
                arc = _arc(path)
                s = 0.0
                m['replans'] += 1
            except RuntimeError:
                m['replan_failures'] += 1          # no way past: wait for now

        # Stop for anyone right in front of the robot
        in_front = (dist < STOP_DISTANCE) & (rel @ heading > 0.0)
        blocked = bool(in_front.any())
        v_target = 0.0 if blocked else V_MAX
        v = float(np.clip(v_target, v - A_MAX * DT, v + A_MAX * DT))
        step = min(v * DT, arc[-1] - s)
        s += step
        m['driven_m'] += step
        t += DT
        if v < 0.05:
            m['stopped_s'] += DT
        status = 'stopped: person in the way' if blocked else (
            'replanned around people' if t - last_replan < 1.5 and m['replans'] else 'driving')

        # Safety metrics: gap = distance between the edges of robot and person
        pos = _at(path, arc, s)
        gap = np.linalg.norm(crowd.pos - pos, axis=1) - ROBOT_RADIUS - PED_RADIUS
        if len(gap):
            m['min_gap_m'] = min(m['min_gap_m'], float(gap.min()))
            close = gap < NEAR_MISS
            touching = gap < 0
            m['near_misses'] += int((close & ~close_now).sum())   # count each approach once
            m['contacts'] += int((touching & ~touching_now).sum())
            close_now, touching_now = close, touching
        trail.append(pos.copy())

        if t >= next_frame:
            next_frame += record_every
            frames.append({'t': t, 'pos': pos.copy(), 'heading': heading.copy(), 'v': v,
                           'path_ahead': path[arc >= s].copy(), 'status': status,
                           'peds': crowd.pos.copy(), 'seen': seen.copy()})
        if s >= arc[-1] - 1e-6:
            break

    m['time_s'] = t
    m['arrived'] = bool(s >= arc[-1] - 1e-6)
    m['trail'] = np.array(trail)
    m['frames'] = frames
    return m


def _background(gt, png_path, step=2):
    """RGB campus image (every step-th cell) and its cell size, row 0 = south."""
    free = cg.free_mask(gt)[::step, ::step]
    rgb = np.empty(free.shape + (3,), dtype=np.uint8)
    rgb[:] = (38, 52, 44)
    b = cg.load_buildings(png_path)
    if b is not None:
        rgb[b[::step, ::step]] = (150, 158, 172)
    s = cg.load_layer(png_path, 'stairs')
    if s is not None:
        rgb[s[::step, ::step] & ~free] = (214, 69, 80)
    rgb[free] = (244, 241, 234)
    return rgb, gt.resolution * step


def save_animation(result, rgb, cell, out, window_m=50.0, fps=6):
    """GIF zoomed in on the robot, following it (a 50 m window)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle

    H, W = rgb.shape[:2]
    half = int(window_m / 2 / cell) + 2
    fig, ax = plt.subplots(figsize=(6, 6.4))
    # Only the part of campus around the robot is drawn each frame (fast)
    background = ax.imshow(rgb[:2, :2], origin='lower', interpolation='nearest')
    trail_line, = ax.plot([], [], color='#E8833A', lw=2.5)
    plan_line, = ax.plot([], [], color='#2F6FB0', lw=1.5, ls='--')
    # Robot and people drawn at their true size (circles in meters)
    robot = Circle((0, 0), ROBOT_RADIUS, color='#E8833A', ec='white', lw=1, zorder=5)
    sensor = Circle((0, 0), SENSOR_RANGE, fill=False, ec='#E8833A', ls=':', lw=1)
    ax.add_patch(robot)
    ax.add_patch(sensor)
    people = [Circle((0, 0), PED_RADIUS, lw=0.8, zorder=4) for _ in result['frames'][0]['peds']]
    for p in people:
        ax.add_patch(p)
    title = ax.set_title('', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.text(0.5, 0.02, f'orange: robot ({2 * ROBOT_RADIUS:.1f} m) and path driven   '
                        f'blue dashes: current plan\ndark: people the robot sees '
                        f'({2 * PED_RADIUS:.1f} m; dotted circle = 8 m sensor)   gray: out of range',
             ha='center', fontsize=8, color='#3F4A5C')
    trail = result['trail']

    def draw(i):
        f = result['frames'][i]
        x, y = f['pos']
        c, r = int(x / cell), int(y / cell)
        r0, r1 = max(r - half, 0), min(r + half, H)
        c0, c1 = max(c - half, 0), min(c + half, W)
        background.set_data(rgb[r0:r1, c0:c1])
        background.set_extent([c0 * cell, c1 * cell, r0 * cell, r1 * cell])
        ax.set_xlim(x - window_m / 2, x + window_m / 2)
        ax.set_ylim(y - window_m / 2, y + window_m / 2)
        n = int(round(f['t'] / DT))
        trail_line.set_data(trail[:n + 1, 0], trail[:n + 1, 1])
        plan_line.set_data(f['path_ahead'][:, 0], f['path_ahead'][:, 1])
        robot.center = (x, y)
        sensor.center = (x, y)
        for p, xy, seen in zip(people, f['peds'], f['seen']):
            p.center = tuple(xy)
            p.set_facecolor('#14213D' if seen else '#8A94A6')
            p.set_edgecolor('white' if seen else '#5B6576')
        title.set_text(f"t = {f['t']:5.1f} s   speed {f['v']:.1f} m/s   {f['status']}")
        return [background, trail_line, plan_line, robot, sensor, title] + people

    anim = FuncAnimation(fig, draw, frames=len(result['frames']), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


def save_map_animation(result, rgb, cell, start, goal, out, fps=6, shrink=3):
    """GIF of the whole campus map: the robot and pedestrians move on a fixed view."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle

    small = rgb[::shrink, ::shrink]          # the whole map, drawn once per frame
    H, W = rgb.shape[:2]
    W_m, H_m = W * cell, H * cell
    fig, ax = plt.subplots(figsize=(7.5, 7.6))
    ax.imshow(small, origin='lower', extent=[0, W_m, 0, H_m], interpolation='antialiased')
    trail = result['trail']
    first = result['frames'][0]
    route_line, = ax.plot(*first['path_ahead'].T, color='#2F6FB0', lw=1.2, ls='--')
    trail_line, = ax.plot([], [], color='#E8833A', lw=2.5)
    ax.plot(*start['delivery_xy'], 'o', color='#2BB673', ms=9, mec='white', mew=1.2, zorder=6)
    ax.plot(*goal['delivery_xy'], '*', color='#E63946', ms=15, mec='white', mew=1, zorder=6)
    sensor = Circle((0, 0), SENSOR_RANGE, fill=False, ec='#E8833A', lw=1)
    ax.add_patch(sensor)
    # Small bright points: at whole-map scale a real-size person is under a pixel
    peds_seen = ax.scatter([], [], s=7, c='#FFD23F', linewidths=0, zorder=5)
    peds_far = ax.scatter([], [], s=5, c='#4CC9F0', linewidths=0, zorder=4)
    robot = ax.scatter([], [], s=90, c='#E8833A', edgecolors='white', linewidths=1.5, zorder=7)
    title = ax.set_title('', fontsize=10)
    ax.set_xlim(0, W_m)
    ax.set_ylim(0, H_m)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m, east)')
    ax.set_ylabel('y (m, north)')
    fig.text(0.5, 0.01, 'green: pickup   red star: drop-off   orange: robot and path driven   '
                        'blue dashes: current plan\nyellow: people the robot sees '
                        '(orange circle = 8 m sensor)   light blue: other people   red: stairs',
             ha='center', fontsize=8, color='#3F4A5C')

    def draw(i):
        f = result['frames'][i]
        n = int(round(f['t'] / DT))
        trail_line.set_data(trail[:n + 1, 0], trail[:n + 1, 1])
        route_line.set_data(f['path_ahead'][:, 0], f['path_ahead'][:, 1])
        robot.set_offsets([f['pos']])
        sensor.center = tuple(f['pos'])
        peds_seen.set_offsets(f['peds'][f['seen']] if f['seen'].any() else np.empty((0, 2)))
        peds_far.set_offsets(f['peds'][~f['seen']] if (~f['seen']).any() else np.empty((0, 2)))
        title.set_text(f"{start['building']} → {goal['building']}\n"
                       f"t = {f['t']:5.1f} s   speed {f['v']:.1f} m/s   {f['status']}")
        return route_line, trail_line, robot, sensor, peds_seen, peds_far, title

    anim = FuncAnimation(fig, draw, frames=len(result['frames']), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


def save_overview(result, rgb, cell, start, goal, crowd_start, out):
    """Picture of the whole delivery: the driven path and where people stood."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    H, W = rgb.shape[:2]
    trail = result['trail']
    lo, hi = trail.min(axis=0) - 25, trail.max(axis=0) + 25
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(rgb, origin='lower', extent=[0, W * cell, 0, H * cell], interpolation='nearest')
    ax.plot(*trail.T, color='#E8833A', lw=2.5, label='robot path driven')
    ax.scatter(*crowd_start.T, s=30, c='#14213D', edgecolors='white', linewidths=0.8,
               zorder=4, label='pedestrians')
    ax.plot(*start['delivery_xy'], 'o', color='#2BB673', ms=11, mec='white', mew=1.5,
            label=f"pickup: {start['building']}")
    ax.plot(*goal['delivery_xy'], '*', color='#E63946', ms=18, mec='white', mew=1,
            label=f"drop-off: {goal['building']}")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal')
    ax.set_title('Delivery among pedestrians (red = stairs the robot cannot use)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)


def simulate(pickup, dropoff, n_peds=20, ped_speed=0.0, floor=None, from_floor=None,
             seed=0, gif=True, png_path=cg.DEFAULT_MAP, out_dir=HERE / 'docs'):
    """Plan a delivery, place pedestrians, run it, print metrics, save pictures."""
    gt, _ = cg.load_campus_grid(png_path)
    entrances = cg.load_entrances(png_path)
    elev, elev_res = cg.load_elevation(png_path)
    plan = cg.plan_delivery(gt, entrances, elev, elev_res, pickup, dropoff,
                            floor=floor, from_floor=from_floor)
    route, start, goal = plan['route'], plan['start'], plan['goal']

    speed = (0.0, 0.0) if ped_speed <= 0 else (0.8 * ped_speed, 1.2 * ped_speed)
    crowd = Crowd(gt, route, n_peds, speed=speed, seed=seed, png_path=png_path)
    crowd_start = crowd.pos.copy()
    what = 'standing still' if ped_speed <= 0 else f'walking at about {ped_speed:.1f} m/s'
    print(f'{n_peds} pedestrians near the route, {what}')

    t0 = time.perf_counter()
    result = run(gt, elev, elev_res, route, np.array(goal['delivery_xy']), crowd)
    m = result
    print(f'Simulated in {time.perf_counter() - t0:.1f} s:')
    if m['arrived']:
        print(f"  Delivered in {m['time_s']:.0f} s, driving {m['driven_m']:.0f} m "
              f"(average {m['driven_m'] / m['time_s']:.2f} m/s)")
    else:
        print(f"  Did NOT arrive within {m['time_s']:.0f} s: people blocked the way "
              f"({m['driven_m']:.0f} m driven)")
    print(f"  Stopped for {m['stopped_s']:.0f} s; replanned around people {m['replans']} times"
          + (f" ({m['replan_failures']} times there was no way past)"
             if m['replan_failures'] else ''))
    gap = m['min_gap_m']
    print(f"  Closest approach: {gap:.2f} m between robot and person"
          + (f"; near-misses (< {NEAR_MISS} m): {m['near_misses']}; contacts: {m['contacts']}"
             if np.isfinite(gap) else ''))

    rgb, cell = _background(gt, png_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_overview(result, rgb, cell, start, goal, crowd_start, out_dir / 'simulation.png')
    print(f"Saved {out_dir / 'simulation.png'}")
    if gif:
        for name, draw in (('simulation_follow.gif',      # zoomed in, follows the robot
                            lambda o: save_animation(result, rgb, cell, o)),
                           ('simulation_map.gif',         # zoomed out, whole campus map
                            lambda o: save_map_animation(result, rgb, cell, start, goal, o))):
            t0 = time.perf_counter()
            draw(out_dir / name)
            print(f"Saved {out_dir / name} ({len(result['frames'])} frames, "
                  f"{time.perf_counter() - t0:.0f} s to draw)")
    return result


def interactive(png_path=cg.DEFAULT_MAP, gif=True):
    """Ask for the delivery and the crowd in the terminal, then simulate."""
    entrances = cg.load_entrances(png_path)
    names = cg.building_names(entrances)
    print('Buildings the robot can deliver between (~ = floor estimated):')
    for i, name in enumerate(names, 1):
        print(f'  {i:2d}. {cg.describe_building(name, entrances)}')
    print('Type a number or part of a name. Floors are optional: press Enter to skip.\n')
    pickup = cg.ask_building('Pickup building: ', names)
    from_floor = cg.ask_floor(f'Pickup floor at {pickup} (optional, Enter to skip): ',
                              pickup, entrances)
    dropoff = cg.ask_building('Drop-off building: ', names)
    floor = cg.ask_floor(f'Drop-off floor at {dropoff} (optional, Enter to skip): ',
                         dropoff, entrances)
    while True:
        answer = input('Number of pedestrians (Enter for 20): ').strip()
        if not answer:
            n = 20
            break
        if answer.isdigit():
            n = int(answer)
            break
        print('  Type a whole number, e.g. 20.')
    print()
    simulate(pickup, dropoff, n_peds=n, floor=floor, from_floor=from_floor, gif=gif,
             png_path=png_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simulate a delivery among pedestrians. Run with no options to be asked.')
    parser.add_argument('--from', dest='pickup', default=None, help='pickup building')
    parser.add_argument('--to', dest='dropoff', default=None, help='drop-off building')
    parser.add_argument('--floor', default=None, help='drop-off floor (optional)')
    parser.add_argument('--from-floor', default=None, help='pickup floor (optional)')
    parser.add_argument('--pedestrians', type=int, default=20, help='number of pedestrians')
    parser.add_argument('--ped-speed', type=float, default=0.0,
                        help='pedestrian walking speed in m/s (0 = standing still)')
    parser.add_argument('--seed', type=int, default=0, help='random seed for the crowd')
    parser.add_argument('--no-gif', action='store_true', help='skip the animation')
    parser.add_argument('--map', type=Path, default=cg.DEFAULT_MAP)
    args = parser.parse_args()
    try:
        if args.pickup is None and args.dropoff is None:
            interactive(args.map, gif=not args.no_gif)
        elif args.pickup is None or args.dropoff is None:
            parser.error('give both --from and --to, or neither to be asked')
        else:
            simulate(args.pickup, args.dropoff, n_peds=args.pedestrians,
                     ped_speed=args.ped_speed, floor=args.floor, from_floor=args.from_floor,
                     seed=args.seed, gif=not args.no_gif, png_path=args.map)
    except (KeyboardInterrupt, EOFError):
        print('\nBye.')
    except ValueError as err:
        raise SystemExit(str(err))
