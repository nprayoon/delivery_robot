# planners/

The route-planning part of the delivery robot, split out from `campus_grid.py`.

| File | What it does |
|---|---|
| `problem.py` | `PlanningProblem`: the map, start, goal, rules and costs. Every planner reads it. |
| `astar.py` | `astar(problem)`: A* on a 0.5 m grid, returns the path as an (N, 2) array of (x, y) meters. |
| `test_planners.py` | Checks A* paths are valid (on the sidewalk, no stairs, not too steep, clear of people). |

`campus_grid.plan_route` and `simulate.py` call `astar` through `PlanningProblem`, so their results are unchanged.

## The problem

The robot drives from a start door to a goal door on CMU's sidewalks.

- **Coordinates:** meters from the map's south-west corner, x east, y north. Internally a grid of 0.5 m cells indexed `(row, col)`, row 0 at the south edge.
- **Where it may drive:** sidewalk cells at least 30 cm from the edge, no stairs, inside the one connected piece of sidewalk the robot can reach (`problem.drivable`), and at least `obstacle_radius` from any obstacle such as a pedestrian (`problem.blocked`).
- **Slopes:** never steeper than 20%.
- **Cost** of driving straight a distance L with height change dz:

  ```
  L * (1 + edge penalty)                        distance, plus extra near the sidewalk edge
  + L * STEEP_WEIGHT * (grade - 8.3%)           only on slopes steeper than a legal ramp
  + CLIMB_WEIGHT * meters climbed
  + DESCENT_WEIGHT * meters descended
  ```

  The constants live in `campus_grid.py`.

## Using it

```python
from planners.problem import PlanningProblem
from planners.astar import astar

problem = PlanningProblem.between_buildings('Cohon University Center', 'Doherty Hall')
path = astar(problem)                  # (N, 2) points in meters
print(problem.evaluate(path))          # valid, length_m, cost, climb_m, max_grade, ...
```

Useful helpers on `problem`:

| Helper | Gives |
|---|---|
| `start_xy`, `goal_xy` | start and goal points (meters) |
| `cell_of(xy)`, `xy_of(cell)` | point to cell and back |
| `is_free(cell)` | whether the robot may be in a cell |
| `random_free_xy(rng)` | a random drivable point |
| `segment_cost(a, b)` | cost of driving straight from a to b, or `inf` if it leaves the drivable area, hits stairs, is too steep or passes too close to an obstacle |
| `path_cost(path)`, `evaluate(path)` | total cost, and numbers for comparing paths |

## A* in short

8-connected grid search. The heuristic is octile distance plus `CLIMB_WEIGHT` times the net height still to climb, which never overestimates, so the route is the cheapest on the grid.

## Run the checks

From the `delivery_robot` folder:

```
python planners/test_planners.py
```
