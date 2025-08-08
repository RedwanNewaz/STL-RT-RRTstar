import numpy as np
import os

MAX_PEPPER_SPEED = 20
TREE_SIZE = 1500
obstacle_list = [
    (470.0, 390.0, 20)
]  # [x,y,size(radius)]

from rrt_star_stl.RRTstar import *


def get_measured_position(t, path, move_pepper):
    """Get robot's measured position with noise."""
    try:
        # Check if path exists and has elements
        if not path:
            return None

        if t in move_pepper:
            # Use second position if available, fallback to first
            pos_index = 1 if len(path) > 1 else 0
        else:
            pos_index = 0

        base_pos = path[pos_index]
        return [base_pos[0] + random.random(), base_pos[1] + random.random()]
    except (IndexError, UnboundLocalError):
        return None


def get_current_position(measured_position, rrt_star, path_nodes):
    """Get current position from measured position or fallback to start."""
    if measured_position is None:
        return rrt_star.start

    current_pos = rrt_star.find_nearest_node_path(measured_position, path_nodes)
    print("measured position", measured_position, current_pos)
    return current_pos


def run_planning(rrt_star, current_pos, previous_human_position, environment_t, specification, show_animation):
    """Execute RRT* planning and return results."""
    start_time = time.time()

    (path, path_nodes), cost_updating_costs, nb_of_rewiring_checks = rrt_star.planning(
        animation=show_animation,
        current_pos=current_pos,
        previous_human_position=previous_human_position,
        updated_human_position=environment_t[0],
        stl_specification=specification
    )

    elapsed = time.time() - start_time
    path.reverse()
    path_nodes.reverse()

    return path, path_nodes, elapsed, cost_updating_costs


def is_goal_reached(path, rrt_star):
    """Check if goal position is reached."""
    goal_pos = [rrt_star.end.x, rrt_star.end.y]

    # Check second position if available, fallback to first
    check_pos = path[1] if len(path) > 1 else path[0]
    return check_pos == goal_pos


def update_followed_path(path_nodes, followed_path_human_referential):
    """Update the followed path with human referential coordinates."""
    # Use second node if available, fallback to first
    node = path_nodes[1] if len(path_nodes) > 1 else path_nodes[0]
    followed_path_human_referential.append(
        (round(node.x_humanreferential), round(node.y_humanreferential))
    )


def visualize_path(rrt_star, path):
    """Draw the graph and current path."""
    rrt_star.draw_graph(room_area=[0, 520, 0, 440])
    plt.plot([x for x, y in path], [y for x, y in path], 'r--')
    plt.grid(True)
    plt.pause(0.001)

def main(specification):
    # Set Initial parameters
    env_data = np.loadtxt("../config/dyn_env.csv", delimiter=',', skiprows=1)
    move_pepper = np.loadtxt("../config/move_pepper.csv", delimiter=',').tolist()
    # rrt_star = dill.load(open("../stl_specifications/stl_tree_200_1500_rewired_fragile_right.dill", "rb"))
    # rrt_star = dill.load(open("../stl_specifications/stl_tree_200_1500_rewired_fragile_left.dill", "rb"))
    rrt_star = RRTStar(
        start=[50, 50],
        goal=[470, 390],
        rand_area=[50, 470, 50, 390],
        obstacle_list=obstacle_list,
        expand_dis=MAX_PEPPER_SPEED,
        max_iter=10000,
        max_time=0.1,
        goal_sample_rate=5,
        path_resolution=MAX_PEPPER_SPEED,
        grid_size=20,
        warm_start=True,
        warm_start_tree_size=TREE_SIZE,
        robot_radius=24)
    ENVIRONMENT = {event[0]: [tuple(event[1:])] for event in env_data.tolist()}
    rrt_star.human_referential = True

    followed_path_human_referential = []
    previous_human_position = [478.4, 396.8]
    # Initialize variables
    path = []
    path_nodes = []

    # Main loop
    for t in ENVIRONMENT:
        # Get measured position
        measured_position = get_measured_position(t, path, move_pepper)
        current_pos = get_current_position(measured_position, rrt_star, path_nodes)

        # Run planning
        path, path_nodes, elapsed, cost_updating_costs = run_planning(
            rrt_star, current_pos, previous_human_position,
            ENVIRONMENT[t], specification, show_animation
        )

        print(f"t={t}, found path in {elapsed} seconds. Updated costs in {cost_updating_costs}")
        print(path)

        # Check if goal reached
        if is_goal_reached(path, rrt_star):
            break

        # Update followed path
        update_followed_path(path_nodes, followed_path_human_referential)

        # Visualize
        visualize_path(rrt_star, path)

    print(followed_path_human_referential)
    # plot_human_referential(followed_path_human_referential)

    exit()


if __name__ == '__main__':
    from specifications import untimed_specification_fragile
    main(untimed_specification_fragile)