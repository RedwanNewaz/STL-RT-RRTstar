import numpy as np

from rrt_star_stl.RRTstar import *
def main():
    # Testing STL specification (in the room's referential)
    alpha, beta, gamma, delta, t1, t2 = 150, 200, 300, 350, 20, 40
    test_pepper_condition = STLPredicate2D(0, 1, alpha, beta, gamma, delta)
    test_eventually_untimed = Untimed_Eventually(test_pepper_condition)
    test_eventually_timed = Eventually(test_pepper_condition, t1, t2)
    test_always_untimed = Untimed_Always(test_pepper_condition)
    test_always_timed = Always(test_pepper_condition, t1, t2)

    test_several_zones = Conjunction([Always(STLPredicate2D(0, 1, 400, 450, 50, 100), 15, 25), Always(STLPredicate2D(0, 1, 100, 150, 350, 400), 35, 55)])

    # Specification for human-robot encounters (in the human/obstacle referential)
    hurry_left = STLPredicate2D(0, 1, -80, -70, -100, 0)
    hurry_right = STLPredicate2D(0, 1, 60, 75, -60, 50)
    hurry = Disjunction([hurry_left, hurry_right])
    untimed_specification_hurry = Untimed_Eventually(hurry)
    timed_specification_hurry = Eventually(hurry, 25, 35)

    walk_left = STLPredicate2D(0, 1, -90, -80, -90, 0)
    walk_right = STLPredicate2D(0, 1, 70, 85, -60, 50)
    walk = Disjunction([walk_left, walk_right])
    untimed_specification_walk = Untimed_Eventually(walk)
    timed_specification_walk = Eventually(walk, 30, 40)

    fragile_left = STLPredicate2D(0, 1, -95, -80, -150, -45)
    fragile_right = STLPredicate2D(0, 1, 80, 95, -60, 50)
    fragile = Disjunction([fragile_left, fragile_right])
    untimed_specification_fragile = Untimed_Eventually(fragile)
    timed_specification_fragile = Eventually(fragile, 35, 45)

    # ====Search Path with RRT====
    obstacle_list = [
        (470.0, 390.0, 20)
    ]  # [x,y,size(radius)]

    # Set Initial parameters
    env_data = np.loadtxt("../config/dyn_env.csv", delimiter=',', skiprows=1)
    print(env_data.shape)
    ENVIRONMENT = {}
    for event in env_data.tolist():
        ENVIRONMENT[event[0]] = [(event[1], event[2], event[3])]


    """
    Instantiation
    """

    MAX_PEPPER_SPEED = 20
    TREE_SIZE = 1500

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

    # for alex, 2022-05-31
    path = rrt_star.planning(animation=show_animation,current_pos=[50, 50])
    path.reverse()
    print(path)
    exit()

    # If we want to print the warm start tree
    # rrt_star.draw_graph(room_area=[0,520,0,440])
    # plt.grid(True)
    # plt.show()

    # save a pre-computed tree as pickle
    # with open("stl_specifications/stl_tree_200_1500.dill", "wb") as dill_file:
    # dill.dump(rrt_star, dill_file)
    # exit()

    # load a pre-existing pickled tree
    # rrt_star = dill.load(open("stl_specifications/stl_tree_200_1500.dill", "rb"))

    # specification = specification_hurry_left
    # specification = specification_hurry_right
    # specification = specification_walk_left
    # specification = specification_walk_right
    # specification = specification_fragile_left
    # specification = specification_fragile_right

    # specification = specification_hurry
    # specification = specification_walk
    specification = Eventually(hurry_right, 25, 35)
    specification = untimed_specification_fragile

    # #update stl costs and rewire from tree root
    # print("update stl costs")
    # first_human_position = [470,390]
    # rrt_star.warmstart_update_stl_cost_humanref(first_human_position,specification)

    # with open("stl_specifications/stl_tree_200_1500_rewired_fragile_right.dill", "wb") as dill_file:
    # dill.dump(rrt_star, dill_file)
    # print("rewiring done")
    # exit()

    rrt_star = dill.load(open("../stl_specifications/stl_tree_200_1500_rewired_fragile_right.dill", "rb"))
    rrt_star.human_referential = True
    # Useful when restarting from not exact position, or with a custom goal
    # new_start = [65,60]
    # new_goal = [420,350]
    # rrt_star.set_new_start_new_goal(new_start,new_goal,specification)

    move_pepper = np.loadtxt("../config/move_pepper.csv", delimiter=',').tolist()


    followed_path_human_referential = []
    previous_human_position = [478.4, 396.8]

    for t in ENVIRONMENT:

        # Update current position with the first goal of last calculated path
        # TODO REAL EXP PEPPER: replace path[1] by the measured current position of Pepper

        # Before July 5 - assuming that the agent would go to the next waypoint specified in the path
        # current_pos = path[1]

        # July 5 - finding the nearest node in the path to the current position
        # fake this in simulation
        try:
            if t in move_pepper:
                try:
                    measured_robot_position = [path[1][0] + random.random(), path[1][1] + random.random()]
                except IndexError:
                    measured_robot_position = [path[0][0] + random.random(), path[0][1] + random.random()]
                current_pos = rrt_star.find_nearest_node_path(measured_robot_position, path_nodes)
                print("measured position", measured_robot_position, current_pos)
            else:
                measured_robot_position = [path[0][0] + random.random(), path[0][1] + random.random()]
                current_pos = rrt_star.find_nearest_node_path(measured_robot_position, path_nodes)
                print("measured position", measured_robot_position, current_pos)
        except UnboundLocalError:
            current_pos = rrt_star.start

        # Plan for the current iteration of given frequency
        # TODO REAL EXP PEPPER: replace "ENVIRONMENT[t][0]" in the "updated_human_position=ENVIRONMENT[t][0]" by the measured positions of the human/obstacle
        start_time = time.time()
        (path, path_nodes), cost_updating_costs, nb_of_rewiring_checks = rrt_star.planning(animation=show_animation,
                                                                                           current_pos=current_pos,
                                                                                           previous_human_position=previous_human_position,
                                                                                           updated_human_position=
                                                                                           ENVIRONMENT[t][0],
                                                                                           stl_specification=specification)
        elapsed = (time.time() - start_time)

        # if t==1.2:
        # dico = {}
        # for node in rrt_star.node_list:
        # dico[node] = node.stl_cost
        # sorteddico = {k: v for k, v in sorted(dico.items(), key=lambda item: item[1])}
        # for node in sorteddico:
        # print(node,sorteddico[node])
        # exit()

        path.reverse()
        path_nodes.reverse()
        # !! path[0] is the recorded position of the agent, and path[1] is the next waypoint !!

        print("t=", t, ", found path in", elapsed, "seconds. Updated costs in", cost_updating_costs)
        print(path)

        try:
            if path[1] == [rrt_star.end.x, rrt_star.end.y]:
                break
        except IndexError:
            if path[0] == [rrt_star.end.x, rrt_star.end.y]:
                break

        try:
            followed_path_human_referential.append(
                (round(path_nodes[1].x_humanreferential), round(path_nodes[1].y_humanreferential)))
        except IndexError:
            followed_path_human_referential.append(
                (round(path_nodes[0].x_humanreferential), round(path_nodes[0].y_humanreferential)))
        # TODO REAL EXP PEPPER: use path[1] as the next goal to reach, then path[2], path[3] etc...
        # RESOLUTION 10Hz = 1 point per 100ms

        rrt_star.draw_graph(room_area=[0, 520, 0, 440])
        plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r--')
        plt.grid(True)
        # plt.show()
        # plt.savefig('img/' + str(t) + '_stl.png')
        plt.pause(0.001)

        print("\n\n\n")

    print(followed_path_human_referential)
    # plot_human_referential(followed_path_human_referential)

    exit()


if __name__ == '__main__':
    main()