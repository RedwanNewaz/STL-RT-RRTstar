from rrt_star_stl.STLFormula import *
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

specification = Eventually(hurry_right, 25, 35)
