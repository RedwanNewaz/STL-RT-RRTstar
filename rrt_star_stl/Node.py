import math

class Node(object):
    """RRT Node"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_humanreferential = None
        self.y_humanreferential = None
        self.path_x = []
        self.path_y = []
        self.parent = None
        self.children = {}
        self.cost = 0.0
        self.blocking_cost = 0.0
        self.stl_cost = 0.0
        self.rho_bar = float("-inf")
        self.trajectory_until_node = []
        self.trajectory_until_node_humanreferential = []

    def total_cost(self):
        return self.cost + self.blocking_cost + self.stl_cost

    def __str__(self):
        try:
            return f'({round(float(self.x), 1)}, {round(float(self.y), 1)})'
        except Exception:
            return "(invalid coords)"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.x, self.y))

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        if self is other:
            return True
        try:
            return math.isclose(float(self.x), float(other.x), abs_tol=1e-9) and \
                math.isclose(float(self.y), float(other.y), abs_tol=1e-9)
        except (TypeError, ValueError):
            return False