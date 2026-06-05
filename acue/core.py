# acue/core.py

from typing import Callable, List, Any


class System:
    def __init__(
        self,
        X: List[Any],
        O: Callable[[Any], Any],
        V: Callable[[Any], bool],
    ):
        self.X = X  # state space
        self.O = O  # transition operator
        self.V = V  # admissibility constraint

    def admissible(self, x):
        return self.V(x)

    def capability(self):
        return [x for x in self.X if self.V(x)]

    def step(self, x):
        return self.O(x)

    def evolve(self, x0, steps=10):
        trajectory = [x0]
        x = x0
        for _ in range(steps):
            x = self.O(x)
            trajectory.append(x)
        return trajectory

    def admissible_trajectory(self, x0, steps=10):
        traj = self.evolve(x0, steps)
        return all(self.V(x) for x in traj)
