# experiments/exp_01_basic_universe.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from acue.core import System
from acue.constraints import bound_constraint


# State space
X = list(range(-20, 21))

# Transition: simple dynamics
def O(x):
    return x + 1


# Constraint: admissibility
V = bound_constraint(-10, 10)


# Build system
system = System(X, O, V)


# Run experiment
print("Admissible states:")
print(system.capability())

print("\nTrajectory from 0:")
traj = system.evolve(0, steps=15)
print(traj)

print("\nIs trajectory admissible?")
print(system.admissible_trajectory(0, steps=15))
