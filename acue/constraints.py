# acue/constraints.py

def bound_constraint(min_val, max_val):
    def V(x):
        return min_val <= x <= max_val
    return V


def parity_constraint(even=True):
    def V(x):
        return (x % 2 == 0) if even else (x % 2 == 1)
    return V


def energy_constraint(max_energy):
    def V(x):
        return abs(x) <= max_energy
    return V
