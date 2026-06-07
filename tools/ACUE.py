import random

# -------------------------
# 1. State Space
# -------------------------
X = list(range(0, 50))

# -------------------------
# 2. World Dynamics O
# -------------------------
def O(x):
    return max(0, min(49, x + random.choice([-3, -1, 1, 3])))

# -------------------------
# 3. Constraint Family {V_i}
# -------------------------
class Constraint:
    def __init__(self, bias):
        self.bias = bias
    
    def V(self, x):
        score = (x % 10) / 10 + self.bias
        return 1 if score > 0.5 else 0
    
    def update(self, x):
        # constraint evolution rule
        self.bias += 0.01 * ((x % 2) - 0.5)

# initialize constraint universe
constraints = [Constraint(random.random()) for _ in range(5)]

# -------------------------
# 4. Selection Process
# -------------------------
def select_states(states, constraints):
    new_states = []
    for x in states:
        if any(V.V(x) == 1 for V in constraints):
            new_states.append(x)
    return new_states

# -------------------------
# 5. Simulation Loop
# -------------------------
state = random.choice(X)

trajectory = []

for t in range(100):
    state = O(state)
    
    # constraint evaluation
    allowed = any(V.V(state) for V in constraints)
    
    if allowed:
        trajectory.append(state)
    else:
        state = random.choice(X)
    
    # constraint evolution (self-reference)
    for V in constraints:
        V.update(state)

print("trajectory:", trajectory[:20])
