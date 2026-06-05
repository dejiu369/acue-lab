class System:
    def __init__(self, X, O, V):
        self.X = X
        self.O = O
        self.V = V

    def admissible(self, x):
        return self.V(x)

    def capability(self):
        return [x for x in self.X if self.V(x)]
