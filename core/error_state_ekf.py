import numpy as np

class INSState:
    def __init__(self, p, v, q, ba, bg):
        self.p = p.copy()
        self.v = v.copy()
        self.q = q.copy()
        self.ba = ba.copy()
        self.bg = bg.copy()

class ErrorStateEKF:
    def __init__(self):
        self.x = np.zeros(15)
        self.P = np.eye(15) * 0.01
        self.Q = np.eye(15) * 1e-5

    def propagate(self, dt):
        F = np.eye(15)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q * dt

    def inject(self, state: INSState):
        state.ba -= self.x[9:12]
        state.bg -= self.x[12:15]
        self.x[:] = 0.0
