class INSState:
    def __init__(self, p, v, q, ba, bg):
        self.p  = p.copy()
        self.v  = v.copy()
        self.q  = q.copy()
        self.ba = ba.copy()
        self.bg = bg.copy()
