class State:

    def __init__(self, cell, gVal, hVal):
        self.cell = cell
        self.gVal = gVal
        self.hVal = hVal

    def get_fVal(self):
        return self.gVal + self.hVal

    def tieBreaker(self, other, smaller_g):
        c = 101**2
        if (self.get_fVal() - other.get_fVal()) == 0:
            if smaller_g:
                return (c*self.get_fVal() - self.gVal) - (c*other.get_fVal() - other.gVal)
            else:
                return (c*other.get_fVal() - other.gVal) - (c*self.get_fVal() - self.gVal)
        return (self.get_fVal() - other.get_fVal()) 