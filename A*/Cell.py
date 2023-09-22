class Cell:
    WALL = '#'
    VALID = '.'   
    PATH = '*' 
    # A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x, y, type=WALL):
        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}
        self.type = type
        self.gVal = None
        self.hVal = 0

    def __str__(self):
        return self.type

    def getNeighbours(self):
        return self.walls

    def has_all_walls(self):
        return all(self.walls.values())

    def knock_down_wall(self, other, wall):
        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False