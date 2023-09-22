from datetime import date
import datetime
import random
import sys
import numpy as np
from BinaryHeap import BinaryHeap
from Cell import Cell
from State import State

sys.setrecursionlimit(10000)
class Agent:
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

    DIRECTIONS = [NORTH, EAST, SOUTH, WEST]

    def __init__(self, maze):
        self.maze = maze

    def move(self, start, direction):
        directions = {
            Agent.NORTH: (lambda start: self.maze.maze_map[start.x - 1][start.y] if start.x >= 1 else None),
            Agent.SOUTH: (lambda start: self.maze.maze_map[start.x + 1][start.y] if start.x + 1 < self.maze.nx else None),
            Agent.EAST: (lambda start: self.maze.maze_map[start.x][start.y + 1] if start.y + 1 < self.maze.ny else None),
            Agent.WEST: (lambda start: self.maze.maze_map[start.x][start.y - 1] if start.y >= 1 else None),
        }
        if direction in directions:
            return directions[direction](start)
        else:
            return None

    # get unvisited neighbors of the agent's current cell
    def validNeigbhors(self, cell, invalid) :
        res = list()
        walls = list()
        for direction in Agent.DIRECTIONS:
            neighbor = self.move(cell, direction)
            if neighbor is not None:
                if self.maze.maze_map[neighbor.x][neighbor.y].type == Cell.VALID:
                    res.append(neighbor)
                else:
                    walls.append(neighbor)
        return res if not invalid else walls

    def AStar(self, start, end, forward, smaller, adaptive):
        iterations = 0
        expandedCells = 0
        search = [[0 for i in range(self.maze.nx)] for j in range(self.maze.ny)]
        for i in range(self.maze.nx):
            for j in range(self.maze.ny):
                self.maze.maze_map[i][j].hVal = Maze.manhattanDistance(self.maze, j, end.x, i, end.y)

        walls = []
        finalPath = list()
        if forward:
            finalPath.insert(0,(start.x,start.y))
        else:
            finalPath.insert(0,(end.x,end.y))

        while not (start.x == end.x and start.y == end.y):
            # Updating variables
            iterations += 1
            self.maze.maze_map[start.x][start.y].gVal = 0
            search[start.x][start.y] = iterations
            self.maze.maze_map[end.x][end.y].gVal = np.Infinity
            search[end.x][end.y] = iterations

            # Initializing Open and Closed list
            open_list = BinaryHeap(smaller, minBool=True)
            open_list.insert(State(start, self.maze.maze_map[start.x][start.y].gVal, self.maze.maze_map[start.x][start.y].hVal))
            closed_list = []

            # Adding all neighbors of start node that are walls to walls list
            [walls.append(neighbor) if neighbor not in walls else None for neighbor in self.validNeigbhors(start, invalid=True)]
            
            # Initializing tree pointers list
            tree_pointers = [[None for i in range(self.maze.nx)] for j in range(self.maze.ny)]

            while self.maze.maze_map[end.x][end.y].gVal > open_list.peek().get_fVal():
                # Remove root of open list and add to closed list
                s = open_list.pop()
                if s not in closed_list:
                    closed_list.append(s) 
                else:
                    continue
                
                # Going through all actions from node popped from open list
                neighbors = self.validNeigbhors(s.cell, invalid=False)
                for neighbor in neighbors:
                    if search[neighbor.x][neighbor.y] < iterations:
                        self.maze.maze_map[neighbor.x][neighbor.y].gVal = np.Infinity
                        search[neighbor.x][neighbor.y] = iterations
                    if self.maze.maze_map[neighbor.x][neighbor.y].gVal > s.gVal + 1:
                        self.maze.maze_map[neighbor.x][neighbor.y].gVal = s.gVal + 1
                        tree_pointers[neighbor.x][neighbor.y] = s
                        index = open_list.index_of(neighbor)
                        if index != -1:
                            open_list.remove(index)
                        open_list.insert(State(neighbor, self.maze.maze_map[neighbor.x][neighbor.y].gVal, self.maze.maze_map[neighbor.x][neighbor.y].hVal))
            expandedCells += len(closed_list)
            # Adaptive A*
            if adaptive:
                for expanded in closed_list:
                    self.maze.maze_map[expanded.cell.x][expanded.cell.y].hVal = self.maze.maze_map[end.x][end.y].gVal - self.maze.maze_map[expanded.cell.x][expanded.cell.y].gVal

            if len(open_list) == 0:
                print('Unable to reach target')
                raise Exception("unable to reach")

            # Following tree pointers
            path = list()
            if forward:
                path.insert(0, State(end, self.maze.maze_map[end.x][end.y].gVal, self.maze.maze_map[end.x][end.y].hVal))
                while path[0].cell.x != start.x or path[0].cell.y != start.y:
                    path.insert(0,tree_pointers[path[0].cell.x][path[0].cell.y])
            else:
                path.append(State(end, self.maze.maze_map[end.x][end.y].gVal, self.maze.maze_map[end.x][end.y].hVal))
                while path[-1].cell.x != start.x or path[-1].cell.y != start.y:
                    path.append(tree_pointers[path[-1].cell.x][path[-1].cell.y])

            temp = [(tempcell.cell.x,tempcell.cell.y) for tempcell in path]

            # Iterating current path and forming finalPath to print
            for i in range(1,len(path)):
                if path[i].cell.type == Cell.WALL:
                    walls.add(path[i].cell) if path[i].cell not in walls else None
                    [finalPath.append((tempcell.cell.x,tempcell.cell.y)) for tempcell in path[1:i]]
                    break
                else:
                    if forward:
                        start = path[i].cell
                    else:
                        end = path[i].cell
            finalPath = self.removePalindrome(finalPath)

        print('Target Reached')
        [finalPath.append(tup) for tup in temp]
        finalPath = self.removePalindrome(finalPath)
        return finalPath, expandedCells
    
    def removePalindrome(self, finalPath):
        palindromes = self.find_palindromes(finalPath)
        for item in palindromes:
            for i in range(item[0]+1,item[1]+1):
                finalPath[i] = -1
        return [step for step in finalPath if step != -1]

    def find_palindromes(self, s):
        n = len(s)
        palindromes = []

        for i in range(n):
            l = i
            r = i
            while l >= 0 and r < n and s[l] == s[r]:
                if r - l + 1 >= 3:
                    palindromes.append((l, r))
                l -= 1
                r += 1

        for i in range(n-1):
            l = i
            r = i + 1
            while l >= 0 and r < n and s[l] == s[r]:
                if r - l + 1 >= 3:
                    palindromes.append((l, r))
                l -= 1
                r += 1
        
        for item in palindromes:
            temp = set()
            temp.update(s[item[0]:item[1]+1])
            if len(temp) == 1:
                palindromes.remove(item)

        return palindromes

# Source for Maze: https://ideone.com/oufifB        
class Maze:

    def __init__(self, graph_size_x, graph_size_y):
        self.nx, self.ny = graph_size_x, graph_size_y
        self.ix, self.iy = random.choice(range(graph_size_x)), random.choice(range(graph_size_y))
        self.maze_map = np.array([[Cell(x, y) for y in range(graph_size_y)] for x in range(graph_size_x)])
    
    def __len__(self):
        return len(self.maze_map)

    # Return the Cell object at (x,y).
    def getCell(self, x, y):
        return self.maze_map[x][y]

    # Return Neighbour of given.
    def getNeighbours(self, x, y):
        return self.maze_map[x][y].getNeighbours()

    def manhattanDistance(self, row1, row2, col1, col2):
        return (np.abs(row2 - row1) + np.abs(col2 - col1))

    def print2d(self,array):
        for row in array:
            for col in row:
                print(col.type, end=' ')
            print()
        print('\n')

    def isGoodPath(self, point):
        if point[0] < 0 or point[0] >= len(self.maze_map) or point[1] < 0 or point[1] >= len(self.maze_map[0]):
            return False

        dirs = [
            [0,-1],
            [1,0],
            [0,1],
            [-1,0],
        ]

        countPath = 0
        for dir in dirs:
            newPoint = [point[0] + dir[0], point[1] + dir[1]]
            if (
                newPoint[0] >= 0 and newPoint[0] < len(self.maze_map) and
                newPoint[1] >= 0 and newPoint[1] < len(self.maze_map[0]) and
                self.maze_map[newPoint[0]][newPoint[1]].type == Cell.VALID
            ):
                countPath += 1

        if countPath > 1:
            return False

        return True

    def generateMaze(self, point):
        dirs = [
            [0,-1],
            [1,0],
            [0,1],
            [-1,0],
        ]

        random.shuffle(dirs)

        for dir in dirs:
            newPoint = [point[0] + dir[0], point[1] + dir[1]]

            if self.isGoodPath(newPoint):
                self.maze_map[newPoint[0]][newPoint[1]].type = Cell.VALID
                self.generateMaze(newPoint)

        self.maze_map
    
def remove_duplicate(start,oldlist,newlist):
        if start==len(oldlist):return newlist  #base condition
        if oldlist[start] not in newlist:   #checking whether element present in new list  or not
            newlist.append(oldlist[start])
        return remove_duplicate(start+1,oldlist,newlist)

class MazeStore():

    def __init__(self, nx = 101, ny = 101, n=50, path='Mazes\\'):
        self.nx = nx
        self.ny = ny
        self.n = n
        self.path = path
    
    def loadMazes(self, name):
        return np.load(f'{self.path}{name}.npy', allow_pickle=True)
    
    def saveMazes(self):
        maze = Maze(self.nx, self.ny)
        for count in range(1, self.n+1):
            maze = Maze(self.nx, self.ny)
            maze.generateMaze((0,0))
            with open(f"{self.path}{count}.npy", 'wb') as file:
                np.save(file, maze.maze_map)
            print(f"{count} done! on path: {self.path}{count}.npy")

if __name__ == "__main__":

    # Uncomment the two line below to generate 50 random mazes and save to your directory.
    # temp = MazeStore()
    # temp.saveMazes()

    forward_smaller_tie_time = []
    forward_larger_tie_time = []
    smaller_expanded = []
    larger_expanded = []
    forward_time = []
    backward_time = []
    forward2_time = []
    adaptive_time = []

    getMaze = MazeStore()
    maze = Maze(101,101)

    for i in range(1):

        loadedMaze = getMaze.loadMazes(i+1)
        maze.maze_map = loadedMaze

        temp = maze.getCell(0, 0)
        print(temp)
        print(temp.getNeighbours())
        my_agent = Agent(maze)

        emptys = [cell for cell in my_agent.maze.maze_map.flatten() if cell.type == Cell.VALID]
        start, end = np.random.choice(emptys, 2, replace=False)
        print(start.x,start.y)
        print(end.x,end.y)
        # Part 2
        print("Part 2 Results \n")
        start_time = datetime.datetime.now()
        forward_smaller_tie = remove_duplicate(0,my_agent.AStar(start, end, forward=True, smaller=True, adaptive=False),[])
        # print("Runtime: ", (datetime.datetime.now()-start_time).microseconds)
        forward_smaller_tie_time.append((datetime.datetime.now()-start_time).microseconds)
        smaller_expanded.append(forward_smaller_tie[1])
        print("-----------------------------")

        start_time = datetime.datetime.now()
        forward_larger_tie = remove_duplicate(0,my_agent.AStar(start, end, forward=True, smaller=False, adaptive=False),[])
        # print("Runtime: ", (datetime.datetime.now()-start_time).microseconds)
        forward_larger_tie_time.append((datetime.datetime.now()-start_time).microseconds)
        larger_expanded.append(forward_larger_tie[1])
        print("-----------------------------")

        # Part 3
        print("Part 3 Results \n")
        start_time = datetime.datetime.now()
        forward = remove_duplicate(0,my_agent.AStar(start, end, forward=True, smaller=False, adaptive=False),[])
        # print("Runtime: ", (datetime.datetime.now()-start_time).microseconds)
        forward_time.append((datetime.datetime.now()-start_time).microseconds)
        # print(forward[0], forward[1])
        print("-----------------------------")

        start_time = datetime.datetime.now()
        backward = remove_duplicate(0,my_agent.AStar(start, end, forward=False, smaller=False, adaptive=False),[])
        # print("Runtime: ", (datetime.datetime.now()-start_time).microseconds)
        backward_time.append((datetime.datetime.now()-start_time).microseconds)
        # print(backward[0], backward[1])
        print(len(forward)==len(backward))
        # maze.print2d(maze.maze_map)
        print(forward)
        print(backward)

        # Part 5
        print("Part 5 Results \n")
        start_time = datetime.datetime.now()
        forward2 = remove_duplicate(0,my_agent.AStar(start, end, forward=True, smaller=False, adaptive=False),[])
        # print("Runtime: ", (datetime.datetime.now()-start_time).microseconds)
        forward2_time.append((datetime.datetime.now()-start_time).microseconds)
        print("Iterations: ", forward2[1])
        print("-----------------------------")

        start_time = datetime.datetime.now()
        adaptive = remove_duplicate(0,my_agent.AStar(start, end, forward=True, smaller=False, adaptive=True),[])
        # print("Runtime: ", (datetime.datetime.now()-start_time).microseconds)
        adaptive_time.append((datetime.datetime.now()-start_time).microseconds)
        print("Iterations: ", adaptive[1])
        print("-----------------------------")
    
    print("Repeated Forward A* break tie with smaller g-value runtime: ",sum(forward_smaller_tie_time)/(10**6))
    print("Repeated Forward A* break tie with larger g-value runtime: ",sum(forward_larger_tie_time)/(10**6))
    print("Average Expanded Cells Repeated Forward A* break tie with smaller g-value: ", sum(smaller_expanded)/len(smaller_expanded))
    print("Average Expanded Cells Repeated Forward A* break tie with larger g-value: ", sum(larger_expanded)/len(larger_expanded))
    print("Repeated Forward A* runtime: ",sum(forward_time)/(10**6))
    print("Repeated Backward A* runtime: ",sum(backward_time)/(10**6))

    print("Repeated Forward A* runtime: ",sum(forward2_time)/(10**6))
    print("Adaptive A* Runtime",sum(adaptive_time)/(10**6))
