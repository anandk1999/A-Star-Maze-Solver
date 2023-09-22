class BinaryHeap():

    def __init__(self, smaller_g, minBool):
        self.arr = list()
        self.smaller_g = smaller_g
        self.minBool = minBool
    
    def __len__(self):
        return len(self.arr)

    def left(self, k):
        return 2 * k + 1

    def right(self, k):
        return 2 * k + 2

    def min_heapify(self,k):
        l = self.left(k)
        r = self.right(k)
        if self.smaller_g:
            if l < len(self.arr) and self.arr[l].tieBreaker(self.arr[k],self.smaller_g) < 0:
                smallest = l
            else:
                smallest = k
            if r < len(self.arr) and self.arr[r].tieBreaker(self.arr[smallest],self.smaller_g) < 0:
                smallest = r
            if smallest != k:
                self.arr[k], self.arr[smallest] = self.arr[smallest], self.arr[k]
                self.min_heapify(smallest)

    def build_min_heap(self):
        n = int((len(self.arr)//2)-1)
        for k in range(n, -1, -1):
            self.min_heapify(k)
    
    def peek(self):
        return self.arr[0]
    
    def insert(self, cell):
        self.arr.append(cell)
        self.min_heapify(len(self.arr) - 1)

    def remove(self,index):
        self.arr.pop(index)
    
    def pop(self):
        if len(self.arr) == 0:
            print("No elements")
            return
        cell = self.arr.pop(0)
        self.min_heapify(0)
        return cell
    
    def validHeap(self):
            if len(self.arr) <= 1:
                return True
        
            for i in range((len(self.arr) - 2) // 2 + 1):
                if self.arr[i] < self.arr[2*i + 1] or (2*i + 2 != len(self.arr) and self.arr[i] < self.arr[2*i + 2]):
                    return False
        
            return True

    def index_of(self, cell):
        cells = [tempstate.cell for tempstate in self.arr]
        if cell in cells:
            return cells.index(cell)
        return -1