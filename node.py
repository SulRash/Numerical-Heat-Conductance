class Node:

    def __init__(self, i, j, value) -> None:
        self.i = i
        self.j = j
        self.value = value
        self.code = "T{i}{j}".format(i=i,j=j)