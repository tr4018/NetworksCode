"""
    simple  network model
    based on C++ implementation from Tim Evans
    author - Talia Rahall, March 2021
"""


class SimpleGraph:
    def __init__(self):
        self.v2v = []  # graph is represented by a list of lists

    def add_vertex(self) -> int:
        # add a vertex to the graph by extending the list and adding a new empty list
        self.v2v.append([])
        return len(self.v2v) - 1

    def add_edge(self, s: int, t: int):
        # add an edge - sets the entries for source (s) and target (t)
        self.v2v[s].append(t)
        self.v2v[t].append(s)

    def add_edge_slowly(self, s: int, t: int) -> bool:
        # adds an edge by ensuring the list of vertices is large enough to hold source and target
        if s == t:
            return False
        max_size = max(s, t) + 1
        for i in range(len(self.v2v), max_size):
            self.v2v.append([])
        self.add_edge(s, t)
        return True

    def get_edges(self) -> []:
        edges = []
        for s in range(len(self.v2v)):
            for t in range(len(self.v2v[s])):
                if (s, t) not in edges and (t, s) not in edges:
                    edges.append((s, t))
        return edges

    def get_neighbour(self, s: int, n: int) -> int:
        return self.v2v[s][n]

    def get_neighbours(self, s: int) -> []:
        return self.v2v[s]

    def get_num_stubs(self) -> int:
        stubs = sum([self.get_vertex_degree(v) for v in range(self.get_num_vertices())])
        return stubs

    def get_vertex_degree(self, v: int) -> int:
        return len(self.v2v[v])

    def get_num_vertices(self) -> int:
        return len(self.v2v)

    def get_num_edges(self) -> int:
        return self.get_num_stubs() // 2

    def get_max_vertex_degree(self) -> int:
        mvd = 0
        for v in self.v2v:
            mvd = max(len(v), mvd)
        return mvd
        #return max([self.get_vertex_degree(v) for v in range(self.get_num_vertices())])

    def get_degree_dist(self) -> []:
        degrees = [self.get_vertex_degree(v) for v in range(self.get_num_vertices())]
        max_degree = max(degrees)
        dd = [0] * (1 + max_degree)
        for d in range(len(degrees)):
            dd[degrees[d]] += 1
        return dd

    def print(self):
        for source in range(self.get_num_vertices()):
            for n in range(self.get_vertex_degree(source)):
                target = self.get_neighbour(source, n)
                if source < target:
                    print(source, target)

    def print_graph(self):
        for v in range(self.get_num_vertices()):
            print(v, self.get_neighbours(v))

