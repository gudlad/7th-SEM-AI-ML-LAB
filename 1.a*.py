def aStarAlgo(start_node, stop_node):
    open_set = set(start_node)
    closed_set = set()
    g = {}  # actual cost
    parents = {}
    g[start_node] = 0
    parents[start_node] = start_node

    while len(open_set) > 0:
        n = None

        for v in open_set:  # deciding the optimal neighbors n b/w A,B ie the BESTNODE
            if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v  # n = S (BESTNODE)

        if n == stop_node or Graph_nodes[n] == None:
            pass
        else:
            for (m, weight) in get_neighbors(n):
                # generate the successors of
                # BESTNODE
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m] = n
                    # g(SUCCESSOR) = g(BESTNODE)+ the cost of getting from BESTNODE to SUCCESSOR.
                    g[m] = g[n] + weight
                else:
                    # If SUCCESSOR is same as the node on OPEN, then
                    # take this OLD node to the BESTNODE and update cost value.
                    # If the successor not on OPEN, but in CLOSED, then remove it from CLOSED and add it to  OPEN
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n
                        if m in closed_set:
                            closed_set.remove(m)
                            open_set.add(m)
        if n == None:
            # If no node on OPEN, report failure.
            print('Path does not exist!')
            return None
        if n == stop_node:
            # print(n)
            # print(parents)
            path = []
            while parents[n] != n:   # B != E, S!=B
                path.append(n)       # [E, B]
                n = parents[n]
            path.append(start_node)  # S -->[E,B,S]
            path.reverse()           # [S,B,E]
            print('Path found: {}'.format(path))
            return path
        open_set.remove(n)
        closed_set.add(n)
    print('Path does not exist!')
    return None


def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None


def heuristic(n):
    H_dist = {
        'S': 5,
        'A': 4,
        'B': 5,
        'E': 0,
    }
    return H_dist[n]


Graph_nodes = {
    'S': [('A', 1), ('B', 2)],
    'A': [('E', 13)],
    'B': [('E', 5)]
}
aStarAlgo('S', 'E')
