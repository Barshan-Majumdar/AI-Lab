import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

G = nx.Graph()
edges = [
    ('A', 'B'), ('A', 'C'),
    ('B', 'D'), ('B', 'E'),
    ('C', 'F'), ('C', 'G'),
    ('E', 'H'),
]
G.add_edges_from(edges)

start = 'A'
visited = []
queue = deque([start])

while queue:
    node = queue.popleft()
    if node not in visited:
        visited.append(node)
        queue.extend(sorted(set(G.neighbors(node)) - set(visited)))

print("BFS Traversal:", visited)

pos = nx.spring_layout(G, seed=42)
node_colors = ['red' if node in visited else 'lightblue' for node in G.nodes()]
nx.draw(G, pos,
        with_labels=True,
        node_size=1000,
        node_color=node_colors,
        font_size=12)
plt.title("Breadth First Search")
plt.show()
