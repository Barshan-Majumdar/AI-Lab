# AO* ALGORITHM
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
graph = {
    'A': ['B', ['C', 'D']],
    'B': ['E', 'F'],
    'C': [],
    'D': [],
    'E': [],
    'F': []
}
heuristic = {
    'A': 10,
    'B': 6,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 2
}
G = nx.DiGraph()
for node in graph:
    G.add_node(node)
G.add_edge('A', 'B', relation='OR')
G.add_edge('A', 'C', relation='AND')
G.add_edge('A', 'D', relation='AND')

G.add_edge('B', 'E', relation='OR')
G.add_edge('B', 'F', relation='OR')
pos = {
    'A': (0, 3),

    'B': (-2, 2),

    'C': (1, 2),
    'D': (3, 2),

    'E': (-3, 1),
    'F': (-1, 1)
}
solved = set()

cost = heuristic.copy()

states = []
def add_state(title, current=None, selected=None, message=""):

    states.append({
        "title": title,
        "current": current,
        "selected": selected,
        "message": message,
        "cost": cost.copy(),
        "solved": solved.copy()
    })
add_state(
    title="STEP 1: Start AO*",
    current='A',
    message="Start from root node A"
)

add_state(
    title="STEP 2: Expand A",
    current='A',
    message="A has two choices: B OR (C AND D)"
)

add_state(
    title="STEP 3: Expand B",
    current='B',
    message="B has two OR choices: E or F"
)
add_state(
    title="STEP 4: Compare E and F",
    current='B',
    message="E = 4 and F = 2 → Choose F"
)
cost['B'] = min(cost['E'], cost['F'])

solved.add('E')
solved.add('F')
solved.add('B')


add_state(
    title="STEP 5: B is SOLVED",
    current='B',
    selected='F',
    message="Cost(B) = min(4, 2) = 2"
)
cost['C'] = 2
cost['D'] = 3

and_cost = cost['C'] + cost['D']

solved.add('C')
solved.add('D')


add_state(
    title="STEP 6: Calculate C AND D",
    current='A',
    selected=['C', 'D'],
    message="AND cost = C + D = 2 + 3 = 5"
)
option1 = cost['B']
option2 = and_cost


add_state(
    title="STEP 7: Compare A's Solutions",
    current='A',
    message=f"A → B = {option1}     |     A → C AND D = {option2}"
)
if option1 < option2:

    cost['A'] = option1

    solved.add('A')

    final_solution = ['A', 'B', 'F']

    add_state(
        title="STEP 8: OPTIMAL SOLUTION FOUND",
        current='A',
        selected='F',
        message="Minimum cost = 2 → Optimal solution: A → B → F"
    )

else:

    cost['A'] = option2

    solved.add('A')

    final_solution = ['A', 'C', 'D']

    add_state(
        title="STEP 8: OPTIMAL SOLUTION FOUND",
        current='A',
        selected=['C', 'D'],
        message="Minimum cost = 5 → Optimal solution: A → C AND D"
    )
fig, ax = plt.subplots(figsize=(12, 7))
def update(frame):

    ax.clear()

    state = states[frame]

    current = state["current"]

    selected = state["selected"]
    node_colors = []

    for node in G.nodes:
        if node in state["solved"]:
            node_colors.append("lightgreen")

        # Current node
        elif node == current:
            node_colors.append("orange")

        # Normal nodes
        else:
            node_colors.append("lightblue")
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=2000,
        edgecolors="black",
        linewidths=2,
        ax=ax
    )
    edge_colors = []

    for u, v in G.edges:

        selected_edge = False
        if frame == len(states) - 1:

            if (u == 'A' and v == 'B'):

                selected_edge = True

            if (u == 'B' and v == 'F'):

                selected_edge = True
        if selected == 'F':

            if u == 'B' and v == 'F':

                selected_edge = True
        if isinstance(selected, list):

            if u == 'A' and v in selected:

                selected_edge = True


        if selected_edge:

            edge_colors.append("red")

        else:

            edge_colors.append("gray")
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        width=3,
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3",
        ax=ax
    )
    labels = {}

    for node in G.nodes:

        labels[node] = (
            f"{node}\n"
            f"h = {state['cost'][node]}"
        )


    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=12,
        font_weight="bold",
        ax=ax
    )
    edge_labels = {

        ('A', 'B'): 'OR',

        ('A', 'C'): 'AND',

        ('A', 'D'): 'AND',

        ('B', 'E'): 'OR',

        ('B', 'F'): 'OR'
    }


    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=10,
        ax=ax
    )
    ax.set_title(
        state["title"],
        fontsize=20,
        fontweight="bold"
    )
    ax.text(
        0.5,
        -0.08,
        state["message"],
        transform=ax.transAxes,
        ha="center",
        fontsize=14,
        fontweight="bold"
    )
    cost_text = (

        "CURRENT COSTS\n\n"

        f"A = {state['cost']['A']}\n"

        f"B = {state['cost']['B']}\n"

        f"C = {state['cost']['C']}\n"

        f"D = {state['cost']['D']}\n"

        f"E = {state['cost']['E']}\n"

        f"F = {state['cost']['F']}"
    )


    ax.text(
        1.02,
        0.70,
        cost_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black"
        )
    )
    solved_text = (
        "SOLVED:\n"
        + ", ".join(sorted(state["solved"]))
    )


    ax.text(
        1.02,
        0.25,
        solved_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black"
        )
    )
    ax.axis("off")

animation = FuncAnimation(
    fig,
    update,
    frames=len(states),
    interval=2000,
    repeat=False
)
plt.close(fig)

display(
    HTML(
        animation.to_jshtml()
    )
)

print("\n")
print("=" * 50)
print("              AO* FINAL RESULT")
print("=" * 50)

print()

print("Optimal Cost:", cost['A'])

print(
    "Optimal Solution:",
    " → ".join(final_solution)
)

print(
    "Solved Nodes:",
    solved
)

print()

print("=" * 50)