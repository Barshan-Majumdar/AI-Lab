# ============================================================
# AO* ALGORITHM
# MINIMUM-COST AUTOMATED MODEL TRAINING PIPELINE
#
# Full solution for the graph given in the problem image
#
# Works in:
#   Google Colab
#   Jupyter Notebook
# ============================================================

import matplotlib.pyplot as plt
import networkx as nx

from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display


# ============================================================
# 1. GRAPH DEFINITION
# ============================================================
#
# OR  = choose minimum-cost alternative
# AND = all children are required
#
# PROCESS PIPELINE
#       OR
#   /         \
# DEEP       ENSEMBLE
# LEARNING   METHODS
#
# DEEP LEARNING:
#       AND
#    /       \
# FEATURE   ARCHITECTURE
#
# ENSEMBLE:
#       AND
#    /       \
# ENSEMBLE   MODEL
# CONSTRUCT. EVALUATION
# ============================================================


graph = {

    # Root:
    # PROCESS = DEEP LEARNING OR ENSEMBLE METHODS
    "PROCESS": [
        "DEEP_LEARNING",
        "ENSEMBLE_METHODS"
    ],

    # Deep Learning:
    # FEATURE EXTRACTION AND ARCHITECTURE OPTIMIZATION
    "DEEP_LEARNING": [
        [
            "FEATURE_EXTRACTION",
            "ARCHITECTURE_OPTIMIZATION"
        ]
    ],

    # Feature Extraction
    "FEATURE_EXTRACTION": [
        "CONV_EXTRACTOR"
    ],

    # Conv Extractor
    "CONV_EXTRACTOR": [
        "VISION_TRANSFORMER"
    ],

    # Architecture Optimization
    "ARCHITECTURE_OPTIMIZATION": [
        "GRID_SEARCH"
    ],

    # Grid Search
    "GRID_SEARCH": [
        "BAYESIAN_TUNING"
    ],

    # Ensemble Methods:
    # Ensemble Construction AND Model Evaluation
    "ENSEMBLE_METHODS": [
        [
            "ENSEMBLE_CONSTRUCTION",
            "MODEL_EVALUATION"
        ]
    ],

    # Ensemble Construction
    "ENSEMBLE_CONSTRUCTION": [
        "GRADIENT_BOOSTING"
    ],

    # Gradient Boosting
    "GRADIENT_BOOSTING": [
        "RANDOM_FOREST"
    ],

    # Terminal nodes
    "VISION_TRANSFORMER": [],
    "BAYESIAN_TUNING": [],
    "RANDOM_FOREST": [],
    "MODEL_EVALUATION": []
}


# ============================================================
# 2. ACTUAL COSTS (C)
# ============================================================
#
# C = actual operational cost
#
# We use the C values shown in the graph.
# ============================================================

actual_cost = {

    "CONV_EXTRACTOR": 4,
    "VISION_TRANSFORMER": 6,

    "GRID_SEARCH": 5,
    "BAYESIAN_TUNING": 3,

    "GRADIENT_BOOSTING": 4,
    "RANDOM_FOREST": 3,

    "MODEL_EVALUATION": 5
}


# ============================================================
# 3. HEURISTIC VALUES (h)
# ============================================================
#
# h = estimated remaining cost
#
# These are used to guide the search.
# ============================================================

heuristic = {

    "PROCESS": 12,

    "DEEP_LEARNING": 6,

    "FEATURE_EXTRACTION": 6,
    "CONV_EXTRACTOR": 2,
    "VISION_TRANSFORMER": 3,

    "ARCHITECTURE_OPTIMIZATION": 4,
    "GRID_SEARCH": 1,
    "BAYESIAN_TUNING": 0,

    "ENSEMBLE_METHODS": 40,

    "ENSEMBLE_CONSTRUCTION": 7,
    "GRADIENT_BOOSTING": 3,
    "RANDOM_FOREST": 2,

    "MODEL_EVALUATION": 0
}


# ============================================================
# 4. INITIAL ESTIMATED COST
# ============================================================

estimated_cost = heuristic.copy()


# ============================================================
# 5. AO* RESULT STORAGE
# ============================================================

solved = set()

states = []


# ============================================================
# 6. FUNCTION TO SAVE ANIMATION STATE
# ============================================================

def save_state(
    title,
    current=None,
    selected_edges=None,
    message=""
):

    states.append({

        "title": title,

        "current": current,

        "selected_edges": selected_edges or [],

        "message": message,

        "cost": estimated_cost.copy(),

        "solved": solved.copy()
    })


# ============================================================
# 7. CREATE VISUALIZATION GRAPH
# ============================================================

G = nx.DiGraph()


# Add all nodes
for node in graph:

    G.add_node(node)


# ============================================================
# Add edges
# ============================================================

edges = [

    # PROCESS
    ("PROCESS", "DEEP_LEARNING"),
    ("PROCESS", "ENSEMBLE_METHODS"),

    # DEEP LEARNING
    ("DEEP_LEARNING", "FEATURE_EXTRACTION"),
    ("DEEP_LEARNING", "ARCHITECTURE_OPTIMIZATION"),

    # FEATURE EXTRACTION
    ("FEATURE_EXTRACTION", "CONV_EXTRACTOR"),
    ("CONV_EXTRACTOR", "VISION_TRANSFORMER"),

    # ARCHITECTURE
    ("ARCHITECTURE_OPTIMIZATION", "GRID_SEARCH"),
    ("GRID_SEARCH", "BAYESIAN_TUNING"),

    # ENSEMBLE
    ("ENSEMBLE_METHODS", "ENSEMBLE_CONSTRUCTION"),
    ("ENSEMBLE_METHODS", "MODEL_EVALUATION"),

    # ENSEMBLE CONSTRUCTION
    ("ENSEMBLE_CONSTRUCTION", "GRADIENT_BOOSTING"),
    ("GRADIENT_BOOSTING", "RANDOM_FOREST")
]


G.add_edges_from(edges)


# ============================================================
# 8. NODE POSITIONS
# ============================================================

pos = {

    # Root
    "PROCESS": (0, 6),

    # Two strategies
    "DEEP_LEARNING": (-4, 5),
    "ENSEMBLE_METHODS": (4, 5),

    # Deep Learning branches
    "FEATURE_EXTRACTION": (-6, 4),
    "ARCHITECTURE_OPTIMIZATION": (-2, 4),

    # Feature extraction
    "CONV_EXTRACTOR": (-6, 3),
    "VISION_TRANSFORMER": (-6, 2),

    # Architecture optimization
    "GRID_SEARCH": (-2, 3),
    "BAYESIAN_TUNING": (-2, 2),

    # Ensemble
    "ENSEMBLE_CONSTRUCTION": (3, 4),
    "MODEL_EVALUATION": (6, 4),

    # Ensemble construction
    "GRADIENT_BOOSTING": (3, 3),
    "RANDOM_FOREST": (3, 2)
}


# ============================================================
# 9. EDGE RELATIONSHIPS
# ============================================================

edge_relations = {

    ("PROCESS", "DEEP_LEARNING"): "OR",
    ("PROCESS", "ENSEMBLE_METHODS"): "OR",

    ("DEEP_LEARNING", "FEATURE_EXTRACTION"): "AND",
    ("DEEP_LEARNING", "ARCHITECTURE_OPTIMIZATION"): "AND",

    ("FEATURE_EXTRACTION", "CONV_EXTRACTOR"): "SEQ",
    ("CONV_EXTRACTOR", "VISION_TRANSFORMER"): "SEQ",

    ("ARCHITECTURE_OPTIMIZATION", "GRID_SEARCH"): "SEQ",
    ("GRID_SEARCH", "BAYESIAN_TUNING"): "SEQ",

    ("ENSEMBLE_METHODS", "ENSEMBLE_CONSTRUCTION"): "AND",
    ("ENSEMBLE_METHODS", "MODEL_EVALUATION"): "AND",

    ("ENSEMBLE_CONSTRUCTION", "GRADIENT_BOOSTING"): "SEQ",
    ("GRADIENT_BOOSTING", "RANDOM_FOREST"): "SEQ"
}


# ============================================================
# 10. FUNCTION TO CALCULATE TERMINAL PATH COST
# ============================================================

def calculate_chain_cost(node):

    """
    Calculate actual cost of a sequential chain.
    """

    if node not in graph or len(graph[node]) == 0:

        return actual_cost.get(node, 0)


    option = graph[node][0]


    if isinstance(option, list):

        return sum(
            calculate_chain_cost(child)
            for child in option
        )


    else:

        return (
            actual_cost.get(node, 0)
            + calculate_chain_cost(option)
        )


# ============================================================
# 11. CALCULATE DEEP LEARNING
# ============================================================

feature_extraction_cost = (
    actual_cost["CONV_EXTRACTOR"]
    + actual_cost["VISION_TRANSFORMER"]
)

architecture_cost = (
    actual_cost["GRID_SEARCH"]
    + actual_cost["BAYESIAN_TUNING"]
)

deep_learning_cost = (
    feature_extraction_cost
    + architecture_cost
)


# ============================================================
# 12. CALCULATE ENSEMBLE
# ============================================================

ensemble_construction_cost = (
    actual_cost["GRADIENT_BOOSTING"]
    + actual_cost["RANDOM_FOREST"]
)

model_evaluation_cost = (
    actual_cost["MODEL_EVALUATION"]
)

ensemble_cost = (
    ensemble_construction_cost
    + model_evaluation_cost
)


# ============================================================
# 13. ANIMATION STATES
# ============================================================


# ------------------------------------------------------------
# STEP 1
# ------------------------------------------------------------

save_state(

    title="STEP 1 — Start AO*",

    current="PROCESS",

    message=(
        "Start from root: PROCESS PIPELINE"
    )
)


# ------------------------------------------------------------
# STEP 2
# ------------------------------------------------------------

save_state(

    title="STEP 2 — Expand Root",

    current="PROCESS",

    message=(
        "PROCESS has two OR choices: "
        "DEEP LEARNING or ENSEMBLE METHODS"
    )
)


# ------------------------------------------------------------
# STEP 3
# ------------------------------------------------------------

save_state(

    title="STEP 3 — Evaluate Deep Learning",

    current="DEEP_LEARNING",

    message=(
        "Deep Learning requires FEATURE EXTRACTION "
        "AND ARCHITECTURE OPTIMIZATION"
    )
)


# ------------------------------------------------------------
# STEP 4
# ------------------------------------------------------------

estimated_cost["FEATURE_EXTRACTION"] = (
    actual_cost["CONV_EXTRACTOR"]
    + actual_cost["VISION_TRANSFORMER"]
)

save_state(

    title="STEP 4 — Feature Extraction",

    current="FEATURE_EXTRACTION",

    selected_edges=[
        ("FEATURE_EXTRACTION", "CONV_EXTRACTOR"),
        ("CONV_EXTRACTOR", "VISION_TRANSFORMER")
    ],

    message=(
        "Feature Extraction = 4 + 6 = 10"
    )
)


# ------------------------------------------------------------
# STEP 5
# ------------------------------------------------------------

estimated_cost["ARCHITECTURE_OPTIMIZATION"] = (
    actual_cost["GRID_SEARCH"]
    + actual_cost["BAYESIAN_TUNING"]
)

save_state(

    title="STEP 5 — Architecture Optimization",

    current="ARCHITECTURE_OPTIMIZATION",

    selected_edges=[
        ("ARCHITECTURE_OPTIMIZATION", "GRID_SEARCH"),
        ("GRID_SEARCH", "BAYESIAN_TUNING")
    ],

    message=(
        "Architecture Optimization = 5 + 3 = 8"
    )
)


# ------------------------------------------------------------
# STEP 6
# ------------------------------------------------------------

estimated_cost["DEEP_LEARNING"] = deep_learning_cost

solved.update([
    "VISION_TRANSFORMER",
    "BAYESIAN_TUNING",
    "FEATURE_EXTRACTION",
    "ARCHITECTURE_OPTIMIZATION",
    "DEEP_LEARNING"
])

save_state(

    title="STEP 6 — Deep Learning Cost",

    current="DEEP_LEARNING",

    message=(
        "Deep Learning = Feature Extraction + "
        "Architecture Optimization = 10 + 8 = 18"
    )
)


# ------------------------------------------------------------
# STEP 7
# ------------------------------------------------------------

save_state(

    title="STEP 7 — Evaluate Ensemble Methods",

    current="ENSEMBLE_METHODS",

    message=(
        "Ensemble Methods requires "
        "Ensemble Construction AND Model Evaluation"
    )
)


# ------------------------------------------------------------
# STEP 8
# ------------------------------------------------------------

estimated_cost["ENSEMBLE_CONSTRUCTION"] = (
    actual_cost["GRADIENT_BOOSTING"]
    + actual_cost["RANDOM_FOREST"]
)

save_state(

    title="STEP 8 — Ensemble Construction",

    current="ENSEMBLE_CONSTRUCTION",

    selected_edges=[
        ("ENSEMBLE_CONSTRUCTION", "GRADIENT_BOOSTING"),
        ("GRADIENT_BOOSTING", "RANDOM_FOREST")
    ],

    message=(
        "Ensemble Construction = 4 + 3 = 7"
    )
)


# ------------------------------------------------------------
# STEP 9
# ------------------------------------------------------------

estimated_cost["MODEL_EVALUATION"] = (
    actual_cost["MODEL_EVALUATION"]
)

save_state(

    title="STEP 9 — Model Evaluation",

    current="MODEL_EVALUATION",

    message=(
        "Model Evaluation = 5"
    )
)


# ------------------------------------------------------------
# STEP 10
# ------------------------------------------------------------

estimated_cost["ENSEMBLE_METHODS"] = ensemble_cost

solved.update([
    "RANDOM_FOREST",
    "GRADIENT_BOOSTING",
    "ENSEMBLE_CONSTRUCTION",
    "MODEL_EVALUATION",
    "ENSEMBLE_METHODS"
])

save_state(

    title="STEP 10 — Ensemble Methods Cost",

    current="ENSEMBLE_METHODS",

    message=(
        "Ensemble = 7 + 5 = 12"
    )
)


# ------------------------------------------------------------
# STEP 11
# ------------------------------------------------------------

save_state(

    title="STEP 11 — Compare OR Choices",

    current="PROCESS",

    message=(
        f"Deep Learning = {deep_learning_cost}    "
        f"vs    Ensemble Methods = {ensemble_cost}"
    )
)


# ============================================================
# 14. FINAL AO* DECISION
# ============================================================

if deep_learning_cost < ensemble_cost:

    optimal_cost = deep_learning_cost

    optimal_strategy = "DEEP LEARNING"

    optimal_edges = [

        ("PROCESS", "DEEP_LEARNING"),

        ("DEEP_LEARNING", "FEATURE_EXTRACTION"),
        ("DEEP_LEARNING", "ARCHITECTURE_OPTIMIZATION"),

        ("FEATURE_EXTRACTION", "CONV_EXTRACTOR"),
        ("CONV_EXTRACTOR", "VISION_TRANSFORMER"),

        ("ARCHITECTURE_OPTIMIZATION", "GRID_SEARCH"),
        ("GRID_SEARCH", "BAYESIAN_TUNING")
    ]

    optimal_sequence = (
        "Process → Deep Learning → "
        "Feature Extraction → Conv Extractor → "
        "Vision Transformer"
    )


else:

    optimal_cost = ensemble_cost

    optimal_strategy = "ENSEMBLE METHODS"

    optimal_edges = [

        ("PROCESS", "ENSEMBLE_METHODS"),

        ("ENSEMBLE_METHODS", "ENSEMBLE_CONSTRUCTION"),
        ("ENSEMBLE_METHODS", "MODEL_EVALUATION"),

        ("ENSEMBLE_CONSTRUCTION", "GRADIENT_BOOSTING"),
        ("GRADIENT_BOOSTING", "RANDOM_FOREST")
    ]

    optimal_sequence = (
        "Process → Ensemble Methods → "
        "Ensemble Construction → "
        "Gradient Boosting → Random Forest"
        " + Model Evaluation"
    )


# ------------------------------------------------------------
# FINAL STATE
# ------------------------------------------------------------

estimated_cost["PROCESS"] = optimal_cost

solved.add("PROCESS")

save_state(

    title="STEP 12 — 🏆 OPTIMAL SOLUTION FOUND",

    current="PROCESS",

    selected_edges=optimal_edges,

    message=(
        f"Optimal Strategy = {optimal_strategy}    |    "
        f"Total Cost = {optimal_cost}"
    )
)


# ============================================================
# 15. DRAWING / ANIMATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(15, 9)
)


def update(frame):

    ax.clear()

    state = states[frame]

    current = state["current"]

    selected_edges = state["selected_edges"]


    # ========================================================
    # NODE COLORS
    # ========================================================

    node_colors = []

    for node in G.nodes:

        if node == current:

            node_colors.append("orange")

        elif node in state["solved"]:

            node_colors.append("lightgreen")

        else:

            node_colors.append("lightblue")


    # ========================================================
    # DRAW NODES
    # ========================================================

    nx.draw_networkx_nodes(

        G,

        pos,

        node_color=node_colors,

        node_size=2200,

        edgecolors="black",

        linewidths=2,

        ax=ax
    )


    # ========================================================
    # EDGE COLORS
    # ========================================================

    edge_colors = []

    edge_widths = []

    for edge in G.edges:

        if edge in selected_edges:

            edge_colors.append("red")

            edge_widths.append(5)

        else:

            edge_colors.append("gray")

            edge_widths.append(2)


    # ========================================================
    # DRAW EDGES
    # ========================================================

    nx.draw_networkx_edges(

        G,

        pos,

        edge_color=edge_colors,

        width=edge_widths,

        arrows=True,

        arrowsize=20,

        ax=ax
    )


    # ========================================================
    # NODE LABELS
    # ========================================================

    labels = {}

    for node in G.nodes:

        if node in actual_cost:

            c = actual_cost[node]

        else:

            c = "-"


        h = heuristic.get(node, "-")

        current_cost = state["cost"].get(
            node,
            h
        )


        # Shorter names for display

        display_name = {

            "PROCESS": "PROCESS\nPIPELINE",

            "DEEP_LEARNING": "DEEP\nLEARNING",

            "FEATURE_EXTRACTION": "FEATURE\nEXTRACTION",

            "CONV_EXTRACTOR": "CONV\nEXTRACTOR",

            "VISION_TRANSFORMER": "VISION\nTRANSFORMER",

            "ARCHITECTURE_OPTIMIZATION": "ARCHITECTURE\nOPTIMIZATION",

            "GRID_SEARCH": "GRID\nSEARCH",

            "BAYESIAN_TUNING": "BAYESIAN\nTUNING",

            "ENSEMBLE_METHODS": "ENSEMBLE\nMETHODS",

            "ENSEMBLE_CONSTRUCTION": "ENSEMBLE\nCONSTRUCTION",

            "GRADIENT_BOOSTING": "GRADIENT\nBOOSTING",

            "RANDOM_FOREST": "RANDOM\nFOREST",

            "MODEL_EVALUATION": "MODEL\nEVALUATION"
        }.get(node, node)


        labels[node] = (

            f"{display_name}\n"

            f"C={c}  h={h}\n"

            f"f={current_cost}"
        )


    nx.draw_networkx_labels(

        G,

        pos,

        labels=labels,

        font_size=8,

        font_weight="bold",

        ax=ax
    )


    # ========================================================
    # AND / OR LABELS
    # ========================================================

    ax.text(

        -4.0,
        5.35,

        "OR",

        fontsize=13,

        fontweight="bold"
    )

    ax.text(

        3.8,
        5.35,

        "OR",

        fontsize=13,

        fontweight="bold"
    )


    ax.text(

        -4.0,
        4.35,

        "AND",

        fontsize=13,

        fontweight="bold"
    )


    ax.text(

        4.5,
        4.35,

        "AND",

        fontsize=13,

        fontweight="bold"
    )


    # ========================================================
    # TITLE
    # ========================================================

    ax.set_title(

        state["title"],

        fontsize=20,

        fontweight="bold",

        pad=20
    )


    # ========================================================
    # EXPLANATION
    # ========================================================

    ax.text(

        0.5,

        -0.03,

        state["message"],

        transform=ax.transAxes,

        ha="center",

        fontsize=13,

        fontweight="bold"
    )


    # ========================================================
    # COST PANEL
    # ========================================================

    cost_panel = (

        "CURRENT COSTS\n\n"

        f"Deep Learning : {state['cost'].get('DEEP_LEARNING', '-')}\n"

        f"Ensemble      : {state['cost'].get('ENSEMBLE_METHODS', '-')}\n"

        f"Process       : {state['cost'].get('PROCESS', '-')}\n\n"

        "RULES\n"

        "OR  → MIN\n"

        "AND → SUM"
    )


    ax.text(

        1.02,

        0.75,

        cost_panel,

        transform=ax.transAxes,

        fontsize=11,

        verticalalignment="top",

        bbox=dict(

            boxstyle="round",

            facecolor="white",

            edgecolor="black"
        )
    )


    # ========================================================
    # FINAL SOLUTION
    # ========================================================

    if frame == len(states) - 1:

        final_text = (

            "🏆 FINAL SOLUTION\n\n"

            f"Strategy:\n{optimal_strategy}\n\n"

            f"Total Cost: {optimal_cost}\n\n"

            "Optimal Subgraph:\n"

            "PROCESS\n"

            "   ↓\n"

            "ENSEMBLE METHODS\n"

            "   ↓\n"

            "ENSEMBLE CONSTRUCTION\n"

            "   ↓\n"

            "GRADIENT BOOSTING\n"

            "   ↓\n"

            "RANDOM FOREST\n\n"

            "+ MODEL EVALUATION"
        )


        ax.text(

            1.02,

            0.35,

            final_text,

            transform=ax.transAxes,

            fontsize=10,

            verticalalignment="top",

            bbox=dict(

                boxstyle="round",

                facecolor="white",

                edgecolor="red",

                linewidth=2
            )
        )


    ax.axis("off")


# ============================================================
# 16. CREATE ANIMATION
# ============================================================

animation = FuncAnimation(

    fig,

    update,

    frames=len(states),

    interval=1800,

    repeat=False
)


# ============================================================
# 17. DISPLAY ANIMATION
# ============================================================

plt.close(fig)

display(
    HTML(
        animation.to_jshtml()
    )
)


# ============================================================
# 18. PRINT FINAL RESULT
# ============================================================

print("\n")
print("=" * 70)
print("                    AO* FINAL RESULT")
print("=" * 70)

print()

print(
    "Deep Learning Cost     :",
    deep_learning_cost
)

print(
    "Ensemble Methods Cost  :",
    ensemble_cost
)

print()

print(
    "Optimal Strategy       :",
    optimal_strategy
)

print(
    "Optimal Cost           :",
    optimal_cost
)

print()

print(
    "Optimal Sequence       :"
)

print(
    optimal_sequence
)

print()

print("=" * 70)

print(
    "AO* DECISION:"
)

print(
    f"min({deep_learning_cost}, {ensemble_cost}) "
    f"= {optimal_cost}"
)

print()

print(
    "OR  → Choose minimum cost"
)

print(
    "AND → Add all required costs"
)

print()

print("=" * 70)