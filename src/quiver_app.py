import matplotlib
matplotlib.use("TkAgg")  # Use the TkAgg backend
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import ast
import networkx as nx

# Helper functions
def labels_to_mat(labels):
    n = len(labels)  # Number of labels (should match the dimension of the matrix)
    gamma = [sp.Symbol(f'γ_{i}') for i in range(1, n + 1)]
    coefficient_matrix = sp.zeros(n)

    for i in range(n):
        for j in range(n):
            coefficient_matrix[i, j] = sp.expand(labels[i]).coeff(gamma[j])

    return np.array(coefficient_matrix)
def mat_to_labels(gamma_matrix):
    n = len(gamma_matrix)
    gamma = [sp.Symbol(f'γ_{i}') for i in range(1, n + 1)]
    labels = []

    for i in range(n):
        linear_combination = sum(gamma_matrix[i, j] * gamma[j] for j in range(n))
        labels.append(linear_combination)

    return labels
def node_topology(quiver_matrix):
    num_nodes = quiver_matrix.shape[0]

    # Function to compute a stable canonical representation for a node
    def compute_canonical_representation(node):
        # Get incoming and outgoing edges
        incoming = [i for i in range(num_nodes) if quiver_matrix[i, node] != 0]
        outgoing = [i for i in range(num_nodes) if quiver_matrix[node, i] != 0]

        # Combine neighbors and sort consistently
        combined_neighbors = tuple(sorted(incoming) + sorted(outgoing))
        return combined_neighbors

    # Compute canonical representations for all nodes
    canonical_representations = [
        compute_canonical_representation(node) for node in range(num_nodes)
    ]

    # Group nodes by their canonical representations
    unique_representations = list(set(canonical_representations))
    automorphic_classes = {chr(97 + i): [] for i in range(len(unique_representations))}

    for node, representation in enumerate(canonical_representations):
        class_label = chr(97 + unique_representations.index(representation))
        automorphic_classes[class_label].append(node)

    return automorphic_classes
def plot_quiver(quiver_matrix, ax):
    num_cols = quiver_matrix.shape[0]
    adj_matrix = quiver_matrix[:, :num_cols]
    gamma_matrix = quiver_matrix[:, num_cols:]

    G = nx.DiGraph()
    node_labels = mat_to_labels(gamma_matrix)
    node_labels = [sp.pretty(node_labels[i], use_unicode=True) for i in range(len(node_labels))]

    for i, label in enumerate(node_labels):
        G.add_node(i, label=str(label))

    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i, j] != 0:
                G.add_edge(i, j, weight=adj_matrix[i, j])

    labels = {i: str(label) for i, label in enumerate(node_labels)}
    pos = nx.circular_layout(G)

    ax.clear()
    nx.draw(
        G,
        pos=pos,
        labels=labels,
        with_labels=True,
        node_color='lightblue',
        edgecolors='black',
        node_size=2000,
        font_size=10,
        font_weight='bold',
        arrows=True,
        linewidths=2,
        ax=ax
    )

    edge_labels = {(i, j): adj_matrix[i, j] for i in range(len(adj_matrix)) for j in range(len(adj_matrix[i])) if
                   adj_matrix[i, j] > 1}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
def mutate_quiver(quiver_matrix, node):
    import numpy as np

    num_cols = quiver_matrix.shape[0]
    adj_matrix = quiver_matrix[:, :num_cols]
    gamma_matrix = quiver_matrix[:, num_cols:]
    n = len(adj_matrix)

    # Update labels (gamma vector)
    labels = mat_to_labels(gamma_matrix)
    mutated_label = labels[node]
    labels[node] = -mutated_label

    for i in range(n):
        if i == node:
            continue
        inner_product = adj_matrix[node, i] - adj_matrix[i, node]
        if inner_product > 0:
            labels[i] += inner_product * mutated_label

    # Process two-paths
    endpoints = [i for i in range(n) if adj_matrix[i, node] > 0]
    startpoints = [i for i in range(n) if adj_matrix[node, i] > 0]
    for end in endpoints:
        for start in startpoints:
            adj_matrix[end, start] += adj_matrix[end, node] * adj_matrix[node, start]

    # Reverse arrows connected to the mutated node
    for i in range(n):
        adj_matrix[node, i], adj_matrix[i, node] = adj_matrix[i, node], adj_matrix[node, i]

    # Cancel double arrows
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j] > 0 and adj_matrix[j, i] > 0:
                net_arrows = adj_matrix[i, j] - adj_matrix[j, i]
                adj_matrix[i, j] = max(0, net_arrows)
                adj_matrix[j, i] = max(0, -net_arrows)

    # Reconstruct gamma matrix and return updated quiver
    gamma_matrix = labels_to_mat(labels)
    return np.hstack((adj_matrix, gamma_matrix))
class QuiverApp:
    def __init__(self, quiver_matrix):
        self.quiver_matrix = quiver_matrix
        self.num_cols = quiver_matrix.shape[0]
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.update_plot()

        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def update_plot(self):
        plot_quiver(self.quiver_matrix, self.ax)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        """Handle node clicks to mutate."""
        if event.inaxes != self.ax:
            return

        # Create a dummy graph for layout calculation
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_cols))
        pos = nx.circular_layout(G)

        # Determine the clicked node based on position
        click_point = np.array([event.xdata, event.ydata])
        distances = {
            node: np.linalg.norm(click_point - pos[node])
            for node in pos
        }
        clicked_node = min(distances, key=distances.get)

        print(f"Clicked on node: {clicked_node}")

        # Mutate the quiver
        self.quiver_matrix = mutate_quiver(self.quiver_matrix, clicked_node)

        # Update the plot
        self.update_plot()
def main():
    # Welcome message
    def print_centered(text, width=50, style=None):
        if style == 'bold':
            start, end = '\033[1m', '\033[0m'  # Bold
        elif style == 'italic':
            start, end = '\033[3m', '\033[0m'  # Italic
        else:
            start, end = '', ''
        print(start + text.center(width) + end)

    divider = '=' * 60  # Adjust width as needed

    print_centered(divider)
    print_centered("BPS Spectroscopy with Reinforcement Learning", style='bold')
    print_centered("Interactive Quiver App", style='bold')
    print_centered("arXiv:2501.14863")
    print_centered("Federico Carta, Asa Gauntlett, Finley Griffin, Yang-Hui He", style='italic')
    print_centered(divider)

    matrix_input = input("Enter adjacency matrix in array form [[],[],...]: ")
    adj_matrix = np.asarray(ast.literal_eval(matrix_input), dtype=int)
    initial_quiver = np.hstack((adj_matrix, np.eye(len(adj_matrix), dtype=int)))

    app = QuiverApp(initial_quiver)
    print("QuiverApp instance created.")  # Debugging message
    plt.show()  # Blocking show, keeps the plot open
    print("Plot displayed.")  # Debugging message

if __name__ == "__main__":
    main()

