import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
import matplotlib.pyplot as plt
import csv
import numpy as np
import sympy as sp
import gymnasium as gym
import pandas as pd
import copy
import ast
import torch as th
from gymnasium.utils import seeding
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from colorama import Fore, Back, Style, init
init(autoreset=True)

# Quiver and chamber functions

def labels_to_mat(labels):
    n = len(labels)  # Number of labels (should match the dimension of the matrix)

    # Create symbolic variables \gamma_1, \gamma_2, ..., \gamma_n
    gamma = [sp.Symbol(f'γ_{i}') for i in range(1, n + 1)]

    # Initialize an empty matrix to store the coefficients
    coefficient_matrix = sp.zeros(n)

    # Loop through each label (which is a linear combination of the gamma symbols)
    for i in range(n):
        for j in range(n):
            # Extract the coefficient of \gamma_j from the i-th label
            coefficient_matrix[i, j] = sp.expand(labels[i]).coeff(gamma[j])

    # Convert to np
    np_matrix = np.array(coefficient_matrix)

    return np_matrix
def mat_to_labels(gamma_matrix):
    n = len(gamma_matrix)

    # Create symbolic variables \gamma_1, \gamma_2, ..., \gamma_n
    gamma = [sp.Symbol(f'γ_{i}') for i in range(1, n + 1)]

    # Initialize an empty tuple to store the linear combinations
    labels = []

    # Loop through the rows of the matrix to create linear combinations
    for i in range(n):
        linear_combination = sum(gamma_matrix[i, j] * gamma[j] for j in range(n))
        labels.append(linear_combination)

    # Convert the list to a tuple and return
    return labels
def all_automorphisms(quiver_matrix):
    # Extract adj matrix
    num_cols = quiver_matrix.shape[0]
    adj_matrix = quiver_matrix[:, :num_cols]

    # Create a directed graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # Create a graph matcher object, which can find automorphisms
    matcher = GraphMatcher(G, G)

    # Find and return all isomorphisms (automorphisms in this case)
    automorphisms = list(matcher.isomorphisms_iter())
    return automorphisms
def all_chambers(history: list, automorphisms: list) -> list:
    # Ensure history is a NumPy array
    history = np.array(history)

    # Store debugging info and resulting chambers
    substitution_results = []  # To maintain the order of substitutions and results

    # Generate all unique chambers
    unique_histories = set()
    for automorphism in automorphisms:
        # Apply this automorphism rule globally to each row of the history
        permuted_history = []
        for row in history:
            # Apply the automorphism to reorder the elements in each row
            new_row = [row[automorphism[i]] if i in automorphism else row[i] for i in range(len(row))]
            permuted_history.append(tuple(new_row))

        # Add debugging info and check uniqueness
        if tuple(permuted_history) not in unique_histories:
            substitution_results.append((automorphism, permuted_history))
            unique_histories.add(tuple(permuted_history))

    # Convert unique histories back to finite chambers in the order of the substitutions
    finite_chambers = []
    for substitution_rule, permuted_history in substitution_results:
        # Convert the permuted history to a finite chamber (mock function, replace with actual conversion)
        finite_chamber = history_to_finite_chamber(np.array(permuted_history))
        finite_chambers.append(finite_chamber)

    return finite_chambers
def plot_quiver(quiver_matrix):
    # Extract adjacency and gamma matrices
    num_cols = quiver_matrix.shape[0]
    adj_matrix = quiver_matrix[:, :num_cols]
    gamma_matrix = quiver_matrix[:, num_cols:]

    # Create a directed graph
    G = nx.DiGraph()
    node_labels = mat_to_labels(gamma_matrix)
    node_labels = [sp.pretty(node_labels[i], use_unicode=True) for i in range(len(node_labels))]

    # Add nodes with labels
    for i, label in enumerate(node_labels):
        G.add_node(i, label=str(label))

    # Add edges based on the adjacency matrix
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i, j] != 0:
                G.add_edge(i, j, weight=adj_matrix[i, j])

    # Get node labels for the plot
    labels = {i: str(label) for i, label in enumerate(node_labels)}

    # Define layout
    pos = nx.circular_layout(G)

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos=pos,
        labels=labels,
        with_labels=True,
        node_color='lightsteelblue',  # Updated to use colors based on topological classes
        edgecolors='black',  # Outline color: black
        node_size=2000,  # Size of the nodes
        font_size=10,
        font_weight='bold',
        arrows=True,
        linewidths=2  # Thickness of node outlines
    )

    # Draw edge labels (for values > 1)
    edge_labels = {(i, j): adj_matrix[i, j] for i in range(len(adj_matrix)) for j in range(len(adj_matrix[i])) if
                   adj_matrix[i, j] > 1}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Get figure object
    plt.show()
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
def mutate_sequence(quiver_matrix,order):
    rewards = [0]
    quivers = [quiver_matrix]
    final_quiver = antiparticle_quiver(quiver_matrix)

    n = len(order)
    for i in range(n):
        quiver_matrix = mutate_quiver(quiver_matrix,order[i])
        reward = calculate_reward(quiver_matrix,final_quiver)
        rewards.append(reward)
        quivers.append(quiver_matrix)

    return quivers, rewards
def history_to_finite_chamber(history):
    # Ensure the history is a NumPy array for consistent handling
    history = np.array(history)

    finite_chamber = []

    # Total number of nodes
    n = history.shape[1]  # Number of columns in the history (one per node)

    # Create symbolic gamma terms for each node
    gamma = [
        sp.Symbol(f"gamma_{i + 1}", real=True)  # Use indices 1-based for readability
        for i in range(n)
    ]

    # Compute the symbolic finite chamber representation
    for hyper in history:
        finite_chamber.append(np.dot(hyper, gamma))

    return finite_chamber
def topological_chamber(chamber):
    def remove_subscripts(symbol):
        # Convert the symbol to string, strip numeric subscripts, and return as a new symbol
        symbol_str = str(symbol)
        cleaned_str = "_".join(part for part in symbol_str.split("_") if not part.isdigit())
        return sp.Symbol(cleaned_str, real=True)

    cleaned_chamber = []
    for expr in chamber:
        # Replace all symbols in the expression with their cleaned versions
        cleaned_expr = expr.subs({
            symbol: remove_subscripts(symbol)
            for symbol in expr.free_symbols
        })
        cleaned_chamber.append(cleaned_expr)

    return cleaned_chamber
def calculate_reward(quiver_matrix, final_quiver):
    # Ensure quiver_matrix is an np.ndarray
    if isinstance(quiver_matrix, tuple):
        quiver_matrix = np.array(quiver_matrix)

    # Extract adjacency and gamma matrices
    num_cols = quiver_matrix.shape[0]
    adj_matrix = quiver_matrix[:, :num_cols]
    anti_adj_matrix = final_quiver[:, :num_cols]
    gamma_matrix = quiver_matrix[:, num_cols:]

    # Convert
    adj_matrix = np.array(adj_matrix, dtype=np.int32)
    anti_adj_matrix = np.array(anti_adj_matrix, dtype=np.int32)

    # Calculate mean square distance in labels
    square_label_distance = np.sum((np.sum(np.abs(gamma_matrix), axis=1)-1) ** 2)
    max_distance = num_cols*(num_cols-1)**2
    if square_label_distance > max_distance:
        return -max_reward
    label_distance_ratio = square_label_distance/max_distance
    correct_indices = [
        (row_idx, row.tolist().index(-1))
        for row_idx, row in enumerate(gamma_matrix)
        if (row == -1).sum() == 1 and (row != -1).sum() == len(row) - 1
    ]
    correct = len(correct_indices)
    correct_ratio = (correct / num_cols)**2
    sigma_gamma = max(1-label_distance_ratio-correct_ratio,0)

    # Step 1: Create a dictionary mapping old node indices to new based on gamma matrix
    correct_node_dict = {}
    for i, row in enumerate(gamma_matrix):
        if np.sum(row == -1) == 1 and np.sum(row != -1) == len(row) - 1:
            new_node_index = np.where(row == -1)[0][0]
            correct_node_dict[i] = int(new_node_index)

    # Step 2: Generate list of arrows based on new node definitions
    # This involves creating a new adjacency matrix with redefined nodes
    new_adj_matrix = np.zeros_like(adj_matrix)
    for old_idx, new_idx in correct_node_dict.items():
        for j in range(num_cols):
            if adj_matrix[old_idx, j] > 0 and j in correct_node_dict:
                new_adj_matrix[new_idx, correct_node_dict[j]] = adj_matrix[old_idx, j]

    # Step 3: Compare new_adj_matrix with anti_adj_matrix to count correct arrows
    correct_rows = 0
    for row_new, row_anti in zip(new_adj_matrix, anti_adj_matrix):
        if np.array_equal(row_new, row_anti):  # Check if entire rows are equal
            correct_rows += 1

    # Step 4: Calculate sigma_M
    sigma_M = 1-correct_rows/num_cols

    # Calculate reward
    reward = int(max_reward*(1-structure_label_ratio*sigma_M-(1-structure_label_ratio)*sigma_gamma))

    return reward
def antiparticle_quiver(quiver_matrix):
    # Extract adjacency and gamma matrices
    num_cols = quiver_matrix.shape[0]
    adj_matrix = quiver_matrix[:, :num_cols]
    gamma_matrix = quiver_matrix[:, num_cols:]

    # Update labels
    gamma_matrix = -gamma_matrix

    # Return result
    return np.hstack((adj_matrix,gamma_matrix))
def generate_sequences(num_nodes, max_steps, current_sequence=None):
    if current_sequence is None:
        current_sequence = []

    if len(current_sequence) == max_steps:
        yield current_sequence
        return

    nodes = range(num_nodes)  # Generate node indices from 0 to num_nodes - 1
    for node in nodes:
        # Ensure no two consecutive nodes are the same
        if not current_sequence or current_sequence[-1] != node:
            yield from generate_sequences(num_nodes, max_steps, current_sequence + [node])
def smart_walk(quiver_matrix, max_steps, max_length, find_all_finite_chambers_flag):
    # Initialize random walk with a deep copy to prevent altering the original during mutations
    initial_quiver = copy.deepcopy(quiver_matrix)
    quiver = copy.deepcopy(quiver_matrix)
    final_quiver = antiparticle_quiver(quiver)
    num_nodes = quiver.shape[0]  # Define num_nodes based on the quiver matrix
    gamma_matrix = quiver[:, num_nodes:]
    finite_chamber = []
    row_history = []
    action_history = []
    illegal_flag = False
    illegal_prefix = []
    total_steps = 0

    # Generate sequence at the start of each episode
    generator = generate_sequences(num_nodes, max_length)
    mutation_sequence = next(generator)

    while not finite_chamber and total_steps <= max_steps:
        for action in mutation_sequence:
            # Update available nodes at each step
            available_nodes = np.array([
                i for i in range(num_nodes)
                if not np.any(gamma_matrix[i, :] < 0)
                   and not any(
                    np.array_equal(gamma_matrix[i, :], row) or np.array_equal(gamma_matrix[i, :], 2 * row)
                    for row in row_history)
            ])

            # Check if the action is legal
            if action not in available_nodes:
                print(Fore.RED + f"Illegal node encountered: {action}. Terminating episode.")
                illegal_flag = True
                action_history.append(action)
                illegal_prefix = action_history
                break
            else: illegal_flag = False

            print(Fore.MAGENTA + "Following predetermined sequence...")
            print(f"\nMutating on node {action}")

            # Mutate the quiver and calculate the reward
            row_history.append(gamma_matrix[action, :])
            action_history.append(action)
            quiver = mutate_quiver(quiver, action)
            gamma_matrix = quiver[:, num_nodes:]
            print("Resulting gamma matrix:")
            print(gamma_matrix)
            reward = calculate_reward(quiver, final_quiver)

            # Terminate episode if terminal reward given
            if reward == -max_reward:
                print(Fore.RED + "Terminating episode: maximum negative label distance")
                illegal_flag = True
                illegal_prefix = action_history
                break
            else: illegal_flag = False

            if reward == max_reward:
                finite_chamber = history_to_finite_chamber(row_history)
                print(Fore.GREEN + f"Finite chamber (particles only) found at timestep {total_steps + 1}")
                print(Fore.GREEN + sp.pretty(finite_chamber))
                break

            total_steps += 1
            if total_steps > max_steps:
                break

        if finite_chamber:
            break
        else:
            print(Fore.BLUE + "Resetting episode...")
            quiver = copy.deepcopy(initial_quiver)
            gamma_matrix = quiver[:, num_nodes:]
            if illegal_flag:
                print(f"I'm gonna go to the next sequence without {illegal_prefix}")
                mutation_sequence = next(generator)  # Generate a new sequence
                while mutation_sequence[:len(illegal_prefix)] == illegal_prefix:
                    mutation_sequence = next(generator)
                print(f"Found: {mutation_sequence}")
            else: mutation_sequence = next(generator)
            row_history = []
            action_history = []
            print(Fore.MAGENTA + f"Mutation sequence: {mutation_sequence}")

    if total_steps > max_steps:
        print(Fore.RED + f"No finite chamber found in {max_steps} steps.")

    return quiver_matrix, max_steps, max_length, find_all_finite_chambers_flag
def random_cyclic_quiver(num_nodes):
    # Create random pairs (arrows)
    pairs = []
    for i in range(num_nodes):
        if num_nodes > 3:
            num_arrows = np.random.randint(1,num_nodes-2)
        else: num_arrows = 1
        for j in range(num_arrows):
            # Determine the nodes that i is already connected to in order to avoid repeats and reversals
            connected_to = set(pair[0] for pair in pairs if pair[1] == i)
            duplicates = set(pair[1] for pair in pairs if pair[0] == i)
            # Compute the set of available nodes that i can still be connected to
            available = set(range(num_nodes)) - {i} - connected_to - duplicates
            if available:
                chosen = np.random.choice(list(available))
                pairs.append((i,int(chosen)))

    # Create adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    adj_matrix[tuple(zip(*pairs))] = 1
    # Add random double arrows
    adj_matrix += (np.random.rand(num_nodes, num_nodes) < 0.05) & (adj_matrix == 1)

    # Create quiver matrix
    gamma_matrix = np.eye(num_nodes, dtype=int)
    quiver_matrix = np.hstack((adj_matrix,gamma_matrix))

    return quiver_matrix

# Reward parameters and hyperparameters

max_reward = 100  # Reward for reaching final state or penalty for early termination
structure_label_ratio = 0.1 # Ratio for favouring structural vs. label differences
max_steps = 12 # Maximum number of states in a finite chamber permitted before terminating
epsilon = 1 # Inital exploration rate
epsilon_decay = 0.999 # Decay of exploration rate
epsilon_min = 0.2 # Minimum exploration rate
gamma = 0.995 # Discount factor

# RL environments
class quiver_env(gym.Env):
    def __init__(self, quiver_matrix, antiparticle_quiver, mutate_quiver, calculate_reward, max_steps, epsilon, callback):
        self.initial_quiver_matrix = quiver_matrix
        self.antiparticle_quiver = antiparticle_quiver
        self.mutate_quiver = mutate_quiver
        self.calculate_reward = calculate_reward
        self.quiver_matrix = None
        self.np_random = None
        self.max_steps = max_steps
        self.step_counter = 0
        self.total_steps = 0
        self.callback = callback
        self.mutation_history = []
        self.chamber_history = []
        self.chamber_key = []
        self.action_history = []
        self.available_nodes = []
        self.optimal_path = []
        self.total_reward = 0
        self.rewards = []
        self.use_optimal_path = False
        self.illegal_action = False

        # Epsilon-greedy exploration
        self.epsilon = epsilon
        self.epsilon_decay_factor = epsilon_decay

        # Define the shape of each half of quiver
        num_rows, num_cols = quiver_matrix.shape
        first_half_cols = num_cols // 2
        second_half_cols = num_cols - first_half_cols
        self.num_nodes = num_rows

        # Define the observation spaces for each half of the matrix
        first_half_space = gym.spaces.Box(low=0, high=2, shape=(num_rows * first_half_cols,), dtype=int)
        second_half_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_rows * second_half_cols,), dtype=int)

        # Add available nodes to observation space as one-hot vector
        available_nodes_space = gym.spaces.Box(low=0, high=1, shape=(self.num_nodes,), dtype=int)

        # Combine all parts of the observation space
        self.observation_space = gym.spaces.Dict({
            'first_half': first_half_space,
            'second_half': second_half_space,
            'available_nodes': available_nodes_space
        })

        # Initialise action space
        self.action_space = gym.spaces.Discrete(self.num_nodes)

    def _prepare_observation(self):
        # One-hot encode available nodes
        available_nodes_array = np.zeros(self.num_nodes, dtype=int)
        available_nodes_array[self.available_nodes] = 1

        # Return a dictionary with all parts of the observation
        return {
            'first_half': self.quiver_matrix[:, :self.quiver_matrix.shape[1] // 2].flatten().astype(np.int64),
            'second_half': self.quiver_matrix[:, self.quiver_matrix.shape[1] // 2:].flatten().astype(np.int64),
            'available_nodes': available_nodes_array
        }

    def reset(self, seed=None, options=None):
        print(Fore.BLUE + "Resetting episode...")
        self.epsilon = max(self.epsilon*self.epsilon_decay_factor, epsilon_min)
        print(Fore.BLUE + f"Decreasing exploration rate, ε = {self.epsilon}")
        self.np_random, _ = seeding.np_random(seed)
        self.quiver_matrix = np.array(self.initial_quiver_matrix, copy=True)
        self.step_counter = 0
        self.illegal_action = False
        self.mutation_history = []
        self.action_history = []
        self.total_reward = 0
        self.rewards = []
        self.available_nodes = list(range(self.num_nodes))
        self.action_space = gym.spaces.Discrete(len(self.available_nodes))
        mask = np.zeros(self.num_nodes, dtype=bool)
        mask[self.available_nodes] = True

        # Flatten each half and return as a dictionary
        return self._prepare_observation(), {'action_mask': mask}

    def update_available_nodes(self):
        gamma_matrix = self.quiver_matrix[:, self.quiver_matrix.shape[1] // 2:]

        self.available_nodes = []
        for i in range(gamma_matrix.shape[0]):
            if any(np.array_equal(gamma_matrix[i], row) for row in self.mutation_history):
                continue
            if np.any(gamma_matrix[i] < 0):
                continue
            self.available_nodes.append(i)

    def step(self, action):
        # Ensure quiver_matrix is an np.ndarray
        if isinstance(self.quiver_matrix, tuple):
            self.quiver_matrix = np.array(self.quiver_matrix)

        # Reset chamber history here
        if self.step_counter == 0:
            self.chamber_history = []

        # Update available nodes for this step
        self.update_available_nodes()

        # Check if there are any available nodes; if not, terminate with a penalty
        if not self.available_nodes:
            reward = -self.total_reward
            print(Fore.RED + f"Terminating episode ({-self.total_reward}): no nodes left for mutation")
            terminated = True
            return self._prepare_observation(), reward, terminated, False, {}

        # Define the action mask based on available nodes
        mask = np.zeros(self.num_nodes, dtype=bool)
        mask[self.available_nodes] = True

        # Check if the selected action is within the available nodes
        if action >= len(mask) or not mask[action]:
            # Apply a very negative reward for selecting an invalid action and terminate
            reward = -self.total_reward
            print(Fore.RED + f"Invalid action {action}. Terminating with penalty ({reward}).")
            terminated = True
            return self._prepare_observation(), reward, terminated, False, {}

        # Epsilon-greedy exploration
        exploration_log = ""
        if self.np_random.random() < self.epsilon and len(self.available_nodes) > 1 and not self.use_optimal_path:
            action = self.np_random.choice(self.available_nodes)
            exploration_log = " (exploration)"

        # Use optimal path if specified
        if self.use_optimal_path and self.step_counter < len(self.optimal_path):
            action = self.optimal_path[self.step_counter]

        # Determine actual node index for mutation
        actual_node_index = action  # Since action is already within available nodes
        print(f"\nMutating on node {actual_node_index}" + exploration_log)

        # Record mutation action
        gamma_matrix = self.quiver_matrix[:, self.quiver_matrix.shape[1] // 2:]
        mutated_row = gamma_matrix[actual_node_index]
        self.mutation_history.append(mutated_row)
        self.chamber_history.append(mutated_row)
        self.mutation_history.append(2*mutated_row)
        self.action_history.append(action)

        # Mutate the quiver
        self.quiver_matrix = self.mutate_quiver(self.quiver_matrix, actual_node_index)
        gamma_matrix = self.quiver_matrix[:, self.quiver_matrix.shape[1] // 2:]
        print("Resulting gamma matrix:")
        print(gamma_matrix)

        # Calculate the reward
        reward = self.calculate_reward(self.quiver_matrix, self.antiparticle_quiver)
        if reward == -max_reward:
            reward = -self.total_reward
            trunc_flag = True
        else:
            trunc_flag = False
        print("Reward:", reward)
        self.total_steps += 1
        print("Training step:", self.total_steps)
        self.total_reward += reward*gamma**self.step_counter

        # Check if the antiparticle quiver has been reached
        if reward == max_reward:
            print(Fore.GREEN + "Antiparticle quiver reached!")

            # Save the optimal path if it hasn't been saved already
            if not self.optimal_path:
                self.optimal_path = self.action_history.copy()
                self.use_optimal_path = True
                print(Fore.GREEN + f"Finite chamber (particles only) found of length {len(self.chamber_history)}")
                self.actions = self.chamber_history
                self.actions2 = self.action_history
                print(Fore.GREEN + f"{sp.pretty(history_to_finite_chamber(self.actions))}")

                # Check if the chamber is in finite_chambers (from the callback)
                if self.callback and hasattr(self.callback, "finite_chambers"):
                    to_check = history_to_finite_chamber(self.actions)

                    # Check if the current chamber is in finite_chambers
                    if to_check in self.callback.finite_chambers:
                        print(Fore.YELLOW + "I've seen this chamber before!")

                        # Apply a penalty for revisiting a finite chamber
                        penalty = -max_reward
                        reward += penalty
                        print(Fore.YELLOW + f"Applying penalty {penalty} for revisiting this chamber.")

            # Terminate the episode successfully
            terminated = True
            return self._prepare_observation(), reward, terminated, False, {}

        # Check if the episode should be truncated
        self.step_counter += 1
        truncated = self.step_counter >= self.max_steps or reward <= -max_reward or trunc_flag

        # Return the observation, reward, termination, and truncation status
        self.update_available_nodes()
        obs = self._prepare_observation()
        terminated = False
        return obs, reward, terminated, truncated, {'action_mask': mask}
class stop_training(BaseCallback):
    def __init__(self, verbose=0):
        super(stop_training, self).__init__(verbose)
        self.timesteps = 0
        self.episodes = 0

    def _on_step(self) -> bool:
        # Increment timesteps
        self.timesteps += 1

        # Check if we're at the start of a new episode
        if "infos" in self.locals and "episode" in self.locals["infos"][0]:
            self.episodes = self.locals["infos"][0]["episode"]["l"]

        # Access the environment and check for the optimal path flag
        env = self.training_env.get_attr("env")[0]

        # Check if the optimal path has been found in the environment
        if env.use_optimal_path:
            if self.verbose > 0:
                print(Fore.GREEN + f"Optimal path found: Stopping training early.")
            return False  # Stop training

        return True  # Continue training otherwise
class find_all_finite_chambers(BaseCallback):
    def __init__(self, verbose=0, max_timesteps=1000000):
        super(find_all_finite_chambers, self).__init__(verbose)
        self.timesteps = 0
        self.finite_chambers = []  # Store all finite chambers
        self.topological_chambers = []
        self.max_timesteps = max_timesteps
        self.to_plot = [0]
        self.stagnation_counter = 0
        self.compute_automorphisms = True

    def _on_step(self) -> bool:
        # Increment timesteps
        self.timesteps += 1

        # Access the environment
        env = self.training_env.get_attr("env")[0]
        if self.compute_automorphisms:
            self.automorphisms = all_automorphisms(env.initial_quiver_matrix)
            self.compute_automorphisms = False

        # Check if the optimal path has been found in the environment
        if env.use_optimal_path:
            # Save the chamber if it's unique
            action_history = env.actions2
            actions = env.actions
            chamber = history_to_finite_chamber(actions)
            env.use_optimal_path = False
            env.optimal_path = []
            if chamber not in self.finite_chambers:
                self.stagnation_counter = 0
                self.topological_chambers.append(chamber)
                equiv_chambers = all_chambers(actions,self.automorphisms)
                self.to_plot.append(self.to_plot[-1] + 1) # For plotting
                for chamber in equiv_chambers:
                    self.finite_chambers.append(chamber)
                if self.verbose > 0:
                    print(Fore.GREEN + f"Found new finite chamber (degeneracy {len(equiv_chambers)}): Length = {len(chamber)}")
        else:
            self.to_plot.append(self.to_plot[-1])

        # Stop training if maximum timesteps are reached or stagnation condition is met
        self.stagnation_counter += 1
        print(Fore.MAGENTA + f"Time since no finite chamber: {self.stagnation_counter}")
        self.stagnation_lim = 1e6
        if self.timesteps >= self.max_timesteps or self.stagnation_counter > self.stagnation_lim:
            if self.verbose > 0:
                print(Fore.GREEN + f"Training complete. Total finite chambers found: {len(self.finite_chambers)}, Total unique chambers found: {len(self.topological_chambers)}")
                for idx, chamber in enumerate(self.topological_chambers):
                    print(Fore.GREEN + f"Unique chamber {idx + 1}:")
                    print(Fore.GREEN + sp.pretty(chamber))

            # # Update data
            # self.to_plot.pop(0)
            # timesteps = [i + 1 for i in range(len(self.to_plot))]
            #
            # # Save `to_plot` to a data file (CSV)
            # data_file = "finite_chambers_performance_2.csv"
            # with open(data_file, mode='w', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(["Timestep", "Number of Finite Chambers"])
            #     for t, chambers in zip(timesteps, self.to_plot):
            #         writer.writerow([t, chambers])
            # print(f"Performance data saved to {data_file}")
            #
            # # Save finite chambers
            # df = pd.DataFrame(self.finite_chambers)
            # df.to_csv('finite_chambers_2.csv', index=False)
            # print(f"Chamber data saved to finite_chambers_2.csv")
            #
            # # Create a single plot
            # plt.figure(figsize=(8, 6))
            # plt.plot(timesteps, self.to_plot, linestyle='-', label='Finite Chambers Found')  # Just the line
            #
            # # Add labels and title
            # plt.xlabel('Timestep')
            # plt.ylabel('Number of Finite Chambers')
            # plt.title('Finite Chambers Found Over Time')
            # plt.legend()
            # plt.grid(True)
            #
            # # Show the plot
            # plt.show()

            return False
        return True
class masked_ppo_policy(MultiInputActorCriticPolicy):
    def __init__(self, *args, input_size, **kwargs):
        super(masked_ppo_policy, self).__init__(*args, **kwargs)

        # Override action_net and value_net to accept a 78-dimensional input
        self.action_net = th.nn.Linear(input_size, self.action_space.n)  # Adjusts action_net input size to 78
        self.value_net = th.nn.Linear(input_size, 1)  # Adjusts value_net input size to 78

    def forward(self, obs, deterministic=False):
        # Retrieve the action mask from the observation
        action_mask = obs["available_nodes"].bool()

        # Extract features from the observation
        latent = self.extract_features(obs)

        # Get action logits using the modified action_net
        logits = self.action_net(latent)

        # Apply the mask to set the logits of invalid actions to a very low number
        masked_logits = th.where(action_mask, logits, th.tensor(-1e20))

        # Create a categorical distribution from masked logits manually
        distribution = self.action_dist.proba_distribution(masked_logits)

        # Sample action from the distribution
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        # Get the value estimate using the modified value_net
        value = self.value_net(latent)

        return actions, value, log_prob
def record_episode(env, model):
    obs, _ = env.reset()  # Separate observation from info, as only obs is needed
    done = False
    cumulative_rewards = []
    intermediate_quivers = []
    total_reward = 0

    # Store the initial quiver
    intermediate_quivers.append(env.quiver_matrix.copy())

    # Track and use the optimal path if it exists
    step_counter = 0  # Tracks position in the optimal path

    while not done:
        # Use the optimal path action if it exists
        if env.use_optimal_path and step_counter < len(env.optimal_path):
            action = env.optimal_path[step_counter]
            print(Fore.GREEN + f"Using optimal path action: {action} at step {step_counter}")  # DEBUG
        else:
            # Get action from the trained model
            action, _ = model.predict(obs, deterministic=True)
            print(f"Agent selected action: {action}")  # DEBUG

        # Take the action in the environment
        obs, reward, done, truncated, _ = env.step(action)

        # Update done status to include truncated
        done = done or truncated

        # Accumulate rewards
        total_reward += reward
        cumulative_rewards.append(total_reward)

        # Store intermediate quivers during each step
        intermediate_quivers.append(env.quiver_matrix.copy())

        # Increment the step counter
        step_counter += 1

    return cumulative_rewards, intermediate_quivers
def build_model():
    choices = {}

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
    print_centered("Chamber App", style='bold')
    print_centered("arXiv:2501.14863")
    print_centered("Federico Carta, Asa Gauntlett, Finley Griffin, Yang-Hui He", style='italic')
    print_centered(divider)

    # Choice for model type
    print("\nSelect the model to train:")
    print("1. PPO")
    print("2. SW (baseline)")
    choice_model = input("Enter the number of your choice (1 or 2): ")
    while choice_model not in ["1", "2"]:
        choice_model = input("Invalid input. Please enter 1 for PPO or 2 for SW: ")
    choices["model_type"] = "SW" if choice_model == "2" else "PPO"

    choices["action"] = "Single"
    if choices["model_type"] == "PPO":
        print("\nSelect an action:")
        print("1. Find a single finite chamber")
        print("2. Find all finite chambers")
        choice_action = input("Enter the number of your choice: ")
        while choice_action not in ["1", "2"]:
            choice_action = input("Invalid input. Please enter 1 or 2: ")
        choices["action"] = "Single" if choice_action == "1" else "Find all finite chambers"

    # Choice for random matrix or pre-defined matrix
    print("\nNew (1) or pre-defined (2) quiver?")
    do_new = None
    do_new = input("Enter the number of your choice: ") == "1"

    # Choice for adjacency matrix
    if do_new:
        matrix_input = input("Enter adjacency matrix in array form [[],[],...]: ")
        choices["matrix"] = np.asarray(ast.literal_eval(matrix_input), dtype=int)
        num_nodes = choices["matrix"].shape[0]
        hidden_size = 2 * int(num_nodes) ** 2 + int(num_nodes)
        choices["matrix_name"] = "user_input"

    else:
        print("\nSelect the quiver to train on:")
        matrices = {
            "1": ("SU(2) Nf = 4 Q1", np.array([[0, 2, 0, 0, 0, 0],
                                      [0, 0, 1, 1, 1, 1],
                                      [1, 0, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0, 0]])),
            "2": ("SU(2) Nf = 4 Q2", np.array(
                [[0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 0, 0]])),
            "3": ("SU(2) Nf = 4 Q3", np.array(
                [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 1, 0, 1, 0, 0],
                 [1, 0, 1, 0, 0, 0]])),
            "4": ("SU(2) Nf = 4 Q4", np.array(
                [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1],
                 [1, 0, 1, 1, 0, 0]])),
            "5": ("X6 Derksen-Owen", np.array([[0,2,0,0,0,0],[0,0,1,0,0,0],[1,0,0,1,0,1],[0,0,0,0,2,0],[0,0,1,0,0,0],[0,0,0,0,0,0]])),
            "6": ("X7 Derksen-Owen", np.array(
                [[0, 2, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 2, 0, 0], [0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 2],[0, 0, 1, 0, 0, 0, 0]])),
            "7": ("Elliptic E6", np.array([
    [0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]])),
            "8": ("Elliptic E7", np.array([
                [0, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]])),
            "9": ("Elliptic E8", np.array([
                [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])),
        }
        for key, (name, matrix) in matrices.items():
            print(f"{key}. {name}")
        choice_matrix = input("Enter the number of your choice: ")
        while choice_matrix not in matrices:
            choice_matrix = input("Invalid input. Please select a valid number: ")
        choices["matrix_name"], choices["matrix"] = matrices[choice_matrix]
        n = matrices[choice_matrix][1].shape[0]
        hidden_size = 2 * n ** 2 + n

    # Choice for max timesteps
    choice_steps = input("\nEnter the maximum number of training steps (e.g. 1e3):")
    choice_steps = int(float(choice_steps))
    choices["max_timesteps"] = choice_steps
    if choice_model == "3":
        return choices

    # Choice for network architecture
    print("\nSelect the network architecture:")
    print("1. Small (128, 256, 128)")
    print("2. Medium (256, 512, 256)")
    print("3. Large (512, 1024, 512)")
    print(f"")
    choice_architecture = input("Enter the number of your choice (1, 2, or 3): ")
    while choice_architecture not in ["1", "2", "3"]:
        choice_architecture = input("Invalid input. Please enter 1, 2, or 3: ")
    architecture_map = {
        "1": [hidden_size, 128, 256, 128, hidden_size],
        "2": [hidden_size, 256, 512, 256, hidden_size],
        "3": [hidden_size, 512, 1024, 512, hidden_size]
    }
    choices["architecture"] = architecture_map[choice_architecture]

    # Choice for learning rate
    print("\nSelect the learning rate:")
    print("1. 0.6")
    print("2. 0.5")
    print("3. 0.1")
    print("4. 0.01")
    print("5. 0.001")
    choice_lr = input("Enter the number of your choice (1, 2, 3, 4 or 5): ")
    while choice_lr not in ["1", "2", "3", "4", "5"]:
        choice_lr = input("Invalid input. Please enter 1, 2, 3, 4 or 5: ")
    lr_map = {
        "1": 0.6,
        "2": 0.5,
        "3": 0.1,
        "4": 0.01,
        "5": 0.001,
    }
    choices["learning_rate"] = lr_map[choice_lr]

    return choices
def train_model(choices):
    # Extract the adjacency matrix and environment setup
    adj_matrix = choices["matrix"]
    input_size = 2 * adj_matrix.shape[0] ** 2 + adj_matrix.shape[0]
    max_timesteps = choices["max_timesteps"]

    initial_quiver = np.hstack((adj_matrix, np.eye(len(adj_matrix), dtype=int)))
    user_input = input("Do you want to plot the quiver? (yes/no): ").strip().lower()
    if user_input == 'yes':
        plot_quiver(initial_quiver)  # Assuming this function is defined elsewhere
    final_quiver = antiparticle_quiver(initial_quiver)

    if choices["model_type"] == "SW":
        find_all_chambers_flag = True if choices["action"] == "Find all finite chambers" else False
        finite_chamber = smart_walk(initial_quiver, max_timesteps, max_steps, find_all_chambers_flag)
        return finite_chamber

    # Define the correct callback based on the user's choice
    if choices["action"] == "Find all finite chambers":
        callback = find_all_finite_chambers(verbose=1, max_timesteps=max_timesteps)
        print("Finding all finite chambers...")
    else:
        callback = stop_training(verbose=1)
        print("Training to find an optimal path...")

    # Define the environment
    env = quiver_env(initial_quiver, final_quiver, mutate_quiver, calculate_reward, max_steps, epsilon, callback)

    # Check if the environment is compatible
    check_env(env)

    # Define network architecture and other hyperparameters
    policy_kwargs = dict(net_arch=choices["architecture"])
    learning_rate = choices["learning_rate"]
    gamma = 0.95  # Example gamma, you can make this a choice too

    # Instantiate the model based on the user's choice
    model = None
    if choices["model_type"] == "PPO" or "SW":
        policy_kwargs["input_size"] = input_size
        model = PPO(masked_ppo_policy, env, policy_kwargs=policy_kwargs, learning_rate=learning_rate, gamma=gamma,
                    verbose=1)
    elif choices["model_type"] == "DQN":
        model = DQN("MultiInputPolicy", env, policy_kwargs=policy_kwargs, learning_rate=learning_rate, gamma=gamma,
                    buffer_size=1000, verbose=1)
    else:
        raise ValueError("Unsupported model type.")

    # Train the model
    print(f"Training {choices['model_type']} model on {choices['matrix_name']}...")
    model.learn(total_timesteps=max_timesteps, callback=callback)

    if choices["action"] == "Find all finite chambers":
        return callback.finite_chambers

    # # Test and evaluate the trained model
    # cumulative_rewards, intermediate_quivers = record_episode(env, model)  # Replace with your evaluation function
    #
    # # Plot cumulative rewards
    # plt.figure(figsize=(10, 5))
    # plt.plot(cumulative_rewards, label=f"{choices['model_type']} Training")
    # plt.xlabel("Timestep")
    # plt.ylabel("Cumulative Reward")
    # plt.title(f"Training Performance ({choices['model_type']} on {choices['matrix_name']})")
    # plt.legend()
    # plt.show()
    #
    # # Save cumulative rewards to a CSV file
    # csv_filename = f"{choices['model_type']}_cumulative_rewards.csv"
    # with open(csv_filename, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     # Write header
    #     writer.writerow(['Timestep', 'Cumulative Reward'])
    #     # Write data rows using the index as timestep
    #     for timestep, reward in enumerate(cumulative_rewards):
    #         writer.writerow([timestep, reward])
    #
    # print(f"Cumulative rewards saved to {csv_filename}")
    #
    # # Optionally, plot intermediate quivers
    # print("Plotting final quiver:")
    # for quiver in intermediate_quivers:
    #     plot_quiver(quiver)

choices = build_model()
train_model(choices)