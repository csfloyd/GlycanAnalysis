import sympy
import numpy as np
import time
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from scipy.signal import savgol_filter, find_peaks

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Reaction:
    reactants: List[Tuple[int, int]]  # (species_index, stoichiometry)
    products: List[Tuple[int, int]]   # (species_index, stoichiometry)

        
def generate_stoichiometry_matrix(L, n_reactions, reac_order=1, force_reverse_reactions=False, seed=None, timeout = 3):
    """
    Generate a stoichiometry matrix A given a matrix L (whose nullspace defines conservation laws),
    the number of reactions, and the maximum reaction order.
    
    Args:
        L: sympy.Matrix, the matrix whose nullspace defines conservation laws
        n_reac: int, number of reactions (columns in A)
        reac_order: int, maximum allowed absolute value for stoichiometric coefficients (default 1)
        seed: int or None, random seed for reproducibility
        
    Returns:
        A: sympy.Matrix, the generated stoichiometry matrix
    """
    if seed is not None:
        np.random.seed(seed)
    nullspace = L.nullspace()
    # Check for non-integer elements in nullspace vectors
    for v in nullspace:
        if not all(x == int(x) for x in v):
            raise ValueError("Nullspace vector contains non-integer elements. Please provide a matrix L with integer nullspace vectors.")
    n_null = len(nullspace)
    cols = []
    sub_start_time = time.time()
    for m in range(n_reactions):
        while True:
            coefs = [np.random.randint(-2, 2) for _ in range(n_null)]  
            zero_col = sympy.zeros(*nullspace[0].shape)
            col = sum((c * v for c, v in zip(coefs, nullspace)), zero_col)
            sum_neg = sum(-x for x in col if x < 0)
            sum_pos = sum(x for x in col if x > 0)
            if (
                all(abs(x) <= reac_order for x in col)
                and sum_neg <= reac_order
                and sum_pos <= reac_order
                and sum_neg != 0
                and sum_pos != 0
                and not any(col == existing for existing in cols)
            ):
                break
        
            if time.time() - sub_start_time > timeout:
                raise ValueError("Timeout reached while generating stoichiometry matrix.")
        cols.append(col)
        if force_reverse_reactions:
            cols.append(-col)
    A = sympy.Matrix.hstack(*cols)
    return A

def build_mass_action_odes(A, Reaction, species_prefix='S', rate_prefix='k'):
    """
    Given a stoichiometry matrix A and a Reaction class, build the mass action ODEs.
    Returns:
        S: tuple of sympy symbols for species concentrations
        k: tuple of sympy symbols for rate constants
        odes: list of sympy expressions for the ODEs
        reaction_network: list of Reaction objects
        rate_exprs: list of sympy expressions for reaction rates
    """
    n_species = A.shape[0]
    n_reactions = A.shape[1]

    # Build reaction network
    reaction_network = []
    for j in range(n_reactions):
        reactants = []
        products = []
        for i in range(n_species):
            stoich = A[i, j]
            if stoich < 0:
                reactants.append((i, -stoich))
            elif stoich > 0:
                products.append((i, stoich))
        reaction_network.append(Reaction(reactants, products))

    # Define symbols
    S = sympy.symbols(f'{species_prefix}1:{n_species+1}')
    k = sympy.symbols(f'{rate_prefix}1:{n_reactions+1}')

    # Build rate expressions
    rate_exprs = []
    for j, rxn in enumerate(reaction_network):
        rate = k[j]
        for idx, stoich in rxn.reactants:
            rate *= S[idx]**stoich
        rate_exprs.append(rate)

    # Build ODEs
    odes = [0 for _ in range(n_species)]
    for j, rxn in enumerate(reaction_network):
        rate = rate_exprs[j]
        for idx, stoich in rxn.reactants:
            odes[idx] -= stoich * rate
        for idx, stoich in rxn.products:
            odes[idx] += stoich * rate

    return S, k, odes, reaction_network, rate_exprs

def random_conservation_laws(n_cons, n_species, probs, seed=None):
    """
    Generate a random set of conservation laws as a sympy.Matrix.

    Args:
        n_cons (int): Number of conservation laws (rows).
        n_species (int): Number of species (columns).
        mc (int): Maximum absolute value for integer elements.
        seed (int or None): Random seed for reproducibility.

    Returns:
        sympy.Matrix: Matrix of shape (n_cons, n_species).
    """
    if seed is not None:
        np.random.seed(seed)
    L = []
    mc = len(probs)
    for _ in range(n_cons):
        not_all_zero = True
        while not_all_zero:
            values = np.arange(mc)  
            row = np.random.choice(values, size=n_species, p=probs).tolist()
            if sum(row) != 0:  # Exclude all-zero row
                not_all_zero = False
        L.append(row)
    return sympy.Matrix(L)


def print_reaction_network(reaction_network):
    for idx, rxn in enumerate(reaction_network):
        react_str = ' + '.join(
            [ ' + '.join([f"S{r[0]+1}"] * r[1]) for r in rxn.reactants ]
        ) or '∅'
        prod_str = ' + '.join(
            [ ' + '.join([f"S{p[0]+1}"] * p[1]) for p in rxn.products ]
        ) or '∅'
        print(f"R{idx+1}: {react_str} -> {prod_str}")
      

def compute_ss_response_as_function_of_input(rhs, k_values, y0, t_span, t_eval, input_type, input_idx, input_vals, method = 'LSODA', rtol = 1e-3, atol = 1e-6):
    ss_response = []
    for input_val in input_vals:
        if input_type == 'rate':
            k_values_copy = k_values.copy()
            k_values_copy[input_idx] = k_values_copy[input_idx] * np.exp(input_val)
            sol = solve_ivp(rhs, t_span, y0, args=(k_values_copy,), t_eval=t_eval, method=method, rtol=rtol, atol=atol)
        elif input_type == 'concentration':
            y0_copy = y0.copy()
            y0_copy[input_idx] = input_val
            sol = solve_ivp(rhs, t_span, y0_copy, args=(k_values,), t_eval=t_eval, method=method, rtol=rtol, atol=atol)
        ss_response.append(sol.y[:, -1])
    return np.array(ss_response)


def smooth_curve(y, window_length=3, polyorder=1):
    """
    Applies a Savitzky-Golay filter to smooth a sequence of y-values.

    Parameters:
        y (array-like): Sequence of y-values at equally spaced x-values.
        window_length (int): Window length for Savitzky-Golay filter (must be odd and >= polyorder+2).
        polyorder (int): Polynomial order for Savitzky-Golay filter.

    Returns:
        array-like: Smoothed y-values.
    """
    y = np.asarray(y)
    # Ensure window_length is valid
    if window_length < polyorder + 2:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1
    if window_length > len(y):
        window_length = len(y) if len(y) % 2 == 1 else len(y) - 1
    # Smooth y and compute its first derivative 
    y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
    return y_smooth


# def count_turning_points(y, eps=1e-8, window_length=5, polyorder=2, smooth=False):
#     """
#     Counts the number of turning points (where the derivative changes sign)
#     in a sequence of equally spaced points, ignoring small numerical fluctuations.

#     Parameters:
#         y (array-like): Sequence of y-values at equally spaced x-values.
#         eps (float): Threshold below which the derivative is considered zero.

#     Returns:
#         int: Number of turning points.
#     """
#     if smooth:
#         y = smooth_curve(y, window_length=window_length, polyorder=polyorder)
#     y = np.asarray(y)
#     dy = np.diff(y)
#     # Set small derivatives to zero
#     dy[np.abs(dy) < eps] = 0
#     sign_dy = np.sign(dy)
#     # Only count sign changes where both sides are nonzero (not flat)
#     sign_changes = (sign_dy[1:] * sign_dy[:-1] < 0) & (sign_dy[1:] != 0) & (sign_dy[:-1] != 0)
#     return np.sum(sign_changes)

def count_turning_points(y, eps=1e-8, window_length=5, polyorder=2, smooth=False, prominence=None, wlen=None):
    """
    Counts the number of turning points (peaks + valleys) in a sequence of equally spaced points.
    Uses scipy.signal.find_peaks for robust detection, with optional smoothing.

    Parameters:
        y (array-like): Sequence of y-values at equally spaced x-values.
        eps (float): Minimum height difference to consider a peak/valley (used as prominence if not specified).
        window_length (int): Window length for smoothing (if smooth=True).
        polyorder (int): Polynomial order for smoothing (if smooth=True).
        smooth (bool): Whether to smooth the curve before finding peaks/valleys.
        prominence (float or None): Prominence parameter for find_peaks. If None, uses eps.
        wlen (int or None): Window length for find_peaks (in samples).

    Returns:
        int: Number of turning points (peaks + valleys).
    """
    if smooth:
        y = smooth_curve(y, window_length=window_length, polyorder=polyorder)
    y = np.asarray(y)
    if prominence is None:
        prominence = eps
    # Find peaks
    peaks, _ = find_peaks(y, prominence=prominence, wlen=wlen)
    # Find valleys (peaks in -y)
    valleys, _ = find_peaks(-y, prominence=prominence, wlen=wlen)
    return len(peaks) + len(valleys)


def count_turning_points_columns(arr, eps=1e-8, window_length=5, polyorder=2, smooth=False, prominence=None, wlen=None):
    """
    Counts the number of turning points for each column in a 2D numpy array.
    
    Parameters:
        arr (numpy.ndarray): 2D array where each column contains a sequence of values
        
    Returns:
        numpy.ndarray: Array containing the number of turning points for each column
    """
    arr = np.asarray(arr)
    n_cols = arr.shape[1]
    turning_points = np.zeros(n_cols)
    
    for i in range(n_cols):
        turning_points[i] = count_turning_points(arr[:, i], eps=eps, window_length=window_length, polyorder=polyorder, smooth=smooth, prominence=prominence, wlen=wlen)
        
    return turning_points

def recompute_turning_points_list(ss_response_list, eps=1e-8, window_length=5, polyorder=2, smooth=False, prominence=None, wlen=None):
    """
    Given a list of steady-state response arrays, recompute the number of turning points for each entry.
    Args:
        ss_response_list (list of np.ndarray): Each entry is a 2D array (responses for one system).
        eps (float): Threshold for numerical noise in turning point detection.
    Returns:
        list of np.ndarray: Each entry is the turning points array for the corresponding ss_response.
    """
    return [[count_turning_points_columns(ss_response, eps=eps, window_length=window_length, polyorder=polyorder, smooth=smooth, prominence=prominence, wlen=wlen) for ss_response in ss_responses] for ss_responses in ss_response_list]


def reaction_network_to_bipartite_graph(reaction_network, n_species):
    G = nx.DiGraph()
    # Add species nodes
    for i in range(n_species):
        G.add_node(f"S{i+1}", bipartite='species')
    # Add reaction nodes and edges
    for idx, rxn in enumerate(reaction_network):
        rxn_node = f"R{idx+1}"
        G.add_node(rxn_node, bipartite='reaction')
        # Reactants: edge from species to reaction
        for sidx, stoich in rxn.reactants:
            G.add_edge(f"S{sidx+1}", rxn_node, stoich=stoich, role='reactant')
        # Products: edge from reaction to species
        for sidx, stoich in rxn.products:
            G.add_edge(rxn_node, f"S{sidx+1}", stoich=stoich, role='product')
    return G


def export_data_as_apkl(filename, S_list, k_list, odes_list, reaction_network_list, ss_response_list, G_list):
    """
    Exports the provided data lists to a .apkl file using pickle.
    
    Args:
        filename (str): The path to the output .apkl file.
        S_list, k_list, odes_list, reaction_network_list, turning_points_list, ss_response_list, G_list: Data to export.
    """
    data = {
        'S_list': S_list,
        'k_list': k_list,
        'odes_list': odes_list,
        'reaction_network_list': reaction_network_list,
        'ss_response_list': ss_response_list,
        'G_list': G_list
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data exported to {filename}")

def import_data_from_apkl(filename):
    """
    Loads data from a .apkl file created by export_data_as_apkl.
    
    Args:
        filename (str): The path to the .apkl file.
    
    Returns:
        dict: Dictionary containing all the exported lists.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def draw_bipartite_reaction_network(G, pos=None, ax=None):
    """
    Draws a bipartite reaction network graph with custom node and edge colors/styles.
    Args:
        G (networkx.Graph): The bipartite reaction network graph.
        pos (dict or None): Node positions. If None, uses spring_layout.
        figsize (tuple): Figure size for the plot.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    color_map = []
    for node in G.nodes():
        if node.startswith('S'):  # Species nodes
            color_map.append('skyblue')
        else:  # Reaction nodes
            color_map.append('salmon')

    edge_colors = []
    edge_widths = []
    edge_styles = []
    for u, v in G.edges():
        if u.startswith('S') and v.startswith('R'):
            edge_colors.append('black')
            edge_widths.append(1.75)
            edge_styles.append('solid')
        elif u.startswith('R') and v.startswith('S'):
            edge_colors.append('green')
            edge_widths.append(1.75)
            edge_styles.append('dashed')

    # If no ax is provided, create a new figure and axes
    if ax is None:
        fig, ax = plt.figure(figsize=(10, 6))

    nx.draw(
        G, pos,
        with_labels=True,
        node_color=color_map,
        edge_color=edge_colors,
        width=edge_widths,
        style=edge_styles,
        node_size=800,
        arrows=True,
        arrowsize=20, 
        ax=ax
    )
    if ax is None:
        plt.show()
