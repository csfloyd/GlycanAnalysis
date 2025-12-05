"""
Generation module for reaction networks.

This module contains functions for generating random reaction networks,
linkage groups, and other network structures.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Iterator
import sympy
from scipy.optimize import nnls
from scipy.optimize import minimize


def latin_hypercube_sampling(ranges: List[Tuple[float, float]], K: int, seed: int = None) -> Iterator[List[float]]:
    """
    Generate Latin Hypercube Sampling (LHS) samples.
    
    Latin Hypercube Sampling ensures good coverage of the parameter space by dividing
    each dimension into K bins and ensuring each bin is sampled exactly once.
    
    Args:
        ranges: List of (min, max) tuples defining the range for each dimension
        K: Number of bins per dimension (and total number of samples)
        seed: Random seed for reproducibility (optional)
    
    Yields:
        List[float]: A sample point with one value per dimension
        
    Example:
        >>> ranges = [(0, 1), (0, 10), (-1, 1)]
        >>> samples = latin_hypercube_sampling(ranges, K=5)
        >>> for sample in samples:
        ...     print(sample)  # [0.2, 3.4, -0.1], [0.8, 7.2, 0.6], etc.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_dims = len(ranges)
    
    # Create a K x n_dims matrix to store bin assignments
    # Each row represents one sample, each column represents one dimension
    bin_assignments = np.zeros((K, n_dims), dtype=int)
    
    # For each dimension, randomly assign each bin (0 to K-1) to exactly one sample
    for dim in range(n_dims):
        bin_assignments[:, dim] = np.random.permutation(K)
    
    # Generate samples
    for sample_idx in range(K):
        sample = []
        for dim in range(n_dims):
            min_val, max_val = ranges[dim]
            bin_idx = bin_assignments[sample_idx, dim]
            
            # Calculate bin boundaries
            bin_width = (max_val - min_val) / K
            bin_min = min_val + bin_idx * bin_width
            bin_max = bin_min + bin_width
            
            # Sample uniformly within the bin
            sample_value = np.random.uniform(bin_min, bin_max)
            sample.append(sample_value)
        
        yield sample


def random_connected_graph(n_nodes, n_edges, rng, seed, directed=False):
    """
    Generate a random connected graph with n_nodes and n_edges.
    Uses local numpy and Python RNGs for full reproducibility.
    Args:
        n_nodes: Number of nodes
        n_edges: Number of edges
        directed: Whether the graph is directed
        rng: numpy.random.Generator
        py_rng: random.Random
    Returns:
        G: networkx Graph or DiGraph
    """

    nodes = list(range(n_nodes))
    if directed:
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        # Generate a random tree and orient edges randomly
        tree = nx.random_tree(n_nodes, seed=seed)
        for u, v in tree.edges():
            if rng.random() < 0.5:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)
        # Add extra edges
        possible_edges = [(i, j) for i in nodes for j in nodes if i != j and not G.has_edge(i, j)]
        extra_edges_needed = n_edges - (n_nodes - 1)
        if extra_edges_needed > 0:
            extra_edges_indices = rng.choice(len(possible_edges), size=extra_edges_needed, replace=False)
            for idx in extra_edges_indices:
                u, v = possible_edges[idx]
                G.add_edge(u, v)
    else:
        G = nx.random_tree(n_nodes, seed=seed)
        # Add extra edges
        possible_edges = [(i, j) for i in nodes for j in nodes if i < j and not G.has_edge(i, j)]
        extra_edges_needed = n_edges - (n_nodes - 1)
        if extra_edges_needed > 0:
            extra_edges_indices = rng.choice(len(possible_edges), size=extra_edges_needed, replace=False)
            for idx in extra_edges_indices:
                u, v = possible_edges[idx]
                G.add_edge(u, v)
    return G


def random_conservation_laws(n_cons, n_species, probs):
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
    return np.array(L)


def generate_thermodynamic_rates(r_n, C0=1, beta=1, E_range=3, B_range=3, F_range=0, seed=None):
    """
    Generate reaction rates based on species energies and reaction barriers.
    
    Args:
        r_n: ReactionNetwork instance
        C0: Base concentration (default 1)
        beta: Inverse temperature parameter (default 1) 
        E_range: Range for random species energies (default 3)
        B_range: Range for random reaction barriers (default 3)
        F_range: Range for random reaction affinities (default 0)
        seed: Random seed for reproducibility (default None)
    
    Returns:
        List of reaction rates
    """
    if seed is not None:
        np.random.seed(seed)
        
    assert r_n.force_reverse
    n_reversible_reactions = int(len(r_n.reactions)/2)

    species_energies = np.random.uniform(-E_range, E_range, r_n.n_species)
    species_energies_dict = dict(zip(r_n.species_names, species_energies))
    reaction_barriers = np.random.uniform(-B_range, B_range, n_reversible_reactions)
    reaction_affinities = np.random.uniform(-F_range, F_range, n_reversible_reactions)

    reac_rates = []
    
    for reac_ind in range(n_reversible_reactions):
        i, j, _ = r_n.reactions[2*reac_ind]
        src_species = r_n.all_complexes[i].split("+")
        stoich_src = len(src_species)
        dst_species = r_n.all_complexes[j].split("+")
        stoich_dst = len(dst_species)

        src_energy = np.sum([species_energies_dict[s] for s in src_species])
        dst_energy = np.sum([species_energies_dict[s] for s in dst_species])

        fwd_rate = np.exp(beta*(-reaction_barriers[reac_ind] + src_energy + reaction_affinities[reac_ind]/2)) * (C0**(1-stoich_src))
        rev_rate = np.exp(beta*(-reaction_barriers[reac_ind] + dst_energy - reaction_affinities[reac_ind]/2)) * (C0**(1-stoich_dst))
        reac_rates.append(fwd_rate)
        reac_rates.append(rev_rate)

    return reac_rates

def get_rates_from_exponents(r_n, species_energies, reaction_barriers, reaction_affinities,C0=1, beta=1):
    """
    Generate reaction rates based on species energies and reaction barriers.
    
    Args:
        r_n: ReactionNetwork instance
        C0: Base concentration (default 1)
        beta: Inverse temperature parameter (default 1) 
        species_energies: Dictionary of species energies
        reaction_barriers: Dictionary of reaction barriers
        reaction_affinities: Dictionary of reaction affinities
        seed: Random seed for reproducibility (default None)
    
    Returns:
        List of reaction rates
    """
    assert r_n.force_reverse
    n_reversible_reactions = int(len(r_n.reactions)/2)
    species_energies_dict = dict(zip(r_n.species_names, species_energies))

    reac_rates = []
    
    for reac_ind in range(n_reversible_reactions):
        i, j, _ = r_n.reactions[2*reac_ind]
        src_species = r_n.all_complexes[i].split("+")
        stoich_src = len(src_species)
        dst_species = r_n.all_complexes[j].split("+")
        stoich_dst = len(dst_species)

        src_energy = np.sum([species_energies_dict[s] for s in src_species])
        dst_energy = np.sum([species_energies_dict[s] for s in dst_species])

        fwd_rate = np.exp(beta*(-reaction_barriers[reac_ind] + src_energy + reaction_affinities[reac_ind]/2)) * (C0**(1-stoich_src))
        rev_rate = np.exp(beta*(-reaction_barriers[reac_ind] + dst_energy - reaction_affinities[reac_ind]/2)) * (C0**(1-stoich_dst))
        reac_rates.append(fwd_rate)
        reac_rates.append(rev_rate)

    return reac_rates


def generate_initial_concentrations(L, base_concentrations=None, scale=4, offset_scale=3):
    # Get nullspace basis vectors
    nullspace = sympy.Matrix(L).nullspace()
    
    # Create random offset from nullspace vectors
    if nullspace:
        offset = offset_scale * sum(np.random.random() * np.array(v, dtype=float) for v in nullspace)
        offset = offset.flatten()
    else:
        offset = np.zeros(L.shape[1])

    # Create initial concentrations and add offset
    if base_concentrations is None:
        base_concentrations = np.ones(L.shape[1])
    C = scale * base_concentrations
    C = C + offset
    
    return C


def generate_positive_initial_concentrations_nnls(L, const_vals, min_conc=1e-8):
    n_species = L.shape[1]
    # Start from the NNLS solution as an initial guess
    C0, _ = nnls(L, const_vals)
    C0 = np.maximum(C0, min_conc)

    def variance(C):
        return np.var(C)

    constraints = [
        {'type': 'eq', 'fun': lambda C: L @ C - const_vals},
        {'type': 'ineq', 'fun': lambda C: C - min_conc}
    ]

    result = minimize(
        variance,
        C0,
        constraints=constraints,
        method='SLSQP',
        options={'ftol': 1e-6, 'maxiter': 5000}
    )

    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)
    return result.x


# def generate_feedforward_signaling_network(NR, NS_vec, include_reverse=True, include_uncatalyzed=True):
#     """
#     Generate a signaling network with NR receptors and layers of ligands specified by NS_vec.
    
#     Args:
#         NR: Number of receptors
#         NS_vec: List specifying number of ligands in each layer
        
#     Returns:
#         species_names: List of species names
#         reaction_strings: List of reaction strings
#         L: Conservation law matrix
#     """
#     reaction_strings = []
#     species_names = []

#     # Generate species names
#     for i in range(NR):
#         species_names.append(str(f'R{i}'))
#     for l in range(len(NS_vec)):
#         for j in range(NS_vec[l]):
#             species_names.append(f'S{l}{j}s') # first number is layer, second is index
#             species_names.append(f'S{l}{j}') # s indicates that it is in active form 

#     # Generate receptor-ligand reactions
#     for i in range(NR):
#         NS = NS_vec[0]
#         for j in range(NS):
#             reac = f'R{i}+S0{j}s -> R{i}+S0{j}'
#             reaction_strings.append(reac)
#             if include_reverse:
#                 reac = f'R{i}+S0{j} -> R{i}+S0{j}s'
#                 reaction_strings.append(reac)
                

#     # Generate ligand-ligand reactions between layers
#     for l in range(len(NS_vec)-1):
#         NS = NS_vec[l+1]
#         for i in range(NS_vec[0]):
#             for j in range(NS):
#                 reac = f'S{l}{i}+S{l+1}{j}s -> S{l}{i}+S{l+1}{j}'
#                 reaction_strings.append(reac)
#                 if include_reverse:
#                     reac = f'S{l}{i} + S{l+1}{j} -> S{l}{i} + S{l+1}{j}s'
#                     reaction_strings.append(reac)

#     # Generate ligand-ligand reactions between layers
#     if include_uncatalyzed:
#         for l in range(len(NS_vec)):
#             for j in range(NS_vec[l]):
#                 reac = f'S{l}{j}s -> S{l}{j}'
#                 reaction_strings.append(reac)
#                 if include_reverse:
#                     reac = f'S{l}{j} -> S{l}{j}s'
#                     reaction_strings.append(reac)

#     # Generate conservation law matrix
#     L = np.zeros((NR + np.sum(NS_vec), len(species_names)))
#     for i in range(NR):
#         L[i, i] = 1
#     flat_idx = 0
#     for l in range(len(NS_vec)):
#         for j in range(NS_vec[l]):
#             L[NR + flat_idx, species_names.index(f'S{l}{j}s')] = 1
#             L[NR + flat_idx, species_names.index(f'S{l}{j}')] = 1
#             flat_idx += 1

#     return species_names, reaction_strings, L


def generate_layered_feedforward_signaling_network(NR, NS_vec, p_r, p_f = 1, include_reverse=True, include_uncatalyzed=True, seed=None):
    """
    Generate a layered feedforward signaling network with NR receptors and layers of ligands specified by NS_vec.
    Similar interface to generate_dag_signaling_network but maintains layered structure.
    
    Args:
        NR: Number of receptors (upstream nodes)
        NS_vec: List specifying number of ligands in each layer
        p_f: Probability for forward connections between layers
        p_r: Probability for recurrent connections (backward connections within/across layers)
        include_reverse: Whether to include reverse reactions
        include_uncatalyzed: Whether to include uncatalyzed activation/deactivation
        seed: Random seed for reproducibility
        
    Returns:
        species_names: List of species names
        reaction_strings: List of reaction strings
        L: Conservation law matrix
        adjacency_matrix: The generated adjacency matrix (flattened to 2D for all substrates)
        input_substrates_list: List of substrate indices that are input substrates for each receptor
    """
    if seed is not None:
        np.random.seed(seed)
    
    reaction_strings = []
    species_names = []
    
    # Calculate total number of substrates across all layers
    total_NS = sum(NS_vec)
    
    # Generate species names (using flat indexing)
    for i in range(NR):
        species_names.append(f'R{i}')
    flat_idx = 0
    for l in range(len(NS_vec)):
        for j in range(NS_vec[l]):
            species_names.append(f'S{flat_idx}s')  # inactive form
            species_names.append(f'S{flat_idx}')   # active form
            flat_idx += 1
    
    # Create adjacency matrix for all substrates (flattened across layers)
    # This will be a total_NS x total_NS matrix
    adjacency_matrix = np.zeros((total_NS, total_NS))
    
    # Create mapping from (layer, index) to flattened index
    layer_to_flat = {}
    flat_idx = 0
    for l in range(len(NS_vec)):
        for j in range(NS_vec[l]):
            layer_to_flat[(l, j)] = flat_idx
            flat_idx += 1
    
    # Fill forward connections between layers with probability p_f
    for l in range(len(NS_vec) - 1):  # For each layer except the last
        for i in range(NS_vec[l]):    # For each substrate in current layer
            for j in range(NS_vec[l + 1]):  # For each substrate in next layer
                if np.random.rand() < p_f:
                    flat_i = layer_to_flat[(l, i)]
                    flat_j = layer_to_flat[(l + 1, j)]
                    adjacency_matrix[flat_i, flat_j] = 1
    
    # Generate receptor-substrate reactions (receptors can activate randomly selected input substrates)
    # Randomly select input substrates from first layer with probability p_f
    input_substrates_list = []
    for i in range(NR):
        input_substrates = []
        for j in range(NS_vec[0]):  # Only first layer can be input substrates
            if np.random.rand() < p_f:
                input_substrates.append(layer_to_flat[(0, j)])
        if len(input_substrates) == 0:  # ensure that there is at least one input substrate
            input_substrates = [0]
        input_substrates_list.append(input_substrates)
    
    # Add recurrent connections with probability p_r
    # These can be within layers or backward across layers
    for i in range(total_NS):
        for j in range(total_NS):
            if i != j and adjacency_matrix[i, j] == 0:  # Don't overwrite existing forward connections
                if np.random.rand() < p_r:
                    adjacency_matrix[i, j] = 1
    
    # Generate receptor-substrate reactions
    for i in range(NR):
        for flat_j in input_substrates_list[i]:
            reac = f'R{i}+S{flat_j}s -> R{i}+S{flat_j}'
            reaction_strings.append(reac)
            if include_reverse:
                reac = f'R{i}+S{flat_j} -> R{i}+S{flat_j}s'
                reaction_strings.append(reac)
    
    # Generate substrate-substrate reactions based on adjacency matrix
    for i in range(total_NS):
        for j in range(total_NS):
            if adjacency_matrix[i, j] == 1:  # If there's a connection from S_i to S_j
                reac = f'S{i}+S{j}s -> S{i}+S{j}'
                reaction_strings.append(reac)
                if include_reverse:
                    reac = f'S{i}+S{j} -> S{i}+S{j}s'
                    reaction_strings.append(reac)
    
    # Generate uncatalyzed activation/deactivation reactions
    if include_uncatalyzed:
        flat_idx = 0
        for l in range(len(NS_vec)):
            for j in range(NS_vec[l]):
                reac = f'S{flat_idx}s -> S{flat_idx}'
                reaction_strings.append(reac)
                if include_reverse:
                    reac = f'S{flat_idx} -> S{flat_idx}s'
                    reaction_strings.append(reac)
                flat_idx += 1
    
    # Generate conservation law matrix
    L = np.zeros((NR + total_NS, len(species_names)))
    
    # Conservation laws for receptors (each receptor is conserved)
    for i in range(NR):
        L[i, i] = 1
    
    # Conservation laws for substrates (each substrate has conserved total: S{flat_idx}s + S{flat_idx})
    flat_idx = 0
    for l in range(len(NS_vec)):
        for j in range(NS_vec[l]):
            L[NR + flat_idx, species_names.index(f'S{flat_idx}s')] = 1
            L[NR + flat_idx, species_names.index(f'S{flat_idx}')] = 1
            flat_idx += 1
    
    return species_names, reaction_strings, L, adjacency_matrix, input_substrates_list

def generate_random_signaling_network(NR, NS, NS_in, prob, include_reverse=False, include_uncatalyzed=True):
    """
    Generate a random signaling network with NR receptors and NS ligands, where NS_in ligands are randomly selected as inputs.
    
    Args:
        NR: Number of receptors
        NS: Total number of ligands
        NS_in: Number of input ligands that can be activated by receptors
        prob: Probability of including a catalytic activation between any two ligands
        include_reverse: Whether to include reverse reactions
        include_uncatalyzed: Whether to include uncatalyzed activation/deactivation
        
    Returns:
        species_names: List of species names
        reaction_strings: List of reaction strings
        L: Conservation law matrix
    """
    reaction_strings = []
    species_names = []

    # Generate species names
    for i in range(NR):
        species_names.append(str(f'R{i}'))
        for j in range(NS):
            species_names.append(f'S{j}s') # first number is layer, second is index
            species_names.append(f'S{j}') # s indicates that it is in active form

    # Randomly select NS_in indices from range(NS)
    input_indices = np.random.choice(NS, size=NS_in, replace=False)
    for i in range(NR):
        for input_index in input_indices:
            reac = f'R{i}+S{input_index}s -> R{i}+S{input_index}'
            reaction_strings.append(reac)
            if include_reverse:
                reac = f'R{i}+S{input_index} -> R{i}+S{input_index}s'
                reaction_strings.append(reac)

    
    # Generate ligand-ligand reactions between layers
    for i in range(NS):
        for j in range(NS):
            if i != j:
                if np.random.rand() < prob:
                    reac = f'S{i}+S{j}s -> S{i}+S{j}'
                    reaction_strings.append(reac)
                    if include_reverse:
                        reac = f'S{i} + S{j} -> S{i} + S{j}s'
                        reaction_strings.append(reac)
    

    if include_uncatalyzed:
        for i in range(NS):
            reac = f'S{i}s -> S{i}'
            reaction_strings.append(reac)
            if include_reverse:
                reac = f'S{i} -> S{i}s'
                reaction_strings.append(reac)


    # Generate conservation law matrix
    L = np.zeros((NR + NS, len(species_names)))
    for i in range(NR):
        L[i, i] = 1
    for j in range(NS):
        L[NR + j, species_names.index(f'S{j}s')] = 1
        L[NR + j, species_names.index(f'S{j}')] = 1

    return species_names, reaction_strings, L, input_indices



def generate_dag_signaling_network(NR, NS, p_f, p_r, include_reverse=False, include_uncatalyzed=True, seed=None):
    """
    Generate a signaling network based on a directed acyclic graph (DAG) structure.
    
    Args:
        NR: Number of receptors (upstream nodes)
        NS: Number of substrates in the DAG
        p_f: Sparsity fraction for forward connections (upper triangular matrix)
        p_r: Probability for backward connections (lower triangular matrix)
        include_reverse: Whether to include reverse reactions
        include_uncatalyzed: Whether to include uncatalyzed activation/deactivation
        seed: Random seed for reproducibility
        
    Returns:
        species_names: List of species names
        reaction_strings: List of reaction strings
        L: Conservation law matrix
        adjacency_matrix: The generated adjacency matrix
        input_substrates: List of substrate indices that are input substrates
    """
    if seed is not None:
        np.random.seed(seed)
    
    reaction_strings = []
    species_names = []
    
    # Generate species names
    for i in range(NR):
        species_names.append(f'R{i}')
    for j in range(NS):
        species_names.append(f'S{j}s')  # inactive form
        species_names.append(f'S{j}')   # active form
    
    # Create adjacency matrix for the DAG
    # Start with upper triangular matrix (forward connections)
    adjacency_matrix = np.zeros((NS, NS))
    
    # Fill upper triangular part with probability p_f
    for i in range(NS):
        for j in range(i + 1, NS):
            if np.random.rand() < p_f:
                adjacency_matrix[i, j] = 1

    # Generate receptor-substrate reactions (receptors can activate randomly selected input substrates)
    # Randomly select input substrates with probability p_f
    input_substrates_list = []
    for i in range(NR):
        input_substrates = []
        for j in range(NS):
            if np.random.rand() < p_f:
                input_substrates.append(j)
        if len(input_substrates) == 0: # ensure that there is at least one input substrate
            input_substrates = [0]
        input_substrates_list.append(input_substrates)
    
    # Add some backward connections in lower triangular part with probability p_r
    # for i in range(1, NS):
    #     for j in range(i):
    #         if np.random.rand() < p_r:
    #             adjacency_matrix[i, j] = 1

    G_original = get_digraph_from_adjacency_matrix(adjacency_matrix, input_substrates_list, NR, NS)
    adjacency_matrix_original = adjacency_matrix.copy()
    n_paths = count_simple_paths(G_original, f'R0', f'S{NS-1}')
    for i in range(1, NS):
        for j in range(i):
            if np.random.rand() < p_r:
                adjacency_matrix[i, j] = 1
                G = get_digraph_from_adjacency_matrix(adjacency_matrix, input_substrates_list, NR, NS)
                if count_simple_paths(G, f'R0', f'S{NS-1}') != n_paths:
                    adjacency_matrix[i, j] = 0
    
    for i in range(NR):
        for j in input_substrates_list[i]:
            reac = f'R{i}+S{j}s -> R{i}+S{j}'
            reaction_strings.append(reac)
            if include_reverse:
                reac = f'R{i}+S{j} -> R{i}+S{j}s'
                reaction_strings.append(reac)
    
    # Generate substrate-substrate reactions based on adjacency matrix
    # Only include reactions for directed edges in the DAG (i -> j means active S_i can catalyze activation of S_j)
    for i in range(NS):
        for j in range(NS):
            if adjacency_matrix[i, j] == 1:  # If there's a connection from S_i to S_j
                reac = f'S{i}+S{j}s -> S{i}+S{j}'
                reaction_strings.append(reac)
                if include_reverse:
                    reac = f'S{i}+S{j} -> S{i}+S{j}s'
                    reaction_strings.append(reac)
    
    # Generate uncatalyzed activation/deactivation reactions
    if include_uncatalyzed:
        for j in range(NS):
            reac = f'S{j}s -> S{j}'
            reaction_strings.append(reac)
            if include_reverse:
                reac = f'S{j} -> S{j}s'
                reaction_strings.append(reac)
    
    # Generate conservation law matrix
    L = np.zeros((NR + NS, len(species_names)))
    
    # Conservation laws for receptors (each receptor is conserved)
    for i in range(NR):
        L[i, i] = 1
    
    # Conservation laws for substrates (each substrate has conserved total: S{j}s + S{j})
    for j in range(NS):
        L[NR + j, species_names.index(f'S{j}s')] = 1
        L[NR + j, species_names.index(f'S{j}')] = 1
    
    return species_names, reaction_strings, L, adjacency_matrix, input_substrates_list

def add_recurrent_connections(adjacency_matrix, reaction_strings, include_reverse, input_substrates_list, NR, NS, p_r, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    G_original = get_digraph_from_adjacency_matrix(adjacency_matrix, input_substrates_list, NR, NS)
    adjacency_matrix_copy = adjacency_matrix.copy()
    n_paths = count_simple_paths(G_original, f'R0', f'S{NS-1}')
    for i in range(1, NS):
        for j in range(i):
            if np.random.rand() < p_r:
                adjacency_matrix_copy[i, j] = 1
                G = get_digraph_from_adjacency_matrix(adjacency_matrix_copy, input_substrates_list, NR, NS)
                if count_simple_paths(G, f'R0', f'S{NS-1}') != n_paths:
                    adjacency_matrix_copy[i, j] = 0

    for i in range(NR):
        for j in input_substrates_list[i]:
            reac = f'R{i}+S{j}s -> R{i}+S{j}'
            reaction_strings.append(reac)
            if include_reverse:
                reac = f'R{i}+S{j} -> R{i}+S{j}s'
                reaction_strings.append(reac)

    return reaction_strings, adjacency_matrix_copy

def generate_specified_dag_signaling_network(adjacency_matrix, input_substrates_list, include_reverse=False, include_uncatalyzed=True, seed=None):
    """
    Generate a signaling network based on a directed acyclic graph (DAG) structure.
    
    Args:
        NR: Number of receptors (upstream nodes)
        NS: Number of substrates in the DAG
        adjacency_matrix: The adjacency matrix for substrate-substrate connections
        input_substrates_list: List of substrate indices that are input substrates
        include_reverse: Whether to include reverse reactions
        include_uncatalyzed: Whether to include uncatalyzed activation/deactivation
        seed: Random seed for reproducibility
        
    Returns:
        species_names: List of species names    
        reaction_strings: List of reaction strings
        L: Conservation law matrix
        adjacency_matrix: The adjacency matrix for substrate-substrate connections
        input_substrates_list: List of substrate indices that are input substrates
    """
    if seed is not None:
        np.random.seed(seed)
    
    reaction_strings = []
    species_names = []
    
    NS = adjacency_matrix.shape[0]
    NR = len(input_substrates_list)
    
    # Generate species names
    for i in range(NR):
        species_names.append(f'R{i}')
    for j in range(NS):
        species_names.append(f'S{j}s')  # inactive form
        species_names.append(f'S{j}')   # active form
    
    
    for i in range(NR):
        for j in input_substrates_list[i]:
            reac = f'R{i}+S{j}s -> R{i}+S{j}'
            reaction_strings.append(reac)
            if include_reverse:
                reac = f'R{i}+S{j} -> R{i}+S{j}s'
                reaction_strings.append(reac)
    
    # Generate substrate-substrate reactions based on adjacency matrix
    # Only include reactions for directed edges in the DAG (i -> j means active S_i can catalyze activation of S_j)
    for i in range(NS):
        for j in range(NS):
            if adjacency_matrix[i, j] == 1:  # If there's a connection from S_i to S_j
                reac = f'S{i}+S{j}s -> S{i}+S{j}'
                reaction_strings.append(reac)
                if include_reverse:
                    reac = f'S{i}+S{j} -> S{i}+S{j}s'
                    reaction_strings.append(reac)
    
    # Generate uncatalyzed activation/deactivation reactions
    if include_uncatalyzed:
        for j in range(NS):
            reac = f'S{j}s -> S{j}'
            reaction_strings.append(reac)
            if include_reverse:
                reac = f'S{j} -> S{j}s'
                reaction_strings.append(reac)
    
    # Generate conservation law matrix
    L = np.zeros((NR + NS, len(species_names)))
    
    # Conservation laws for receptors (each receptor is conserved)
    for i in range(NR):
        L[i, i] = 1
    
    # Conservation laws for substrates (each substrate has conserved total: S{j}s + S{j})
    for j in range(NS):
        L[NR + j, species_names.index(f'S{j}s')] = 1
        L[NR + j, species_names.index(f'S{j}')] = 1
    
    return species_names, reaction_strings, L

def expand_catalyzed_reactions(species_names, reaction_strings, L):
    """
    Expand catalyzed reactions into elementary steps with intermediate complexes.
    
    Args:
        species_names: List of species names
        reaction_strings: List of reaction strings
        L: Linkage matrix (numpy array)
        
    Returns:
        tuple: (new_species_names, new_reaction_strings, new_L)
            - new_species_names: Updated list of species names including intermediates
            - new_reaction_strings: Updated list of reaction strings with expanded reactions
            - new_L: Updated linkage matrix
    """
    def has_catalyst(reaction_string):
        """
        Check if any species appears on both sides of the reaction.
        
        Args:
            reaction_string: Reaction string like 'S0+S1s -> S0+S1'
            
        Returns:
            bool: True if any species appears on both sides
        """
        # Split into reactants and products
        parts = reaction_string.split('->')
        if len(parts) != 2:
            return False
        
        reactants_str = parts[0].strip()
        products_str = parts[1].strip()
        
        # Parse species from each side
        reactants = set(s.strip() for s in reactants_str.split('+'))
        products = set(s.strip() for s in products_str.split('+'))
        
        # Check if any species appears in both
        return len(reactants & products) > 0


    def split_catalyzed_reaction(reaction_string):
        """
        Split a catalyzed reaction into two elementary steps with an intermediate complex.
        Assumes a catalyst species appears on both sides.
        
        Args:
            reaction_string: Reaction string like 'S0+S1s -> S0+S1'
            
        Returns:
            tuple: (intermediate_name, reaction1, reaction2)
                - intermediate_name: Name of the intermediate complex (e.g., 'S0_S1s')
                - reaction1: Complex formation reaction (e.g., 'S0+S1s -> S0_S1s')
                - reaction2: Complex dissociation reaction (e.g., 'S0_S1s -> S0+S1')
        """
        # Split into reactants and products
        parts = reaction_string.split('->')
        reactants_str = parts[0].strip()
        products_str = parts[1].strip()
        
        # Parse species from each side
        reactants = [s.strip() for s in reactants_str.split('+')]
        products = [s.strip() for s in products_str.split('+')]
        
        # Create intermediate name by joining all reactants with underscores
        intermediate_name = '_'.join(reactants)
        
        # Create reaction 1: reactants -> intermediate (complex formation)
        reaction1 = f"{'+'.join(reactants)} -> {intermediate_name}"
        
        # Create reaction 2: intermediate -> products (complex dissociation)
        reaction2 = f"{intermediate_name} -> {'+'.join(products)}"
        
        return intermediate_name, reaction1, reaction2, list(set(reactants) | set(products))

    new_species_names = species_names.copy()
    new_reaction_strings = reaction_strings.copy()
    new_L = L.copy()
    for reaction_string in reaction_strings:
        if has_catalyst(reaction_string):
            intermediate_name, reaction1, reaction2, involved_species = split_catalyzed_reaction(reaction_string)
            involved_indices = [species_names.index(species) for species in involved_species]
            new_species_names.append(intermediate_name)
            new_reaction_strings.remove(reaction_string)
            new_reaction_strings.append(reaction1)
            new_reaction_strings.append(reaction2)
            new_l = np.zeros(len(new_L))
            for i in range(len(new_L)):
                if L[i, involved_indices].sum() > 0:
                    new_l[i] = 1
            new_L = np.column_stack([new_L, new_l])
    
    return new_species_names, new_reaction_strings, new_L

def get_digraph_from_adjacency_matrix(adjacency_matrix, input_substrates_list, NR, NS):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add receptor nodes
    for i in range(NR):
        G.add_node(f'R{i}', node_type='receptor')
    
    # Add substrate nodes
    for j in range(NS):
        G.add_node(f'S{j}', node_type='substrate')
    
    # Add receptor-substrate edges (for input substrates)
    for i in range(NR):
        for j in input_substrates_list[i]:
            G.add_edge(f'R{i}', f'S{j}', edge_type='receptor')
    
    # Add substrate-substrate edges based on adjacency matrix
    for i in range(NS):
        for j in range(NS):
            if adjacency_matrix[i, j] == 1:
                G.add_edge(f'S{i}', f'S{j}', edge_type='substrate')
    return G

def count_simple_paths(G, source, target):
    try:
        # Get all simple paths and count them
        return list(nx.all_simple_paths(G, source, target))
    except nx.NetworkXNoPath:
        return 0

def visualize_dag_signaling_network(adjacency_matrix, input_substrates_list, NR, NS, num_outs = 1,node_size=600, arrow_size=25, font_size=12, ax=None):
    """
    Visualize the DAG signaling network including receptor connections.
    
    Args:
        adjacency_matrix: The adjacency matrix for substrate-substrate connections
        input_substrates_list: List of substrate indices that are input substrates
        NR: Number of receptors
        NS: Number of substrates
        ax: Optional matplotlib axis to plot into. If None, creates a new figure and axis.

        title: Title for the plot
        
    Returns:
        matplotlib figure object (if ax is None) or the provided axis
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    
    # Create a directed graph
    G = get_digraph_from_adjacency_matrix(adjacency_matrix, input_substrates_list, NR, NS)
       
    # Create the plot if axis not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        return_fig = True
    else:
        fig = ax.get_figure()
        return_fig = False

    #print(nx.is_directed_acyclic_graph(G))
    if nx.is_directed_acyclic_graph(G):

        for layer, nodes in enumerate(nx.topological_generations(G)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
            for node in nodes:
                G.nodes[node]["layer"] = layer

        # Check if G is a directed acyclic graph
    
        # If DAG, use multipartite layout
        pos = nx.multipartite_layout(G, subset_key="layer")
        rotation_matrix = np.array([[0, 1], [-1, 0]])  # 90 degree counterclockwise rotation
        for node in pos:
            pos[node] = rotation_matrix @ np.array(pos[node])
    else:
        # If not DAG, use spring layout
        #pos = nx.spring_layout(G, k=2, iterations=50)
        pos = nx.circular_layout(G)
    
    # Draw nodes
    receptor_nodes = [node for node in G.nodes() if node.startswith('R')]
    substrate_nodes = [node for node in G.nodes() if node.startswith('S')]
    
    nx.draw_networkx_nodes(G, pos, nodelist=receptor_nodes, 
                          node_color='lightblue', node_size=node_size, 
                          node_shape='s', ax=ax, label='Receptors')
    nx.draw_networkx_nodes(G, pos, nodelist=substrate_nodes[:-num_outs], 
                          node_color='pink', node_size=node_size, 
                          node_shape='o', ax=ax, label='Substrates')
    nx.draw_networkx_nodes(G, pos, nodelist=substrate_nodes[-num_outs:], 
                          node_color='lightgreen', node_size=node_size, 
                          node_shape='o', ax=ax, label='Substrates')
    
    # Draw edges with different colors
    receptor_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'receptor']
    substrate_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'substrate']
    
    nx.draw_networkx_edges(G, pos, edgelist=receptor_edges, 
                          edge_color='blue', arrows=True, arrowsize=arrow_size, 
                          ax=ax, label='Receptor activation', connectionstyle="arc3,rad=-0.4")
    nx.draw_networkx_edges(G, pos, edgelist=substrate_edges, 
                          edge_color='red', arrows=True, arrowsize=arrow_size, 
                          ax=ax, label='Substrate catalysis', connectionstyle="arc3,rad=0.4")
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, font_family='Arial')
    
    if return_fig:
        plt.tight_layout()
        return fig
    else:
        return ax

def generate_random_dimerization_network(NS, prob, include_reverse=False):
    """
    Generate a random signaling network with NR receptors and NS ligands, where NS_in ligands are randomly selected as inputs.
    
    Args:
        NR: Number of receptors
        NS: Total number of ligands
        NS_in: Number of input ligands that can be activated by receptors
        prob: Probability of including a catalytic activation between any two ligands
        include_reverse: Whether to include reverse reactions
        include_uncatalyzed: Whether to include uncatalyzed activation/deactivation
        
    Returns:
        species_names: List of species names
        reaction_strings: List of reaction strings
        L: Conservation law matrix
    """
    reaction_strings = []
    species_names = []
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    monomer_names = [f'{alphabet[i]}' for i in range(NS)]
    
    dimer_names = []
    for i in range(NS):
        for j in range(i+1, NS):
            if np.random.rand() < prob:
                dimer_names.append(f'{monomer_names[i]}{monomer_names[j]}')
                reac = f'{monomer_names[i]}+{monomer_names[j]} -> {dimer_names[-1]}'
                reaction_strings.append(reac)
                if include_reverse:
                    reac = f'{dimer_names[-1]} -> {monomer_names[i]}+{monomer_names[j]}'
                    reaction_strings.append(reac)
    
    species_names = monomer_names + dimer_names
    

    # Get indices of all species containing 'A'
    L = []
    for monomer in monomer_names:
        mon_indices = [i for i, name in enumerate(species_names) if monomer in name]
        lrow = np.zeros(len(species_names))
        lrow[mon_indices] = 1
        L.append(lrow)
    L = np.array(L)
    
    return species_names, reaction_strings, L

def generate_random_first_order_network(n_nodes, input_reactions_list, prob, extra_reactions = [], include_reverse=False):
    """
    Generate a random signaling network with NR receptors and NS ligands, where NS_in ligands are randomly selected as inputs.
    
    Args:
        NR: Number of receptors
        n_nodes: Total number of ligands
        NS_in: Number of input ligands that can be activated by receptors
        prob: Probability of including a catalytic activation between any two ligands
        include_reverse: Whether to include reverse reactions
        include_uncatalyzed: Whether to include uncatalyzed activation/deactivation
        
    Returns:
        species_names: List of species names
        reaction_strings: List of reaction strings
        L: Conservation law matrix
    """
    reaction_strings = []
    species_names = []
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    monomer_names = [f'{alphabet[i]}' for i in range(n_nodes)]

    NR = len(input_reactions_list)
    
    # Generate species names
    for i in range(NR):
        species_names.append(f'R{i}')
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if np.random.rand() < prob:
                reac = f'{monomer_names[i]} -> {monomer_names[j]}'
                reaction_strings.append(reac)
                if include_reverse:
                    reac = f'{monomer_names[j]} -> {monomer_names[i]}'
                    reaction_strings.append(reac)
    
    species_names = species_names + monomer_names
    reaction_strings = reaction_strings + extra_reactions
    for i in range(NR):
        input_reactions = input_reactions_list[i]
        reaction_strings = reaction_strings + input_reactions


#    for i in range(NR):
#     input_reactions = input_reactions_list[i]
#     for reaction in input_reactions:
#         parts = reaction.split('->')
#         reac = f'R{i}+{parts[0]} -> R{i}+{parts[1]}'
#         reaction_strings.append(reac)


    
    # Generate conservation law matrix
    L = np.zeros((NR + 1, len(species_names)))
    
    # Conservation laws for receptors (each receptor is conserved)
    for i in range(NR):
        L[i, i] = 1
    
    # Conservation laws for substrates (each substrate has conserved total: S{j}s + S{j})
    L[NR, NR:] = np.ones(n_nodes)
    
    
    return species_names, reaction_strings, L

