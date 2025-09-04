"""
Generation module for reaction networks.

This module contains functions for generating random reaction networks,
linkage groups, and other network structures.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
import sympy
from scipy.optimize import nnls
from scipy.optimize import minimize


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


def generate_feedforward_signaling_network(NR, NL_vec, include_reverse=False, include_uncatalyzed=True):
    """
    Generate a signaling network with NR receptors and layers of ligands specified by NL_vec.
    
    Args:
        NR: Number of receptors
        NL_vec: List specifying number of ligands in each layer
        
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
    for l in range(len(NL_vec)):
        for j in range(NL_vec[l]):
            species_names.append(f'L{l}{j}') # first number is layer, second is index
            species_names.append(f'L{l}{j}s') # s indicates that it is in active form 

    # Generate receptor-ligand reactions
    for i in range(NR):
        NL = NL_vec[0]
        for j in range(NL):
            reac = f'R{i}+L0{j} -> R{i}+L0{j}s'
            reaction_strings.append(reac)
            if include_reverse:
                reac = f'R{i}+L0{j}s -> R{i}+L0{j}'
                reaction_strings.append(reac)
                

    # Generate ligand-ligand reactions between layers
    for l in range(len(NL_vec)-1):
        NL = NL_vec[l+1]
        for i in range(NL_vec[0]):
            for j in range(NL):
                reac = f'L{l}{i}s+L{l+1}{j} -> L{l}{i}s+L{l+1}{j}s'
                reaction_strings.append(reac)
                if include_reverse:
                    reac = f'L{l}{i}s + L{l+1}{j}s -> L{l}{i}s + L{l+1}{j}'
                    reaction_strings.append(reac)

    # Generate ligand-ligand reactions between layers
    if include_uncatalyzed:
        for l in range(len(NL_vec)):
            for j in range(NL_vec[l]):
                reac = f'L{l}{j} -> L{l}{j}s'
                reaction_strings.append(reac)
                if include_reverse:
                    reac = f'L{l}{j}s -> L{l}{j}'
                    reaction_strings.append(reac)

    # Generate conservation law matrix
    L = np.zeros((NR + np.sum(NL_vec), len(species_names)))
    for i in range(NR):
        L[i, i] = 1
    for l in range(len(NL_vec)):
        for j in range(NL_vec[l]):
            L[NR + l*NL + j, species_names.index(f'L{l}{j}')] = 1
            L[NR + l*NL + j, species_names.index(f'L{l}{j}s')] = 1

    return species_names, reaction_strings, L

def generate_random_signaling_network(NR, NL, NL_in, prob, include_reverse=False, include_uncatalyzed=True):
    """
    Generate a random signaling network with NR receptors and NL ligands, where NL_in ligands are randomly selected as inputs.
    
    Args:
        NR: Number of receptors
        NL: Total number of ligands
        NL_in: Number of input ligands that can be activated by receptors
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
        for j in range(NL):
            species_names.append(f'L{j}') # first number is layer, second is index
            species_names.append(f'L{j}s') # s indicates that it is in active form

    # Randomly select NL_in indices from range(NL)
    input_indices = np.random.choice(NL, size=NL_in, replace=False)
    for i in range(NR):
        for input_index in input_indices:
            reac = f'R{i}+L{input_index} -> R{i}+L{input_index}s'
            reaction_strings.append(reac)
            if include_reverse:
                reac = f'R{i}+L{input_index}s -> R{i}+L{input_index}'
                reaction_strings.append(reac)

    
    # Generate ligand-ligand reactions between layers
    for i in range(NL):
        for j in range(NL):
            if i != j:
                if np.random.rand() < prob:
                    reac = f'L{i}s+L{j} -> L{i}s+L{j}s'
                    reaction_strings.append(reac)
                    if include_reverse:
                        reac = f'L{i}s+L{j}s -> L{i}s+L{j}'
                        reaction_strings.append(reac)
    

    if include_uncatalyzed:
        for i in range(NL):
            reac = f'L{i} -> L{i}s'
            reaction_strings.append(reac)
            if include_reverse:
                reac = f'L{i}s -> L{i}'
                reaction_strings.append(reac)


    # Generate conservation law matrix
    L = np.zeros((NR + NL, len(species_names)))
    for i in range(NR):
        L[i, i] = 1
    for j in range(NL):
        L[NR + j, species_names.index(f'L{j}')] = 1
        L[NR + j, species_names.index(f'L{j}s')] = 1

    return species_names, reaction_strings, L, input_indices

def generate_random_dimerization_network(NL, prob, include_reverse=False):
    """
    Generate a random signaling network with NR receptors and NL ligands, where NL_in ligands are randomly selected as inputs.
    
    Args:
        NR: Number of receptors
        NL: Total number of ligands
        NL_in: Number of input ligands that can be activated by receptors
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
    monomer_names = [f'{alphabet[i]}' for i in range(NL)]
    
    dimer_names = []
    for i in range(NL):
        for j in range(i+1, NL):
            if np.random.rand() < prob:
                dimer_names.append(f'{monomer_names[i]}{monomer_names[j]}')
                reac = f'{monomer_names[i]}+{monomer_names[j]} -> {dimer_names[-1]}'
                reaction_strings.append(reac)
                if include_reverse:
                    reac = f'{dimer_names[-1]} -> {monomer_names[i]}+{monomer_names[j]}'
                    reaction_strings.append(reac)
    
    species_names = monomer_names + dimer_names
    

    # Get indices of all species containing 'A'
    A_indices = [i for i, name in enumerate(species_names) if 'A' in name]
    L = []
    for monomer in monomer_names:
        mon_indices = [i for i, name in enumerate(species_names) if monomer in name]
        lrow = np.zeros(len(species_names))
        lrow[mon_indices] = 1
        L.append(lrow)
    L = np.array(L)
    
    return species_names, reaction_strings, L