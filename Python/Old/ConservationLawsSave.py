import sympy
import numpy as np
import time
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import string
import random
import itertools
import sympy

from dataclasses import dataclass
from typing import List, Tuple

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def random_connected_graph(n_nodes, n_edges, directed=False, seed=None):
    rng = np.random.default_rng(seed)
    nodes = list(range(n_nodes))
    if directed:
        # Start with a random spanning arborescence (directed tree)
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
        # Start with a random spanning tree
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


def random_conservation_laws(n_cons, n_species, probs, seed=None):
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
    return np.array(L)


def generate_string_vector_dict(n):
    """
    Generates a dictionary mapping complex names (as strings) to their
    vector representations. Correctly handles complexes like "A+E" and "E+A"
    as identical by creating a canonical (sorted) representation.
    """
    letters = list(string.ascii_uppercase[:n])
    letter_indices = {letter: i for i, letter in enumerate(letters)}
    result = {}

    # Single letters
    for letter in letters:
        vec = np.zeros(n, dtype=int)
        vec[letter_indices[letter]] = 1
        result[letter] = vec

    # Pairs
    # Use itertools.combinations_with_replacement to get unique pairs (order doesn't matter)
    for a, b in itertools.combinations_with_replacement(letters, 2):
        vec = np.zeros(n, dtype=int)
        vec[letter_indices[a]] += 1
        vec[letter_indices[b]] += 1
        
        # Create a canonical key. Since combinations_with_replacement produces
        # sorted output, we can just join them.
        key = f"{a}+{b}"
        result[key] = vec
        
    return result


def find_valid_partners(c_b, L, complexes_dict):
    s1 = complexes_dict[c_b]
    valid_partners = {}
    for c in complexes_dict:
        s_test = complexes_dict[c]
        diff = s1 - s_test
        # Check if this complex satisfies conservation laws
        if all(np.dot(l_vec, diff) == 0 for l_vec in L):
            valid_partners[c] = s_test
    return valid_partners


def generate_pairing_groups(L, complexes_dict):
    pairing_groups = []
    available_complexes = set(complexes_dict.keys())
    while available_complexes:
        c = available_complexes.pop()
        valid_partners = find_valid_partners(c, L, complexes_dict)
        # Ensure the group is self-contained and we only store it once
        canonical_group = frozenset(valid_partners.keys())
        is_new = True
        for group in pairing_groups:
            if canonical_group == frozenset(group.keys()):
                is_new = False
                break
        if is_new:
             pairing_groups.append(valid_partners)
        
    return pairing_groups


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

class ReactionNetwork:
    """
    A class to build, store, and simulate a chemical reaction network.

    This class encapsulates the entire structure of a reaction network,
    including its species, complexes, reactions, and conservation laws.
    The constructor attempts to build a valid network with a specified
    deficiency.
    """

    def __init__(self, n_complexes, n_reactions, n_lcs, n_species, L, n_cons, seed=None, force_reverse=True):
        """
        Constructs a ReactionNetwork instance. This is equivalent to the
        old `build_reaction_network` function.
        """
        if seed is None:
            seed = random.randint(0, 1_000_000)
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.n_tries = 100

        for n in range(self.n_tries):
            try:
                self.force_reverse = force_reverse
                self.complexes_per_class, self.reactions_per_class = self._random_partition_complexes_and_reactions(
                    n_complexes, n_reactions, n_lcs)
                self.complexes_dict = generate_string_vector_dict(n_species)
                self.pairing_groups = generate_pairing_groups(L, self.complexes_dict)
                
                self.linkage_groups = self._generate_linkage_class_graphs()
                self.assignments = self._assign_complexes_to_linkage_groups()

                self.species_names = list(string.ascii_uppercase[:n_species])
                self.Y, self.all_complexes = self._build_Y_matrix()
                self.reactions = self._build_reaction_list(random_rates=True)
                self.A = self._build_A_matrix_incidence()
                self.Psi = self._build_Psi_function()
                self.rhs = self._make_ode_rhs()
                self.L = L

                # deficiency check is now inside the try block
                rank_YA = np.linalg.matrix_rank(self.Y @ self.A)
                if n_species - rank_YA == n_cons:
                    # Success, break the loop
                    break

            except (ValueError, nx.NetworkXError):
                # If any part of the setup fails, continue to the next try
                continue
        else:
            # This 'else' block executes only if the 'for' loop completes without a 'break'.
            # This means all 'n_tries' attempts failed.
            raise ValueError(f"Could not find a valid reaction network with the specified parameters after {self.n_tries} attempts.")

    def _random_partition_complexes_and_reactions(self, n_complexes, n_reactions, n_lcs):
        min_complexes_per_class = 2
        min_total_complexes = n_lcs * min_complexes_per_class
        if n_complexes < min_total_complexes:
            raise ValueError("Not enough complexes to give at least 2 to each class.")
        
        remaining_complexes = n_complexes - min_total_complexes
        extra_complexes = self.rng.multinomial(remaining_complexes, [1/n_lcs]*n_lcs) if remaining_complexes > 0 else np.zeros(n_lcs, dtype=int)
        complexes_per_class = min_complexes_per_class + extra_complexes

        min_reactions_per_class = complexes_per_class - 1
        if self.force_reverse:
            max_reactions_per_class = (complexes_per_class * (complexes_per_class - 1) // 2).astype(int)
        else:
            max_reactions_per_class = (complexes_per_class * (complexes_per_class - 1)).astype(int)
        
        min_total_reactions = min_reactions_per_class.sum()
        max_total_reactions = max_reactions_per_class.sum()
        
        if n_reactions < min_total_reactions:
            print(f"Setting n_reactions to minimum value: {min_total_reactions}")
            n_reactions = min_total_reactions
        if n_reactions > max_total_reactions:
            print(f"Setting n_reactions to maximum value: {max_total_reactions}")
            n_reactions = max_total_reactions

        remaining_reactions = n_reactions - min_total_reactions
        reactions_per_class = min_reactions_per_class.copy()
        
        for _ in range(remaining_reactions):
            eligible = np.where(reactions_per_class < max_reactions_per_class)[0]
            if not eligible.any(): break
            reactions_per_class[self.rng.choice(eligible)] += 1

        return complexes_per_class, reactions_per_class

    def _generate_linkage_class_graphs(self):
        graphs = []
        for n_nodes, n_edges in zip(self.complexes_per_class, self.reactions_per_class):
            g_seed = int(self.rng.integers(1e9))
            G = random_connected_graph(n_nodes, n_edges, directed=not self.force_reverse, seed=g_seed)
            graphs.append(G)
        return graphs

    def _assign_complexes_to_linkage_groups(self):
        pairing_groups_copy = self.pairing_groups[:]
        rng = random.Random(int(self.rng.integers(1e9)))
        used_complexes = set()
        assignments = []

        for G in self.linkage_groups:
            n_nodes = G.number_of_nodes()
            found = False
            rng.shuffle(pairing_groups_copy)
            for pairing_group in pairing_groups_copy:
                available_complexes = [c for c in pairing_group if c not in used_complexes]
                if len(available_complexes) >= n_nodes:
                    chosen_complexes = rng.sample(available_complexes, n_nodes)
                    node_assignment = {node: complex_name for node, complex_name in zip(G.nodes(), chosen_complexes)}
                    assignments.append(node_assignment)
                    used_complexes.update(chosen_complexes)
                    found = True
                    break
            if not found:
                raise ValueError(f"Not enough complexes for linkage group with {n_nodes} nodes.")

        for G, node_assignment in zip(self.linkage_groups, assignments):
            nx.relabel_nodes(G, node_assignment, copy=False)
        return assignments

    def _build_Y_matrix(self):
        all_complexes = []
        for group in self.assignments:
            all_complexes.extend(sorted(group.values()))
        all_complexes = list(dict.fromkeys(all_complexes))
        
        Y = np.zeros((len(self.species_names), len(all_complexes)), dtype=int)
        for j, cname in enumerate(all_complexes):
            Y[:, j] = self.complexes_dict[cname]
        return Y, all_complexes

    def _build_reaction_list(self, random_rates=True):
        complex_to_idx = {c: i for i, c in enumerate(self.all_complexes)}
        reactions = []
        for G in self.linkage_groups:
            for u, v in G.edges():
                # Add the forward reaction
                forward_rate = self.rng.uniform(0.1, 2.0) if random_rates else 1.0
                reactions.append((complex_to_idx[u], complex_to_idx[v], forward_rate))
                
                # If reversible, add the backward reaction with its own rate
                if self.force_reverse:
                    backward_rate = self.rng.uniform(0.1, 2.0) if random_rates else 1.0
                    reactions.append((complex_to_idx[v], complex_to_idx[u], backward_rate))
        return reactions
        
    def _build_A_matrix_incidence(self):
        n = len(self.all_complexes)
        A = np.zeros((n, n))
        for i, j, k in self.reactions:
            A[j, i] += k
            A[i, i] -= k
        return A

    def _build_Psi_function(self):
        def Psi(C):
            vals = []
            for i, j, k in self.reactions:
                monomial = np.prod(C ** self.Y[:, i])
                vals.append(monomial)
            return np.array(vals)
        return Psi

    def _make_ode_rhs(self):
        def rhs(C):
            v_monomials = self.Psi(C)
            dCdt = np.zeros(self.Y.shape[0])
            for idx, (i, j, k) in enumerate(self.reactions):
                rate = k * v_monomials[idx]
                stoichiometric_change = self.Y[:, j] - self.Y[:, i]
                dCdt += rate * stoichiometric_change
            return dCdt
        return rhs

    def update_rates(self, new_rates: List[float]):
        """
        Updates the reaction rate constants in place.
        `new_rates` should be a list of floats of the same length as the
        number of reactions.
        """
        if len(new_rates) != len(self.reactions):
            raise ValueError(f"Expected {len(self.reactions)} rates, but got {len(new_rates)}.")

        self.reactions = [(i, j, new_rate) for (i, j, _), new_rate in zip(self.reactions, new_rates)]
        self.A = self._build_A_matrix_incidence()
        self.rhs = self._make_ode_rhs()

    def integrate(self, C0, t_span=(0, 5), num_points=20000):
        """
        Integrates the ODE system for the reaction network.
        """
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        sol = solve_ivp(lambda t, C: self.rhs(C), t_span, C0, t_eval=t_eval, vectorized=False)
        return sol, sol.y[:,-1]
    
    def get_symbolic_rhs(self):
        """
        Returns the symbolic right-hand side of the ODEs for the reaction network.
        """
        import sympy

        # Create symbolic variables for each species
        species_syms = sympy.symbols(self.species_names)
        
        # Create descriptive names for the symbolic rate constants
        rate_names = []
        for r_idx, (i, j, _) in enumerate(self.reactions):
            # Creates names like: k_0_C+C_B+C
            reactant_str = self.all_complexes[i]
            product_str = self.all_complexes[j]
            # rate_names.append(f'k_{r_idx};{reactant_str}->{product_str}')
            rate_names.append(f'k_{r_idx}')

        rate_syms = sympy.symbols(rate_names)

        # Initialize a list to hold the symbolic ODE for each species
        dCdt_sym = [sympy.Integer(0)] * len(species_syms)

        # Iterate through each reaction to build the symbolic ODEs
        for idx, (i, j, k_val) in enumerate(self.reactions):
            rate_k = rate_syms[idx]
            
            # Build the monomial for the reactant complex
            reactant_complex_stoich = self.Y[:, i]
            monomial = sympy.Integer(1)
            for species_idx, exponent in enumerate(reactant_complex_stoich):
                if exponent > 0:
                    monomial *= species_syms[species_idx]**exponent
            
            # Symbolic representation of the reaction rate
            reaction_rate_sym = rate_k * monomial
            
            # Stoichiometric change vector for the reaction
            stoich_change = self.Y[:, j] - self.Y[:, i]
            
            # Add this reaction's contribution to each species' ODE
            for species_idx, change in enumerate(stoich_change):
                if change != 0:
                    dCdt_sym[species_idx] += change * reaction_rate_sym
                
        return dCdt_sym, species_syms, rate_syms
    

def make_reduced_rhs_with_conservation(r_n):

    dCdt_sym, all_species_syms, rate_syms = r_n.get_symbolic_rhs()
    species_names = [str(s) for s in all_species_syms]
    species_map = {str(s): s for s in all_species_syms}

    # 1. Build conservation law equations and choose eliminated species
    const_syms = [sympy.symbols(f'const{i}') for i in range(len(r_n.L))]
    conservation_eqns = []
    eliminated_species = []
    for i, conservation_vector in enumerate(r_n.L):
        nonzero_indices = np.nonzero(conservation_vector)[0]
        species_indices = [(idx, str(all_species_syms[idx])) for idx in nonzero_indices]
        species_indices.sort(key=lambda x: x[1])
        if not species_indices:
            continue
        elim_idx, elim_species = next((idx, species) for idx, species in species_indices if species not in eliminated_species)
        eliminated_species.append(elim_species)
        eqn = np.dot(conservation_vector, all_species_syms) - const_syms[i]
        conservation_eqns.append(eqn)

    print(conservation_eqns)
    # 2. Solve all conservation laws simultaneously for all eliminated species
    elim_syms = [species_map[s] for s in eliminated_species]
    print(elim_syms)
    solutions_list = sympy.solve(conservation_eqns, elim_syms, dict=True)
    if not solutions_list:
        raise RuntimeError("Could not solve conservation laws for eliminated species.")
    solutions = solutions_list[0]  # dict: sympy.Symbol -> expression

    # 3. Substitute into ODEs
    remaining_species = [s for s in species_names if s not in eliminated_species]
    remaining_syms = [species_map[s] for s in remaining_species]
    substituted_rhs = [expr.subs(solutions, simultaneous=True) for expr in dCdt_sym]
    rhs_kept = [substituted_rhs[species_names.index(s)] for s in remaining_species]

    # 4. Lambdify
    func_args = remaining_syms + const_syms + list(rate_syms)
    rhs_func = sympy.lambdify(func_args, rhs_kept, modules=['numpy'])

    def ode_rhs(t, C, consts, rate_vals):
        args = list(C) + list(consts) + list(rate_vals)
        return np.array(rhs_func(*args)).flatten()

    # Convert solutions to string keys for consistency
    solutions = {str(k): v for k, v in solutions.items()}

    return ode_rhs, remaining_species, const_syms, solutions


def get_const_and_reduced_init_from_rn(r_n, remaining_species, C):

    # Get all species names and initial concentrations from r_n
    all_species_names = r_n.species_names

    # Compute the values of the conservation constants from the initial conditions
    const_vals = np.dot(r_n.L, C)

    # Get the indices of the remaining species
    idx_remain = [all_species_names.index(s) for s in remaining_species]
    reduced_init = C[idx_remain]

    return const_vals, reduced_init

def recover_eliminated_species(solutions, remaining_species, const_syms, const_vals, C_reduced):
    # Build the substitution dictionary for sympy
    subs = dict(zip(remaining_species, C_reduced))
    subs.update(zip(const_syms, const_vals))

    # Recursively substitute eliminated species
    recovered = {}
    for elim_name, expr in solutions.items():
        expr_sub = expr
        # Substitute other eliminated species if present
        for other_name, other_expr in solutions.items():
            if other_name != elim_name:
                expr_sub = expr_sub.subs(other_name, other_expr)
        # Now substitute numeric values
        val = expr_sub.evalf(subs=subs)
        try:
            recovered[elim_name] = float(val)
        except TypeError:
            recovered[elim_name] = val  # fallback: return symbolic if not numeric
    return recovered