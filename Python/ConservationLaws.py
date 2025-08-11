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
from scipy.optimize import nnls
from scipy.optimize import minimize

from dataclasses import dataclass
from typing import List, Dict, Tuple

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt

###################################################################
######################## Class Definitions ########################
################################################################### 

class ReactionNetwork:
    """
    Represents the structure and symbolic properties of a chemical reaction network.
    Handles network construction, symbolic ODE generation, and conservation law management.
    """
    def __init__(self, n_species: int, 
                 n_complexes: int, n_reactions: int, n_lcs: int,  
                 L: np.ndarray, 
                 seed: int, force_reverse: bool = True, subset_group_ind: int = None,
                 complexes_per_class: List[int] = None, reactions_per_class: List[int] = None):
        """
        Constructs a ReactionNetwork instance. This is equivalent to the
        old `build_reaction_network` function.
        Ensures full reproducibility by using local RNGs for both numpy and Python random.
        
        Args:
            n_species: Number of species in the network
            n_complexes: Total number of complexes (used only if complexes_per_class is None)
            n_reactions: Total number of reactions (used only if reactions_per_class is None)
            n_lcs: Number of linkage classes (used only if complexes_per_class is None)
            L: Conservation law matrix
            seed: Random seed for reproducibility
            force_reverse: Whether to force reversible reactions
            subset_group_ind: Index of conservation group to focus on (optional)
            complexes_per_class: List specifying number of complexes per linkage class (optional)
            reactions_per_class: List specifying number of reactions per linkage class (optional)
        """
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.py_rng = random.Random(self.seed)
        self.n_tries = 1000


        for n in range(self.n_tries):
            try:
                self.force_reverse = force_reverse
                self.L = L
                self.subset_group_ind = subset_group_ind
                self.species_names = list(string.ascii_uppercase[:n_species])

                # Use provided values or generate randomly
                if complexes_per_class is not None and reactions_per_class is not None:
                    # Validate provided values
                    if len(complexes_per_class) != len(reactions_per_class):
                        raise ValueError("complexes_per_class and reactions_per_class must have the same length")
                    
                    # Check that each class has at least 2 complexes
                    if any(n < 2 for n in complexes_per_class):
                        raise ValueError("Each linkage class must have at least 2 complexes")
                    
                    # Check that reactions per class is valid
                    for i, (n_complexes, n_reactions) in enumerate(zip(complexes_per_class, reactions_per_class)):
                        min_reactions = n_complexes - 1
                        if self.force_reverse:
                            max_reactions = n_complexes * (n_complexes - 1) // 2
                        else:
                            max_reactions = n_complexes * (n_complexes - 1)
                        
                        if n_reactions < min_reactions:
                            raise ValueError(f"Linkage class {i} needs at least {min_reactions} reactions for {n_complexes} complexes")
                        if n_reactions > max_reactions:
                            raise ValueError(f"Linkage class {i} can have at most {max_reactions} reactions for {n_complexes} complexes")
                    
                    self.complexes_per_class = complexes_per_class
                    self.reactions_per_class = reactions_per_class
                else:
                    self.complexes_per_class, self.reactions_per_class = self._random_partition_complexes_and_reactions(
                        n_complexes, n_reactions, n_lcs)
                
                self.complexes_dict = generate_string_vector_dict(n_species)
                
                self.pairing_groups = generate_pairing_groups(L, self.complexes_dict)
                
                self.conservation_groups = self.get_conservation_groups()
                self.within_pairing_groups, self.between_pairing_groups = self.get_within_between_pairing_groups()
                
                
                self.linkage_groups = self._generate_linkage_class_graphs()
                self.assignments = self._assign_complexes_to_linkage_groups()

                
                self.Y, self.all_complexes = self._build_Y_matrix()
                self.reactions = self._build_reaction_list(random_rates=True)
                self.A = self._build_A_matrix_incidence()
                self.Psi = self._build_Psi_function()
                self.n_species = n_species
                self.n_reactions = len(self.reactions)
                n_cons = len(self.conservation_groups)

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

    @classmethod
    def from_reaction_strings(cls, reaction_strings: List[str], L: np.ndarray,
                             seed: int = 42, force_reverse: bool = True, 
                             subset_group_ind: int = None, random_rates: bool = True):
        """
        Alternate constructor that creates a ReactionNetwork from a list of reaction strings.
        Validates that the provided conservation laws are consistent with the stoichiometric matrix.
        
        Args:
            reaction_strings: List of reaction strings like ['A+B->C', 'C->D+E']
            L: Conservation law matrix to validate against the stoichiometric matrix
            seed: Random seed for reproducibility (used for rate generation)
            force_reverse: If True, automatically adds reverse reactions for each input reaction
            subset_group_ind: Index of conservation group to focus on (optional)
            random_rates: Whether to assign random rates (True) or unit rates (False)
            
        Returns:
            ReactionNetwork instance
            
        Raises:
            ValueError: If the provided conservation laws are inconsistent with the stoichiometric matrix
        """
        # Parse reaction strings to extract complexes and reactions
        complexes, reactions = parse_reaction_strings(reaction_strings, force_reverse)
        
        # Determine number of species from complexes
        all_species = set()
        for complex_str in complexes:
            species_list = complex_str.split('+')
            all_species.update(species_list)
        n_species = len(all_species)
        
        # Create a minimal instance to set up basic attributes
        instance = cls.__new__(cls)
        instance.seed = seed
        instance.rng = np.random.default_rng(instance.seed)
        instance.py_rng = random.Random(instance.seed)
        instance.force_reverse = force_reverse
        instance.subset_group_ind = subset_group_ind
        instance.species_names = sorted(list(all_species))
        
        # Build complexes dictionary for the actual species present
        instance.complexes_dict = build_complexes_dict_from_complexes(complexes, instance.species_names)
        
        # Build Y matrix and all_complexes from the actual complexes
        instance.Y, instance.all_complexes = build_Y_matrix_from_complexes(complexes, instance.complexes_dict, instance.species_names)
        
        # Build reactions list from parsed reactions
        instance.reactions = build_reactions_from_parsed_reactions(reactions, instance.all_complexes, 
                                                                  instance.rng, random_rates, force_reverse)
        
        # Build stoichiometric matrix and validate conservation laws
        instance.A = instance._build_A_matrix_incidence()
        validate_conservation_laws_against_stoichiometry(instance, L)
        instance.L = L
        
        
        # Build pairing groups and conservation groups
        instance.pairing_groups = generate_pairing_groups(instance.L, instance.complexes_dict)
        # Call these methods after the instance is fully set up
        instance.conservation_groups = instance.get_conservation_groups()
        instance.n_cons = len(instance.conservation_groups)
        instance.within_pairing_groups, instance.between_pairing_groups = instance.get_within_between_pairing_groups()
        
        # Build remaining matrices and functions
        instance.Psi = instance._build_Psi_function()
        instance.n_species = n_species
        instance.n_reactions = len(instance.reactions)
        
        # Set linkage groups and assignments (simplified for deterministic case)
        instance.linkage_groups = build_linkage_groups_from_reactions(reactions, instance.all_complexes)
        instance.assignments = build_assignments_from_linkage_groups(instance.linkage_groups, instance.all_complexes)
        
        # Compute and store complexes_per_class and reactions_per_class
        instance.complexes_per_class, instance.reactions_per_class = compute_complexes_and_reactions_per_class(instance.linkage_groups)
        
        return instance

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
            if len(eligible) == 0: break
            reactions_per_class[self.rng.choice(eligible)] += 1


        return complexes_per_class, reactions_per_class

    def _generate_linkage_class_graphs(self):
        graphs = []
        for n_nodes, n_edges in zip(self.complexes_per_class, self.reactions_per_class):
            G = random_connected_graph(n_nodes, n_edges, self.rng, self.seed, directed=not self.force_reverse)
            graphs.append(G)
        return graphs

    def _assign_complexes_to_linkage_groups(self):
        pairing_groups_copy = self.pairing_groups[:]
        if self.subset_group_ind is not None:
            pairing_groups_copy = self.within_pairing_groups[:] + self.split_between_pairing_groups()[:]
        else:
            pairing_groups_copy = self.pairing_groups[:]

        used_complexes = set()
        assignments = []

        for G in self.linkage_groups:
            n_nodes = G.number_of_nodes()
            found = False
            self.py_rng.shuffle(pairing_groups_copy)
            for pairing_group in pairing_groups_copy:
                available_complexes = [c for c in pairing_group if c not in used_complexes]
                if len(available_complexes) >= n_nodes:
                    chosen_complexes = self.py_rng.sample(available_complexes, n_nodes)
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
    
    def get_conservation_groups(self):
        """Get groups of species that are conserved together"""
        conservation_groups = []
        for cg in range(len(self.L)):
            conservation_groups.append([self.species_names[i] for i, val in enumerate(self.L[cg]) if val != 0])
        return conservation_groups

    def get_within_between_pairing_groups(self):
        """Split pairing groups into within and between conservation groups"""
        within_pairing_groups = []
        between_pairing_groups = []
        
        for pg in self.pairing_groups:
            complexes_in_pg = list(pg.keys())
            mixed_bool = False
            for c in complexes_in_pg:
                if(len(c)) == 3:
                    if not any(c[0] in cg and c[2] in cg for cg in self.conservation_groups):
                        mixed_bool = True
            if not mixed_bool:
                within_pairing_groups.append(pg)
            else:  
                between_pairing_groups.append(pg)
                
        return within_pairing_groups, between_pairing_groups

    def split_between_pairing_groups(self, subset_group_ind = None):
        """Split between pairing groups based on a subset of species"""
        split_groups = []
        if subset_group_ind is None:
            subset_group_ind = self.subset_group_ind
        subset_group = self.conservation_groups[subset_group_ind]
        for pg in self.between_pairing_groups:
            for s in subset_group:
                split_groups.append(dict())
                for c in pg.keys():
                    if s in c:
                        split_groups[-1][c] = pg[c]
        return split_groups
    
    def count_conservation_group_changes(self):
        # Get conservation groups
        conservation_groups = self.get_conservation_groups()

        M_0t1 = 0
        M_1t0 = 0 
        M_0b1 = 0 # symmetric

        # Loop through reactions and check conservation group membership
        for i, (c1_idx, c2_idx, _) in enumerate(self.reactions):
            c1 = self.all_complexes[c1_idx]
            c2 = self.all_complexes[c2_idx]
            
            for pairing_group in self.within_pairing_groups:
                if c1 in pairing_group and c2 in pairing_group.keys():
                    break

            for pairing_group in self.between_pairing_groups:
                if c1 in pairing_group and c2 in pairing_group.keys():
                    
                    # Check stoichiometry changes for each conservation group
                    cg_change_bool_vec = []
                    for group_idx, group in enumerate(conservation_groups):
                        # Count species from this group on each side
                        # Get species in this group from each side
                        c1_species = set(species for species in (c1.split('+') if len(c1)==3 else [c1]) if species in group)
                        c2_species = set(species for species in (c2.split('+') if len(c2)==3 else [c2]) if species in group)
                        # If counts differ, this group has a net change
                        cg_change_bool_vec.append(c1_species != c2_species)
                    
                    if cg_change_bool_vec == [True, False]:
                        M_1t0 += 1
                    elif cg_change_bool_vec == [False, True]:
                        M_0t1 += 1
                    elif cg_change_bool_vec == [True, True]:
                        M_0b1 += 1
        
                    break

        return M_0t1, M_1t0, M_0b1
        
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

    def get_stoichiometric_matrix(self) -> np.ndarray:
        """
        Returns the stoichiometric matrix S with rows = species and columns = reactions.

        For each reaction e corresponding to (i -> j), the e-th column of S is
        given by Y[:, j] - Y[:, i]. The resulting matrix has shape
        (n_species, n_reactions).
        """
        # Return cached matrix if available and consistent
        if hasattr(self, "S") and self.S is not None and self.S.shape == (self.Y.shape[0], len(self.reactions)):
            return self.S

        n_species = self.Y.shape[0]
        n_reactions = len(self.reactions)
        S = np.zeros((n_species, n_reactions), dtype=int)
        for e, (i, j, _) in enumerate(self.reactions):
            S[:, e] = self.Y[:, j] - self.Y[:, i]

        self.S = S
        return S

    def get_reaction_strings(self, include_rates: bool = True) -> List[str]:
        """
        Returns a list of human-readable strings describing each reaction
        in terms of its source and target complexes.

        Example item: "r3: A+B -> C ; k_3=1.25"

        Args:
            include_rates: whether to append numeric rate values
        """
        reaction_strings = []
        for r_idx, (i, j, k_val) in enumerate(self.reactions):
            src = self.all_complexes[i]
            dst = self.all_complexes[j]
            if include_rates:
                reaction_strings.append(f"r{r_idx}: {src} -> {dst} ; k_{r_idx}={k_val:.6g}")
            else:
                reaction_strings.append(f"r{r_idx}: {src} -> {dst}")
        return reaction_strings

    def print_reactions(self, include_rates: bool = True) -> None:
        """
        Prints each reaction as a symbolic mapping of complexes.
        """
        for s in self.get_reaction_strings(include_rates=include_rates):
            print(s)

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

    def compute_cycles(self):
        """
        Compute cycles in the reaction network by finding the nullspace of the stoichiometric matrix.
        
        Args:
            r_n: ReactionNetwork instance
            force_reverse: Boolean indicating if reactions are forced to be reversible
            
        Returns:
            Z: Matrix of cycles
            cycles: List of cycles where each cycle is a list of reaction strings
        """
        S = self.get_stoichiometric_matrix()
 
        if self.force_reverse:
            S_red = S[:,0:-1:2] 
        else:
            S_red = S

        ns = sympy.Matrix(S_red).nullspace()  # list of column vectors
        Z = sympy.Matrix.hstack(*ns) if ns else sympy.Matrix.zeros(S_red.shape[1], 0)
        
        
        cycles = []
        reaction_strings = self.get_reaction_strings(include_rates=False)
        
        for col in range(Z.shape[1]):
            cycle = Z[:,col]
            cycle_reactions = []
            for ind in range(len(cycle)):
                if cycle[ind] == 1:
                    cycle_reactions.append(reaction_strings[2*ind])
                if cycle[ind] == -1:
                    cycle_reactions.append(reaction_strings[2*ind+1])
            cycles.append(cycle_reactions)
            
        return Z, cycles

    
    

class ReactionNetworkSimulator:
    """
    Handles numeric simulation and integration of a ReactionNetwork, including ODE reduction via conservation laws.
    """
    def __init__(self, reaction_network: ReactionNetwork):
        """
        Args:
            reaction_network: An instance of ReactionNetwork
        """
        self.r_n = reaction_network
        self.rhs = self._make_ode_rhs()
        self.reduced_ode_rhs = None
        self.remaining_species = None
        self.const_syms = None
        self.elimination_solutions = None

    def _make_ode_rhs(self):
        def rhs(C):
            v_monomials = self.r_n.Psi(C)
            dCdt = np.zeros(self.r_n.Y.shape[0])
            for idx, (i, j, k) in enumerate(self.r_n.reactions):
                rate = k * v_monomials[idx]
                stoichiometric_change = self.r_n.Y[:, j] - self.r_n.Y[:, i]
                dCdt += rate * stoichiometric_change
            return dCdt
        return rhs

    def integrate(
        self,
        ode_rhs,
        initial_conditions: np.ndarray,
        t_span: Tuple[float, float] = (0, 5),
        num_points: int = 20000,
        **kwargs
        ) -> Tuple[object, np.ndarray]:
        """
        Integrate an ODE system using the provided right-hand side function.

        Args:
            ode_rhs: Callable (t, y) -> dy/dt (the ODE right-hand side)
            initial_conditions: Initial concentrations (array)
            t_span: Tuple (start, end) for integration time
            t_eval: Array of time points to evaluate at (optional)
            kwargs: Additional arguments to pass to solve_ivp

        Returns:
            sol: The solution object from solve_ivp
            final_concentrations: Concentrations at the final time point
        """
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        sol = solve_ivp(lambda t, C: ode_rhs(C), t_span, initial_conditions, t_eval=t_eval, **kwargs)
        return sol, sol.y[:,-1]

    def solve_conservation_laws(self):
        """
        Symbolically solves the conservation laws for the eliminated variables.
        """
        _, all_species_syms, _ = self.get_symbolic_rhs()  # Use sympy symbols, not string names
        species_names = [str(s) for s in all_species_syms]
        species_map = {str(s): s for s in all_species_syms}
        const_syms = [sympy.symbols(f'const{i}') for i in range(len(self.r_n.L))]
        conservation_eqns = []
        eliminated_species = []
        for i, conservation_vector in enumerate(self.r_n.L):
            nonzero_indices = np.nonzero(conservation_vector)[0]
            species_indices = [(idx, str(all_species_syms[idx])) for idx in nonzero_indices]
            species_indices.sort(key=lambda x: x[1])
            if not species_indices:
                continue
            elim_idx, elim_species = next((idx, species) for idx, species in species_indices if species not in eliminated_species)
            eliminated_species.append(elim_species)
            # Use sympy sum for symbolic computation
            eqn = sum(conservation_vector[j] * all_species_syms[j] for j in range(len(all_species_syms))) - const_syms[i]
            conservation_eqns.append(eqn)
        elim_syms = [species_map[s] for s in eliminated_species]
        solutions_list = sympy.solve(conservation_eqns, elim_syms, dict=True)
        if not solutions_list:
            raise RuntimeError("Could not solve conservation laws for eliminated species.")
        solutions = solutions_list[0]  # dict: sympy.Symbol -> expression
        elimination_solutions = {str(k): v for k, v in solutions.items()}
        remaining_species = [s for s in species_names if s not in eliminated_species]
        self.eliminated_species = eliminated_species
        self.remaining_species = remaining_species
        self.const_syms = const_syms
        self.elimination_solutions = elimination_solutions

    def make_reduced_rhs_with_conservation(self, const_vals=None, rate_vals=None):
        """
        Constructs a reduced ODE rhs function with conservation laws substituted in.
        Args:
            const_vals: values for the conservation constants (list or array) - optional for symbolic setup
            rate_vals: values for the rate constants (optional, defaults to r_n.rate_constants or [r[2] for r in r_n.reactions])
        Returns:
            ode_rhs(C): function of concentrations for the remaining species
        """
        
        # Only do symbolic substitution once if not already done
        if not hasattr(self, '_substituted_rhs_func'):
            dCdt_sym, all_species_syms, rate_syms = self.get_symbolic_rhs()
            species_names = [str(s) for s in all_species_syms]
            species_map = {str(s): s for s in all_species_syms}
            remaining_syms = [species_map[s] for s in self.remaining_species]
            
            # Substitute solutions into the ODEs
            solutions = {species_map[k]: v for k, v in self.elimination_solutions.items()}
            substituted_rhs = [expr.subs(solutions, simultaneous=True) for expr in dCdt_sym]
            rhs_kept = [substituted_rhs[species_names.index(s)] for s in self.remaining_species]
            
            # Create symbolic function with all parameters
            func_args = remaining_syms + self.const_syms + list(rate_syms)
            self._substituted_rhs_func = sympy.lambdify(func_args, rhs_kept, modules=['numpy'])
            self._rate_syms = rate_syms
        
        # Get rate values (default if not provided)
        if rate_vals is None:
            rate_vals = [r[2] for r in self.r_n.reactions]
        
        # Create the numerical function with current const_vals
        def reduced_ode_rhs(C, const_vals=const_vals):
            args = list(C) + list(const_vals) + list(rate_vals)
            return np.array(self._substituted_rhs_func(*args)).flatten()
        
        # Store for later use
        self.reduced_ode_rhs = reduced_ode_rhs
        return reduced_ode_rhs
    
    def make_reduced_rhs_with_conservation_flexible(self, rate_vals=None):
        """
        Creates a flexible reduced ODE rhs function that accepts const_vals as a parameter.
        This allows changing const_vals without recreating the function.
        
        Args:
            rate_vals: values for the rate constants (optional)
        Returns:
            ode_rhs(C, const_vals): function that takes both concentrations and const_vals
        """
        # Ensure symbolic substitution is done
        if not hasattr(self, '_substituted_rhs_func'):
            self.make_reduced_rhs_with_conservation()
        
        # Get rate values (default if not provided)
        if rate_vals is None:
            rate_vals = [r[2] for r in self.r_n.reactions]
        
        # Create flexible function that accepts const_vals as parameter
        def flexible_reduced_ode_rhs(C, const_vals):
            args = list(C) + list(const_vals) + list(rate_vals)
            return np.array(self._substituted_rhs_func(*args)).flatten()
        
        return flexible_reduced_ode_rhs

    def get_const_and_reduced_init(self, C):

        # Compute the values of the conservation constants from the initial conditions
        const_vals = np.dot(self.r_n.L, C)

        # Get the indices of the remaining species
        idx_remain = [self.r_n.species_names.index(s) for s in self.remaining_species]
        reduced_init = C[idx_remain]

        return const_vals, reduced_init
    
    def recover_eliminated_species(self, const_vals, C_reduced):
        # Build the substitution dictionary for sympy
        subs = dict(zip(self.remaining_species, C_reduced))
        subs.update(zip(self.const_syms, const_vals))

        # Recursively substitute eliminated species
        recovered = {}
        for elim_name, expr in self.elimination_solutions.items():
            expr_sub = expr
            # Substitute other eliminated species if present
            for other_name, other_expr in self.elimination_solutions.items():
                if other_name != elim_name:
                    expr_sub = expr_sub.subs(other_name, other_expr)
            # Now substitute numeric values
            val = expr_sub.evalf(subs=subs)
            try:
                recovered[elim_name] = float(val)
            except TypeError:
                recovered[elim_name] = val  # fallback: return symbolic if not numeric

        # Build complete concentration vector with all species
        C_complete = np.zeros(len(self.r_n.species_names))
        
        # Fill in the remaining species values
        for species, value in zip(self.remaining_species, C_reduced):
            idx = self.r_n.species_names.index(species)
            C_complete[idx] = value
            
        # Fill in the recovered eliminated species values
        for species, value in recovered.items():
            idx = self.r_n.species_names.index(str(species))
            C_complete[idx] = value

        return C_complete

    def get_symbolic_rhs(self) -> Tuple[List[sympy.Expr], List[sympy.Symbol], List[sympy.Symbol]]:
        """
        Returns the symbolic right-hand side of the ODEs for the reaction network.
        Returns:
            dCdt_sym: List of sympy expressions for dC/dt for each species
            species_syms: List of sympy symbols for each species
            rate_syms: List of sympy symbols for each rate constant
        """

        # Create symbolic variables for each species
        species_syms = sympy.symbols(self.r_n.species_names)
        
        # Create descriptive names for the symbolic rate constants
        rate_names = []
        for r_idx, (i, j, _) in enumerate(self.r_n.reactions):
            # Creates names like: k_0_C+C_B+C
            # reactant_str = self.r_n.all_complexes[i]
            # product_str = self.r_n.all_complexes[j]
            # rate_names.append(f'k_{r_idx};{reactant_str}->{product_str}')
            rate_names.append(f'k_{r_idx}')

        rate_syms = sympy.symbols(rate_names)

        # Initialize a list to hold the symbolic ODE for each species
        dCdt_sym = [sympy.Integer(0)] * len(species_syms)

        # Iterate through each reaction to build the symbolic ODEs
        for idx, (i, j, k_val) in enumerate(self.r_n.reactions):
            rate_k = rate_syms[idx]
            
            # Build the monomial for the reactant complex
            reactant_complex_stoich = self.r_n.Y[:, i]
            monomial = sympy.Integer(1)
            for species_idx, exponent in enumerate(reactant_complex_stoich):
                if exponent > 0:
                    monomial *= species_syms[species_idx]**exponent
            
            # Symbolic representation of the reaction rate
            reaction_rate_sym = rate_k * monomial
            
            # Stoichiometric change vector for the reaction
            stoich_change = self.r_n.Y[:, j] - self.r_n.Y[:, i]
            
            # Add this reaction's contribution to each species' ODE
            for species_idx, change in enumerate(stoich_change):
                if change != 0:
                    dCdt_sym[species_idx] += change * reaction_rate_sym
                
        return dCdt_sym, species_syms, rate_syms
    
    def get_symbolic_reduced_rhs(self):
        """
        Returns the symbolic right-hand side of the reduced ODEs (after eliminating variables using conservation laws).
        Returns:
            reduced_rhs: List of sympy expressions for dC/dt for each remaining species
            remaining_syms: List of sympy symbols for the remaining species
            const_syms: List of sympy symbols for the conservation constants
            rate_syms: List of sympy symbols for each rate constant
        """
        # Ensure conservation laws have been solved
        if self.elimination_solutions is None or self.remaining_species is None:
            self.solve_conservation_laws()

        # Get the full symbolic ODEs
        dCdt_sym, all_species_syms, rate_syms = self.get_symbolic_rhs()
        species_names = [str(s) for s in all_species_syms]
        species_map = {str(s): s for s in all_species_syms}
        remaining_syms = [species_map[s] for s in self.remaining_species]

        # Substitute eliminated species in the ODEs
        solutions = {species_map[k]: v for k, v in self.elimination_solutions.items()}
        substituted_rhs = [expr.subs(solutions, simultaneous=True) for expr in dCdt_sym]

        # Only keep the ODEs for the remaining species
        reduced_rhs = [substituted_rhs[species_names.index(s)] for s in self.remaining_species]

        return reduced_rhs, remaining_syms, self.const_syms, rate_syms
    
    def get_derivatives(self):
        """
        Returns symbolic and lambdified derivatives of the reduced ODE system with respect to remaining species, conservation constants, and rate constants.
        
        Tensor and function naming/indexing convention:
            dX_dY_dZ[i, j, k] = ∂²X_i / ∂Y_j ∂Z_k
        where X is the numerator (e.g., R), Y and Z are the denominator variables (e.g., C, l, k), and indices follow this order.
        All lambdified functions return arrays with the same indexing convention.
        
        Returns:
            dR_dC, dR_dC_func: Jacobian w.r.t. remaining species
            dR_dl, dR_dl_func: Jacobian w.r.t. conservation constants
            dR_dC_dk, dR_dC_dk_func: 2nd derivative w.r.t. species and rate constants
            dR_dl_dk, dR_dl_dk_func: 2nd derivative w.r.t. conservation constants and rate constants
            dR_dC_dl, dR_dC_dl_func: 2nd derivative w.r.t. species and conservation constants
            dR_dl_dl, dR_dl_dl_func: 2nd derivative w.r.t. conservation constants (twice)
            remaining_syms: list of sympy symbols for the remaining species
        """
        reduced_rhs, remaining_syms, const_syms, rate_syms = self.get_symbolic_reduced_rhs()
        arg_syms = list(remaining_syms) + list(const_syms) + list(rate_syms)
        dR_dC = sympy.Matrix(reduced_rhs).jacobian(remaining_syms)
        dR_dC_func = sympy.lambdify(
            arg_syms,
            dR_dC,
            modules='numpy'
        )

        dR_dl = sympy.Matrix(reduced_rhs).jacobian(const_syms)
        dR_dl_func = sympy.lambdify(
            arg_syms,
            dR_dl,
            modules='numpy'
        )

        dR_dk = sympy.Matrix(reduced_rhs).jacobian(rate_syms)
        dR_dk_func = sympy.lambdify(
            arg_syms,
            dR_dk,
            modules='numpy'
        )

        # dR_dC_dk: (m, n, p)
        m, n = dR_dC.shape
        p = len(rate_syms)
        dR_dC_dk = sympy.MutableDenseNDimArray(
            [[[dR_dC[i, j].diff(rate_syms[k]) for k in range(p)] for j in range(n)] for i in range(m)]
        )
        # Lambdify each entry individually, then wrap in a function that evaluates the whole tensor
        def evaluate_tensor_func(func_array, *args):
            out = np.empty(func_array.shape, dtype=float)
            it = np.nditer(func_array, flags=['multi_index', 'refs_ok'])
            for x in it:
                idx = it.multi_index
                out[idx] = func_array[idx](*args)
            return out

        dR_dC_dk_func_array = np.empty((m, n, p), dtype=object)
        
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    dR_dC_dk_func_array[i, j, k] = sympy.lambdify(arg_syms, dR_dC_dk[i, j, k], modules='numpy')
        def dR_dC_dk_func(*args):
            return evaluate_tensor_func(dR_dC_dk_func_array, *args)

        m_dl, n_dl = dR_dl.shape
        dR_dl_dk = sympy.MutableDenseNDimArray(
            [[[dR_dl[i, j].diff(rate_syms[k]) for k in range(p)] for j in range(n_dl)] for i in range(m_dl)]
        )
        dR_dl_dk_func_array = np.empty((m_dl, n_dl, p), dtype=object)
        for i in range(m_dl):
            for j in range(n_dl):
                for k in range(p):
                    dR_dl_dk_func_array[i, j, k] = sympy.lambdify(arg_syms, dR_dl_dk[i, j, k], modules='numpy')
        def dR_dl_dk_func(*args):
            return evaluate_tensor_func(dR_dl_dk_func_array, *args)
        
        # n_dl = len(const_syms)
        # dR_dC_dl = sympy.MutableDenseNDimArray(
        #     [[[dR_dC[i, j].diff(const_syms[k]) for k in range(n_dl)] for j in range(n)] for i in range(m)]
        # )
        # dR_dC_dl_func_array = np.empty((m, n, n_dl), dtype=object)
        # for i in range(m):
        #     for j in range(n):
        #         for k in range(n_dl):
        #             dR_dC_dl_func_array[i, j, k] = sympy.lambdify(arg_syms, dR_dC_dl[i, j, k], modules='numpy')
        # def dR_dC_dl_func(*args):
        #     return evaluate_tensor_func(dR_dC_dl_func_array, *args)
        
        # # dR_dl_dl: (m_dl, n_dl, n_dl)
        # dR_dl_dl = sympy.MutableDenseNDimArray(
        #     [[[dR_dl[i, j].diff(const_syms[k]) for k in range(n_dl)] for j in range(n_dl)] for i in range(m_dl)]
        # )
        # dR_dl_dl_func_array = np.empty((m_dl, n_dl, n_dl), dtype=object)
        # for i in range(m_dl):
        #     for j in range(n_dl):
        #         for k in range(n_dl):
        #             dR_dl_dl_func_array[i, j, k] = sympy.lambdify(arg_syms, dR_dl_dl[i, j, k], modules='numpy')
        # def dR_dl_dl_func(*args):
        #     return evaluate_tensor_func(dR_dl_dl_func_array, *args)


        return (
            dR_dC, dR_dC_func,
            dR_dl, dR_dl_func,
            dR_dk, dR_dk_func,
            dR_dC_dk, dR_dC_dk_func,
            dR_dl_dk, dR_dl_dk_func,
            #dR_dC_dl, dR_dC_dl_func,
            #dR_dl_dl, dR_dl_dl_func,
            remaining_syms
        )

    def dC_dl_func(self, C, const_vals, rates, dR_dC_func, dR_dl_func):
        args = list(C) + list(const_vals) + list(rates)
        dR_dC_eval = dR_dC_func(*args) 
        dR_dl_eval = dR_dl_func(*args)
        dC_dl = -np.tensordot(np.linalg.inv(dR_dC_eval), dR_dl_eval, axes = ([1], [0]))
        return dC_dl
    
    def dC_dk_func(self, C, const_vals, rates, dR_dC_func, dR_dk_func):
        args = list(C) + list(const_vals) + list(rates)
        dR_dC_eval = dR_dC_func(*args) 
        dR_dk_eval = dR_dk_func(*args)
        dC_dk = -np.tensordot(np.linalg.inv(dR_dC_eval), dR_dk_eval, axes = ([1], [0]))
        return dC_dk

    def dC_dl_laplacian_func(self, C, const_vals, rates, dR_dC_func, dR_dl_func, dR_dl_dl_func, dR_dC_dl_func):
        args = list(C) + list(const_vals) + list(rates)
        dR_dC_eval = dR_dC_func(*args) 
        dR_dl_eval = dR_dl_func(*args)
        dR_dl_dl_eval = dR_dl_dl_func(*args)
        dR_dC_dl_eval = dR_dC_dl_func(*args)

        M = np.linalg.inv(dR_dC_eval)
        Q = - np.tensordot(np.tensordot(M, dR_dC_dl_eval, axes = ([1], [0])), M, axes = ([1], [0])).swapaxes(-1,-2) # dM_{ij} / dl_k

        term_1 = - np.einsum('ijj->i', np.tensordot(Q, dR_dl_eval, axes = ([1], [0])))
        dR_dl_dl_contr = np.einsum('ijj->i', dR_dl_dl_eval)
        term_2 = - np.tensordot(M, dR_dl_dl_contr, axes = ([1], [0]))

        return term_1 + term_2

    def dC_func(self, t, C, rates, l_unit_vec, l_base, dR_dC_func, dR_dl_func):
        const_vals = l_unit_vec * t + l_base
        dC_dl = self.dC_dl_func(C, const_vals, rates, dR_dC_func, dR_dl_func)
        dC = np.tensordot(dC_dl, l_unit_vec, axes = ([1], [0]))
        return dC
    
    def dC_dl_dk_func(self, t, C, rates, l_unit_vec, l_base, dR_dC_func, dR_dl_func, dR_dC_dk_func, dR_dl_dk_func):
        const_vals = l_unit_vec * t + l_base
        args = list(C) + list(const_vals) + list(rates)

        dR_dC_eval = dR_dC_func(*args)
        M = np.linalg.inv(dR_dC_eval)
        dR_dl_eval = dR_dl_func(*args)
        dR_dC_dk_eval = dR_dC_dk_func(*args)
        dR_dl_dk_eval = dR_dl_dk_func(*args)

        term_1 = np.tensordot(
            np.tensordot(
                np.tensordot(
                    M, dR_dC_dk_eval, axes = ([1], [0])), 
                M, axes = ([1], [0])),
            dR_dl_eval, axes = ([2], [0])).swapaxes(-2,-1)

        term_2 = -np.tensordot(M, dR_dl_dk_eval, axes = ([1], [0]))

        dC_dl_dk = term_1 + term_2
        return dC_dl_dk
    
    def dC_and_dC_dk_combined_func(self, t, C_combined, rates, l_unit_vec, l_base, dR_dC_func, dR_dl_func, dR_dC_dk_func, dR_dl_dk_func):
        dC = self.dC_func(t, C_combined[:len(self.remaining_species)], rates, l_unit_vec, l_base, dR_dC_func, dR_dl_func)
        dC_dk = np.tensordot(self.dC_dl_dk_func(t, C_combined[:len(self.remaining_species)], rates, l_unit_vec, l_base, dR_dC_func, dR_dl_func, dR_dC_dk_func, dR_dl_dk_func), l_unit_vec, axes = ([1], [0]))
        dC_combined = np.concatenate([dC, dC_dk.flatten()], axis = 0)
        return dC_combined
    
    def convert_dC_combined(self, dC_combined):
        dC = dC_combined[:len(self.remaining_species)]
        dC_dk = dC_combined[len(self.remaining_species):].reshape(len(self.remaining_species), -1)
        return dC, dC_dk
    
    def compute_dC_dk_full(self, dC_dk):
        """
        Computes the full dC_dk matrix including both eliminated and remaining species.
        
        Args:
            sim: Simulation object containing elimination solutions and remaining species
            dC_dk: Matrix of derivatives for remaining species
        
        Returns:
            numpy array: Full dC_dk matrix for all species
        """
        dsp_dk_list = []
        for sp, expr in self.elimination_solutions.items():
            dsp_dk = np.zeros_like(dC_dk[0])
            for (i, rem_sp) in enumerate(self.remaining_species):
                coef = float(expr.coeff(sympy.Symbol(rem_sp)))
                dsp_dk = dsp_dk + dC_dk[i] * coef
            dsp_dk_list.append(dsp_dk)

        # Create new dC_dk list with interleaved values
        new_dC_dk = []
        eliminated_species = list(self.elimination_solutions.keys())
        all_species = eliminated_species + self.remaining_species

        for sp in self.r_n.species_names:
            if sp in eliminated_species:
                # Get index in eliminated species list to find corresponding dsp_dk
                idx = eliminated_species.index(sp)
                new_dC_dk.append(dsp_dk_list[idx])
            else:
                # Get index in remaining species list to find corresponding dC_dk
                idx = self.remaining_species.index(sp)
                new_dC_dk.append(dC_dk[idx])

        return np.array(new_dC_dk)

    # --- JAX-based autodiff ODE integration and sensitivity ---
    def jax_rhs(self, C, rates):
        """
        JAX-compatible ODE right-hand side for the full system.
        C: jnp.array, concentrations
        rates: jnp.array, rate constants
        Returns: dC/dt as jnp.array
        """

        # Re-implement the ODE system using JAX operations only
        # This is a direct translation of self.r_n.Psi and self.r_n.reactions logic
        Y = jnp.array(self.r_n.Y)
        reactions = self.r_n.reactions
        n_species = Y.shape[0]
        v_monomials = []
        for idx, (i, j, _) in enumerate(reactions):
            monomial = jnp.prod(C ** Y[:, i])
            v_monomials.append(monomial)
        v_monomials = jnp.array(v_monomials)
        dCdt = jnp.zeros(n_species)
        for idx, (i, j, _) in enumerate(reactions):
            rate = rates[idx] * v_monomials[idx]
            stoich_change = Y[:, j] - Y[:, i]
            dCdt = dCdt + rate * stoich_change
        return dCdt

    def integrate_final_conc_jax(self, C0, rates, t0=0.0, t1=1000.0, dt0=1e-2, max_steps=100000):
        """
        Integrate the ODE using diffrax and return the final concentrations.
        C0: initial concentrations (jnp.array)
        rates: rate constants (jnp.array)
        Returns: final concentrations (jnp.array)
        """
 
        def ode_func(t, C, args):
            rates = args
            return self.jax_rhs(C, rates)
        term = ODETerm(ode_func)
        sol = diffeqsolve(
            term,
            solver=Dopri5(),
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=C0,
            args=rates,
            saveat=SaveAt(t1=True),
            max_steps=max_steps  # or higher if needed
        )
        return sol.ys[-1]

    def sensitivity_wrt_rates_jax(self, C0, rates, t0=0.0, t1=1000.0, dt0=1e-2, max_steps=100000, mode='fwd'):
        """
        Compute the Jacobian of the final concentrations with respect to the rates using JAX autodiff.
        C0: initial concentrations (jnp.array)
        rates: rate constants (jnp.array)
        Returns: Jacobian (n_species, n_rates)
        """
        import jax
        if mode == 'fwd':
            jac_fn = jax.jacfwd(lambda r: self.integrate_final_conc_jax(C0, r, t0, t1, dt0, max_steps))
        else:
            jac_fn = jax.jacrev(lambda r: self.integrate_final_conc_jax(C0, r, t0, t1, dt0, max_steps))
        return jac_fn(rates)

class InputData:
    """
    Manages labeled input data for training and testing.
    """

    def __init__(self, n_classes, data_list, split_fac=0.75):
        """
        Initializes training and testing datasets.

        Parameters:
        - n_classes: Number of output classes
        - data_list: List of data samples per class
        - split_fac: Fraction of data used for training
        """
        self.n_classes = n_classes
        self.labels = self._create_labels()
        self.data_list = data_list 
        self.split_fac = split_fac
        self.training_data, self.testing_data = self._split_shuffle_data(data_list, split_fac)

    def _create_labels(self):
        """Creates one-hot encoded labels for classification."""
        return [np.eye(self.n_classes)[n] for n in range(self.n_classes)]

    def _split_shuffle_data(self, data_list, split_fac):
        """Splits data into training and testing sets."""
        tr_data, te_data = [], []
        for nc in range(self.n_classes):
            sub_data = data_list[nc]
            random.shuffle(sub_data)
            n_train = round(split_fac * len(sub_data))
            tr_data.append(iter(sub_data[:n_train]))
            te_data.append(iter(sub_data[n_train:]))
        return tr_data, te_data

    def refill_iterators(self):
        """Refills the training and testing iterators from data_list."""
        self.training_data, self.testing_data = self._split_shuffle_data(self.data_list, self.split_fac)

    def get_next_training_sample(self, class_number):
        """Returns the next training sample for the given class, refilling iterators as needed."""
        try:
            return next(self.training_data[class_number])
        except StopIteration:
            self.refill_iterators()
            return next(self.training_data[class_number])
        

##################################################################
######################## Helper Functions ########################
################################################################## 

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
        
    species_energies = np.random.rand(r_n.n_species)*E_range
    species_energies_dict = dict(zip(r_n.species_names, species_energies))
    reaction_barriers = np.random.rand(r_n.n_reactions)*B_range
    reaction_affinities = np.random.rand(r_n.n_reactions)*F_range

    reac_rates = []
    assert r_n.force_reverse
    for reac_ind in range(int(len(r_n.reactions)/2)):
        i, j, _ = r_n.reactions[3]
        src_species = r_n.all_complexes[i].split("+")
        stoich_src = len(src_species)
        dst_species = r_n.all_complexes[j].split("+")
        stoich_dst = len(dst_species)

        src_energy = np.sum([species_energies_dict[s] for s in src_species])
        dst_energy = np.sum([species_energies_dict[s] for s in dst_species])

        fwd_rate = np.exp(beta*(reaction_barriers[reac_ind] - src_energy + reaction_affinities[reac_ind]/2)) * (C0**(1-stoich_src))
        rev_rate = np.exp(beta*(reaction_barriers[reac_ind] - dst_energy - reaction_affinities[reac_ind]/2)) * (C0**(1-stoich_dst))
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
    from scipy.optimize import nnls
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

def parse_reaction_strings(reaction_strings: List[str], force_reverse: bool = False) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Parse a list of reaction strings to extract complexes and reactions.
    
    Args:
        reaction_strings: List of strings like ['A+B->C', 'C->D+E']
        force_reverse: If True, automatically add reverse reactions
        
    Returns:
        complexes: List of unique complex strings
        reactions: List of (reactant, product) tuples
    """
    complexes = set()
    reactions = []
    
    for reaction_str in reaction_strings:
        # Clean up whitespace and split on arrow
        reaction_str = reaction_str.strip()
        if '->' in reaction_str:
            parts = reaction_str.split('->')
        elif '→' in reaction_str:
            parts = reaction_str.split('→')
        else:
            raise ValueError(f"Invalid reaction string: {reaction_str}. Must contain '->' or '→'")
        
        if len(parts) != 2:
            raise ValueError(f"Invalid reaction string: {reaction_str}. Must have exactly one arrow.")
        
        reactant = parts[0].strip()
        product = parts[1].strip()
        
        # Add complexes to set
        complexes.add(reactant)
        complexes.add(product)
        
        # Add forward reaction
        reactions.append((reactant, product))
        
        # Add reverse reaction if force_reverse is True
        if force_reverse:
            reactions.append((product, reactant))
    
    return list(complexes), reactions

def build_complexes_dict_from_complexes(complexes: List[str], species_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Build complexes dictionary from a list of complex strings.
    
    Args:
        complexes: List of complex strings like ['A', 'B', 'A+B']
        species_names: List of species names in order
        
    Returns:
        Dictionary mapping complex strings to stoichiometric vectors
    """
    species_indices = {species: i for i, species in enumerate(species_names)}
    complexes_dict = {}
    
    for complex_str in complexes:
        vec = np.zeros(len(species_names), dtype=int)
        species_list = complex_str.split('+')
        
        for species in species_list:
            species = species.strip()
            if species in species_indices:
                vec[species_indices[species]] += 1
            else:
                raise ValueError(f"Species {species} in complex {complex_str} not found in species_names")
        
        complexes_dict[complex_str] = vec
    
    return complexes_dict

def build_Y_matrix_from_complexes(complexes: List[str], complexes_dict: Dict[str, np.ndarray], 
                                 species_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Build Y matrix and all_complexes list from complexes.
    
    Args:
        complexes: List of complex strings
        complexes_dict: Dictionary mapping complex strings to stoichiometric vectors
        species_names: List of species names
        
    Returns:
        Y: Stoichiometric matrix
        all_complexes: List of complex strings in order
    """
    all_complexes = sorted(complexes)  # Sort for deterministic ordering
    Y = np.zeros((len(species_names), len(all_complexes)), dtype=int)
    
    for j, complex_name in enumerate(all_complexes):
        Y[:, j] = complexes_dict[complex_name]
    
    return Y, all_complexes

def build_reactions_from_parsed_reactions(reactions: List[Tuple[str, str]], all_complexes: List[str], 
                                         rng: np.random.Generator, random_rates: bool = True, 
                                         force_reverse: bool = False) -> List[Tuple[int, int, float]]:
    """
    Build reactions list from parsed reactions.
    
    Args:
        reactions: List of (reactant, product) tuples
        all_complexes: List of complex strings in order
        rng: Random number generator
        random_rates: Whether to assign random rates
        force_reverse: Whether reverse reactions were added during parsing
        
    Returns:
        List of (reactant_idx, product_idx, rate) tuples
    """
    complex_to_idx = {complex_name: i for i, complex_name in enumerate(all_complexes)}
    reaction_list = []
    
    for reactant, product in reactions:
        reactant_idx = complex_to_idx[reactant]
        product_idx = complex_to_idx[product]
        
        if random_rates:
            rate = rng.uniform(0.1, 2.0)
        else:
            rate = 1.0
            
        reaction_list.append((reactant_idx, product_idx, rate))
    
    return reaction_list

def build_linkage_groups_from_reactions(reactions: List[Tuple[str, str]], all_complexes: List[str]) -> List[nx.Graph]:
    """
    Build linkage groups from reactions by finding connected components.
    
    Args:
        reactions: List of (reactant, product) tuples
        all_complexes: List of complex strings
        
    Returns:
        List of NetworkX graphs representing linkage classes
    """
    # Create a graph with complexes as nodes and reactions as edges
    G = nx.Graph()
    G.add_nodes_from(all_complexes)
    
    for reactant, product in reactions:
        G.add_edge(reactant, product)
    
    # Find connected components (linkage classes)
    linkage_groups = []
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        linkage_groups.append(subgraph.copy())
    
    return linkage_groups

def build_assignments_from_linkage_groups(linkage_groups: List[nx.Graph], all_complexes: List[str]) -> List[Dict]:
    """
    Build assignments from linkage groups.
    
    Args:
        linkage_groups: List of NetworkX graphs
        all_complexes: List of complex strings
        
    Returns:
        List of assignment dictionaries
    """
    assignments = []
    
    for G in linkage_groups:
        # Create a mapping from node indices to complex names
        # Since the graph nodes are already the complex names, this is straightforward
        node_assignment = {node: node for node in G.nodes()}
        assignments.append(node_assignment)
    
    return assignments

def compute_complexes_and_reactions_per_class(linkage_groups: List[nx.Graph]) -> Tuple[List[int], List[int]]:
    """
    Compute complexes_per_class and reactions_per_class from linkage groups.
    
    Args:
        linkage_groups: List of NetworkX graphs representing linkage classes
        
    Returns:
        complexes_per_class: List of number of complexes per linkage class
        reactions_per_class: List of number of reactions per linkage class
    """
    complexes_per_class = []
    reactions_per_class = []
    
    for G in linkage_groups:
        n_complexes = G.number_of_nodes()
        n_reactions = G.number_of_edges()
        
        complexes_per_class.append(n_complexes)
        reactions_per_class.append(n_reactions)
    
    return complexes_per_class, reactions_per_class

def validate_conservation_laws_against_stoichiometry(rn_instance, L: np.ndarray) -> None:
    """
    Validate that the provided conservation laws are consistent with the stoichiometric matrix.
    
    Args:
        rn_instance: ReactionNetwork instance with stoichiometric matrix built
        L: Conservation law matrix to validate
        
    Raises:
        ValueError: If the conservation laws are inconsistent with the stoichiometric matrix
    """
    # Get the stoichiometric matrix
    S = rn_instance.get_stoichiometric_matrix()
    
    # Check dimensions
    if L.shape[1] != S.shape[0]:
        raise ValueError(f"Conservation law matrix L has {L.shape[1]} columns but stoichiometric matrix S has {S.shape[0]} rows")
    
    # Check that L * S = 0 (conservation laws are in the left nullspace of S)
    L_times_S = L @ S
    if not np.allclose(L_times_S, 0, atol=1e-10):
        raise ValueError("Conservation laws are inconsistent with the stoichiometric matrix. L * S should be zero.")
    
    # Check that L has full row rank (no redundant conservation laws)
    if np.linalg.matrix_rank(L) < L.shape[0]:
        raise ValueError("Conservation law matrix L has redundant rows (not full row rank)")
    
    # Check that the number of conservation laws is correct
    # The number of conservation laws should equal n_species - rank(S)
    expected_num_conservation_laws = S.shape[0] - np.linalg.matrix_rank(S)
    if L.shape[0] != expected_num_conservation_laws:
        raise ValueError(f"Expected {expected_num_conservation_laws} conservation laws but got {L.shape[0]}")


def compute_probs(C_full, L_vec, l0):
    return (L_vec[:, np.newaxis] * C_full / l0) if C_full.ndim == 2 else (L_vec * C_full / l0)

