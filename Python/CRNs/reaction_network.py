"""
Reaction Network module.

This module contains the main ReactionNetwork class with all its methods,
including enhanced functionality for arbitrary species names and conservation law computation.
"""

import numpy as np
import sympy
import networkx as nx
import string
import random
from typing import List, Dict, Tuple, Optional
import itertools

from .generation import (
    random_connected_graph
)


class ReactionNetwork:
    """
    Represents the structure and symbolic properties of a chemical reaction network.
    Handles network construction, symbolic ODE generation, and conservation law management.
    """
    
    def __init__(self, n_species: int, 
                 n_complexes: int, n_reactions: int, n_lcs: int,  
                 L: np.ndarray, 
                 seed: int, force_reverse: bool = True, subset_group_ind: int = None,
                 complexes_per_class: List[int] = None, reactions_per_class: List[int] = None,
                 species_names: List[str] = None):
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
            species_names: List of species names. If None, uses default single-letter names (A, B, C, ...)
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
                
                # Set species names - use provided names or default to single letters
                if species_names is not None:
                    if len(species_names) != n_species:
                        raise ValueError(f"Expected {n_species} species names, but got {len(species_names)}")
                    self.species_names = species_names
                else:
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
                
                self.complexes_dict = self._generate_string_vector_dict(n_species)
                
                self.pairing_groups = self._generate_pairing_groups(L, self.complexes_dict)
                
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
                self.n_cons = len(self.conservation_groups)

                # deficiency check is now inside the try block
                rank_YA = np.linalg.matrix_rank(self.Y @ self.A)
                if n_species - rank_YA == self.n_cons:
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
                             subset_group_ind: int = None, random_rates: bool = True,
                             species_names: List[str] = None):
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
            species_names: List of species names. If None, will be inferred from reactions
            
        Returns:
            ReactionNetwork instance
            
        Raises:
            ValueError: If the provided conservation laws are inconsistent with the stoichiometric matrix
        """
        # Parse reaction strings to extract complexes and reactions
        complexes, reactions = cls._parse_reaction_strings(reaction_strings, force_reverse)
        
        # Determine species names and number of species
        if species_names is not None:
            # Use provided species names
            instance_species_names = species_names
            n_species = len(species_names)
        else:
            # Infer species names from complexes
            all_species = set()
            for complex_str in complexes:
                species_list = complex_str.split('+')
                all_species.update(species_list)
            instance_species_names = sorted(list(all_species))
            n_species = len(all_species)
        
        # Create a minimal instance to set up basic attributes
        instance = cls.__new__(cls)
        instance.seed = seed
        instance.rng = np.random.default_rng(instance.seed)
        instance.py_rng = random.Random(instance.seed)
        instance.force_reverse = force_reverse
        instance.subset_group_ind = subset_group_ind
        instance.species_names = instance_species_names
        
        # Build complexes dictionary for the actual species present
        instance.complexes_dict = instance._build_complexes_dict(complexes)
        
        # Build Y matrix and all_complexes from the actual complexes
        instance.Y, instance.all_complexes = instance._build_Y_matrix_from_complexes(complexes)
        
        # Build reactions list from parsed reactions
        instance.reactions = instance._build_reactions_from_parsed(reactions, instance.all_complexes, instance.rng, random_rates, force_reverse)
        
        # Build stoichiometric matrix and validate conservation laws
        instance.A = instance._build_A_matrix_incidence()
        if L is None:
            # Get the stoichiometric matrix and compute conservation laws
            S = instance.get_stoichiometric_matrix()
            L = instance.compute_conservation_laws()
        instance.validate_conservation_laws(L)
        instance.L = L
        
        # Build pairing groups and conservation groups
        instance.pairing_groups = instance._generate_pairing_groups(instance.L, instance.complexes_dict)
        # Call these methods after the instance is fully set up
        instance.conservation_groups = instance.get_conservation_groups()
        instance.n_cons = len(instance.conservation_groups)
        instance.within_pairing_groups, instance.between_pairing_groups = instance.get_within_between_pairing_groups()
        
        # Build remaining matrices and functions
        instance.Psi = instance._build_Psi_function()
        instance.n_species = n_species
        instance.n_reactions = len(instance.reactions)
        
        # Set linkage groups and assignments (simplified for deterministic case)
        instance.linkage_groups = instance._build_linkage_groups(reactions)
        instance.assignments = instance._build_assignments()
        
        # Compute and store complexes_per_class and reactions_per_class
        instance.complexes_per_class, instance.reactions_per_class = instance._compute_complexes_and_reactions_per_class()
        
        return instance

    def compute_conservation_laws(self) -> np.ndarray:
        """
        Compute conservation laws from stoichiometric matrix.
        
        This function finds the left nullspace of the stoichiometric matrix S,
        then systematically searches for non-negative integer combinations
        of the basis vectors to find physically meaningful conservation laws.
        
        Args:
            S: Stoichiometric matrix with shape (n_species, n_reactions)
            
        Returns:
            L: Conservation law matrix with shape (n_conservation_laws, n_species)
            
        Raises:
            ValueError: If no valid conservation laws are found
        """
        # Get the nullspace using SymPy for exact computation
        S = self.get_stoichiometric_matrix()
        S_sympy = sympy.Matrix(S)
        S_transpose = S_sympy.T
        nullspace_vectors = S_transpose.nullspace()
        
        if not nullspace_vectors:
            # No conservation laws found
            return np.array([]).reshape(0, S.shape[0])
        
        # Convert sympy vectors to numpy array
        nullspace_basis = np.array([list(vector) for vector in nullspace_vectors], dtype=float)
        
        # Ensure we have a 2D array even if only one conservation law
        if nullspace_basis.ndim == 1:
            nullspace_basis = nullspace_basis.reshape(1, -1)
        
        n_basis_vectors = nullspace_basis.shape[0]
        
        # Try integer combinations of the basis vectors
        max_coeff = 3  # Maximum coefficient to try
        
        conservation_laws = []
        
        # Generate all combinations of coefficients
        coeff_ranges = [range(-max_coeff, max_coeff + 1)] * n_basis_vectors
        
        # Try all combinations in one loop
        for coeffs in itertools.product(*coeff_ranges):
            # Skip the zero combination
            if all(c == 0 for c in coeffs):
                continue
                
            # Compute the linear combination
            combination = np.dot(coeffs, nullspace_basis)
            
            # Check if it's integer and non-negative
            if np.all(np.mod(combination, 1) == 0) and np.all(combination >= 0) and not np.allclose(combination, 0):
                # Check if this combination is in the span of the current conservation laws
                if len(conservation_laws) == 0:
                    conservation_laws.append(combination)
                else:
                    # Create matrix of current conservation laws
                    current_matrix = np.array(conservation_laws)
                    
                    # Try to solve: current_matrix * x = combination
                    # If there's a solution, the combination is in the span
                    try:
                        # Use least squares to find coefficients
                        coeffs, residuals, rank, s = np.linalg.lstsq(current_matrix.T, combination, rcond=None)
                        
                        # Check if the residual is small (meaning it's in the span)
                        if len(residuals) > 0 and residuals[0] < 1e-10:
                            # It's in the span, so skip it
                            continue
                        else:
                            # Not in the span, so add it
                            conservation_laws.append(combination)
                    except np.linalg.LinAlgError:
                        # If the system is underdetermined, it might still be in the span
                        # Try a different approach: check if the augmented matrix has the same rank
                        augmented_matrix = np.vstack([current_matrix, combination])
                        if np.linalg.matrix_rank(augmented_matrix) == np.linalg.matrix_rank(current_matrix):
                            # Same rank means it's in the span
                            continue
                        else:
                            # Different rank means it's not in the span
                            conservation_laws.append(combination)
        
        if not conservation_laws:
            raise ValueError("No non-negative integer conservation laws found. "
                            "The nullspace vectors do not admit non-negative integer combinations.")
        
        # Convert to numpy array
        L = np.array(conservation_laws, dtype=int)
        
        return L
    
    def validate_conservation_laws(self, L: np.ndarray) -> None:
        """
        Validate conservation laws against the stoichiometric matrix.
        
        Args:
            L: Conservation law matrix to validate
            
        Raises:
            ValueError: If conservation laws are inconsistent with stoichiometric matrix
        """
        S = self.get_stoichiometric_matrix()
        
        # Check that L * S = 0 (conservation laws are in the left nullspace of S)
        product = np.dot(L, S)
        if not np.allclose(product, 0, atol=1e-10):
            raise ValueError("Conservation laws are not consistent with stoichiometric matrix: L * S != 0")
        
        # Check that conservation laws are linearly independent
        rank_L = np.linalg.matrix_rank(L)
        if rank_L != L.shape[0]:
            raise ValueError("Conservation laws are not linearly independent")
        
    @classmethod
    def _parse_reaction_strings(cls, reaction_strings: List[str], force_reverse: bool = False) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Parse a list of reaction strings to extract complexes and reactions.
        
        Args:
            reaction_strings: List of reaction strings like ['A+B->C', 'C->D+E']
            force_reverse: If True, automatically adds reverse reactions for each input reaction
            
        Returns:
            complexes: List of unique complex strings
            reactions: List of (reactant, product) tuples
        """
        """
        Parse a list of reaction strings to extract complexes and reactions.
        
        Args:
            reaction_strings: List of strings like ['A+B->C', 'C->D+E', '2*A + B -> C']
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


    @classmethod
    def _parse_complex_string(cls, complex_str: str, species_names: List[str]) -> Dict[str, int]:
        """
        Parse a complex string to extract species and their stoichiometric coefficients.
        
        Args:
            complex_str: Complex string like '2*A + B' or 'A+B'
            species_names: List of valid species names
            
        Returns:
            Dictionary mapping species names to their stoichiometric coefficients
        """
        species_coeffs = {}
        
        # Split on '+' and handle stoichiometric coefficients
        parts = [part.strip() for part in complex_str.split('+')]
        
        for part in parts:
            if '*' in part:
                # Handle stoichiometric coefficients like '2*A'
                coeff_part, species_part = part.split('*', 1)
                try:
                    coeff = int(coeff_part.strip())
                except ValueError:
                    coeff = 1
                species = species_part.strip()
            else:
                # No stoichiometric coefficient, assume 1
                coeff = 1
                species = part.strip()
            
            if species in species_names:
                species_coeffs[species] = species_coeffs.get(species, 0) + coeff
            else:
                raise ValueError(f"Unknown species '{species}' in complex '{complex_str}'. Valid species: {species_names}")
        
        return species_coeffs
    
    def _build_complexes_dict(self, complexes: List[str]) -> Dict[str, np.ndarray]:
        """
        Build complexes dictionary from a list of complex strings.
        
        Args:
            complexes: List of complex strings like ['A', 'B', 'A+B', '2*A + B']
            
        Returns:
            Dictionary mapping complex strings to stoichiometric vectors
        """
        complexes_dict = {}
        
        for complex_str in complexes:
            vec = np.zeros(len(self.species_names), dtype=int)
            species_coeffs = self.__class__._parse_complex_string(complex_str, self.species_names)
            
            # Fill the vector with stoichiometric coefficients
            for species, coeff in species_coeffs.items():
                species_index = self.species_names.index(species)
                vec[species_index] = coeff
            
            complexes_dict[complex_str] = vec
        
        return complexes_dict
    
    def _build_Y_matrix_from_complexes(self, complexes: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Build Y matrix and all_complexes list from complexes.
        
        Args:
            complexes: List of complex strings
            
        Returns:
            Y: Stoichiometric matrix
            all_complexes: List of complex strings in order
        """
        all_complexes = sorted(complexes)  # Sort for deterministic ordering
        Y = np.zeros((len(self.species_names), len(all_complexes)), dtype=int)
        
        for j, complex_name in enumerate(all_complexes):
            Y[:, j] = self.complexes_dict[complex_name]
        
        return Y, all_complexes
    
    def _build_reactions_from_parsed(self,reactions: List[Tuple[str, str]], all_complexes: List[str], 
                                         rng: np.random.Generator, random_rates: bool = True, 
                                         force_reverse: bool = False) -> List[Tuple[int, int, float]]:
        """
        Build reactions list from parsed reactions.
        
        Args:
            reactions: List of (reactant, product) tuples
            random_rates: Whether to use random rates
            force_reverse: Whether reactions are forced to be reversible
            
        Returns:
            List of (reactant_index, product_index, rate) tuples
        """
        complex_to_idx = {complex_name: i for i, complex_name in enumerate(self.all_complexes)}
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
    
    def _build_linkage_groups(self, reactions: List[Tuple[str, str]]) -> List[nx.Graph]:
        """
        Build linkage groups from reactions.
        
        Args:
            reactions: List of (reactant, product) tuples
            
        Returns:
            List of NetworkX graphs representing linkage classes
        """
        # Create a graph with complexes as nodes and reactions as edges
        G = nx.Graph()
        G.add_nodes_from(self.all_complexes)
    
        for reactant, product in reactions:
            G.add_edge(reactant, product)
    
        # Find connected components (linkage classes)
        linkage_groups = []
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            linkage_groups.append(subgraph.copy())
        
        return linkage_groups
    
    def _build_assignments(self) -> List[Dict]:
        """
        Build assignments from linkage groups.
        
        Returns:
            List of assignment dictionaries
        """
        assignments = []
        for G in self.linkage_groups:
            node_assignment = {node: node for node in G.nodes()}
            assignments.append(node_assignment)
        return assignments
    
    def _compute_complexes_and_reactions_per_class(self) -> Tuple[List[int], List[int]]:
        """
        Compute complexes_per_class and reactions_per_class from linkage groups.
        
        Returns:
            complexes_per_class: List of number of complexes per linkage class
            reactions_per_class: List of number of reactions per linkage class
        """
        complexes_per_class = []
        reactions_per_class = []
        
        for G in self.linkage_groups:
            n_complexes = G.number_of_nodes()
            n_reactions = G.number_of_edges()
            
            complexes_per_class.append(n_complexes)
            reactions_per_class.append(n_reactions)
        
        return complexes_per_class, reactions_per_class 
    
    def _generate_string_vector_dict(self, n: int) -> Dict[str, np.ndarray]:
        """
        Generate a dictionary mapping string representations to vectors.
        
        Args:
            n: Number of species
            
        Returns:
            Dictionary mapping complex strings to stoichiometric vectors
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
    
    def _generate_pairing_groups(self, L: np.ndarray, complexes_dict: Dict[str, np.ndarray]) -> List[List[str]]:
        """
        Generate pairing groups from conservation laws and complexes dictionary.
        
        Args:
            L: Conservation law matrix
            complexes_dict: Dictionary mapping complex strings to stoichiometric vectors
            
        Returns:
            List of pairing groups, where each group is a list of complex strings
        """
        pairing_groups = []
        available_complexes = set(complexes_dict.keys())
        while available_complexes:
            c = available_complexes.pop()
            valid_partners = self._find_valid_partners(c, L, complexes_dict)
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
    
    def _find_valid_partners(self, c_b: str, L: np.ndarray, complexes_dict: Dict[str, np.ndarray]) -> List[str]:
        """
        Find valid reaction partners for a given complex based on conservation laws.
        
        Args:
            c_b: Complex string
            L: Conservation law matrix
            complexes_dict: Dictionary mapping complex strings to stoichiometric vectors
            
        Returns:
            List of valid partner complex strings
        """
        s1 = complexes_dict[c_b]
        valid_partners = {}
        for c in complexes_dict:
            s_test = complexes_dict[c]
            diff = s1 - s_test
            # Check if this complex satisfies conservation laws
            if all(np.dot(l_vec, diff) == 0 for l_vec in L):
                valid_partners[c] = s_test
        return valid_partners

    def _random_partition_complexes_and_reactions(self, n_complexes, n_reactions, n_lcs):
        """Randomly partition complexes and reactions into linkage classes."""
        # Randomly partition complexes into linkage classes
        complexes_per_class = []
        remaining_complexes = n_complexes
        
        for i in range(n_lcs - 1):
            if remaining_complexes <= n_lcs - i:
                # Distribute remaining complexes evenly
                complexes_per_class.append(1)
            else:
                # Randomly choose number of complexes for this class
                max_complexes = remaining_complexes - (n_lcs - i - 1)
                n_complexes_class = self.rng.integers(2, max_complexes + 1)
                complexes_per_class.append(n_complexes_class)
                remaining_complexes -= n_complexes_class
        
        # Put remaining complexes in the last class
        complexes_per_class.append(remaining_complexes)
        
        # Randomly partition reactions into linkage classes
        reactions_per_class = []
        remaining_reactions = n_reactions
        
        for i, n_complexes in enumerate(complexes_per_class):
            if i == n_lcs - 1:
                # Put remaining reactions in the last class
                reactions_per_class.append(remaining_reactions)
            else:
                # Randomly choose number of reactions for this class
                min_reactions = n_complexes - 1
                if self.force_reverse:
                    max_reactions = n_complexes * (n_complexes - 1) // 2
                else:
                    max_reactions = n_complexes * (n_complexes - 1)
                
                max_possible = min(max_reactions, remaining_reactions - (n_lcs - i - 1))
                n_reactions_class = self.rng.integers(min_reactions, max_possible + 1)
                reactions_per_class.append(n_reactions_class)
                remaining_reactions -= n_reactions_class
        
        return complexes_per_class, reactions_per_class
    
    def _generate_linkage_class_graphs(self):
        """Generate linkage class graphs."""
        linkage_groups = []
        
        for n_complexes, n_reactions in zip(self.complexes_per_class, self.reactions_per_class):
            # Create a random connected graph for this linkage class
            G = random_connected_graph(
                n_complexes, n_reactions, self.rng, self.seed, 
                directed=not self.force_reverse
            )
            linkage_groups.append(G)
        
        return linkage_groups
    
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
        """Build Y matrix."""
        all_complexes = []
        for group in self.assignments:
            all_complexes.extend(sorted(group.values()))
        all_complexes = list(dict.fromkeys(all_complexes))
        
        Y = np.zeros((len(self.species_names), len(all_complexes)), dtype=int)
        for j, cname in enumerate(all_complexes):
            Y[:, j] = self.complexes_dict[cname]
        return Y, all_complexes

    
    def _build_reaction_list(self, random_rates=True):
        """Build reaction list."""
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
    
    def _build_A_matrix_incidence(self):
        """Build A matrix incidence"""
        n = len(self.all_complexes)
        A = np.zeros((n, n))
        for i, j, k in self.reactions:
            A[j, i] += k
            A[i, i] -= k
        return A

    def _build_Psi_function(self):
        """Build Psi function"""
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
        `new_rates` should be a list of floats. If force_reverse=True, it should have
        2 rates per reaction (forward and reverse). If force_reverse=False, it should
        have 1 rate per reaction.
        """
        if self.force_reverse:
            # For reversible reactions, we expect 2 rates per reaction
            expected_rates = len(self.reactions)
            if len(new_rates) != expected_rates:
                raise ValueError(f"Expected {expected_rates} rates for reversible reactions, but got {len(new_rates)}.")
        else:
            # For irreversible reactions, we expect 1 rate per reaction
            expected_rates = len(self.reactions)
            if len(new_rates) != expected_rates:
                raise ValueError(f"Expected {expected_rates} rates for irreversible reactions, but got {len(new_rates)}.")

        self.reactions = [(i, j, new_rate) for (i, j, _), new_rate in zip(self.reactions, new_rates)]
        self.A = self._build_A_matrix_incidence()

    
    
    