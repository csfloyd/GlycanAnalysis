"""
Analysis module for reaction networks.

This module contains functions for analyzing reaction networks, including
conservation law computation and validation.
"""

import numpy as np
import sympy
import itertools
from typing import List, Dict, Tuple

def count_conservation_group_changes(r_n):
    # Get conservation groups
    conservation_groups = r_n.get_conservation_groups()

    M_0t1 = 0
    M_1t0 = 0 
    M_0b1 = 0 # symmetric

    # Loop through reactions and check conservation group membership
    for i, (c1_idx, c2_idx, _) in enumerate(r_n.reactions):
        c1 = r_n.all_complexes[c1_idx]
        c2 = r_n.all_complexes[c2_idx]
        
        for pairing_group in r_n.within_pairing_groups:
            if c1 in pairing_group and c2 in pairing_group.keys():
                break

        for pairing_group in r_n.between_pairing_groups:
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

def compute_cycles(r_n):
        """
        Compute cycles in the reaction network by finding the nullspace of the stoichiometric matrix.
        
        Returns:
            Z: Matrix of cycles
            cycles: List of cycles where each cycle is a list of reaction strings
        """
        """
        Compute cycles in the reaction network by finding the nullspace of the stoichiometric matrix.
        
        Args:
            r_n: ReactionNetwork instance
            force_reverse: Boolean indicating if reactions are forced to be reversible
            
        Returns:
            Z: Matrix of cycles
            cycles: List of cycles where each cycle is a list of reaction strings
        """
        S = r_n.get_stoichiometric_matrix()
 
        if r_n.force_reverse:
            S_red = S[:,0:-1:2] 
        else:
            S_red = S

        ns = sympy.Matrix(S_red).nullspace()  # list of column vectors
        Z = sympy.Matrix.hstack(*ns) if ns else sympy.Matrix.zeros(S_red.shape[1], 0)
        
        
        cycles = []
        reaction_strings = r_n.get_reaction_strings(include_rates=False)
        
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