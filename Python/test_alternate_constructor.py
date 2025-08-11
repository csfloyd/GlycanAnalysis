#!/usr/bin/env python3
"""
Test script for the new ReactionNetwork.from_reaction_strings constructor.
"""

import numpy as np
from ConservationLaws import ReactionNetwork

def test_alternate_constructor():
    """Test the new from_reaction_strings constructor."""
    
    # Define a simple reaction network
    reaction_strings = [
        'A+B->C',
        'C->D+E',
        'D->A+B'
    ]
    
    # Define conservation laws (A+B+C+D+E = constant)
    L = np.array([[1, 1, 1, 1, 1]])
    
    print("Creating ReactionNetwork from reaction strings...")
    print(f"Reaction strings: {reaction_strings}")
    print(f"Conservation law matrix L:\n{L}")
    print("Conservation laws will be validated against stoichiometric matrix")
    
    # Create network using the new constructor
    rn = ReactionNetwork.from_reaction_strings(
        reaction_strings=reaction_strings,
        L=L,
        seed=42,
        force_reverse=False,  # Not forcing reversible reactions
        random_rates=True
    )
    
    print(f"\nNetwork created successfully!")
    print(f"Number of species: {rn.n_species}")
    print(f"Species names: {rn.species_names}")
    print(f"Number of complexes: {len(rn.all_complexes)}")
    print(f"Complexes: {rn.all_complexes}")
    print(f"Number of reactions: {rn.n_reactions}")
    
    print(f"\nReactions:")
    rn.print_reactions(include_rates=True)
    
    print(f"\nConservation groups: {rn.conservation_groups}")
    print(f"Number of pairing groups: {len(rn.pairing_groups)}")
    
    # Show complexes_per_class and reactions_per_class
    print(f"\nLinkage class information:")
    print(f"Complexes per class: {rn.complexes_per_class}")
    print(f"Reactions per class: {rn.reactions_per_class}")
    
    # Show validated conservation laws
    print(f"\nValidated conservation laws:")
    print(f"Conservation law matrix L:\n{rn.L}")
    print(f"Number of conservation laws: {rn.L.shape[0]}")
    
    # Test that the network is functionally equivalent to a regular one
    print(f"\nTesting network functionality...")
    
    # Test stoichiometric matrix
    S = rn.get_stoichiometric_matrix()
    print(f"Stoichiometric matrix shape: {S.shape}")
    
    # Test reaction strings
    reaction_strings_output = rn.get_reaction_strings(include_rates=False)
    print(f"Generated reaction strings: {reaction_strings_output}")
    
    # Test that we can create a simulator
    from ConservationLaws import ReactionNetworkSimulator
    simulator = ReactionNetworkSimulator(rn)
    print(f"Simulator created successfully!")
    
    print(f"\nTest completed successfully!")

def test_complex_reaction_network():
    """Test with a more complex reaction network."""
    
    # Define a more complex reaction network
    reaction_strings = [
        'A+B->C',
        'C->D+E', 
        'D+E->F',
        'F->A+B',
        'C->G',
        'G->H',
        'H->I'
    ]
    
    # Define conservation laws
    L = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0, 0],  # A+B+C+D+E+F = constant
        [0, 0, 0, 0, 0, 1, 1, 1, 1]   # G+H+I = constant
    ])
    
    print("\n" + "="*50)
    print("Testing complex reaction network...")
    print(f"Reaction strings: {reaction_strings}")
    print(f"Conservation law matrix L:\n{L}")
    print("Conservation laws will be validated against stoichiometric matrix")
    
    rn = ReactionNetwork.from_reaction_strings(
        reaction_strings=reaction_strings,
        L=L,
        seed=123,
        force_reverse=False,
        random_rates=True
    )
    
    print(f"Network created successfully!")
    print(f"Species: {rn.species_names}")
    print(f"Complexes: {rn.all_complexes}")
    print(f"Number of reactions: {rn.n_reactions}")
    
    print(f"\nReactions:")
    rn.print_reactions(include_rates=True)
    
    print(f"Conservation groups: {rn.conservation_groups}")
    print(f"Number of linkage groups: {len(rn.linkage_groups)}")
    
    # Show complexes_per_class and reactions_per_class
    print(f"\nLinkage class information:")
    print(f"Complexes per class: {rn.complexes_per_class}")
    print(f"Reactions per class: {rn.reactions_per_class}")
    
    # Show validated conservation laws
    print(f"\nValidated conservation laws:")
    print(f"Conservation law matrix L:\n{rn.L}")
    print(f"Number of conservation laws: {rn.L.shape[0]}")
    
    # Test cycles computation
    Z, cycles = rn.compute_cycles()
    print(f"Number of cycles: {len(cycles)}")
    if cycles:
        print(f"First cycle: {cycles[0]}")

def test_validation_errors():
    """Test validation errors with inconsistent conservation laws."""
    
    print("\n" + "="*50)
    print("Testing validation errors...")
    
    # Define a simple reaction network
    reaction_strings = ['A+B->C', 'C->D+E']
    
    # Test 1: Wrong number of conservation laws
    print("\nTest 1: Too many conservation laws")
    L_wrong_count = np.array([
        [1, 1, 1, 1, 1],  # A+B+C+D+E = constant
        [1, 0, 0, 0, 0]   # A = constant (redundant)
    ])
    
    try:
        rn = ReactionNetwork.from_reaction_strings(
            reaction_strings=reaction_strings,
            L=L_wrong_count,
            seed=42
        )
    except ValueError as e:
        print(f"Expected error: {e}")
    
    # Test 2: Inconsistent conservation law
    print("\nTest 2: Inconsistent conservation law")
    L_inconsistent = np.array([[1, 1, 0, 0, 0]])  # A+B = constant (inconsistent with reactions)
    
    try:
        rn = ReactionNetwork.from_reaction_strings(
            reaction_strings=reaction_strings,
            L=L_inconsistent,
            seed=42
        )
    except ValueError as e:
        print(f"Expected error: {e}")
    
    # Test 3: Wrong dimensions
    print("\nTest 3: Wrong dimensions")
    L_wrong_dim = np.array([[1, 1, 1]])  # Only 3 columns instead of 5
    
    try:
        rn = ReactionNetwork.from_reaction_strings(
            reaction_strings=reaction_strings,
            L=L_wrong_dim,
            seed=42
        )
    except ValueError as e:
        print(f"Expected error: {e}")

def test_force_reverse():
    """Test the force_reverse functionality."""
    
    print("\n" + "="*50)
    print("Testing force_reverse functionality...")
    
    # Define a simple reaction network
    reaction_strings = [
        'A->B',
        'B->C',
        'C->D'
    ]
    
    # Define conservation laws (A+B+C+D = constant)
    L = np.array([[1, 1, 1, 1]])
    
    print(f"Original reaction strings: {reaction_strings}")
    print(f"Conservation law matrix L:\n{L}")
    
    # Test without force_reverse
    print("\n--- Without force_reverse ---")
    rn_no_reverse = ReactionNetwork.from_reaction_strings(
        reaction_strings=reaction_strings,
        L=L,
        seed=42,
        force_reverse=False,
        random_rates=True
    )
    
    print(f"Number of reactions: {rn_no_reverse.n_reactions}")
    print("Reactions:")
    rn_no_reverse.print_reactions(include_rates=True)
    print(f"Complexes per class: {rn_no_reverse.complexes_per_class}")
    print(f"Reactions per class: {rn_no_reverse.reactions_per_class}")
    
    # Test with force_reverse
    print("\n--- With force_reverse ---")
    rn_with_reverse = ReactionNetwork.from_reaction_strings(
        reaction_strings=reaction_strings,
        L=L,
        seed=42,
        force_reverse=True,
        random_rates=True
    )
    
    print(f"Number of reactions: {rn_with_reverse.n_reactions}")
    print("Reactions:")
    rn_with_reverse.print_reactions(include_rates=True)
    print(f"Complexes per class: {rn_with_reverse.complexes_per_class}")
    print(f"Reactions per class: {rn_with_reverse.reactions_per_class}")
    
    # Show that the number of reactions doubled
    print(f"\nReaction count comparison:")
    print(f"Without force_reverse: {rn_no_reverse.n_reactions} reactions")
    print(f"With force_reverse: {rn_with_reverse.n_reactions} reactions")
    print(f"Expected doubling: {rn_no_reverse.n_reactions * 2} reactions")

if __name__ == "__main__":
    test_alternate_constructor()
    test_complex_reaction_network()
    test_validation_errors()
    test_force_reverse()
