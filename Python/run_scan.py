from CRNs import *
import numpy as np
import argparse
import time
import copy
import pickle


# Create argument parser
parser = argparse.ArgumentParser(description="SLURM job script with arguments.")

# Define command-line arguments
parser.add_argument("--param1", type=int, required=True, help="An integer parameter")
parser.add_argument("--param2", type=int, required=False, help="An integer parameter")
parser.add_argument("--output", type=str, required=True, help="A string parameter")

# Parse arguments
args = parser.parse_args()

output_dir = args.output

n_species = 8
n_complexes = 9
n_reactions = args.param1
force_reverse = True
L = np.array([[2, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 2, 1, 1, 1]])
n_cons = len(L)
n_lcs = n_complexes - n_species + n_cons
complexes_per_class = [2, 3, 4]
reactions_per_class = [1, 3, 6]
subset_group_ind = None

t_span = (0, 10000)
num_points = 10000
int_method = 'Radau'
r_tol = 1e-3    
a_tol = 1e-3
sign_conditions_list = []
M_0t1_list = []
M_1t0_list = []
M_0b1_list = []
r_n_list = []

# Profiling variables
total_integration_time = 0.0
total_network_generation_time = 0.0
total_symbolic_time = 0.0
total_sensitivity_time = 0.0
total_nnls_time = 0.0
total_conservation_laws_time = 0.0
total_symbolic_rhs_time = 0.0
total_derivatives_time = 0.0
total_conservation_analysis_time = 0.0
total_simulator_init_time = 0.0
total_rates_creation_time = 0.0
total_flexible_rhs_time = 0.0
total_initial_conditions_time = 0.0
total_species_recovery_time = 0.0
total_sign_processing_time = 0.0
total_data_storage_time = 0.0

start_time = time.time()
for iter in range(25):
    seed = np.random.randint(1000000)

    # Time network generation
    network_start = time.time()
    try:
        r_n = ReactionNetwork(  
            n_species, 
            n_complexes, n_reactions, n_lcs, 
            L, seed, force_reverse=force_reverse, subset_group_ind=subset_group_ind, 
            complexes_per_class=complexes_per_class, reactions_per_class=reactions_per_class
    )
        network_generation_time = time.time() - network_start
        total_network_generation_time += network_generation_time
    except Exception as e:
        print(f"Error creating reaction network: {e}, skipping...")
        continue

    # Time conservation group analysis
    conservation_analysis_start = time.time()
    M_0t1, M_1t0, M_0b1 = count_conservation_group_changes(r_n)
    conservation_analysis_time = time.time() - conservation_analysis_start
    total_conservation_analysis_time += conservation_analysis_time
    
    # Time data storage operations
    data_storage_start = time.time()
    M_0t1_list.append(M_0t1)
    M_1t0_list.append(M_1t0)
    M_0b1_list.append(M_0b1)
    # Store the parameters used to create the network (can recreate later)
    r_n_params = {
        'all_complexes': r_n.all_complexes,
        'reactions': r_n.reactions
    }
    r_n_list.append(r_n_params)
    data_storage_time = time.time() - data_storage_start
    total_data_storage_time += data_storage_time

    # Time simulator initialization
    simulator_init_start = time.time()
    sim = ReactionNetworkSimulator(r_n)
    simulator_init_time = time.time() - simulator_init_start
    total_simulator_init_time += simulator_init_time
    
    # Time symbolic computations
    symbolic_start = time.time()
    
    # Time conservation laws solving
    conservation_start = time.time()
    sim.solve_conservation_laws()
    conservation_time = time.time() - conservation_start
    total_conservation_laws_time += conservation_time
    
    # Time symbolic RHS generation
    rhs_start = time.time()
    reduced_rhs, remaining_syms, const_syms, rate_syms = sim.get_symbolic_reduced_rhs()
    rhs_time = time.time() - rhs_start
    total_symbolic_rhs_time += rhs_time
    
    # Time derivatives computation
    derivatives_start = time.time()
    # Only compute the derivatives we actually use
    dR_dC, dR_dC_func, dR_dl, dR_dl_func, dR_dk, dR_dk_func, remaining_syms = sim.get_first_order_derivatives()
    derivatives_time = time.time() - derivatives_start
    total_derivatives_time += derivatives_time
    
    symbolic_time = time.time() - symbolic_start
    total_symbolic_time += symbolic_time


    # Time rates creation
    rates_creation_start = time.time()
    rates = np.array([r_n.reactions[r_idx][2] for r_idx in range(len(r_n.reactions))])
    rates_creation_time = time.time() - rates_creation_start
    total_rates_creation_time += rates_creation_time
    
    # Time flexible RHS creation
    flexible_rhs_start = time.time()
    flexible_reduced_ode_rhs = sim.make_reduced_rhs_with_conservation_flexible()
    flexible_rhs_time = time.time() - flexible_rhs_start
    total_flexible_rhs_time += flexible_rhs_time
 
    sign_conditions = []    
    for n in range(500):

        l0 = np.exp(np.random.uniform(np.log(0.0001), np.log(1000.0), size=L.shape[0]))  # Sample uniformly in log space between 0.1 and 1000

        # Time NNLS
        nnls_start = time.time()
        C_full = generate_positive_initial_concentrations_nnls(L, l0)
        nnls_time = time.time() - nnls_start
        total_nnls_time += nnls_time
        
        # Time initial conditions processing
        initial_conditions_start = time.time()
        _, C_reduced_init = sim.get_const_and_reduced_init(C_full)
        initial_conditions_time = time.time() - initial_conditions_start
        total_initial_conditions_time += initial_conditions_time

        # Time ODE integration
        integration_start = time.time()
        # Set up timeout for ODE integration (30 seconds)
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(1)  # 30 second timeout
            # Try fast integration first, fall back to standard if needed
            sol_reduced, C_reduced_final = sim.integrate(
                lambda C: flexible_reduced_ode_rhs(C, l0), C_reduced_init, t_span=t_span,
                num_points=num_points, method=int_method, rtol=r_tol, atol=a_tol
            )
            signal.alarm(0)   # Cancel timeout
            integration_time = time.time() - integration_start
            total_integration_time += integration_time
        except TimeoutError:
            signal.alarm(0)   # Cancel timeout
            print(f"  Integration {n} timed out, skipping...")
            continue
        except Exception as e:
            signal.alarm(0)   # Cancel timeout
            print(f"  Integration {n} failed with error: {e}, skipping...")
            continue

        # Time species recovery
        species_recovery_start = time.time()
        C_full = sim.recover_eliminated_species(l0, C_reduced_final)
        species_recovery_time = time.time() - species_recovery_start
        total_species_recovery_time += species_recovery_time
        
        # Time sensitivity analysis
        sensitivity_start = time.time()
        dC_dl = sim.dC_dl_func(C_reduced_final, l0, rates, dR_dC_func, dR_dl_func)
        dC_dl_full = sim.compute_dC_dk_full(dC_dl)
        sensitivity_time = time.time() - sensitivity_start
        total_sensitivity_time += sensitivity_time
        
        # Time sign processing
        sign_processing_start = time.time()
        signs = np.sign(np.round(dC_dl_full, decimals=12)).tolist()
        if signs not in sign_conditions:
            sign_conditions.append(signs)
        sign_processing_time = time.time() - sign_processing_start
        total_sign_processing_time += sign_processing_time
    
    # Time final data storage
    final_storage_start = time.time()
    sign_conditions_list.append(sign_conditions)
    final_storage_time = time.time() - final_storage_start
    total_data_storage_time += final_storage_time

    if iter%5 == 0:
        print(f"Iteration {iter} complete")
    

end_time = time.time()
total_time = end_time - start_time

# Print detailed timing information
print(f"\n=== TIMING SUMMARY ===")
print(f"Total time: {total_time:.2f} seconds")
print(f"Network generation time: {total_network_generation_time:.2f} seconds ({total_network_generation_time/total_time*100:.1f}%)")
print(f"ODE integration time: {total_integration_time:.2f} seconds ({total_integration_time/total_time*100:.1f}%)")
print(f"Symbolic computations: {total_symbolic_time:.2f} seconds ({total_symbolic_time/total_time*100:.1f}%)")
print(f"  - Conservation laws solving: {total_conservation_laws_time:.2f} seconds ({total_conservation_laws_time/total_time*100:.1f}%)")
print(f"  - Symbolic RHS generation: {total_symbolic_rhs_time:.2f} seconds ({total_symbolic_rhs_time/total_time*100:.1f}%)")
print(f"  - Derivatives computation: {total_derivatives_time:.2f} seconds ({total_derivatives_time/total_time*100:.1f}%)")
print(f"Sensitivity analysis: {total_sensitivity_time:.2f} seconds ({total_sensitivity_time/total_time*100:.1f}%)")
print(f"NNLS (initial concentrations): {total_nnls_time:.2f} seconds ({total_nnls_time/total_time*100:.1f}%)")
print(f"Conservation group analysis: {total_conservation_analysis_time:.2f} seconds ({total_conservation_analysis_time/total_time*100:.1f}%)")
print(f"Simulator initialization: {total_simulator_init_time:.2f} seconds ({total_simulator_init_time/total_time*100:.1f}%)")
print(f"Rates creation: {total_rates_creation_time:.2f} seconds ({total_rates_creation_time/total_time*100:.1f}%)")
print(f"Flexible RHS creation: {total_flexible_rhs_time:.2f} seconds ({total_flexible_rhs_time/total_time*100:.1f}%)")
print(f"Initial conditions processing: {total_initial_conditions_time:.2f} seconds ({total_initial_conditions_time/total_time*100:.1f}%)")
print(f"Species recovery: {total_species_recovery_time:.2f} seconds ({total_species_recovery_time/total_time*100:.1f}%)")
print(f"Sign processing: {total_sign_processing_time:.2f} seconds ({total_sign_processing_time/total_time*100:.1f}%)")
print(f"Data storage: {total_data_storage_time:.2f} seconds ({total_data_storage_time/total_time*100:.1f}%)")
other_time = total_time - total_network_generation_time - total_integration_time - total_symbolic_time - total_sensitivity_time - total_nnls_time - total_conservation_analysis_time - total_simulator_init_time - total_rates_creation_time - total_flexible_rhs_time - total_initial_conditions_time - total_species_recovery_time - total_sign_processing_time - total_data_storage_time
print(f"Other operations: {other_time:.2f} seconds ({other_time/total_time*100:.1f}%)")
print(f"=====================\n")

save_data = (M_0t1_list, M_1t0_list, M_0b1_list, r_n_list, sign_conditions_list)

# Save to a file
with open(output_dir + "/SavedData.pkl", "wb") as file:
    pickle.dump(save_data, file)
