from CRNs import *
from CRNs.utils import NetworkDataLogger
import numpy as np
import argparse
import time
import copy
import pickle


# Create argument parser
parser = argparse.ArgumentParser(description="SLURM job script with arguments.")

# Define command-line arguments
parser.add_argument("--param1", type=float, required=True, help="An integer parameter")
parser.add_argument("--param2", type=int, required=False, help="An integer parameter")
parser.add_argument("--output", type=str, required=True, help="A string parameter")

# Parse arguments
args = parser.parse_args()

output_dir = args.output

# n_species = 8
# n_complexes = 9
# n_reactions = 8
# force_reverse = True
# L = np.array([[2, 1, 1, 1, 0, 0, 0, 0],
#               [0, 0, 0, 0, 2, 1, 1, 1]])
# n_cons = len(L)
# n_lcs = n_complexes - n_species + n_cons
# complexes_per_class = [2, 3, 4]
# reactions_per_class = [1, 3, 6]
# subset_group_ind = None
# seed = 30

# n_species = 6
# n_complexes = 7
# n_reactions = 4
# force_reverse = True
# L = np.array([[2, 1, 1, 0, 0, 0],
#               [0, 0, 0, 1, 1, 1]])
# n_cons = len(L)
# n_lcs = n_complexes - n_species + n_cons
# seed = None
# subset_group_ind = None

# n_species = 10
# n_complexes = 11
# n_reactions = 10
# force_reverse = True
# L = np.array([[2, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 2, 1, 1, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])

n_species = 8
n_complexes = 9
n_reactions = 9
force_reverse = True
L = np.array([[2, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 2, 1, 1, 1]])



n_cons = len(L)
n_lcs = n_complexes - n_species + n_cons
subset_group_ind = None
seed = None

E_range = 1
B_range = 1
F_range = args.param1
C0 = 1
beta = 1

n_graph_samples = 2500
# input_dim = 2
# default_l0 = np.array([1.0, 1.0, 1.0])
# sc_grad_dims = [[6,7,8,9],[0,1]]

input_dim = 1
default_l0 = np.array([1.0, 1.0])
sc_grad_dims = [[4,5,6,7],[0]]

t_span = (0, 10000)
num_points = 10000
int_method = 'LSODA'
r_tol = 1e-12   
a_tol = 1e-12

if seed is not None:
    np.random.seed(seed)
    random.seed(seed)

# Initialize data logger and time profiler
data_logger = NetworkDataLogger()
profiler = TimeProfiler()
profiler.start_total_timer()

sampler = "grid"

if sampler == "adaptive":
    ad_len = 50
    sampler = AdaptiveSampler(
        input_dims=[0],
        sc_grad_dims=sc_grad_dims,
        default_l0=default_l0,
        min_samples=ad_len,           # Minimum samples before checking convergence
        max_samples=1000,          # Maximum samples to prevent infinite loops
        convergence_window=ad_len,    # Window size for convergence check
        convergence_threshold=2/ad_len, # Stop when <% of recent samples are new
        timeout_seconds=5,       # Timeout for individual integrations
        l0_range=(1e-4, 1e1),
        profiler=profiler,         # Use existing profiler for timing
        round_decimals=6,
        use_signal_alarms=True
    )
else:
    sampler = GridSampler(
        input_dims=[0],
        default_l0=default_l0,
        sc_grad_dims=sc_grad_dims,
        l0_range=(1e-3, 1e1),
        l0_grid_size=20,
        grid_dim=1,
        timeout_seconds=5,
        profiler=profiler,
        round_decimals=6,
        use_signal_alarms=True
    )

for iter in range(n_graph_samples):
    seed = np.random.randint(1000000)

    # Time network generation
    profiler.start_timer("network_generation")
    try:
        r_n = ReactionNetwork(  
            n_species, 
            n_complexes, n_reactions, n_lcs, 
            L, seed, force_reverse=force_reverse, subset_group_ind=subset_group_ind, 
            #complexes_per_class=complexes_per_class, reactions_per_class=reactions_per_class
        )
        reac_rates = generate_thermodynamic_rates(r_n, C0, beta, E_range, B_range, F_range)
        r_n.update_rates(reac_rates)
        profiler.end_timer("network_generation")
    except Exception as e:
        profiler.end_timer("network_generation")
        print(f"Error creating reaction network: {e}, skipping...")
        continue

    # Time conservation group analysis
    profiler.start_timer("conservation_analysis")
    # M_0t1, M_1t0, M_0b1 = count_conservation_group_changes(r_n)
    interaction_matrix = get_interaction_matrix(r_n)
    profiler.end_timer("conservation_analysis")
    
    # Time data storage operations
    profiler.start_timer("data_storage")
    cycles = compute_cycles(r_n)
    profiler.end_timer("data_storage")

    # Time simulator initialization
    profiler.start_timer("simulator_init")
    sim = ReactionNetworkSimulator(r_n)
    profiler.end_timer("simulator_init")
    
    
    # Time conservation laws solving
    profiler.start_timer("conservation_laws")
    sim.solve_conservation_laws()
    profiler.end_timer("conservation_laws")
    
    # Time symbolic RHS generation
    profiler.start_timer("symbolic_rhs")
    reduced_rhs, remaining_syms, const_syms, rate_syms = sim.get_symbolic_reduced_rhs()
    profiler.end_timer("symbolic_rhs")
    
    # Time derivatives computation
    profiler.start_timer("derivatives")
    # Only compute the derivatives we actually use
    dR_dC, dR_dC_func, dR_dl, dR_dl_func, dR_dk, dR_dk_func, remaining_syms = sim.get_first_order_derivatives()
    # Store the derivative functions that will be reused in adaptive sampling
    precomputed_derivatives = (dR_dC_func, dR_dl_func)

    profiler.end_timer("derivatives")
    
    # Time rates creation
    profiler.start_timer("rates_creation")
    rates = np.array([r_n.reactions[r_idx][2] for r_idx in range(len(r_n.reactions))])
    profiler.end_timer("rates_creation")
    
    # Time flexible RHS creation
    profiler.start_timer("flexible_rhs")
    flexible_reduced_ode_rhs = sim.make_reduced_rhs_with_conservation_flexible()
    profiler.end_timer("flexible_rhs")
 
    # Use adaptive sampling instead of fixed sampling
    # Note: The adaptive sampler handles its own internal timing with "adaptive_" prefixed timers
    profiler.start_timer("adaptive_sampling")
    sign_conditions, C_full_list, dC_dl_list, l0_list, sample_count, convergence_reached = sampler.sample_sign_conditions(
        sim=sim,
        L=L,
        t_span=t_span,
        num_points=num_points,
        int_method=int_method,
        r_tol=r_tol,
        a_tol=a_tol,
        precomputed_derivatives=precomputed_derivatives
    )
    profiler.end_timer("adaptive_sampling")
    
    # Reset sampler for next network
    sampler.reset()
    
    # Log the network data using the new logger
    profiler.start_timer("data_logging")
    data_logger.log_network(
        r_n=r_n,
        interaction_matrix=interaction_matrix,
        cycles=cycles,
        sign_conditions=sign_conditions,
        C_full_list=C_full_list,
        dC_dl_list=dC_dl_list,
        l0_list=l0_list,
        iteration=iter,
        seed=seed,
        E_range=E_range,
        B_range=B_range,
        F_range=F_range,
        C0=C0,
        beta=beta,
        sample_count=sample_count,  # Add sampling statistics
        convergence_reached=convergence_reached
    )
    profiler.end_timer("data_logging")

    if iter%5 == 0:
        print(f"Iteration {iter} complete")

profiler.end_total_timer()

# Print detailed timing information
profiler.print_summary()

# Save data using the logger
data_logger.save_data(output_dir + "/SavedData.pkl")
