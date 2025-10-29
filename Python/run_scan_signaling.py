from CRNs import *
from CRNs.utils import NetworkDataLogger
import numpy as np
import argparse
import time
import copy
import pickle
import ast


# Create argument parser
parser = argparse.ArgumentParser(description="SLURM job script with arguments.")

arg1_type = str
# Define command-line arguments
parser.add_argument("--param1", type=arg1_type, required=True, help="An integer parameter")
parser.add_argument("--param2", type=int, required=False, help="An integer parameter")
parser.add_argument("--output", type=str, required=True, help="A string parameter")

# Parse arguments
args = parser.parse_args()

output_dir = args.output

#seed = args.param2
seed = 40

### Signaling network 
NR = 1
# NS = 3
# p_f = 0.8
p_r = 0
# Convert string to list of integers
NS_vec = [int(digit) for digit in args.param1]
NS_vec.append(1)
NS = sum(NS_vec)
target_node = 'S'+str(NS-1)

E_range = 8.0
B_range = 8.0
F_range = 8.0
C0 = 1
beta = 1
l0_range = (1e-3, 1e3)


n_graph_samples = 10000

t_span = (0, 10000)
num_points = 10000
int_method = 'LSODA'
r_tol = 1e-12  
a_tol = 1e-12


if seed is not None:
    np.random.seed(seed)
    random.seed(seed)

#species_names, reaction_strings, L, adjacency_matrix, input_substrates = generate_dag_signaling_network(NR, NS, p_f, p_r, include_reverse=True, include_uncatalyzed=True)
species_names, reaction_strings, L, adjacency_matrix, input_substrates_list = generate_layered_feedforward_signaling_network(NR, NS_vec, p_r, include_reverse=False, include_uncatalyzed=True)

G = get_digraph_from_adjacency_matrix(adjacency_matrix, input_substrates_list, NR, NS)
target_node = 'S'+str(NS-1)
target_node_idx = species_names.index(target_node)

print("Number of paths is", len(count_simple_paths(G, 'R0', target_node)))

r_n = ReactionNetwork.from_reaction_strings(
    reaction_strings=reaction_strings,
    L=L,
    seed=seed,
    species_names=species_names,
    force_reverse=True
)
n_species = r_n.n_species
n_complexes = r_n.n_complexes
n_reactions = int(r_n.n_reactions / 2)
n_lcs = r_n.n_lcs
n_cons = len(L)
r_n_base = r_n

lhs_bool = True

if lhs_bool:
    E_lists = [(-E_range, E_range) for _ in range(n_species)]
    B_lists = [(-B_range, B_range) for _ in range(n_reactions)]
    F_lists = [(-F_range, F_range) for _ in range(n_reactions)]
    l0_lists = [(np.log10(l0_range[0]), np.log10(l0_range[1])) for _ in range(n_cons)]
    ranges = E_lists + B_lists + F_lists + l0_lists
    samples = list(latin_hypercube_sampling(ranges, n_graph_samples, seed))


input_dims = [0]
default_l0 = np.ones(NR+NS)
sc_grad_dims = [[NR+NS-1],[0]]


# Initialize data logger and time profiler
data_logger = NetworkDataLogger()
profiler = TimeProfiler()
profiler.start_total_timer()

sampler = GridSampler(
    input_dims=input_dims,
    default_l0=default_l0,
    sc_grad_dims=sc_grad_dims,
    l0_range=l0_range,
    l0_grid_size=50,
    grid_dim=1,
    timeout_seconds=5,
    profiler=profiler,
    round_decimals=6,
    use_signal_alarms=False,
    use_contour_integration=True
)

for iter in range(n_graph_samples):

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)

    # Time network generation
    profiler.start_timer("network_generation")
    try:
        if r_n_base is None:
            r_n = ReactionNetwork(  
                n_species, 
                n_complexes, n_reactions, n_lcs, 
                L, seed, force_reverse=force_reverse, subset_group_ind=subset_group_ind, 
                #complexes_per_class=complexes_per_class, reactions_per_class=reactions_per_class
            )
        else:
            r_n = r_n_base
        if lhs_bool:
            sample = samples[iter]
            E_list = sample[:n_species]
            B_list = sample[n_species:n_species+n_reactions]
            F_list = sample[n_species+n_reactions:n_species+n_reactions+n_reactions]
            reac_rates = get_rates_from_exponents(r_n, E_list, B_list, F_list, C0, beta)
            r_n.update_rates(reac_rates)

            l0_list = 10**np.array(sample[n_species+n_reactions+n_reactions:])
            sampler.default_l0 = default_l0

        else:
            reac_rates = generate_thermodynamic_rates(r_n, C0, beta, E_range, B_range, F_range)
            r_n.update_rates(reac_rates)

            default_l0 = np.exp(np.random.uniform(np.log(l0_range[0]), np.log(l0_range[1]), n_cons))
            sampler.default_l0 = default_l0

        profiler.end_timer("network_generation")
    except Exception as e:
        profiler.end_timer("network_generation")
        print(f"Error creating reaction network: {e}, skipping...")
        continue

    if r_n_base is None or iter == 0:
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


    dC_dl_list_sub = [dC_dl[target_node_idx, input_dims[0]] for dC_dl in dC_dl_list]
    C_full_list_sub = [C_full[target_node_idx] for C_full in C_full_list]

    if signal.alarm(0) == 0:
        sig_bool = True
    else:
        sig_bool = False

    data_logger.log_network(
        r_n=r_n,
        interaction_matrix=interaction_matrix,
        input_substrates_list=input_substrates_list,
        NR=NR,
        NS=NS,
        target_node=target_node,
        cycles=cycles,
        adjacency_matrix=adjacency_matrix,
        C_full_list=C_full_list_sub,
        dC_dl_list=dC_dl_list_sub,
        l0_list=l0_list,
        iteration=iter,
        seed=seed,
        sig_bool=sig_bool
    )
    profiler.end_timer("data_logging")

    if iter%20 == 0:
        print(f"Iteration {iter} complete")

profiler.end_total_timer()

# Print detailed timing information
profiler.print_summary()

# Save data using the logger
data_logger.save_data(output_dir + "/SavedData.pkl")
