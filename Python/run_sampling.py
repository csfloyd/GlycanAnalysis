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

arg1_type = int
# Define command-line arguments
parser.add_argument("--param1", type=int, required=True, help="An integer parameter")
parser.add_argument("--param2", type=int, required=False, help="An integer parameter")
parser.add_argument("--param3", type=int, required=False, help="An integer parameter")
parser.add_argument("--output", type=str, required=True, help="A string parameter")

# Parse arguments
args = parser.parse_args()

output_dir = args.output

seed = args.param3
#seed = 40

if seed is not None:
    np.random.seed(seed)
    random.seed(seed)

sample_network_bool = False

# Recurrences use p_r; MP options use num_P. Internal/external catalysis
# require extra phosphoforms, so expand_to_MP runs if either catalysis flag is on.
p_r = 1.0
add_recurrences = True          # add backward edges that do not create new I/O paths
allow_output_recurrence = True  # if False, skip recurrences sourced at output nodes
expand_to_mp = False             # add phosphoforms S, Ss, ..., up to num_P trailing s's
num_P = 2                        # max phosphorylation level (Sss when num_P=2)
expand_mp_external = False       # copy C+Xs->C+X templates onto higher phosphoforms
expand_mp_internal = False       # add self-catalysis Sis+Siss -> Sis+Si
expand_catalyzed = False         # split catalyzed reactions into elementary complexes
n_outputs = 1                    # class/output nodes at S{NS-n_outputs} .. S{NS-1}

if sample_network_bool:
    ### Signaling network 
    NR = 1
    # NS = 3
    # p_f = 0.8
    # Convert string to list of integers
    if args.param1 == "n":
        var = ""
    else:
        var = args.param1

    #NS_vec = [int(digit) for digit in var]
    #NS_vec.append(1)
    #NS = sum(NS_vec)

    p_f = 1.0
    NS = 5

    # Forward DAG only (p_r=0); recurrences applied below via add_recurrent_connections.
    #species_names, reaction_strings, L, adjacency_matrix, input_substrates_list = generate_dag_signaling_network(NR, NS, p_f, 0.0, include_reverse=False, include_uncatalyzed=True)
    species_names, reaction_strings, L, adjacency_matrix, input_substrates_list = generate_layered_feedforward_signaling_network(NR, NS_vec, p_r, include_reverse=False, include_uncatalyzed=True)

else:
    NR = 1
    NS = 4
    save_path = "/project/svaikunt/csfloyd/TrainingCRNs/Python/SavedNetworks/"
    sub_path = "NS" + str(NS) + ".pkl"  # Replace with your desired path
    filepath = save_path + sub_path
    with open(filepath, 'rb') as f:
        networks = pickle.load(f)

    n_paths = args.param1
    sample_idx = args.param2
    adjacency_matrix = networks[n_paths][sample_idx]['adjacency_matrix_0']
    input_substrates_list = networks[n_paths][sample_idx]['input_substrates_list']

    species_names, reaction_strings, L = generate_specified_dag_signaling_network(adjacency_matrix, input_substrates_list, include_reverse=False, include_uncatalyzed=True)

if add_recurrences:
    reaction_strings, adjacency_matrix = add_recurrent_connections(
        adjacency_matrix, reaction_strings, False, input_substrates_list, NR, NS, p_r,
        seed=seed, n_outputs=n_outputs,
        allow_output_sources=allow_output_recurrence,
    )
else:
    p_r = 0.0

if expand_to_mp or expand_mp_external or expand_mp_internal:
    species_names, reaction_strings, L = expand_to_MP(
        species_names, reaction_strings, L, num_P=num_P
    )
if expand_mp_external:
    species_names, reaction_strings, L = expand_MP_catalysis(
        species_names, reaction_strings, L
    )
if expand_mp_internal:
    species_names, reaction_strings, L = expand_MP_self_catalysis(
        species_names, reaction_strings, L
    )
if expand_catalyzed:
    species_names, reaction_strings, L = expand_catalyzed_reactions(
        species_names, reaction_strings, L
    )

target_node = 'S'+str(NS-1)

E_range = 10.0
B_range = 10.0
F_range = 10.0
C0 = 1
beta = 1
l0_range = (1e-9, 1e9)
l0_range_sub = (1e0, 1e0)

# True: thermodynamic E/B/F rates. False: independent log-normal rates (training-style).
use_thermodynamic_rates = True
rate_log_std = 1.0  # used only when use_thermodynamic_rates is False

n_graph_samples = 10000

t_span = (0, 10000)
num_points = 2  # endpoint only; matches CRNModel ODE training
int_method = 'LSODA'
r_tol = 1e-12
a_tol = 1e-12
timeout_seconds = 5

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

interaction_matrix = get_interaction_matrix(r_n)
cycles = compute_cycles(r_n)

lhs_bool = False

if lhs_bool:
    E_lists = [(-E_range, E_range) for _ in range(n_species)]
    B_lists = [(-B_range, B_range) for _ in range(n_reactions)]
    F_lists = [(-F_range, F_range) for _ in range(n_reactions)]
    l0_lists = [(np.log10(l0_range_sub[0]), np.log10(l0_range_sub[1])) for _ in range(n_cons)]
    ranges = E_lists + B_lists + F_lists + l0_lists
    samples = list(latin_hypercube_sampling(ranges, n_graph_samples, seed))


input_dims = [0]
default_l0 = np.ones(n_cons)
sc_grad_dims = [[target_node_idx], input_dims]

data_logger = NetworkDataLogger()
data_logger.set_shared(
    network_params=NetworkDataLogger.extract_network_params(r_n, include_rates=False),
    interaction_matrix=interaction_matrix,
    input_substrates_list=input_substrates_list,
    NR=NR,
    NS=NS,
    target_node=target_node,
    target_node_idx=target_node_idx,
    cycles=cycles,
    adjacency_matrix=adjacency_matrix,
    seed=seed,
    input_dims=input_dims,
    l0_range=l0_range,
    use_thermodynamic_rates=use_thermodynamic_rates,
    p_r=p_r,
    add_recurrences=add_recurrences,
    allow_output_recurrence=allow_output_recurrence,
    expand_to_mp=expand_to_mp,
    num_P=num_P,
    expand_mp_external=expand_mp_external,
    expand_mp_internal=expand_mp_internal,
    expand_catalyzed=expand_catalyzed,
)

# Initialize time profiler
profiler = TimeProfiler()
profiler.start_total_timer()

sampler = GridSampler(
    input_dims=input_dims,
    default_l0=default_l0,
    sc_grad_dims=sc_grad_dims,
    l0_range=l0_range,
    l0_grid_size=100,
    grid_dim=1,
    timeout_seconds=timeout_seconds,
    profiler=profiler,
    round_decimals=6,
    use_signal_alarms=True,
    use_contour_integration=True
)

for iter in range(n_graph_samples):

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
            default_l0 = 10**np.array(sample[n_species+n_reactions+n_reactions:])

        else:
            if use_thermodynamic_rates:
                reac_rates = generate_thermodynamic_rates(r_n, C0, beta, E_range, B_range, F_range)
            else:
                n_rates = len(r_n.get_rates())
                reac_rates = np.exp(np.random.randn(n_rates) * rate_log_std)
            default_l0 = 10**(np.random.uniform(
                np.log10(l0_range_sub[0]), np.log10(l0_range_sub[1]), n_cons
            ))
        
        r_n.update_rates(reac_rates)
        sampler.default_l0 = default_l0

        profiler.end_timer("network_generation")
    except Exception as e:
        profiler.end_timer("network_generation")
        print(f"Error creating reaction network: {e}, skipping...")
        continue

    if r_n_base is None or iter == 0:
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

    # Per-integrate SIGALRM is handled inside GridSampler (try/finally), same as CRNModel.
    profiler.start_timer("adaptive_sampling")
    timed_out = False
    try:
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
    except TimeoutError:
        timed_out = True
        sign_conditions, C_full_list, dC_dl_list, l0_list, sample_count, convergence_reached = [], [], [], [], 0, False
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
    profiler.end_timer("adaptive_sampling")
    
    # Reset sampler for next network
    sampler.reset()
    
    # Log the network data using the new logger
    profiler.start_timer("data_logging")


    dC_dl_list_sub = np.asarray(
        [dC_dl[target_node_idx, input_dims[0]] for dC_dl in dC_dl_list], dtype=float
    )
    C_full_list_sub = np.asarray(
        [C_full[target_node_idx] for C_full in C_full_list], dtype=float
    )
    l0_scan = np.asarray(
        [l0_vec[input_dims[0]] for l0_vec in l0_list], dtype=float
    )

    sig_bool = timed_out or (len(C_full_list) == 0)

    data_logger.log_network(
        rates=np.asarray(r_n.get_rates(), dtype=float),
        C_full_list=C_full_list_sub,
        dC_dl_list=dC_dl_list_sub,
        l0_list=l0_scan,
        iteration=iter,
        sig_bool=sig_bool,
    )
    profiler.end_timer("data_logging")

    if iter%20 == 0:
        print(f"Iteration {iter} complete")

profiler.end_total_timer()

# Print detailed timing information
profiler.print_summary()

# Save data using the logger
data_logger.save_data(output_dir + "/SavedData.pkl")
