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
arg2_type = float
# Define command-line arguments
parser.add_argument("--param1", type=arg1_type, required=True, help="An integer parameter")
parser.add_argument("--param2", type=arg2_type, required=False, help="An integer parameter")
parser.add_argument("--param3", type=int, required=False, help="An integer parameter")
parser.add_argument("--output", type=str, required=True, help="A string parameter")
# Parse arguments
args = parser.parse_args()

output_dir = args.output

# Separate seeds for data generation and network initialization
seed_1 = args.param3
random.seed(seed_1)
np.random.seed(seed_1)
new_seed = np.random.randint(0, 1000000)

data_seed = new_seed  # Fixed data seed for consistency
network_seed = new_seed

print(f"Data seed: {data_seed}, Network seed: {network_seed}")

data_setup = 'gmm'
if data_setup == 'gmm':
    n_classes = 5
    n_modes = n_classes
    input_dim = 10
    center_noise_scale = 2.0
    center_variance = center_noise_scale ** 2
#    input_noise_scale = np.sqrt(0.5)
    input_noise_scale = 0.5
    log_variance = input_noise_scale ** 2
    proj_dim = args.param1
    NR = proj_dim
    data_gen_method = 'random'  # 'lattice', 'random', or 'cartesian'
    log_scale = True
elif data_setup == 'lattice':
    clb = 3
    n_classes = 3 #clb**2
    n_modes = 9
    input_dim = 2
    lattice_space = 2.0
    center_variance = (input_dim * lattice_space**2)
    input_noise_scale = np.sqrt(0.1)
    log_variance = input_noise_scale ** 2
    proj_dim = input_dim
    NR = proj_dim
    data_gen_method = 'lattice'  # 'lattice', 'random', or 'cartesian'
    log_scale = True

# Set True to train an MLP with the same hidden_dim x hidden_depth.
# MLP inputs stay in log-space (no 10^x / exp concentration scaling).
use_mlp = False
mlp_activation = 'tanh'  # 'tanh', 'relu', 'sigmoid', or 'linear'

readout_type = 'biochemical'  # 'biochemical' or 'linear'
p_mode_keep = 1.0          # Fraction of lattice modes to keep (1.0=all, 0.5=half randomly deleted)

# CRN-only below: ignored when use_mlp is True (graph/ODE, p_r/p_f, recurrences, MP).
use_graph_forward = False  # Set False to use ODE integration instead
atol = 1e-12
rtol = 1e-12
t_span = (0, 10000)
num_points = 2
timeout_seconds = 5  # Skip samples whose ODE integration exceeds this (None/0 disables)

hidden_dim = 4
hidden_depth = 1
#p_r = args.param2
#p_f = args.param1

p_r = 0.0
p_f = args.param2

# Network expansion options (all off = original 2-state feedforward CRN).
# Recurrences use p_r above; MP options use num_P. Internal/external catalysis
# require extra phosphoforms, so expand_to_MP runs if either catalysis flag is on.
# Ignored when use_mlp is True.
add_recurrences = True          # add backward edges that do not create new I/O paths
expand_to_mp = False             # add phosphoforms S, Ss, ..., up to num_P trailing s's
num_P = 2                        # max phosphorylation level (Sss when num_P=2)
expand_mp_external = False       # copy C+Xs->C+X templates onto higher phosphoforms
expand_mp_internal = False       # add self-catalysis Sis+Siss -> Sis+Si
expand_catalyzed = False         # split catalyzed reactions into elementary complexes

NS_vec = [hidden_dim] * hidden_depth + [n_classes]
NS = sum(NS_vec)
input_nodes = [f'R{i}' for i in range(NR)]
target_nodes = [f'S{NS - n_classes + i}' for i in range(n_classes)]


lr_rate = 0.01
readout_lr = lr_rate
reg_weight = 0.0

reg_schedule_type = 'linear_warmup'
reg_schedule_params = {
    'warmup_batches': 500,
}

num_batches=1000
batch_size=32
T_start=0.1
T_end=T_start
T_decay=0.99
noise_start=0.0
noise_end=0.0
noise_decay=0.99
print_every=250

if use_mlp:
    log_scale = False

#################################################
############### Data generation #################
#################################################

# Set seed for data generation
np.random.seed(data_seed)
random.seed(data_seed)

if data_gen_method == 'cartesian':
    alphabet = None
    psi = None
    mode_labels = None

    Q = 2   # alphabet size
    L = 3   # number of vectors to concatenate
    N = 4   # length of each vector
    M = 4   # random subset size per block (M <= Q^N) - end up with M^L modes

    input_dim = L * N
    proj_dim = input_dim
    NR = proj_dim

    n_classes, data_list, log_means, alphabet, psi, mode_labels = generate_cartesian_alphabet_data(
        Q=Q, L=L, N=N, C=n_classes, M=M,
        n_samples_per_class=10000,
        log_variance=log_variance,
        center_variance=center_variance,
        random_state=data_seed,
    )

elif data_gen_method == 'lattice':
    n_classes, data_list, log_means = generate_lognormal_mixture_lattice_centers(
        n_classes=n_classes,
        n_modes=n_modes,
        n_samples_per_class=10000,
        input_dim=input_dim,
        center_variance=center_variance,
        log_variance=log_variance,
        center_offset=0.0,
        random_state=data_seed,
        p_mode_keep=p_mode_keep,
    )
    
    # Update n_modes if modes were deleted
    n_modes = len(log_means)

else:  # 'random'
    n_classes, data_list, log_means = generate_lognormal_mixture_random_centers(
        n_classes=n_classes,
        n_modes=n_modes,
        n_samples_per_class=10000,
        input_dim=input_dim,
        center_variance=center_variance,
        log_variance=log_variance,
        center_offset=0.0,
        random_state=data_seed,
    )

data_list = project_data_list(data_list, d=proj_dim)
input_data = InputData(n_classes, data_list)

if not log_scale:
    input_data = convert_input_data_to_log_scale(input_data)

# Print mode deletion info if applicable
if data_gen_method == 'lattice' and p_mode_keep < 1.0:
    n_modes_original = int(np.round(n_modes / p_mode_keep))
    print(f"Randomly kept {n_modes}/{n_modes_original} lattice modes (p_mode_keep={p_mode_keep})")


#################################################
############### Network generation ##############
#################################################

# Reset seed for network initialization and training
np.random.seed(network_seed)
random.seed(network_seed)

if use_mlp:
    n_inputs = proj_dim
    layer_sizes = [n_inputs] + [hidden_dim] * hidden_depth + [n_classes]
    mlp = SimpleMLP(layer_sizes, activation=mlp_activation)
    model = MLPModel(mlp, n_classes=n_classes)
    print(
        f"Model: MLP {layer_sizes} | activation={mlp_activation} | "
        f"{mlp.get_param_count()} parameters | log_scale={log_scale}"
    )

    regularizers = []
    print(f"Using {len(regularizers)} regularizers:")
    for reg in regularizers:
        print(f"  - {reg.name} (weight={reg.weight})")

    lr_dict = {'params': lr_rate}
    trainer = UnifiedTrainer(
        model,
        optimizer_type='adam',
        lr_dict=lr_dict,
        max_grad_norm=50.0,
        regularizers=regularizers,
        reg_schedule_type=reg_schedule_type,
        reg_schedule_params=reg_schedule_params
    )

    print(f"\nStarting MLP training...")
    history = run_training(
        trainer=trainer,
        input_data=input_data,
        n_classes=n_classes,
        num_batches=num_batches,
        batch_size=batch_size,
        T_start=T_start,
        T_end=T_end,
        T_decay=T_decay,
        noise_start=noise_start,
        noise_end=noise_end,
        noise_decay=noise_decay,
        print_every=print_every
    )

    history.loss_by_class = []

    save_data = {
        'model_type': 'mlp',
        'model_params': trainer.model.get_params(),
        'layer_sizes': layer_sizes,
        'mlp_activation': mlp_activation,
        'n_mlp_params': mlp.get_param_count(),

        'data_seed': data_seed,
        'network_seed': network_seed,
        'n_classes': n_classes,
        'NR': NR,
        'NS_vec': NS_vec,
        'NS': NS,
        'hidden_dim': hidden_dim,
        'hidden_depth': hidden_depth,
        'n_inputs': n_inputs,

        'data_gen_method': data_gen_method,
        'log_scale': log_scale,
        'input_dim': input_dim,
        'proj_dim': proj_dim,
        'n_modes': n_modes,
        'center_variance': center_variance,
        'log_variance': log_variance,
        'center_offset': 0.0,
        'n_samples_per_class': 10000,
        'log_means': log_means,
        'p_mode_keep': p_mode_keep,

        'num_batches': num_batches,
        'batch_size': batch_size,
        'T_start': T_start,
        'T_end': T_end,
        'T_decay': T_decay,
        'noise_start': noise_start,
        'noise_end': noise_end,
        'noise_decay': noise_decay,
        'readout_type': 'mlp',
        'lr_dict': lr_dict,
        'max_grad_norm': 50.0,

        'regularizers': [
            {
                'name': reg.name,
                'weight': reg.weight,
                'class': reg.__class__.__name__,
                'params': {k: v for k, v in reg.__dict__.items() if k not in ['name', 'weight']}
            }
            for reg in regularizers
        ],
        'reg_schedule_type': reg_schedule_type,
        'reg_schedule_params': reg_schedule_params,

        'history': history,
        'n_integration_timeouts': getattr(history, 'n_integration_timeouts', 0),
    }

    if data_gen_method == 'cartesian':
        save_data.update({
            'Q': Q,
            'cart_L': L,
            'N': N,
            'M': M,
            'alphabet': alphabet,
            'psi': psi,
            'mode_labels': mode_labels,
        })

    with open(f'{output_dir}/training_results.pkl', 'wb') as f:
        pickle.dump(save_data, f)

    print(f"\n✓ Training complete!")
    print(f"Final accuracy: {history.get_recent_avg('accuracy'):.1%}")

else:
    # Build a feedforward layered graph first (p_r=0). Recurrences, extra phosphoforms,
    # and elementary expansions are applied below only if their flags are on.
    species_names, reaction_strings, L, adjacency_matrix, input_substrates_list = generate_layered_feedforward_signaling_network(NR, NS_vec, 0.0, p_f = p_f, include_reverse=False, include_uncatalyzed=True)

    if add_recurrences:
        # Backward edges that do not create new simple paths from any R{i} to any class node.
        reaction_strings, adjacency_matrix = add_recurrent_connections(
            adjacency_matrix, reaction_strings, False, input_substrates_list, NR, NS, p_r,
            seed=network_seed, n_outputs=n_classes
        )
    else:
        p_r = 0.0

    # Extra states must exist before internal/external MP catalysis can be added.
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
    # Elementary-complex expansion should come last so it sees the final catalytic set.
    if expand_catalyzed:
        species_names, reaction_strings, L = expand_catalyzed_reactions(
            species_names, reaction_strings, L
        )

    target_node_idxs = [species_names.index(node) for node in target_nodes]

    G = get_digraph_from_adjacency_matrix(adjacency_matrix, input_substrates_list, NR, NS)
    n_paths = count_simple_paths(G, f'R0', f'S{NS-1}')


    ################################################
    ############### Create network #################
    ################################################

    r_n = ReactionNetwork.from_reaction_strings(
        reaction_strings=reaction_strings,
        L=L,
        seed=network_seed,
        species_names=species_names,
        force_reverse=True
    )

    sim = ReactionNetworkSimulator(r_n)
    symbolic_rhs, species, rates = sim.get_symbolic_rhs()
    reduced_rhs, remaining_syms, const_syms, rate_syms = sim.get_symbolic_reduced_rhs()
    dR_dC, dR_dC_func, dR_dl, dR_dl_func, dR_dk, dR_dk_func, remaining_syms = sim.get_first_order_derivatives()
    sim.solve_conservation_laws()
    rates_orig = np.array([r_n.reactions[r_idx][2] for r_idx in range(len(r_n.reactions))])


    ###############################################
    ############### Train network #################
    ###############################################

    n_rates = len(r_n.get_rates())
    rates = np.exp(np.random.randn(n_rates) * 1.0)
    r_n.update_rates(rates)

    # ============== GRAPH COMPUTATION ==============
    if use_graph_forward:
        graph_comp = GraphComputation(G, input_nodes, target_nodes)
        graph_comp.build_r_n_maps(r_n)
        n_nodes = len(graph_comp.nodes)
        forward_method = 'graph'
    else:
        n_nodes = len(L)
        graph_comp = None
        forward_method = 'ode'

    default_l0 = np.ones(n_nodes)
    # Target node indices in species list
    target_node_idxs = [r_n.species_names.index(node) for node in target_nodes]
    n_inputs = len(input_nodes)

    # ============== MODEL ==============
    crn_model = CRNModel(
        r_n=r_n,
        sim=sim,
        L=L,
        class_ids=target_node_idxs,
        n_inputs=n_inputs,
        default_l0=default_l0,
        forward_method=forward_method,
        graph_comp=graph_comp,
        dR_dC_func=dR_dC_func,
        dR_dk_func=dR_dk_func,
        dR_dl_func=dR_dl_func,
        generate_init_func=generate_positive_initial_concentrations_nnls,
        readout_type=readout_type,
        readout_seed=network_seed,
        atol=atol,
        rtol=rtol,
        t_span=t_span,
        num_points=num_points,
        timeout_seconds=timeout_seconds,
    )

    if readout_type == 'linear':
        print(
            f"Readout: {readout_type} | "
            f"h dim={len(crn_model._hidden_groups)}, "
            f"W shape={crn_model._W.shape}"
        )
    else:
        print(
            f"Readout: {readout_type} | "
            f"class logits from species {[species_names[i] for i in target_node_idxs]}"
        )

    # ============== REGULARIZERS ==============
    # regularizers = [
    #     IPRSparsityRegularizer(
    #         weight=reg_weight,
    #         target_ipr=1.0,
    #         penalty_type='above'
    #     ),
    # ]

    regularizers = []

    print(f"Using {len(regularizers)} regularizers:")
    for reg in regularizers:
        print(f"  - {reg.name} (weight={reg.weight})")
        if hasattr(reg, 'target_ipr'):
            print(f"    → target_ipr={reg.target_ipr}, penalty_type={reg.penalty_type}")
        if hasattr(reg, 'sharpness'):
            print(f"    → sharpness={reg.sharpness}")

    # ============== TRAINER ==============
    lr_dict = {'log_rates': lr_rate, 'log_l0': lr_rate}
    if readout_type == 'linear':
        lr_dict.update({'W': readout_lr, 'b': readout_lr})

    trainer = UnifiedTrainer(
        crn_model,
        optimizer_type='adam',
        lr_dict=lr_dict,
        max_grad_norm=50.0,
        frozen_params=['log_l0'],
        regularizers=regularizers,
        reg_schedule_type=reg_schedule_type,
        reg_schedule_params=reg_schedule_params
    )

    # ============== TRAIN ==============
    print(f"\nStarting training with regularization...")
    history = run_training_crn(
        trainer=trainer,
        input_data=input_data,
        n_classes=n_classes,
        num_batches=num_batches,
        batch_size=batch_size,
        T_start=T_start,
        T_end=T_end,
        T_decay=T_decay,
        noise_start=noise_start,
        noise_end=noise_end,
        noise_decay=noise_decay,
        print_every=print_every
    )

    # Clear loss_by_class to save storage (not needed for analysis)
    history.loss_by_class = []

    # ============== SAVE EVERYTHING ==============
    save_data = {
        # Model parameters (learned)
        'model_params': trainer.model.get_params(),
        
        # Network structure
        'reaction_strings': reaction_strings,
        'species_names': species_names,
        'L': L,
        'adjacency_matrix': adjacency_matrix,
        'input_substrates_list': input_substrates_list,
        
        # Network configuration
        'data_seed': data_seed,
        'network_seed': network_seed,
        'n_classes': n_classes,
        'NR': NR,
        'NS_vec': NS_vec,
        'NS': NS,
        'p_r': p_r,
        'p_f': p_f,
        'add_recurrences': add_recurrences,
        'expand_to_mp': expand_to_mp,
        'num_P': num_P,
        'expand_mp_external': expand_mp_external,
        'expand_mp_internal': expand_mp_internal,
        'expand_catalyzed': expand_catalyzed,
        'hidden_dim': hidden_dim,
        'input_nodes': input_nodes,
        'target_nodes': target_nodes,
        'target_node_idxs': target_node_idxs,
        'n_inputs': n_inputs,
        'default_l0': default_l0,
        'forward_method': forward_method,
        'timeout_seconds': timeout_seconds,
        't_span': t_span,
        'num_points': num_points,
        'rtol': rtol,
        'atol': atol,
        'n_integration_timeouts': history.n_integration_timeouts,
        
        # Data generation parameters
        'data_gen_method': data_gen_method,
        'log_scale': log_scale,
        'input_dim': input_dim,
        'proj_dim': proj_dim,
        'n_modes': n_modes,
        'center_variance': center_variance,
        'log_variance': log_variance,
        'center_offset': 0.0,
        'n_samples_per_class': 10000,
        'log_means': log_means,
        'p_mode_keep': p_mode_keep,  # Fraction of modes kept (for lattice method)
        
        # Training hyperparameters
        'num_batches': num_batches,
        'batch_size': batch_size,
        'T_start': T_start,
        'T_end': T_end,
        'T_decay': T_decay,
        'noise_start': noise_start,
        'noise_end': noise_end,
        'noise_decay': noise_decay,
        'readout_type': readout_type,
        'lr_dict': lr_dict,
        'max_grad_norm': 50.0,
        
        # Regularization settings
        'regularizers': [
            {
                'name': reg.name,
                'weight': reg.weight,
                'class': reg.__class__.__name__,
                'params': {k: v for k, v in reg.__dict__.items() if k not in ['name', 'weight']}
            }
            for reg in regularizers
        ],
        'reg_schedule_type': reg_schedule_type,
        'reg_schedule_params': reg_schedule_params,
        
        # Training history
        'history': history,
    }

    if data_gen_method == 'cartesian':
        save_data.update({
            'Q': Q,
            'cart_L': L,
            'N': N,
            'M': M,
            'alphabet': alphabet,
            'psi': psi,
            'mode_labels': mode_labels,
        })

    with open(f'{output_dir}/training_results.pkl', 'wb') as f:
        pickle.dump(save_data, f)

    print(f"\n✓ Training complete!")
    print(f"Final accuracy: {history.get_recent_avg('accuracy'):.1%}")
    print(f"ODE integration timeouts: {history.n_integration_timeouts}")
