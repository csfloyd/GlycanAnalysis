from CRNs import *
from CRNs.training import (
    GraphComputation, CRNModel,
    convert_input_data_to_log_scale,
)
from CRNs.mi_training import MITrainer
from CRNs.mi_estimators import create_mi_estimator
from CRNs.regularizers import compute_code_dim_for_indices
import numpy as np
import argparse
import time
import copy
import pickle
import random


# Create argument parser
parser = argparse.ArgumentParser(description="MI training job script with arguments.")

# Define command-line arguments
arg1_type = float
arg2_type = float
arg3_type = int
parser.add_argument("--param1", type=arg1_type, required=True, help="Learning rate for MI training")
parser.add_argument("--param2", type=arg2_type, required=True, help="Hidden dimension")
parser.add_argument("--param3", type=arg3_type, required=False, help="Network seed")
parser.add_argument("--output", type=str, required=True, help="Output directory")
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

# ============== HARDCODED PARAMETERS ==============
# Data generation
clb = 5
n_outputs = 5              # network output-layer size == MI code dimension
n_modes = clb**2
#n_modes = 256
#n_modes = args.param2
n_data_classes = n_modes   # data sampling buckets; labels are irrelevant to I(R;C)
input_dim = 2 
lattice_space = 1.0
center_variance = (input_dim * lattice_space**2)
input_noise_scale = args.param2
log_variance = input_noise_scale ** 2
proj_dim = input_dim
NR = proj_dim
data_gen_method = 'lattice'  # 'lattice', 'random', or 'cartesian' ##CHANGE
log_scale = True
readout_type = 'biochemical'  # 'biochemical' or 'linear'
p_mode_keep = 1.0          # Fraction of lattice modes to keep (1.0=all, 0.5=half randomly deleted)

# Network architecture
hidden_dim = 3 #3 #args.param2
p_f = args.param1 #1.0
p_r = 0.0

NS_vec = [hidden_dim, n_outputs]
NS = sum(NS_vec)
input_nodes = [f'R{i}' for i in range(NR)]
target_nodes = [f'S{NS - n_outputs + i}' for i in range(n_outputs)]

# MI Training parameters (LOG-NORMAL CHANNEL, hardcoded)
mi_lr_rate = 0.05
mi_num_batches = 1000
mi_batch_size = 32
optimizer_type = 'adam'  # 'adam', 'sgd', or 'fletcher_reeves'
mi_method = 'lognormal'  # log-normal channel R -> C (hardcoded)
use_graph_forward = True  # True = fast graph computation, False = ODE integration
log_input_space = False  # Optimize I(ln(R); C) for better gradient scaling

# Which layer to compute MI over: 'hidden', 'output', or a list of species names/indices
mi_layer = target_nodes

# Log-normal channel hyperparameters (fixed noise, not optimized)
mi_mc_seed = data_seed
output_noise_scale = 0.1           # scalar or array (code_dim,); fixed log-space noise std
noise_scale = output_noise_scale
lognormal_mc_samples = 256   # MC draws S per mixture component
resample_every = 0          # 0 => freeze common random numbers (recommended)

# Evaluation diagnostics (held-out set for a clean MI curve)
eval_batch_size = lognormal_mc_samples
eval_every = 50

print("="*80)
print("MUTUAL INFORMATION TRAINING SETUP")
print("="*80)
print(f"Data: {n_data_classes} classes, {input_dim}D input, {data_gen_method} method")
print(f"Output layer: {n_outputs} nodes, {n_modes} data modes")
print(f"Mode keep fraction: {p_mode_keep} (lattice mode deletion)")
print(f"Network: {hidden_dim} hidden nodes, {readout_type} readout")
print(f"Training: {mi_num_batches} batches × {mi_batch_size} samples/batch")
print(f"Learning rate: {mi_lr_rate}")
print(f"MI layer: {mi_layer}")
print(f"Seeds: data={data_seed}, network={network_seed}")
print("="*80)

#################################################
############### Data generation #################
#################################################

# Set seed for data generation
np.random.seed(data_seed)
random.seed(data_seed)

print("\n[1/6] Generating data...")

if data_gen_method == 'cartesian':
    alphabet = None
    psi = None
    mode_labels = None
    
    Q = 2   # alphabet size
    L = 3   # number of vectors to concatenate
    N = 4   # length of each vector
    M = 4   # random subset size per block
    
    input_dim = L * N
    proj_dim = input_dim
    NR = proj_dim
    
    n_data_classes, data_list, log_means, alphabet, psi, mode_labels = generate_cartesian_alphabet_data(
        Q=Q, L=L, N=N, C=n_data_classes, M=M,
        n_samples_per_class=10000,
        log_variance=log_variance,
        center_variance=center_variance,
        random_state=data_seed,
    )

elif data_gen_method == 'lattice':
    n_data_classes, data_list, log_means = generate_lognormal_mixture_lattice_centers(
        n_classes=n_data_classes,
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
    n_data_classes, data_list, log_means = generate_lognormal_mixture_random_centers(
        n_classes=n_data_classes,
        n_modes=n_modes,
        n_samples_per_class=10000,
        input_dim=input_dim,
        center_variance=center_variance,
        log_variance=log_variance,
        center_offset=0.0,
        random_state=data_seed,
    )

input_data = InputData(n_data_classes, data_list)

if not log_scale:
    input_data = convert_input_data_to_log_scale(input_data)

print(f"✓ Generated {n_data_classes} classes with 10000 samples each")
if data_gen_method == 'lattice' and p_mode_keep < 1.0:
    n_modes_original = int(np.round(n_modes / p_mode_keep))
    print(f"  Randomly kept {n_modes}/{n_modes_original} lattice modes (p_mode_keep={p_mode_keep})")

#################################################
############### Network generation ##############
#################################################

# Reset seed for network initialization
np.random.seed(network_seed)
random.seed(network_seed)

print("\n[2/6] Generating network...")


species_names, reaction_strings, L, adjacency_matrix, input_substrates_list = \
    generate_layered_feedforward_signaling_network(
        NR, NS_vec, p_r, p_f=p_f, 
        include_reverse=False, include_uncatalyzed=True
    )

target_node_idxs = [species_names.index(node) for node in target_nodes]

G = get_digraph_from_adjacency_matrix(adjacency_matrix, input_substrates_list, NR, NS)
n_paths = count_simple_paths(G, f'R0', f'S{NS-1}')

print(f"✓ Created network: {len(reaction_strings)} reactions, {n_paths} paths")

################################################
############### Create CRN model ###############
################################################

print("\n[3/6] Initializing CRN model...")

r_n = ReactionNetwork.from_reaction_strings(
    reaction_strings=reaction_strings,
    L=L,
    seed=network_seed,
    species_names=species_names,
    force_reverse=True,
)

sim = ReactionNetworkSimulator(r_n)
symbolic_rhs, species, rates = sim.get_symbolic_rhs()
reduced_rhs, remaining_syms, const_syms, rate_syms = sim.get_symbolic_reduced_rhs()
dR_dC, dR_dC_func, dR_dl, dR_dl_func, dR_dk, dR_dk_func, remaining_syms = \
    sim.get_first_order_derivatives()
sim.solve_conservation_laws()

# Initialize rates
n_rates = len(r_n.get_rates())
rates = np.exp(np.random.randn(n_rates) * 1.0)
r_n.update_rates(rates)

# Graph computation setup
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
target_node_idxs = [r_n.species_names.index(node) for node in target_nodes]
n_inputs = len(input_nodes)

# Create CRN model
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
)

print(f"✓ Model created: {readout_type} readout, {forward_method} forward pass")

###############################################
############### Setup MI training #############
###############################################

print("\n[4/6] Setting up MI training...")

dummy_R = input_data.get_next_training_sample(0)
_ = crn_model.forward(dummy_R)
state = crn_model.get_regularization_state()

# Choose which layer to compute MI over
if isinstance(mi_layer, list):
    layer_indices = []
    layer_names = []
    for item in mi_layer:
        if isinstance(item, str):
            idx = state['species_names'].index(item)
            layer_indices.append(idx)
            layer_names.append(item)
        else:
            layer_indices.append(item)
            layer_names.append(state['species_names'][item])
    layer_description = f"custom layer ({len(layer_indices)} species: {layer_names})"
elif mi_layer == 'hidden':
    layer_indices = state['hidden_indices']
    layer_description = f"hidden layer ({len(layer_indices)} species)"
elif mi_layer == 'output':
    layer_indices = state['class_ids']
    layer_description = f"output layer ({len(layer_indices)} species)"
else:
    raise ValueError(
        f"mi_layer must be 'hidden', 'output', or a list of species names/indices. Got: {mi_layer}"
    )

if readout_type == 'biochemical':
    code_dim = compute_code_dim_for_indices(state['species_names'], layer_indices)
elif readout_type == 'linear':
    code_dim = len(crn_model._hidden_groups)
else:
    raise ValueError(f"Unknown readout_type: {readout_type}")

print(f"Code dimension: {code_dim} ({layer_description})")

# Expand scalar noise to a per-dimension vector if needed
if np.size(noise_scale) == 1:
    noise_scale_vec = np.full(code_dim, float(noise_scale))
else:
    noise_scale_vec = np.asarray(noise_scale, dtype=float)
    if noise_scale_vec.shape != (code_dim,):
        raise ValueError(f"noise_scale must be scalar or shape ({code_dim},)")

# Create log-normal channel MI estimator
mi_estimator = create_mi_estimator(
    code_dim=code_dim,
    method=mi_method,
    noise_scale=noise_scale_vec,
    lognormal_mc_samples=lognormal_mc_samples,
    resample_every=resample_every,
    seed=mi_mc_seed,
)
estimator_name = type(mi_estimator).__name__
print(f"MI estimator: {estimator_name} (sigma mean={noise_scale_vec.mean():.3g})")

# Create learning rate dict
lr_dict = {'log_rates': mi_lr_rate, 'log_l0': mi_lr_rate}
if readout_type == 'linear':
    lr_dict.update({'W': mi_lr_rate, 'b': mi_lr_rate})

# Create MI trainer
mi_trainer = MITrainer(
    model=crn_model,
    mi_estimator=mi_estimator,
    layer_indices=layer_indices,
    optimizer_type=optimizer_type,
    lr_dict=lr_dict,
    max_grad_norm=50.0,
    frozen_params=['log_l0'],  # Don't update conservation constants
    log_input_space=log_input_space
)

opt_mode = "log-input-space" if log_input_space else "standard"
assert mi_trainer.channel == 'lognormal', "Estimator is not the log-normal channel!"
print(f"✓ Trainer ready: channel={mi_trainer.channel}, {optimizer_type} optimizer, "
      f"lr={mi_lr_rate}, mode={opt_mode}, layer={layer_description}")

###############################################
############### Run MI training ###############
###############################################

print("\n[5/6] Running MI training (maximize I(R;C))...")
print("="*80)


def sample_R(n):
    return [input_data.get_next_training_sample(random.randrange(n_data_classes)) for _ in range(n)]


def codes_for(R_list):
    return np.array([
        mi_trainer.extract_code_from_indices(R, layer_indices) for R in R_list
    ])


# Fixed held-out set for a clean, comparable MI curve (frozen eps in estimator).
eval_R = sample_R(eval_batch_size)

hist = {'train_mi': [], 'grad_norm_history': [], 'cbar_mean': [], 'cbar_std': [],
        'eval_batch': [], 'eval_mi': [], 'eval_Hc': [], 'eval_HcR': []}

start_time = time.time()

init_eval = mi_estimator.entropy_breakdown(codes_for(eval_R))
print(f"Initial (untrained) held-out I(R;C) = {init_eval['mutual_information']:.4f} nats")

for batch in range(mi_num_batches):
    R_batch = sample_R(mi_batch_size)
    try:
        mi_value, diag = mi_trainer.train_step_batch(R_batch, compute_diagnostics=True)
    except Exception as e:
        print(f"  batch {batch}: step failed ({e})")
        continue
    if np.isnan(mi_value) or np.isinf(mi_value):
        continue

    hist['train_mi'].append(mi_value)
    hist['grad_norm_history'].append(diag['grad_norms'])
    hist['cbar_mean'].append(diag['h_mean'].mean())   # 'h_*' slots now hold Cbar stats
    hist['cbar_std'].append(diag['h_std'].mean())

    if batch % eval_every == 0 or batch == mi_num_batches - 1:
        ev = mi_estimator.entropy_breakdown(codes_for(eval_R))
        hist['eval_batch'].append(batch)
        hist['eval_mi'].append(ev['mutual_information'])
        hist['eval_Hc'].append(ev['H_marginal'])
        hist['eval_HcR'].append(ev['H_conditional'])
        gnorm = diag['grad_norms'].get('log_rates', np.nan)
        print(f"Batch {batch:4d}/{mi_num_batches} | "
              f"train I(R;C)={mi_value:.4f} | held-out I(R;C)={ev['mutual_information']:.4f} | "
              f"H(C)={ev['H_marginal']:.3f} H(C|R)={ev['H_conditional']:.3f} | "
              f"grad={gnorm:.2e}")

training_time = time.time() - start_time

final_eval = mi_estimator.entropy_breakdown(codes_for(eval_R))
print("="*80)
print(f"✓ Training complete!")
print(f"Held-out I(R;C): initial={init_eval['mutual_information']:.4f} -> "
      f"final={final_eval['mutual_information']:.4f} nats "
      f"({final_eval['mutual_information']/np.log(2):.4f} bits)")
print(f"Training time: {training_time:.1f}s")

###############################################
############### Analyze results ###############
###############################################

print("\n[6/6] Analyzing results...")

# Sample mean codes (Cbar) from the trained model across classes
test_cbar_samples = []
for class_idx in range(n_data_classes):
    for _ in range(50):
        R = input_data.get_next_training_sample(class_idx)
        test_cbar_samples.append(
            mi_trainer.extract_code_from_indices(R, layer_indices)
        )
test_cbar_samples = np.array(test_cbar_samples)

# Information decomposition on the fixed held-out set
I_R_C = final_eval['mutual_information']
H_C = final_eval['H_marginal']
H_C_given_R = final_eval['H_conditional']

print(f"\nInformation decomposition (log-normal channel):")
print(f"  H(C|R) = {H_C_given_R:.4f} nats (conditional entropy)")
print(f"  H(C)   = {H_C:.4f} nats (marginal entropy)")
print(f"  I(R;C) = {I_R_C:.4f} nats = {I_R_C/np.log(2):.4f} bits")

print(f"\nCbar statistics ({mi_layer} layer):")
print(f"  mean={test_cbar_samples.mean():.3g}  std(dim-mean)={test_cbar_samples.mean(0).std():.3g}")
print(f"  per-dim std (avg)={test_cbar_samples.std(0).mean():.3g}")
print(f"  range=[{test_cbar_samples.min():.3g}, {test_cbar_samples.max():.3g}]")

###############################################
############### Save everything ###############
###############################################

print("\nSaving results...")

save_data = {
    # Model parameters (learned)
    'model_params': mi_trainer.model.get_params(),
    
    # Network structure
    'reaction_strings': reaction_strings,
    'species_names': species_names,
    'L': L,
    'adjacency_matrix': adjacency_matrix,
    'input_substrates_list': input_substrates_list,
    
    # Network configuration
    'data_seed': data_seed,
    'network_seed': network_seed,
    'n_classes': n_data_classes,   # data class count (what the analysis code reconstructs from)
    'n_outputs': n_outputs,        # network output-layer size (MI code dimension)
    'NR': NR,
    'NS_vec': NS_vec,
    'NS': NS,
    'p_r': p_r,
    'p_f': p_f,
    'hidden_dim': hidden_dim,
    'input_nodes': input_nodes,
    'target_nodes': target_nodes,
    'target_node_idxs': target_node_idxs,
    'n_inputs': n_inputs,
    'default_l0': default_l0,
    'code_dim': code_dim,
    'mi_layer': mi_layer,
    'layer_indices': layer_indices,
    'layer_description': layer_description,
    
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
    
    # MI Training hyperparameters
    'mi_lr_rate': mi_lr_rate,
    'mi_num_batches': mi_num_batches,
    'mi_batch_size': mi_batch_size,
    'optimizer_type': optimizer_type,
    'mi_method': mi_method,
    'readout_type': readout_type,
    'lr_dict': lr_dict,
    'max_grad_norm': 50.0,
    'frozen_params': ['log_l0'],
    'log_input_space': log_input_space,
    'use_graph_forward': use_graph_forward,
    'forward_method': forward_method,
    
    # MI Training history (log-normal channel)
    # Note: loss_by_class intentionally excluded to save storage
    'mi_history': {
        'train_mi': hist['train_mi'],
        'eval_batch': hist['eval_batch'],
        'eval_mi': hist['eval_mi'],
        'eval_Hc': hist['eval_Hc'],
        'eval_HcR': hist['eval_HcR'],
        'cbar_mean': hist['cbar_mean'],
        'cbar_std': hist['cbar_std'],
        'grad_norm_history': hist['grad_norm_history'],
    },
    
    # Final results
    'final_mi_nats': I_R_C,
    'final_mi_bits': I_R_C / np.log(2),
    'final_H_C': H_C,
    'final_H_C_given_R': H_C_given_R,
    'initial_mi_nats': init_eval['mutual_information'],
    'test_cbar_samples': test_cbar_samples,
    'eval_R': eval_R,
    'training_time_seconds': training_time,
    
    # Log-normal channel config
    'noise_scale': noise_scale_vec,
    'lognormal_mc_samples': lognormal_mc_samples,
    'resample_every': resample_every,
    'mi_mc_seed': mi_mc_seed,
    'eval_batch_size': eval_batch_size,
    'eval_every': eval_every,
    
    # MI estimator info
    'mi_estimator_name': estimator_name,
    'mi_estimator_type': mi_method,
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

with open(f'{output_dir}/mi_training_results.pkl', 'wb') as f:
    pickle.dump(save_data, f)

print(f"✓ Results saved to {output_dir}/mi_training_results.pkl")

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Final I(R;C): {I_R_C:.4f} nats = {I_R_C/np.log(2):.4f} bits")
print(f"Gain over untrained: {I_R_C - init_eval['mutual_information']:+.4f} nats")
print(f"MI layer: {layer_description}")
print(f"Code dimension: {code_dim}")
print(f"Hidden dimension: {hidden_dim}")
print(f"Learning rate: {mi_lr_rate}")
print(f"Fixed noise sigma (mean): {noise_scale_vec.mean():.3g}")
print(f"Training time: {training_time:.1f}s")
print("="*80)
