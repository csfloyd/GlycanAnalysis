from CRNs import *
from CRNs.training import (
    GraphComputation, CRNModel,
    convert_input_data_to_log_scale,
)
from CRNs.mi_training import MITrainer, run_training_mi
from CRNs.mi_estimators import create_mi_estimator
from CRNs.regularizers import _group_substrate_forms
import numpy as np
import argparse
import time
import copy
import pickle
import random


# Create argument parser
parser = argparse.ArgumentParser(description="MI training job script with arguments.")

# Define command-line arguments
parser.add_argument("--param1", type=int, required=True, help="Learning rate for MI training")
parser.add_argument("--param2", type=int, required=True, help="Hidden dimension")
parser.add_argument("--param3", type=int, required=True, help="Network seed")
parser.add_argument("--output", type=str, required=True, help="Output directory")

# Parse arguments
args = parser.parse_args()

output_dir = args.output

# Separate seeds for data generation and network initialization
data_seed = 10  # Fixed data seed for consistency
network_seed = args.param3

print(f"Data seed: {data_seed}, Network seed: {network_seed}")

# ============== HARDCODED PARAMETERS ==============
# Data generation
clb = args.param1
n_classes = clb**2
n_modes = n_classes
input_dim = 2
lattice_space = 2
center_variance = (input_dim * lattice_space**2)
log_variance = 0.15
proj_dim = input_dim
NR = proj_dim
data_gen_method = 'lattice'  # 'lattice', 'random', or 'cartesian'
log_scale = True

# Network architecture
hidden_dim = args.param2
readout_type = 'biochemical'  # 'biochemical' or 'linear'
p_f = 1.0
p_r = 0.0

# MI Training parameters
mi_lr_rate = 0.05
mi_num_batches = 500
mi_batch_size = 32
optimizer_type = 'adam'  # 'adam', 'sgd', or 'fletcher_reeves'
mi_method = 'analytical'  # 'analytical', 'mc', or 'mean_field'
use_graph_forward = True  # True = fast graph computation, False = ODE integration
log_input_space = False  # Optimize I(ln(R); z) for better gradient scaling
print_every = 250

print("="*80)
print("MUTUAL INFORMATION TRAINING SETUP")
print("="*80)
print(f"Data: {n_classes} classes, {input_dim}D input, {data_gen_method} method")
print(f"Network: {hidden_dim} hidden nodes, {readout_type} readout")
print(f"Training: {mi_num_batches} batches × {mi_batch_size} samples/batch")
print(f"Learning rate: {mi_lr_rate}")
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
    )

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

input_data = InputData(n_classes, data_list)

if not log_scale:
    input_data = convert_input_data_to_log_scale(input_data)

print(f"✓ Generated {n_classes} classes with 10000 samples each")

#################################################
############### Network generation ##############
#################################################

# Reset seed for network initialization
np.random.seed(network_seed)
random.seed(network_seed)

print("\n[2/6] Generating network...")

NS_vec = [hidden_dim, n_classes]
NS = sum(NS_vec)
input_nodes = [f'R{i}' for i in range(NR)]
target_nodes = [f'S{NS - n_classes + i}' for i in range(n_classes)]

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

# Determine code dimension from model architecture
if readout_type == 'biochemical':
    # Need a forward pass to get state
    dummy_R = input_data.get_next_training_sample(0)
    _ = crn_model.forward(dummy_R)
    state = crn_model.get_regularization_state()
    substrate_groups = _group_substrate_forms(
        state['species_names'],
        state['hidden_indices']
    )
    code_dim = len(substrate_groups)
elif readout_type == 'linear':
    code_dim = len(crn_model._hidden_groups)
else:
    raise ValueError(f"Unknown readout_type: {readout_type}")

print(f"Code dimension: {code_dim} bits (2^{code_dim} = {2**code_dim} possible codes)")

# Create MI estimator (auto-selects best method based on code_dim)
mi_estimator = create_mi_estimator(code_dim=code_dim, method=mi_method)
estimator_name = type(mi_estimator).__name__
print(f"MI estimator: {estimator_name}")

if code_dim <= 15:
    print(f"  → Using analytical enumeration (exact)")
else:
    print(f"  → Using mean-field approximation (code_dim > 15)")

# Create learning rate dict
lr_dict = {'log_rates': mi_lr_rate, 'log_l0': mi_lr_rate}
if readout_type == 'linear':
    lr_dict.update({'W': mi_lr_rate, 'b': mi_lr_rate})

# Create MI trainer
mi_trainer = MITrainer(
    model=crn_model,
    mi_estimator=mi_estimator,
    optimizer_type=optimizer_type,
    lr_dict=lr_dict,
    max_grad_norm=50.0,
    frozen_params=['log_l0'],  # Don't update conservation constants
    log_input_space=log_input_space
)

opt_mode = "log-input-space" if log_input_space else "standard"
print(f"✓ Trainer ready: {optimizer_type} optimizer, lr={mi_lr_rate}, mode={opt_mode}")

###############################################
############### Run MI training ###############
###############################################

print("\n[5/6] Running MI training...")
print("="*80)

start_time = time.time()

mi_history = run_training_mi(
    trainer=mi_trainer,
    input_data=input_data,
    num_batches=mi_num_batches,
    batch_size=mi_batch_size,
    print_every=print_every,
    verbose=True
)

training_time = time.time() - start_time

print("="*80)
print(f"✓ Training complete!")
print(f"Final I(R;z) = {mi_history.mi_values[-1]:.4f} nats = {mi_history.mi_values[-1]/np.log(2):.4f} bits")
print(f"Average I(R;z) (last 50 batches) = {np.mean(mi_history.mi_values[-50:]):.4f} nats")
print(f"Training time: {training_time:.1f}s")

###############################################
############### Analyze results ###############
###############################################

print("\n[6/6] Analyzing results...")

# Sample h from trained model
test_h_samples = []
for class_idx in range(n_classes):
    for _ in range(50):
        R = input_data.get_next_training_sample(class_idx)
        h = mi_trainer.extract_activation_parameters(R)
        test_h_samples.append(h)
test_h_samples = np.array(test_h_samples)

# Compute information-theoretic quantities
H_z_given_R = np.mean([
    mi_estimator.bernoulli_entropy(h).sum() 
    for h in test_h_samples
])
H_z = mi_estimator.estimate(test_h_samples[:mi_batch_size])
I_R_z = H_z - H_z_given_R

print(f"\nInformation decomposition:")
print(f"  H(z|R) = {H_z_given_R:.4f} nats (conditional entropy)")
print(f"  H(z)   = {H_z:.4f} nats (marginal entropy)")
print(f"  I(R;z) = {I_R_z:.4f} nats = {I_R_z/np.log(2):.4f} bits")

print(f"\nActivation statistics:")
print(f"  h mean: {test_h_samples.mean():.3f} ± {test_h_samples.mean(axis=0).std():.3f}")
print(f"  h std:  {test_h_samples.std(axis=0).mean():.3f}")
print(f"  h range: [{test_h_samples.min():.3f}, {test_h_samples.max():.3f}]")

# Compare with random baseline
random_h = np.random.rand(mi_batch_size * 10, code_dim)
random_mi = mi_estimator.estimate(random_h)
print(f"\nComparison with random encoding:")
print(f"  Random:  I(R;z) ≈ {random_mi:.4f} nats = {random_mi/np.log(2):.4f} bits")
print(f"  Learned: I(R;z) = {I_R_z:.4f} nats = {I_R_z/np.log(2):.4f} bits")
print(f"  Improvement: {I_R_z - random_mi:+.4f} nats ({(I_R_z/random_mi - 1)*100:+.1f}%)")

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
    'n_classes': n_classes,
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
    
    # MI Training history
    'mi_history': {
        'mi_values': mi_history.mi_values,
        'h_statistics': mi_history.h_statistics,
        'ipr_values': mi_history.ipr_values,
        'grad_norm_history': mi_history.grad_norm_history,
        'loss_history': mi_history.loss_history,  # Should be zeros for MI
    },
    
    # Final results
    'final_mi_nats': I_R_z,
    'final_mi_bits': I_R_z / np.log(2),
    'final_H_z': H_z,
    'final_H_z_given_R': H_z_given_R,
    'random_baseline_mi': random_mi,
    'test_h_samples': test_h_samples,
    'training_time_seconds': training_time,
    
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
print(f"Final I(R;z): {I_R_z:.4f} nats = {I_R_z/np.log(2):.4f} bits")
print(f"Improvement over random: {(I_R_z/random_mi - 1)*100:+.1f}%")
print(f"Code dimension: {code_dim} bits")
print(f"Hidden dimension: {hidden_dim}")
print(f"Learning rate: {mi_lr_rate}")
print(f"Training time: {training_time:.1f}s")
print("="*80)
