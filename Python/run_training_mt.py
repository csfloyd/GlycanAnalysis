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
parser.add_argument("--param2", type=float, required=False, help="An integer parameter")
parser.add_argument("--param3", type=int, required=False, help="An integer parameter")
parser.add_argument("--output", type=str, required=True, help="A string parameter")

# Parse arguments
args = parser.parse_args()

output_dir = args.output


seed = args.param3
n_classes = 5
input_dim = 10
center_variance = 20.0
log_variance = 1
proj_dim = 5
NR = proj_dim
hidden_dim = 5 #args.param2
n_tasks = args.param1  # Number of tasks for multi-task learning


num_batches=5000
batch_size=25
T_start=0.1
T_end=0.1
T_decay=0.99
noise_start=0.0
noise_end=0.0
noise_decay=0.99
print_every=250

# Multi-task l0 parameters
l0_log_mean = 0.0
l0_log_std = 3.0

np.random.seed(seed)
random.seed(seed)


#################################################
############### Network structure ###############
#################################################


NS_vec = [hidden_dim,n_classes]
NS = sum(NS_vec)
input_nodes = [f'R{i}' for i in range(NR)]
target_nodes = [f'S{NS - n_classes + i}' for i in range(n_classes)]

p_r = 0.0
p_f = args.param2
species_names, reaction_strings, L, adjacency_matrix, input_substrates_list = generate_layered_feedforward_signaling_network(NR, NS_vec, p_r, p_f = p_f, include_reverse=False, include_uncatalyzed=True)
target_node_idxs = [species_names.index(node) for node in target_nodes]

G = get_digraph_from_adjacency_matrix(adjacency_matrix, input_substrates_list, NR, NS)
n_paths = count_simple_paths(G, f'R0', f'S{NS-1}')


################################################
############### Create network #################
################################################

r_n = ReactionNetwork.from_reaction_strings(
    reaction_strings=reaction_strings,
    L=L,
    seed=seed,
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
graph_comp = GraphComputation(G, input_nodes, target_nodes)
graph_comp.build_r_n_maps(r_n)

# Build l0 to match graph node ordering
n_nodes = len(graph_comp.nodes)
n_inputs = len(input_nodes)

# Target node indices in species list
target_node_idxs = [r_n.species_names.index(node) for node in target_nodes]

# ============== GENERATE MULTI-TASK DATA ==============
multi_task_data = generate_multitask_data(
    n_tasks=n_tasks,
    n_classes=n_classes,
    n_samples_per_class=10000,
    input_dim=input_dim,
    proj_dim=proj_dim,
    center_variance=center_variance,
    log_variance=log_variance,
    center_offset=0.0,
    n_nodes=n_nodes,
    NR=NR,
    hidden_dim=hidden_dim,
    input_data_class=InputData,
    l0_log_mean=l0_log_mean,
    l0_log_std=l0_log_std,
    base_seed=seed,
    permute_labels_only=True
)

# Use first task's l0 as initial default (will be overwritten during training)
default_l0 = multi_task_data.get_l0(multi_task_data.task_ids[0]).copy()

# ============== MODEL ==============
crn_model = CRNModel(
    r_n=r_n,
    sim=sim,
    L=L,
    class_ids=target_node_idxs,
    n_inputs=n_inputs,
    default_l0=default_l0,
    forward_method='graph',
    graph_comp=graph_comp,
    dR_dC_func=dR_dC_func,
    dR_dk_func=dR_dk_func,
    dR_dl_func=dR_dl_func,
    generate_init_func=generate_positive_initial_concentrations_nnls,
)

# ============== TRAINER (with frozen l0) ==============
trainer = UnifiedTrainer(
    crn_model,
    optimizer_type='adam',
    lr_dict={'log_rates': 0.001},
    max_grad_norm=50.0,
    frozen_params=['log_l0']  # Freeze l0 - it's set per-task
)

# ============== TRAIN MULTI-TASK ==============
history = run_training_crn_multitask(
    trainer=trainer,
    multi_task_data=multi_task_data,
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

# ============== SAVE EVERYTHING ==============
# Extract per-task info for saving
task_data = {}
for task_id in multi_task_data.task_ids:
    task = multi_task_data.get_task(task_id)
    task_data[task_id] = {
        'l0': task['l0'],
        'log_means': task['log_means'],
        'seed': task['seed'],
        'n_classes': task['n_classes'],
    }

save_data = {
    # Model parameters (learned rates only, l0 was frozen)
    'model_params': trainer.model.get_params(),
    
    # Network structure
    'reaction_strings': reaction_strings,
    'species_names': species_names,
    'L': L,
    'adjacency_matrix': adjacency_matrix,
    'input_substrates_list': input_substrates_list,
    
    # Network configuration
    'seed': seed,
    'n_classes': n_classes,
    'NR': NR,
    'NS_vec': NS_vec,
    'NS': NS,
    'p_r': p_r,
    'hidden_dim': hidden_dim,
    'input_nodes': input_nodes,
    'target_nodes': target_nodes,
    'target_node_idxs': target_node_idxs,
    'n_inputs': n_inputs,
    'n_nodes': n_nodes,
    
    # Multi-task configuration
    'n_tasks': n_tasks,
    'task_data': task_data,  # Per-task l0 and log_means
    'l0_log_mean': l0_log_mean,
    'l0_log_std': l0_log_std,
    
    # Data generation parameters
    'input_dim': input_dim,
    'proj_dim': proj_dim,
    'center_variance': center_variance,
    'log_variance': log_variance,
    'center_offset': 0.0,
    'n_samples_per_class': 10000,
    
    # Training hyperparameters
    'num_batches': num_batches,
    'batch_size': batch_size,
    'T_start': T_start,
    'T_end': T_end,
    'T_decay': T_decay,
    'noise_start': noise_start,
    'noise_end': noise_end,
    'noise_decay': noise_decay,
    'lr_dict': {'log_rates': 0.001},
    'max_grad_norm': 50.0,
    'frozen_params': ['log_l0'],
    
    # Training history
    'history': history,
}

with open(f'{output_dir}/training_results.pkl', 'wb') as f:
    pickle.dump(save_data, f)
