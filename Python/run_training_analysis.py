"""
Analyze a single training run (CE or MI) saved by submit_training_jobs / run_training*.

Loads the training pickle from --input, computes per-sample activations and class
separation metrics, and writes separate pickles to --output (same style as run_analysis.py):

  model_info.pkl   - serializable subset of model_info (no live CRN objects)
  activations.pkl  - per-sample activations / IPR
  separation.pkl   - class-separation metrics
  accuracy.pkl     - overall and per-class accuracy on test samples
"""
from CRNs import *
from CRNs.mi_training import MITrainingHistory
from CRNs.mi_estimators import create_mi_estimator
from CRNs.regularizers import (
    compute_activation_fractions,
    compute_fractional_ipr,
    remap_hidden_activations_to_unit_interval,
)
from CRNs.training import (
    GraphComputation,
    CRNModel,
    SimpleMLP,
    MLPModel,
    _assign_modes_to_labels,
    generate_lognormal_mixture,
    project_data_list,
    convert_input_data_to_log_scale,
)
from CRNs.utils import InputData
from CRNs.generation import generate_positive_initial_concentrations_nnls
import argparse
import os
import pickle
import time
import types
import sys
import numpy as np
import networkx as nx



# ============== WHICH OUTPUTS TO WRITE (edit these) ==============
# Same style as run_analysis.py: flip flags here before submitting jobs.
# separation requires computing activations even if write_activations=False.
write_model_info = True
write_activations = True
write_separation = True

param_type1 = float
param_type2 = float
param_type3 = int

MI = False
if MI:
    write_accuracy = True
    layer_indices_default = "output"
    use_raw_concentrations_default = True
else:
    write_accuracy = False
    layer_indices_default = "hidden"
    use_raw_concentrations_default = False


# Defaults for activation sampling (CLI --n_samples_per_mode / --plot_seed override)
n_samples_per_mode_default = 100
plot_seed_default = 0

# Defaults for accuracy computation
n_test_samples_default = 1000

# ============== WHICH LAYER TO EXTRACT (CE + MI share one path) ==============
# Activation extraction always goes through compute_per_sample_mi_activations
# (a superset of the old CE compute_per_sample_activations).
#
# layer_indices_default resolution order inside that helper:
#   1) this setting if not None
#   2) model_info['layer_indices'] from the training pickle (MI runs)
#   3) hidden_indices after forward (matches old CE behavior)
#
# Examples:
#   layer_indices_default = None          # auto (pickle → hidden)
#   layer_indices_default = "hidden"      # force hidden substrates
#   layer_indices_default = "output"      # force target/output nodes
#   layer_indices_default = [12, 13, 14]  # explicit species indices
#
# use_raw_concentrations_default:
#   None  -> auto (True if resolved indices == target_node_idxs)
#   True  -> raw C_full values (typical for MI output / log-normal Cbar)
#   False -> activation fractions (typical for hidden / CE)

# Intrinsic dimensionality estimators.
# Avoid `from skdim import id`: skdim/__init__.py eagerly imports FisherS →
# matplotlib → pyparsing, which breaks on some Midway Python 3.6 installs.
# Register package stubs first so those __init__.py files never run.
SKDIM_AVAILABLE = False
_SKDIM_ESTIMATORS = {}


def _find_skdim_root():
    candidates = []
    try:
        import site
        candidates.append(site.getusersitepackages())
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    candidates.extend(sys.path)
    for base in candidates:
        if not base:
            continue
        root = os.path.join(base, "skdim")
        if os.path.isfile(os.path.join(root, "id", "_MLE.py")):
            return root
    return None


def _register_pkg(name, path):
    """Put a package in sys.modules without executing its __init__.py."""
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    mod.__file__ = os.path.join(path, "__init__.py")
    sys.modules[name] = mod
    return mod


def _load_skdim_estimators():
    root = _find_skdim_root()
    if root is None:
        raise ImportError("skdim package not found on sys.path")

    skdim_mod = _register_pkg("skdim", root)
    id_mod = _register_pkg("skdim.id", os.path.join(root, "id"))
    skdim_mod.id = id_mod

    # These imports no longer execute skdim/__init__.py or id/__init__.py.
    from skdim.id._MLE import MLE
    from skdim.id._TwoNN import TwoNN
    from skdim.id._MOM import MOM
    from skdim.id._TLE import TLE

    return {
        "id_mle": MLE,
        "id_twonn": TwoNN,
        "id_mom": MOM,
        "id_tle": TLE,
    }


# try:
#     _SKDIM_ESTIMATORS = _load_skdim_estimators()
#     SKDIM_AVAILABLE = True
# except Exception as e:
#     print(f"Warning: scikit-dimension ID estimators unavailable ({e}). "
#           "ID estimation will be skipped.")



def _is_mi_results(data):
  """Return True if pickle came from run_training_mi.py."""
  return 'mi_history' in data or 'final_mi_nats' in data


def _is_mlp_results(data):
  """Return True if pickle came from MLP training in run_training.py."""
  if data.get('model_type') == 'mlp':
    return True
  if data.get('model_type') == 'crn':
    return False
  return 'layer_sizes' in data and 'reaction_strings' not in data


def load_mi_history(data):
  """
  Rebuild an MITrainingHistory object from a saved pickle dict.

  run_training_mi.py stores mi_history as a plain dict; this restores
  the object interface used elsewhere in the codebase.
  """
  raw = data.get('mi_history')
  if raw is None:
    return None

  if isinstance(raw, MITrainingHistory):
    return raw

  history = MITrainingHistory(n_classes=data.get('n_classes', 1))
  history.mi_values = list(raw.get('mi_values', []))
  history.h_statistics = list(raw.get('h_statistics', []))
  history.ipr_values = list(raw.get('ipr_values', []))
  history.grad_norm_history = list(raw.get('grad_norm_history', []))
  history.loss_history = list(raw.get('loss_history', []))
  return history


def extract_activation_parameters(crn_model, inputs):
  """
  Extract Bernoulli code parameters h ∈ [0, 1]^d from a loaded CRN or MLP model.

  Mirrors MITrainer.extract_activation_parameters without needing a trainer.
  """
  inputs = np.asarray(inputs)
  if inputs.ndim == 1:
    _ = crn_model.forward(inputs)
  else:
    _ = crn_model.forward(inputs[0])

  if isinstance(crn_model, MLPModel):
    return crn_model.get_penultimate_activations()

  readout_type = crn_model.readout_type

  if readout_type == 'biochemical':
    state = crn_model.get_regularization_state()
    activation_dict = compute_activation_fractions(
      state['C_full'],
      state['species_names'],
      state['hidden_indices'],
    )
    if len(activation_dict) == 0:
      raise RuntimeError("No hidden substrates found for MI extraction")
    return np.array([activation_dict[i] for i in sorted(activation_dict.keys())])

  if readout_type == 'linear':
    return np.clip(crn_model._h.copy(), 0.0, 1.0)

  raise ValueError(f"Unknown readout_type: {readout_type}")


def _softmax_rows(x):
  exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
  return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def _load_trained_mlp_model(data):
  """Rebuild SimpleMLP / MLPModel from an MLP training pickle."""
  layer_sizes = list(data['layer_sizes'])
  mlp_activation = data.get('mlp_activation', 'tanh')
  n_classes = data['n_classes']
  network_seed = data.get('network_seed', data.get('seed', 0))

  np.random.seed(network_seed)
  mlp = SimpleMLP(layer_sizes, activation=mlp_activation)
  mlp_model = MLPModel(mlp, n_classes=n_classes)
  mlp_model.set_params(data['model_params'])

  training_type = 'mi' if _is_mi_results(data) else 'ce'
  print(
    f"MLP settings: layer_sizes={layer_sizes}, activation={mlp_activation}, "
    f"n_params={mlp.get_param_count()}, log_scale={data.get('log_scale', False)}"
  )

  def evaluate_model(input_data_array, return_full_state=False):
    input_data_array = np.atleast_2d(input_data_array)
    predictions = []
    full_states = [] if return_full_state else None
    for i in range(input_data_array.shape[0]):
      output = mlp_model.forward(np.asarray(input_data_array[i], dtype=float).flatten())
      predictions.append(output)
      if return_full_state:
        full_states.append(mlp_model.mlp.activations[-2].copy())

    if not predictions:
      empty = np.zeros((0, mlp_model.n_classes))
      empty_cls = np.array([], dtype=int)
      if return_full_state:
        return empty, empty_cls, empty, np.array([])
      return empty, empty_cls, empty

    predictions = np.array(predictions)
    probabilities = _softmax_rows(predictions)
    predicted_classes = np.argmax(predictions, axis=1)
    if return_full_state:
      return predictions, predicted_classes, probabilities, np.array(full_states)
    return predictions, predicted_classes, probabilities

  evaluate_model.n_integration_timeouts = 0

  def evaluate_codes(input_data_array):
    input_data_array = np.atleast_2d(input_data_array)
    return np.array([
      extract_activation_parameters(mlp_model, input_data_array[i])
      for i in range(input_data_array.shape[0])
    ])

  if training_type == 'mi':
    training_history = load_mi_history(data)
  else:
    training_history = data.get('history', None)

  model_info = {
    'crn_model': mlp_model,
    'r_n': None,
    'n_classes': n_classes,
    'proj_dim': data['proj_dim'],
    'input_dim': data['input_dim'],
    'seed': data.get('data_seed', data.get('seed', 0)),
    'network_seed': network_seed,
    'target_nodes': data.get('target_nodes'),
    'training_type': training_type,
    'training_history': training_history,
    'readout_type': data.get('readout_type', 'mlp'),
    'adjacency_matrix': None,
    'input_substrates_list': None,
    'NR': data.get('NR', data['input_dim']),
    'NS': data.get('NS'),
    'species_names': None,
    'data': data,
    'evaluate_codes': evaluate_codes,
    'mi_estimator': None,
    'layer_indices': data.get('layer_indices', None),
    'model_type': 'mlp',
    'layer_sizes': layer_sizes,
    'mlp_activation': mlp_activation,
  }
  return evaluate_model, model_info


def load_trained_crn_model(filepath):
  """
  Load a trained CRN or MLP model from a pickle file and return evaluation helpers.

  Works with both:
    - training_results.pkl      (classification training, CRN or MLP)
    - mi_training_results.pkl   (MI training)

  Returns
  -------
  evaluate_model : callable
      (inputs) -> predictions, predicted_classes, probabilities
      or with return_full_state=True -> ..., full_states
  model_info : dict
      Metadata, loaded model, and training/MI history
  """
  with open(filepath, "rb") as f:
    data = pickle.load(f)

  if _is_mlp_results(data):
    return _load_trained_mlp_model(data)

  training_type = 'mi' if _is_mi_results(data) else 'ce'
  readout_type = data.get('readout_type', 'biochemical')
  readout_seed = data.get(
    'readout_seed',
    data.get('network_seed', 0),
  )
  forward_method = data.get('forward_method', 'graph')
  network_seed = data.get('network_seed', data.get('seed', 0))

  r_n = ReactionNetwork.from_reaction_strings(
    reaction_strings=data['reaction_strings'],
    L=data['L'],
    seed=network_seed,
    species_names=data['species_names'],
    force_reverse=True,
  )

  sim = ReactionNetworkSimulator(r_n)
  symbolic_rhs, species, rates = sim.get_symbolic_rhs()
  reduced_rhs, remaining_syms, const_syms, rate_syms = sim.get_symbolic_reduced_rhs()
  sim.solve_conservation_laws()
  dR_dC, dR_dC_func, dR_dl, dR_dl_func, dR_dk, dR_dk_func, remaining_syms = \
    sim.get_first_order_derivatives()

  graph_comp = None
  if forward_method == 'graph':
    G = get_digraph_from_adjacency_matrix(
      data['adjacency_matrix'],
      data['input_substrates_list'],
      data['NR'],
      data['NS'],
    )
    # GraphComputation uses topological_sort; recurrent nets (p_r > 0) are cyclic.
    if nx.is_directed_acyclic_graph(G):
      graph_comp = GraphComputation(G, data['input_nodes'], data['target_nodes'])
      graph_comp.build_r_n_maps(r_n)
    else:
      print("Network has cycles (p_r > 0); using ODE forward instead of GraphComputation")
      forward_method = 'ode'

  # Match run_training.py when the pickle has no integrator fields.
  # CRNModel defaults (t_span=(0, 1e5), num_points=10000) are much heavier and
  # routinely hit the 5s SIGALRM timeout on recurrent/MP nets.
  t_span = data.get('t_span', (0, 10000))
  num_points = data.get('num_points', 2)
  rtol = data.get('rtol', 1e-12)
  atol = data.get('atol', 1e-12)
  timeout_seconds = data.get('timeout_seconds', 5)

  crn_model = CRNModel(
    r_n=r_n,
    sim=sim,
    L=data['L'],
    class_ids=data['target_node_idxs'],
    n_inputs=data['n_inputs'],
    default_l0=data['default_l0'],
    forward_method=forward_method,
    graph_comp=graph_comp,
    dR_dC_func=dR_dC_func,
    dR_dk_func=dR_dk_func,
    dR_dl_func=dR_dl_func,
    generate_init_func=generate_positive_initial_concentrations_nnls,
    readout_type=readout_type,
    readout_seed=readout_seed,
    t_span=t_span,
    num_points=num_points,
    rtol=rtol,
    atol=atol,
    timeout_seconds=timeout_seconds,
  )
  crn_model.set_params(data['model_params'])
  print(
    f"ODE settings: t_span={t_span}, num_points={num_points}, "
    f"rtol={rtol}, atol={atol}, timeout={timeout_seconds}s, "
    f"forward={forward_method}"
  )

  def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

  def evaluate_model(input_data_array, return_full_state=False):
    input_data_array = np.atleast_2d(input_data_array)
    predictions = []
    full_states = [] if return_full_state else None

    n_timeouts = 0
    for i in range(input_data_array.shape[0]):
      try:
        output = crn_model.forward(input_data_array[i])
      except TimeoutError:
        n_timeouts += 1
        continue
      predictions.append(output)
      if return_full_state:
        full_states.append(crn_model.get_C_full())

    if n_timeouts:
      print(f"Warning: skipped {n_timeouts} samples due to ODE integration timeout")
    evaluate_model.n_integration_timeouts = (
      getattr(evaluate_model, 'n_integration_timeouts', 0) + n_timeouts
    )

    if not predictions:
      empty = np.zeros((0, crn_model.n_classes))
      empty_cls = np.array([], dtype=int)
      if return_full_state:
        return empty, empty_cls, empty, np.array([])
      return empty, empty_cls, empty

    predictions = np.array(predictions)
    probabilities = softmax(predictions)
    predicted_classes = np.argmax(predictions, axis=1)

    if return_full_state:
      return predictions, predicted_classes, probabilities, np.array(full_states)
    return predictions, predicted_classes, probabilities

  evaluate_model.n_integration_timeouts = 0

  def evaluate_codes(input_data_array):
    """Return h codes for each input sample."""
    input_data_array = np.atleast_2d(input_data_array)
    return np.array([
      extract_activation_parameters(crn_model, input_data_array[i])
      for i in range(input_data_array.shape[0])
    ])

  if training_type == 'mi':
    training_history = load_mi_history(data)
  else:
    training_history = data.get('history', None)

  mi_estimator = None
  if training_type == 'mi' and data.get('code_dim') is not None:
    mi_estimator = create_mi_estimator(
      code_dim=data['code_dim'],
      method=data.get('mi_method', data.get('mi_estimator_type', 'analytical')),
    )

  model_info = {
    'crn_model': crn_model,
    'r_n': r_n,
    'n_classes': data['n_classes'],
    'proj_dim': data['proj_dim'],
    'input_dim': data['input_dim'],
    'seed': data.get('data_seed', data.get('seed', 0)),
    'network_seed': network_seed,
    'target_nodes': data['target_nodes'],
    'training_type': training_type,
    'training_history': training_history,
    'readout_type': readout_type,
    'adjacency_matrix': data['adjacency_matrix'],
    'input_substrates_list': data['input_substrates_list'],
    'NR': data['NR'],
    'NS': data['NS'],
    'species_names': data['species_names'],
    'data': data,
    'evaluate_codes': evaluate_codes,
    'mi_estimator': mi_estimator,
    'layer_indices': data.get('layer_indices', None),
    'model_type': data.get('model_type', 'crn'),
  }

  if training_type == 'mi':
    model_info.update({
      'code_dim': data.get('code_dim'),
      'final_mi_nats': data.get('final_mi_nats'),
      'final_mi_bits': data.get('final_mi_bits'),
      'final_H_z': data.get('final_H_z'),
      'final_H_z_given_R': data.get('final_H_z_given_R'),
      'random_baseline_mi': data.get('random_baseline_mi'),
      'test_h_samples': data.get('test_h_samples'),
      'log_input_space': data.get('log_input_space', False),
      'mi_estimator_name': data.get('mi_estimator_name'),
      'mi_layer': data.get('mi_layer', None),  # ADD THIS TOO (optional)
      'layer_description': data.get('layer_description', None)  # AND THIS (optional)
    })

  return evaluate_model, model_info


def get_final_mi(model_info, window=50):
  """Convenience accessor for final/average MI from loaded MI results."""
  if model_info.get('final_mi_nats') is not None:
    return model_info['final_mi_nats']

  history = model_info.get('training_history')
  if history is not None and hasattr(history, 'get_recent_mi'):
    return history.get_recent_mi(window=window)

  raw = model_info['data'].get('mi_history', {})
  mi_values = raw.get('mi_values', [])
  if len(mi_values) == 0:
    return np.nan
  return float(np.mean(mi_values[-window:]))


def load_training_data(model_info):
  """
  Reconstruct the InputData object used during training.
  Works for both classification and MI pickles.
  """
  data = model_info['data']
  data_gen_method = data.get('data_gen_method', 'lattice')
  extra_info = {}

  if data_gen_method == 'cartesian':
    n_classes, data_list, log_means, alphabet, psi, mode_labels = generate_cartesian_alphabet_data(
      Q=data['Q'],
      L=data['cart_L'],
      N=data['N'],
      C=data['n_classes'],
      M=data['M'],
      n_samples_per_class=data['n_samples_per_class'],
      log_variance=data['log_variance'],
      center_variance=data['center_variance'],
      random_state=data.get('data_seed', data.get('seed', 0)),
    )
    extra_info = {
      'alphabet': alphabet,
      'psi': psi,
      'mode_labels': mode_labels,
    }

  elif data_gen_method == 'lattice':
    lattice_kwargs = dict(
      n_classes=data['n_classes'],
      n_modes=data.get('n_modes', data['n_classes']),
      n_samples_per_class=data['n_samples_per_class'],
      input_dim=data['input_dim'],
      center_variance=data['center_variance'],
      log_variance=data['log_variance'],
      center_offset=data.get('center_offset', 0.0),
      random_state=data.get('data_seed', data.get('seed', 0)),
    )
    if _is_mlp_results(data):
      lattice_kwargs['p_mode_keep'] = data.get('p_mode_keep', 1.0)
    n_classes, data_list, log_means = generate_lognormal_mixture_lattice_centers(
      **lattice_kwargs
    )

  else:
    n_classes, data_list, log_means = generate_lognormal_mixture_random_centers(
      n_classes=data['n_classes'],
      n_modes=data.get('n_modes', data['n_classes']),
      n_samples_per_class=data['n_samples_per_class'],
      input_dim=data['input_dim'],
      center_variance=data['center_variance'],
      log_variance=data['log_variance'],
      center_offset=data.get('center_offset', 0.0),
      random_state=data.get('data_seed', data.get('seed', 0)),
    )

  if data_gen_method != 'cartesian':
    if 'proj_dim' in data and data['proj_dim'] != data['input_dim']:
      data_list = project_data_list(data_list, d=data['proj_dim'])

  if not data.get('log_scale', True):
    input_data = convert_input_data_to_log_scale(InputData(n_classes, data_list))
  else:
    input_data = InputData(n_classes, data_list)

  return input_data, data_list, log_means, extra_info

def extract_mi_code_vector(crn_model, sample, layer_indices=None, use_raw_concentrations=False):
    """
    Extract MI code vector h in [0, 1]^d for one input sample.

    Matches MITrainer.extract_activation_parameters.
    
    Args:
        crn_model: The CRN or MLP model
        sample: Input sample
        layer_indices: Optional list of species indices to extract from.
                      If None, uses hidden_indices (default behavior)
        use_raw_concentrations: If True, return raw concentrations instead of
                               activation fractions. Useful for output layer.
    """
    if isinstance(crn_model, MLPModel):
        x = np.asarray(sample, dtype=float).flatten()
        logits = crn_model.forward(x)
        if layer_indices == "mlp_output":
            return np.asarray(logits, dtype=float)
        h = np.asarray(crn_model.mlp.activations[-2], dtype=float)
        if use_raw_concentrations:
            return h
        return remap_hidden_activations_to_unit_interval(h, crn_model.mlp.activation)

    x = np.asarray(sample, dtype=float).flatten()[:crn_model.n_inputs]
    try:
        _ = crn_model.forward(x)
    except TimeoutError:
        return None

    if crn_model.readout_type == 'linear':
        return np.clip(crn_model._h.copy(), 0.0, 1.0)

    state = crn_model.get_regularization_state()
    
    # Use provided layer_indices or fall back to hidden_indices
    indices_to_use = layer_indices if layer_indices is not None else state['hidden_indices']
    
    if use_raw_concentrations:
        # Extract raw concentrations directly
        C_full = state['C_full']
        return np.array([C_full[i] for i in indices_to_use], dtype=float)
    
    # Use activation fractions (original behavior)
    activation_dict = compute_activation_fractions(
        state['C_full'],
        state['species_names'],
        indices_to_use,
    )
    if len(activation_dict) == 0:
        return np.array([], dtype=float)
    return np.array(
        [activation_dict[i] for i in sorted(activation_dict.keys())],
        dtype=float,
    )

def _replay_mode_labels(
    n_modes, n_classes, input_dim, n_samples_per_mode_training,
    center_variance, log_variance, center_offset, data_seed, data_gen_method='random',
):
    np.random.seed(data_seed)
    per_dim_center_variance = center_variance / input_dim
    per_dim_log_variance = log_variance / input_dim

    if data_gen_method == 'random':
        replay_log_means = []
        for _ in range(n_modes):
            center = np.random.normal(0, np.sqrt(per_dim_center_variance), size=input_dim)
            replay_log_means.append(center + center_offset)
    elif data_gen_method == 'lattice':
        grid_size_per_dim = round(n_modes ** (1.0 / input_dim))
        if grid_size_per_dim ** input_dim != n_modes:
            raise ValueError(f"n_modes={n_modes} must be a perfect {input_dim}-th power")
        grid_coords = np.meshgrid(
            *[np.arange(grid_size_per_dim) for _ in range(input_dim)],
            indexing='ij',
        )
        grid_coords = np.stack([g.flatten() for g in grid_coords], axis=1)
        lattice_spacing = np.sqrt(per_dim_center_variance)
        grid_coords_centered = (grid_coords - (grid_size_per_dim - 1) / 2.0) * lattice_spacing
        replay_log_means = [grid_coords_centered[i] + center_offset for i in range(n_modes)]
    else:
        raise ValueError(f"Unsupported data_gen_method={data_gen_method!r}")

    log_variances = [per_dim_log_variance] * n_modes
    generate_lognormal_mixture(
        n_classes=n_modes,
        n_samples_per_class=n_samples_per_mode_training,
        input_dim=input_dim,
        log_means=replay_log_means,
        log_variances=log_variances,
        random_state=None,
    )

    if n_modes == n_classes:
        return np.arange(n_classes, dtype=int)
    return np.asarray(_assign_modes_to_labels(n_modes, n_classes), dtype=int)


def _resolve_mode_labels(
    log_means, n_classes, mode_labels=None, data_gen_method=None,
    data_seed=None, n_samples_per_mode_training=None, input_dim=None,
    center_variance=None, log_variance=None, center_offset=None,
):
    n_modes = len(log_means)

    if mode_labels is not None:
        ml = np.asarray(mode_labels, dtype=int).flatten()
        if ml.shape[0] == n_modes:
            return ml

    if data_gen_method == 'cartesian':
        raise ValueError(
            "cartesian data requires mode_labels with length len(log_means)."
        )

    if data_gen_method in ('random', 'lattice'):
        return _replay_mode_labels(
            n_modes=n_modes,
            n_classes=n_classes,
            input_dim=input_dim,
            n_samples_per_mode_training=n_samples_per_mode_training,
            center_variance=center_variance,
            log_variance=log_variance,
            center_offset=center_offset,
            data_seed=data_seed,
            data_gen_method=data_gen_method,
        )

    if n_modes == n_classes:
        return np.arange(n_classes, dtype=int)

    raise ValueError(
        f"Could not resolve mode_labels: len(log_means)={n_modes}, n_classes={n_classes}."
    )


def _draw_mode_samples(mode_idx, n_samples, log_means, log_variance, rng, in_log_space=True):
    mu = np.asarray(log_means[mode_idx], dtype=float).flatten()
    dim = mu.shape[0]
    per_dim_var = log_variance / dim
    cov = per_dim_var * np.eye(dim)
    log_draws = rng.multivariate_normal(mu, cov, size=n_samples)
    if in_log_space:
        return [ld for ld in log_draws]
    return [np.exp(ld) for ld in log_draws]


def resolve_layer_indices(model_info, layer_indices=None):
    """
    Resolve which species indices to extract.

    layer_indices may be:
      None / "auto" -> pickle layer_indices, else None (hidden after forward)
      "hidden"      -> None (extract_mi_code_vector uses hidden_indices)
      "output"      -> data['target_node_idxs']
      list/tuple    -> used as-is
    """
    if isinstance(layer_indices, str):
        key = layer_indices.lower()
        if key in ("auto", "none", ""):
            layer_indices = None
        elif key == "hidden":
            return None
        elif key == "output":
            if model_info.get("model_type") == "mlp" or model_info.get("data", {}).get("model_type") == "mlp":
                return "mlp_output"
            idxs = model_info.get("data", {}).get("target_node_idxs")
            if idxs is None:
                raise ValueError("layer_indices='output' but target_node_idxs missing from pickle")
            return list(idxs)
        else:
            raise ValueError(
                f"Unknown layer_indices={layer_indices!r}; "
                "use None/'auto', 'hidden', 'output', or a list of indices"
            )

    if layer_indices is not None:
        if isinstance(layer_indices, str):
            return layer_indices
        return list(layer_indices)

    saved = model_info.get("layer_indices", None)
    if saved is not None:
        print(f"Using layer_indices from saved model: {saved}")
        if isinstance(saved, str):
            return saved
        return list(saved)
    return None


def resolve_use_raw_concentrations(model_info, layer_indices, use_raw_concentrations=None):
    """None -> True when resolved indices match the output/target layer."""
    if use_raw_concentrations is not None:
        return bool(use_raw_concentrations)
    if layer_indices == "mlp_output":
        return True
    if layer_indices is None:
        return False
    target_node_idxs = model_info.get("data", {}).get("target_node_idxs", [])
    if list(layer_indices) == list(target_node_idxs):
        print("Auto-detected output layer - using raw concentrations")
        return True
    return False


def compute_per_sample_mi_activations(
    filepath=None,
    n_samples_per_mode=8,
    plot_seed=0,
    classes_to_plot=None,
    evaluate_model=None,
    model_info=None,
    input_data=None,
    log_means=None,
    extra_info=None,
    layer_indices=None,
    use_raw_concentrations=None,
):
    """
    Unified per-sample activation extractor for CE and MI pickles.

    layer_indices / use_raw_concentrations: see resolve_* helpers and the
    top-of-file layer_indices_default / use_raw_concentrations_default knobs.
    """
    if model_info is None:
        if filepath is None:
            raise ValueError("Must provide filepath or pre-loaded model_info")
        evaluate_model, model_info = load_trained_crn_model(filepath)
        input_data, data_list, log_means, extra_info = load_training_data(model_info)
    else:
        if log_means is None:
            log_means = model_info['data']['log_means']
        if extra_info is None:
            extra_info = {}

    layer_indices = resolve_layer_indices(model_info, layer_indices)
    use_raw_concentrations = resolve_use_raw_concentrations(
        model_info, layer_indices, use_raw_concentrations
    )
    print(
        f"Extracting activations: layer_indices={layer_indices}, "
        f"use_raw_concentrations={use_raw_concentrations}"
    )

    crn_model = model_info['crn_model']
    n_classes = model_info['n_classes']
    input_dim = model_info['input_dim']
    data = model_info['data']

    center_variance = data['center_variance']
    log_variance = data['log_variance']
    center_offset = data.get('center_offset', 0.0)
    data_gen_method = data.get('data_gen_method', 'lattice')
    data_seed = model_info['seed']
    n_samples_per_class = data['n_samples_per_class']
    log_scale = data.get('log_scale', True)

    if classes_to_plot is None:
        classes_to_plot = list(range(n_classes))

    rng = np.random.default_rng(plot_seed)
    n_modes_actual = len(log_means)

    mode_labels_arr = _resolve_mode_labels(
        log_means=log_means,
        n_classes=n_classes,
        mode_labels=extra_info.get('mode_labels', None),
        data_gen_method=data_gen_method,
        data_seed=data_seed,
        n_samples_per_mode_training=n_samples_per_class,
        input_dim=input_dim,
        center_variance=center_variance,
        log_variance=log_variance,
        center_offset=center_offset,
    )
    assert mode_labels_arr.shape[0] == n_modes_actual

    use_log_samples = not log_scale

    per_sample_activations_per_class = []
    ipr_per_class = []
    mode_indices_per_class = []
    mode_sample_counts_per_class = []
    n_timeouts = 0
    n_attempted = 0
    code_dim_found = None

    for class_idx in classes_to_plot:
        mode_indices = np.flatnonzero(mode_labels_arr == class_idx)
        mode_indices_per_class.append(mode_indices)

        class_activations = []
        class_iprs = []
        mode_counts = []

        for mode_idx in sorted(mode_indices):
            mode_idx = int(mode_idx)
            samples = _draw_mode_samples(
                mode_idx,
                n_samples_per_mode,
                log_means,
                log_variance,
                rng=rng,
                in_log_space=use_log_samples,
            )

            mode_activations = []
            mode_iprs = []
            for sample in samples:
                n_attempted += 1
                h = extract_mi_code_vector(crn_model, sample, 
                                          layer_indices=layer_indices,
                                          use_raw_concentrations=use_raw_concentrations)
                if h is None:
                    n_timeouts += 1
                    continue
                if code_dim_found is None:
                    code_dim_found = int(h.size)
                mode_activations.append(h)
                # Only compute IPR if we have data
                if h.size > 0:
                    if use_raw_concentrations:
                        # For raw concentrations, normalize before computing IPR
                        h_sum = h.sum()
                        if h_sum > 0:
                            h_normalized = h / h_sum
                            mode_iprs.append(compute_fractional_ipr(h_normalized))
                        else:
                            mode_iprs.append(np.nan)
                    else:
                        mode_iprs.append(compute_fractional_ipr(h))
                else:
                    mode_iprs.append(np.nan)

            class_activations.extend(mode_activations)
            class_iprs.extend(mode_iprs)
            mode_counts.append(len(mode_activations))

        per_sample_activations_per_class.append(class_activations)
        ipr_per_class.append(np.asarray(class_iprs))
        mode_sample_counts_per_class.append(mode_counts)

    if n_timeouts:
        print(
            f"Warning: skipped {n_timeouts}/{n_attempted} samples "
            f"due to ODE integration timeout"
        )

    if code_dim_found is not None:
        code_dim = code_dim_found
    else:
        code_dim = int(model_info.get('code_dim', 0) or 0)

    stacked = []
    for acts in per_sample_activations_per_class:
        if acts:
            stacked.append(np.asarray(acts))
        else:
            stacked.append(np.zeros((0, code_dim), dtype=float))
    per_sample_activations_per_class = stacked

    return {
        'per_sample_activations_per_class': per_sample_activations_per_class,
        'ipr_per_class': ipr_per_class,
        'mode_indices_per_class': mode_indices_per_class,
        'mode_sample_counts_per_class': mode_sample_counts_per_class,
        'n_integration_timeouts': n_timeouts,
        'n_samples_attempted': n_attempted,
        'n_classes': n_classes,
        'code_dim': code_dim,
        'model_info': model_info,
        'log_means': log_means,
        'extra_info': extra_info,
        'classes_analyzed': list(classes_to_plot),
        'training_type': model_info.get('training_type', 'mi'),
        'final_mi_nats': get_final_mi(model_info),
        'layer_indices_used': layer_indices,
        'use_raw_concentrations': use_raw_concentrations,
    }

def _pca_eigenvalues(H, standardize=True, log_space=False, eps=1e-12):
    """
    Sample-covariance PCA eigenvalues of activation matrix H (N, D).

    Defaults match the notebook: column standardization on, no log transform.
    Returns eigvals (length r) and per-component variance fractions.
    """
    H = np.asarray(H, dtype=float)
    if H.ndim != 2:
        raise ValueError(f"Expected 2D activation matrix, got shape {H.shape}")
    N, D = H.shape
    if N < 2 or D < 1:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    if log_space:
        H = np.log(np.clip(H, eps, None))

    Hc = H - H.mean(axis=0, keepdims=True)
    if standardize:
        sd = Hc.std(axis=0, ddof=1, keepdims=True)
        sd = np.where(sd < eps, 1.0, sd)
        Hc = Hc / sd

    _, S, _ = np.linalg.svd(Hc, full_matrices=False)
    eigvals = (S ** 2) / max(N - 1, 1)
    total = eigvals.sum()
    frac = eigvals / (total + 1e-300)
    return eigvals, frac


def _participation_ratio(eigvals):
    """Effective dimensionality: (sum λ)^2 / sum(λ^2)."""
    eigvals = np.asarray(eigvals, dtype=float)
    if eigvals.size == 0:
        return float("nan")
    s = eigvals.sum()
    return float((s ** 2) / ((eigvals ** 2).sum() + 1e-300))


def _groups_from_mode_counts(per_sample_activations_per_class, mode_sample_counts_per_class):
    """
    Split class-stacked activation rows back into per-GMM-mode blocks.

    mode_sample_counts_per_class[c] is a list of sample counts for modes in class c
    (same order as used when building the class stack).
    """
    groups = []
    for acts, counts in zip(per_sample_activations_per_class, mode_sample_counts_per_class):
        acts = np.asarray(acts, dtype=float)
        start = 0
        for n in counts:
            n = int(n)
            if n <= 0:
                continue
            groups.append(acts[start:start + n])
            start += n
        if start != acts.shape[0]:
            raise ValueError(
                f"mode_sample_counts sum {start} != class activation rows {acts.shape[0]}"
            )
    return groups


def compute_activation_pca_pr_metrics(
    per_sample_activations_per_group,
    standardize=True,
    log_space=False,
    prefix="pca",
):
    """
    Participation-ratio metrics from activation samples grouped by class or mode.

    - total: PCA on all stacked samples
    - between: PCA on group means
    - within_pooled: PCA eigenvalues of the pooled within-group covariance
    - within_mean_pr: average of per-group PRs (groups with >= 2 samples)
    """
    groups = [
        np.asarray(g, dtype=float)
        for g in per_sample_activations_per_group
        if g is not None and np.asarray(g).size > 0
    ]
    out = {
        f"{prefix}_standardize": bool(standardize),
        f"{prefix}_log_space": bool(log_space),
        f"{prefix}_n_groups": len(groups),
    }
    if len(groups) == 0:
        out[f"{prefix}_pr_total"] = float("nan")
        out[f"{prefix}_pr_between"] = float("nan")
        out[f"{prefix}_pr_within_pooled"] = float("nan")
        out[f"{prefix}_pr_within_mean"] = float("nan")
        out[f"{prefix}_first_pc_frac"] = float("nan")
        return out

    H = np.vstack(groups)
    eig_tot, frac_tot = _pca_eigenvalues(H, standardize=standardize, log_space=log_space)
    out[f"{prefix}_pr_total"] = _participation_ratio(eig_tot)
    out[f"{prefix}_first_pc_frac"] = float(frac_tot[0]) if frac_tot.size else float("nan")
    out[f"{prefix}_n_samples"] = int(H.shape[0])
    out[f"{prefix}_n_dims"] = int(H.shape[1])

    means = []
    for g in groups:
        g = np.asarray(g, dtype=float)
        if log_space:
            g = np.log(np.clip(g, 1e-12, None))
        means.append(g.mean(axis=0))
    means = np.array(means, dtype=float)
    # Between PCA: never re-log means (already in log space if requested).
    eig_b, _ = _pca_eigenvalues(means, standardize=standardize, log_space=False)
    out[f"{prefix}_pr_between"] = _participation_ratio(eig_b)

    # Pooled within-group scatter (unnormalized sum of squares), then PCA via SVD
    # after the same preprocess applied independently inside each group is awkward;
    # instead: preprocess all samples with global μ/sd from H, then center per group
    # and pool. For PR we use eigenvalues of the pooled within matrix.
    eps = 1e-12
    Hp = H.astype(float, copy=True)
    if log_space:
        Hp = np.log(np.clip(Hp, eps, None))
    mu = Hp.mean(axis=0, keepdims=True)
    Hc = Hp - mu
    if standardize:
        sd = Hc.std(axis=0, ddof=1, keepdims=True)
        sd = np.where(sd < eps, 1.0, sd)
        Hc = Hc / sd

    # Re-split Hc into groups and form pooled within scatter
    sizes = [g.shape[0] for g in groups]
    pieces = np.split(Hc, np.cumsum(sizes[:-1]), axis=0)
    within_rows = []
    per_group_prs = []
    for piece in pieces:
        if piece.shape[0] < 2:
            continue
        piece_c = piece - piece.mean(axis=0, keepdims=True)
        within_rows.append(piece_c)
        # Per-group PR on the already column-scaled coordinates
        _, S, _ = np.linalg.svd(piece_c, full_matrices=False)
        eig_g = (S ** 2) / max(piece_c.shape[0] - 1, 1)
        per_group_prs.append(_participation_ratio(eig_g))

    if within_rows:
        W = np.vstack(within_rows)
        n_within = W.shape[0]
        _, S_w, _ = np.linalg.svd(W, full_matrices=False)
        # Use n_within - n_groups_used dof roughly via n_within - 1 (same as notebook total)
        eig_w = (S_w ** 2) / max(n_within - 1, 1)
        out[f"{prefix}_pr_within_pooled"] = _participation_ratio(eig_w)
    else:
        out[f"{prefix}_pr_within_pooled"] = float("nan")

    out[f"{prefix}_pr_within_mean"] = (
        float(np.mean(per_group_prs)) if per_group_prs else float("nan")
    )
    return out


def compute_class_separation_metrics(
    per_sample_activations_per_class,
    noise_scale=None,
    mode_sample_counts_per_class=None,
    pca_standardize=True,
    pca_log_space=False,
):
    """
    Compute single-number metrics for class separation suitable for comparing training conditions.
    
    Parameters:
    -----------
    per_sample_activations_per_class : list of arrays
        List of per-sample activations (Cbar) for each class
    noise_scale : float, optional
        If provided, compute noise-corrected metrics assuming lognormal channel:
        log(C) ~ Normal(log(Cbar), noise_scale²)
    mode_sample_counts_per_class : list of lists, optional
        Per-class mode sample counts from compute_per_sample_mi_activations.
        If provided, also compute GMM-mode PCA participation ratios.
    pca_standardize, pca_log_space : bool
        Preprocess for PCA/PR (defaults: standardize=True, log_space=False).
    
    Returns a dictionary with several aggregate metrics. Recommended ones:
    - 'avg_fisher_ratio': Simple average of per-dimension Fisher ratios
    - 'weighted_fisher_ratio': Fisher ratio weighted by inter-class variance
    - 'explained_variance_ratio': Inter / (Inter + Intra), like R² [RECOMMENDED]
    - 'trace_ratio': Classic LDA objective
    - 'mean_pairwise_angle': Average angle between class mean vectors (degrees)
    - 'pca_pr_*': participation-ratio (effective dimensionality) of activations
    
    If noise_scale is provided, also returns '_noisy' versions of all metrics.
    """
    # Get basic variance decomposition
    all_activations = np.vstack(per_sample_activations_per_class)
    global_mean = np.mean(all_activations, axis=0)
    
    n_classes = len(per_sample_activations_per_class)
    n_hidden = all_activations.shape[1]
    
    class_means = np.array([np.mean(acts, axis=0) for acts in per_sample_activations_per_class])
    class_sizes = np.array([acts.shape[0] for acts in per_sample_activations_per_class])
    
    # Between-class scatter
    inter_class_scatter = np.zeros(n_hidden)
    for class_idx in range(n_classes):
        n_c = class_sizes[class_idx]
        diff = class_means[class_idx] - global_mean
        inter_class_scatter += n_c * (diff ** 2)
    inter_class_variance = inter_class_scatter / np.sum(class_sizes)
    
    # Within-class scatter
    intra_class_scatter = np.zeros(n_hidden)
    for class_idx in range(n_classes):
        acts = per_sample_activations_per_class[class_idx]
        class_mean = class_means[class_idx]
        for sample in acts:
            intra_class_scatter += (sample - class_mean) ** 2
    intra_class_variance = intra_class_scatter / np.sum(class_sizes)
    
    # Compute aggregate metrics (UNCORRECTED)
    eps = 1e-10
    
    # 1. Average Fisher Ratio (simple mean)
    fisher_ratio_per_dim = inter_class_variance / (intra_class_variance + eps)
    avg_fisher_ratio = np.mean(fisher_ratio_per_dim)
    
    # 2. Weighted Fisher Ratio (weight by inter-class variance)
    weights = inter_class_variance / (np.sum(inter_class_variance) + eps)
    weighted_fisher_ratio = np.sum(weights * fisher_ratio_per_dim)
    
    # 3. Trace Ratio (classic LDA objective)
    trace_ratio = np.sum(inter_class_variance) / (np.sum(intra_class_variance) + eps)
    
    # 4. Explained Variance Ratio (like R² in regression)
    total_variance = inter_class_variance + intra_class_variance
    explained_var_ratio_per_dim = inter_class_variance / (total_variance + eps)
    explained_variance_ratio = np.mean(explained_var_ratio_per_dim)
    
    # 5. Harmonic mean of Fisher ratios
    harmonic_fisher = len(fisher_ratio_per_dim) / np.sum(1.0 / (fisher_ratio_per_dim + eps))
    
    # 6. Max-normalized Fisher ratio
    max_fisher = np.max(fisher_ratio_per_dim)
    normalized_avg_fisher = avg_fisher_ratio / (max_fisher + eps)
    
    # 7. Pairwise angles between class mean vectors
    dot_matrix = class_means @ class_means.T
    norms = np.linalg.norm(class_means, axis=1)
    norm_outer = np.outer(norms, norms)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_matrix = np.divide(
            dot_matrix,
            norm_outer,
            out=np.zeros_like(dot_matrix, dtype=float),
            where=norm_outer > 1e-10,
        )
    cos_matrix = np.clip(cos_matrix, -1.0, 1.0)
    angle_matrix_deg = np.degrees(np.arccos(cos_matrix))
    
    upper_tri_indices = np.triu_indices(n_classes, k=1)
    pairwise_angles = angle_matrix_deg[upper_tri_indices]
    
    mean_pairwise_angle = np.mean(pairwise_angles)
    min_pairwise_angle = np.min(pairwise_angles)
    max_pairwise_angle = np.max(pairwise_angles)
    std_pairwise_angle = np.std(pairwise_angles)
    
    # Build base results dict
    results = {
        'avg_fisher_ratio': avg_fisher_ratio,
        'weighted_fisher_ratio': weighted_fisher_ratio,
        'trace_ratio': trace_ratio,
        'explained_variance_ratio': explained_variance_ratio,
        'harmonic_fisher_ratio': harmonic_fisher,
        'normalized_avg_fisher': normalized_avg_fisher,
        # Pairwise angle metrics
        'mean_pairwise_angle': mean_pairwise_angle,
        'min_pairwise_angle': min_pairwise_angle,
        'max_pairwise_angle': max_pairwise_angle,
        'std_pairwise_angle': std_pairwise_angle,
        'angle_matrix': angle_matrix_deg,
        # Additional components for reference
        'inter_total': np.sum(inter_class_variance),
        'intra_total': np.sum(intra_class_variance),
        'n_dimensions': n_hidden,
        'class_means': class_means,
        'global_mean': global_mean,
    }
    
    # ========== NOISE CORRECTION (APPROACH 3) ==========
    if noise_scale is not None:
        sigma_sq = noise_scale ** 2
        exp_sigma_sq = np.exp(sigma_sq)
        
        # Compute expected squared codes for measurement noise term
        # E[Cbar²] per dimension
        expected_cbar_squared = np.mean(all_activations ** 2, axis=0)
        
        # Corrected inter-class variance: exp(σ²) × inter
        # (accounts for mean shift by exp(σ²/2))
        inter_class_variance_noisy = exp_sigma_sq * inter_class_variance
        
        # Corrected intra-class variance: exp(σ²) × intra + E[Cbar²] × (exp(σ²) - 1)
        # First term: original variance propagated through lognormal
        # Second term: measurement noise variance
        measurement_noise_variance = expected_cbar_squared * (exp_sigma_sq - 1)
        intra_class_variance_noisy = exp_sigma_sq * intra_class_variance + measurement_noise_variance
        
        # Recompute all metrics with corrected variances
        fisher_ratio_per_dim_noisy = inter_class_variance_noisy / (intra_class_variance_noisy + eps)
        avg_fisher_ratio_noisy = np.mean(fisher_ratio_per_dim_noisy)
        
        weights_noisy = inter_class_variance_noisy / (np.sum(inter_class_variance_noisy) + eps)
        weighted_fisher_ratio_noisy = np.sum(weights_noisy * fisher_ratio_per_dim_noisy)
        
        trace_ratio_noisy = np.sum(inter_class_variance_noisy) / (np.sum(intra_class_variance_noisy) + eps)
        
        total_variance_noisy = inter_class_variance_noisy + intra_class_variance_noisy
        explained_var_ratio_per_dim_noisy = inter_class_variance_noisy / (total_variance_noisy + eps)
        explained_variance_ratio_noisy = np.mean(explained_var_ratio_per_dim_noisy)
        
        harmonic_fisher_noisy = len(fisher_ratio_per_dim_noisy) / np.sum(1.0 / (fisher_ratio_per_dim_noisy + eps))
        
        max_fisher_noisy = np.max(fisher_ratio_per_dim_noisy)
        normalized_avg_fisher_noisy = avg_fisher_ratio_noisy / (max_fisher_noisy + eps)
        
        # Add noise-corrected metrics to results
        results.update({
            # Noise-corrected separation metrics
            'avg_fisher_ratio_noisy': avg_fisher_ratio_noisy,
            'weighted_fisher_ratio_noisy': weighted_fisher_ratio_noisy,
            'trace_ratio_noisy': trace_ratio_noisy,
            'explained_variance_ratio_noisy': explained_variance_ratio_noisy,
            'harmonic_fisher_ratio_noisy': harmonic_fisher_noisy,
            'normalized_avg_fisher_noisy': normalized_avg_fisher_noisy,
            # Noise-corrected variance components
            'inter_total_noisy': np.sum(inter_class_variance_noisy),
            'intra_total_noisy': np.sum(intra_class_variance_noisy),
            'measurement_noise_total': np.sum(measurement_noise_variance),
            # Noise parameters used
            'noise_scale': noise_scale,
            'exp_sigma_squared': exp_sigma_sq,
        })

    # PCA participation ratios (effective dimensionality of activation cloud)
    # Class-grouped: between/within refer to label classes (same grouping as Fisher).
    results.update(
        compute_activation_pca_pr_metrics(
            per_sample_activations_per_class,
            standardize=pca_standardize,
            log_space=pca_log_space,
            prefix="pca",
        )
    )
    # Alias the class-grouped between/within keys for clarity next to Fisher metrics.
    results["pca_pr_between_class"] = results.pop("pca_pr_between")
    results["pca_pr_within_class_pooled"] = results.pop("pca_pr_within_pooled")
    results["pca_pr_within_class_mean"] = results.pop("pca_pr_within_mean")

    # GMM-mode-grouped: between/within refer to input modes (notebook-style).
    if mode_sample_counts_per_class is not None:
        mode_groups = _groups_from_mode_counts(
            per_sample_activations_per_class,
            mode_sample_counts_per_class,
        )
        mode_pr = compute_activation_pca_pr_metrics(
            mode_groups,
            standardize=pca_standardize,
            log_space=pca_log_space,
            prefix="pca_mode",
        )
        # Rename between/within for mode grouping; keep total as pca_mode_pr_total.
        mode_pr["pca_mode_pr_between_mode"] = mode_pr.pop("pca_mode_pr_between")
        mode_pr["pca_mode_pr_within_mode_pooled"] = mode_pr.pop("pca_mode_pr_within_pooled")
        mode_pr["pca_mode_pr_within_mode_mean"] = mode_pr.pop("pca_mode_pr_within_mean")
        results.update(mode_pr)

    # ========== INTRINSIC DIMENSIONALITY ESTIMATION ==========
    if SKDIM_AVAILABLE and all_activations.shape[0] >= 10:
        # Global ID estimation (all samples)
        for name, Estimator in _SKDIM_ESTIMATORS.items():
            try:
                id_est = Estimator().fit_transform(all_activations)
                results[name] = float(id_est)
            except Exception as e:
                results[name] = float("nan")
                print(f"  ID estimation {name} failed: {e}")

        # Per-class / per-mode / centers use MLE (most robust)
        MLE = _SKDIM_ESTIMATORS["id_mle"]

        per_class_ids = []
        for class_idx, acts in enumerate(per_sample_activations_per_class):
            acts = np.asarray(acts, dtype=float)
            if acts.shape[0] < 10:  # Need enough samples
                per_class_ids.append(float("nan"))
                continue
            try:
                id_est = MLE().fit_transform(acts)
                per_class_ids.append(float(id_est))
            except Exception:
                per_class_ids.append(float("nan"))

        results["id_per_class"] = per_class_ids
        results["id_mean_per_class"] = float(np.nanmean(per_class_ids))
        results["id_std_per_class"] = float(np.nanstd(per_class_ids))

        # Per-mode ID estimation (if mode grouping available)
        if mode_sample_counts_per_class is not None:
            per_mode_ids = []
            for mode_acts in mode_groups:
                mode_acts = np.asarray(mode_acts, dtype=float)
                if mode_acts.shape[0] < 10:
                    per_mode_ids.append(float("nan"))
                    continue
                try:
                    id_est = MLE().fit_transform(mode_acts)
                    per_mode_ids.append(float(id_est))
                except Exception:
                    per_mode_ids.append(float("nan"))

            results["id_per_mode"] = per_mode_ids
            results["id_mean_per_mode"] = float(np.nanmean(per_mode_ids))
            results["id_std_per_mode"] = float(np.nanstd(per_mode_ids))

        # ID of class means (cluster center dimensionality)
        if class_means.shape[0] >= 10:
            try:
                id_centers = MLE().fit_transform(class_means)
                results["id_class_centers"] = float(id_centers)
            except Exception:
                results["id_class_centers"] = float("nan")
        else:
            results["id_class_centers"] = float("nan")

    return results



# ============== SERIALIZATION / I/O HELPERS ==============

_LIVE_MODEL_INFO_KEYS = (
    "crn_model",
    "r_n",
    "evaluate_codes",
    "mi_estimator",
)


def strip_model_info(model_info):
    """Drop live objects/callables; keep scalars, arrays, and nested training `data`."""
    out = {}
    for key, value in model_info.items():
        if key in _LIVE_MODEL_INFO_KEYS:
            continue
        if key == "training_history":
            # Prefer plain dict already stored in the training pickle.
            data = model_info.get("data", {})
            if isinstance(data, dict) and data.get("mi_history") is not None:
                out["training_history"] = data.get("mi_history")
            else:
                out["training_history"] = None
            continue
        out[key] = value
    return out


def find_training_pickle(input_dir):
    """Prefer MI pickle if present, else classification pickle."""
    candidates = (
        os.path.join(input_dir, "mi_training_results.pkl"),
        os.path.join(input_dir, "training_results.pkl"),
    )
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No training_results.pkl or mi_training_results.pkl in {input_dir}"
    )


def _noise_scale_for_separation(model_info):
    """Scalar noise_scale for log-normal MI; None otherwise."""
    data = model_info.get("data", {})
    method = data.get("mi_method", data.get("mi_estimator_type", ""))
    if model_info.get("training_type") != "mi":
        return None
    if method != "lognormal":
        return None
    noise_scale = data.get("noise_scale")
    if noise_scale is None:
        return None
    return float(np.mean(np.asarray(noise_scale, dtype=float)))


def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved {path}")


def compute_model_accuracy(evaluate_model, input_data, model_info, n_test_samples=100, seed=None):
    """
    Compute overall and per-class accuracy by evaluating the model on test samples.
    
    Parameters
    ----------
    evaluate_model : callable
        Function that takes input samples and returns (outputs, predicted_classes, probs)
    input_data : InputData
        Input data object with get_next_training_sample method
    model_info : dict
        Model info dict containing n_classes
    n_test_samples : int
        Number of test samples per class
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing accuracy metrics
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_classes = model_info['n_classes']
    total_correct = 0
    total_samples = 0
    per_class_accuracy = []
    per_class_correct = []
    per_class_total = []
    
    for class_idx in range(n_classes):
        samples = []
        for _ in range(n_test_samples):
            sample = input_data.get_next_training_sample(class_idx)
            samples.append(sample)
        
        samples = np.array(samples)
        outputs, predicted_classes, probs = evaluate_model(samples)
        
        n_ok = len(predicted_classes)
        class_correct = int(np.sum(predicted_classes == class_idx)) if n_ok else 0
        class_accuracy = class_correct / n_ok if n_ok else float('nan')
        per_class_accuracy.append(class_accuracy)
        per_class_correct.append(class_correct)
        per_class_total.append(n_ok)
        
        total_correct += class_correct
        total_samples += n_ok
    
    overall_accuracy = total_correct / total_samples if total_samples else float('nan')
    valid_acc = [a for a in per_class_accuracy if np.isfinite(a)]
    avg_class_accuracy = np.mean(valid_acc) if valid_acc else float('nan')
    std_class_accuracy = np.std(valid_acc) if valid_acc else float('nan')
    min_class_accuracy = np.min(valid_acc) if valid_acc else float('nan')
    max_class_accuracy = np.max(valid_acc) if valid_acc else float('nan')
    
    return {
        'overall_accuracy': overall_accuracy,
        'avg_class_accuracy': avg_class_accuracy,
        'std_class_accuracy': std_class_accuracy,
        'min_class_accuracy': min_class_accuracy,
        'max_class_accuracy': max_class_accuracy,
        'per_class_accuracy': per_class_accuracy,
        'per_class_correct': per_class_correct,
        'per_class_total': per_class_total,
        'n_test_samples': n_test_samples,
        'n_samples_evaluated': total_samples,
        'n_integration_timeouts': getattr(evaluate_model, 'n_integration_timeouts', 0),
        'n_classes': n_classes,
    }


def run_analysis(
    input_dir,
    output_dir,
    n_samples_per_mode=None,
    plot_seed=None,
    param1=None,
    param2=None,
    param3=None,
    do_write_model_info=None,
    do_write_activations=None,
    do_write_separation=None,
    do_write_accuracy=None,
    n_test_samples=None,
    layer_indices=None,
    use_raw_concentrations=None,
):
    if n_samples_per_mode is None:
        n_samples_per_mode = n_samples_per_mode_default
    if plot_seed is None:
        plot_seed = plot_seed_default
    if do_write_model_info is None:
        do_write_model_info = write_model_info
    if do_write_activations is None:
        do_write_activations = write_activations
    if do_write_separation is None:
        do_write_separation = write_separation
    if do_write_accuracy is None:
        do_write_accuracy = write_accuracy
    if n_test_samples is None:
        n_test_samples = n_test_samples_default
    if layer_indices is None:
        layer_indices = layer_indices_default
    if use_raw_concentrations is None:
        use_raw_concentrations = use_raw_concentrations_default

    need_activations = do_write_activations or do_write_separation
    if not (do_write_model_info or need_activations or do_write_accuracy):
        raise ValueError("Nothing to write: enable at least one write_* flag")

    os.makedirs(output_dir, exist_ok=True)
    print(
        "Writing: "
        f"model_info={do_write_model_info}, "
        f"activations={do_write_activations}, "
        f"separation={do_write_separation}, "
        f"accuracy={do_write_accuracy}"
    )

    pickle_path = find_training_pickle(input_dir)
    print(f"Loading training results from: {pickle_path}")
    t0 = time.time()

    evaluate_model, model_info = load_trained_crn_model(pickle_path)
    input_data, data_list, log_means, extra_info = load_training_data(model_info)
    print(f"Loaded {model_info['training_type']} model in {time.time() - t0:.1f}s")

    act_results = None
    activations_save = None
    separation_save = None
    accuracy_save = None

    if need_activations:
        # Unified CE/MI activation path
        t1 = time.time()
        act_results = compute_per_sample_mi_activations(
            n_samples_per_mode=n_samples_per_mode,
            plot_seed=plot_seed,
            evaluate_model=evaluate_model,
            model_info=model_info,
            input_data=input_data,
            log_means=log_means,
            extra_info=extra_info,
            layer_indices=layer_indices,
            use_raw_concentrations=use_raw_concentrations,
        )
        print(f"Activations computed in {time.time() - t1:.1f}s")

        activations_save = {
            "per_sample_activations_per_class": act_results["per_sample_activations_per_class"],
            "ipr_per_class": act_results["ipr_per_class"],
            "mode_indices_per_class": act_results["mode_indices_per_class"],
            "mode_sample_counts_per_class": act_results["mode_sample_counts_per_class"],
            "n_classes": act_results["n_classes"],
            "code_dim": act_results["code_dim"],
            "classes_analyzed": act_results["classes_analyzed"],
            "training_type": act_results["training_type"],
            "final_mi_nats": act_results.get("final_mi_nats"),
            "layer_indices_used": act_results.get("layer_indices_used"),
            "use_raw_concentrations": act_results.get("use_raw_concentrations"),
            "log_means": act_results.get("log_means"),
            "n_integration_timeouts": act_results.get("n_integration_timeouts", 0),
            "n_samples_attempted": act_results.get("n_samples_attempted"),
            "n_samples_per_mode": n_samples_per_mode,
            "plot_seed": plot_seed,
            "param1": param1,
            "param2": param2,
            "param3": param3,
        }

    if do_write_separation:
        per_class = act_results["per_sample_activations_per_class"]
        noise_scale = _noise_scale_for_separation(model_info)
        t2 = time.time()
        separation = compute_class_separation_metrics(
            per_class,
            noise_scale=noise_scale,
            mode_sample_counts_per_class=act_results.get(
                "mode_sample_counts_per_class"
            ),
            pca_standardize=True,
            pca_log_space=False,
        )
        print(f"Separation metrics computed in {time.time() - t2:.1f}s")
        print(
            f"  PCA PR total={separation.get('pca_pr_total', float('nan')):.2f}, "
            f"between_class={separation.get('pca_pr_between_class', float('nan')):.2f}, "
            f"within_class={separation.get('pca_pr_within_class_pooled', float('nan')):.2f}"
        )
        if "pca_mode_pr_total" in separation:
            print(
                f"  PCA mode PR total={separation['pca_mode_pr_total']:.2f}, "
                f"between_mode={separation.get('pca_mode_pr_between_mode', float('nan')):.2f}, "
                f"within_mode={separation.get('pca_mode_pr_within_mode_pooled', float('nan')):.2f}"
            )
        if "id_mle" in separation:
            print(
                f"  ID estimates: MLE={separation.get('id_mle', float('nan')):.2f}, "
                f"TwoNN={separation.get('id_twonn', float('nan')):.2f}, "
                f"mean_per_class={separation.get('id_mean_per_class', float('nan')):.2f}"
            )
        separation_save = dict(separation)
        separation_save.update({
            "training_type": model_info["training_type"],
            "param1": param1,
            "param2": param2,
            "param3": param3,
        })

    if do_write_accuracy:
        t3 = time.time()
        accuracy = compute_model_accuracy(
            evaluate_model=evaluate_model,
            input_data=input_data,
            model_info=model_info,
            n_test_samples=n_test_samples,
            seed=plot_seed,
        )
        print(f"Accuracy computed in {time.time() - t3:.1f}s")
        print(f"  Overall accuracy: {accuracy['overall_accuracy']:.1%}")
        print(f"  Per-class: {[f'{acc:.1%}' for acc in accuracy['per_class_accuracy']]}")
        accuracy_save = dict(accuracy)
        accuracy_save.update({
            "training_type": model_info["training_type"],
            "param1": param1,
            "param2": param2,
            "param3": param3,
        })

    model_info_save = None
    if do_write_model_info:
        model_info_save = strip_model_info(model_info)
        model_info_save["source_pickle"] = os.path.basename(pickle_path)
        model_info_save["input_dir"] = input_dir
        model_info_save["output_dir"] = output_dir
        model_info_save["param1"] = param1
        model_info_save["param2"] = param2
        model_info_save["param3"] = param3
        model_info_save["n_samples_per_mode"] = n_samples_per_mode
        model_info_save["plot_seed"] = plot_seed
        model_info_save["write_flags"] = {
            "model_info": do_write_model_info,
            "activations": do_write_activations,
            "separation": do_write_separation,
            "accuracy": do_write_accuracy,
        }
        save_pickle(os.path.join(output_dir, "model_info.pkl"), model_info_save)

    if do_write_activations:
        save_pickle(os.path.join(output_dir, "activations.pkl"), activations_save)

    if do_write_separation:
        save_pickle(os.path.join(output_dir, "separation.pkl"), separation_save)

    if do_write_accuracy:
        save_pickle(os.path.join(output_dir, "accuracy.pkl"), accuracy_save)

    print(f"Done in {time.time() - t0:.1f}s total")
    return model_info_save, activations_save, separation_save, accuracy_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze trained CE/MI CRN results and save pickles."
    )
    parser.add_argument("--param1", type=param_type1, required=False, help="Sweep param1")
    parser.add_argument("--param2", type=param_type2, required=False, help="Sweep param2")
    parser.add_argument("--param3", type=param_type3, required=False, help="Sweep param3")
    parser.add_argument("--input", type=str, required=True, help="Training output directory")
    parser.add_argument("--output", type=str, required=True, help="Analysis output directory")
    parser.add_argument(
        "--n_samples_per_mode",
        type=int,
        default=None,
        help="Samples per mode (default: n_samples_per_mode_default in this file)",
    )
    parser.add_argument(
        "--plot_seed",
        type=int,
        default=None,
        help="RNG seed for activation sampling (default: plot_seed_default in this file)",
    )
    args = parser.parse_args()

    run_analysis(
        input_dir=args.input,
        output_dir=args.output,
        n_samples_per_mode=args.n_samples_per_mode,
        plot_seed=args.plot_seed,
        param1=args.param1,
        param2=args.param2,
        param3=args.param3,
    )
