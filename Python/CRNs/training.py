"""
Training module for reaction networks.

This module contains functions for training reaction networks.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Iterator, Optional
from abc import ABC, abstractmethod
import sympy
from scipy.optimize import nnls
from scipy.optimize import minimize


# ============== ABSTRACT BASE CLASS ==============

class ForwardModel(ABC):
    """Abstract base class for trainable forward models."""
    
    @abstractmethod
    def forward(self, inputs) -> np.ndarray:
        """Compute class outputs from inputs.
        
        Args:
            inputs: input values (model-specific format)
            
        Returns:
            class_outputs: array of shape (n_classes,)
        """
        pass
    
    @abstractmethod
    def backward(self, probs: np.ndarray, target_idx: int, temperature: float = 1.0) -> Dict[str, np.ndarray]:
        """Compute parameter gradients for cross-entropy loss.
        
        Args:
            probs: softmax probabilities (n_classes,)
            target_idx: index of correct class
            temperature: softmax temperature
            
        Returns:
            dict mapping parameter names to gradient arrays
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, np.ndarray]:
        """Return dict of parameter_name -> parameter array."""
        pass
    
    @abstractmethod
    def set_params(self, params_dict: Dict[str, np.ndarray]):
        """Set parameters from dict."""
        pass
    
    @abstractmethod
    def get_param_shapes(self) -> Dict[str, Tuple]:
        """Return dict of parameter_name -> shape tuple."""
        pass


# ============== MLP MODEL ==============

class MLPModel(ForwardModel):
    """MLP-based forward model with backprop gradients."""
    
    def __init__(self, mlp: 'SimpleMLP', n_classes: int):
        """
        Args:
            mlp: SimpleMLP instance
            n_classes: number of output classes
        """
        self.mlp = mlp
        self.n_classes = n_classes
    
    def forward(self, inputs) -> np.ndarray:
        """Forward pass returning logits."""
        x = np.asarray(inputs).flatten()
        logits = self.mlp.forward(x)
        return logits
    
    def backward(self, probs: np.ndarray, target_idx: int, temperature: float = 1.0) -> Dict[str, np.ndarray]:
        """Backprop gradients for cross-entropy loss."""
        grad_weights, grad_biases = self.mlp.backward(probs, target_idx, temperature)
        
        # Flatten to single gradient vector
        grads = []
        for dW, db in zip(grad_weights, grad_biases):
            grads.append(dW.flatten())
            grads.append(db.flatten())
        
        return {'params': np.concatenate(grads)}
    
    def backward_mse(self, probs: np.ndarray, target_vec: np.ndarray, temperature: float = 1.0) -> Dict[str, np.ndarray]:
        """Backprop gradients for MSE loss."""
        grad_weights, grad_biases = self.mlp.backward_mse(probs, target_vec, temperature)
        
        grads = []
        for dW, db in zip(grad_weights, grad_biases):
            grads.append(dW.flatten())
            grads.append(db.flatten())
        
        return {'params': np.concatenate(grads)}
    
    def get_params(self) -> Dict[str, np.ndarray]:
        return {'params': self.mlp.get_flat_params()}
    
    def set_params(self, params_dict: Dict[str, np.ndarray]):
        self.mlp.set_flat_params(params_dict['params'])
    
    def get_param_shapes(self) -> Dict[str, Tuple]:
        return {'params': (self.mlp.get_param_count(),)}


# ============== CRN MODEL ==============

class CRNModel(ForwardModel):
    """CRN-based forward model with analytical gradients.
    
    Supports multiple forward computation methods:
    - 'ode': Full ODE integration to steady state (accurate, slow)
    - 'graph': Fast graph-based approximation
    
    Both methods use the same analytical gradient computation.
    """
    
    def __init__(self, 
                 r_n,
                 sim,
                 L: np.ndarray,
                 class_ids: List[int],
                 n_inputs: int,
                 default_l0: np.ndarray,
                 forward_method: str = 'ode',
                 graph_comp: Optional['GraphComputation'] = None,
                 dR_dC_func=None,
                 dR_dk_func=None,
                 dR_dl_func=None,
                 generate_init_func=None,
                 t_span: Tuple[float, float] = (0, 100000),
                 num_points: int = 10000,
                 rtol: float = 1e-12,
                 atol: float = 1e-12):
        """
        Args:
            r_n: ReactionNetwork instance
            sim: Simulator instance
            L: Conservation law matrix
            class_ids: indices of output species (target nodes)
            n_inputs: number of input species (receptors)
            default_l0: default conservation constants
            forward_method: 'ode' or 'graph'
            graph_comp: GraphComputation instance (required if forward_method='graph')
            dR_dC_func: Jacobian dR/dC function
            dR_dk_func: Jacobian dR/dk function  
            dR_dl_func: Jacobian dR/dl function
            generate_init_func: function to generate initial concentrations from l0
            t_span: ODE integration time span
            num_points: number of ODE integration points
            rtol, atol: ODE integration tolerances
        """
        self.r_n = r_n
        self.sim = sim
        self.L = L
        self.class_ids = class_ids
        self.n_classes = len(class_ids)
        self.n_inputs = n_inputs
        
        self.forward_method = forward_method
        self.graph_comp = graph_comp
        
        # Jacobian functions for analytical gradients
        self.dR_dC_func = dR_dC_func
        self.dR_dk_func = dR_dk_func
        self.dR_dl_func = dR_dl_func
        self.generate_init_func = generate_init_func
        
        # ODE settings
        self.t_span = t_span
        self.num_points = num_points
        self.rtol = rtol
        self.atol = atol
        
        # Parameters
        self._rates = np.array(r_n.get_rates())
        self._default_l0 = default_l0.copy()
        
        # Indices for trainable l0 (typically exclude inputs and outputs)
        self.l0_train_range = (n_inputs, len(default_l0) - len(class_ids))
        
        # Cached forward pass results
        self._C_full = None
        self._C_reduced_final = None
        self._current_l0 = None
    
    def set_forward_method(self, method: str):
        """Switch forward computation method ('ode' or 'graph')."""
        if method not in ('ode', 'graph'):
            raise ValueError(f"Unknown forward method: {method}")
        if method == 'graph' and self.graph_comp is None:
            raise ValueError("GraphComputation not provided")
        self.forward_method = method
    
    def forward(self, inputs) -> np.ndarray:
        """Compute steady-state class outputs.
        
        Args:
            inputs: array of input values (receptor concentrations)
            
        Returns:
            class_outputs: array of shape (n_classes,)
        """
        # Build l0 with inputs
        l0 = self._default_l0.copy()
        l0[:self.n_inputs] = inputs
        self._current_l0 = l0
        
        if self.forward_method == 'ode':
            C_full, C_reduced = self._forward_ode(l0)
        elif self.forward_method == 'graph':
            C_full, C_reduced = self._forward_graph(l0)
        else:
            raise ValueError(f"Unknown forward method: {self.forward_method}")
        
        self._C_full = C_full
        self._C_reduced_final = C_reduced
        
        # Extract class outputs
        return np.array([C_full[idx] for idx in self.class_ids])
    
    def _forward_ode(self, l0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass via ODE integration."""
        C_full = self.generate_init_func(self.L, l0)
        _, C_reduced_init = self.sim.get_const_and_reduced_init(C_full)
        self.sim.make_reduced_rhs_with_conservation(l0)
        
        sol_reduced, C_reduced_final = self.sim.integrate(
            self.sim.reduced_ode_rhs, C_reduced_init,
            t_span=self.t_span, num_points=self.num_points,
            method='LSODA', rtol=self.rtol, atol=self.atol
        )
        
        C_full = self.sim.recover_eliminated_species(l0, C_reduced_final)
        return C_full, C_reduced_final
    
    def _forward_graph(self, l0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass via graph computation."""
        # Graph forward returns C_full directly
        C_full = self.graph_comp.forward(self._rates, l0)
        
        # Derive C_reduced from C_full for gradient computation
        _, C_reduced = self.sim.get_const_and_reduced_init(C_full)
        
        return C_full, C_reduced
    
    def backward(self, probs: np.ndarray, target_idx: int, temperature: float = 1.0) -> Dict[str, np.ndarray]:
        """Compute analytical gradients w.r.t. rates and l0.
        
        Uses dC_dk and dC_dl sensitivity functions evaluated at steady state.
        """
        l0 = self._current_l0
        rates = self._rates
        C_reduced = self._C_reduced_final
        
        # Compute sensitivity matrices at steady state
        dC_dk = self.sim.dC_dk_func(
            C_reduced, l0, rates, self.dR_dC_func, self.dR_dk_func
        )
        dC_dk_full = self.sim.compute_dC_dk_full(dC_dk, l_bool=False)
        
        dC_dl = self.sim.dC_dl_func(
            C_reduced, l0, rates, self.dR_dC_func, self.dR_dl_func
        )
        dC_dl_full = self.sim.compute_dC_dk_full(dC_dl, l_bool=True)
        
        # Compute softmax Jacobian: dp/dC = (diag(p) - outer(p,p)) / T
        softmax_jacobian = (np.diag(probs) - np.outer(probs, probs)) / temperature
        
        # Extract sensitivities at class nodes
        dC_dk_class = np.array([dC_dk_full[idx] for idx in self.class_ids])
        dC_dl_class = np.array([dC_dl_full[idx] for idx in self.class_ids])
        
        # dp/dk = (dp/dC) @ (dC/dk)
        dprobs_dk = softmax_jacobian @ dC_dk_class
        dprobs_dl = softmax_jacobian @ dC_dl_class
        
        # Cross-entropy gradient: dL/dk = -dp[target]/dk / (p[target] + eps)
        eps = 1e-8
        grad_k = -dprobs_dk[target_idx] / (probs[target_idx] + eps)
        grad_l = -dprobs_dl[target_idx] / (probs[target_idx] + eps)
        
        # Chain rule for log-space parameters
        return {
            'log_rates': grad_k * self._rates,
            'log_l0': grad_l * self._default_l0
        }
    
    def backward_mse(self, probs: np.ndarray, target_vec: np.ndarray, temperature: float = 1.0) -> Dict[str, np.ndarray]:
        """Compute analytical gradients for MSE loss."""
        l0 = self._current_l0
        rates = self._rates
        C_reduced = self._C_reduced_final
        
        # Compute sensitivity matrices
        dC_dk = self.sim.dC_dk_func(
            C_reduced, l0, rates, self.dR_dC_func, self.dR_dk_func
        )
        dC_dk_full = self.sim.compute_dC_dk_full(dC_dk, l_bool=False)
        
        dC_dl = self.sim.dC_dl_func(
            C_reduced, l0, rates, self.dR_dC_func, self.dR_dl_func
        )
        dC_dl_full = self.sim.compute_dC_dk_full(dC_dl, l_bool=True)
        
        # Softmax Jacobian
        softmax_jacobian = (np.diag(probs) - np.outer(probs, probs)) / temperature
        
        # Extract at class nodes
        dC_dk_class = np.array([dC_dk_full[idx] for idx in self.class_ids])
        dC_dl_class = np.array([dC_dl_full[idx] for idx in self.class_ids])
        
        dprobs_dk = softmax_jacobian @ dC_dk_class
        dprobs_dl = softmax_jacobian @ dC_dl_class
        
        # MSE gradient: dL/dp = (p - y), then chain through softmax
        diff = probs - target_vec
        grad_k = np.sum(diff[:, None] * dprobs_dk, axis=0)
        grad_l = np.sum(diff[:, None] * dprobs_dl, axis=0)
        
        return {
            'log_rates': grad_k * self._rates,
            'log_l0': grad_l * self._default_l0
        }
    
    def get_params(self) -> Dict[str, np.ndarray]:
        return {
            'log_rates': np.log(self._rates),
            'log_l0': np.log(self._default_l0)
        }
    
    def set_params(self, params_dict: Dict[str, np.ndarray]):
        self._rates = np.exp(params_dict['log_rates'])
        self._default_l0 = np.exp(params_dict['log_l0'])
        self.r_n.update_rates(self._rates)
    
    def get_param_shapes(self) -> Dict[str, Tuple]:
        return {
            'log_rates': self._rates.shape,
            'log_l0': self._default_l0.shape
        }
    
    def get_C_full(self) -> np.ndarray:
        """Return the full concentration vector from last forward pass."""
        return self._C_full


# ============== UNIFIED TRAINER ==============

class UnifiedTrainer:
    """Unified training loop for any ForwardModel."""
    
    def __init__(self, 
                 model: ForwardModel,
                 optimizer_type: str = 'adam',
                 lr: float = 0.1,
                 lr_dict: Optional[Dict[str, float]] = None,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8,
                 max_grad_norm: float = 50.0,
                 loss_type: str = 'cross_entropy',
                 frozen_params: Optional[List[str]] = None):
        """
        Args:
            model: ForwardModel instance (MLPModel or CRNModel)
            optimizer_type: 'adam' or 'sgd'
            lr: default learning rate
            lr_dict: optional per-parameter learning rates {'param_name': lr}
            beta1, beta2, eps: Adam hyperparameters
            max_grad_norm: gradient clipping threshold
            loss_type: 'cross_entropy' or 'mse'
            frozen_params: list of parameter names to freeze (e.g. ['log_l0'])
        """
        self.model = model
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.lr_dict = lr_dict or {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.loss_type = loss_type
        self.frozen_params = set(frozen_params) if frozen_params else set()
        
        # Initialize Adam state for each parameter group
        self.m = {}  # first moment
        self.v = {}  # second moment
        self.t = 0   # timestep
        
        for name, shape in model.get_param_shapes().items():
            self.m[name] = np.zeros(shape)
            self.v[name] = np.zeros(shape)
    
    def softmax(self, outputs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax with temperature scaling."""
        scaled = outputs / temperature
        shifted = scaled - np.max(scaled)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals)
    
    def compute_loss(self, probs: np.ndarray, target_idx: int) -> float:
        """Compute loss value."""
        if self.loss_type == 'cross_entropy':
            return -np.log(probs[target_idx] + 1e-8)
        elif self.loss_type == 'mse':
            target_vec = np.zeros(len(probs))
            target_vec[target_idx] = 1.0
            return 0.5 * np.sum((probs - target_vec) ** 2)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
    
    def clip_gradient(self, grad: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Clip gradient by global norm."""
        grad_norm = np.linalg.norm(grad)
        if grad_norm > self.max_grad_norm:
            return grad * (self.max_grad_norm / grad_norm), grad_norm, True
        return grad, grad_norm, False
    
    def train_step(self, inputs, target_idx: int, temperature: float = 1.0,
                   noise_scale: float = 0.0) -> Tuple[float, np.ndarray, Dict[str, float]]:
        """Single training step.
        
        Args:
            inputs: input values for the model
            target_idx: index of correct class
            temperature: softmax temperature
            noise_scale: gradient noise scale (0 = no noise)
            
        Returns:
            loss: scalar loss value
            probs: probability vector
            grad_norms: dict of gradient norms per parameter
        """
        # Forward pass
        outputs = self.model.forward(inputs)
        probs = self.softmax(outputs, temperature)
        
        # Compute loss
        loss = self.compute_loss(probs, target_idx)
        
        # Backward pass
        if self.loss_type == 'cross_entropy':
            grads = self.model.backward(probs, target_idx, temperature)
        elif self.loss_type == 'mse':
            target_vec = np.zeros(len(probs))
            target_vec[target_idx] = 1.0
            grads = self.model.backward_mse(probs, target_vec, temperature)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # Get current params
        params = self.model.get_params()
        grad_norms = {}
        clipped = False
        
        # Process each parameter group (skip frozen params)
        trainable_params = [name for name in params if name not in self.frozen_params]
        
        for name in trainable_params:
            g = grads[name]
            
            # Gradient clipping
            g, grad_norm, was_clipped = self.clip_gradient(g)
            grad_norms[name] = grad_norm
            clipped = clipped or was_clipped
            
            # Add gradient noise
            if noise_scale > 0:
                noise = np.random.randn(*g.shape) * noise_scale * (np.abs(g).mean() + 1e-8)
                g = g + noise
            
            grads[name] = g
        
        # Optimizer step
        self.t += 1
        lr_effective = {}
        
        for name in trainable_params:
            g = grads[name]
            lr_param = self.lr_dict.get(name, self.lr)
            lr_effective[name] = lr_param
            
            if self.optimizer_type == 'adam':
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (g ** 2)
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)
                params[name] = params[name] - lr_param * m_hat / (np.sqrt(v_hat) + self.eps)
            elif self.optimizer_type == 'sgd':
                params[name] = params[name] - lr_param * g
            else:
                raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")
        
        self.model.set_params(params)
        
        return loss, probs, grad_norms
    
    def compute_accuracy(self, probs: np.ndarray, target_idx: int) -> bool:
        """Check if prediction is correct."""
        return np.argmax(probs) == target_idx
    
    def reset_optimizer_state(self):
        """Reset Adam momentum variables."""
        self.t = 0
        for name in self.m:
            self.m[name] = np.zeros_like(self.m[name])
            self.v[name] = np.zeros_like(self.v[name])
    
    def freeze_params(self, param_names: List[str]):
        """Freeze parameters (stop updating them)."""
        self.frozen_params.update(param_names)
    
    def unfreeze_params(self, param_names: List[str]):
        """Unfreeze parameters (resume updating them)."""
        self.frozen_params.difference_update(param_names)
    
    def get_trainable_params(self) -> List[str]:
        """Return list of currently trainable parameter names."""
        return [name for name in self.model.get_param_shapes().keys() 
                if name not in self.frozen_params]


def run_training(trainer: UnifiedTrainer,
                 input_data,
                 n_classes: int,
                 num_batches: int,
                 batch_size: int,
                 T_start: float = 1.0,
                 T_end: float = 0.2,
                 T_decay: float = 0.99,
                 noise_start: float = 1.0,
                 noise_end: float = 0.0,
                 noise_decay: float = 0.9,
                 print_every: int = 50,
                 history: 'TrainingHistory' = None,
                 verbose: bool = True) -> 'TrainingHistory':
    """
    Run a complete training loop.
    
    Args:
        trainer: UnifiedTrainer instance
        input_data: object with get_next_training_sample(class_idx) method
        n_classes: number of classes
        num_batches: number of batches to train
        batch_size: samples per batch
        T_start: initial softmax temperature
        T_end: final softmax temperature
        T_decay: temperature decay rate per batch
        noise_start: initial gradient noise scale
        noise_end: final gradient noise scale
        noise_decay: noise decay rate per batch
        print_every: print diagnostics every N batches
        history: TrainingHistory instance (created if None)
        verbose: whether to print progress
        
    Returns:
        TrainingHistory with recorded metrics
    """
    import time
    import random
    
    if history is None:
        history = TrainingHistory(n_classes)
    
    start_time = time.time()
    
    if verbose:
        print(f"Starting training: {num_batches} batches, batch_size={batch_size}")
        print("=" * 80)
    
    for batch in range(num_batches):
        # Annealing schedules
        temperature = max(T_end, T_start * (T_decay ** batch))
        noise_scale = max(noise_end, noise_start * (noise_decay ** batch))
        
        batch_loss = 0.0
        batch_correct = 0
        batch_valid = 0
        batch_grad_norms = None
        
        for sample in range(batch_size):
            target_idx = random.randrange(n_classes)
            inputs = input_data.get_next_training_sample(target_idx)
            
            try:
                loss, probs, grad_norms = trainer.train_step(
                    inputs=np.array(inputs).flatten(),
                    target_idx=target_idx,
                    temperature=temperature,
                    noise_scale=noise_scale
                )
                
                # Check for numerical issues
                if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                    if verbose:
                        print(f"  Warning: Invalid probs at batch {batch}, sample {sample}")
                    continue
                
                batch_loss += loss
                history.record_sample_loss(target_idx, loss)
                batch_correct += int(trainer.compute_accuracy(probs, target_idx))
                batch_valid += 1
                batch_grad_norms = grad_norms
                
            except Exception as e:
                if verbose:
                    print(f"  Warning: Training step failed at batch {batch}, sample {sample}: {e}")
                continue
        
        # Skip if no valid samples
        if batch_valid == 0:
            if verbose:
                print(f"  Batch {batch}: No valid samples, skipping")
            continue
        
        # Get param stats
        params = trainer.model.get_params()
        param_values = np.concatenate([p.flatten() for p in params.values()])
        param_stats = (param_values.min(), param_values.max(), param_values.mean())
        
        # Record batch metrics
        history.record_batch(
            avg_loss=batch_loss / batch_valid,
            accuracy=batch_correct / batch_valid,
            grad_norms=batch_grad_norms if batch_grad_norms else {},
            temperature=temperature,
            noise_scale=noise_scale,
            param_stats=param_stats
        )
        
        # Print diagnostics
        if verbose and (batch % print_every == 0 or batch == num_batches - 1):
            recent_loss = history.get_recent_avg('loss')
            recent_acc = history.get_recent_avg('accuracy')
            
            grad_str = ""
            if batch_grad_norms:
                if len(batch_grad_norms) == 1:
                    grad_str = f"GradNorm: {list(batch_grad_norms.values())[0]:.2e}"
                else:
                    grad_str = " ".join([f"{k}:{v:.2e}" for k, v in batch_grad_norms.items()])
            
            print(f"Batch {batch:4d}/{num_batches} | "
                  f"Loss: {batch_loss/batch_valid:.4f} (avg: {recent_loss:.4f}) | "
                  f"Acc: {batch_correct/batch_valid:.1%} (avg: {recent_acc:.1%}) | "
                  f"{grad_str} | "
                  f"T: {temperature:.2f}")
    
    training_time = time.time() - start_time
    
    if verbose:
        print("=" * 80)
        history.print_summary(training_time)
    
    return history


def run_training_crn(trainer: UnifiedTrainer,
                     input_data,
                     n_classes: int,
                     num_batches: int,
                     batch_size: int,
                     T_start: float = 1.0,
                     T_end: float = 0.2,
                     T_decay: float = 0.99,
                     noise_start: float = 5.0,
                     noise_end: float = 0.0,
                     noise_decay: float = 0.99,
                     print_every: int = 50,
                     history: 'TrainingHistory' = None,
                     verbose: bool = True) -> 'TrainingHistory':
    """
    Run training loop for CRN models (handles potential ODE integration failures).
    
    Same interface as run_training but with CRN-specific defaults and error handling.
    """
    import time
    import random
    
    if history is None:
        history = TrainingHistory(n_classes)
    
    start_time = time.time()
    
    if verbose:
        model_type = "CRN"
        if hasattr(trainer.model, 'forward_method'):
            model_type = f"CRN ({trainer.model.forward_method})"
        print(f"Starting {model_type} training: {num_batches} batches, batch_size={batch_size}")
        print("=" * 80)
    
    for batch in range(num_batches):
        # Annealing schedules
        temperature = max(T_end, T_start * (T_decay ** batch))
        noise_scale = max(noise_end, noise_start * (noise_decay ** batch))
        
        batch_loss = 0.0
        batch_correct = 0
        batch_valid = 0
        batch_grad_norms = None
        
        for sample in range(batch_size):
            target_idx = random.randrange(n_classes)
            inputs = input_data.get_next_training_sample(target_idx)
            
            try:
                loss, probs, grad_norms = trainer.train_step(
                    inputs=inputs,  # CRN models handle input format internally
                    target_idx=target_idx,
                    temperature=temperature,
                    noise_scale=noise_scale
                )
                
                # Check for numerical issues
                if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                    if verbose:
                        print(f"  Warning: Invalid probs at batch {batch}, sample {sample}")
                    continue
                
                # Check for invalid gradients
                has_invalid_grad = any(
                    np.any(np.isnan(g)) or np.any(np.isinf(g)) 
                    for g in grad_norms.values()
                )
                if has_invalid_grad:
                    if verbose:
                        print(f"  Warning: Invalid gradient at batch {batch}, sample {sample}")
                    continue
                
                batch_loss += loss
                history.record_sample_loss(target_idx, loss)
                batch_correct += int(trainer.compute_accuracy(probs, target_idx))
                batch_valid += 1
                batch_grad_norms = grad_norms
                
            except Exception as e:
                if verbose:
                    print(f"  Warning: Training step failed at batch {batch}, sample {sample}: {e}")
                continue
        
        # Skip if no valid samples
        if batch_valid == 0:
            if verbose:
                print(f"  Batch {batch}: No valid samples, skipping")
            continue
        
        # Get param stats (for CRN, show rate range)
        params = trainer.model.get_params()
        if 'log_rates' in params:
            rates = np.exp(params['log_rates'])
            param_stats = (rates.min(), rates.max(), rates.mean())
        else:
            param_values = np.concatenate([p.flatten() for p in params.values()])
            param_stats = (param_values.min(), param_values.max(), param_values.mean())
        
        # Record batch metrics
        history.record_batch(
            avg_loss=batch_loss / batch_valid,
            accuracy=batch_correct / batch_valid,
            grad_norms=batch_grad_norms if batch_grad_norms else {},
            temperature=temperature,
            noise_scale=noise_scale,
            param_stats=param_stats
        )
        
        # Print diagnostics
        if verbose and (batch % print_every == 0 or batch == num_batches - 1):
            recent_loss = history.get_recent_avg('loss')
            recent_acc = history.get_recent_avg('accuracy')
            
            grad_str = ""
            if batch_grad_norms:
                grad_str = " ".join([f"{k[:6]}:{v:.2e}" for k, v in batch_grad_norms.items()])
            
            pmin, pmax, _ = param_stats
            print(f"Batch {batch:4d}/{num_batches} | "
                  f"Loss: {batch_loss/batch_valid:.4f} (avg: {recent_loss:.4f}) | "
                  f"Acc: {batch_correct/batch_valid:.1%} (avg: {recent_acc:.1%}) | "
                  f"{grad_str} | "
                  f"Rates: [{pmin:.2e}, {pmax:.2e}]")
    
    training_time = time.time() - start_time
    
    if verbose:
        print("=" * 80)
        history.print_summary(training_time)
    
    return history


class GraphComputation:
    """Computational graph for reaction networks using NumPy."""
    
    def __init__(self, G, input_nodes, output_nodes):
        self.G = G
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.topo_order = list(nx.topological_sort(G))
        self.edges = list(G.edges)
        self.nodes = list(G.nodes)
        self.n_nodes = len(self.nodes)
        self.n_edges = len(self.edges)
        
        # Create node index mappings
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.edge_to_idx = {e: i for i, e in enumerate(self.edges)}
        
        # Pre-compute indices for fast access
        self.input_idxs = np.array([self.node_to_idx[n] for n in input_nodes])
        self.output_idxs = np.array([self.node_to_idx[n] for n in output_nodes])
        
        # Pre-compute NON-INPUT nodes in topological order
        input_set = set(input_nodes)
        self.compute_node_idxs = [
            self.node_to_idx[n] for n in self.topo_order if n not in input_set
        ]
        
        # Pre-compute predecessor structure
        self._build_predecessor_lists()
    
    def _build_predecessor_lists(self):
        """Pre-compute predecessor indices for each node."""
        self.pred_info = {}  # node_idx -> list of (pred_node_idx, edge_idx)
        
        for node in self.nodes:
            node_idx = self.node_to_idx[node]
            preds = list(self.G.predecessors(node))
            self.pred_info[node_idx] = [
                (self.node_to_idx[pred], self.edge_to_idx[(pred, node)])
                for pred in preds
            ]

    def build_r_n_maps(self, r_n):
        """Build index mappings from reaction network."""
        self.node_f_idx = np.zeros(self.n_nodes, dtype=np.int32)
        self.node_r_idx = np.zeros(self.n_nodes, dtype=np.int32)
        self.edge_f_idx = np.zeros(self.n_edges, dtype=np.int32)
        self.edge_r_idx = np.zeros(self.n_edges, dtype=np.int32)

        for (i, reaction) in enumerate(r_n.reactions):
            src = r_n.all_complexes[reaction[0]].split('+')
            dst = r_n.all_complexes[reaction[1]].split('+')
            num_src, num_dst = len(src), len(dst)
            
            # Unimolecular reactions: Xs <-> X
            if num_src == 1 and num_dst == 1:
                if src[0].endswith('s'):
                    node = src[0][:-1]
                    if node in self.node_to_idx:
                        self.node_f_idx[self.node_to_idx[node]] = i
                if dst[0].endswith('s'):
                    node = dst[0][:-1]
                    if node in self.node_to_idx:
                        self.node_r_idx[self.node_to_idx[node]] = i

            # Bimolecular reactions: A+Bs -> A+B or A+B -> A+Bs
            if num_src == 2 and num_dst == 2:
                src_set, dst_set = set(src), set(dst)
                common = src_set & dst_set
                
                if len(common) == 1:
                    upstream_node = common.pop()
                    src_only = (src_set - {upstream_node}).pop()
                    dst_only = (dst_set - {upstream_node}).pop()
                    
                    if src_only.endswith('s') and not dst_only.endswith('s'):
                        downstream_node = dst_only
                        is_forward = True
                    elif dst_only.endswith('s') and not src_only.endswith('s'):
                        downstream_node = src_only
                        is_forward = False
                    else:
                        continue
                    
                    edge_key = (upstream_node, downstream_node)
                    if edge_key in self.edge_to_idx:
                        edge_idx = self.edge_to_idx[edge_key]
                        if is_forward:
                            self.edge_f_idx[edge_idx] = i
                        else:
                            self.edge_r_idx[edge_idx] = i

        self.species_node_map = {}
        self.num_species = len(r_n.species_names)
        for node in self.nodes:
            self.species_node_map[node] = r_n.species_names.index(node)
            if node[0] == 'S':
                self.species_node_map[node+'s'] = r_n.species_names.index(node+'s')



    def forward(self, rates, l0):
        """
        Compute steady-state node values.
        
        Args:
            rates: (n_reactions,) array of rate constants
            l0: (n_nodes,) array of conservation constants
            input_vals: (n_inputs,) array of input values
            
        Returns:
            (n_outputs,) array of output node values
        """
        rates = np.asarray(rates)
        l0 = np.asarray(l0)
        # Initialize node values
        node_values = np.zeros(self.n_nodes)
        node_values[self.input_idxs] = l0[self.input_idxs]
        # Get rate parameters via indexing
        node_kf = rates[self.node_f_idx]
        node_kr = rates[self.node_r_idx]
        edge_kf = rates[self.edge_f_idx]
        edge_kr = rates[self.edge_r_idx]
        
        # Process nodes in topological order (skip inputs)
        for node_idx in self.compute_node_idxs:
            kf_node = node_kf[node_idx]
            kr_node = node_kr[node_idx]
            numerator = kf_node
            denominator = kf_node + kr_node
            
            # Sum contributions from predecessors
            for pred_idx, edge_idx in self.pred_info[node_idx]:
                kf_edge = edge_kf[edge_idx]
                kr_edge = edge_kr[edge_idx]
                pred_val = node_values[pred_idx]
                
                numerator += kf_edge * pred_val
                denominator += (kf_edge + kr_edge) * pred_val
            
            node_values[node_idx] = l0[node_idx] * numerator / (denominator + 1e-10)

        # Recover full concentrations
        C_full = np.zeros(self.num_species)
        for (i, node) in enumerate(self.nodes):
            C_full[self.species_node_map[node]] = node_values[i]
            if node[0] == 'S':
                C_full[self.species_node_map[node+'s']] = l0[i] - node_values[i]
        
        return C_full
    
    def loss(self, rates, l0, input_vals, targets):
        """MSE loss between predictions and targets."""
        preds = self.forward(rates, l0, input_vals)
        return np.mean((preds - targets) ** 2)


# ============== MLP IMPLEMENTATION ==============
class SimpleMLP:
    """
    Simple MLP with manual backpropagation for full control over gradients.
    """
    def __init__(self, layer_sizes, activation='relu'):
        """
        layer_sizes: list of ints, e.g. [input_dim, hidden1, hidden2, n_classes]
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.activation = activation
        
        # Initialize weights and biases (Xavier initialization)
        self.weights = []
        self.biases = []
        for i in range(self.n_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            W = np.random.randn(fan_in, fan_out) * std
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)
    
    def _activation(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            return x  # linear
    
    def _activation_derivative(self, x):
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return s * (1 - s)
        else:
            return np.ones_like(x)
    
    def forward(self, x):
        """Forward pass, storing activations for backprop."""
        self.activations = [x]
        self.pre_activations = []
        
        for i in range(self.n_layers):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.pre_activations.append(z)
            
            if i < self.n_layers - 1:
                # Hidden layer: apply activation
                a = self._activation(z)
            else:
                # Output layer: no activation (we apply softmax separately)
                a = z
            self.activations.append(a)
        
        return self.activations[-1]  # logits
    
    def softmax(self, logits, temperature=1.0):
        """Softmax with temperature."""
        scaled = logits / temperature
        shifted = scaled - np.max(scaled)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals)
    
    def backward(self, probs, target_idx, temperature=1.0):
        """
        Backward pass for cross-entropy loss with softmax output.
        Returns gradients for weights and biases.
        """
        grad_weights = []
        grad_biases = []
        
        # Gradient of cross-entropy + softmax: (p - y) / temperature
        # where y is one-hot target
        delta = probs.copy()
        delta[target_idx] -= 1.0
        delta /= temperature
        
        # Backpropagate through layers
        for i in range(self.n_layers - 1, -1, -1):
            # Gradient w.r.t. weights and biases
            a_prev = self.activations[i]
            dW = np.outer(a_prev, delta)
            db = delta.copy()
            
            grad_weights.insert(0, dW)
            grad_biases.insert(0, db)
            
            if i > 0:
                # Backprop through weights
                delta = self.weights[i] @ delta
                # Backprop through activation
                delta = delta * self._activation_derivative(self.pre_activations[i - 1])
        
        return grad_weights, grad_biases
    
    def backward_mse(self, probs, target_vec, temperature=1.0):
        """
        Backward pass for MSE loss with softmax output.
        """
        grad_weights = []
        grad_biases = []
        
        # d(MSE)/d(logits) via chain rule through softmax
        # d(MSE)/dp = (p - y)
        # dp/dz (softmax jacobian) = diag(p) - outer(p, p)
        diff = probs - target_vec
        softmax_jacobian = (np.diag(probs) - np.outer(probs, probs)) / temperature
        delta = softmax_jacobian.T @ diff
        
        for i in range(self.n_layers - 1, -1, -1):
            a_prev = self.activations[i]
            dW = np.outer(a_prev, delta)
            db = delta.copy()
            
            grad_weights.insert(0, dW)
            grad_biases.insert(0, db)
            
            if i > 0:
                delta = self.weights[i] @ delta
                delta = delta * self._activation_derivative(self.pre_activations[i - 1])
        
        return grad_weights, grad_biases
    
    def get_flat_params(self):
        """Flatten all parameters into a single vector."""
        params = []
        for W, b in zip(self.weights, self.biases):
            params.append(W.flatten())
            params.append(b.flatten())
        return np.concatenate(params)
    
    def set_flat_params(self, flat_params):
        """Set parameters from a flat vector."""
        idx = 0
        for i in range(self.n_layers):
            W_shape = self.weights[i].shape
            W_size = np.prod(W_shape)
            self.weights[i] = flat_params[idx:idx + W_size].reshape(W_shape)
            idx += W_size
            
            b_size = self.biases[i].shape[0]
            self.biases[i] = flat_params[idx:idx + b_size]
            idx += b_size
    
    def get_param_count(self):
        """Return total number of parameters."""
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))


# ============== TRAINING HISTORY & DIAGNOSTICS ==============

class TrainingHistory:
    """Track and store training metrics."""
    
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.loss_history = []
        self.accuracy_history = []
        self.grad_norm_history = []  # Can be dict or scalar
        self.temperature_history = []
        self.noise_scale_history = []
        self.loss_by_class = [[] for _ in range(self.n_classes)]
        self.param_stats_history = []  # (min, max, mean) tuples
    
    def record_batch(self, 
                     avg_loss: float,
                     accuracy: float,
                     grad_norms: dict,
                     temperature: float = None,
                     noise_scale: float = None,
                     param_stats: Tuple[float, float, float] = None):
        """Record metrics for a batch."""
        self.loss_history.append(avg_loss)
        self.accuracy_history.append(accuracy)
        self.grad_norm_history.append(grad_norms)
        
        if temperature is not None:
            self.temperature_history.append(temperature)
        if noise_scale is not None:
            self.noise_scale_history.append(noise_scale)
        if param_stats is not None:
            self.param_stats_history.append(param_stats)
    
    def record_sample_loss(self, class_idx: int, loss: float):
        """Record loss for a specific class sample."""
        self.loss_by_class[class_idx].append(loss)
    
    def get_recent_avg(self, metric: str, window: int = 100) -> float:
        """Get recent average of a metric."""
        history = getattr(self, f'{metric}_history', [])
        if len(history) == 0:
            return 0.0
        window = min(window, len(history))
        return np.mean(history[-window:])
    
    def print_summary(self, training_time: float = None):
        """Print training summary statistics."""
        print(f"\n{'='*60}")
        print(f"Training Summary")
        print(f"{'='*60}")
        
        if training_time is not None:
            print(f"Training time: {training_time:.1f} seconds")
        
        if len(self.loss_history) > 0:
            print(f"Final loss (last 100 batches): {self.get_recent_avg('loss'):.4f}")
            print(f"Min loss achieved: {min(self.loss_history):.4f}")
        
        if len(self.accuracy_history) > 0:
            print(f"Final accuracy (last 100 batches): {self.get_recent_avg('accuracy'):.1%}")
            print(f"Best accuracy: {max(self.accuracy_history):.1%}")
        
        if len(self.param_stats_history) > 0:
            pmin, pmax, pmean = self.param_stats_history[-1]
            print(f"Final param range: [{pmin:.2e}, {pmax:.2e}], mean: {pmean:.2e}")


def plot_training_diagnostics(history: TrainingHistory,
                               title: str = "Training Diagnostics",
                               max_grad_norm: float = None,
                               figsize: Tuple[int, int] = (12, 10),
                               smoothing_window: int = None,
                               show: bool = True):
    """
    Plot comprehensive training diagnostics.
    
    Args:
        history: TrainingHistory object with recorded metrics
        title: Overall figure title
        max_grad_norm: Gradient clipping threshold (for reference line)
        figsize: Figure size
        smoothing_window: Window size for smoothing (auto if None)
        show: Whether to call plt.show()
    
    Returns:
        fig, axes: matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    
    n_batches = len(history.loss_history)
    if smoothing_window is None:
        smoothing_window = min(50, n_batches // 10 + 1)
    
    # Determine layout based on available data
    has_grad_norms = len(history.grad_norm_history) > 0
    has_class_loss = any(len(losses) > 0 for losses in history.loss_by_class)
    has_annealing = len(history.temperature_history) > 0 or len(history.noise_scale_history) > 0
    
    n_plots = 2 + int(has_grad_norms) + int(has_class_loss or has_annealing)
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten() if n_plots > 2 else [axes] if n_plots == 1 else axes.flatten()
    
    plot_idx = 0
    
    # ===== Plot 1: Loss over training =====
    ax = axes[plot_idx]
    ax.plot(history.loss_history, alpha=0.3, label='Batch loss', color='blue')
    
    if n_batches >= smoothing_window:
        smoothed = np.convolve(history.loss_history, 
                               np.ones(smoothing_window)/smoothing_window, mode='valid')
        ax.plot(range(smoothing_window-1, n_batches), smoothed, 
                'r-', linewidth=2, label=f'Smoothed (w={smoothing_window})')
    
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1
    
    # ===== Plot 2: Accuracy over training =====
    ax = axes[plot_idx]
    ax.plot(history.accuracy_history, alpha=0.3, label='Batch accuracy', color='green')
    
    if n_batches >= smoothing_window:
        smoothed = np.convolve(history.accuracy_history,
                               np.ones(smoothing_window)/smoothing_window, mode='valid')
        ax.plot(range(smoothing_window-1, n_batches), smoothed,
                'darkgreen', linewidth=2, label=f'Smoothed (w={smoothing_window})')
    
    ax.axhline(y=1/history.n_classes, color='k', linestyle='--', 
               alpha=0.5, label='Random chance')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1
    
    # ===== Plot 3: Gradient norm =====
    if has_grad_norms:
        ax = axes[plot_idx]
        
        # Handle both dict and scalar grad norms
        grad_norms = history.grad_norm_history
        if isinstance(grad_norms[0], dict):
            # Plot each parameter group
            param_names = list(grad_norms[0].keys())
            colors = plt.cm.tab10(np.linspace(0, 1, len(param_names)))
            
            for i, name in enumerate(param_names):
                norms = [g.get(name, 0) for g in grad_norms]
                ax.semilogy(norms, alpha=0.7, color=colors[i], label=name)
        else:
            # Single scalar
            ax.semilogy(grad_norms, alpha=0.7, color='purple')
        
        if max_grad_norm is not None:
            ax.axhline(y=max_grad_norm, color='r', linestyle='--', 
                       alpha=0.7, label='Clip threshold')
        
        ax.set_xlabel('Batch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # ===== Plot 4: Loss by class OR Annealing schedules =====
    if has_class_loss:
        ax = axes[plot_idx]
        colors = plt.cm.tab10(np.linspace(0, 1, history.n_classes))
        
        for k in range(history.n_classes):
            losses = history.loss_by_class[k]
            if len(losses) > 0:
                ax.plot(losses, alpha=0.2, color=colors[k])
                
                class_window = min(20, len(losses) // 5 + 1)
                if len(losses) >= class_window:
                    smoothed = np.convolve(losses, 
                                          np.ones(class_window)/class_window, mode='valid')
                    ax.plot(range(class_window-1, len(losses)), smoothed,
                           color=colors[k], linewidth=2, label=f'Class {k}')
                else:
                    ax.plot(losses, color=colors[k], linewidth=2, label=f'Class {k}')
        
        ax.set_xlabel('Sample (per class)')
        ax.set_ylabel('Loss')
        ax.set_title('Loss by Class')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    elif has_annealing:
        ax = axes[plot_idx]
        
        if len(history.temperature_history) > 0:
            ax.plot(history.temperature_history, 'b-', linewidth=2, label='Temperature')
        
        if len(history.noise_scale_history) > 0:
            ax2 = ax.twinx()
            ax2.plot(history.noise_scale_history, 'orange', linewidth=2, label='Noise scale')
            ax2.set_ylabel('Noise Scale', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
        
        ax.set_xlabel('Batch')
        ax.set_ylabel('Temperature', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_title('Annealing Schedules')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        if len(history.noise_scale_history) > 0:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax.legend()
        
        plot_idx += 1
    
    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, axes


def plot_comparison(histories: Dict[str, TrainingHistory],
                    metric: str = 'loss',
                    title: str = None,
                    smoothing_window: int = 50,
                    figsize: Tuple[int, int] = (10, 6),
                    show: bool = True):
    """
    Plot comparison of multiple training runs.
    
    Args:
        histories: dict mapping run name to TrainingHistory
        metric: 'loss' or 'accuracy'
        title: Plot title
        smoothing_window: Smoothing window size
        figsize: Figure size
        show: Whether to call plt.show()
    
    Returns:
        fig, ax: matplotlib figure and axis
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for i, (name, history) in enumerate(histories.items()):
        data = history.loss_history if metric == 'loss' else history.accuracy_history
        
        if len(data) == 0:
            continue
        
        ax.plot(data, alpha=0.2, color=colors[i])
        
        if len(data) >= smoothing_window:
            smoothed = np.convolve(data, np.ones(smoothing_window)/smoothing_window, mode='valid')
            ax.plot(range(smoothing_window-1, len(data)), smoothed,
                   color=colors[i], linewidth=2, label=name)
        else:
            ax.plot(data, color=colors[i], linewidth=2, label=name)
    
    ax.set_xlabel('Batch')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title or f'Training {metric.capitalize()} Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if metric == 'accuracy':
        ax.set_ylim([0, 1.05])
        n_classes = list(histories.values())[0].n_classes
        ax.axhline(y=1/n_classes, color='k', linestyle='--', alpha=0.5, label='Random')
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, ax

def generate_lognormal_mixture(
    n_classes: int,
    n_samples_per_class: int,
    input_dim: int,
    log_means: list,
    log_variances: list,
    random_state: int = None
):
    if random_state is not None:
        np.random.seed(random_state)
    
    data_list = []
    for c in range(n_classes):
        mu = np.array(log_means[c]).flatten()
        var = log_variances[c]
        if np.isscalar(var):
            cov = var * np.eye(input_dim)
        else:
            cov = np.diag(np.array(var).flatten())
        
        log_samples = np.random.multivariate_normal(mu, cov, n_samples_per_class)
        samples = np.exp(log_samples)
        data_list.append([sample for sample in samples])
    
    return n_classes, data_list


def generate_lognormal_mixture_random_centers(
    n_classes: int,
    n_samples_per_class: int,
    input_dim: int = 1,
    center_variance: float = 1.0,
    log_variance: float = 0.3,
    center_offset: float = 3.0,
    random_state: int = None
):
    """
    Both center_variance and log_variance are TOTAL variances.
    They are divided by input_dim to get per-dimension variance,
    ensuring comparable spread across different dimensionalities.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Scale both by dimension for comparable total variance
    per_dim_center_variance = center_variance / input_dim
    per_dim_log_variance = log_variance / input_dim
    
    log_means = []
    for c in range(n_classes):
        center = np.random.normal(0, np.sqrt(per_dim_center_variance), size=input_dim)
        log_mean = center + center_offset
        log_means.append(log_mean)
    
    log_variances = [per_dim_log_variance] * n_classes
    
    n_classes_out, data_list = generate_lognormal_mixture(
        n_classes=n_classes,
        n_samples_per_class=n_samples_per_class,
        input_dim=input_dim,
        log_means=log_means,
        log_variances=log_variances,
        random_state=None
    )
    
    return n_classes_out, data_list, log_means


def project_data_list(data_list, d):

    projected_data_list = []
    for class_data in data_list:
        projected_class = [sample[:d] for sample in class_data]
        projected_data_list.append(projected_class)
    return projected_data_list


# ============== MULTI-TASK LEARNING ==============

class MultiTaskInputData:
    """Container for multiple classification tasks with task-specific l0 values."""
    
    def __init__(self, tasks: Dict[str, dict]):
        """
        Args:
            tasks: dict mapping task_id -> {
                'input_data': InputData instance,
                'l0': np.array of l0 values,
                'n_classes': int,
                'log_means': list (optional, for reproducibility)
            }
        """
        self.tasks = tasks
        self.task_ids = list(tasks.keys())
        self.n_tasks = len(self.task_ids)
    
    def sample_task(self) -> str:
        """Randomly sample a task id."""
        import random
        return random.choice(self.task_ids)
    
    def get_task(self, task_id: str) -> dict:
        """Get task info by id."""
        return self.tasks[task_id]
    
    def get_l0(self, task_id: str) -> np.ndarray:
        """Get l0 values for a task."""
        return self.tasks[task_id]['l0']
    
    def get_input_data(self, task_id: str):
        """Get InputData for a task."""
        return self.tasks[task_id]['input_data']
    
    def get_next_training_sample(self, task_id: str, class_idx: int):
        """Get next training sample from a specific task."""
        return self.tasks[task_id]['input_data'].get_next_training_sample(class_idx)
    
    def get_n_classes(self, task_id: str) -> int:
        """Get number of classes for a task."""
        return self.tasks[task_id]['n_classes']


def generate_multitask_data(
    n_tasks: int,
    n_classes: int,
    n_samples_per_class: int,
    input_dim: int,
    proj_dim: int,
    center_variance: float,
    log_variance: float,
    center_offset: float,
    n_nodes: int,
    NR: int,
    hidden_dim: int,
    input_data_class,
    l0_log_mean: float = 0.0,
    l0_log_std: float = 1.0,
    base_seed: int = None
) -> MultiTaskInputData:
    """
    Generate multi-task data with random l0 values for internal nodes.
    
    Args:
        n_tasks: number of tasks to generate
        n_classes: number of classes per task
        n_samples_per_class: samples per class
        input_dim: original input dimension (before projection)
        proj_dim: projected dimension (= NR)
        center_variance: variance for class center generation
        log_variance: variance within each class
        center_offset: offset for log means
        n_nodes: total number of nodes in graph
        NR: number of input (receptor) nodes
        hidden_dim: number of hidden nodes
        input_data_class: InputData class to use for wrapping data
        l0_log_mean: mean of log(l0) for internal nodes
        l0_log_std: std of log(l0) for internal nodes
        base_seed: random seed for reproducibility
        
    Returns:
        MultiTaskInputData container
    """
    import random as random_module
    
    tasks = {}
    
    for task_idx in range(n_tasks):
        # Set seed for this task if base_seed provided
        task_seed = base_seed + task_idx if base_seed is not None else None
        
        if task_seed is not None:
            np.random.seed(task_seed)
            random_module.seed(task_seed)
        
        # Generate data clouds for this task
        _, data_list, log_means = generate_lognormal_mixture_random_centers(
            n_classes=n_classes,
            n_samples_per_class=n_samples_per_class,
            input_dim=input_dim,
            center_variance=center_variance,
            log_variance=log_variance,
            center_offset=center_offset,
            random_state=task_seed
        )
        
        # Project to lower dimension
        data_list = project_data_list(data_list, d=proj_dim)
        input_data = input_data_class(n_classes, data_list)
        
        # Generate l0 values
        l0 = np.ones(n_nodes)
        
        # Randomize internal nodes only (indices NR to NR + hidden_dim)
        internal_start = NR
        internal_end = NR + hidden_dim
        n_internal = internal_end - internal_start
        
        # Log-normal distribution for l0 at internal nodes
        if n_internal > 0:
            log_l0_internal = np.random.randn(n_internal) * l0_log_std + l0_log_mean
            l0[internal_start:internal_end] = np.exp(log_l0_internal)
        
        # Store task
        task_id = f"task_{task_idx}"
        tasks[task_id] = {
            'input_data': input_data,
            'l0': l0,
            'n_classes': n_classes,
            'log_means': log_means,
            'seed': task_seed,
        }
    
    return MultiTaskInputData(tasks)


def run_training_crn_multitask(
    trainer: UnifiedTrainer,
    multi_task_data: MultiTaskInputData,
    n_classes: int,
    num_batches: int,
    batch_size: int,
    T_start: float = 1.0,
    T_end: float = 0.2,
    T_decay: float = 0.99,
    noise_start: float = 5.0,
    noise_end: float = 0.0,
    noise_decay: float = 0.99,
    print_every: int = 50,
    history: 'TrainingHistory' = None,
    verbose: bool = True
) -> 'TrainingHistory':
    """
    Run training loop for CRN models with multiple tasks.
    
    Each task has its own data distribution and l0 values.
    The model learns shared rates across all tasks.
    
    Args:
        trainer: UnifiedTrainer instance (should have frozen_params=['log_l0'])
        multi_task_data: MultiTaskInputData with task-specific data and l0
        n_classes: number of classes (assumed same for all tasks)
        num_batches: number of batches to train
        batch_size: samples per batch
        T_start: initial softmax temperature
        T_end: final softmax temperature
        T_decay: temperature decay rate per batch
        noise_start: initial gradient noise scale
        noise_end: final gradient noise scale
        noise_decay: noise decay rate per batch
        print_every: print diagnostics every N batches
        history: TrainingHistory instance (created if None)
        verbose: whether to print progress
        
    Returns:
        TrainingHistory with recorded metrics
    """
    import time
    import random
    
    if history is None:
        history = TrainingHistory(n_classes)
    
    start_time = time.time()
    
    if verbose:
        model_type = "CRN"
        if hasattr(trainer.model, 'forward_method'):
            model_type = f"CRN ({trainer.model.forward_method})"
        print(f"Starting {model_type} multi-task training: {num_batches} batches, "
              f"batch_size={batch_size}, n_tasks={multi_task_data.n_tasks}")
        print("=" * 80)
    
    for batch in range(num_batches):
        # Annealing schedules
        temperature = max(T_end, T_start * (T_decay ** batch))
        noise_scale = max(noise_end, noise_start * (noise_decay ** batch))
        
        batch_loss = 0.0
        batch_correct = 0
        batch_valid = 0
        batch_grad_norms = None
        
        for sample in range(batch_size):
            # Sample a task
            task_id = multi_task_data.sample_task()
            
            # Set l0 for this task (context switch)
            trainer.model._default_l0 = multi_task_data.get_l0(task_id).copy()
            
            # Sample class and get input from this task
            target_idx = random.randrange(n_classes)
            inputs = multi_task_data.get_next_training_sample(task_id, target_idx)
            
            try:
                loss, probs, grad_norms = trainer.train_step(
                    inputs=inputs,
                    target_idx=target_idx,
                    temperature=temperature,
                    noise_scale=noise_scale
                )
                
                # Check for numerical issues
                if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                    if verbose:
                        print(f"  Warning: Invalid probs at batch {batch}, sample {sample}")
                    continue
                
                # Check for invalid gradients
                has_invalid_grad = any(
                    np.any(np.isnan(g)) or np.any(np.isinf(g)) 
                    for g in grad_norms.values()
                )
                if has_invalid_grad:
                    if verbose:
                        print(f"  Warning: Invalid gradient at batch {batch}, sample {sample}")
                    continue
                
                batch_loss += loss
                history.record_sample_loss(target_idx, loss)
                batch_correct += int(trainer.compute_accuracy(probs, target_idx))
                batch_valid += 1
                batch_grad_norms = grad_norms
                
            except Exception as e:
                if verbose:
                    print(f"  Warning: Training step failed at batch {batch}, sample {sample}: {e}")
                continue
        
        # Skip if no valid samples
        if batch_valid == 0:
            if verbose:
                print(f"  Batch {batch}: No valid samples, skipping")
            continue
        
        # Get param stats (for CRN, show rate range)
        params = trainer.model.get_params()
        if 'log_rates' in params:
            rates = np.exp(params['log_rates'])
            param_stats = (rates.min(), rates.max(), rates.mean())
        else:
            param_values = np.concatenate([p.flatten() for p in params.values()])
            param_stats = (param_values.min(), param_values.max(), param_values.mean())
        
        # Record batch metrics
        history.record_batch(
            avg_loss=batch_loss / batch_valid,
            accuracy=batch_correct / batch_valid,
            grad_norms=batch_grad_norms if batch_grad_norms else {},
            temperature=temperature,
            noise_scale=noise_scale,
            param_stats=param_stats
        )
        
        # Print diagnostics
        if verbose and (batch % print_every == 0 or batch == num_batches - 1):
            recent_loss = history.get_recent_avg('loss')
            recent_acc = history.get_recent_avg('accuracy')
            
            grad_str = ""
            if batch_grad_norms:
                grad_str = " ".join([f"{k[:6]}:{v:.2e}" for k, v in batch_grad_norms.items()])
            
            pmin, pmax, _ = param_stats
            print(f"Batch {batch:4d}/{num_batches} | "
                  f"Loss: {batch_loss/batch_valid:.4f} (avg: {recent_loss:.4f}) | "
                  f"Acc: {batch_correct/batch_valid:.1%} (avg: {recent_acc:.1%}) | "
                  f"{grad_str} | "
                  f"Rates: [{pmin:.2e}, {pmax:.2e}]")
    
    training_time = time.time() - start_time
    
    if verbose:
        print("=" * 80)
        history.print_summary(training_time)
    
    return history


# class GraphComputationJIT:
#     """Optimized computational graph using JAX arrays and JIT compilation."""
    
#     def __init__(self, G, input_nodes, output_nodes):
#         self.G = G
#         self.input_nodes = input_nodes
#         self.output_nodes = output_nodes
#         self.topo_order = list(nx.topological_sort(G))
#         self.edges = list(G.edges)
#         self.nodes = list(G.nodes)
#         self.n_nodes = len(self.nodes)
#         self.n_edges = len(self.edges)
        
#         # Create node index mappings
#         self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
#         self.edge_to_idx = {e: i for i, e in enumerate(self.edges)}
        
#         # Pre-compute indices for fast access
#         self.input_idxs = jnp.array([self.node_to_idx[n] for n in input_nodes])
#         self.output_idxs = jnp.array([self.node_to_idx[n] for n in output_nodes])
        
#         # Pre-compute NON-INPUT nodes in topological order (avoid conditional in JIT)
#         input_set = set(input_nodes)
#         self.compute_node_idxs = tuple(
#             self.node_to_idx[n] for n in self.topo_order if n not in input_set
#         )
        
#         # Pre-compute predecessor structure as arrays
#         self._build_predecessor_arrays()
        
#     def _build_predecessor_arrays(self):
#         """Pre-compute predecessor indices for vectorized access."""
#         max_preds = max((len(list(self.G.predecessors(n))) for n in self.nodes), default=1)
#         max_preds = max(max_preds, 1)
        
#         self.pred_node_idxs = np.full((self.n_nodes, max_preds), 0, dtype=np.int32)
#         self.pred_edge_idxs = np.full((self.n_nodes, max_preds), 0, dtype=np.int32)
#         self.pred_mask = np.zeros((self.n_nodes, max_preds), dtype=bool)
        
#         for node in self.nodes:
#             node_idx = self.node_to_idx[node]
#             preds = list(self.G.predecessors(node))
            
#             for j, pred in enumerate(preds):
#                 pred_idx = self.node_to_idx[pred]
#                 edge_idx = self.edge_to_idx[(pred, node)]
#                 self.pred_node_idxs[node_idx, j] = pred_idx
#                 self.pred_edge_idxs[node_idx, j] = edge_idx
#                 self.pred_mask[node_idx, j] = True
        
#         self.pred_node_idxs = jnp.array(self.pred_node_idxs)
#         self.pred_edge_idxs = jnp.array(self.pred_edge_idxs)
#         self.pred_mask = jnp.array(self.pred_mask)

#     def build_r_n_maps(self, r_n):
#         """Build index mappings from reaction network."""
#         node_f_idx = np.zeros(self.n_nodes, dtype=np.int32)
#         node_r_idx = np.zeros(self.n_nodes, dtype=np.int32)
#         edge_f_idx = np.zeros(self.n_edges, dtype=np.int32)
#         edge_r_idx = np.zeros(self.n_edges, dtype=np.int32)

#         for (i, reaction) in enumerate(r_n.reactions):
#             src = r_n.all_complexes[reaction[0]].split('+')
#             dst = r_n.all_complexes[reaction[1]].split('+')
#             num_src, num_dst = len(src), len(dst)
            
#             if num_src == 1 and num_dst == 1:
#                 if src[0].endswith('s'):
#                     node = src[0][:-1]
#                     if node in self.node_to_idx:
#                         node_f_idx[self.node_to_idx[node]] = i
#                 if dst[0].endswith('s'):
#                     node = dst[0][:-1]
#                     if node in self.node_to_idx:
#                         node_r_idx[self.node_to_idx[node]] = i

#             if num_src == 2 and num_dst == 2:
#                 src_set, dst_set = set(src), set(dst)
#                 common = src_set & dst_set
                
#                 if len(common) == 1:
#                     upstream_node = common.pop()
#                     src_only = (src_set - {upstream_node}).pop()
#                     dst_only = (dst_set - {upstream_node}).pop()
                    
#                     if src_only.endswith('s') and not dst_only.endswith('s'):
#                         downstream_node = dst_only
#                         is_forward = True
#                     elif dst_only.endswith('s') and not src_only.endswith('s'):
#                         downstream_node = src_only
#                         is_forward = False
#                     else:
#                         continue
                    
#                     edge_key = (upstream_node, downstream_node)
#                     if edge_key in self.edge_to_idx:
#                         edge_idx = self.edge_to_idx[edge_key]
#                         if is_forward:
#                             edge_f_idx[edge_idx] = i
#                         else:
#                             edge_r_idx[edge_idx] = i
        
#         self.node_f_idx = jnp.array(node_f_idx)
#         self.node_r_idx = jnp.array(node_r_idx)
#         self.edge_f_idx = jnp.array(edge_f_idx)
#         self.edge_r_idx = jnp.array(edge_r_idx)
        
#         self._compile_forward()

#     def _compile_forward(self):
#         """Create JIT-compiled forward pass."""
        
#         # Static values (known at compile time)
#         compute_node_idxs = self.compute_node_idxs  # tuple = static
#         input_idxs = self.input_idxs
#         output_idxs = self.output_idxs
#         pred_node_idxs = self.pred_node_idxs
#         pred_edge_idxs = self.pred_edge_idxs
#         pred_mask = self.pred_mask
#         node_f_idx = self.node_f_idx
#         node_r_idx = self.node_r_idx
#         edge_f_idx = self.edge_f_idx
#         edge_r_idx = self.edge_r_idx
#         n_nodes = self.n_nodes
        
#         #@jax.jit
#         def _forward_jit(rates, l0, input_vals):
#             """
#             rates: (n_reactions,) array of rate constants
#             l0: (n_nodes,) array of conservation constants
#             input_vals: (n_inputs,) array of input values
#             """
#             # Initialize node values with inputs
#             node_values = jnp.zeros(n_nodes)
#             node_values = node_values.at[input_idxs].set(input_vals)
            
#             # Get rate parameters via indexing
#             node_kf = rates[node_f_idx]
#             node_kr = rates[node_r_idx]
#             edge_kf = rates[edge_f_idx]
#             edge_kr = rates[edge_r_idx]
            
#             # Process only non-input nodes (static tuple, unrolled by JIT)
#             for node_idx in compute_node_idxs:
#                 kf_node = node_kf[node_idx]
#                 kr_node = node_kr[node_idx]
#                 numerator = kf_node
#                 denominator = kf_node + kr_node
                
#                 # Vectorized predecessor contribution
#                 pred_nodes = pred_node_idxs[node_idx]
#                 pred_edges = pred_edge_idxs[node_idx]
#                 mask = pred_mask[node_idx]
                
#                 pred_vals = jnp.where(mask, node_values[pred_nodes], 0.0)
#                 kf_edges = jnp.where(mask, edge_kf[pred_edges], 0.0)
#                 kr_edges = jnp.where(mask, edge_kr[pred_edges], 0.0)
#                 kt_edges = kf_edges + kr_edges
                
#                 numerator = numerator + jnp.sum(kf_edges * pred_vals)
#                 denominator = denominator + jnp.sum(kt_edges * pred_vals)
                
#                 node_val = l0[node_idx] * numerator / (denominator + 1e-10)
#                 node_values = node_values.at[node_idx].set(node_val)
            
#             return node_values[output_idxs]
        
#         self._forward_jit = _forward_jit
    
#     def forward(self, rates, l0, input_vals):
#         return self._forward_jit(
#             jnp.asarray(rates),
#             jnp.asarray(l0),
#             jnp.asarray(input_vals)
#         )


# class GraphComputation:
#     """Turn a NetworkX graph into a differentiable computational graph."""
    
#     def __init__(self, G, input_nodes, output_nodes):
#         self.G = G
#         self.input_nodes = input_nodes
#         self.output_nodes = output_nodes
#         self.topo_order = list(nx.topological_sort(G))
#         self.edges = list(G.edges)
#         self.nodes = list(G.nodes)
#         self.n_edges = len(self.edges)
#         self.edge_to_idx = {e: i for i, e in enumerate(self.edges)}

#     def build_r_n_maps(self, r_n):
#         self.node_params_f_map = {}
#         self.node_params_r_map = {}
#         self.edge_params_f_map = {}
#         self.edge_params_r_map = {}
#         self.node_params_f = {}
#         self.node_params_r = {}
#         self.edge_params_f = {}
#         self.edge_params_r = {}

#         for (i, reaction) in enumerate(r_n.reactions):
#             src = r_n.all_complexes[reaction[0]].split('+')
#             dst = r_n.all_complexes[reaction[1]].split('+')
#             rate = reaction[2]
#             num_src = len(src)
#             num_dst = len(dst)
            
#             # Unimolecular reactions: X <-> Xs
#             if num_src == 1 and num_dst == 1:
#                 if src[0].endswith('s'):
#                     node = src[0][:-1]  # remove 's'
#                     #node_params_f[node] = (rate, i)
#                     self.node_params_f_map[node] = i
#                     self.node_params_f[node] = rate
#                 if dst[0].endswith('s'):
#                     node = dst[0][:-1]  # remove 's'
#                     #node_params_r[node] = (rate, i)
#                     self.node_params_r_map[node] = i
#                     self.node_params_r[node] = rate

#             # Bimolecular reactions: A+Bs -> A+B or A+B -> A+Bs
#             if num_src == 2 and num_dst == 2:
#                 src_set = set(src)
#                 dst_set = set(dst)
                
#                 # Find the species that appears on both sides (upstream/catalyst)
#                 common = src_set & dst_set
                
#                 if len(common) == 1:
#                     upstream_node = common.pop()
                    
#                     # Find the species that changes (has 's' on one side)
#                     src_only = (src_set - {upstream_node}).pop()
#                     dst_only = (dst_set - {upstream_node}).pop()
                    
#                     # Determine downstream node and direction
#                     if src_only.endswith('s') and not dst_only.endswith('s'):
#                         # Xs -> X : forward activation, 's' on left
#                         downstream_node = dst_only
#                         is_forward = True
#                     elif dst_only.endswith('s') and not src_only.endswith('s'):
#                         # X -> Xs : reverse reaction, 's' on right
#                         downstream_node = src_only
#                         is_forward = False
#                     else:
#                         print(f"Warning: Unexpected reaction pattern at {i}: {src} -> {dst}")
#                         continue
                    
#                     edge_key = (upstream_node, downstream_node)
                    
#                     if is_forward:
#                         self.edge_params_f_map[edge_key] = i
#                         self.edge_params_f[edge_key] = rate
#                     else:
#                         self.edge_params_r_map[edge_key] = i
#                         self.edge_params_r[edge_key] = rate
                    
#                     # print(f"Reaction {i}: {'+'.join(src)} -> {'+'.join(dst)}")
#                     # print(f"  Upstream: {upstream_node}, Downstream: {downstream_node}, Forward: {is_forward}")
#                 else:
#                     # Both species change or neither - different reaction type
#                     print(f"Reaction {i}: Non-catalytic bimolecular: {'+'.join(src)} -> {'+'.join(dst)}")

#     def build_node_params_l0(self, l0):
#         self.node_params_l0 = {}
#         for (i, node) in enumerate(self.nodes):
#             self.node_params_l0[node] = l0[i]
    
#     def build_rate_params(self, rates):
#         for key in self.node_params_f_map:
#             self.node_params_f[key] = rates[self.node_params_f_map[key]]
#         for key in self.node_params_r_map:
#             self.node_params_r[key] = rates[self.node_params_r_map[key]]
#         for key in self.edge_params_f_map:
#             self.edge_params_f[key] = rates[self.edge_params_f_map[key]]
#         for key in self.edge_params_r_map:
#             self.edge_params_r[key] = rates[self.edge_params_r_map[key]]
            

#     def forward(self, inputs):
#         """
#         params: array of shape (n_edges,) - one parameter per edge
#         inputs: dict mapping input_node -> value
#         """
#         node_values = {n: inputs[n] for n in self.input_nodes}
        
#         for node in self.topo_order:
#             if node in self.input_nodes:
#                 continue
            
#             # Aggregate incoming edges
#             numerator = self.node_params_f[node]
#             denominator = self.node_params_f[node] + self.node_params_r[node]
#             for pred in self.G.predecessors(node):
#                 edge_idx = self.edge_to_idx[(pred, node)]
#                 edge = self.edges[edge_idx]
#                 kf_couple = self.edge_params_f[edge]
#                 kr_couple = self.edge_params_r[edge]
#                 kt_couple = kf_couple + kr_couple
#                 numerator += kf_couple * node_values[pred]
#                 denominator += kt_couple * node_values[pred]
            
#             # Node activation (customize as needed)
#             node_values[node] = self.node_params_l0[node] * numerator / denominator
        
#         return jnp.array([node_values[n] for n in self.output_nodes])

    
#     def loss(self, params, inputs, targets):
#         preds = self.forward(params, inputs)
#         return jnp.mean((preds - targets) ** 2)