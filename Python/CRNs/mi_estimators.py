"""
Mutual Information estimation for binary stochastic codes.

This module provides estimators for I(R; z) where:
- R: continuous input from data distribution
- z: binary codes sampled from z ~ Bernoulli(h(R))
- h: activation parameters in [0,1]^d from the model

Supports fully analytical computation (no z sampling needed).
"""

import numpy as np
import itertools
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
from scipy.special import logsumexp


class MutualInformationEstimator(ABC):
    """
    Abstract base class for I(R; z) estimation.
    
    Subclasses implement different estimation strategies.
    """
    
    @abstractmethod
    def estimate(self, h_samples: np.ndarray) -> float:
        """
        Estimate I(R; z) from Bernoulli parameter samples.
        
        Args:
            h_samples: Bernoulli parameters, shape (n_samples, code_dim)
                      Each h[i] parameterizes p(z | R_i)
        
        Returns:
            mi_estimate: Estimated mutual information I(R; z)
        """
        pass
    
    @abstractmethod
    def compute_gradients(self, h_samples: np.ndarray) -> np.ndarray:
        """
        Compute ∇_h I(R; z) for each sample.
        
        Args:
            h_samples: Bernoulli parameters, shape (n_samples, code_dim)
        
        Returns:
            gradients: ∇_h I(R; z), shape (n_samples, code_dim)
        """
        pass
    
    @staticmethod
    def bernoulli_entropy(h: np.ndarray) -> np.ndarray:
        """
        Entropy of Bernoulli distribution: H(Bernoulli(h)).
        
        H = -h log h - (1-h) log(1-h)
        
        Args:
            h: Bernoulli parameters, any shape
        
        Returns:
            entropy: Same shape as h
        """
        h = np.clip(h, 1e-8, 1 - 1e-8)
        return -h * np.log(h) - (1 - h) * np.log(1 - h)
    
    @staticmethod
    def grad_bernoulli_entropy(h: np.ndarray) -> np.ndarray:
        """
        Gradient of Bernoulli entropy: ∇_h H(Bernoulli(h)).
        
        ∇_h H = -log h + log(1-h) = log((1-h)/h)
        
        Args:
            h: Bernoulli parameters, any shape
        
        Returns:
            gradient: Same shape as h
        """
        h = np.clip(h, 1e-8, 1 - 1e-8)
        return np.log((1 - h) / h)


class AnalyticalBernoulliMI(MutualInformationEstimator):
    """
    Fully analytical MI estimation for binary codes.
    
    I(R; z) = H(z) - H(z|R)
    
    Both terms computed analytically from h samples:
    - H(z|R): E_R[H(Bernoulli(h(R)))] - analytical
    - H(z): Enumerate all z ∈ {0,1}^d, compute p(z) analytically
    
    No sampling of z required!
    """
    
    def __init__(self, max_code_dim: int = 15, use_vectorized: bool = True):
        """
        Args:
            max_code_dim: Maximum code dimension for enumeration (2^d states)
            use_vectorized: Use vectorized computation (faster but more memory)
        """
        self.max_code_dim = max_code_dim
        self.use_vectorized = use_vectorized
    
    def estimate(self, h_samples: np.ndarray) -> float:
        """
        I(R; z) = H(z) - H(z|R)
        
        Args:
            h_samples: shape (n_samples, code_dim)
        
        Returns:
            mi_estimate: Mutual information
        """
        n_samples, code_dim = h_samples.shape
        
        if code_dim > self.max_code_dim:
            raise ValueError(
                f"Code dimension {code_dim} exceeds max {self.max_code_dim}. "
                f"Use MeanFieldMI for high-dimensional codes."
            )
        
        # H(z|R) = E_R[H(Bernoulli(h))]
        H_z_given_R = np.mean(self.bernoulli_entropy(h_samples).sum(axis=1))
        
        # H(z) via enumeration
        H_z = self._marginal_entropy_analytical(h_samples)
        
        return H_z - H_z_given_R
    
    def compute_gradients(self, h_samples: np.ndarray) -> np.ndarray:
        """
        ∇_h I(R; z) = ∇_h H(z) - ∇_h H(z|R)
        
        Note: H(z|R) = (1/N) Σ_n H(z|R_n), so gradient is scaled by 1/N.
        
        Args:
            h_samples: shape (n_samples, code_dim)
        
        Returns:
            gradients: shape (n_samples, code_dim)
        """
        n_samples, code_dim = h_samples.shape
        
        # ∇_{h_n} H(z|R) = (1/N) ∇_{h_n} H(z|R_n) = (1/N) log((1-h_n)/h_n)
        grad_H_conditional = self.grad_bernoulli_entropy(h_samples) / n_samples
        
        # ∇_h H(z) - each sample contributes to marginal
        grad_H_marginal = self._marginal_entropy_gradient(h_samples)
        
        # ∇_h I = ∇_h H(z) - ∇_h H(z|R)
        return grad_H_marginal - grad_H_conditional
    
    def _marginal_entropy_analytical(self, h_samples: np.ndarray) -> float:
        """
        Compute H(z) by enumerating all z ∈ {0,1}^d.
        
        p(z) = (1/N) Σ_n p(z | h_n)
        H(z) = -Σ_z p(z) log p(z)
        """
        if self.use_vectorized:
            return self._marginal_entropy_vectorized(h_samples)
        else:
            return self._marginal_entropy_loop(h_samples)
    
    def _marginal_entropy_vectorized(self, h_samples: np.ndarray) -> float:
        """
        Vectorized computation of H(z).
        
        Fast but memory-intensive: O(N × 2^d).
        """
        N, code_dim = h_samples.shape
        
        # Generate all 2^d binary codes
        all_codes = np.array(list(itertools.product([0, 1], repeat=code_dim)))
        
        # Broadcast for vectorized computation
        h_expanded = h_samples[:, None, :]      # (N, 1, d)
        z_expanded = all_codes[None, :, :]      # (1, 2^d, d)
        
        # Clip for numerical stability
        h_safe = np.clip(h_expanded, 1e-8, 1 - 1e-8)
        
        # log p(z|h_n) = Σ_i [z_i log h_i + (1-z_i) log(1-h_i)]
        log_p_z_given_h = (
            z_expanded * np.log(h_safe) + 
            (1 - z_expanded) * np.log(1 - h_safe)
        ).sum(axis=2)  # (N, 2^d)
        
        # p(z) = (1/N) Σ_n p(z|h_n)
        p_z_given_h = np.exp(log_p_z_given_h)
        p_z = p_z_given_h.mean(axis=0)  # (2^d,)
        
        # Normalize and compute entropy
        p_z = p_z / p_z.sum()
        entropy = -np.sum(p_z * np.log(p_z + 1e-10))
        
        return entropy
    
    def _marginal_entropy_loop(self, h_samples: np.ndarray) -> float:
        """
        Loop-based computation of H(z).
        
        Memory-efficient: O(d) but slower.
        """
        N, code_dim = h_samples.shape
        all_codes = np.array(list(itertools.product([0, 1], repeat=code_dim)))
        
        # Precompute log probabilities
        log_h = np.log(np.clip(h_samples, 1e-8, 1 - 1e-8))
        log_1_minus_h = np.log(np.clip(1 - h_samples, 1e-8, 1))
        
        log_probs_z = []
        for z in all_codes:
            # log p(z|h_n) for all n
            log_p_z_given_h = (
                z * log_h + (1 - z) * log_1_minus_h
            ).sum(axis=1)  # (N,)
            
            # log p(z) = log[(1/N) Σ_n p(z|h_n)]
            log_p_z = logsumexp(log_p_z_given_h) - np.log(N)
            log_probs_z.append(log_p_z)
        
        log_probs_z = np.array(log_probs_z)
        probs_z = np.exp(log_probs_z)
        probs_z = probs_z / probs_z.sum()
        
        entropy = -np.sum(probs_z * np.log(probs_z + 1e-10))
        return entropy
    
    def _marginal_entropy_gradient(self, h_samples: np.ndarray) -> np.ndarray:
        """
        Compute ∇_{h_n} H(z) for each sample n.
        
        Uses chain rule through mixture:
        ∇_{h_n} H(z) = ∇_{h_n} [-Σ_z p(z) log p(z)]
        
        where p(z) = (1/N) Σ_m p(z|h_m)
        so ∇_{h_n} p(z) = (1/N) ∇_{h_n} p(z|h_n)
        
        CRITICAL: When p(z) is normalized (p_z / p_z.sum()), we need to account
        for this in the gradient. The normalization factor cancels in the entropy
        gradient formula, so we work with unnormalized p(z).
        """
        N, code_dim = h_samples.shape
        all_codes = np.array(list(itertools.product([0, 1], repeat=code_dim)))
        
        # First, compute p(z) for all z (UNNORMALIZED)
        h_safe = np.clip(h_samples, 1e-8, 1 - 1e-8)
        log_h = np.log(h_safe)
        log_1_minus_h = np.log(1 - h_safe)
        
        # Compute p(z) for all codes
        p_z_list = []
        p_z_given_h_all = []  # (2^d, N) - need for gradient
        
        for z in all_codes:
            log_p_z_given_h = (
                z * log_h + (1 - z) * log_1_minus_h
            ).sum(axis=1)  # (N,)
            
            p_z_given_h = np.exp(log_p_z_given_h)  # (N,)
            p_z = p_z_given_h.mean()  # This is (1/N) Σ_m p(z|h_m)
            
            p_z_list.append(p_z)
            p_z_given_h_all.append(p_z_given_h)
        
        p_z = np.array(p_z_list)  # (2^d,)
        Z = p_z.sum()  # Normalization constant (should be ≈1 but may have numerical error)
        p_z_normalized = p_z / Z  # normalized distribution
        p_z_given_h_all = np.array(p_z_given_h_all)  # (2^d, N)
        
        # Now compute gradient for each sample
        gradients = np.zeros((N, code_dim))
        
        for n in range(N):
            for i in range(code_dim):
                # For normalized p(z), we need:
                # ∇_{h_{n,i}} H(z) = -Σ_z [∇_{h_{n,i}} (p(z)/Z)] [1 + log(p(z)/Z)]
                #                  = -Σ_z [(∇_{h_{n,i}} p(z))/Z - p(z)·(∇_{h_{n,i}} Z)/Z^2] [1 + log(p(z)/Z)]
                #
                # Note: ∇_{h_{n,i}} Z = ∇_{h_{n,i}} [Σ_z' p(z')]
                #                      = Σ_z' ∇_{h_{n,i}} p(z')
                #                      = (1/N) Σ_z' ∇_{h_{n,i}} p(z'|h_n)
                #                      = 0 (since Σ_z' p(z'|h_n) = 1)
                #
                # So: ∇_{h_{n,i}} (p(z)/Z) = (∇_{h_{n,i}} p(z)) / Z
                
                grad_sum = 0.0
                
                for z_idx, z in enumerate(all_codes):
                    # ∇_{h_{n,i}} p(z) = (1/N) ∇_{h_{n,i}} p(z|h_n)
                    # ∇_{h_{n,i}} p(z|h_n) = p(z|h_n) [z_i/h_{n,i} - (1-z_i)/(1-h_{n,i})]
                    
                    p_z_given_hn = p_z_given_h_all[z_idx, n]
                    h_ni = h_safe[n, i]
                    z_i = z[i]
                    
                    grad_p_z_given_h = p_z_given_hn * (z_i / h_ni - (1 - z_i) / (1 - h_ni))
                    grad_p_z_unnormalized = grad_p_z_given_h / N
                    
                    # Gradient of normalized p(z)
                    grad_p_z = grad_p_z_unnormalized / Z
                    
                    # Entropy gradient contribution using normalized p(z)
                    if p_z_normalized[z_idx] > 1e-10:
                        grad_sum += grad_p_z * (1 + np.log(p_z_normalized[z_idx]))
                
                gradients[n, i] = -grad_sum
        
        return gradients


class MonteCarloMI(MutualInformationEstimator):
    """
    Monte Carlo MI estimation for medium-dimensional codes.
    
    Uses sampling to estimate p(z) instead of enumeration.
    Good for code_dim in range 15-25 where analytical is too slow
    and mean-field is too inaccurate.
    
    WARNING: Gradient estimator has known bias due to mixture sampling challenges.
    For training, prefer analytical (code_dim ≤ 15) or mean-field (code_dim > 15).
    This estimator is primarily useful for MI evaluation, not optimization.
    
    TODO: Implement unbiased gradient estimator (requires more sophisticated
    importance sampling or Rao-Blackwellization).
    """
    
    def __init__(self, n_samples_per_h: int = 100, use_baseline: bool = True):
        """
        Args:
            n_samples_per_h: Number of z samples to draw from each h
            use_baseline: Whether to use baseline for gradient variance reduction
        """
        self.K = n_samples_per_h
        self.use_baseline = use_baseline
    
    def estimate(self, h_samples: np.ndarray) -> float:
        """
        I(R; z) = H(z) - H(z|R) with MC estimation of H(z).
        
        Args:
            h_samples: shape (n_samples, code_dim)
        
        Returns:
            mi_estimate: Mutual information
        """
        n_samples, code_dim = h_samples.shape
        
        # H(z|R) - exact (no sampling needed)
        H_z_given_R = np.mean(self.bernoulli_entropy(h_samples).sum(axis=1))
        
        # H(z) - Monte Carlo estimation
        H_z = self._marginal_entropy_mc(h_samples)
        
        return H_z - H_z_given_R
    
    def _marginal_entropy_mc(self, h_samples: np.ndarray) -> float:
        """
        Estimate H(z) = -Σ_z p(z) log p(z) via Monte Carlo sampling.
        
        Sample z ~ p(z|h_n) for each h_n, build histogram, compute entropy.
        """
        n_samples, code_dim = h_samples.shape
        
        # Sample K codes from each h_n
        all_z_samples = []
        for h_n in h_samples:
            # Draw K binary codes
            z_samples = (np.random.rand(self.K, code_dim) < h_n).astype(np.uint8)
            all_z_samples.append(z_samples)
        
        all_z_samples = np.vstack(all_z_samples)  # (n_samples * K, code_dim)
        
        # Convert binary vectors to integers for efficient histogram
        # Pack bits into integers (supports up to 64 bits)
        if code_dim <= 64:
            z_integers = self._pack_binary(all_z_samples)
            
            # Build histogram
            unique_codes, counts = np.unique(z_integers, return_counts=True)
            p_z = counts / (n_samples * self.K)
            
            # Compute entropy
            entropy = -np.sum(p_z * np.log(p_z + 1e-10))
        else:
            # For very high dimensions, use tuple hashing (slower but works)
            from collections import Counter
            z_tuples = [tuple(z) for z in all_z_samples]
            counter = Counter(z_tuples)
            counts = np.array(list(counter.values()))
            p_z = counts / (n_samples * self.K)
            entropy = -np.sum(p_z * np.log(p_z + 1e-10))
        
        return entropy
    
    def _pack_binary(self, binary_array: np.ndarray) -> np.ndarray:
        """Convert binary vectors to integers for efficient comparison."""
        n_samples, code_dim = binary_array.shape
        
        if code_dim <= 32:
            # Use 32-bit integers
            powers = 2 ** np.arange(code_dim, dtype=np.uint32)
            return binary_array @ powers
        elif code_dim <= 64:
            # Use 64-bit integers
            powers = 2 ** np.arange(code_dim, dtype=np.uint64)
            return binary_array @ powers
        else:
            raise ValueError(f"code_dim={code_dim} too large for integer packing")
    
    def compute_gradients(self, h_samples: np.ndarray) -> np.ndarray:
        """
        ∇_h I with Monte Carlo gradients using score function estimator.
        
        Uses REINFORCE-style gradient with optional baseline for variance reduction.
        
        Args:
            h_samples: shape (n_samples, code_dim)
        
        Returns:
            gradients: shape (n_samples, code_dim)
        """
        n_samples, code_dim = h_samples.shape
        
        # ∇_{h_n} H(z|R) = (1/N) log((1-h_n)/h_n)
        grad_H_conditional = self.grad_bernoulli_entropy(h_samples) / n_samples
        
        # ∇_{h_n} H(z) - Monte Carlo with score function
        grad_H_marginal = self._marginal_entropy_gradient_mc(h_samples)
        
        return grad_H_marginal - grad_H_conditional
    
    def _marginal_entropy_gradient_mc(self, h_samples: np.ndarray) -> np.ndarray:
        """
        MC gradient of H(z) using importance sampling.
        
        We need: ∇_{h_n} H(z) = -Σ_z [∇_{h_n} p(z)] [1 + log p(z)]
        
        Where p(z) = (1/N) Σ_m p(z|h_m), so:
        ∇_{h_n} p(z) = (1/N) ∇_{h_n} p(z|h_n)
        
        MC approximation: Sample z from each h_m, estimate p(z) from histogram,
        then for each sampled z, compute gradient contribution.
        """
        n_samples, code_dim = h_samples.shape
        h_safe = np.clip(h_samples, 1e-8, 1 - 1e-8)
        
        # Sample K codes from each h_n
        z_samples_per_h = []
        for h_n in h_safe:
            z_samples = (np.random.rand(self.K, code_dim) < h_n).astype(np.uint8)
            z_samples_per_h.append(z_samples)
        
        # Build global histogram for p(z) estimation
        all_z = np.vstack(z_samples_per_h)  # (n_samples * K, code_dim)
        
        # Build histogram and p(z) estimates
        if code_dim <= 64:
            all_z_int = self._pack_binary(all_z)
            unique_codes, counts = np.unique(all_z_int, return_counts=True)
            code_to_count = dict(zip(unique_codes, counts))
            code_to_logp = {
                code: np.log(count / (n_samples * self.K)) 
                for code, count in zip(unique_codes, counts)
            }
        else:
            from collections import Counter
            all_z_tuples = [tuple(z) for z in all_z]
            counter = Counter(all_z_tuples)
            code_to_logp = {
                code: np.log(count / (n_samples * self.K))
                for code, count in counter.items()
            }
        
        # Compute gradients
        gradients = np.zeros((n_samples, code_dim))
        
        for n in range(n_samples):
            h_n = h_safe[n]
            z_samples = z_samples_per_h[n]
            
            # For each code sampled from h_n
            for z_k in z_samples:
                # Get log p(z_k) from histogram
                if code_dim <= 64:
                    z_int = self._pack_binary(z_k.reshape(1, -1))[0]
                    log_p_z = code_to_logp.get(z_int, -20.0)  # Default for unseen
                else:
                    z_tuple = tuple(z_k)
                    log_p_z = code_to_logp.get(z_tuple, -20.0)
                
                # ∇_{h_n} p(z|h_n) = p(z|h_n) * [z/h - (1-z)/(1-h)]
                # But we want: -[∇_{h_n} p(z)] [1 + log p(z)]
                # Where ∇_{h_n} p(z) = (1/N) * ∇_{h_n} p(z|h_n)
                
                # Compute p(z|h_n) for this specific z
                log_p_z_given_hn = np.sum(
                    z_k * np.log(h_n) + (1 - z_k) * np.log(1 - h_n)
                )
                p_z_given_hn = np.exp(log_p_z_given_hn)
                
                # Gradient of log p(z|h_n)
                grad_log_p = z_k / h_n - (1 - z_k) / (1 - h_n)
                
                # Gradient of p(z|h_n)
                grad_p_z_given_hn = p_z_given_hn * grad_log_p
                
                # Gradient contribution (scaled by 1/N and weighted by entropy derivative)
                weight = -(1.0 + log_p_z) / n_samples
                gradients[n] += weight * grad_p_z_given_hn / self.K
        
        return gradients


class MeanFieldMI(MutualInformationEstimator):
    """
    Mean-field approximation for high-dimensional codes.
    
    Assumes independence in marginal: p(z) ≈ ∏_i Bernoulli(μ_i)
    where μ_i = E_R[h_i(R)]
    
    Faster but less accurate than analytical enumeration.
    Use for code_dim > 15.
    """
    
    def estimate(self, h_samples: np.ndarray) -> float:
        """
        I(R; z) ≈ H(z) - H(z|R) with mean-field H(z).
        """
        # H(z|R) - exact
        H_z_given_R = np.mean(self.bernoulli_entropy(h_samples).sum(axis=1))
        
        # H(z) - mean-field approximation
        h_mean = h_samples.mean(axis=0)  # (code_dim,)
        H_z = self.bernoulli_entropy(h_mean).sum()
        
        return H_z - H_z_given_R
    
    def compute_gradients(self, h_samples: np.ndarray) -> np.ndarray:
        """
        ∇_h I with mean-field approximation.
        """
        n_samples, code_dim = h_samples.shape
        
        # ∇_{h_n} H(z|R) = (1/N) log((1-h_n)/h_n)
        grad_H_conditional = self.grad_bernoulli_entropy(h_samples) / n_samples
        
        # ∇_h H(z) - mean-field
        # Each sample contributes (1/N) to mean, so:
        # ∇_{h_n} H(z) = (1/N) ∇_μ H(Bernoulli(μ))
        h_mean = h_samples.mean(axis=0)
        grad_mean_field = self.grad_bernoulli_entropy(h_mean)  # (code_dim,)
        grad_H_marginal = np.tile(grad_mean_field, (n_samples, 1)) / n_samples
        
        return grad_H_marginal - grad_H_conditional


class LogNormalChannelMI(MutualInformationEstimator):
    r"""
    Mutual information for a log-normal channel R -> C.

    Models the downstream activation as
        ln C | R ~ Normal(mu(R), Sigma),
        mu_i = ln(Cbar_i(R)) - 0.5 * Sigma_ii,
    so that E[C|R] = Cbar(R), the mass-action steady state passed in by the
    caller. Sigma is diagonal with per-element, *fixed* noise std sigma_i
    (a modeling hyperparameter, not optimized).

    Because mutual information is invariant under the invertible map
    C = exp(X), we compute everything in log space with X = ln C, a Gaussian
    mixture q(x) = (1/N) sum_n N(x; mu_n, Sigma). The exp() Jacobian terms and
    the common -0.5*Sigma_ii mean shift cancel in I and are irrelevant to the
    gradient, so:

        I(R;C) = I(R;X) = H(X) - H(X|R)

    with the closed-form conditional entropy
        H(X|R) = sum_i ln sigma_i + 0.5 d (1 + ln 2pi)
    and H(X) estimated by nested Monte Carlo over the mixture using
    reparameterized draws with common random numbers (stable gradients).

    The estimator returns I and dI/dCbar (shape (N, d)), where the trainer
    chains dCbar/dC and dC/dtheta exactly as for the Bernoulli channel.
    """

    # Marks the channel type so the trainer knows to feed mean concentrations
    # (Cbar) rather than Bernoulli activation fractions.
    channel_type = 'lognormal'

    def __init__(self,
                 code_dim: int,
                 noise_scale=0.5,
                 n_mc_samples: int = 8,
                 resample_every: int = 1,
                 seed: int = 0,
                 min_conc: float = 1e-12):
        """
        Args:
            code_dim: Dimensionality d of the code (number of activation groups).
            noise_scale: Log-space noise std sigma. Scalar (shared) or array of
                shape (code_dim,) for independent per-element noise. Fixed.
            n_mc_samples: Number of reparameterized draws S per mixture component.
            resample_every: Redraw the common random numbers every this many
                estimate() calls. 1 = redraw each batch; large = fully frozen.
            seed: RNG seed for the common random numbers.
            min_conc: Floor applied to Cbar before taking the log (positivity).
        """
        self.code_dim = int(code_dim)

        sigma = np.atleast_1d(np.asarray(noise_scale, dtype=float))
        if sigma.size == 1:
            sigma = np.full(self.code_dim, float(sigma[0]))
        if sigma.size != self.code_dim:
            raise ValueError(
                f"noise_scale must be scalar or shape ({self.code_dim},), "
                f"got shape {sigma.shape}"
            )
        if np.any(sigma <= 0):
            raise ValueError("noise_scale entries must be positive")
        self.sigma = sigma

        self.S = int(n_mc_samples)
        self.resample_every = int(resample_every)
        self.min_conc = float(min_conc)

        self._rng = np.random.default_rng(seed)
        self._eps = self._rng.standard_normal((self.S, self.code_dim))
        self._call_count = 0

    def _maybe_resample(self):
        """Redraw common random numbers on the configured schedule."""
        if self.resample_every > 0 and (self._call_count % self.resample_every == 0):
            self._eps = self._rng.standard_normal((self.S, self.code_dim))
        self._call_count += 1

    def _means(self, mean_code_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return floored concentrations C (N,d) and log-space means mu (N,d)."""
        C = np.clip(np.asarray(mean_code_batch, dtype=float), self.min_conc, None)
        if C.ndim != 2 or C.shape[1] != self.code_dim:
            raise ValueError(
                f"mean_code_batch must have shape (N, {self.code_dim}), "
                f"got {C.shape}"
            )
        # Common shift -0.5 sigma^2 keeps E[C|R]=Cbar but does not affect MI.
        mu = np.log(C) - 0.5 * self.sigma ** 2
        return C, mu

    def _forward(self, mu: np.ndarray):
        """
        Shared computation of the Gaussian-mixture quantities.

        Returns:
            x:        (N, S, d) reparameterized samples
            logq:     (N, S) log mixture density at each sample
            w:        (N, S, N) responsibilities over mixture components
            log_self: (N, S) self log-density log N(x^{n,s}; mu_n, Sigma)
        """
        N, d = mu.shape
        sigma = self.sigma
        eps = self._eps  # (S, d) common random numbers

        # Reparameterized samples from each component: x^{n,s} = mu_n + sigma * eps^s
        x = mu[:, None, :] + sigma[None, None, :] * eps[None, :, :]  # (N, S, d)

        # Per-component log densities ell[n,s,m] = log N(x^{n,s}; mu_m, Sigma)
        z = (x[:, :, None, :] - mu[None, None, :, :]) / sigma[None, None, None, :]
        quad = -0.5 * np.sum(z ** 2, axis=-1)  # (N, S, N)
        log_norm = -np.sum(np.log(sigma)) - 0.5 * d * np.log(2.0 * np.pi)
        ell = quad + log_norm  # (N, S, N)

        lse = logsumexp(ell, axis=-1)          # (N, S)
        logq = lse - np.log(N)                 # log q(x) = log[(1/N) sum_m N(...)]
        w = np.exp(ell - lse[..., None])       # (N, S, N) responsibilities

        # Self log-density: quad term is -0.5||eps||^2 (independent of mu), so it
        # shares the SAME Monte-Carlo noise as logq. Using it for H(C|R) makes the
        # noise cancel exactly in I = mean(log_self - logq), removing the finite-S
        # bias that a purely analytic H(C|R) would leave behind.
        quad_self = -0.5 * np.sum(eps ** 2, axis=-1)  # (S,)
        log_self = quad_self[None, :] + log_norm      # (N, S) via broadcast

        return x, logq, w, log_self

    def estimate(self, mean_code_batch: np.ndarray) -> float:
        """
        I(R;C) = H(X) - H(X|R) with X = ln C.

        Args:
            mean_code_batch: mean downstream activations Cbar, shape (N, d).

        Returns:
            mi_estimate: mutual information in nats.
        """
        self._maybe_resample()
        _, mu = self._means(mean_code_batch)

        _, logq, _, log_self = self._forward(mu)

        # Matched estimator: I = mean(log_self - logq). The shared MC noise in
        # log_self and logq cancels, so this is unbiased for separated clusters
        # (-> ln K exactly) regardless of S or the frozen eps buffer.
        return float(np.mean(log_self - logq))

    def compute_gradients(self, mean_code_batch: np.ndarray) -> np.ndarray:
        """
        dI/dCbar for each sample.

        Uses the reparameterized gradient of the mixture entropy (density path
        + sample path). H(X|R) is independent of the means, so only H(X)
        contributes. Reuses the currently cached common random numbers so it is
        consistent with the preceding estimate() call.

        Args:
            mean_code_batch: mean downstream activations Cbar, shape (N, d).

        Returns:
            grad_cbar: dI/dCbar, shape (N, d).
        """
        C, mu = self._means(mean_code_batch)
        N, d = mu.shape
        sigma2 = self.sigma ** 2

        x, _, w, _ = self._forward(mu)

        # diff[n,s,m,i] = x^{n,s}_i - mu_{m,i}
        diff = x[:, :, None, :] - mu[None, None, :, :]  # (N, S, N, d)

        # Density path: dens[k,i] = sum_{n,s} w[n,s,k] * (x^{n,s}_i - mu_{k,i}) / sigma_i^2
        dens = np.einsum('nsk,nski->ki', w, diff) / sigma2  # (N, d) indexed by k

        # Sample path (only own component contributes):
        #   samp[k,i] = sum_s (x^{k,s}_i - xbar^{k,s}_i) / sigma_i^2
        xbar = np.einsum('nsm,mi->nsi', w, mu)              # (N, S, d)
        samp = np.sum((x - xbar) / sigma2[None, None, :], axis=1)  # (N, d) indexed by k

        # dI/dmu_k = -(1/(N S)) [ density - sample ]
        grad_mu = -(dens - samp) / (N * self.S)

        # Chain to Cbar: mu_i = ln Cbar_i (+ const), so dmu/dCbar = 1/Cbar.
        grad_cbar = grad_mu / C
        return grad_cbar

    def entropy_breakdown(self, mean_code_batch: np.ndarray) -> Dict[str, float]:
        """Diagnostic helper: return H(X), H(X|R), and I in nats."""
        self._maybe_resample()
        _, mu = self._means(mean_code_batch)
        _, logq, _, log_self = self._forward(mu)
        # Report the matched (empirical) conditional entropy so that
        # I = H_marginal - H_conditional holds exactly with the estimate().
        H_marginal = float(-np.mean(logq))
        H_conditional = float(-np.mean(log_self))
        return {
            'H_marginal': H_marginal,
            'H_conditional': H_conditional,
            'mutual_information': H_marginal - H_conditional,
        }


def create_mi_estimator(code_dim: int, method: str = 'auto', 
                        n_mc_samples: int = 100,
                        enable_marginal_approx: bool = True,
                        noise_scale=0.5,
                        lognormal_mc_samples: int = 8,
                        resample_every: int = 1,
                        seed: int = 0) -> MutualInformationEstimator:
    """
    Factory function to create appropriate MI estimator.
    
    Args:
        code_dim: Dimensionality of binary codes
        method: MI estimation method
            - 'auto': Choose based on code_dim (analytical if ≤15, else mean-field)
            - 'analytical': Always use exact enumeration (slow for code_dim > 15)
            - 'mc': Monte Carlo sampling (MI estimate OK, gradients biased - use with caution)
            - 'mean_field': Always use mean-field approximation (fast, approximate)
            - 'lognormal': Log-normal channel R->C (continuous). Models downstream
                           activations as log-normal around the mass-action mean and
                           estimates I(R;C) via a Gaussian-mixture entropy in log space.
        n_mc_samples: Number of samples per h for MC method (default: 100)
        enable_marginal_approx: For analytical estimator, whether to use
                                approximate marginal entropy for code_dim > 15.
                                Ignored for other methods.
        noise_scale: (lognormal only) Fixed log-space noise std. Scalar (shared)
                     or array of shape (code_dim,) for independent per-element noise.
        lognormal_mc_samples: (lognormal only) MC draws S per mixture component.
        resample_every: (lognormal only) Redraw common random numbers every this
                        many batches (1 = each batch).
        seed: (lognormal only) RNG seed for the common random numbers.
    
    Returns:
        MutualInformationEstimator instance
        
    Recommendations:
        - code_dim ≤ 15: Use analytical (exact, fast)
        - code_dim > 15: Use mean-field (fast, works well if low correlations)
        - MC method: Good for MI evaluation but NOT recommended for training (biased gradients)
        
    Warnings:
        - Analytical method with code_dim > 15 enumerates 2^code_dim states,
          which can be extremely slow (2^20 = 1 million states)
        - MC method has biased gradients for mixture distributions
        - Mean-field assumes independence and may be inaccurate with correlations
    """
    if method == 'auto':
        if code_dim <= 15:
            return AnalyticalBernoulliMI(use_vectorized=True)
        else:
            import warnings
            warnings.warn(
                f"code_dim={code_dim} > 15: using mean-field approximation. "
                f"To force analytical (slow), use method='analytical'. "
                f"Note: MC method available but has biased gradients.",
                UserWarning
            )
            return MeanFieldMI()
    
    elif method == 'analytical':
        if code_dim > 15:
            import warnings
            warnings.warn(
                f"Analytical MI estimation with code_dim={code_dim} will enumerate "
                f"2^{code_dim} = {2**code_dim:,} states. This may be very slow!",
                UserWarning
            )
        return AnalyticalBernoulliMI(use_vectorized=True)
    
    elif method == 'mc':
        if code_dim > 25:
            import warnings
            warnings.warn(
                f"Monte Carlo with code_dim={code_dim} > 25 may have sparse histograms. "
                f"Consider increasing n_mc_samples or using method='mean_field'.",
                UserWarning
            )
        return MonteCarloMI(n_samples_per_h=n_mc_samples, use_baseline=True)
    
    elif method == 'mean_field':
        if code_dim <= 10:
            import warnings
            warnings.warn(
                f"Using mean-field approximation with code_dim={code_dim}. "
                f"Consider method='analytical' for exact computation.",
                UserWarning
            )
        return MeanFieldMI()
    
    elif method == 'lognormal':
        return LogNormalChannelMI(
            code_dim=code_dim,
            noise_scale=noise_scale,
            n_mc_samples=lognormal_mc_samples,
            resample_every=resample_every,
            seed=seed,
        )
    
    else:
        raise ValueError(
            f"Unknown method: {method!r}. "
            f"Must be 'auto', 'analytical', 'mc', 'mean_field', or 'lognormal'."
        )
