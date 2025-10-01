#################################################
################  Import things #################
#################################################

import numpy as np
import timeit
import random
import copy
import matplotlib.pyplot as plt
import pickle
from sklearn import datasets
from scipy import stats
import glycowork
import pandas as pd


from scipy.stats import dirichlet
import dirichlet as dirichlet_mle #https://github.com/ericsuh/dirichlet

from glycowork.glycan_data.loader import glycan_binding as gb
from glycowork.glycan_data.loader import df_glycan as df_glycan
from glycowork.glycan_data.loader import glycomics_data_loader

glycans_in_gb = list(gb.columns)


from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from sklearn.manifold import TSNE

############################################
################  Functions ################
############################################


from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def get_group_indices(df, groups):
    """
    Given a DataFrame and a list of group names, return a dict mapping each group to a list of column names containing that group string.
    """
    group_indices = dict(zip(groups, [[] for _ in groups]))
    for col in df.columns:
        for group in groups:
            if group in col:
                group_indices[group].append(col)
    return group_indices

def build_prot_seq_list(prot_names):
    prot_seq_list = []
    for prot_name in prot_names:
        prot_seq_list.append(gb[gb['protein'] == prot_name]['target'].iloc[0])
    return prot_seq_list

def build_z_score_mat(glycans_in_df, prot_seq_list, emb, use_emb = True, force_nearest = False, no_nan = True):
    n_gly = len(glycans_in_df)
    n_prots = len(prot_seq_list)
    z_score_mat = np.zeros((n_prots, n_gly))

    for (p, prot_seq) in enumerate(prot_seq_list):
        prot_row = gb[gb['target'] == prot_seq]
        for (i, gly) in enumerate(glycans_in_df):
            if use_emb:
                aff, is_glycan = get_nearest_affinity(gly, prot_row, emb, 1, force_nearest, no_nan)
                # if not is_glycan:
                #     print(gly, aff)
            else:
                if gly in prot_row.columns:
                    gly_ind = glycans_in_df.index(gly)
                    aff = prot_row[gly].to_numpy()[0]
                else:
                    aff = np.nan
            z_score_mat[p, i] = 0 if np.isnan(aff) else aff
    return z_score_mat

def build_random_z_score_mat(glycans_in_df, n_prots, random_func=np.random.normal, **random_kwargs):
    n_gly = len(glycans_in_df)
    z_score_mat = random_func(size=(n_prots, n_gly), **random_kwargs)
    return z_score_mat

# def build_activation_arrays(z_score_mat, group_indices, df):
#     aff_mat = z_score_mat
#     activation_arrays = {}
#     for group, col_list in group_indices.items():
#         activation_array = np.zeros((aff_mat.shape[0], len(col_list)))
#         for i, col in enumerate(col_list):
#             glycan_dist = prob_dist(np.array(df[col]))
#             activation_array[:, i] = np.dot(aff_mat, glycan_dist)
#         activation_arrays[group] = activation_array
#     return activation_arrays


def build_distribution_arrays(group_indices, df):
    distribution_arrays = {}
    for group, col_list in group_indices.items():
        distribution_array = np.zeros((df.shape[0], len(col_list)))
        for i, col in enumerate(col_list):
            glycan_dist = prob_dist(np.array(df[col]))
            distribution_array[:, i] = glycan_dist
        distribution_arrays[group] = distribution_array
    return distribution_arrays

def pca_distribution_arrays(distribution_arrays, n_components=2):
    all_samples = []
    group_sample_counts = {}
    for group, arr in distribution_arrays.items():
        arr = np.asarray(arr)
        all_samples.append(arr.T)  # shape: (samples, features)
        group_sample_counts[group] = arr.shape[1]
    all_samples = np.vstack(all_samples)  # shape: (total_samples, features)

    # Fit PCA on all samples
    pca = PCA(n_components=n_components)
    all_samples_pca = pca.fit_transform(all_samples)

    # Split back into groups
    pca_arrays = {}
    start = 0
    for group, count in group_sample_counts.items():
        end = start + count
        # Transpose back to (n_components, samples_in_group)
        pca_arrays[group] = all_samples_pca[start:end].T
        start = end
    return pca_arrays

def tsne_distribution_arrays(distribution_arrays, n_components=2, random_state=None, **tsne_kwargs):
    method = tsne_kwargs.get('method', 'barnes_hut')
    if n_components >= 4 and method == 'barnes_hut':
        method = 'exact'
    tsne_kwargs['method'] = method
    # Stack all samples from all groups for fitting t-SNE
    all_samples = []
    group_sample_counts = {}
    for group, arr in distribution_arrays.items():
        arr = np.asarray(arr)
        all_samples.append(arr.T)  # shape: (samples, features)
        group_sample_counts[group] = arr.shape[1]
    all_samples = np.vstack(all_samples)  # shape: (total_samples, features)

    # Fit t-SNE on all samples
    tsne = TSNE(n_components=n_components, random_state=random_state, **tsne_kwargs)
    all_samples_tsne = tsne.fit_transform(all_samples)

    # Split back into groups
    tsne_arrays = {}
    start = 0
    for group, count in group_sample_counts.items():
        end = start + count
        # Transpose back to (n_components, samples_in_group)
        tsne_arrays[group] = all_samples_tsne[start:end].T
        start = end
    return tsne_arrays

def compute_silhouette_score(activation_arrays):
    # Suppose activation_arrays is your dict as above
    X = []      # List to hold all activation vectors (samples)
    labels = [] # List to hold group labels for each sample

    for group, arr in activation_arrays.items():
        # arr shape: (n_proteins, n_samples_in_group)
        arr = np.asarray(arr)
        X.append(arr.T)  # Transpose so each row is a sample
        labels.extend([group] * arr.shape[1])

    X = np.vstack(X)  # Shape: (total_samples, n_proteins)
    labels = np.array(labels)

    score = silhouette_score(X, labels)
    return score

def sample_proteins_and_get_silhouette_score(n_proteins, glycans_in_df, emb, group_indices, df):

    prot_inds = np.random.choice(range(len(gb)), n_proteins, replace=False)
    prot_names = gb['protein'].iloc[prot_inds]
    prot_seq_list = build_prot_seq_list(prot_names)
    z_score_mat = build_z_score_mat(glycans_in_df, prot_seq_list, emb, use_emb = True, force_nearest = False, no_nan = True)
    activation_arrays = build_activation_arrays(z_score_mat, group_indices, df)
    distribution_arrays = build_distribution_arrays(group_indices, df)
    return compute_silhouette_score(activation_arrays), compute_silhouette_score(distribution_arrays), prot_names

def sample_random_matrix_and_get_silhouette_score(n_proteins, glycans_in_df, group_indices, df, random_func=np.random.normal, **random_kwargs):

    z_score_mat = build_random_z_score_mat(glycans_in_df, n_proteins, random_func, **random_kwargs)
    activation_arrays = build_activation_arrays(z_score_mat, group_indices, df)
    return compute_silhouette_score(activation_arrays), z_score_mat

def fit_dirichlet_distribution(df, group_indices, lambda_val = 1):
    group_alphas = {}
    n_glys = df.shape[0]
    for group in group_indices:
        alphas = np.zeros(n_glys)
        for i, g in enumerate(group_indices[group]):
            glycan_dist = prob_dist(np.array(df[g]))
            alphas += glycan_dist / len(group_indices[group])
        group_alphas[group] = lambda_val * alphas
    return group_alphas

def fit_dirichlet_distribution_mle(df, group_indices):
    group_alphas = {}
    n_glys = df.shape[0]
    for group in group_indices:
        dists = np.zeros((len(group_indices[group]), n_glys))
        for i, g in enumerate(group_indices[group]):
            glycan_dist = prob_dist(np.array(df[g]))
            dists[i,:] = glycan_dist
        alphas = dirichlet_mle.mle(dists, method="fixedpoint")
        group_alphas[group] = alphas
    return group_alphas

def compute_activation(aff_mat, dist_vec, non_sat = False, safe_val=1e-2):
    """
    Compute the activation given an affinity matrix and a distribution vector.
    If invert is True, invert the affinity matrix as in build_activation_arrays.
    """
    if non_sat:
        aff_mat = aff_mat - np.min(aff_mat) + safe_val
        aff_mat = 1 / aff_mat
        numerator = np.dot(aff_mat, dist_vec)
        denominator = 1 + numerator
    else:
        numerator = np.dot(aff_mat, dist_vec)
        denominator = 1
    return numerator / denominator

def build_activation_arrays(z_score_mat, group_indices, df):
    aff_mat = z_score_mat
    activation_arrays = {}
    for group, col_list in group_indices.items():
        activation_array = np.zeros((aff_mat.shape[0], len(col_list)))
        for i, col in enumerate(col_list):
            glycan_dist = prob_dist(np.array(df[col]))
            activation_array[:, i] = compute_activation(aff_mat, glycan_dist)
        activation_arrays[group] = activation_array
    return activation_arrays

def build_activation_arrays_with_sampling(z_score_mat, group_indices, df, n_samples, lambda_val = 20):
    group_alphas = fit_dirichlet_distribution(df, group_indices, lambda_val)
    aff_mat = z_score_mat
    activation_arrays = {}
    for group in group_indices:
        activation_array = np.zeros((aff_mat.shape[0], n_samples))
        for i in range(n_samples):
            samp = dirichlet.rvs(group_alphas[group], size = 1)[0]
            activation_array[:, i] = compute_activation(aff_mat, samp)
        activation_arrays[group] = activation_array
    return activation_arrays

def build_activation_arrays_with_sampling_mle(z_score_mat, group_indices, df, n_samples):
    group_alphas = fit_dirichlet_distribution_mle(df, group_indices)
    aff_mat = z_score_mat
    activation_arrays = {}
    for group in group_indices:
        activation_array = np.zeros((aff_mat.shape[0], n_samples))
        for i in range(n_samples):
            samp = dirichlet.rvs(group_alphas[group], size = 1)[0]
            activation_array[:, i] = compute_activation(aff_mat, samp)
        activation_arrays[group] = activation_array
    return activation_arrays

def export_data_for_sampling(z_score_mat, group_indices, df, filepath):
    group_alphas = fit_dirichlet_distribution_mle(df, group_indices)
    exp_data = {'group_alphas': group_alphas, 'z_score_mat': z_score_mat, 'group_indices': group_indices, 'df': df}
    with open(filepath, 'wb') as f:
        pickle.dump(exp_data, f)
    

def get_activation_extrema(activation_arrays):
    """Get min/max values across all activation arrays"""
    xmin, xmax = float('inf'), float('-inf')
    ymin, ymax = float('inf'), float('-inf')
    
    for group, activation_array in activation_arrays.items():
        x = activation_array[0,:]
        y = activation_array[1,:]
        
        xmin = min(xmin, x.min())
        xmax = max(xmax, x.max()) 
        ymin = min(ymin, y.min())
        ymax = max(ymax, y.max())
        
    return xmin, xmax, ymin, ymax
    
    
def get_embeddings(  ## gets glycan embeddings
   glycans: List[str], # List of IUPAC-condensed glycan sequences
   emb: Optional[Union[Dict[str, np.ndarray], pd.DataFrame]] = None, # Glycan embeddings dict/DataFrame; defaults to SweetNet embeddings
   label_list: Optional[List[Any]] = None, # Labels for coloring points
   shape_feature: Optional[str] = None, # Monosaccharide/bond for point shapes
   filepath: Union[str, Path] = '', # Path to save plot
   alpha: float = 0.8, # Point transparency
   palette: str = 'colorblind', # Color palette for groups
   **kwargs: Any # Keyword args passed to seaborn scatterplot
   ) -> None:
    "Visualizes learned glycan embeddings using t-SNE dimensionality reduction with optional group coloring"
    idx = [i for i, g in enumerate(glycans) if '{' not in g]
    glycans = [glycans[i] for i in idx]
    if label_list is not None:
      label_list = [label_list[i] for i in idx]
    # Get all glycan embeddings
    if emb is None:
      if not Path('glycan_representations_v1_4.pkl').exists():
          download_model("https://drive.google.com/file/d/1--tf0kyea9jFLfffUtICKkyIw36E9hJ3/view?usp=sharing", local_path = 'glycan_representations_v1_4.pkl')
      emb = pickle.load(open('glycan_representations_v1_4.pkl', 'rb'))
    # Get the subset of embeddings corresponding to 'glycans'
    if isinstance(emb, pd.DataFrame):
      embs = emb.values
      glycans_used = glycans
    else:
      glycans_used = [g for g in glycans if g in emb]
      embs = np.vstack([emb[g] for g in glycans_used])
    # # Calculate t-SNE of embeddings
    # n_samples = embs.shape[0]
    # perplexity = min(30, n_samples - 1)
    # embs = TSNE(random_state = 42, perplexity = perplexity,
    #             init = 'pca', learning_rate = 'auto').fit_transform(embs)
    return embs, glycans_used

emb = pickle.load(open('glycan_representations_v1_4.pkl', 'rb'))
embs, glycans_in_gb_and_emb = get_embeddings(glycans_in_gb)
emb_dict = dict(zip(glycans_in_gb_and_emb, embs))

def get_affinity_relative(glycan, prot_row, emb, threshold = 0.5, force_nearest = False, no_nan = False):
    # Step 1: If the glycan is in the reference set and we are not forcing nearest neighbor search,
    #         return the glycan itself and True (indicating a direct match).
    if (glycan in glycans_in_gb) and (not force_nearest) and ((not no_nan) or (not np.isnan(prot_row[glycan].to_numpy()[0]))):
        return glycan, True
    else:
        if glycan not in emb:
            return None, False
        # Step 2: Otherwise, get the embedding vector for the input glycan.
        embedding = np.array(emb[glycan])
        # Step 3: Initialize variables to track the minimum distance and the closest glycan found so far.
        min_dist = float('inf')
        closest_glycan = None
        # Step 4: Iterate over all glycans in the reference set that have embeddings.
        for g in glycans_in_gb_and_emb:
            # Step 5: If not forcing nearest, skip the input glycan itself.
            if g != glycan or not force_nearest:
                # Step 6: Compute the Euclidean distance between the input glycan and the current glycan.
                dist = np.linalg.norm(embedding - np.array(emb[g]))
                # Step 7: If this is the smallest distance so far, and (optionally) the protein row is not NaN,
                #         update the closest glycan and minimum distance.
                if dist < min_dist:
                    if not no_nan or not np.isnan(prot_row[g].to_numpy()[0]):
                        min_dist = dist
                        closest_glycan = g
        # Step 8: Return the closest glycan and whether its distance is below the threshold.
        return closest_glycan, min_dist < threshold
    
    
def get_nearest_affinity(glycan, prot_row, emb, threshold = 0.5, force_nearest = False, no_nan = False):
    aff_gly, is_close = get_affinity_relative(glycan, prot_row, emb, threshold, force_nearest, no_nan)
    if is_close:
        aff = prot_row[aff_gly].to_numpy()[0]
    else:
        aff = np.nan
    return aff, aff_gly==glycan

def get_affinity(glycan, prot_row):
    if glycan in prot_row.columns:
        return prot_row[glycan].to_numpy()[0]
    else:
        print("not in prot_row")
        return np.nan

def prob_dist(vals):
    vals_new = vals + 1e-8
    return vals_new / np.sum(vals_new)
    