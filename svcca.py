# Copyright 2018 Google Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
PyTorch + GPU version of CCA for deep networks.

This is a GPU-accelerated version using PyTorch instead of NumPy.
All computations are performed on GPU for faster processing.
"""
import json
import torch
import numpy as np

from open_file import load_feature, get_layers
import time

num_cca_trials = 5


def randomized_svd_torch(M, n_components, n_oversamples=10, n_iter=2):
    """
    Randomized SVD using PyTorch (GPU accelerated)
    
    Much faster than full SVD for large matrices when only top-k components needed.
    Complexity: O(n_components * n * m) vs O(min(n,m)^2 * max(n,m)) for full SVD
    
    Args:
        M: (m, n) torch tensor
        n_components: number of singular values/vectors to compute
        n_oversamples: additional random vectors for accuracy
        n_iter: number of power iterations for accuracy
    
    Returns:
        U: (m, n_components) left singular vectors
        S: (n_components,) singular values
        Vt: (n_components, n) right singular vectors transposed
    
    Reference: "Finding structure with randomness" (Halko et al., 2011)
    """
    m, n = M.shape
    k = min(n_components + n_oversamples, min(m, n))
    
    # Generate random matrix
    Omega = torch.randn(n, k, device=M.device, dtype=M.dtype)
    
    # Power iteration for better accuracy
    Y = M @ Omega
    for _ in range(n_iter):
        Y = M @ (M.T @ Y)
    
    # QR decomposition
    Q, _ = torch.linalg.qr(Y)
    
    # Project M onto Q
    B = Q.T @ M
    
    # SVD of smaller matrix
    Uhat, S, Vt = torch.linalg.svd(B, full_matrices=False)
    
    # Recover U
    U = Q @ Uhat
    
    # Return only n_components
    return U[:, :n_components], S[:n_components], Vt[:n_components, :]


def positivedef_matrix_sqrt(array):
    """Stable method for computing matrix square roots, supports complex matrices.
    
    Args:
        array: A torch 2d tensor, can be complex valued that is a positive
               definite symmetric (or hermitian) matrix
    
    Returns:
        sqrtarray: The matrix square root of array
    """
    w, v = torch.linalg.eigh(array)
    wsqrt = torch.sqrt(w)
    sqrtarray = torch.matmul(v, torch.matmul(torch.diag(wsqrt), v.conj().T))
    return sqrtarray


def remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon):
    """Takes covariance between X, Y, and removes values of small magnitude.
    
    Args:
        sigma_xx: 2d torch tensor, variance matrix for x
        sigma_xy: 2d torch tensor, crossvariance matrix for x,y
        sigma_yx: 2d torch tensor, crossvariance matrix for x,y,
                  (conjugate) transpose of sigma_xy
        sigma_yy: 2d torch tensor, variance matrix for y
        epsilon: cutoff value for norm below which directions are thrown away
    
    Returns:
        sigma_xx_crop: 2d tensor with low x norm directions removed
        sigma_xy_crop: 2d tensor with low x and y norm directions removed
        sigma_yx_crop: 2d tensor with low x and y norm directions removed
        sigma_yy_crop: 2d tensor with low y norm directions removed
        x_idxs: indexes of sigma_xx that were removed
        y_idxs: indexes of sigma_yy that were removed
    """
    x_diag = torch.abs(torch.diagonal(sigma_xx))
    y_diag = torch.abs(torch.diagonal(sigma_yy))
    x_idxs = (x_diag >= epsilon)
    y_idxs = (y_diag >= epsilon)
    
    sigma_xx_crop = sigma_xx[x_idxs][:, x_idxs]
    sigma_xy_crop = sigma_xy[x_idxs][:, y_idxs]
    sigma_yx_crop = sigma_yx[y_idxs][:, x_idxs]
    sigma_yy_crop = sigma_yy[y_idxs][:, y_idxs]
    
    return (sigma_xx_crop, sigma_xy_crop, sigma_yx_crop, sigma_yy_crop,
            x_idxs, y_idxs)


def compute_ccas(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon,
                 verbose=False):
    """Main cca computation function, takes in variances and crossvariances.
    
    Args:
        sigma_xx: 2d torch tensor, (num_neurons_x, num_neurons_x)
                  variance matrix for x
        sigma_xy: 2d torch tensor, (num_neurons_x, num_neurons_y)
                  crossvariance matrix for x,y
        sigma_yx: 2d torch tensor, (num_neurons_y, num_neurons_x)
                  crossvariance matrix for x,y (conj) transpose of sigma_xy
        sigma_yy: 2d torch tensor, (num_neurons_y, num_neurons_y)
                  variance matrix for y
        epsilon: small float to help with stabilizing computations
        verbose: boolean on whether to print intermediate outputs
    
    Returns:
        [u, s, v]: CCA results
        invsqrt_xx: Inverse square root of sigma_xx
        invsqrt_yy: Inverse square root of sigma_yy
        x_idxs: The indexes that were pruned
        y_idxs: The indexes that were pruned
    """
    (sigma_xx, sigma_xy, sigma_yx, sigma_yy,
     x_idxs, y_idxs) = remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon)
    
    numx = sigma_xx.shape[0]
    numy = sigma_yy.shape[0]
    
    if numx == 0 or numy == 0:
        device = sigma_xx.device
        return ([torch.tensor(0, device=device), torch.tensor(0, device=device), torch.tensor(0, device=device)],
                torch.zeros_like(sigma_xx),
                torch.zeros_like(sigma_yy), x_idxs, y_idxs)
    
    if verbose:
        print("adding eps to diagonal and taking inverse")
    sigma_xx += epsilon * torch.eye(numx, device=sigma_xx.device, dtype=sigma_xx.dtype)
    sigma_yy += epsilon * torch.eye(numy, device=sigma_yy.device, dtype=sigma_yy.dtype)
    inv_xx = torch.linalg.pinv(sigma_xx)
    inv_yy = torch.linalg.pinv(sigma_yy)
    
    if verbose:
        print("taking square root")
    invsqrt_xx = positivedef_matrix_sqrt(inv_xx)
    invsqrt_yy = positivedef_matrix_sqrt(inv_yy)
    
    if verbose:
        print("dot products...")
    arr = torch.matmul(invsqrt_xx, torch.matmul(sigma_xy, invsqrt_yy))
    
    if verbose:
        print("trying to take final svd")
    print(arr.shape)
    
    # Use randomized SVD for faster computation (especially for large matrices)
    # This computes only top-k singular values which is what we need for CCA
    n_components = min(arr.shape[0], arr.shape[1])  # Compute all components for accurate CCA
    
    if arr.shape[0] > 1000 or arr.shape[1] > 1000:  # Only use randomized for very large matrices
        # Use randomized SVD for large matrices
        if verbose:
            print(f"Using randomized SVD with {n_components} components")
        u, s, vh = randomized_svd_torch(arr, n_components=n_components)
        v = vh.T
    else:
        # Use full SVD for small matrices
        u, s, vh = torch.linalg.svd(arr, full_matrices=False)
        v = vh.T  # torch returns V^H, we need V
    
    if verbose:
        print("computed everything!")
    
    return [u, torch.abs(s), v], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs


def sum_threshold(array, threshold):
    """Computes threshold index of decreasing nonnegative array by summing.
    
    Args:
        array: a 1d torch tensor of decreasing, nonnegative floats
        threshold: a number between 0 and 1
    
    Returns:
        i: index at which torch.sum(array[:i]) >= threshold
    """
    assert (threshold >= 0) and (threshold <= 1), "incorrect threshold"
    
    cumsum = torch.cumsum(array, dim=0)
    total = torch.sum(array)
    idxs = (cumsum / total >= threshold).nonzero(as_tuple=True)[0]
    
    if len(idxs) > 0:
        return idxs[0].item()
    return len(array)


def create_zero_dict(compute_dirns, dimension, device='cuda'):
    """Outputs a zero dict when neuron activation norms too small.
    
    Args:
        compute_dirns: boolean, whether to have zero vectors for directions
        dimension: int, defines shape of directions
        device: torch device ('cuda' or 'cpu')
    
    Returns:
        return_dict: a dict of appropriately shaped zero entries
    """
    return_dict = {}
    return_dict["mean"] = (torch.tensor(0.0, device=device), torch.tensor(0.0, device=device))
    return_dict["sum"] = (torch.tensor(0.0, device=device), torch.tensor(0.0, device=device))
    return_dict["cca_coef1"] = torch.tensor(0.0, device=device)
    return_dict["cca_coef2"] = torch.tensor(0.0, device=device)
    return_dict["idx1"] = 0
    return_dict["idx2"] = 0
    
    if compute_dirns:
        return_dict["cca_dirns1"] = torch.zeros((1, dimension), device=device)
        return_dict["cca_dirns2"] = torch.zeros((1, dimension), device=device)
    
    return return_dict


def get_cca_similarity(acts1, acts2, epsilon=0., threshold=0.98,
                       compute_coefs=True,
                       compute_dirns=False,
                       verbose=False,
                       device='cuda',
                       use_svcca=True,
                       svd_components=None):
    """The main function for computing CCA or SVCCA similarities on GPU.
    
    NOTE: This is vanilla CCA by default. For true SVCCA (as in the paper),
    set use_svcca=True which performs SVD preprocessing before CCA.
    
    Args:
        acts1: (num_neurons1, data_points) a 2d torch tensor on GPU
        acts2: (num_neurons2, data_points) same as above
        epsilon: small float to help stabilize computations
        threshold: float between 0, 1 for trailing zeros threshold
        compute_coefs: boolean value determining whether coefficients are computed
        compute_dirns: boolean value determining whether actual cca directions are computed
        verbose: Boolean, whether intermediate outputs are printed
        device: torch device ('cuda' or 'cpu')
        use_svcca: If True, apply SVD preprocessing before CCA (true SVCCA from paper)
        svd_components: Number of SVD components to keep. If None, keeps 98% variance.
    
    Returns:
        return_dict: A dictionary with outputs from the cca computations
    """
    # Convert to torch tensors if needed and move to device
    if isinstance(acts1, np.ndarray):
        acts1 = torch.from_numpy(acts1).to(device)
    elif acts1.device.type != device:
        acts1 = acts1.to(device)
    
    if isinstance(acts2, np.ndarray):
        acts2 = torch.from_numpy(acts2).to(device)
    elif acts2.device.type != device:
        acts2 = acts2.to(device)
    
    # Convert to float32 if not already
    if acts1.dtype == torch.float16:
        acts1 = acts1.float()
    if acts2.dtype == torch.float16:
        acts2 = acts2.float()
    
    # Assert dimensionality equal
    assert acts1.shape[1] == acts2.shape[1], "dimensions don't match"
    
    # Recommendation: num_samples > num_neurons for statistical stability
    # But this is not strictly required due to regularization
    
    return_dict = {}
    
    # SVCCA Step 1: SVD preprocessing (as in the paper)
    if use_svcca:
        if verbose:
            print("Applying SVD preprocessing (SVCCA)...")
        
        # SVD for acts1
        u1, s1, v1 = torch.linalg.svd(acts1, full_matrices=False)
        
        # Determine number of components to keep (default: high variance threshold for more components)
        # IMPORTANT: Ensure minimum of 3 components to avoid degenerate cases
        # where a single component explains most variance, leading to 1x1 covariance matrices
        min_components = 5  # Minimum components to maintain statistical stability
        variance_threshold = 0.98  # High threshold to keep more components
        
        if svd_components is None:
            # Keep components that explain variance_threshold of total variance
            variance_explained = torch.cumsum(s1**2, dim=0) / torch.sum(s1**2)
            svd_components_1 = torch.searchsorted(variance_explained, variance_threshold).item() + 1
            # Ensure we keep at least min_components (but not more than available)
            svd_components_1 = max(min_components, min(svd_components_1, len(s1)))
        else:
            svd_components_1 = min(svd_components, len(s1))
        
        # Project onto top components
        acts1_reduced = u1[:, :svd_components_1].T @ acts1  # (k1, samples)
        
        # SVD for acts2
        u2, s2, v2 = torch.linalg.svd(acts2, full_matrices=False)
        
        if svd_components is None:
            variance_explained = torch.cumsum(s2**2, dim=0) / torch.sum(s2**2)
            svd_components_2 = torch.searchsorted(variance_explained, variance_threshold).item() + 1
            # Ensure we keep at least min_components (but not more than available)
            svd_components_2 = max(min_components, min(svd_components_2, len(s2)))
        else:
            svd_components_2 = min(svd_components, len(s2))
        
        acts2_reduced = u2[:, :svd_components_2].T @ acts2  # (k2, samples)
        
        if verbose:
            print(f"  acts1: {acts1.shape} → {acts1_reduced.shape} (kept {svd_components_1} components)")
            print(f"  acts2: {acts2.shape} → {acts2_reduced.shape} (kept {svd_components_2} components)")
        
        # Use reduced representations for CCA
        acts1 = acts1_reduced
        acts2 = acts2_reduced
    
    # Compute covariance
    numx = acts1.shape[0]
    numy = acts2.shape[0]
    
    # Concatenate and compute covariance
    combined = torch.cat([acts1, acts2], dim=0)
    # Center the data
    combined = combined - combined.mean(dim=1, keepdim=True)
    covariance = torch.matmul(combined, combined.T) / (combined.shape[1] - 1)
    
    sigmaxx = covariance[:numx, :numx]
    sigmaxy = covariance[:numx, numx:]
    sigmayx = covariance[numx:, :numx]
    sigmayy = covariance[numx:, numx:]
    
    # Rescale covariance to make cca computation more stable
    xmax = torch.max(torch.abs(sigmaxx))
    ymax = torch.max(torch.abs(sigmayy))
    sigmaxx /= xmax
    sigmayy /= ymax
    sigmaxy /= torch.sqrt(xmax * ymax)
    sigmayx /= torch.sqrt(xmax * ymax)
    
    ([u, s, v], invsqrt_xx, invsqrt_yy,
     x_idxs, y_idxs) = compute_ccas(sigmaxx, sigmaxy, sigmayx, sigmayy,
                                     epsilon=epsilon,
                                     verbose=verbose)
    
    # If x_idxs or y_idxs is all false, return_dict has zero entries
    if (not torch.any(x_idxs)) or (not torch.any(y_idxs)):
        return create_zero_dict(compute_dirns, acts1.shape[1], device=device)
    
    # Store coefficients and valid indices
    return_dict["coef_x"] = u.T
    return_dict["invsqrt_xx"] = invsqrt_xx
    return_dict["coef_y"] = v
    return_dict["invsqrt_yy"] = invsqrt_yy
    return_dict["x_idxs"] = x_idxs
    return_dict["y_idxs"] = y_idxs
    
    if compute_dirns:
        # Compute means
        neuron_means1 = torch.mean(acts1, dim=1, keepdim=True)
        neuron_means2 = torch.mean(acts2, dim=1, keepdim=True)
        return_dict["neuron_means1"] = neuron_means1
        return_dict["neuron_means2"] = neuron_means2
        
        # Compute CCA directions in reduced space
        acts1_centered = acts1 - neuron_means1
        acts2_centered = acts2 - neuron_means2
        
        cca_dirns1 = torch.matmul(torch.matmul(u.T, invsqrt_xx), acts1_centered) + neuron_means1
        cca_dirns2 = torch.matmul(torch.matmul(v.T, invsqrt_yy), acts2_centered) + neuron_means2
        
        return_dict["cca_dirns1"] = cca_dirns1
        return_dict["cca_dirns2"] = cca_dirns2
    
    # Get rid of trailing zeros in the cca coefficients
    idx1 = sum_threshold(s, threshold)
    idx2 = sum_threshold(s, threshold)
    
    return_dict["cca_coef1"] = s
    return_dict["cca_coef2"] = s
    return_dict["idx1"] = idx1
    return_dict["idx2"] = idx2
    
    # Summary statistics
    return_dict["mean"] = (torch.mean(s[:idx1]).item(), torch.mean(s[:idx2]).item())
    return_dict["sum"] = (torch.sum(s).item(), torch.sum(s).item())

    return return_dict


def robust_cca_similarity(acts1, acts2, threshold=0.98, epsilon=1e-6,
                          compute_dirns=True, device='cuda', use_svcca=True, svd_components=None):
    """Calls get_cca_similarity multiple times while adding noise (GPU version).
    
    Args:
        acts1: (num_neurons1, data_points) a 2d torch tensor
        acts2: (num_neurons2, data_points) same as above
        threshold: float between 0, 1
        epsilon: small float to help stabilize computations
        compute_dirns: boolean value determining whether actual cca directions are computed
        device: torch device ('cuda' or 'cpu')
        use_svcca: If True, apply SVD preprocessing before CCA (true SVCCA from paper)
        svd_components: Number of SVD components to keep. If None, keeps 98% variance.
    
    Returns:
        return_dict: A dictionary with outputs from the cca computations
    """
    # Move to device if needed
    if isinstance(acts1, np.ndarray):
        acts1 = torch.from_numpy(acts1).to(device)
    elif acts1.device.type != device:
        acts1 = acts1.to(device)
    
    if isinstance(acts2, np.ndarray):
        acts2 = torch.from_numpy(acts2).to(device)
    elif acts2.device.type != device:
        acts2 = acts2.to(device)
    
    for trial in range(num_cca_trials):
        try:
            return_dict = get_cca_similarity(
                acts1, acts2, 
                threshold=threshold,
                compute_dirns=compute_dirns, 
                device=device,
                use_svcca=use_svcca,
                svd_components=svd_components
            )
            return return_dict
        except RuntimeError:  # torch uses RuntimeError instead of LinAlgError
            acts1 = acts1 * 1e-1 + torch.randn_like(acts1) * epsilon
            acts2 = acts2 * 1e-1 + torch.randn_like(acts2) * epsilon
            if trial + 1 == num_cca_trials:
                raise
    
    return return_dict


# Convenience function to convert results back to numpy if needed
def cca_dict_to_numpy(return_dict):
    """Convert all torch tensors in return_dict to numpy arrays (CPU)."""
    numpy_dict = {}
    for key, value in return_dict.items():
        if isinstance(value, torch.Tensor):
            numpy_dict[key] = value.cpu().numpy()
        elif isinstance(value, tuple):
            numpy_dict[key] = tuple(v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in value)
        else:
            numpy_dict[key] = value
    return numpy_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a_path", type=str, help="Path to hdf file of cached features of model A")
    parser.add_argument("--model_b_path", type=str, help="Path to hdf file of cached features of model B")
    args = parser.parse_args()
    
    print("Loading features...")
    start = time.time()
    
    mean_cca_similarity = {}
    layers = get_layers("Path to hdf file of cached features of model A or model B")  # NOTE: need to change this 
    
    for layer in layers:
        print(layer)
        actv_model_a = load_feature(
            args.model_a_path,  
            layer)
        actv_model_b = load_feature(
            args.model_b_path, 
            layer)
        
        print(f"Loading took {time.time() - start:.2f}s")
        print(f"Activation A shape: {actv_model_a.shape}")
        print(f"Activation B shape: {actv_model_b.shape}")
        
        # Convert to torch and move to GPU
        print("\nPreparing activations...")
        start = time.time()
        h_dim = actv_model_a.shape[-1]
        if len(actv_model_a.shape) == 3:
            act1 = torch.from_numpy(actv_model_a).type(torch.float32).cuda().reshape(-1, h_dim)
            act2 = torch.from_numpy(actv_model_b).type(torch.float32).cuda().reshape(-1, h_dim)
        
        # Transpose to (D, N) format required by CCA
        act1 = act1.t()  # (D, N)
        act2 = act2.t()  # (D, N)
        
        #print(f"Preparation took {time.time() - start:.2f}s")
        print(f"act1 shape: {act1.shape} (neurons, samples)")
        print(f"act2 shape: {act2.shape} (neurons, samples)")
        
        # Compute SVCCA (with SVD preprocessing)
        print("\nComputing SVCCA similarity (SVD + CCA)...")
        start = time.time()
        return_dict = get_cca_similarity(
            act1, act2, 
            threshold=0.98, 
            epsilon=1e-6, 
            compute_dirns=False,  # Set to False for faster computation if directions not needed
            device='cuda',
            use_svcca=False,  # Use true SVCCA from paper (SVD preprocessing)
            svd_components=None  # Auto: keep 98% variance
        )
        elapsed = time.time() - start
        
        print(f"SVCCA computation took {elapsed:.2f}s")
        print(f"Results:")
        print(f"  Mean CCA similarity: {return_dict['mean']}")
        print(f"{'='*50}")
        mean_cca_similarity[layer] = return_dict['mean'][0]
    
    # summarize the results
    for k, v in mean_cca_similarity.items():
        print(f"Layer: {k}, Mean CCA similarity: {v}\n")
    
    with open('svcca_results/similarity_results.json', 'w') as f:
        json.dump(mean_cca_similarity, f)
