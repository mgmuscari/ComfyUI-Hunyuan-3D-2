# Embedding Mixer and List Management Nodes
# N-input aggregation for multi-image sculptural composition
#
# Categories:
#   - List Management: Create, Append, Split, Map
#   - Aggregation: Mean, Weighted Mix, PCA
#   - Statistical: Variance, Distance Matrix

import torch
import numpy as np
from typing import List, Dict, Any


# Import slerp from parent module
def slerp(v0, v1, t):
    """Spherical linear interpolation between two tensors."""
    orig_shape = v0.shape
    v0_flat = v0.reshape(v0.shape[0], -1)
    v1_flat = v1.reshape(v1.shape[0], -1)

    v0_norm = v0_flat / (torch.norm(v0_flat, dim=-1, keepdim=True) + 1e-8)
    v1_norm = v1_flat / (torch.norm(v1_flat, dim=-1, keepdim=True) + 1e-8)

    dot = torch.sum(v0_norm * v1_norm, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    mask = (sin_theta.abs() < 1e-6).float()

    s0 = torch.sin((1.0 - t) * theta) / (sin_theta + 1e-8)
    s1 = torch.sin(t * theta) / (sin_theta + 1e-8)

    result = (s0 * v0_flat + s1 * v1_flat) * (1 - mask) + ((1 - t) * v0_flat + t * v1_flat) * mask

    return result.reshape(orig_shape)


# =============================================================================
# LIST MANAGEMENT NODES
# =============================================================================

class Hunyuan3DEmbeddingList:
    """Create an embedding list from individual embeddings.

    Start of a list for N-input operations.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_1": ("EMBEDDING",),
            },
            "optional": {
                "embedding_2": ("EMBEDDING",),
                "embedding_3": ("EMBEDDING",),
                "embedding_4": ("EMBEDDING",),
            }
        }

    RETURN_TYPES = ("EMBEDDING_LIST", "INT")
    RETURN_NAMES = ("embedding_list", "count")
    FUNCTION = "create_list"
    CATEGORY = "3d/meltdown-spectre/aggregate"

    def create_list(self, embedding_1, embedding_2=None, embedding_3=None, embedding_4=None):
        embeddings = [embedding_1]

        if embedding_2 is not None:
            embeddings.append(embedding_2)
        if embedding_3 is not None:
            embeddings.append(embedding_3)
        if embedding_4 is not None:
            embeddings.append(embedding_4)

        return (embeddings, len(embeddings))


class Hunyuan3DEmbeddingListAppend:
    """Append an embedding to an existing list."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_list": ("EMBEDDING_LIST",),
                "embedding": ("EMBEDDING",),
            }
        }

    RETURN_TYPES = ("EMBEDDING_LIST", "INT")
    RETURN_NAMES = ("embedding_list", "count")
    FUNCTION = "append"
    CATEGORY = "3d/meltdown-spectre/aggregate"

    def append(self, embedding_list, embedding):
        new_list = list(embedding_list) + [embedding]
        return (new_list, len(new_list))


class Hunyuan3DEmbeddingListSplit:
    """Extract individual embeddings from a list by index."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_list": ("EMBEDDING_LIST",),
                "index": ("INT", {"default": 0, "min": 0, "max": 99}),
            }
        }

    RETURN_TYPES = ("EMBEDDING", "INT")
    RETURN_NAMES = ("embedding", "total_count")
    FUNCTION = "split"
    CATEGORY = "3d/meltdown-spectre/aggregate"

    def split(self, embedding_list, index=0):
        if index >= len(embedding_list):
            index = len(embedding_list) - 1
        return (embedding_list[index], len(embedding_list))


class Hunyuan3DEmbeddingListConcat:
    """Concatenate two embedding lists."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "list_a": ("EMBEDDING_LIST",),
                "list_b": ("EMBEDDING_LIST",),
            }
        }

    RETURN_TYPES = ("EMBEDDING_LIST", "INT")
    RETURN_NAMES = ("combined_list", "count")
    FUNCTION = "concat"
    CATEGORY = "3d/meltdown-spectre/aggregate"

    def concat(self, list_a, list_b):
        combined = list(list_a) + list(list_b)
        return (combined, len(combined))


# =============================================================================
# AGGREGATION NODES
# =============================================================================

class Hunyuan3DEmbeddingMean:
    """Compute the centroid (mean) of N embeddings.

    Useful for finding the "average" of multiple aesthetic sources.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_list": ("EMBEDDING_LIST",),
            },
            "optional": {
                "normalize": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("EMBEDDING",)
    RETURN_NAMES = ("mean",)
    FUNCTION = "compute_mean"
    CATEGORY = "3d/meltdown-spectre/aggregate"

    def compute_mean(self, embedding_list, normalize=False):
        if len(embedding_list) == 0:
            raise ValueError("Empty embedding list")

        # Get reference structure from first embedding
        reference = embedding_list[0]
        result = {}

        for key in reference.keys():
            if isinstance(reference[key], torch.Tensor):
                # Stack all tensors and compute mean
                tensors = [emb[key] for emb in embedding_list]
                stacked = torch.stack(tensors, dim=0)
                mean_tensor = stacked.mean(dim=0)

                if normalize:
                    norm = torch.norm(mean_tensor.reshape(mean_tensor.shape[0], -1), dim=-1, keepdim=True)
                    # Expand norm to match tensor shape
                    if len(mean_tensor.shape) == 3:
                        norm = norm.unsqueeze(-1)
                    mean_tensor = mean_tensor / (norm + 1e-8)

                result[key] = mean_tensor
            else:
                result[key] = reference[key]

        return (result,)


class Hunyuan3DEmbeddingWeightedMix:
    """Mix N embeddings with configurable weights.

    Multiple blend modes:
    - linear: Simple weighted average
    - slerp_chain: Sequential spherical interpolation
    - softmax: Weights normalized via softmax
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_list": ("EMBEDDING_LIST",),
            },
            "optional": {
                "weights": ("STRING", {"default": "1.0, 1.0, 1.0, 1.0"}),
                "blend_mode": (["linear", "slerp_chain", "softmax_linear"],),
                "normalize_output": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("EMBEDDING",)
    RETURN_NAMES = ("mixed",)
    FUNCTION = "mix"
    CATEGORY = "3d/meltdown-spectre/aggregate"

    def mix(self, embedding_list, weights="1.0, 1.0, 1.0, 1.0",
            blend_mode="linear", normalize_output=False):

        # Parse weights string
        weight_values = [float(w.strip()) for w in weights.split(",")]

        # Pad or truncate weights to match list length
        n = len(embedding_list)
        if len(weight_values) < n:
            weight_values.extend([1.0] * (n - len(weight_values)))
        weight_values = weight_values[:n]

        # Convert to tensor
        device = embedding_list[0]['main'].device
        dtype = embedding_list[0]['main'].dtype
        w = torch.tensor(weight_values, device=device, dtype=dtype)

        # Normalize weights based on mode
        if blend_mode == "softmax_linear":
            w = torch.softmax(w, dim=0)
        else:
            w = w / (w.sum() + 1e-8)

        reference = embedding_list[0]
        result = {}

        for key in reference.keys():
            if isinstance(reference[key], torch.Tensor):
                if blend_mode in ["linear", "softmax_linear"]:
                    # Weighted sum
                    mixed = torch.zeros_like(reference[key])
                    for i, emb in enumerate(embedding_list):
                        mixed = mixed + w[i] * emb[key]

                elif blend_mode == "slerp_chain":
                    # Sequential SLERP
                    mixed = embedding_list[0][key]
                    remaining_weight = 1.0

                    for i in range(1, n):
                        if remaining_weight < 1e-8:
                            break
                        # Interpolation factor for this step
                        t = weight_values[i] / (remaining_weight * sum(weight_values[i:]) + 1e-8)
                        t = min(max(t, 0.0), 1.0)

                        mixed = slerp(mixed, embedding_list[i][key], t)
                        remaining_weight -= weight_values[i] / sum(weight_values)

                if normalize_output:
                    flat = mixed.reshape(mixed.shape[0], -1)
                    norm = torch.norm(flat, dim=-1, keepdim=True)
                    if len(mixed.shape) == 3:
                        norm = norm.unsqueeze(-1)
                    mixed = mixed / (norm + 1e-8)

                result[key] = mixed
            else:
                result[key] = reference[key]

        return (result,)


class Hunyuan3DEmbeddingMixer4:
    """Convenience node: Mix exactly 4 embeddings with individual weight sliders.

    More ergonomic than the list-based mixer for common 4-image workflows.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_1": ("EMBEDDING",),
                "embedding_2": ("EMBEDDING",),
            },
            "optional": {
                "embedding_3": ("EMBEDDING",),
                "embedding_4": ("EMBEDDING",),
                "weight_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "weight_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "weight_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "weight_4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "blend_mode": (["linear", "slerp_chain", "softmax"],),
                "normalize": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("EMBEDDING",)
    RETURN_NAMES = ("mixed",)
    FUNCTION = "mix"
    CATEGORY = "3d/meltdown-spectre/aggregate"

    def mix(self, embedding_1, embedding_2, embedding_3=None, embedding_4=None,
            weight_1=1.0, weight_2=1.0, weight_3=1.0, weight_4=1.0,
            blend_mode="linear", normalize=False):

        embeddings = [embedding_1, embedding_2]
        weights = [weight_1, weight_2]

        if embedding_3 is not None:
            embeddings.append(embedding_3)
            weights.append(weight_3)
        if embedding_4 is not None:
            embeddings.append(embedding_4)
            weights.append(weight_4)

        # Delegate to weighted mix
        mixer = Hunyuan3DEmbeddingWeightedMix()
        weights_str = ", ".join(str(w) for w in weights)

        mode = "softmax_linear" if blend_mode == "softmax" else blend_mode
        return mixer.mix(embeddings, weights_str, mode, normalize)


# =============================================================================
# STATISTICAL OPERATORS
# =============================================================================

class Hunyuan3DEmbeddingVariance:
    """Compute variance/spread of N embeddings.

    Returns both variance embedding and scalar spread measure.
    High variance = diverse inputs, low variance = similar inputs.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_list": ("EMBEDDING_LIST",),
            }
        }

    RETURN_TYPES = ("EMBEDDING", "FLOAT")
    RETURN_NAMES = ("variance", "spread")
    FUNCTION = "compute_variance"
    CATEGORY = "3d/meltdown-spectre/aggregate"

    def compute_variance(self, embedding_list):
        if len(embedding_list) < 2:
            raise ValueError("Need at least 2 embeddings for variance")

        reference = embedding_list[0]
        result = {}
        total_spread = 0.0

        for key in reference.keys():
            if isinstance(reference[key], torch.Tensor):
                tensors = [emb[key] for emb in embedding_list]
                stacked = torch.stack(tensors, dim=0)

                # Variance along the list dimension
                variance = stacked.var(dim=0)
                result[key] = variance

                # Scalar spread (mean of variances)
                total_spread += variance.mean().item()
            else:
                result[key] = reference[key]

        return (result, total_spread)


class Hunyuan3DEmbeddingDistanceMatrix:
    """Compute pairwise distances between embeddings in a list.

    Returns distance matrix as an image for visualization.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_list": ("EMBEDDING_LIST",),
            },
            "optional": {
                "metric": (["cosine", "euclidean", "dot"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("distance_matrix", "info")
    FUNCTION = "compute_distances"
    CATEGORY = "3d/meltdown-spectre/aggregate"

    def compute_distances(self, embedding_list, metric="cosine"):
        n = len(embedding_list)

        # Extract and flatten main embeddings
        flattened = []
        for emb in embedding_list:
            main = emb['main']
            flat = main.reshape(main.shape[0], -1)
            flattened.append(flat)

        # Compute pairwise distances
        distances = torch.zeros(n, n, device=flattened[0].device, dtype=flattened[0].dtype)

        for i in range(n):
            for j in range(n):
                if metric == "cosine":
                    # Cosine distance = 1 - cosine_similarity
                    dot = torch.sum(flattened[i] * flattened[j])
                    norm_i = torch.norm(flattened[i])
                    norm_j = torch.norm(flattened[j])
                    sim = dot / (norm_i * norm_j + 1e-8)
                    distances[i, j] = 1.0 - sim

                elif metric == "euclidean":
                    distances[i, j] = torch.norm(flattened[i] - flattened[j])

                elif metric == "dot":
                    # Negative dot product (smaller = more similar)
                    distances[i, j] = -torch.sum(flattened[i] * flattened[j])

        # Normalize for visualization
        d_min, d_max = distances.min(), distances.max()
        normalized = (distances - d_min) / (d_max - d_min + 1e-8)

        # Create visualization image
        # Expand to displayable size
        scale = max(1, 64 // n)
        img_size = n * scale

        # Repeat to make larger
        dist_img = normalized.unsqueeze(0).unsqueeze(0)  # [1, 1, n, n]
        dist_img = torch.nn.functional.interpolate(dist_img, size=(img_size, img_size), mode='nearest')

        # Convert to RGB (colormap: blue=similar, red=different)
        dist_img = dist_img.squeeze()  # [img_size, img_size]
        rgb = torch.zeros(1, img_size, img_size, 3, device=distances.device, dtype=distances.dtype)
        rgb[0, :, :, 0] = dist_img  # Red channel = distance
        rgb[0, :, :, 2] = 1.0 - dist_img  # Blue channel = similarity

        # Generate info string
        info_lines = [f"Distance Matrix ({metric}):", f"  Embeddings: {n}"]
        if n <= 5:
            info_lines.append("  Distances:")
            for i in range(n):
                row = [f"{distances[i, j].item():.3f}" for j in range(n)]
                info_lines.append(f"    [{', '.join(row)}]")

        return (rgb.cpu(), "\n".join(info_lines))


class Hunyuan3DEmbeddingPCA:
    """Extract principal components from N embeddings.

    Returns the main directions of variation in the embedding set.
    Useful for understanding the "axes of difference" between inputs.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_list": ("EMBEDDING_LIST",),
            },
            "optional": {
                "n_components": ("INT", {"default": 3, "min": 1, "max": 10}),
                "return_component": ("INT", {"default": 0, "min": 0, "max": 9}),
            }
        }

    RETURN_TYPES = ("EMBEDDING", "STRING")
    RETURN_NAMES = ("principal_component", "variance_explained")
    FUNCTION = "compute_pca"
    CATEGORY = "3d/meltdown-spectre/aggregate"

    def compute_pca(self, embedding_list, n_components=3, return_component=0):
        if len(embedding_list) < 2:
            raise ValueError("Need at least 2 embeddings for PCA")

        reference = embedding_list[0]
        result = {}
        variance_info = []

        for key in reference.keys():
            if isinstance(reference[key], torch.Tensor):
                # Stack and flatten
                tensors = [emb[key] for emb in embedding_list]
                stacked = torch.stack(tensors, dim=0)  # [N, B, S, D] or [N, B, D]

                orig_shape = tensors[0].shape
                n_samples = len(tensors)

                # Flatten each sample
                flattened = stacked.reshape(n_samples, -1)  # [N, features]

                # Center the data
                mean = flattened.mean(dim=0, keepdim=True)
                centered = flattened - mean

                # SVD for PCA
                U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

                # Principal components are rows of Vh
                n_actual = min(n_components, len(S))
                if return_component >= n_actual:
                    return_component = n_actual - 1

                # Get requested component
                component = Vh[return_component]  # [features]

                # Reshape to original embedding shape (add batch dim)
                pc_tensor = component.reshape(orig_shape)

                result[key] = pc_tensor

                # Variance explained
                total_var = (S ** 2).sum()
                var_explained = [(S[i] ** 2 / total_var).item() * 100 for i in range(n_actual)]
                variance_info = [f"PC{i}: {v:.1f}%" for i, v in enumerate(var_explained)]
            else:
                result[key] = reference[key]

        info_str = f"Variance explained:\n  " + "\n  ".join(variance_info)

        return (result, info_str)


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    # List management
    "Hunyuan3DEmbeddingList": Hunyuan3DEmbeddingList,
    "Hunyuan3DEmbeddingListAppend": Hunyuan3DEmbeddingListAppend,
    "Hunyuan3DEmbeddingListSplit": Hunyuan3DEmbeddingListSplit,
    "Hunyuan3DEmbeddingListConcat": Hunyuan3DEmbeddingListConcat,
    # Aggregation
    "Hunyuan3DEmbeddingMean": Hunyuan3DEmbeddingMean,
    "Hunyuan3DEmbeddingWeightedMix": Hunyuan3DEmbeddingWeightedMix,
    "Hunyuan3DEmbeddingMixer4": Hunyuan3DEmbeddingMixer4,
    # Statistical
    "Hunyuan3DEmbeddingVariance": Hunyuan3DEmbeddingVariance,
    "Hunyuan3DEmbeddingDistanceMatrix": Hunyuan3DEmbeddingDistanceMatrix,
    "Hunyuan3DEmbeddingPCA": Hunyuan3DEmbeddingPCA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # List management
    "Hunyuan3DEmbeddingList": "Create Embedding List",
    "Hunyuan3DEmbeddingListAppend": "Append to Embedding List",
    "Hunyuan3DEmbeddingListSplit": "Get Embedding from List",
    "Hunyuan3DEmbeddingListConcat": "Concatenate Embedding Lists",
    # Aggregation
    "Hunyuan3DEmbeddingMean": "Embedding Mean (Centroid)",
    "Hunyuan3DEmbeddingWeightedMix": "Weighted Embedding Mix (N-input)",
    "Hunyuan3DEmbeddingMixer4": "4-Way Embedding Mixer",
    # Statistical
    "Hunyuan3DEmbeddingVariance": "Embedding Variance (Spread)",
    "Hunyuan3DEmbeddingDistanceMatrix": "Embedding Distance Matrix",
    "Hunyuan3DEmbeddingPCA": "Embedding PCA (Principal Components)",
}
