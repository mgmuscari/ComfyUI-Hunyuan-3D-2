# Extended Embedding Algebra Nodes
# Advanced operators for multi-image sculptural composition
#
# Categories:
#   - Projection: Project, Reject, Orthogonalize
#   - Spectral: Decompose, Filter, Recombine (SVD-based)
#   - Set Operations: Union, Intersection, XOR
#   - Splice: Sequence splice, Spectral splice, Mask

import torch
import numpy as np


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def flatten_embedding(emb_tensor):
    """Flatten embedding to 2D [batch, features] for matrix operations."""
    original_shape = emb_tensor.shape
    if len(original_shape) == 3:
        B, S, D = original_shape
        return emb_tensor.reshape(B, S * D), original_shape
    elif len(original_shape) == 2:
        return emb_tensor, original_shape
    else:
        raise ValueError(f"Unexpected embedding shape: {original_shape}")


def unflatten_embedding(flat_tensor, original_shape):
    """Restore embedding to original shape."""
    if len(original_shape) == 3:
        B, S, D = original_shape
        return flat_tensor.reshape(B, S, D)
    return flat_tensor


def cosine_similarity_tensors(a, b):
    """Compute cosine similarity between two tensors (flattened)."""
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    dot = torch.sum(a_flat * b_flat, dim=-1, keepdim=True)
    norm_a = torch.norm(a_flat, dim=-1, keepdim=True)
    norm_b = torch.norm(b_flat, dim=-1, keepdim=True)
    return dot / (norm_a * norm_b + 1e-8)


# =============================================================================
# PROJECTION OPERATORS
# =============================================================================

class Hunyuan3DEmbeddingProject:
    """Project embedding A onto the direction of embedding B.

    Returns the component of A that lies in B's direction.
    Useful for extracting "how much of B is in A".
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_a": ("EMBEDDING",),
                "embedding_b": ("EMBEDDING",),
            },
            "optional": {
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("EMBEDDING", "FLOAT")
    RETURN_NAMES = ("projection", "similarity")
    FUNCTION = "project"
    CATEGORY = "3d/meltdown-spectre/primitives"

    def project(self, embedding_a, embedding_b, scale=1.0):
        result = {}
        similarity_val = 0.0

        for key in embedding_a.keys():
            if isinstance(embedding_a[key], torch.Tensor):
                a = embedding_a[key]
                b = embedding_b[key]

                # Flatten for projection
                a_flat, orig_shape = flatten_embedding(a)
                b_flat, _ = flatten_embedding(b)

                # Projection: proj_b(a) = (a . b / b . b) * b
                dot_ab = torch.sum(a_flat * b_flat, dim=-1, keepdim=True)
                dot_bb = torch.sum(b_flat * b_flat, dim=-1, keepdim=True)
                proj = (dot_ab / (dot_bb + 1e-8)) * b_flat

                # Compute cosine similarity
                similarity_val = (dot_ab / (torch.norm(a_flat, dim=-1, keepdim=True) *
                                           torch.norm(b_flat, dim=-1, keepdim=True) + 1e-8)).mean().item()

                result[key] = unflatten_embedding(proj * scale, orig_shape)
            else:
                result[key] = embedding_a[key]

        return (result, similarity_val)


class Hunyuan3DEmbeddingReject:
    """Compute the rejection of A from B (orthogonal component).

    Returns the part of A that is perpendicular to B.
    Useful for "removing B's influence from A".
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_a": ("EMBEDDING",),
                "embedding_b": ("EMBEDDING",),
            },
            "optional": {
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("EMBEDDING",)
    RETURN_NAMES = ("rejection",)
    FUNCTION = "reject"
    CATEGORY = "3d/meltdown-spectre/primitives"

    def reject(self, embedding_a, embedding_b, scale=1.0):
        result = {}

        for key in embedding_a.keys():
            if isinstance(embedding_a[key], torch.Tensor):
                a = embedding_a[key]
                b = embedding_b[key]

                a_flat, orig_shape = flatten_embedding(a)
                b_flat, _ = flatten_embedding(b)

                # Projection
                dot_ab = torch.sum(a_flat * b_flat, dim=-1, keepdim=True)
                dot_bb = torch.sum(b_flat * b_flat, dim=-1, keepdim=True)
                proj = (dot_ab / (dot_bb + 1e-8)) * b_flat

                # Rejection = A - projection
                rej = a_flat - proj

                result[key] = unflatten_embedding(rej * scale, orig_shape)
            else:
                result[key] = embedding_a[key]

        return (result,)


class Hunyuan3DEmbeddingOrthogonalize:
    """Make embedding A orthogonal to embedding B (Gram-Schmidt step).

    Similar to reject but normalizes the result.
    Useful for ensuring A has no component in B's direction.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_a": ("EMBEDDING",),
                "embedding_b": ("EMBEDDING",),
            },
            "optional": {
                "preserve_norm": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("EMBEDDING",)
    RETURN_NAMES = ("orthogonalized",)
    FUNCTION = "orthogonalize"
    CATEGORY = "3d/meltdown-spectre/primitives"

    def orthogonalize(self, embedding_a, embedding_b, preserve_norm=True):
        result = {}

        for key in embedding_a.keys():
            if isinstance(embedding_a[key], torch.Tensor):
                a = embedding_a[key]
                b = embedding_b[key]

                a_flat, orig_shape = flatten_embedding(a)
                b_flat, _ = flatten_embedding(b)

                # Original norm
                orig_norm = torch.norm(a_flat, dim=-1, keepdim=True)

                # Gram-Schmidt: a_orth = a - proj_b(a)
                dot_ab = torch.sum(a_flat * b_flat, dim=-1, keepdim=True)
                dot_bb = torch.sum(b_flat * b_flat, dim=-1, keepdim=True)
                proj = (dot_ab / (dot_bb + 1e-8)) * b_flat
                orth = a_flat - proj

                # Optionally preserve original norm
                if preserve_norm:
                    new_norm = torch.norm(orth, dim=-1, keepdim=True)
                    orth = orth / (new_norm + 1e-8) * orig_norm

                result[key] = unflatten_embedding(orth, orig_shape)
            else:
                result[key] = embedding_a[key]

        return (result,)


# =============================================================================
# SPECTRAL OPERATORS (SVD-based)
# =============================================================================

class Hunyuan3DEmbeddingSpectralDecompose:
    """Decompose embedding using SVD into U, S, V components.

    Returns spectral components that can be filtered and recombined.
    - U: Left singular vectors (spatial structure)
    - S: Singular values (importance/energy of each component)
    - V: Right singular vectors (feature patterns)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding": ("EMBEDDING",),
            }
        }

    RETURN_TYPES = ("SPECTRAL_COMPONENTS", "EMBEDDING_INFO")
    RETURN_NAMES = ("spectral", "info")
    FUNCTION = "decompose"
    CATEGORY = "3d/meltdown-spectre/spectral"

    def decompose(self, embedding):
        main_emb = embedding['main']
        flat, orig_shape = flatten_embedding(main_emb)

        # SVD decomposition
        U, S, Vh = torch.linalg.svd(flat, full_matrices=False)

        spectral = {
            'U': U,
            'S': S,
            'Vh': Vh,
            'original_shape': orig_shape,
            'device': main_emb.device,
            'dtype': main_emb.dtype,
            'other_keys': {k: v for k, v in embedding.items() if k != 'main'}
        }

        info = {
            'n_components': len(S),
            'energy_distribution': (S / S.sum()).tolist()[:10],  # Top 10 normalized
            'total_energy': S.sum().item(),
            'shape': list(orig_shape),
        }

        return (spectral, info)


class Hunyuan3DEmbeddingSpectralFilter:
    """Filter spectral components by singular value magnitude.

    Modes:
    - high: Keep large singular values (coarse structure, dominant features)
    - low: Keep small singular values (fine details, subtle textures)
    - band: Keep middle range (selective frequency band)
    - custom: Use explicit mask on components
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "spectral": ("SPECTRAL_COMPONENTS",),
                "mode": (["high", "low", "band", "notch"],),
            },
            "optional": {
                "cutoff": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "cutoff_high": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "sharpness": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("SPECTRAL_COMPONENTS",)
    RETURN_NAMES = ("filtered",)
    FUNCTION = "filter"
    CATEGORY = "3d/meltdown-spectre/spectral"

    def filter(self, spectral, mode="high", cutoff=0.5, cutoff_high=0.8, sharpness=10.0):
        S = spectral['S'].clone()
        n = len(S)

        # Create normalized position (0 = first component, 1 = last)
        positions = torch.arange(n, device=S.device, dtype=S.dtype) / (n - 1 + 1e-8)

        # Create filter mask based on mode
        if mode == "high":
            # Sigmoid filter: keeps components before cutoff
            mask = torch.sigmoid(sharpness * (cutoff - positions))
        elif mode == "low":
            # Inverse: keeps components after cutoff
            mask = torch.sigmoid(sharpness * (positions - cutoff))
        elif mode == "band":
            # Band-pass: between cutoff and cutoff_high
            low_mask = torch.sigmoid(sharpness * (positions - cutoff))
            high_mask = torch.sigmoid(sharpness * (cutoff_high - positions))
            mask = low_mask * high_mask
        elif mode == "notch":
            # Notch filter: removes band between cutoff and cutoff_high
            in_band = torch.sigmoid(sharpness * (positions - cutoff)) * \
                      torch.sigmoid(sharpness * (cutoff_high - positions))
            mask = 1.0 - in_band

        # Apply mask to singular values
        S_filtered = S * mask

        filtered = {
            'U': spectral['U'],
            'S': S_filtered,
            'Vh': spectral['Vh'],
            'original_shape': spectral['original_shape'],
            'device': spectral['device'],
            'dtype': spectral['dtype'],
            'other_keys': spectral['other_keys'],
            'filter_mask': mask,
        }

        return (filtered,)


class Hunyuan3DEmbeddingSpectralRecombine:
    """Reconstruct embedding from (filtered) spectral components.

    Can combine components from different sources for spectral splicing.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "spectral": ("SPECTRAL_COMPONENTS",),
            },
            "optional": {
                "normalize": ("BOOLEAN", {"default": False}),
                "target_norm": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("EMBEDDING",)
    RETURN_NAMES = ("embedding",)
    FUNCTION = "recombine"
    CATEGORY = "3d/meltdown-spectre/spectral"

    def recombine(self, spectral, normalize=False, target_norm=1.0):
        U = spectral['U']
        S = spectral['S']
        Vh = spectral['Vh']
        orig_shape = spectral['original_shape']

        # Reconstruct: U @ diag(S) @ Vh
        reconstructed = U @ torch.diag(S) @ Vh

        if normalize:
            norm = torch.norm(reconstructed, dim=-1, keepdim=True)
            reconstructed = reconstructed / (norm + 1e-8) * target_norm

        # Unflatten
        main_tensor = unflatten_embedding(reconstructed, orig_shape)

        # Rebuild embedding dict
        result = {'main': main_tensor}
        result.update(spectral['other_keys'])

        return (result,)


class Hunyuan3DEmbeddingSpectralSplice:
    """Combine spectral components from two embeddings.

    Take high-frequency (structure) from A and low-frequency (detail) from B,
    or any other combination based on splice_point.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "spectral_a": ("SPECTRAL_COMPONENTS",),
                "spectral_b": ("SPECTRAL_COMPONENTS",),
            },
            "optional": {
                "splice_point": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "blend_width": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05}),
                "source_high": (["A", "B"],),
            }
        }

    RETURN_TYPES = ("SPECTRAL_COMPONENTS",)
    RETURN_NAMES = ("spliced",)
    FUNCTION = "splice"
    CATEGORY = "3d/meltdown-spectre/spectral"

    def splice(self, spectral_a, spectral_b, splice_point=0.5, blend_width=0.1, source_high="A"):
        S_a = spectral_a['S']
        S_b = spectral_b['S']
        n = len(S_a)

        positions = torch.arange(n, device=S_a.device, dtype=S_a.dtype) / (n - 1 + 1e-8)

        # Create smooth blend mask
        if blend_width > 0:
            sharpness = 1.0 / blend_width
            blend = torch.sigmoid(sharpness * (positions - splice_point))
        else:
            blend = (positions >= splice_point).float()

        # Determine which source gets high vs low
        if source_high == "A":
            # A gets high (structure), B gets low (detail)
            mask_a = 1.0 - blend
            mask_b = blend
        else:
            # B gets high, A gets low
            mask_a = blend
            mask_b = 1.0 - blend

        # Blend singular values
        S_spliced = S_a * mask_a + S_b * mask_b

        # Use U from high-frequency source, Vh from low-frequency source (or blend)
        # For simplicity, blend U and Vh as well
        U_spliced = spectral_a['U'] * mask_a.unsqueeze(0) + spectral_b['U'] * mask_b.unsqueeze(0)
        Vh_spliced = spectral_a['Vh'] * mask_a.unsqueeze(-1) + spectral_b['Vh'] * mask_b.unsqueeze(-1)

        spliced = {
            'U': U_spliced,
            'S': S_spliced,
            'Vh': Vh_spliced,
            'original_shape': spectral_a['original_shape'],
            'device': spectral_a['device'],
            'dtype': spectral_a['dtype'],
            'other_keys': spectral_a['other_keys'],
        }

        return (spliced,)


# =============================================================================
# SET OPERATIONS
# =============================================================================

class Hunyuan3DEmbeddingIntersection:
    """Find the common ground between two embeddings.

    Projects A onto B's direction and weights by similarity.
    Returns what A and B "agree on".
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_a": ("EMBEDDING",),
                "embedding_b": ("EMBEDDING",),
            },
            "optional": {
                "method": (["projection", "minimum", "geometric_mean"],),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("EMBEDDING", "FLOAT")
    RETURN_NAMES = ("intersection", "overlap_score")
    FUNCTION = "intersect"
    CATEGORY = "3d/meltdown-spectre/set-ops"

    def intersect(self, embedding_a, embedding_b, method="projection", scale=1.0):
        result = {}
        overlap = 0.0

        for key in embedding_a.keys():
            if isinstance(embedding_a[key], torch.Tensor):
                a = embedding_a[key]
                b = embedding_b[key]

                a_flat, orig_shape = flatten_embedding(a)
                b_flat, _ = flatten_embedding(b)

                # Cosine similarity
                sim = cosine_similarity_tensors(a, b).mean().item()
                overlap = max(0, sim)  # Clamp to positive

                if method == "projection":
                    # Project A onto B, weighted by similarity
                    dot_ab = torch.sum(a_flat * b_flat, dim=-1, keepdim=True)
                    dot_bb = torch.sum(b_flat * b_flat, dim=-1, keepdim=True)
                    proj = (dot_ab / (dot_bb + 1e-8)) * b_flat
                    intersection = proj * max(0, sim)

                elif method == "minimum":
                    # Element-wise minimum (intersection in set theory sense)
                    intersection = torch.minimum(a_flat, b_flat)

                elif method == "geometric_mean":
                    # Geometric mean weighted by similarity
                    # Handles sign by using signed sqrt
                    sign = torch.sign(a_flat * b_flat)
                    abs_geom = torch.sqrt(torch.abs(a_flat) * torch.abs(b_flat))
                    intersection = sign * abs_geom * max(0, sim)

                result[key] = unflatten_embedding(intersection * scale, orig_shape)
            else:
                result[key] = embedding_a[key]

        return (result, overlap)


class Hunyuan3DEmbeddingUnion:
    """Combine unique aspects of both embeddings.

    Computes A + B - intersection, capturing everything in either.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_a": ("EMBEDDING",),
                "embedding_b": ("EMBEDDING",),
            },
            "optional": {
                "intersection_method": (["projection", "minimum", "geometric_mean"],),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("EMBEDDING",)
    RETURN_NAMES = ("union",)
    FUNCTION = "union"
    CATEGORY = "3d/meltdown-spectre/set-ops"

    def union(self, embedding_a, embedding_b, intersection_method="projection", scale=1.0):
        # First compute intersection
        intersector = Hunyuan3DEmbeddingIntersection()
        intersection, _ = intersector.intersect(embedding_a, embedding_b, intersection_method)

        result = {}
        for key in embedding_a.keys():
            if isinstance(embedding_a[key], torch.Tensor):
                # Union = A + B - intersection
                union_val = embedding_a[key] + embedding_b[key] - intersection[key]
                result[key] = union_val * scale
            else:
                result[key] = embedding_a[key]

        return (result,)


class Hunyuan3DEmbeddingXOR:
    """Symmetric difference: what's unique to each embedding.

    Returns aspects that are in A or B but not both.
    Useful for extracting distinctive features.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_a": ("EMBEDDING",),
                "embedding_b": ("EMBEDDING",),
            },
            "optional": {
                "method": (["rejection_sum", "abs_difference"],),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("EMBEDDING",)
    RETURN_NAMES = ("xor",)
    FUNCTION = "xor"
    CATEGORY = "3d/meltdown-spectre/set-ops"

    def xor(self, embedding_a, embedding_b, method="rejection_sum", scale=1.0):
        result = {}

        for key in embedding_a.keys():
            if isinstance(embedding_a[key], torch.Tensor):
                a = embedding_a[key]
                b = embedding_b[key]

                a_flat, orig_shape = flatten_embedding(a)
                b_flat, _ = flatten_embedding(b)

                if method == "rejection_sum":
                    # Reject A from B + Reject B from A
                    # A unique = A - proj_b(A)
                    dot_ab = torch.sum(a_flat * b_flat, dim=-1, keepdim=True)
                    dot_bb = torch.sum(b_flat * b_flat, dim=-1, keepdim=True)
                    dot_aa = torch.sum(a_flat * a_flat, dim=-1, keepdim=True)

                    proj_a_on_b = (dot_ab / (dot_bb + 1e-8)) * b_flat
                    proj_b_on_a = (dot_ab / (dot_aa + 1e-8)) * a_flat

                    unique_a = a_flat - proj_a_on_b
                    unique_b = b_flat - proj_b_on_a
                    xor_result = unique_a + unique_b

                elif method == "abs_difference":
                    # Simple absolute difference
                    xor_result = torch.abs(a_flat - b_flat)

                result[key] = unflatten_embedding(xor_result * scale, orig_shape)
            else:
                result[key] = embedding_a[key]

        return (result,)


# =============================================================================
# SPLICE OPERATORS
# =============================================================================

class Hunyuan3DEmbeddingSpliceSequence:
    """Splice embeddings by sequence position.

    Takes positions [0, splice_point) from A and [splice_point, end] from B.
    Useful when embeddings have spatial/sequential structure.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_a": ("EMBEDDING",),
                "embedding_b": ("EMBEDDING",),
            },
            "optional": {
                "splice_point": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "blend_tokens": ("INT", {"default": 0, "min": 0, "max": 50}),
            }
        }

    RETURN_TYPES = ("EMBEDDING",)
    RETURN_NAMES = ("spliced",)
    FUNCTION = "splice"
    CATEGORY = "3d/meltdown-spectre/splice"

    def splice(self, embedding_a, embedding_b, splice_point=0.5, blend_tokens=0):
        result = {}

        for key in embedding_a.keys():
            if isinstance(embedding_a[key], torch.Tensor):
                a = embedding_a[key]
                b = embedding_b[key]

                if len(a.shape) == 3:
                    B, S, D = a.shape
                    split_idx = int(S * splice_point)

                    if blend_tokens > 0 and split_idx > blend_tokens and split_idx < S - blend_tokens:
                        # Smooth blending around splice point
                        spliced = a.clone()
                        spliced[:, split_idx:, :] = b[:, split_idx:, :]

                        # Blend region
                        for i in range(blend_tokens):
                            t = (i + 1) / (blend_tokens + 1)
                            idx = split_idx - blend_tokens + i
                            if 0 <= idx < S:
                                spliced[:, idx, :] = (1 - t) * a[:, idx, :] + t * b[:, idx, :]
                    else:
                        # Hard splice
                        spliced = torch.cat([a[:, :split_idx, :], b[:, split_idx:, :]], dim=1)

                    result[key] = spliced
                else:
                    # For 2D, splice along feature dimension
                    D = a.shape[-1]
                    split_idx = int(D * splice_point)
                    result[key] = torch.cat([a[..., :split_idx], b[..., split_idx:]], dim=-1)
            else:
                result[key] = embedding_a[key]

        return (result,)


class Hunyuan3DEmbeddingMask:
    """Element-wise masking: A where mask > threshold, B otherwise.

    Mask can be a tensor or generated from embedding features.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_a": ("EMBEDDING",),
                "embedding_b": ("EMBEDDING",),
            },
            "optional": {
                "mask_source": (["magnitude_a", "magnitude_b", "magnitude_diff", "random"],),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "softness": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("EMBEDDING", "IMAGE")
    RETURN_NAMES = ("masked", "mask_preview")
    FUNCTION = "mask"
    CATEGORY = "3d/meltdown-spectre/splice"

    def mask(self, embedding_a, embedding_b, mask_source="magnitude_a",
             threshold=0.5, softness=0.1, seed=42):
        result = {}
        mask_preview = None

        for key in embedding_a.keys():
            if isinstance(embedding_a[key], torch.Tensor):
                a = embedding_a[key]
                b = embedding_b[key]

                a_flat, orig_shape = flatten_embedding(a)
                b_flat, _ = flatten_embedding(b)

                # Generate mask based on source
                if mask_source == "magnitude_a":
                    raw_mask = torch.norm(a_flat, dim=0)
                elif mask_source == "magnitude_b":
                    raw_mask = torch.norm(b_flat, dim=0)
                elif mask_source == "magnitude_diff":
                    raw_mask = torch.abs(torch.norm(a_flat, dim=0) - torch.norm(b_flat, dim=0))
                elif mask_source == "random":
                    torch.manual_seed(seed)
                    raw_mask = torch.rand(a_flat.shape[-1], device=a.device, dtype=a.dtype)

                # Normalize to [0, 1]
                raw_mask = (raw_mask - raw_mask.min()) / (raw_mask.max() - raw_mask.min() + 1e-8)

                # Apply threshold with softness
                if softness > 0:
                    mask = torch.sigmoid((raw_mask - threshold) / softness)
                else:
                    mask = (raw_mask >= threshold).float()

                # Apply mask: A where mask=1, B where mask=0
                masked = a_flat * mask + b_flat * (1 - mask)

                result[key] = unflatten_embedding(masked, orig_shape)

                # Create mask preview (only for main key)
                if key == 'main' and len(orig_shape) == 3:
                    B, S, D = orig_shape
                    # Reshape mask for visualization
                    mask_2d = mask.reshape(1, 1, S, D).expand(1, 3, S, D)
                    mask_preview = mask_2d.permute(0, 2, 3, 1).cpu()  # [1, S, D, 3]
            else:
                result[key] = embedding_a[key]

        if mask_preview is None:
            mask_preview = torch.zeros(1, 64, 64, 3)

        return (result, mask_preview)


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    # Projection operators
    "Hunyuan3DEmbeddingProject": Hunyuan3DEmbeddingProject,
    "Hunyuan3DEmbeddingReject": Hunyuan3DEmbeddingReject,
    "Hunyuan3DEmbeddingOrthogonalize": Hunyuan3DEmbeddingOrthogonalize,
    # Spectral operators
    "Hunyuan3DEmbeddingSpectralDecompose": Hunyuan3DEmbeddingSpectralDecompose,
    "Hunyuan3DEmbeddingSpectralFilter": Hunyuan3DEmbeddingSpectralFilter,
    "Hunyuan3DEmbeddingSpectralRecombine": Hunyuan3DEmbeddingSpectralRecombine,
    "Hunyuan3DEmbeddingSpectralSplice": Hunyuan3DEmbeddingSpectralSplice,
    # Set operations
    "Hunyuan3DEmbeddingIntersection": Hunyuan3DEmbeddingIntersection,
    "Hunyuan3DEmbeddingUnion": Hunyuan3DEmbeddingUnion,
    "Hunyuan3DEmbeddingXOR": Hunyuan3DEmbeddingXOR,
    # Splice operators
    "Hunyuan3DEmbeddingSpliceSequence": Hunyuan3DEmbeddingSpliceSequence,
    "Hunyuan3DEmbeddingMask": Hunyuan3DEmbeddingMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Projection operators
    "Hunyuan3DEmbeddingProject": "Project A onto B",
    "Hunyuan3DEmbeddingReject": "Reject A from B (Orthogonal)",
    "Hunyuan3DEmbeddingOrthogonalize": "Orthogonalize (Gram-Schmidt)",
    # Spectral operators
    "Hunyuan3DEmbeddingSpectralDecompose": "Spectral Decompose (SVD)",
    "Hunyuan3DEmbeddingSpectralFilter": "Spectral Filter (High/Low/Band)",
    "Hunyuan3DEmbeddingSpectralRecombine": "Spectral Recombine",
    "Hunyuan3DEmbeddingSpectralSplice": "Spectral Splice (Structure/Detail)",
    # Set operations
    "Hunyuan3DEmbeddingIntersection": "Embedding Intersection (Common)",
    "Hunyuan3DEmbeddingUnion": "Embedding Union (Combined)",
    "Hunyuan3DEmbeddingXOR": "Embedding XOR (Unique)",
    # Splice operators
    "Hunyuan3DEmbeddingSpliceSequence": "Splice by Sequence Position",
    "Hunyuan3DEmbeddingMask": "Masked Blend",
}
