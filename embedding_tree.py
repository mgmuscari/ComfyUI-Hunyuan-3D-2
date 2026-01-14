# Embedding Tree Structure Nodes
# Hierarchical binary tree operations for multi-image composition
#
# Categories:
#   - Tree Building: Branch, Merge, Root
#   - Convenience: Multi-Meltdown, Aesthetic Blender
#   - Visualization: Tree Preview

import torch
import numpy as np
from typing import List, Dict, Any, Optional


# Import slerp helper
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
# TREE NODE DATA STRUCTURE
# =============================================================================

def create_tree_node(embedding, operation="leaf", label="", children=None):
    """Create a tree node wrapping an embedding with metadata."""
    return {
        'embedding': embedding,
        'operation': operation,
        'label': label,
        'children': children or [],
        'depth': 0 if children is None else max(c.get('depth', 0) for c in children) + 1
    }


# =============================================================================
# TREE BUILDING NODES
# =============================================================================

class Hunyuan3DEmbeddingBranch:
    """Create a named binary operation branch in the tree.

    Wraps a binary operation with a label for visualization.
    The result can be used as input to further tree operations.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_a": ("EMBEDDING",),
                "embedding_b": ("EMBEDDING",),
                "operation": (["subtract", "add", "lerp", "slerp", "project", "reject"],),
            },
            "optional": {
                "label": ("STRING", {"default": ""}),
                "strength": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.1}),
                "t": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("EMBEDDING", "TREE_NODE")
    RETURN_NAMES = ("result", "tree_node")
    FUNCTION = "branch"
    CATEGORY = "3d/meltdown-spectre/tree"

    def branch(self, embedding_a, embedding_b, operation="subtract",
               label="", strength=1.0, t=0.5):

        result = {}

        for key in embedding_a.keys():
            if isinstance(embedding_a[key], torch.Tensor):
                a = embedding_a[key]
                b = embedding_b[key]

                if operation == "subtract":
                    out = (a - b) * strength
                elif operation == "add":
                    out = (a + b) * strength
                elif operation == "lerp":
                    out = ((1 - t) * a + t * b) * strength
                elif operation == "slerp":
                    out = slerp(a, b, t) * strength
                elif operation == "project":
                    # Project A onto B
                    a_flat = a.reshape(a.shape[0], -1)
                    b_flat = b.reshape(b.shape[0], -1)
                    dot_ab = torch.sum(a_flat * b_flat, dim=-1, keepdim=True)
                    dot_bb = torch.sum(b_flat * b_flat, dim=-1, keepdim=True)
                    proj = (dot_ab / (dot_bb + 1e-8)) * b_flat
                    out = proj.reshape(a.shape) * strength
                elif operation == "reject":
                    # Reject A from B (orthogonal component)
                    a_flat = a.reshape(a.shape[0], -1)
                    b_flat = b.reshape(b.shape[0], -1)
                    dot_ab = torch.sum(a_flat * b_flat, dim=-1, keepdim=True)
                    dot_bb = torch.sum(b_flat * b_flat, dim=-1, keepdim=True)
                    proj = (dot_ab / (dot_bb + 1e-8)) * b_flat
                    rej = a_flat - proj
                    out = rej.reshape(a.shape) * strength
                else:
                    out = a  # Fallback

                result[key] = out
            else:
                result[key] = embedding_a[key]

        # Create tree node metadata
        op_symbol = {
            "subtract": "-",
            "add": "+",
            "lerp": f"lerp({t})",
            "slerp": f"slerp({t})",
            "project": "proj",
            "reject": "rej",
        }.get(operation, operation)

        auto_label = label if label else f"({op_symbol})"

        tree_node = create_tree_node(
            embedding=result,
            operation=operation,
            label=auto_label,
            children=[
                {'label': 'A', 'operation': 'leaf'},
                {'label': 'B', 'operation': 'leaf'}
            ]
        )

        return (result, tree_node)


class Hunyuan3DEmbeddingTreeMerge:
    """Merge two tree branches into a parent node.

    Combines results from two branch operations.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "branch_a": ("EMBEDDING",),
                "branch_b": ("EMBEDDING",),
                "operation": (["subtract", "add", "lerp", "slerp"],),
            },
            "optional": {
                "tree_node_a": ("TREE_NODE",),
                "tree_node_b": ("TREE_NODE",),
                "label": ("STRING", {"default": "merged"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.1}),
                "t": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("EMBEDDING", "TREE_NODE")
    RETURN_NAMES = ("merged", "tree_node")
    FUNCTION = "merge"
    CATEGORY = "3d/meltdown-spectre/tree"

    def merge(self, branch_a, branch_b, operation="subtract",
              tree_node_a=None, tree_node_b=None, label="merged",
              strength=1.0, t=0.5):

        result = {}

        for key in branch_a.keys():
            if isinstance(branch_a[key], torch.Tensor):
                a = branch_a[key]
                b = branch_b[key]

                if operation == "subtract":
                    out = (a - b) * strength
                elif operation == "add":
                    out = (a + b) * strength
                elif operation == "lerp":
                    out = ((1 - t) * a + t * b) * strength
                elif operation == "slerp":
                    out = slerp(a, b, t) * strength
                else:
                    out = a

                result[key] = out
            else:
                result[key] = branch_a[key]

        # Build tree structure
        children = []
        if tree_node_a:
            children.append(tree_node_a)
        else:
            children.append({'label': 'A', 'operation': 'leaf'})
        if tree_node_b:
            children.append(tree_node_b)
        else:
            children.append({'label': 'B', 'operation': 'leaf'})

        tree_node = create_tree_node(
            embedding=result,
            operation=operation,
            label=label,
            children=children
        )

        return (result, tree_node)


class Hunyuan3DEmbeddingTreeRoot:
    """Finalize tree and output embedding with visualization.

    Creates a summary of the tree structure for debugging.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding": ("EMBEDDING",),
            },
            "optional": {
                "tree_node": ("TREE_NODE",),
                "root_label": ("STRING", {"default": "root"}),
                "normalize": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("EMBEDDING", "STRING")
    RETURN_NAMES = ("embedding", "tree_structure")
    FUNCTION = "root"
    CATEGORY = "3d/meltdown-spectre/tree"

    def root(self, embedding, tree_node=None, root_label="root", normalize=False):
        result = {}

        for key in embedding.keys():
            if isinstance(embedding[key], torch.Tensor):
                out = embedding[key]

                if normalize:
                    flat = out.reshape(out.shape[0], -1)
                    norm = torch.norm(flat, dim=-1, keepdim=True)
                    if len(out.shape) == 3:
                        norm = norm.unsqueeze(-1)
                    out = out / (norm + 1e-8)

                result[key] = out
            else:
                result[key] = embedding[key]

        # Generate tree visualization string
        def tree_to_string(node, indent=0):
            if node is None:
                return "  " * indent + "[input]"

            prefix = "  " * indent
            label = node.get('label', 'node')
            op = node.get('operation', 'unknown')

            lines = [f"{prefix}{label} ({op})"]

            children = node.get('children', [])
            for child in children:
                lines.append(tree_to_string(child, indent + 1))

            return "\n".join(lines)

        if tree_node:
            tree_str = f"Tree Structure:\n{root_label}\n" + tree_to_string(tree_node, 1)
        else:
            tree_str = f"Tree Structure:\n{root_label}\n  [direct input]"

        return (result, tree_str)


# =============================================================================
# CONVENIENCE NODES
# =============================================================================

class Hunyuan3DMultiMeltdown:
    """Hierarchical subtraction for 4 images.

    Computes ((A-B) - (C-D)) for a balanced subtraction tree.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding_a": ("EMBEDDING",),
                "embedding_b": ("EMBEDDING",),
                "embedding_c": ("EMBEDDING",),
                "embedding_d": ("EMBEDDING",),
            },
            "optional": {
                "structure": (["((A-B)-(C-D))", "((A-C)-(B-D))", "(A-B-C-D)"],),
                "scale_inner": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "scale_outer": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("EMBEDDING", "STRING")
    RETURN_NAMES = ("meltdown", "structure_info")
    FUNCTION = "multi_meltdown"
    CATEGORY = "3d/meltdown-spectre/convenience"

    def multi_meltdown(self, embedding_a, embedding_b, embedding_c, embedding_d,
                       structure="((A-B)-(C-D))", scale_inner=1.0, scale_outer=1.0):

        result = {}

        for key in embedding_a.keys():
            if isinstance(embedding_a[key], torch.Tensor):
                a = embedding_a[key]
                b = embedding_b[key]
                c = embedding_c[key]
                d = embedding_d[key]

                if structure == "((A-B)-(C-D))":
                    left = (a - b) * scale_inner
                    right = (c - d) * scale_inner
                    out = (left - right) * scale_outer
                    info = "((A-B)-(C-D)): Left branch vs Right branch"

                elif structure == "((A-C)-(B-D))":
                    left = (a - c) * scale_inner
                    right = (b - d) * scale_inner
                    out = (left - right) * scale_outer
                    info = "((A-C)-(B-D)): Diagonal pairing"

                elif structure == "(A-B-C-D)":
                    # Sequential subtraction
                    out = a - b - c - d
                    out = out * scale_outer
                    info = "(A-B-C-D): Sequential subtraction"

                result[key] = out
            else:
                result[key] = embedding_a[key]

        return (result, info)


class Hunyuan3DAestheticBlender:
    """Preset mixing patterns for artistic effects.

    Provides intuitive presets for common multi-image compositions.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dominant": ("EMBEDDING",),
                "accent_1": ("EMBEDDING",),
            },
            "optional": {
                "accent_2": ("EMBEDDING",),
                "accent_3": ("EMBEDDING",),
                "preset": (["dominant_with_accents", "balanced_blend", "contrast_boost",
                           "extract_essence", "harmonic_mean"],),
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("EMBEDDING", "STRING")
    RETURN_NAMES = ("blended", "recipe")
    FUNCTION = "blend"
    CATEGORY = "3d/meltdown-spectre/convenience"

    def blend(self, dominant, accent_1, accent_2=None, accent_3=None,
              preset="dominant_with_accents", intensity=0.5):

        accents = [accent_1]
        if accent_2 is not None:
            accents.append(accent_2)
        if accent_3 is not None:
            accents.append(accent_3)

        result = {}
        recipe_parts = [f"Preset: {preset}", f"Intensity: {intensity}"]

        for key in dominant.keys():
            if isinstance(dominant[key], torch.Tensor):
                d = dominant[key]
                a_list = [acc[key] for acc in accents]

                if preset == "dominant_with_accents":
                    # Dominant + small contributions from accents
                    out = d.clone()
                    for i, a in enumerate(a_list):
                        weight = intensity / (i + 1)  # Decreasing weights
                        out = out + weight * (a - d)
                    recipe_parts.append(f"D + sum({intensity}*(Ai-D)/i)")

                elif preset == "balanced_blend":
                    # Equal weighted average
                    all_emb = [d] + a_list
                    out = sum(all_emb) / len(all_emb)
                    out = d + intensity * (out - d)
                    recipe_parts.append(f"D + {intensity}*(mean(all) - D)")

                elif preset == "contrast_boost":
                    # Dominant minus averaged accents
                    accent_mean = sum(a_list) / len(a_list)
                    out = d + intensity * (d - accent_mean)
                    recipe_parts.append(f"D + {intensity}*(D - mean(accents))")

                elif preset == "extract_essence":
                    # What's unique to dominant vs all accents
                    accent_mean = sum(a_list) / len(a_list)
                    diff = d - accent_mean
                    norm_d = torch.norm(d.reshape(d.shape[0], -1), dim=-1, keepdim=True)
                    if len(d.shape) == 3:
                        norm_d = norm_d.unsqueeze(-1)
                    out = diff / (torch.norm(diff.reshape(diff.shape[0], -1), dim=-1, keepdim=True).unsqueeze(-1) + 1e-8) * norm_d * intensity
                    recipe_parts.append(f"normalize(D - mean(accents)) * {intensity}")

                elif preset == "harmonic_mean":
                    # Harmonic mean emphasizes common elements
                    all_emb = [d] + a_list
                    # Avoid division by zero
                    harmonic = len(all_emb) / sum(1.0 / (torch.abs(e) + 1e-8) for e in all_emb)
                    out = d + intensity * (harmonic - d)
                    recipe_parts.append(f"D + {intensity}*(harmonic_mean - D)")

                result[key] = out
            else:
                result[key] = dominant[key]

        return (result, "\n".join(recipe_parts))


class Hunyuan3DExpressionEval:
    """Evaluate simple embedding expressions.

    Supports: A, B, C, D as inputs
    Operations: +, -, *, numbers, parentheses
    Examples: "(A - B) * 0.5 + C", "A + B - 2*C"
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "expression": ("STRING", {"default": "(A - B) * 0.5 + C"}),
                "A": ("EMBEDDING",),
            },
            "optional": {
                "B": ("EMBEDDING",),
                "C": ("EMBEDDING",),
                "D": ("EMBEDDING",),
            }
        }

    RETURN_TYPES = ("EMBEDDING", "STRING")
    RETURN_NAMES = ("result", "parsed_expression")
    FUNCTION = "evaluate"
    CATEGORY = "3d/meltdown-spectre/convenience"

    def evaluate(self, expression, A, B=None, C=None, D=None):
        # Simple expression parser for embedding arithmetic
        # This is a basic implementation - production would use proper parsing

        variables = {'A': A, 'B': B, 'C': C, 'D': D}

        # Clean expression
        expr = expression.strip()

        # Tokenize (very simple)
        import re
        tokens = re.findall(r'[A-D]|[+\-*/]|[0-9.]+|[()]', expr)

        # For safety, evaluate only supported operations
        result = None
        reference = A

        try:
            # Build computation by parsing tokens
            # This is a simplified evaluator - handles basic cases
            current = None
            operation = None
            multiplier = 1.0

            i = 0
            while i < len(tokens):
                token = tokens[i]

                if token in 'ABCD':
                    var = variables.get(token)
                    if var is None:
                        raise ValueError(f"Variable {token} not provided")

                    tensor_result = {}
                    for key in var.keys():
                        if isinstance(var[key], torch.Tensor):
                            val = var[key] * multiplier
                            multiplier = 1.0

                            if current is None:
                                tensor_result[key] = val
                            elif operation == '+':
                                tensor_result[key] = current[key] + val
                            elif operation == '-':
                                tensor_result[key] = current[key] - val
                            elif operation == '*':
                                tensor_result[key] = current[key] * val
                            else:
                                tensor_result[key] = val
                        else:
                            tensor_result[key] = var[key]

                    current = tensor_result

                elif token in '+-':
                    operation = token

                elif token == '*':
                    # Check if next token is a number
                    if i + 1 < len(tokens) and re.match(r'[0-9.]+', tokens[i + 1]):
                        multiplier = float(tokens[i + 1])
                        i += 1
                    else:
                        operation = '*'

                elif re.match(r'[0-9.]+', token):
                    # Number - apply as multiplier to next variable
                    multiplier = float(token)
                    if i + 1 < len(tokens) and tokens[i + 1] == '*':
                        i += 1  # Skip the *

                i += 1

            result = current if current else A

        except Exception as e:
            # On error, return A unchanged
            result = A
            expr = f"Error: {str(e)}"

        return (result, f"Parsed: {expr}")


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    # Tree building
    "Hunyuan3DEmbeddingBranch": Hunyuan3DEmbeddingBranch,
    "Hunyuan3DEmbeddingTreeMerge": Hunyuan3DEmbeddingTreeMerge,
    "Hunyuan3DEmbeddingTreeRoot": Hunyuan3DEmbeddingTreeRoot,
    # Convenience
    "Hunyuan3DMultiMeltdown": Hunyuan3DMultiMeltdown,
    "Hunyuan3DAestheticBlender": Hunyuan3DAestheticBlender,
    "Hunyuan3DExpressionEval": Hunyuan3DExpressionEval,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Tree building
    "Hunyuan3DEmbeddingBranch": "Tree Branch (Binary Op)",
    "Hunyuan3DEmbeddingTreeMerge": "Tree Merge (Combine Branches)",
    "Hunyuan3DEmbeddingTreeRoot": "Tree Root (Finalize)",
    # Convenience
    "Hunyuan3DMultiMeltdown": "Multi-Meltdown (4 Images)",
    "Hunyuan3DAestheticBlender": "Aesthetic Blender (Presets)",
    "Hunyuan3DExpressionEval": "Expression Evaluator",
}
