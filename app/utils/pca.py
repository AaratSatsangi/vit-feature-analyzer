import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def build_pca_map(patch_tokens: torch.Tensor, h_feat: int, w_feat: int, bg_threshold: float, invert: bool = False) -> np.ndarray:
    """
    Compute PCA on patch tokens and return a mapped color image.
    patch_tokens: [N, D] tensor (usually unnormalized)
    Returns: [h_feat, w_feat, 3] numpy array in [0, 1] range.
    """
    N, D = patch_tokens.shape
    if N == 0:
        return np.zeros((h_feat, w_feat, 3), dtype=np.float32)

    device = patch_tokens.device

    # Optional: L2 normalize
    features = F.normalize(patch_tokens.float(), p=2, dim=-1)

    # 1st PCA component for foreground/background separation
    mean_feat = features.mean(dim=0, keepdim=True)
    centered = features - mean_feat
    
    # Use exact SVD rather than pca_lowrank to avoid randomized solver flutters
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    pca_feats = U[:, :3] * S[:3]  # projection is exactly U * S for the fitted data
    
    # Sign ambiguity resolution (to keep signs consistent across thresholds/runs)
    max_abs_idx = torch.argmax(pca_feats.abs(), dim=0)
    signs = torch.sign(pca_feats[max_abs_idx, torch.arange(pca_feats.shape[1])])
    pca_feats = pca_feats * signs

    if invert:
        pca_feats[:, 0] *= -1

    # --- Background suppression via first component threshold ---
    pc1 = pca_feats[:, 0]
    pc1_max = pc1.abs().max()
    if pc1_max > 1e-6:
        pc1_norm = pc1 / pc1_max
    else:
        pc1_norm = pc1
    mask = pc1_norm > bg_threshold

    if mask.sum() <= 3:
        # Not enough foreground points to run PCA
        pca_feats2 = pca_feats
        mask = torch.ones(N, dtype=torch.bool, device=device)
    else:
        # Re-fit PCA on foreground patches only for cleaner colors
        fg_features = features[mask]
        fg_mean = fg_features.mean(dim=0, keepdim=True)
        fg_centered = fg_features - fg_mean
        
        U_fg, S_fg, Vh_fg = torch.linalg.svd(fg_centered, full_matrices=False)
        pca_feats2 = torch.matmul(features - fg_mean, Vh_fg[:3].T)
        
        # Enforce deterministic signs on refit as well
        max_abs_idx2 = torch.argmax(pca_feats2.abs(), dim=0)
        signs2 = torch.sign(pca_feats2[max_abs_idx2, torch.arange(pca_feats2.shape[1])])
        pca_feats2 = pca_feats2 * signs2

    # Scale foreground PCA features to [0, 1]
    pca_rgb = pca_feats2.clone()
    for c in range(3):
        vmin = pca_feats2[mask, c].min()
        vmax = pca_feats2[mask, c].max()
        if vmax - vmin > 1e-6:
            pca_rgb[:, c] = (pca_feats2[:, c] - vmin) / (vmax - vmin)
        else:
            pca_rgb[:, c] = 0.0

    pca_rgb = torch.clamp(pca_rgb, 0, 1)
    pca_rgb[~mask] = 0.0

    pca_map_2d = pca_rgb.view(h_feat, w_feat, 3)
    return pca_map_2d.cpu().numpy()


def build_combined_pca_maps(results, thresholds: list[dict]) -> list[np.ndarray]:
    """
    Compute joint PCA across multiple images for globally consistent coloring.
    Mirrors the exact algorithm from _build_combined_pca_maps in ui_app.py.

    results:    list of InferenceResult objects (must have .patch_tokens, .h_feat, .w_feat,
                .img_display with shape [img_h, img_w, 3])
    thresholds: list of dicts {"t": float, "inv": bool}
    Returns:    list of PIL-upscaled numpy arrays [img_h, img_w, 3] in [0,1] range.
    """
    if not results or len(results) != len(thresholds):
        return []

    fg_masks = []
    all_fg_tokens = []

    # Step 1: Independent PCA per image to find foreground based on 1st principal component
    for res, thresh in zip(results, thresholds):
        tokens = res.patch_tokens.to(dtype=torch.float32)
        x_mean = tokens.mean(dim=0, keepdim=True)
        U, S, _ = torch.linalg.svd(tokens - x_mean, full_matrices=False)
        pca_feats = U[:, :3] * S[:3]

        max_abs_idx = torch.argmax(pca_feats.abs(), dim=0)
        signs = torch.sign(pca_feats[max_abs_idx, torch.arange(pca_feats.shape[1])])
        pca_feats = pca_feats * signs

        bg_val = thresh.get("t", 0.0)
        inv = thresh.get("inv", False)

        if inv:
            pca_feats[:, 0] *= -1

        pc1 = pca_feats[:, 0]
        pc1_max = pc1.abs().max()
        if pc1_max > 1e-6:
            pc1_norm = pc1 / pc1_max
        else:
            pc1_norm = pc1

        fg_mask = pc1_norm > bg_val
        fg_masks.append(fg_mask)

        if fg_mask.sum() > 0:
            all_fg_tokens.append(tokens[fg_mask])

    # Step 2: Joint PCA on concatenated foreground patches across all same-category images
    if len(all_fg_tokens) > 0 and sum(t.shape[0] for t in all_fg_tokens) > 3:
        cat_fg_tokens = torch.cat(all_fg_tokens, dim=0)
        cat_mean = cat_fg_tokens.mean(dim=0, keepdim=True)
        _, _, Vh2 = torch.linalg.svd(cat_fg_tokens - cat_mean, full_matrices=False)

        global_pca_feats2 = torch.matmul(cat_fg_tokens - cat_mean, Vh2[:3].T)

        max_abs_idx2 = torch.argmax(global_pca_feats2.abs(), dim=0)
        signs2 = torch.sign(global_pca_feats2[max_abs_idx2, torch.arange(global_pca_feats2.shape[1])])
        global_pca_feats2 = global_pca_feats2 * signs2

        vmin = global_pca_feats2.min(dim=0)[0]
        vmax = global_pca_feats2.max(dim=0)[0]

        range_val = vmax - vmin
        range_val[range_val < 1e-6] = 1.0

        pca_maps = []

        # Step 3: Reproject each image's tokens into global PCA space and normalize consistently
        for idx_i, res in enumerate(results):
            tokens = res.patch_tokens.to(dtype=torch.float32)
            fg_mask = fg_masks[idx_i]

            img_pca_feats2 = torch.matmul(tokens - cat_mean, Vh2[:3].T) * signs2

            pca_rgb = (img_pca_feats2 - vmin) / range_val
            pca_rgb = torch.clamp(pca_rgb, 0, 1)

            pca_rgb[~fg_mask] = 0.0

            pca_map = pca_rgb.view(res.h_feat, res.w_feat, 3).numpy()
            pca_map_uint8 = (pca_map * 255).astype(np.uint8)
            pca_pil = Image.fromarray(pca_map_uint8, mode="RGB")

            # Upscale to original image resolution
            img_h, img_w = res.img_display.shape[:2]
            pca_pil = pca_pil.resize((img_w, img_h), resample=Image.NEAREST)

            pca_maps.append(np.array(pca_pil).astype(np.float32) / 255.0)

        return pca_maps
    else:
        # Fallback: isolated per-image PCA maps
        fallback = []
        for res, thresh in zip(results, thresholds):
            single_map = build_pca_map(
                res.patch_tokens, res.h_feat, res.w_feat,
                bg_threshold=thresh.get("t", 0.0),
                invert=thresh.get("inv", False),
            )
            # Upscale to image resolution
            h_map, w_map = single_map.shape[:2]
            img_h, img_w = res.img_display.shape[:2]
            map_uint8 = (single_map * 255).astype(np.uint8)
            pil_map = Image.fromarray(map_uint8, mode="RGB")
            pil_map = pil_map.resize((img_w, img_h), resample=Image.NEAREST)
            fallback.append(np.array(pil_map).astype(np.float32) / 255.0)
        return fallback
