import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# Use Agg backend for headless rendering
matplotlib.use('Agg')

def _resize_rgb_float(arr: np.ndarray, size: tuple) -> np.ndarray:
    """Resize [H, W, 3] float array."""
    arr_uint8 = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr_uint8).resize(size, Image.Resampling.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0

def _resize_gray_float(arr: np.ndarray, size: tuple) -> np.ndarray:
    """Resize [H, W] float array."""
    img = Image.fromarray(arr).resize(size, Image.Resampling.BILINEAR)
    return np.array(img)

def render_attention_grid(raw_img: np.ndarray, attn_cls: np.ndarray, h_feat: int, w_feat: int) -> bytes:
    """
    Renders a matplotlib grid showing per-head attention overlays.
    Returns the rendered image as a PNG byte string.
    
    raw_img: [H, W, 3] float 0-1
    attn_cls: [num_heads, h_feat, w_feat]
    """
    num_heads = attn_cls.shape[0]
    
    # Resize original to display format
    h_px, w_px = raw_img.shape[:2]
    max_dim = 256
    scale = max_dim / max(h_px, w_px)
    if scale < 1.0:
        new_h, new_w = int(h_px * scale), int(w_px * scale)
        raw_img_disp = _resize_rgb_float(raw_img, (new_w, new_h))
    else:
        raw_img_disp = raw_img

    num_cols = min(6, num_heads + 3)
    num_rows = (num_heads + 3 + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
    axes = axes.flatten() if num_rows * num_cols > 1 else [axes]
    
    for ax in axes:
        ax.axis('off')
        
    # 1. Original Image
    axes[0].imshow(raw_img_disp)
    axes[0].set_title("Original", fontsize=10)
    
    # 2. Aggregate Attention (Sum of all heads)
    attn_agg = attn_cls.sum(axis=0) # [H, W]
    if attn_agg.max() - attn_agg.min() > 1e-8:
        attn_agg = (attn_agg - attn_agg.min()) / (attn_agg.max() - attn_agg.min())
    attn_agg_resized = _resize_gray_float(attn_agg, (raw_img_disp.shape[1], raw_img_disp.shape[0]))
    
    axes[1].imshow(raw_img_disp)
    axes[1].imshow(attn_agg_resized, cmap='inferno', alpha=0.6)
    axes[1].set_title("Aggregate Attention", fontsize=10)
    
    # 3. Average CLS Attention (Mean of all heads)
    attn_avg = attn_cls.mean(axis=0)
    if attn_avg.max() - attn_avg.min() > 1e-8:
        attn_avg = (attn_avg - attn_avg.min()) / (attn_avg.max() - attn_avg.min())
    attn_avg_resized = _resize_gray_float(attn_avg, (raw_img_disp.shape[1], raw_img_disp.shape[0]))
    
    axes[2].imshow(raw_img_disp)
    axes[2].imshow(attn_avg_resized, cmap='inferno', alpha=0.6)
    axes[2].set_title("CLS Token Avg", fontsize=10)
    
    # 4. Per-head attention
    for i in range(num_heads):
        ax_idx = i + 3
        head_attn = attn_cls[i]
        if head_attn.max() - head_attn.min() > 1e-8:
            head_attn = (head_attn - head_attn.min()) / (head_attn.max() - head_attn.min())
        head_attn_resized = _resize_gray_float(head_attn, (raw_img_disp.shape[1], raw_img_disp.shape[0]))
        
        axes[ax_idx].imshow(raw_img_disp)
        axes[ax_idx].imshow(head_attn_resized, cmap='inferno', alpha=0.6)
        axes[ax_idx].set_title(f"Head {i}", fontsize=10)
        
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='#1e293b')
    plt.close(fig)
    
    return buf.getvalue()
