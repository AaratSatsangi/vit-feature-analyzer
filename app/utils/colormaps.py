import numpy as np
import matplotlib

COLORMAPS = ['magma', 'viridis', 'inferno', 'plasma', 'cividis', 'hot', 'jet', 'coolwarm']

def apply_colormap(values: np.ndarray, cmap_name: str = 'magma') -> np.ndarray:
    """Map 1D or 2D float array [0,1] → RGBA uint8 array [..., 4]."""
    cmap = matplotlib.colormaps[cmap_name]
    rgba = cmap(values)  # float in [0, 1] -> [..., 4]
    return (rgba * 255).astype(np.uint8)
