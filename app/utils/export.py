import os
import io
import numpy as np
from PIL import Image
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image_array(arr: np.ndarray, path: str):
    """Save an RGB float array [0,1] or uint8 array to PNG."""
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def save_qpixmap(pixmap: QPixmap, path: str):
    """Save a QPixmap to PNG."""
    if not pixmap.isNull():
        pixmap.save(path, "PNG")

def save_pdf_from_array(arr: np.ndarray, path: str, interpolation='bilinear'):
    """Save an RGB array to a publication-quality PDF using matplotlib."""
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
        
    h, w = arr.shape[:2]
    fig = plt.figure(figsize=(w/100, h/100), dpi=300)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    ax.imshow(arr, interpolation=interpolation)
    plt.savefig(path, format='pdf', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

def save_pdf_from_bytes(png_bytes: bytes, path: str):
    """Save a PNG byte string to a PDF using matplotlib."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    save_pdf_from_array(arr, path)
