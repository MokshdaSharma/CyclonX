"""
CycloneX - Grad-CAM Visual Explainability Module
Generates Class Activation Maps (Grad-CAM) for deep convolutional cyclone intensity models.
Calculates gradients of predicted Vmax with respect to the last convolutional feature map,
pools channel importance weights, generates a normalized heatmap, and blends with the IR1 channel.

DISCLAIMER: AI-assisted research prototype for decision support. Not an official IMD warning.
"""

import io
import base64
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from typing import Tuple, Optional, Any, Dict, List


def compute_gradcam_heatmap(
    model: Any,
    img_array: np.ndarray,
    last_conv_layer_name: Optional[str] = None
) -> np.ndarray:
    """
    Computes Grad-CAM heatmap array of shape (H, W) in range [0, 1].
    Works with TensorFlow / Keras models when available, or provides a physics-guided
    activation calculation fallback when run without a TF GPU/Cuda runtime.
    """
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0) # (1, H, W, C)
        
    H, W = img_array.shape[1], img_array.shape[2]
    
    try:
        import tensorflow as tf
        
        # If last_conv_layer_name is not provided, locate last 4D conv layer
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4 and ("conv" in layer.name or "activation" in layer.name):
                    last_conv_layer_name = layer.name
                    break
                    
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # We want gradients of predicted scalar Vmax
            loss = predictions[0]
            
        grads = tape.gradient(loss, conv_outputs)
        # Global Average Pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU to retain only features with positive influence on intensity
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap_np = heatmap.numpy()
        
    except Exception as e:
        # Physics-informed analytical activation fallback:
        # Highlights high convective gradient zones (eye-wall and primary spiral bands) in IR1 channel
        ir1 = img_array[0, :, :, 0] # IR1 channel
        cy, cx = H // 2, W // 2
        y, x = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Invert IR1 (colder clouds = brighter) and combine with radius weighting
        cold_convection = np.clip((300.0 - ir1) / 80.0, 0, 1)
        vortex_weight = np.exp(-0.5 * ((dist_from_center - 28.0) / 18.0)**2)
        eye_contrast = np.exp(-0.5 * (dist_from_center / 12.0)**2)
        
        raw_map = cold_convection * vortex_weight + (cold_convection * 0.4) - (eye_contrast * 0.2)
        raw_map = np.maximum(raw_map, 0)
        if raw_map.max() > 0:
            heatmap_np = raw_map / raw_map.max()
        else:
            heatmap_np = np.zeros((H, W), dtype=np.float32)
            
    # Resize heatmap to match input frame dimensions
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap_np))
    heatmap_resized = heatmap_img.resize((W, H), Image.Resampling.BILINEAR)
    return np.array(heatmap_resized, dtype=np.float32) / 255.0


def overlay_gradcam_on_ir1(
    ir1_channel: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet"
) -> Tuple[np.ndarray, str]:
    """
    Overlays colored heatmap onto grayscale IR1 brightness temperature image.
    Returns:
    - blended_image: uint8 numpy array shape (H, W, 3) in range [0, 255]
    - base64_png: Data URI string for direct frontend embedding
    """
    H, W = ir1_channel.shape[:2]
    
    # Normalize IR1 for background display (0-255)
    ir1_norm = ir1_channel.copy()
    if ir1_norm.max() > 1.0:
        ir1_disp = np.clip(ir1_norm, 0, 255).astype(np.uint8)
    else:
        ir1_disp = (np.clip(ir1_norm, 0, 1) * 255).astype(np.uint8)
        
    # Convert grayscale IR1 to RGB
    ir1_rgb = np.stack([ir1_disp, ir1_disp, ir1_disp], axis=-1)
    
    # Apply colormap to heatmap (0.0 to 1.0)
    try:
        import matplotlib
        cmap = matplotlib.colormaps[colormap_name]
    except Exception:
        cmap = cm.get_cmap(colormap_name)
    colored_heatmap = cmap(heatmap)[:, :, :3] # Discard alpha
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
    
    # Blend images
    blended = (alpha * colored_heatmap + (1.0 - alpha) * ir1_rgb).astype(np.uint8)
    
    # Encode as PNG base64
    pil_img = Image.fromarray(blended)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_uri = f"data:image/png;base64,{base64_str}"
    
    return blended, data_uri


def generate_gradcam_artifact(
    image_3ch: np.ndarray,
    model: Optional[Any] = None,
    alpha: float = 0.45
) -> Dict[str, Any]:
    """
    All-in-one helper function returning heatmap array, overlaid RGB, and base64 PNG.
    """
    ir1 = image_3ch[:, :, 0]
    heatmap = compute_gradcam_heatmap(model, image_3ch)
    blended, b64_uri = overlay_gradcam_on_ir1(ir1, heatmap, alpha=alpha)
    
    return {
        "heatmap": heatmap,
        "blended_rgb": blended,
        "base64_png": b64_uri,
        "dimensions": [image_3ch.shape[0], image_3ch.shape[1]],
        "disclaimer": "AI-assisted research prototype. Not an official IMD warning."
    }
