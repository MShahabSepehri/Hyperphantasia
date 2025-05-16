import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def pad_to_square(img_array, pad_color=255):
    """Pad a NumPy image array (HWC or HW) to make it square."""
    h, w = img_array.shape[:2]
    size = max(h, w)
    pad_vert = (size - h)
    pad_horiz = (size - w)
    pad_top = pad_vert // 2
    pad_bottom = pad_vert - pad_top
    pad_left = pad_horiz // 2
    pad_right = pad_horiz - pad_left

    if img_array.ndim == 2:
        padded = np.pad(img_array, ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode='constant', constant_values=pad_color)
    else:
        padded = np.pad(img_array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                        mode='constant', constant_values=pad_color)
    return padded

def save_square_image(image_path, output_path, output_size=(512, 512), pad_color=255, format='jpg'):
    """Load image, pad to square, resize, and save."""
    img = np.array(Image.open(image_path))
    square_img = pad_to_square(img, pad_color=pad_color)

    # Resize to desired output size
    square_img = Image.fromarray(square_img)
    square_img = square_img.resize(output_size, Image.Resampling.LANCZOS)

    # Save using matplotlib (optional for more control)
    plt.imsave(output_path, np.array(square_img), format=format)
