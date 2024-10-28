import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pywt
import pywt.data



def visualize_coeffs(coeffs, n_levels, image_shape):
    """
    Plots the DWT coefficients, grouping all coefficients of each level in one figure.

    Parameters:
    - coeffs: list
        Wavelet coefficients from dwt2d function.
    - n_levels: int
        Number of decomposition levels.
    - image_shape: tuple
        Shape of the original image for size normalization.
    """
    import matplotlib.pyplot as plt

    # Get the maximum shape to normalize figure sizes
    max_shape = max(image_shape)
    max_figsize = 8  # Maximum figure size in inches

    # Plot the approximation coefficients (LL) at the deepest level
    cA = coeffs[0]
    shape = cA.shape
    figsize = (shape[1]/max_shape * max_figsize, shape[0]/max_shape * max_figsize)
    plt.figure(figsize=figsize)
    plt.imshow(cA, cmap='gray', aspect='auto')
    plt.title(f'Deepest Level Approximation (LL) -  Level {n_levels}')
    plt.axis('off')
    plt.show()

    # Iterate through each level's detail coefficients from highest to lowest level
    levels = n_levels
    for idx, (cH, cV, cD) in enumerate(coeffs[1:], start=1):
        # Level number from highest (1) to deepest (n_levels)
        level = levels - idx + 1

        detail_types = ['Horizontal detail (LH)', 'Vertical detail (HL)', 'Diagonal detail (HH)']
        coefficients = [cH, cV, cD]

        # Determine the figure size proportional to the array shape
        shape = cH.shape
        figsize = (shape[1]/max_shape * max_figsize * 3, shape[0]/max_shape * max_figsize)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        for ax, detail_type, c in zip(axes, detail_types, coefficients):
            ax.imshow(c, cmap='gray', aspect='auto')
            ax.set_title(f'{detail_type}')
            ax.axis('off')

        plt.suptitle(f'Detail Coefficients at Level {level}')
        plt.tight_layout()
        plt.show()

def visualize_coeffs_2D(coeffs, n_levels):
    # Function to draw rectangles around subbands and add labels on a single image

    # Convert the coefficients to a single array
    arr, slices = pywt.coeffs_to_array(coeffs)

    # Plot the coefficients array which is already organized in 2D proper form
    plt.figure(figsize=(8, 8))
    plt.imshow(arr, cmap='gray', extent=[0, arr.shape[1], arr.shape[0], 0])
    plt.title('Wavelet Coefficients organized in 2D')
    plt.axis('off')

    # Draw bounding boxes around the sub-bands and add text on each sub-band
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    subband_labels = {
        'aa': 'LL',  # Approximation coefficients
        'ad': 'HL',  # Horizontal details  ToDo: This is a hack. In fact ad is LH but couldn't fix the indexing properly
        'da': 'LH',  # Vertical details
        'dd': 'HH'   # Diagonal details
    }

    for level in range(len(slices)):
        color = colors[level % len(colors)]  # Cycle through colors
        if level == 0:
            # Approximation coefficients
            s = slices[0]
            y_start = s[0].start or 0
            y_end = s[0].stop or arr.shape[0]
            x_start = s[1].start or 0
            x_end = s[1].stop or arr.shape[1]
            width = x_end - x_start
            height = y_end - y_start

            # Draw rectangle
            rect = plt.Rectangle((x_start, y_start), width, height,
                                 edgecolor=color, facecolor='none', linewidth=1)
            plt.gca().add_patch(rect)

            # Add label at the center
            plt.text(x_start + width / 2, y_start + height / 2,
                     f'{subband_labels["aa"]}{n_levels - level}',
                     color=color, fontsize=max(width//10, 6), ha='center', va='center')
        else:
            # Detail coefficients at this level
            for key in ['ad', 'da', 'dd']:
                s = slices[level][key]
                y_start = s[0].start or 0
                y_end = s[0].stop or arr.shape[0]
                x_start = s[1].start or 0
                x_end = s[1].stop or arr.shape[1]
                width = x_end - x_start
                height = y_end - y_start

                # Draw rectangle
                rect = plt.Rectangle((x_start, y_start), width, height,
                                     edgecolor=color, facecolor='none', linewidth=1)
                plt.gca().add_patch(rect)

                # Add label at the center
                plt.text(x_start + width / 2, y_start + height / 2,
                         f'{subband_labels[key]}{n_levels - level + 1}',
                         color=color, fontsize=max(width//10, 6), ha='center', va='center',alpha=1.)
    plt.show()

def binarize_image(gray_image, threshold=.5):
    binary_image = np.where(gray_image >= threshold, 255, 0).astype(np.uint8)
    return binary_image


