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


def visualize_coeffs(coeffs, n_levels, image_shape, max_figsize):
    # Function to draw rectangles around subbands and add labels in individual images
    # Get the maximum shape to normalize figure sizes
    max_shape = max(image_shape)

    # Now, for each coefficient array, plot it in a separate figure
    for level, coeff in enumerate(coeffs):
        if level == 0:
            # Approximation coefficients at the coarsest level
            cA = coeff
            # Determine the figure size proportional to the array shape
            shape = cA.shape
            figsize = (shape[1]/max_shape * max_figsize, shape[0]/max_shape * max_figsize)  # width, height in inches
            plt.figure(figsize=figsize)
            plt.imshow(cA, cmap='gray', aspect='auto')
            plt.title(f'Level {n_levels}: LL (Approximation Coefficients)')
            plt.axis('off')
            plt.show()
        else:
            # Detail coefficients at this level
            cH, cV, cD = coeff
            lev = n_levels - level + 1  # Adjust level numbering
            detail_types = ['LH (Horizontal', 'HL (Vertical', 'HH (Diagonal']
            coefficients = [cH, cV, cD]
            for detail_type, c in zip(detail_types, coefficients):
                shape = c.shape
                figsize = (shape[1]/max_shape * max_figsize, shape[0]/max_shape * max_figsize)
                plt.figure(figsize=figsize)
                plt.imshow(c, cmap='gray', aspect='auto')
                plt.title(f'Level {lev}: {detail_type} Detail Coefficients)')
                plt.axis('off')
                plt.show()


def get_font_size(ax, base_size=8):
    """
    Calculate font size based on the dimensions of the subplot.
    
    Parameters:
    - ax: The subplot axes object.
    - base_size: A base font size for scaling.
    
    Returns:
    - An integer font size adjusted based on subplot dimensions.
    """
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    # Scale font size based on the subplot width
    font_size = base_size * (width + height) / 3  # Adjust the divisor to scale appropriately
    return int(font_size)


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


def filtering(input_image):
    # Vertical Edge Detection Filter (detects vertical edges)
    vertical_edge_filter = np.array([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]])

    # Horizontal Edge Detection Filter (detects horizontal edges)
    horizontal_edge_filter = np.array([[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]])

    # High-Pass Filter (Laplacian)
    high_pass_filter = np.array([[ 0, -1,  0],
                                [-1,  4, -1],
                                [ 0, -1,  0]])

    # Low-Pass Filter (Averaging)
    low_pass_filter = np.ones((3, 3)) / 9

    # Apply filters
    vertical_edges = convolve(input_image, vertical_edge_filter)
    horizontal_edges = convolve(input_image, horizontal_edge_filter)
    high_pass = convolve(input_image, high_pass_filter)
    low_pass = convolve(input_image, low_pass_filter)

    # Visualize results
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.imshow(input_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(vertical_edges, cmap='gray')
    plt.title('Vertical Edges')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(horizontal_edges, cmap='gray')
    plt.title('Horizontal Edges')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(high_pass, cmap='gray')
    plt.title('High-Pass Filter')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(low_pass, cmap='gray')
    plt.title('Low-Pass Filter')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def create_pattern_image(save_dir='.', image_width=512, image_height=512, grayscale=True):
    """
    Creates an image with various horizontal or vertical lines, checkerboard pattern, random noise, etc to evaluate image processing effects
    """
    # Set this parameter to True for grayscale image, False for color

    # Create a blank white image
    if grayscale:
        img = Image.new('L', (image_width, image_height), 'white')  # 'L' mode for grayscale
    else:
        img = Image.new('RGB', (image_width, image_height), 'white')
    draw = ImageDraw.Draw(img)

    # Define grid size
    rows = 3
    cols = 3
    section_width = image_width // cols
    section_height = image_height // rows

    # Adjust fill colors based on grayscale parameter
    black = 0 if grayscale else 'black'
    white = 255 if grayscale else 'white'
    gray = 128 if grayscale else 'gray'
    blue = 128 if grayscale else 'blue'  # Blue will appear as gray in grayscale
    green = 128 if grayscale else 'green'

    # Draw grid lines (optional)
    for i in range(1, cols):
        x = i * section_width
        draw.line([(x, 0), (x, image_height)], fill=black, width=1)
    for i in range(1, rows):
        y = i * section_height
        draw.line([(0, y), (image_width, y)], fill=black, width=1)

    ### Section (0, 0): Horizontal lines with varying widths ###
    x0, y0 = 0, 0
    num_lines = 10
    spacing = section_height // (num_lines + 1)
    for i in range(num_lines):
        y = y0 + spacing * (i + 1)
        width = (i % num_lines) + 1  # Width cycles from 1 to num_lines
        draw.line([(x0 + 10, y), (x0 + section_width - 10, y)], fill=black, width=width)

    ### Section (0, 1): Vertical lines with varying widths ###
    x0, y0 = section_width, 0
    num_lines = 10
    spacing = section_width // (num_lines + 1)
    for i in range(num_lines):
        x = x0 + spacing * (i + 1)
        width = (i % num_lines) + 1  # Width cycles from 1 to num_lines
        draw.line([(x, y0 + 10), (x, y0 + section_height - 10)], fill=black, width=width)

    ### Section (0, 2): Diagonal lines with varying widths ###
    x0, y0 = 2 * section_width, 0
    num_lines = 10
    for i in range(num_lines):
        offset = i * (section_width // num_lines)
        width = (i % num_lines) + 1
        draw.line(
            [(x0 + offset + 10, y0 + 10), (x0 + section_width - 10, y0 + section_height - offset - 10)],
            fill=black, width=width
        )
    for i in range(num_lines):
        offset = i * (section_width // num_lines)
        width = (i % num_lines) + 1
        draw.line(
            [(x0 + 10, y0 + offset + 10), (x0 + section_width - offset - 10, y0 + section_height - 10)],
            fill=black, width=width
        )

    ### Section (1, 0): Concentric Circles ###
    x0, y0 = 0, section_height
    center = (x0 + section_width // 2, y0 + section_height // 2)
    num_circles = 8
    max_radius = min(section_width, section_height) // 2
    radius_step = max_radius // num_circles
    for i in range(num_circles):
        radius = max_radius - i * radius_step
        width = (i % num_circles) + 1
        bbox = [
            center[0] - radius, center[1] - radius,
            center[0] + radius, center[1] + radius
        ]
        draw.ellipse(bbox, outline=black, width=width)

    ### Section (1, 1): Checkerboard Pattern ###
    x0, y0 = section_width, section_height
    checker_size = 30
    for x in range(x0, x0 + section_width, checker_size):
        for y in range(y0, y0 + section_height, checker_size):
            if ((x - x0) // checker_size + (y - y0) // checker_size) % 2 == 0:
                draw.rectangle([x, y, x + checker_size, y + checker_size], fill=gray)

    ### Section (1, 2): Repeated Sinusoidal Curves with Varying Thickness ###
    x0, y0 = 2 * section_width, section_height
    num_curves = 6
    amplitude = section_height // 4
    frequency = 2 * np.pi / (section_width // 2)
    for i in range(num_curves):
        x_values = np.linspace(x0, x0 + section_width, 1000)
        y_values = y0 + section_height // 2 + amplitude * np.sin(frequency * (x_values - x0 + 2*i) + i * np.pi / num_curves)
        points = list(zip(x_values, y_values))
        width = (i % num_curves) + 1
        draw.line(points, fill=blue, width=width)

    ### Section (2, 0): Concentric Squares ###
    x0, y0 = 0, 2 * section_height
    center = (x0 + section_width // 2, y0 + section_height // 2)
    num_squares = 8
    max_offset = min(section_width, section_height) // 2 - 10
    offset_step = max_offset // num_squares
    for i in range(num_squares):
        offset = i * offset_step
        width = (i % num_squares) + 1
        bbox = [
            center[0] - max_offset + offset, center[1] - max_offset + offset,
            center[0] + max_offset - offset, center[1] + max_offset - offset
        ]
        draw.rectangle(bbox, outline=green, width=width)

    ### Section (2, 1): Random Noise Pattern ###
    x0, y0 = section_width, 2 * section_height
    if grayscale:
        noise = np.random.randint(0, 256, (section_height, section_width), dtype=np.uint8)
        noise_img = Image.fromarray(noise, 'L')
    else:
        noise = np.random.randint(0, 256, (section_height, section_width, 3), dtype=np.uint8)
        noise_img = Image.fromarray(noise, 'RGB')
    img.paste(noise_img, (x0, y0))

    ### Section (2, 2): Radial Gradient ###
    x0, y0 = 2 * section_width, 2 * section_height
    for y in range(section_height):
        for x in range(section_width):
            dx = x - section_width // 2
            dy = y - section_height // 2
            distance = np.sqrt(dx**2 + dy**2)
            intensity = int(255 * (1 - distance / (np.sqrt(2) * section_width / 2)))
            intensity = max(0, min(255, intensity))
            if grayscale:
                img.putpixel((x0 + x, y0 + y), intensity)
            else:
                img.putpixel((x0 + x, y0 + y), (intensity, intensity, intensity))

    # Save and show the image
    if grayscale:
        img.save(os.path.join(save_dir, 'patterns_grayscale.png'))
    else:
        img.save(os.path.join(save_dir, 'patterns.png'))
    # img.show()
    return np.array(img)

