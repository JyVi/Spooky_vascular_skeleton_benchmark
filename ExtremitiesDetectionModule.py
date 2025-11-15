'''
Author: Oscar Morand (LRE, CREATIS)
Date: October 2025
Description: Functions to compute extremities and junctions on 2d binary skeletons
'''

from skimage.measure import label
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import matplotlib.pyplot as plt
import warnings
from functools import singledispatch

from skimage.morphology import dilation, square


# ============================================
# Extremities detection
# ============================================

def detect_extremities(skel: np.ndarray) -> np.ndarray:
    """Detect extremities (final pixels) in a skeletonized image using a fast method.

    Args:
        skel (np.ndarray): The skeletonized image.

    Returns:
        np.ndarray: A binary mask of the extremities.
    """
    if skel.ndim != 2:
        raise ValueError("Input mask must be a 2D array.")
    if skel.dtype != bool:
        warnings.warn("Input mask is not boolean. Converting to boolean.")
        skel = (skel > 0)
    # Define the kernels for convolution
    kernels = np.array([
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ],
        [
            [64, 32, 16],
            [128, 0, 8],
            [1, 2, 4]
        ]
    ])

    _, kH, kW = kernels.shape

    # Compute padding sizes
    pad_h = kH // 2
    pad_w = kW // 2

    # Pad the input image
    padded_image = np.pad(skel, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    patches = sliding_window_view(padded_image, (kH, kW))

    # Perform convolution using einsum
    convolved = np.einsum('ijkl,mkl->mij', patches, kernels)

    # Get "simple" final pixels, those with only one neighbor in connectivity 8
    n_neighbors_c8 = convolved[0] * skel
    simple_final_pixels = n_neighbors_c8 == 1

    # Get "complex" final pixels, those with two neighbors in connectivity 8 but with special patterns
    binary_neighborhood = convolved[1] * skel
    consecutives_ones = ((binary_neighborhood << 1) & binary_neighborhood)
    complex_pixels = (consecutives_ones > 0) | (binary_neighborhood == 0b10000001)
    complex_final_pixels = (n_neighbors_c8 == 2) & complex_pixels

    # Combine both types of final pixels
    final = simple_final_pixels | complex_final_pixels
    return final

def get_number_of_extremities_oscar(skel: np.ndarray) -> int:
    """
    Get the number of extremities (final pixels) in a skeletonized image.

    Args:
        skel (np.ndarray): The skeletonized image.

    Returns:
        int: The number of extremities.
    """
    final_pixels = detect_extremities(skel)
    return np.sum(final_pixels)


# ============================================
# Junctions detection
# ============================================

def compute_neighbors_regions(skel: np.ndarray) -> np.ndarray:
    """Compute the number of neighboring regions for each pixel in the skeleton.

    Args:
        skel (np.ndarray): The skeletonized image.

    Returns:
        np.ndarray: The skeleton with each pixel value being the number of neighboring pixels.
    """
    neighbors_regions = np.zeros_like(skel)
    height, width = skel.shape
    skel_p = np.pad(skel, pad_width=2, mode='constant', constant_values=0)

    for i in range(height):
        for j in range(width):
            if skel_p[i + 2, j + 2] == 0:
                continue

            # Get a 5-pixels wide square patch of pixels
            patch = skel_p[i:i+5, j:j+5]

            # Only keep the pixels that are connected to the center pixel
            cc = label(patch, connectivity=2)
            center_label = cc[2, 2]
            center_component = (cc == center_label)

            # Get only the 1-pixel wide band of pixels that constitute the edge of the 5-pixels square patch
            neighbor_5 = center_component.copy()
            neighbor_5[1:4, 1:4] = 0

            # Get the number of connected components on this band, to get the number of independant branches
            _, n_neighbors = label(neighbor_5, connectivity=2, return_num=True)

            neighbors_regions[i, j] = n_neighbors

    return neighbors_regions


def _clean_neighbors_regions(real_neighbors: np.ndarray) -> np.ndarray:
    """
    Cleans the regions of more-than-2 neighbors pixels by keeping only the pixel with the maximum
    number of neighbors in each region.

    Args:
        real_neighbors (np.ndarray): The skeleton with each pixel value being the number of neighbors.

    Returns:
        np.ndarray: The cleaned skeleton with only one pixel per region of more-than-2 neighbors pixels.
    """

    # Indentify regions of more-than-2 neighbors pixels
    junctions = real_neighbors > 2
    neighbors_cc = label(junctions, connectivity=2)

    simple_neighbors = np.zeros_like(real_neighbors)
    for region_label in np.unique(neighbors_cc):
        if region_label == 0:
            continue

        # Get the mask of the current region
        region_mask = (neighbors_cc == region_label)

        # Get the outer edge of the region
        region_mask_dilated = dilation(region_mask, square(3))
        region_mask_edge = region_mask_dilated & (~region_mask)

        # Compute the number of neighboring edges for the region
        region_vessel_neighbors = region_mask_edge * real_neighbors
        _, n_neighbors = label(region_vessel_neighbors, connectivity=2, return_num=True)

        # Set the pixel at the mean position of the region to the number of neighbors
        x, y = np.where(region_mask)
        x_mean, y_mean = np.mean(x).astype(int), np.mean(y).astype(int)
        simple_neighbors[(x_mean, y_mean)] = n_neighbors

    return simple_neighbors


def compute_neighbors_count(skel: np.ndarray) -> int:
    """
    Detects the number of neighbors for each pixel in the skeletonized image.
    WARNING: this function was not formally tested, and there are still some errors,
    so the result is an approximation of the real neighbor count of junctions pixels,
    especially with complex shaped skeletons like in ROSE dataset, or with skeletons
    that weren't simplified, e.g. with manual annotators.

    Args:
        skel (np.ndarray): The input binary skeleton

    Returns:
        np.ndarray: The skeleton with each pixel value being an approximation of its number of neighbors
    """

    if not isinstance(skel, np.ndarray):
        raise ValueError(f"Input skeleton must be a numpy array, got {type(skel)}")
    if skel.ndim != 2:
        raise ValueError(f"Input skeleton must be a 2D array, got {skel.ndim}D array of shape {skel.shape}")

    if skel.dtype == np.bool:
        warnings.warn("Input skeleton is of boolean type. It will be converted to integer type for processing.")
        skel = skel.astype(int)
    if set(np.unique(skel)) != {0, 1}:
        warnings.warn("Input skeleton is not binary. It will be thresholded at >0 for processing.")
        skel = (skel > 0).astype(int)
    if np.sum(skel) == 0:
        warnings.warn("Input skeleton is empty. Returning the original skeleton.")
        return skel

    # First, detect extremities to add them back later
    extremities = detect_extremities(skel)

    # Compute the number of neighbors for each pixel in the skeleton
    neighbors_regions = compute_neighbors_regions(skel)

    # Right now, there are some regions of more-than-2 neighbors pixels that are clamped together, we only need to keep
    # the pixel that has the maximum number of neighbors
    simple_neighbors = _clean_neighbors_regions(neighbors_regions)

    # Add back the extremities
    simple_neighbors = simple_neighbors + extremities

    # Add back the normal skeleton pixels (2 neighbors pixels)
    final_neighbor_count = skel * 2
    final_neighbor_count[simple_neighbors > 0] = simple_neighbors[simple_neighbors > 0]

    # return the final skeleton with each pixel value being an approximation of the number of neighbors
    return final_neighbor_count



def plot_junctions(skel: np.ndarray, neighbor_counts: dict = None) -> None:
    """
    Plots the junctions detected in the skeleton.

    Args:
        neighbor_counts (dict): A dictionary with the count of junctions for each number of neighbors
    """

    if neighbor_counts is None:
        neighbor_counts = compute_neighbors_count(skel)

    neighbor_range = np.array([i for i in np.unique(neighbor_counts) if i > 0])

    plt.figure(figsize=(12, 12))
    plt.imshow(skel, cmap='gray')
    plt.axis('off')
    for n in neighbor_range:
        if n == 2:
            continue
        y_coords, x_coords = np.where(neighbor_counts == n)
        plt.scatter(x_coords, y_coords, s=20, label=f'Junctions with {n} neighbors')
    plt.legend()
    plt.show()



def count_junctions_oscar(skel: np.ndarray, max_n_neighbors: int = 8) -> dict:
    """
    Counts the number of junctions in the skeleton.

    Args:
        skel (np.ndarray): The input binary skeleton.

    Returns:
        dict: A dictionary with the count of junctions for each number of neighbors.
    """

    neighbor_counts = compute_neighbors_count(skel)

    junction_counts = {str(k): np.sum(neighbor_counts == k) for k in range(1, max_n_neighbors + 1)}

    return junction_counts



@singledispatch
def plot_junctions_distribution(arg) -> None:
    raise TypeError(f"Unsupported type: {type(arg)}")

@plot_junctions_distribution.register(dict)
def _(junctions: dict) -> None:
    """
    Plots the distribution of junctions in the skeleton.

    Args:
        junctions (dict): A dictionary with the count of junctions for each number of neighbors.
    """

    plt.figure(figsize=(12, 12))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    junctions = {k: v for k, v in junctions.items() if k != "2"}
    x = np.array([i for i in junctions.keys()])
    y = np.array([j for j in junctions.values()])
    plt.bar(x, y, color=colors[:len(x)])
    plt.xlabel('Junction degree')
    plt.ylabel('Number of junctions')
    plt.title('Junctions in Skeleton')
    plt.show()

@plot_junctions_distribution.register(np.ndarray)
def _(skel: np.ndarray) -> None:
    junctions = count_junctions(skel)
    plot_junctions_distribution(junctions)