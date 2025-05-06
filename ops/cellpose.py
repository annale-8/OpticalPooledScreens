"""
Cellpose-based Image Segmentation
This module provides functions for segmenting microscopy images using the Cellpose algorithm
(relating to SBS base calling and phenotyping -- steps 1 and 2). It includes functions for:

1. Cell and Nuclei Segmentation: Segmenting cells and nuclei from various image types.
2. Image Preprocessing: Applying log scaling and other preprocessing techniques to images.
3. Label Reconciliation: Reconciling nuclei and cell labels based on their spatial relationships.
4. Mask Processing: Manipulating and refining segmentation masks.
5. Utility Functions: Supporting operations for image analysis and segmentation tasks.

"""

from cellpose.models import Cellpose
import numpy as np
import contextlib
import sys

from ops.annotate import relabel_array
from skimage.measure import regionprops
from skimage.segmentation import clear_border



def segment_cellpose(dapi, cyto, nuclei_diameter, cell_diameter, gpu=False, 
                     cyto_model='cyto', reconcile='consensus', logscale=True,
                     remove_edges=True):
    """
    Segment nuclei and cells using the Cellpose algorithm.

    Parameters:
        dapi (numpy.ndarray): DAPI channel image.
        cyto (numpy.ndarray): Cytoplasmic channel image.
        nuclei_diameter (int): Diameter of nuclei for segmentation.
        cell_diameter (int): Diameter of cells for segmentation.
        gpu (bool, optional): Whether to use GPU for segmentation. Default is False.
        cyto_model (str, optional): Type of cytoplasmic model to use. Default is 'cyto'.
        reconcile (str, optional): Method for reconciling nuclei and cells. Default is 'consensus'.
        logscale (bool, optional): Whether to apply log scaling to the cytoplasmic channel. Default is True.
        remove_edges (bool, optional): Whether to remove nuclei and cells touching the image edges. Default is True.

    Returns:
        tuple: A tuple containing:
            - nuclei (numpy.ndarray): Labeled segmentation mask of nuclei.
            - cells (numpy.ndarray): Labeled segmentation mask of cell boundaries.
    """
    # Disable Cellpose logger
    # import logging
    # logging.getLogger('cellpose').setLevel(logging.WARNING)

    # Apply log scaling to the cytoplasmic channel if specified
    if logscale:
        cyto = image_log_scale(cyto)

    # Create a two-channel image array from DAPI and cytoplasmic channels
    img = np.array([dapi, cyto])

    # Instantiate Cellpose models for nuclei and cytoplasmic segmentation
    model_dapi = Cellpose(model_type='nuclei', gpu=gpu)
    model_cyto = Cellpose(model_type=cyto_model, gpu=gpu)
    
    # Segment nuclei and cells using Cellpose
    nuclei, _, _, _ = model_dapi.eval(img, channels=[1, 0], diameter=nuclei_diameter)
    cells, _, _, _  = model_cyto.eval(img, channels=[2, 1], diameter=cell_diameter)
    
    # Print the number of nuclei and cells found before and after removing edges
    print(f'found {len(np.unique(nuclei))} nuclei before removing edges', file=sys.stderr)
    print(f'found {len(np.unique(cells))} cells before removing edges', file=sys.stderr)

    # Remove nuclei and cells touching the image edges if specified
    if remove_edges:
        nuclei = clear_border(nuclei)
        cells = clear_border(cells)
    
    # Print the number of nuclei and cells found before and after reconciliation
    print(f'found {len(np.unique(nuclei))} nuclei before reconciling', file=sys.stderr)
    print(f'found {len(np.unique(cells))} cells before reconciling', file=sys.stderr)
    
    # Reconcile nuclei and cells if specified
    if reconcile:
        nuclei, cells = reconcile_nuclei_cells(nuclei, cells, how=reconcile)
        print(f'found {len(np.unique(cells))} nuclei/cells after reconciling', file=sys.stderr)

    # Return the segmented nuclei and cells
    return nuclei, cells

def segment_cellpose_rgb(rgb, nuclei_diameter, cell_diameter, gpu=False, 
                         cyto_model='cyto', reconcile='consensus', logscale=True,
                         remove_edges=True, return_counts=False):
    """
    Segment nuclei and cells using the Cellpose algorithm from an RGB image.
    Parameters:
        rgb (numpy.ndarray): RGB image.
        nuclei_diameter (int): Diameter of nuclei for segmentation.
        cell_diameter (int): Diameter of cells for segmentation.
        gpu (bool, optional): Whether to use GPU for segmentation. Default is False.
        cyto_model (str, optional): Type of cytoplasmic model to use. Default is 'cyto'.
        reconcile (str, optional): Method for reconciling nuclei and cells. Default is 'consensus'.
        logscale (bool, optional): Whether to apply log scaling to the cytoplasmic channel. Default is True.
        remove_edges (bool, optional): Whether to remove nuclei and cells touching the image edges. Default is True.
        return_counts (bool, optional): Whether to return counts of nuclei and cells before reconciliation. Default is False.
    Returns:
        tuple: A tuple containing:
            - nuclei (numpy.ndarray): Labeled segmentation mask of nuclei.
            - cells (numpy.ndarray): Labeled segmentation mask of cell boundaries.
            - (optional) counts (dict): Counts of nuclei and cells at different stages if return_counts is True.
    """
    # Instantiate Cellpose models for nuclei and cytoplasmic segmentation
    model_dapi = Cellpose(model_type='nuclei', gpu=gpu)
    model_cyto = Cellpose(model_type=cyto_model, gpu=gpu)
    
    counts = {}
    
    # Segment nuclei and cells using Cellpose from the RGB image
    nuclei, _, _, _ = model_dapi.eval(rgb, channels=[3, 0], diameter=nuclei_diameter)
    cells, _, _, _ = model_cyto.eval(rgb, channels=[2, 3], diameter=cell_diameter)
    
    counts['initial_nuclei'] = len(np.unique(nuclei)) - 1  # Subtract 1 to exclude background
    counts['initial_cells'] = len(np.unique(cells)) - 1
    
    print(f'found {counts["initial_nuclei"]} nuclei before removing edges', file=sys.stderr)
    print(f'found {counts["initial_cells"]} cells before removing edges', file=sys.stderr)
    
    # Remove nuclei and cells touching the image edges if specified
    if remove_edges:
        print('removing edges')
        nuclei = clear_border(nuclei)
        cells = clear_border(cells)
    
    counts['after_edge_removal_nuclei'] = len(np.unique(nuclei)) - 1
    counts['after_edge_removal_cells'] = len(np.unique(cells)) - 1
    
    print(f'found {counts["after_edge_removal_nuclei"]} nuclei before reconciling', file=sys.stderr)
    print(f'found {counts["after_edge_removal_cells"]} cells before reconciling', file=sys.stderr)
    
    # Reconcile nuclei and cells if specified
    if reconcile:
        print(f'reconciling masks with method how={reconcile}')
        nuclei, cells = reconcile_nuclei_cells(nuclei, cells, how=reconcile)
        
    counts['final_cells'] = len(np.unique(cells)) - 1
    print(f'found {counts["final_cells"]} nuclei/cells after reconciling', file=sys.stderr)
    
    if return_counts:
        return nuclei, cells, counts
    else:
        return nuclei, cells


def segment_cellpose_nuclei_rgb(rgb, nuclei_diameter, gpu=False, 
                                remove_edges=True, **kwargs):
    """
    Segment nuclei using the Cellpose algorithm from an RGB image.

    Parameters:
        rgb (numpy.ndarray): RGB image.
        nuclei_diameter (int): Diameter of nuclei for segmentation.
        gpu (bool, optional): Whether to use GPU for segmentation. Default is False.
        remove_edges (bool, optional): Whether to remove nuclei touching the image edges. Default is True.
        **kwargs: Additional keyword arguments.

    Returns:
        numpy.ndarray: Labeled segmentation mask of nuclei.
    """
    # Instantiate Cellpose model for nuclei segmentation
    model_dapi = Cellpose(model_type='nuclei', gpu=gpu)
    
    # Segment nuclei using Cellpose from the RGB image
    nuclei, _, _, _ = model_dapi.eval(rgb, channels=[3, 0], diameter=nuclei_diameter)

    # Print the number of nuclei found before and after removing edges
    print(f'found {len(np.unique(nuclei))} nuclei before removing edges', file=sys.stderr)
    
    # Remove nuclei touching the image edges if specified
    if remove_edges:
        print('removing edges')
        nuclei = clear_border(nuclei)

    # Print the final number of nuclei after processing
    print(f'found {len(np.unique(nuclei))} final nuclei', file=sys.stderr)

    # Return the segmented nuclei
    return nuclei


def image_log_scale(data, bottom_percentile=10, floor_threshold=50, ignore_zero=True):
    """
    Apply log scaling to an image.

    Parameters:
        data (numpy.ndarray): Input image data.
        bottom_percentile (int, optional): Percentile value for determining the bottom threshold. Default is 10.
        floor_threshold (int, optional): Floor threshold for cutting out noisy bits. Default is 50.
        ignore_zero (bool, optional): Whether to ignore zero values in the data. Default is True.

    Returns:
        numpy.ndarray: Scaled image data after log scaling.
    """
    import numpy as np
    
    # Convert input data to float
    data = data.astype(float)
    
    # Select data based on whether to ignore zero values or not
    if ignore_zero:
        data_perc = data[data > 0]
    else:
        data_perc = data
    
    # Determine the bottom percentile value
    bottom = np.percentile(data_perc, bottom_percentile)
    
    # Set values below the bottom percentile to the bottom value
    data[data < bottom] = bottom
    
    # Apply log scaling with floor threshold
    scaled = np.log10(data - bottom + 1)
    
    # Cut out noisy bits based on the floor threshold
    floor = np.log10(floor_threshold)
    scaled[scaled < floor] = floor
    
    # Subtract the floor value
    return scaled - floor


def reconcile_nuclei_cells(nuclei, cells, how='consensus'):
    """
    Reconcile nuclei and cells labels based on their overlap.

    Parameters:
        nuclei (numpy.ndarray): Nuclei mask.
        cells (numpy.ndarray): Cell mask.
        how (str, optional): Method to reconcile labels. 
            - 'consensus': Only keep nucleus-cell pairs where label matches are unique.
            - 'contained_in_cells': Keep multiple nuclei for a single cell but merge them.

    Returns:
        tuple: Tuple containing the reconciled nuclei and cells masks.
    """
    from skimage.morphology import erosion

    def get_unique_label_map(regions, keep_multiple=False):
        """
        Get unique label map from regions.

        Parameters:
            regions (list): List of regions.
            keep_multiple (bool, optional): Whether to keep multiple labels for each region.

        Returns:
            dict: Dictionary containing the label map.
        """
        label_map = {}
        for region in regions:
            intensity_image = region.intensity_image[region.intensity_image > 0]
            labels = np.unique(intensity_image)
            if keep_multiple:
                label_map[region.label] = labels
            elif len(labels) == 1:
                label_map[region.label] = labels[0]
        return label_map

    # Erode nuclei to prevent overlapping with cells
    nuclei_eroded = center_pixels(nuclei)

    # Get unique label maps for nuclei and cells
    nucleus_map = get_unique_label_map(regionprops(nuclei_eroded, intensity_image=cells))
    if how == 'contained_in_cells':
        cell_map = get_unique_label_map(regionprops(cells, intensity_image=nuclei_eroded), keep_multiple=True)
    else:
        cell_map = get_unique_label_map(regionprops(cells, intensity_image=nuclei_eroded))

    # Keep only nucleus-cell pairs with matching labels
    keep = []
    for nucleus in nucleus_map:
        try:
            if how == 'contained_in_cells':
                if nucleus in cell_map[nucleus_map[nucleus]]:
                    keep.append([nucleus, nucleus_map[nucleus]])
            else:
                if cell_map[nucleus_map[nucleus]] == nucleus:
                    keep.append([nucleus, nucleus_map[nucleus]])
        except KeyError:
            pass

    # If no matches found, return zero arrays
    if len(keep) == 0:
        return np.zeros_like(nuclei), np.zeros_like(cells)

    # Extract nuclei and cells to keep
    keep_nuclei, keep_cells = zip(*keep)

    # Reassign labels based on the reconciliation method
    if how == 'contained_in_cells':
        nuclei = relabel_array(nuclei, {nuclei_label: cell_label for nuclei_label, cell_label in keep})
        cells[~np.isin(cells, keep_cells)] = 0
        labels, cell_indices = np.unique(cells, return_inverse=True)
        _, nuclei_indices = np.unique(nuclei, return_inverse=True)
        cells = np.arange(0, labels.shape[0])[cell_indices.reshape(*cells.shape)]
        nuclei = np.arange(0, labels.shape[0])[nuclei_indices.reshape(*nuclei.shape)]
    else:
        nuclei = relabel_array(nuclei, {label: i + 1 for i, label in enumerate(keep_nuclei)})
        cells = relabel_array(cells, {label: i + 1 for i, label in enumerate(keep_cells)})

    # Convert arrays to integers
    nuclei, cells = nuclei.astype(int), cells.astype(int)
    return nuclei, cells


def center_pixels(label_image):
    """
    Assign labels to center pixels of regions in a labeled image.

    Parameters:
        label_image (numpy.ndarray): Labeled image.

    Returns:
        numpy.ndarray: Image with labels assigned to center pixels of regions.
    """
    ultimate = np.zeros_like(label_image)  # Initialize an array to store the result
    for r in regionprops(label_image):  # Iterate over regions in the labeled image
        # Calculate the mean coordinates of the bounding box of the region
        i, j = np.array(r.bbox).reshape(2,2).mean(axis=0).astype(int)
        # Assign the label of the region to the center pixel
        ultimate[i, j] = r.label
    return ultimate  # Return the image with labels assigned to center pixels
