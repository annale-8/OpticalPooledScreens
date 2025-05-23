import warnings
from collections import defaultdict
from itertools import product

import skimage
import skimage.feature
import skimage.filters
import numpy as np
import pandas as pd
import scipy.stats

from scipy import ndimage

import ops.io
import ops.utils


# FEATURES
def feature_table(data, labels, features, global_features=None):
    """Apply functions in feature dictionary to regions in data 
    specified by integer labels. If provided, the global feature
    dictionary is applied to the full input data and labels. 

    Results are combined in a dataframe with one row per label and
    one column per feature.
    """
    regions = ops.utils.regionprops(labels, intensity_image=data)
    results = defaultdict(list)
    for region in regions:
        for feature, func in features.items():
            results[feature].append(func(region))
    if global_features:
        for feature, func in global_features.items():
            results[feature] = func(data, labels)
    return pd.DataFrame(results)


def build_feature_table(stack, labels, features, index):
    """Iterate over leading dimensions of stack, applying `feature_table`. 
    Results are labeled by index and concatenated.

        >>> stack.shape 
        (3, 4, 511, 626)
        
        index = (('round', range(1,4)), 
                 ('channel', ('DAPI', 'Cy3', 'A594', 'Cy5')))
    
        build_feature_table(stack, labels, features, index) 

    """
    index_vals = list(product(*[vals for _, vals in index]))
    index_names = [x[0] for x in index]
    
    s = stack.shape
    results = []
    for frame, vals in zip(stack.reshape(-1, s[-2], s[-1]), index_vals):
        df = feature_table(frame, labels, features)
        for name, val in zip(index_names, vals):
            df[name] = val
        results += [df]
    
    return pd.concat(results)


def find_cells(nuclei, mask, remove_boundary_cells=True):
    """Convert binary mask to cell labels, based on nuclei labels.

    Expands labeled nuclei to cells, constrained to where mask is >0.
    """
    distance = ndimage.distance_transform_cdt(nuclei == 0)
    cells = skimage.morphology.watershed(distance, nuclei, mask=mask)
    # remove cells touching the boundary
    if remove_boundary_cells:
        cut = np.concatenate([cells[0,:], cells[-1,:], 
                              cells[:,0], cells[:,-1]])
        cells.flat[np.in1d(cells, np.unique(cut))] = 0

    return cells.astype(np.uint16)


def find_puncta(channel, threshold, radius=3, area_min=0.5, area_max=100,
                score=lambda r: r.mean_intensity,
                smooth=1.35):
    """
    Finds synaptic puncta in a given channel.
    As of 10/3/20, same function as find_nuclei
    """

    mask = binarize(channel, radius, area_min)
    labeled = skimage.measure.label(mask)
    labeled = filter_by_region(labeled, score, threshold, intensity_image=channel) > 0

    # only fill holes below minimum area
    filled = ndimage.binary_fill_holes(labeled)
    difference = skimage.measure.label(filled!=labeled)

    change = filter_by_region(difference, lambda r: r.area < area_min, 0) > 0
    labeled[change] = filled[change]

    puncta = apply_watershed(labeled, smooth=smooth)

    result = filter_by_region(puncta, lambda r: area_min < r.area < area_max, threshold)

    return result


def assign_cells_puncta(cells, puncta):
    """Assigns cell label to each puncta (label) based on most frequent cell assignment."""

    # consider only puncta where cell > 0
    # cell_mask = np.clip(cells, 0, 1)
    # overlap = np.multiply(cell_mask,puncta)

    # cell_label = np.zeros(((np.max(puncta)+1),), dtype=int)
    # for i in np.unique(overlap):
    #     cell_label[i] = scipy.stats.mode(cells[np.where(puncta == i)])[0][0]

    if cells.shape != puncta.shape:
        print('Cell and puncta masks have different shapes')

    p = puncta.flatten()
    c = cells.flatten()

    df = pd.DataFrame({'puncta':p, 'cell':c})
    # assign puncta to most frequent cell label (if multiple modes, assign to lower value)
    df_label = df.groupby('puncta').agg(lambda x:x.value_counts().index[0])
    label_dict = df_label.to_dict()['cell']
    
    # list puncta that do not map to any cells
    df_label = df_label.reset_index()
    zero_list = df_label['puncta'][df_label.cell==0].tolist()
    
    return label_dict, zero_list

def find_peaks(data, n=5):
    """Finds local maxima. At a maximum, the value is max - min in a 
    neighborhood of width `n`. Elsewhere it is zero.
    """
    from scipy.ndimage import filters
    neighborhood_size = (1,)*(data.ndim-2) + (n,n)
    data_max = filters.maximum_filter(data, neighborhood_size)
    data_min = filters.minimum_filter(data, neighborhood_size)
    peaks = data_max - data_min
    peaks[data != data_max] = 0
    
    # remove peaks close to edge
    mask = np.ones(peaks.shape, dtype=bool)
    mask[..., n:-n, n:-n] = False
    peaks[mask] = 0
    
    return peaks

def calculate_illumination_correction(files, smooth=None, rescale=True, threading=False, slicer=slice(None)):
    """calculate illumination correction field for use with apply_illumination_correction 
    Snake method. Equivalent to CellProfiler's CorrectIlluminationCalculate module with 
    option "Regular", "All", "Median Filter"
    Note: algorithm originally benchmarked using ~250 images per plate to calculate plate-wise
    illumination correction functions (Singh et al. J Microscopy, 256(3):231-236, 2014)
    """
    from ops.io import read_stack as read

    N = len(files)

    global data

    data = read(files[0])[slicer]/N

    def accumulate_image(file):
        global data
        data += read(file)[slicer]/N

    if threading:
        from joblib import Parallel, delayed
        Parallel(n_jobs=-1,require='sharedmem')(delayed(accumulate_image)(file) for file in files[1:])
    else:
        for file in files[1:]:
            accumulate_image(file)

    data = np.squeeze(data.astype(np.uint16))

    if not smooth:
        # default is 1/20th area of image
        # smooth = (np.array(data.shape[-2:])/8).mean().astype(int)
        smooth = int(np.sqrt((data.shape[-1]*data.shape[-2])/(np.pi*20)))

    selem = skimage.morphology.disk(smooth)

    median_filter = ops.utils.applyIJ(skimage.filters.median)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smoothed = median_filter(data,selem,behavior='rank')

    if rescale:
        @ops.utils.applyIJ
        def rescale_channels(data):
            # use 2nd percentile for robust minimum
            robust_min = np.quantile(data.reshape(-1),q=0.02)
            robust_min = 1 if robust_min == 0 else robust_min
            data = data/robust_min
            data[data<1] = 1 
            return data

        smoothed = rescale_channels(smoothed)

    return smoothed

@ops.utils.applyIJ
def log_ndi(data, sigma=1, *args, **kwargs):
    """Apply laplacian of gaussian to each image in a stack of shape
    (..., I, J). 
    Extra arguments are passed to scipy.ndimage.filters.gaussian_laplace.
    Inverts output and converts back to uint16.
    """
    f = scipy.ndimage.filters.gaussian_laplace
    arr_ = -1 * f(data.astype(float), sigma, *args, **kwargs)
    arr_ = np.clip(arr_, 0, 65535) / 65535
    
    # skimage precision warning 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return skimage.img_as_uint(arr_)

class Align:
    """Alignment redux, used by snakemake.
    """
    @staticmethod
    def normalize_by_percentile(data_, q_norm=70):
        shape = data_.shape
        shape = shape[:-2] + (-1,)
        p = np.percentile(data_, q_norm, axis=(-2, -1))[..., None, None]
        normed = data_ / p
        return normed

    @staticmethod
    @ops.utils.applyIJ
    def filter_percentiles(data, q1, q2):
        """Replaces data outside of percentile range [q1, q2]
        with uniform noise over the range [q1, q2]. Useful for 
        eliminating alignment artifacts due to bright features or 
        regions of zeros.
        """
        x1, x2 = np.percentile(data, [q1, q2])
        mask = (x1 > data) | (x2 < data)
        return Align.fill_noise(data, mask, x1, x2)

    @staticmethod
    @ops.utils.applyIJ
    def filter_values(data, x1, x2):
        """Replaces data outside of value range [x1, x2]
        with uniform noise over the range [x1, x2]. Useful for 
        eliminating alignment artifacts due to bright features or 
        regions of zeros.
        """
        mask = (x1 > data) | (x2 < data)
        return Align.fill_noise(data, mask, x1, x2)

    @staticmethod
    def fill_noise(data, mask, x1, x2):
        filtered = data.copy()
        rs = np.random.RandomState(0)
        filtered[mask] = rs.uniform(x1, x2, mask.sum()).astype(data.dtype)
        return filtered        

    @staticmethod
    def calculate_offsets(data_, upsample_factor):
        target = data_[0]
        offsets = []
        for i, src in enumerate(data_):
            if i == 0:
                offsets += [(0, 0)]
            else:
                offset, _, _ = skimage.feature.register_translation(
                                src, target, upsample_factor=upsample_factor)
                # offset, _, _ = skimage.registration.phase_cross_correlation(
                #                 src, target, upsample_factor=upsample_factor)
                offsets += [offset]
        return np.array(offsets)

    @staticmethod
    def apply_offsets(data_, offsets):
        warped = []
        for frame, offset in zip(data_, offsets):
            if offset[0] == 0 and offset[1] == 0:
                warped += [frame]
            else:
                # skimage has a weird (i,j) <=> (x,y) convention
                st = skimage.transform.SimilarityTransform(translation=offset[::-1])
                frame_ = skimage.transform.warp(frame, st, preserve_range=True)
                warped += [frame_.astype(data_.dtype)]

        return np.array(warped)

    @staticmethod
    def align_within_cycle(data_, upsample_factor=4, window=1, q1=0, q2=90):
        filtered = Align.filter_percentiles(Align.apply_window(data_, window), 
            q1=q1, q2=q2)

        offsets = Align.calculate_offsets(filtered, upsample_factor=upsample_factor)

        return Align.apply_offsets(data_, offsets)

    @staticmethod
    def align_between_cycles(data, channel_index, upsample_factor=4, window=1,
    		return_offsets=False):
        # offsets from target channel
        target = Align.apply_window(data[:, channel_index], window)
        offsets = Align.calculate_offsets(target, upsample_factor=upsample_factor)

        # apply to all channels
        warped = []
        for data_ in data.transpose([1, 0, 2, 3]):
            warped += [Align.apply_offsets(data_, offsets)]

        aligned = np.array(warped).transpose([1, 0, 2, 3])
        if return_offsets:
        	return aligned, offsets
        else:
        	return aligned

    @staticmethod
    def align_between_pheno_channels(data, channel_index=4, offset_channel=[3,5], upsample_factor=1, window=2,
            return_offsets=False):
        """ Expects `data` to be a list with dimensions (CHANNEL, I, J) """
        # offsets from target channel
        target = Align.apply_window(data[:, channel_index], window)
        offsets = Align.calculate_offsets(target, upsample_factor=upsample_factor)

        # apply to all channels
        warped = []
        for index, data_ in enumerate(data.transpose([1, 0, 2, 3])):
            if index in offset_channel:
                warped += [Align.apply_offsets(data_, offsets)]
            else:
                warped += [data_]

        aligned = np.array(warped).transpose([1, 0, 2, 3])
        if return_offsets:
            return aligned, offsets
        else:
            return aligned

    @staticmethod
    def apply_window(data, window):
        height, width = data.shape[-2:]
        find_border = lambda x: int((x/2.) * (1 - 1/float(window)))
        i, j = find_border(height), find_border(width)
        return data[..., i:height - i, j:width - j]


# SEGMENT
def find_nuclei(dapi, threshold, radius=15, area_min=50, area_max=500, method='mean',
                score=lambda r: r.mean_intensity,
                smooth=1.35):
    """
    """
    mask = binarize(dapi, radius, area_min, method)
    labeled = skimage.measure.label(mask)
    labeled = filter_by_region(labeled, score, threshold, intensity_image=dapi) > 0

    # only fill holes below minimum area
    filled = ndimage.binary_fill_holes(labeled)
    difference = skimage.measure.label(filled!=labeled)

    change = filter_by_region(difference, lambda r: r.area < area_min, 0) > 0
    labeled[change] = filled[change]

    nuclei = apply_watershed(labeled, smooth=smooth)

    result = filter_by_region(nuclei, lambda r: area_min < r.area < area_max, threshold)

    return result
        

def binarize(image, radius, min_size, method='mean',percentile=0.5,equalize=False):
    """Apply local mean threshold to find outlines. Filter out
    background shapes. Otsu threshold on list of region mean intensities will remove a few
    dark cells. Could use shape to improve the filtering.
    """
    # slower than optimized disk in ImageJ
    # scipy.ndimage.uniform_filter with square is fast but crappy
    selem = skimage.morphology.disk(radius)
    if equalize:
        image = skimage.filters.rank.equalize(image,selem=selem)
    
    dapi = skimage.img_as_ubyte(image)

    if method == 'mean':
        filtered = skimage.filters.rank.mean(dapi, selem=selem)
    elif method == 'median':
        filtered = skimage.filters.rank.median(dapi, selem=selem)
    elif method=='otsu':
        filtered = skimage.filters.rank.otsu(dapi, selem=selem)
    elif method=='percentile':
        filtered = skimage.filters.rank.percentile(dapi,selem=selem,p0=percentile)
    
    mask = dapi > filtered
    mask = skimage.morphology.remove_small_objects(mask, min_size=min_size)

    return mask


def filter_by_region(labeled, score, threshold, intensity_image=None, relabel=True):
    """Apply a filter to label image. The `score` function takes a single region 
    as input and returns a score. 
    If scores are boolean, regions where the score is false are removed.
    Otherwise, the function `threshold` is applied to the list of scores to 
    determine the minimum score at which a region is kept.
    If `relabel` is true, the regions are relabeled starting from 1.
    """
    labeled = labeled.copy().astype(int)
    regions = skimage.measure.regionprops(labeled, intensity_image=intensity_image)
    scores = np.array([score(r) for r in regions])

    if all([s in (True, False) for s in scores]):
        cut = [r.label for r, s in zip(regions, scores) if not s]
    else:
        t = threshold(scores)
        cut = [r.label for r, s in zip(regions, scores) if s < t]

    labeled.flat[np.in1d(labeled.flat[:], cut)] = 0
    
    if relabel:
        labeled, _, _ = skimage.segmentation.relabel_sequential(labeled)

    return labeled


def apply_watershed(img, smooth=4):
    distance = ndimage.distance_transform_edt(img)
    if smooth > 0:
        distance = skimage.filters.gaussian(distance, sigma=smooth)
    local_max = skimage.feature.peak_local_max(
                    distance, indices=False, footprint=np.ones((3, 3)), 
                    exclude_border=False)

    markers = ndimage.label(local_max)[0]
    result = skimage.morphology.watershed(-distance, markers, mask=img)
    return result.astype(np.uint16)



def alpha_blend(arr, positions, clip=True, edge=0.95, edge_width=0.02, subpixel=False):
    """Blend array of images, translating image coordinates according to offset matrix.
    arr : N x I x J
    positions : N x 2 (n, i, j)
    """
    
    # @ops.utils.memoize
    def make_alpha(s, edge=0.95, edge_width=0.02):
        """Unity in center, drops off near edge
        :param s: shape
        :param edge: mid-point of drop-off
        :param edge_width: width of drop-off in exponential
        :return:
        """
        sigmoid = lambda r: 1. / (1. + np.exp(-r))

        x, y = np.meshgrid(range(s[0]), range(s[1]))
        xy = np.concatenate([x[None, ...] - s[0] / 2,
                             y[None, ...] - s[1] / 2])
        R = np.max(np.abs(xy), axis=0)

        return sigmoid(-(R - s[0] * edge/2) / (s[0] * edge_width))

    # determine output shape, offset positions as necessary
    if subpixel:
        positions = np.array(positions)
    else:
        positions = np.round(positions)
    # convert from ij to xy
    positions = positions[:, [1, 0]]    

    positions -= positions.min(axis=0)
    shapes = [a.shape for a in arr]
    output_shape = np.ceil((shapes + positions[:,::-1]).max(axis=0)).astype(int)

    # sum data and alpha layer separately, divide data by alpha
    output = np.zeros([2] + list(output_shape), dtype=float)
    for image, xy in zip(arr, positions):
        alpha = 100 * make_alpha(image.shape, edge=edge, edge_width=edge_width)
        if subpixel is False:
            j, i = np.round(xy).astype(int)

            output[0, i:i+image.shape[0], j:j+image.shape[1]] += image * alpha.T
            output[1, i:i+image.shape[0], j:j+image.shape[1]] += alpha.T
        else:
            ST = skimage.transform.SimilarityTransform(translation=xy)

            tmp = np.array([skimage.transform.warp(image, inverse_map=ST.inverse,
                                                   output_shape=output_shape,
                                                   preserve_range=True, mode='reflect'),
                            skimage.transform.warp(alpha, inverse_map=ST.inverse,
                                                   output_shape=output_shape,
                                                   preserve_range=True, mode='constant')])
            tmp[0, :, :] *= tmp[1, :, :]
            output += tmp


    output = (output[0, :, :] / output[1, :, :])

    if clip:
        def edges(n):
            return np.r_[n[:4, :].flatten(), n[-4:, :].flatten(),
                         n[:, :4].flatten(), n[:, -4:].flatten()]

        while np.isnan(edges(output)).any():
            output = output[4:-4, 4:-4]

    return output.astype(arr[0].dtype)