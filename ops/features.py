import numpy as np
import ops.utils

# FUNCTIONS


def correlate_channels(r, first, second):
    """Cross-correlation between non-zero pixels. 
    Uses `first` and `second` to index channels from `r.intensity_image_full`.
    """
    A, B = r.intensity_image_full[[first, second]]

    filt = A > 0
    if filt.sum() == 0:
        return np.nan

    A = A[filt]
    B  = B[filt]
    try:
        corr_array = (A - A.mean()) * (B - B.mean()) / (A.std() * B.std())
        corr = corr_array.mean()
    except:
        corr = float('Nan')

    return corr


def masked(r, index):
    return r.intensity_image_full[index][r.filled_image]
def bounds(r, index):
    return r.intensity_image_full[index][r.image]

# FEATURES
# these functions expect an `skimage.measure.regionprops` region as input

intensity = {
    'mean': lambda r: r.intensity_image[r.image].mean(),
    'median': lambda r: np.median(r.intensity_image[r.image]),
    'max': lambda r: r.intensity_image[r.image].max(),
    'min': lambda r: r.intensity_image[r.image].min(),
    }

geometry = {
    'area'    : lambda r: r.area,
    'i'       : lambda r: r.centroid[0],
    'j'       : lambda r: r.centroid[1],
    'bounds'  : lambda r: r.bbox,
    'contour' : lambda r: ops.utils.binary_contours(r.image, fix=True, labeled=False)[0],
    'label'   : lambda r: r.label,
    'mask':     lambda r: ops.utils.Mask(r.image),
    'eccentricity': lambda r: r.eccentricity,
    'solidity': lambda r: r.solidity,
    'convex_area': lambda r: r.convex_area,
    'perimeter': lambda r: r.perimeter
    }

# DAPI, HA, myc
frameshift = {
    'dapi_ha_corr' : lambda r: correlate_channels(r, 0, 1),
    'dapi_myc_corr': lambda r: correlate_channels(r, 0, 2),
    'ha_median'    : lambda r: np.median(r.intensity_image_full[1]),
    'myc_median'   : lambda r: np.median(r.intensity_image_full[2]),
    'cell'         : lambda r: r.label,
    }

translocation = {
    'dapi_gfp_corr' : lambda r: correlate_channels(r, 0, 1),
    # 'dapi_mean'  : lambda r: masked(r, 0).mean(),
    # 'dapi_median': lambda r: np.median(masked(r, 0)),
    # 'gfp_median' : lambda r: np.median(masked(r, 1)),
    # 'gfp_mean'   : lambda r: masked(r, 1).mean(),
    # 'dapi_int'   : lambda r: masked(r, 0).sum(),
    # 'gfp_int'    : lambda r: masked(r, 1).sum(),
    # 'dapi_max'   : lambda r: masked(r, 0).max(),
    # 'gfp_max'    : lambda r: masked(r, 1).max(),
    }

viewRNA = {
    'cy3_median': lambda r: np.median(masked(r, 1)),
    'cy5_median': lambda r: np.median(masked(r, 2)),
    'cy5_80p'   : lambda r: np.percentile(masked(r, 2), 80),
    'cy3_int': lambda r: masked(r, 1).sum(),
    'cy5_int': lambda r: masked(r, 2).sum(),
    'cy5_mean': lambda r: masked(r, 2).sum(),
    'cy5_max': lambda r: masked(r, 2).max(),
}

synapse = {
    'dapi_a532_corr' : lambda r: correlate_channels(r, 0, 3),
    'dapi_a594_corr' : lambda r: correlate_channels(r, 0, 4),
    'dapi_a647_corr' : lambda r: correlate_channels(r, 0, 5),
    'dapi_a750_corr' : lambda r: correlate_channels(r, 0, 2),
    'gfp_a532_corr' : lambda r: correlate_channels(r, 1, 3),
    'gfp_a594_corr' : lambda r: correlate_channels(r, 1, 4),
    'gfp_a647_corr' : lambda r: correlate_channels(r, 1, 5),
    'gfp_a750_corr' : lambda r: correlate_channels(r, 1, 2),
    'a532_a594_corr' : lambda r: correlate_channels(r, 3, 4),
    'a532_a647_corr' : lambda r: correlate_channels(r, 3, 5),
    'a532_a750_corr' : lambda r: correlate_channels(r, 3, 2),
    'a594_a647_corr' : lambda r: correlate_channels(r, 4, 5),
    'a532_a750_corr' : lambda r: correlate_channels(r, 4, 2),
    'a647_a750_corr' : lambda r: correlate_channels(r, 5, 2),
    'dapi_int' : lambda r: masked(r, 0).sum(),
    'dapi_mean' : lambda r: masked(r, 0).mean(),
    'dapi_std' : lambda r: np.std(masked(r, 0)),
    'dapi_median' : lambda r: np.median(masked(r, 0)),
    'dapi_max' : lambda r: masked(r, 0).max(),
    'dapi_min' : lambda r: masked(r, 0).min(),
    'dapi_lower_quartile' : lambda r: np.percentile(masked(r, 0),25),
    'dapi_upper_quartile' : lambda r: np.percentile(masked(r, 0),75),
    'gfp_int' : lambda r: masked(r, 1).sum(),
    'gfp_mean' : lambda r: masked(r, 1).mean(),
    'gfp_std' : lambda r: np.std(masked(r, 1)),
    'gfp_median' : lambda r: np.median(masked(r, 1)),
    'gfp_max' : lambda r: masked(r, 1).max(),
    'gfp_min' : lambda r: masked(r, 1).min(),
    'gfp_lower_quartile' : lambda r: np.percentile(masked(r, 1),25),
    'gfp_upper_quartile' : lambda r: np.percentile(masked(r, 1),75),
    'a750_int' : lambda r: masked(r, 2).sum(),
    'a750_mean' : lambda r: masked(r, 2).mean(),
    'a750_std' : lambda r: np.std(masked(r, 2)),
    'a750_median' : lambda r: np.median(masked(r, 2)),
    'a750_max' : lambda r: masked(r, 2).max(),
    'a750_min' : lambda r: masked(r, 2).min(),
    'a750_lower_quartile' : lambda r: np.percentile(masked(r, 2),25),
    'a750_upper_quartile' : lambda r: np.percentile(masked(r, 2),75),
    'a532_int' : lambda r: masked(r, 3).sum(),
    'a532_mean' : lambda r: masked(r, 3).mean(),
    'a532_std' : lambda r: np.std(masked(r, 3)),
    'a532_median' : lambda r: np.median(masked(r, 3)),
    'a532_max' : lambda r: masked(r, 3).max(),
    'a532_min' : lambda r: masked(r, 3).min(),
    'a532_lower_quartile' : lambda r: np.percentile(masked(r, 3),25),
    'a532_upper_quartile' : lambda r: np.percentile(masked(r, 3),75),
    'a594_int' : lambda r: masked(r, 4).sum(),
    'a594_mean' : lambda r: masked(r, 4).mean(),
    'a594_std' : lambda r: np.std(masked(r, 4)),
    'a594_median' : lambda r: np.median(masked(r, 4)),
    'a594_max' : lambda r: masked(r, 4).max(),
    'a594_min' : lambda r: masked(r, 4).min(),
    'a594_lower_quartile' : lambda r: np.percentile(masked(r, 4),25),
    'a594_upper_quartile' : lambda r: np.percentile(masked(r, 4),75),
    'a647_int' : lambda r: masked(r, 5).sum(),
    'a647_mean' : lambda r: masked(r, 5).mean(),
    'a647_std' : lambda r: np.std(masked(r, 5)),
    'a647_median' : lambda r: np.median(masked(r, 5)),
    'a647_max' : lambda r: masked(r, 5).max(),
    'a647_min' : lambda r: masked(r, 5).min(),
    'a647_lower_quartile' : lambda r: np.percentile(masked(r, 5),25),
    'a647_upper_quartile' : lambda r: np.percentile(masked(r, 5),75)
}


all_features = [
    intensity, 
    geometry,
    translocation,
    frameshift,
    viewRNA,
    synapse
    ]

def validate_features():
    names = sum(map(list, all_features), [])
    assert len(names) == len(set(names))

def make_feature_dict(feature_names):
    features = {}
    [features.update(d) for d in all_features]
    return {n: features[n] for n in feature_names}

validate_features()

features_basic = make_feature_dict(('area', 'i', 'j', 'label'))

features_geom = make_feature_dict((
    'area', 'eccentricity', 'convex_area', 'perimeter'))

features_translocation_nuclear = make_feature_dict((
	'dapi_gfp_corr', 
	'eccentricity', 'solidity',
	'dapi_median', 'dapi_mean', 'dapi_int', 'dapi_max',
	'gfp_median',  'gfp_mean',  'gfp_int',  'gfp_max',
    'area'))

features_translocation_cell = make_feature_dict((	
	'dapi_gfp_corr', 
	'eccentricity', 'solidity',
	'dapi_median', 'dapi_mean', 'dapi_int', 'dapi_max',
	'gfp_median',  'gfp_mean',  'gfp_int',  'gfp_max',
    'area'))

features_frameshift = make_feature_dict((
    'dapi_ha_corr', 
    'dapi_median', 'dapi_max', 
    'ha_median'))

features_frameshift_myc = make_feature_dict((
    'dapi_ha_corr', 'dapi_myc_corr', 
    'dapi_median', 'dapi_max', 
    'ha_median', 'myc_median'))

features_translocation_nuclear_simple = make_feature_dict((
	'dapi_gfp_corr', 
	'dapi_mean', 'dapi_max', 'gfp_mean', 'gfp_max',
    'area'))

features_synapse_cell = make_feature_dict((
    'area', 'eccentricity', 'solidity') + tuple(synapse.keys()))

features_synapse_edge = make_feature_dict((
    'area', 'eccentricity', 'solidity') + tuple(synapse.keys()))

features_synapse_puncta = make_feature_dict((
    'area', 'eccentricity', 'solidity') + tuple(synapse.keys()))


