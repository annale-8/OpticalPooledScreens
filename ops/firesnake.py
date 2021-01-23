import inspect
import functools
import os
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='regionprops and image moments')
warnings.filterwarnings('ignore', message='non-tuple sequence for multi')
warnings.filterwarnings('ignore', message='precision loss when converting')

import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use('TkAgg') #uncomment when running snakemake
from nd2reader import ND2Reader
import skimage
import ops.features
import ops.process
import ops.io
import ops.in_situ
from ops.process import Align
import ops.filenames
import ops.nd2_to_tif
import glob


class Snake():
    """Container class for methods that act directly on data (names start with
    underscore) and methods that act on arguments from snakemake (e.g., filenames
    provided instead of image and table data). The snakemake methods (no underscore)
    are automatically loaded by `Snake.load_methods`.
    """

    @staticmethod
    def _nd2_to_tif(input_filename,meta=True):
        """
        Convert nd2 to tif files
        """
        # add parse_filename function to get info from nd2 name and convert to tif filename
        info = ops.filenames.parse_filename(input_filename)
        
        file_description={}
        for k,v in sorted(info.items()):
            file_description[k] = v
        file_description['ext']='tif'
        file_description['subdir']=file_description['expt']+'_tif/'+file_description['mag']+'_'+file_description['cycle']


        with ND2Reader(input_filename) as images:
            images.iter_axes='v'
            axes = 'xy'
            if 'c' in images.axes:
                axes = 'c' + axes
            if 'z' in images.axes:
                axes = 'z' + axes
            images.bundle_axes = axes

            if 'z' in images.axes:
                for site,image in zip(images.metadata['fields_of_view'],images):
                    image = image.max(axis=0)

                    output_filename = ops.filenames.name_file(file_description,site=str(site)) 
                    save(output_filename,image[:])
            else:
                for site,image in zip(images.metadata['fields_of_view'],images):
                    output_filename = ops.filenames.name_file(file_description,site=str(site)) 
                    save(output_filename,image[:])
                                 
            # METADATA EXTRACTION
            if meta==True:
                well_metadata = [{
                                    'filename':ops.filenames.name_file(file_description,site=str(site)),
                                    'field_of_view':site,
                                    'x':images.metadata['x_data'][site],
                                    'y':images.metadata['y_data'][site],
                                    'z':images.metadata['z_data'][site],
                                    'pfs_offset':images.metadata['pfs_offset'][0],
                                    'pixel_size':images.metadata['pixel_microns']
                                } for site in images.metadata['fields_of_view']]
                metadata_filename = ops.filenames.name_file(file_description,tag='metadata',ext='pkl')
                pd.DataFrame(well_metadata).to_pickle(metadata_filename)

    @staticmethod
    def _max_project_zstack(stack,slices=3):
        """
        Condense z-stack into a single slice using a simple maximum project through 
        all slices for each channel individually. If slices is a list, then specifies the number 
        of slices for each channel.
        """
        if isinstance(slices,list):
            channels = len(slices)
            maxed = []
            end_ch_slice = 0
            for ch in range(len(slices)):
                end_ch_slice += slices[ch]
                ch_slices = stack[(end_ch_slice-slices[ch]):(end_ch_slice)]
                ch_maxed = np.amax(ch_slices,axis=0)
                maxed.append(ch_maxed)
        else:
            channels = int(stack.shape[-3]/slices)
            assert len(stack) == int(slices)*channels, 'Input data must have leading dimension length slices*channels'
            maxed = []
            for ch in range(channels):
                ch_slices = stack[(ch*slices):((ch+1)*slices)]
                ch_maxed = np.amax(ch_slices,axis=0)
                maxed.append(ch_maxed)
        maxed = np.array(maxed)

        return maxed 

    @staticmethod
    def _merge_csv(tag,filetype='csv',subdir='process'):
        """Reads .csv files, concatenates them into a pandas df, and saves as merged .h5 file
        """
        files = glob.glob('{subdir}/*.{tag}.csv'.format(subdir=subdir,tag=tag))
    
        arr = []
        for f in files:
            try:
                arr += [pd.read_csv(f)]
            except pd.errors.EmptyDataError:
                pass
        df = pd.concat(arr)
        
        if filetype=='csv':
            df.to_csv(tag+'.csv')
        else:
            df.to_hdf(tag+'.h5', tag, mode='w')

        return df


    @staticmethod
    def _apply_illumination_correction(data, correction, n_jobs=1, backend='threading'):
        if n_jobs == 1:
            return (data/correction).astype(np.uint16)
        else:
            return ops.utils.applyIJ_parallel(Snake._apply_illumination_correction,
                arr=data,
                correction=correction,
                backend=backend,
                n_jobs=n_jobs
                )

    @staticmethod
    def _align_SBS(data, method='DAPI', upsample_factor=2, window=2, cutoff=1,
        align_within_cycle=True, keep_trailing=False, n=1):
        """Rigid alignment of sequencing cycles and channels. 

        Expects `data` to be an array with dimensions (CYCLE, CHANNEL, I, J).
        A centered subset of data is used if `window` is greater 
        than one. Subpixel alignment is done if `upsample_factor` is greater than
        one (can be slow).
        """
        data = np.array(data)
        if keep_trailing:
            valid_channels = min([len(x) for x in data])
            data = np.array([x[-valid_channels:] for x in data])

        assert data.ndim == 4, 'Input data must have dimensions CYCLE, CHANNEL, I, J'

        # align between SBS channels for each cycle
        aligned = data.copy()
        if align_within_cycle:
            align_it = lambda x: Align.align_within_cycle(x, window=window, upsample_factor=upsample_factor)
            if data.shape[1] == 4:
                n = 0
                align_it = lambda x: Align.align_within_cycle(x, window=window, 
                    upsample_factor=upsample_factor)
            # else:
            #     n = 1
            
            aligned[:, n:] = np.array([align_it(x) for x in aligned[:, n:]])
            

        if method == 'DAPI':
            # align cycles using the DAPI channel
            aligned = Align.align_between_cycles(aligned, channel_index=0, 
                                window=window, upsample_factor=upsample_factor)
        elif method == 'SBS_mean':
            # calculate cycle offsets using the average of SBS channels
            target = Align.apply_window(aligned[:, 1:], window=window).max(axis=1)
            normed = Align.normalize_by_percentile(target)
            normed[normed > cutoff] = cutoff
            offsets = Align.calculate_offsets(normed, upsample_factor=upsample_factor)
            # apply cycle offsets to each channel
            for channel in range(aligned.shape[1]):
                aligned[:, channel] = Align.apply_offsets(aligned[:, channel], offsets)

        return aligned

    @staticmethod
    def _align_by_DAPI(data_1, data_2, channel_index=0, upsample_factor=2):
        """Align the second image to the first, using the channel at position 
        `channel_index`. The first channel is usually DAPI.
        """
        images = data_1[channel_index], data_2[channel_index]
        _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        offsets = [offset] * len(data_2)
        aligned = ops.process.Align.apply_offsets(data_2, offsets)
        return aligned
    
    @staticmethod
    def _align_phenotype_channels(files,target,source,riders=[],upsample_factor=2, window=2, remove=False):
        """
        For fast-mode imaging: merge separate channel tifs into one stack
        Expects files to be a list of strings of filenames
        Merged data will be in the order of (CYCLE, CHANNEL, I, J)
        
        Target = int 
            Channel index to which source channels are aligned to
        Source = int or list of integers
            Channel with similar pattern to target used to calculate offsets for alignment
            If list, calculate offsets for each channel to target separately 
            10/1/20 NOTE: current code does not accomodate for riders if multiple sources listed
        Riders - list of integers
            Channels to be aligned to target using offset calculated from source
        """
        
        data = np.array(files)

        # in the case that data has shape (CYCLE, CHANNEL, I, J):
        if data.ndim == 4:
            data = data[0]

        if not isinstance(source,list):
            windowed = Align.apply_window(data[[target,source]],window)
            # remove noise?
            offsets = Align.calculate_offsets(windowed,upsample_factor=upsample_factor)
            
            if not isinstance(riders,list):
                riders = [riders]
            
            full_offsets = np.zeros((data.shape[0],2))
            full_offsets[[source]+riders] = offsets[1]
            aligned = Align.apply_offsets(data, full_offsets)
        else:
            full_offsets = np.zeros((data.shape[0],2))
            for src in source:
                windowed = Align.apply_window(data[[target,src]],window)
                offsets = Align.calculate_offsets(windowed,upsample_factor=upsample_factor)
                full_offsets[[src]] = offsets[1]
            aligned = Align.apply_offsets(data, full_offsets)

        if remove == 'target':
            channel_order = list(range(data.shape[0]))
            channel_order.remove(source)
            channel_order.insert(target+1,source)
            aligned = aligned[channel_order]
            aligned = remove_channels(aligned, target)
        elif remove == 'source':
            aligned = remove_channels(aligned, source)

        return aligned

    @staticmethod
    def _merge_pheno(files,aligned_puncta,mode='binned'):
        """
        Merge semi-fast-mode phenotyping images
        Expects files to be (CHANNEL, I, J) for DAPI, GFP, A750 -- 1480x1480
        Expects aligned_puncta to be (CHANNEL, I, J) for A532, A594, A647 -- 2960x2960
        Bin 2x2 if mode='cells' and upsize if mode='puncta'
        """
        
        data = []
        if mode == 'upsized':
            for f in files:
                data.append(scipy.ndimage.zoom(np.array(f), 2, order=0))
            upsized = np.array(data)
            aligned = np.array(aligned_puncta)
            assert upsized.ndim == aligned.ndim
            merged = np.concatenate([upsized,aligned],axis=0)

        elif mode == 'binned':
            binned = np.array(files)
            aligned = np.array(aligned_puncta)
            aligned_binned = skimage.measure.block_reduce(aligned, block_size=(1,2,2), func=np.max)
            assert binned.ndim == aligned.ndim
            merged = np.concatenate([binned,aligned_binned],axis=0)

        return merged


    @staticmethod
    def _bin_image(data):

        data = np.array(data)

        if data.ndim == 4:
            block_size = (1,1,2,2)
        elif data.ndim == 3:
            block_size = (1,2,2)
        if data.ndim == 2:
            block_size = (2,2)   

        binned = skimage.measure.block_reduce(data, block_size=block_size, func=np.max)

        return binned

    @staticmethod
    def _merge_SBS(data, upsample_factor=2, window=2, cutoff=1,
        align_within_cycle=True, keep_trailing=False, n=1, c1=False, binning=False):
        """
        Fast-mode SBS: merge and align based on modified align_SBS (method=SBS_mean)
        Slow-mode SBS:
            Remove DAPI channel
            Based on remove_channels: 
                Remove channel or list of channels from array of shape (..., CHANNELS, I, J).
        """
        
        data = np.array(data)
        if keep_trailing:
            valid_channels = min([len(x) for x in data])
            data = np.array([x[-valid_channels:] for x in data])
        

        # for fast-mode SBS, all channels merged into one array with dimensions CHANNEL, I, J 
        if not c1: 
            aligned = np.expand_dims(data, axis=0)
            assert aligned.ndim == 4, 'Input data must have dimensions CYCLE, CHANNEL, I, J'

            # align between SBS channels for each cycle
            if align_within_cycle:
                align_it = lambda x: Align.align_within_cycle(x, window=window, upsample_factor=upsample_factor)
                if aligned.shape[1] == 4:
                    n = 0
                    align_it = lambda x: Align.align_within_cycle(x, window=window, 
                        upsample_factor=upsample_factor)
                # else:
                #     n = 1
                aligned[:, n:] = np.array([align_it(x) for x in aligned[:, n:]])
            
            # calculate cycle offsets using the average of SBS channels
            target = Align.apply_window(aligned[:, :], window=window).max(axis=1)
            normed = Align.normalize_by_percentile(target)
            normed[normed > cutoff] = cutoff
            offsets = Align.calculate_offsets(normed, upsample_factor=upsample_factor)
            # apply cycle offsets to each channel
            for channel in range(aligned.shape[1]):
                aligned[:, channel] = Align.apply_offsets(aligned[:, channel], offsets)

        # for slow-mode SBS, remove CYCLE dimension and DAPI channel -> array with dimensions CHANNEL, I, J

        if c1:
            assert data.ndim == 4, 'Input data must have dimensions CYCLE, CHANNEL, I, J'
            remove_index = 0 # DAPI channel
            aligned = remove_channels(data, remove_index)

        if binning:
            sbs = skimage.measure.block_reduce(aligned, block_size=(1,1,2,2), func=np.max)
        else: 
            sbs = aligned
        
        return sbs[0]


    @staticmethod
    def _segment_nuclei(data, threshold, area_min, area_max):
        """Find nuclei from DAPI. Find cell foreground from aligned but unfiltered 
        data. Expects data to have shape (CHANNEL, I, J).
        """
        data = np.array(data)
        
        if data.ndim == 4:
            data = data[0]

        if isinstance(data, list):
            dapi = data[0].astype(np.uint16) #[0] indicated DAPI channel #
        elif data.ndim == 3:
            dapi = data[0].astype(np.uint16)
        else:
            dapi = data.astype(np.uint16)

        kwargs = dict(threshold=lambda x: threshold, 
            area_min=area_min, area_max=area_max)

        # skimage precision warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nuclei = ops.process.find_nuclei(dapi, **kwargs)
        return nuclei.astype(np.uint16)

    @staticmethod
    def _segment_nuclei_channel(data, threshold, channel, radius, area_min, area_max):
        """Find nuclei/puncta from DAPI/specified channel. 
        Find cell foreground from aligned but unfiltered data. 
        Expects data to have shape (CHANNEL, I, J)
        """
        data = np.array(data)
        if data.ndim == 4:
            data = data[0]

        if isinstance(data, list):
            dapi = data[int(channel)].astype(np.uint16) 
        elif data.ndim == 3:
            dapi = data[int(channel)].astype(np.uint16) 
        else:
            dapi = data.astype(np.uint16)

        kwargs = dict(threshold=lambda x: threshold, 
            radius=radius, area_min=area_min, area_max=area_max)

        # skimage precision warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nuclei = ops.process.find_nuclei(dapi, **kwargs)

        return nuclei.astype(np.uint16)

    @staticmethod
    def _segment_nuclei_stack(dapi, threshold, area_min, area_max):
        """Find nuclei from a nuclear stain (e.g., DAPI). Expects data to have shape (I, J) 
        (segments one image) or (N, I, J) (segments a series of DAPI images).
        """
        kwargs = dict(threshold=lambda x: threshold, 
            area_min=area_min, area_max=area_max)

        find_nuclei = ops.utils.applyIJ(ops.process.find_nuclei)
        # skimage precision warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nuclei = find_nuclei(dapi, **kwargs)
        return nuclei.astype(np.uint16)

    @staticmethod
    def _segment_cells(data, nuclei, threshold, DAPI=True):
        """Segment cells from aligned data. Matches cell labels to nuclei labels.
        Note that labels can be skipped, for example if cells are touching the 
        image boundary.
        """
        if DAPI:   
            sbs = 1
        else:
            sbs = 0

        if data.ndim == 4:
            # no DAPI, min over cycles, mean over channels
            mask = data[:, sbs:].min(axis=0).mean(axis=0)
        elif data.ndim == 3:
            mask = np.median(data[sbs:], axis=0)
        elif data.ndim == 2:
            mask = data
        else:
            raise ValueError

        mask = mask > threshold

        try:
            # skimage precision warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cells = ops.process.find_cells(nuclei, mask)
        except ValueError:
            print('segment_cells error -- no cells')
            cells = nuclei

        return cells


    @staticmethod
    def _segment_HEKs(data, nuclei, channel, threshold, area_min, filter_nuclei=False):
        """Combines segment_cells_GFP and segment_cells_HEK
        Segment cells from phenotyping data. Matches cell labels to nuclei labels.
        Note that labels can be skipped, for example if cells are touching the 
        image boundary.
        Filters all segmented cells by area (~min nuclear area) and GFP signal to determine mask of
        HEK cells AND nuclei of non-HEK cells. 
        Expects data to have shape (CYCLE, CHANNEL, I, J)
        """
        
        if data.ndim == 4:
            mask = data[0][int(channel)] # channel index for cell background (e.g. GFP)
        elif data.ndim == 3:
            mask = data[int(channel)]
        elif data.ndim == 2:
            mask = data
        else:
            raise ValueError

        # cell_threshold and HEK_nuclei_area_min
        kwargs = dict(threshold=lambda x: threshold, area_min=area_min)

        mask = mask > threshold
        filled = scipy.ndimage.binary_fill_holes(mask)

        
        # remove nuclei from nuclei mask that do not pass GFP threshold
        if filter_nuclei:
            nuclei = np.multiply(nuclei,filled)

        try:
            # skimage precision warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # cells = ops.process.find_cells_GFP(filtered_nuclei, mask, **kwargs)
                cells = ops.process.find_cells(nuclei, filled)
        except ValueError:
            print('segment_cells error -- no cells')
            cells = nuclei

        HEKs = ops.process.filter_by_region(cells, lambda r: area_min < r.area, threshold, relabel=False)

        return HEKs.astype(np.uint16)

    @staticmethod
    def _filter_cell_clumps(data, cells, wildcards, distance_threshold=10):
        """Filter cell mask to remove all cells within distance d of each other, from centroid (i,j).
        For each cell, find all cells within given bounds, 
        calculate Euclidean distance between them, and eliminate clumped cells
        """
        if np.all(cells==0):
            return np.zeros((1480,1480))

        df = (Snake._extract_features(cells, cells, wildcards))
        # add column for [x,y] positions
        df['ij'] = df[['i','j']].values.tolist()
        ij = df['ij'].values.tolist()

        # calculate matrix of Euclidean distance between all cells in FOV
        distance = scipy.spatial.distance.cdist(ij, ij, 'euclidean')
        min_dist = np.where(distance>0, distance,distance.max()).min(1)
        # cells (labels) that pass distance threshold from nearest neighbor
        try:
            min_idx = np.hstack(np.argwhere(min_dist > distance_threshold))
            label = df.iloc[min_idx]
            mask = np.isin(cells, np.array(label['label'].values.tolist()))
            filtered_cells = np.multiply(mask.astype(int),cells)
        except:
            filtered_cells = np.zeros((1480,1480))

        return filtered_cells


    @staticmethod
    def _segment_cell_edges(cells, thickness=4):
        """
        Find edges of HEK cells only
        """
        cells = np.array(cells, dtype=np.uint16)

        boundaries = skimage.segmentation.find_boundaries(cells, connectivity=2, mode='inner').astype(np.uint16)
        expanded = scipy.ndimage.binary_dilation(boundaries, iterations=thickness).astype(np.uint16)

        # keep same label as cell
        edges = np.multiply(cells, expanded)

        return edges


    @staticmethod
    def _segment_puncta(data, threshold, channel, radius, area_min, area_max):
        """Find synaptic marker puncta from phenotyping channel.
        Expects log-filtered data to have shape (CYCLE, CHANNEL, I, J)
        """
        data = np.array(data)
        if data.ndim == 4:
            data = data[0]

        if isinstance(data, list):
            marker = data[int(channel)].astype(np.uint16) 
        elif data.ndim == 3:
            marker = data[int(channel)].astype(np.uint16) 
        else:
            marker = data.astype(np.uint16)

        kwargs = dict(threshold=lambda x: threshold, 
            radius=radius, area_min=area_min, area_max=area_max)

        # skimage precision warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            puncta = ops.process.find_puncta(marker, **kwargs)

        return puncta.astype(np.uint16)

    @staticmethod
    def _transform_log(data, sigma=1, skip_index=None):
        """Apply Laplacian-of-Gaussian filter from scipy.ndimage.
        Use `skip_index` to skip transforming a channel (e.g., DAPI with `skip_index=0`).
        """
        data = np.array(data)
        loged = ops.process.log_ndi(data, sigma=sigma)
        if skip_index is not None:
            loged[..., skip_index, :, :] = data[..., skip_index, :, :]
        return loged

    @staticmethod
    def _compute_std(data, remove_index=None):
        """Use standard deviation to estimate sequencing read locations.
        """
        if remove_index is not None:
            data = remove_channels(data, remove_index)

        leading_dims = tuple(range(0, data.ndim - 2))
        consensus = np.std(data, axis=leading_dims) # this is for 1-cycle data
        # consensus = np.std(data, axis=0).mean(axis=0) # this is for multi-cycle data


        return consensus
    
    @staticmethod
    def _find_peaks(data, width=5, remove_index=None):
        """Find local maxima and label by difference to next-highest neighboring
        pixel.
        """
        if remove_index is not None:
            data = remove_channels(data, remove_index)

        if data.ndim == 2:
            data = [data]

        peaks = [ops.process.find_peaks(x, n=width) 
                    if x.max() > 0 else x 
                    for x in data]
        peaks = np.array(peaks).squeeze()
        return peaks

    @staticmethod
    def _max_filter(data, width, remove_index=None):
        """Apply a maximum filter in a window of `width`.
        """
        import scipy.ndimage.filters

        if data.ndim == 2:
            data = data[None, None]
        if data.ndim == 3:
            data = data[None]

        if remove_index is not None:
            data = remove_channels(data, remove_index)
        
        maxed = scipy.ndimage.filters.maximum_filter(data, size=(1, 1, width, width))
    
        return maxed

    @staticmethod
    def _extract_bases(maxed, peaks, cells, threshold_peaks, wildcards, bases='GTAC'):
        """Find the signal intensity from `maxed` at each point in `peaks` above 
        `threshold_peaks`. Output is labeled by `wildcards` (e.g., well and tile) and 
        label at that position in integer mask `cells`.
        """

        if maxed.ndim == 3:
            maxed = maxed[None]

        if len(bases) != maxed.shape[1]:
            error = 'Sequencing {0} bases {1} but maxed data had shape {2}'
            raise ValueError(error.format(len(bases), bases, maxed.shape))

        # "cycle 0" is reserved for phenotyping
        cycles = list(range(1, maxed.shape[0] + 1))
        bases = list(bases)

        values, labels, positions = (
            ops.in_situ.extract_base_intensity(maxed, peaks, cells, threshold_peaks))

        df_bases = ops.in_situ.format_bases(values, labels, positions, cycles, bases)

        for k,v in sorted(wildcards.items()):
            df_bases[k] = v

        return df_bases

    @staticmethod
    def _call_reads(df_bases, peaks=None, correction_only_in_cells=True):
        """Median correction performed independently for each tile.
        Use the `correction_only_in_cells` flag to specify if correction
        is based on reads within cells, or all reads.
        """
        if df_bases is None:
            return
        if correction_only_in_cells:
            if len(df_bases.query('cell > 0')) == 0:
                return
        
        cycles = len(set(df_bases['cycle']))
        channels = len(set(df_bases['channel']))

        df_reads = (df_bases
            .pipe(ops.in_situ.clean_up_bases)
            .pipe(ops.in_situ.do_median_call, cycles, channels=channels,
                correction_only_in_cells=correction_only_in_cells)
            )

        if peaks is not None:
            i, j = df_reads[['i', 'j']].values.T
            df_reads['peak'] = peaks[i, j]

        return df_reads

    @staticmethod
    def _call_reads_percentiles(df_bases, peaks=None, correction_only_in_cells=True, imaging_order='GTAC', percentile=98, correction_by_cycle=False):
        # print(imaging_order)
        """Median correction performed independently for each tile.
        Use the `correction_only_in_cells` flag to specify if correction
        is based on reads within cells, or all reads.
        """
        if df_bases is None:
            return
        if correction_only_in_cells:
            if len(df_bases.query('cell > 0')) == 0:
                return


        cycles = len(set(df_bases['cycle']))
        channels = len(set(df_bases['channel']))

        df_reads = (df_bases
            .pipe(ops.in_situ.clean_up_bases)
            .pipe(ops.in_situ.do_percentile_call, cycles=cycles, channels=channels, 
                correction_only_in_cells=correction_only_in_cells, percentile=percentile, correction_by_cycle=correction_by_cycle)
            )

        if peaks is not None:
            i, j = df_reads[['i', 'j']].values.T
            df_reads['peak'] = peaks[i, j]

        return df_reads

    @staticmethod
    def _call_cells(df_reads, q_min=0):
        """Median correction performed independently for each tile.
        """
        if df_reads is None:
            return
        
        return (df_reads
            .query('Q_min >= @q_min')
            .pipe(ops.in_situ.call_cells))

    @staticmethod
    def _extract_features(data, labels, wildcards, features=None):
        """Extracts features in dictionary and combines with generic region
        features.
        """
        from ops.process import feature_table
        from ops.features import features_basic
        features = features.copy() if features else dict()
        features.update(features_basic)

        df = feature_table(data, labels, features)

        for k,v in sorted(wildcards.items()):
            df[k] = v
        
        return df

    @staticmethod
    def _extract_phenotype_synapse_puncta(data_phenotype, puncta, cells, wildcards):
        """Expects data_phenotype to have shape (CHANNEL, I, J)"""

        if puncta.max() == 0:
            return
        if np.all(cells==0):
            return
        if data_phenotype.ndim == 4:
            data_phenotype = data_phenotype[0]
        
        import ops.features

        # add cell labels corresponding to each punctae
        cell_label, zero_label = ops.process.assign_cells_puncta(cells, puncta)
        filtered_puncta = np.isin(puncta,zero_label,invert=True)*puncta

        features_p = ops.features.features_synapse_puncta
        features_p = {k + '_puncta': v for k,v in features_p.items()}

        df_p = (Snake._extract_features(data_phenotype, filtered_puncta, wildcards, features_p)
            .rename(columns={'area': 'area_puncta'}))
        
        try:
            df_p['cell'] = df_p['label'].map(cell_label)
            df_p = df_p[df_p.cell != 0]
            df_p.rename(columns={'label': 'punct'},inplace=True)
        except:
            df_p.insert(0,'cell',0)
        
        return df_p

    @staticmethod
    def _extract_phenotype_synapse_cell(data_phenotype, cells, wildcards):
        """Expects data_phenotype to have shape (CHANNEL, I, J)"""

        if cells.max() == 0:
            return
        if data_phenotype.ndim == 4:
            data_phenotype = data_phenotype[0]

        import ops.features

        features_c = ops.features.features_synapse_cell

        features_c = {k + '_cell': v for k,v in features_c.items()}

        df =  (Snake._extract_features(data_phenotype, cells, wildcards, features_c)
            .rename(columns={'area': 'area_cell'}))

        # add column for [x,y] positions
        df['ij'] = df[['i','j']].values.tolist()
        ij = df['ij'].values.tolist()

        # calculate matrix of Euclidean distance between all cells in FOV
        distance = scipy.spatial.distance.cdist(ij, ij, 'euclidean')
        min_dist = np.where(distance>0, distance,distance.max()).min(1)
        df['min_dist'] = min_dist

        return (df.rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_synapse_cell_edge(data_phenotype, cells, edges, wildcards):
        """Expects data_phenotype to have shape (CYCLE, CHANNEL, I, J)"""

        if cells.max() == 0:
            return
        if data_phenotype.ndim == 4:
            data_phenotype = data_phenotype[0]

        import ops.features

        features_c = ops.features.features_synapse_cell
        features_e = ops.features.features_synapse_edge

        features_c = {k + '_cell': v for k,v in features_c.items()}
        features_e = {k + '_edge': v for k,v in features_e.items()}
        

        df_c =  (Snake._extract_features(data_phenotype, cells, wildcards, features_c)
            .rename(columns={'area': 'area_cell'}))
        df_e = (Snake._extract_features(data_phenotype, edges, wildcards, features_e)
            .drop(['i', 'j'], axis=1).rename(columns={'area': 'area_edge'}))

        # inner join discards edges without corresponding cells
        df = (pd.concat([df_c.set_index('label'), df_e.set_index('label')], axis=1, join='inner')
                .reset_index())


        # add column for [x,y] positions
        df['ij'] = df[['i','j']].values.tolist()
        ij = df['ij'].values.tolist()

        # calculate matrix of Euclidean distance between all cells in FOV
        distance = scipy.spatial.distance.cdist(ij, ij, 'euclidean')
        min_dist = np.where(distance>0, distance,distance.max()).min(1)
        df['min_dist'] = min_dist

        return (df.rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_FR(data_phenotype, nuclei, wildcards):
        """Features for frameshift reporter phenotyped in DAPI, HA channels.
        """
        from ops.features import features_frameshift
        return (Snake._extract_features(data_phenotype, nuclei, wildcards, features_frameshift)
             .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_FR_myc(data_phenotype, nuclei, data_sbs_1, wildcards):
        """Features for frameshift reporter phenotyped in DAPI, HA, myc channels.
        """
        from ops.features import features_frameshift_myc
        return (Snake._extract_features(data_phenotype, nuclei, wildcards, features_frameshift_myc)
            .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_translocation(data_phenotype, nuclei, cells, wildcards):
        if (nuclei.max() == 0) or (cells.max() == 0):
            return

        import ops.features

        features_n = ops.features.features_translocation_nuclear
        features_c = ops.features.features_translocation_cell

        features_n = {k + '_nuclear': v for k,v in features_n.items()}
        features_c = {k + '_cell': v    for k,v in features_c.items()}

        df_n = (Snake._extract_features(data_phenotype, nuclei, wildcards, features_n)
            .rename(columns={'area': 'area_nuclear'}))

        df_c =  (Snake._extract_features(data_phenotype, cells, wildcards, features_c)
            .drop(['i', 'j'], axis=1).rename(columns={'area': 'area_cell'}))


        # inner join discards nuclei without corresponding cells
        df = (pd.concat([df_n.set_index('label'), df_c.set_index('label')], axis=1, join='inner')
                .reset_index())

        return (df
            .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_translocation_live(data, nuclei, wildcards):
        def _extract_phenotype_translocation_simple(data, nuclei, wildcards):
            import ops.features
            features = ops.features.features_translocation_nuclear_simple
            
            return (Snake._extract_features(data, nuclei, wildcards, features)
                .rename(columns={'label': 'cell'}))

        extract = _extract_phenotype_translocation_simple
        arr = []
        for i, (frame, nuclei_frame) in enumerate(zip(data, nuclei)):
            arr += [extract(frame, nuclei_frame, wildcards).assign(frame=i)]

        return pd.concat(arr)

    @staticmethod
    def _extract_phenotype_translocation_ring(data_phenotype, nuclei, wildcards, width=3):
        selem = np.ones((width, width))
        perimeter = skimage.morphology.dilation(nuclei, selem)
        perimeter[nuclei > 0] = 0

        inside = skimage.morphology.erosion(nuclei, selem)
        inner_ring = nuclei.copy()
        inner_ring[inside > 0] = 0

        return (Snake._extract_phenotype_translocation(data_phenotype, inner_ring, perimeter, wildcards)
            .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_minimal(data_phenotype, nuclei, wildcards):
        return (Snake._extract_features(data_phenotype, nuclei, wildcards, dict())
            .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_geom(labels, wildcards):
        from ops.features import features_geom
        return Snake._extract_features(labels, labels, wildcards, features_geom)

    @staticmethod
    def _analyze_single(data, alignment_ref, cells, peaks, 
                        threshold_peaks, wildcards, channel_ix=1):
        if alignment_ref.ndim == 3:
            alignment_ref = alignment_ref[0]
        data = np.array([[alignment_ref, alignment_ref], 
                          data[[0, channel_ix]]])
        aligned = ops.process.Align.align_between_cycles(data, 0, window=2)
        loged = Snake._transform_log(aligned[1, 1])
        maxed = Snake._max_filter(loged, width=3)
        return (Snake._extract_bases(maxed, peaks, cells, bases=['-'],
                    threshold_peaks=threshold_peaks, wildcards=wildcards))

    @staticmethod
    def _track_live_nuclei(nuclei, tolerance_per_frame=5):
        
        # if there are no nuclei, we will have problems
        count = nuclei.max(axis=(-2, -1))
        if (count == 0).any():
            error = 'no nuclei detected in frames: {}'
            print(error.format(np.where(count == 0)))
            return np.zeros_like(nuclei)

        import ops.timelapse

        # nuclei coordinates
        arr = []
        for i, nuclei_frame in enumerate(nuclei):
            extract = Snake._extract_phenotype_minimal
            arr += [extract(nuclei_frame, nuclei_frame, {'frame': i})]
        df_nuclei = pd.concat(arr)

        # track nuclei
        motion_threshold = len(nuclei) * tolerance_per_frame
        G = (df_nuclei
          .rename(columns={'cell': 'label'})
          .pipe(ops.timelapse.initialize_graph)
        )

        cost, path = ops.timelapse.analyze_graph(G)
        relabel = ops.timelapse.filter_paths(cost, path, 
                                    threshold=motion_threshold)
        nuclei_tracked = ops.timelapse.relabel_nuclei(nuclei, relabel)

        return nuclei_tracked

    @staticmethod
    def _merge_triangle_hash(df_0,df_1,alignment,threshold=2):
        import ops.triangle_hash as th
        df_1 = df_1.rename(columns={'tile':'site'})
        model = th.build_linear_model(alignment['rotation'],alignment['translation'])
        
        try:
            merged = th.merge_sbs_phenotype(df_0,df_1,model,threshold=threshold)
        except:
            # if info file is empty, return empty dataframe
            merged = pd.DataFrame(columns=['well','tile','cell_0','i_0','j_0','site','cell_1', 'i_1', 'j_1', 'distance'])                   

        return merged

    @staticmethod
    def add_method(class_, name, f):
        f = staticmethod(f)
        exec('%s.%s = f' % (class_, name))

    @staticmethod
    def load_methods():
        methods = inspect.getmembers(Snake)
        for name, f in methods:
            if name not in ('__doc__', '__module__') and name.startswith('_'):
                Snake.add_method('Snake', name[1:], Snake.call_from_snakemake(f))

    @staticmethod
    def call_from_snakemake(f):
        """Turn a function that acts on a mix of image data, table data and other 
        arguments and may return image or table data into a function that acts on 
        filenames for image and table data, plus other arguments.

        If output filename is provided, saves return value of function.

        Supported input and output filetypes are .pkl, .csv, and .tif.
        """
        def g(**kwargs):

            # split keyword arguments into input (needed for function)
            # and output (needed to save result)
            input_kwargs, output_kwargs = restrict_kwargs(kwargs, f)

            # load arguments provided as filenames
            input_kwargs = {k: load_arg(v) for k,v in input_kwargs.items()}

            results = f(**input_kwargs)

            if 'output' in output_kwargs:
                outputs = output_kwargs['output']
                
                if len(outputs) == 1:
                    results = [results]

                if len(outputs) != len(results):
                    error = '{0} output filenames provided for {1} results'
                    raise ValueError(error.format(len(outputs), len(results)))

                for output, result in zip(outputs, results):
                    save_output(output, result, **output_kwargs)

        return functools.update_wrapper(g, f)


Snake.load_methods()


def remove_channels(data, remove_index):
    """Remove channel or list of channels from array of shape (..., CHANNELS, I, J).
    """
    channels_mask = np.ones(data.shape[-3], dtype=bool)
    channels_mask[remove_index] = False
    data = data[..., channels_mask, :, :]
    return data


def saturated_comp(data, threshold):
    """Clip the max intensity to lower intensity of saturated pixels."""

    data[data > threshold] = threshold

    return data


# IO


def load_arg(x):
    """Try loading data from `x` if it is a filename or list of filenames.
    Otherwise just return `x`.
    """
    one_file = load_file
    many_files = lambda x: [load_file(f) for f in x]
    
    for f in one_file, many_files:
        try:
            return f(x)
        except (pd.errors.EmptyDataError, TypeError, IOError) as e:
            if isinstance(e, (TypeError, IOError)):
                # wasn't a file, probably a string arg
                pass
            elif isinstance(e, pd.errors.EmptyDataError):
                # failed to load file
                return None
            pass
    else:
        return x


def save_output(filename, data, **kwargs):
    """Saves `data` to `filename`. Guesses the save function based on the
    file extension. Saving as .tif passes on kwargs (luts, ...) from input.
    """
    filename = str(filename)
    if data is None:
        # need to save dummy output to satisfy Snakemake
        with open(filename, 'w') as fh:
            pass
        return
    if filename.endswith('.tif'):
        return save_tif(filename, data, **kwargs)
    elif filename.endswith('.pkl'):
        return save_pkl(filename, data)
    elif filename.endswith('.csv'):
        return save_csv(filename, data)
    else:
        raise ValueError('not a recognized filetype: ' + f)


def load_csv(filename):
    df = pd.read_csv(filename)
    if len(df) == 0:
        return None
    return df


def load_pkl(filename):
    df = pd.read_pickle(filename)
    if len(df) == 0:
        return None


def load_tif(filename):
    return ops.io.read_stack(filename)


def save_csv(filename, df):
    df.to_csv(filename, index=None)


def save_pkl(filename, df):
    df.to_pickle(filename)


def save_tif(filename, data_, **kwargs):
    kwargs, _ = restrict_kwargs(kwargs, ops.io.save_stack)
    # `data` can be an argument name for both the Snake method and `save_stack`
    # overwrite with `data_` 
    kwargs['data'] = data_
    ops.io.save_stack(filename, **kwargs)


def restrict_kwargs(kwargs, f):
    """Partition `kwargs` into two dictionaries based on overlap with default 
    arguments of function `f`.
    """
    f_kwargs = set(get_kwarg_defaults(f).keys()) | set(get_arg_names(f))
    keep, discard = {}, {}
    for key in kwargs.keys():
        if key in f_kwargs:
            keep[key] = kwargs[key]
        else:
            discard[key] = kwargs[key]
    return keep, discard


def load_file(filename):
    """Attempt to load file, raising an error if the file is not found or 
    the file extension is not recognized.
    """
    if not isinstance(filename, str):
        raise TypeError
    if not os.path.isfile(filename):
        raise IOError(2, 'Not a file: {0}'.format(filename))
    if filename.endswith('.tif'):
        return load_tif(filename)
    elif filename.endswith('.pkl'):
        return load_pkl(filename)
    elif filename.endswith('.csv'):
        return load_csv(filename)
    else:
        raise IOError(filename)


def get_arg_names(f):
    """List of regular and keyword argument names from function definition.
    """
    argspec = inspect.getargspec(f)
    if argspec.defaults is None:
        return argspec.args
    n = len(argspec.defaults)
    return argspec.args[:-n]


def get_kwarg_defaults(f):
    """Get the kwarg defaults as a dictionary.
    """
    argspec = inspect.getargspec(f)
    if argspec.defaults is None:
        defaults = {}
    else:
        defaults = {k: v for k,v in zip(argspec.args[::-1], argspec.defaults[::-1])}
    return defaults


def load_well_tile_list(filename):
    if filename.endswith('pkl'):
        wells, tiles = pd.read_pickle(filename)[['well', 'tile']].values.T
    elif filename.endswith('csv'):
        wells, tiles = pd.read_csv(filename)[['well', 'tile']].values.T
    return wells, tiles
