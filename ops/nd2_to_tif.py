import matplotlib
matplotlib.use('TkAgg') #uncomment when running from Terminal
from nd2reader import ND2Reader
from ops.imports import *
import ops.filenames

def nd2_to_tif(input_filename,meta=True):
    
    # add parse_filename function to get info from nd2 name and convert to tif filename
    info = ops.filenames.parse_filename(input_filename)
    
    file_description={}
    for k,v in sorted(info.items()):
        file_description[k] = v
    file_description['ext']='tif'
    file_description['subdir']=file_description['plate']+'_tif/'+file_description['mag']+'_'+file_description['cycle']


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
    