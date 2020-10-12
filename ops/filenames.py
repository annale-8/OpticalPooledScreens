import re
import os
import time
from urllib.parse import urlparse
from glob import glob

from ops.constants import FILE


FILE_PATTERN = [
        r'((?P<home>.*)\/)?',
        r'(?P<dataset>(?P<date>[0-9]{8}).*?)\/',
        r'(?:(?P<subdir>.*)\/)*',
        r'(MAX_)?(?P<mag>[0-9]+[xX]).',
        r'(?:(?P<cycle>[^_\.]*).*?(?:.*MMStack)?.)?',
        r'(?P<well>[A-H][0-9]*)',
        r'(?:[_-]Site[_-](?P<site>([0-9]+)))?',
        r'(?:[_-]Channel[_-](?P<channel>.*)\_.*)?',
        r'(?:_Tile-(?P<tile>([0-9]+)))?',
        r'(?:\.(?P<tag>.*))*\.(?P<ext>.*)']

FOLDER_PATTERN = [
        r'(?P<mag>[0-9]+X).',
        r'(?:(?P<cycle>[^_\.]*).*?)\/',
        r'(?P<well>[A-H][0-9]+)',
        r'(?:[_-]Site[_-](?P<site>([0-9]+)))?',
        r'\/?']

ND2_FILE_PATTERN = [
        r'((?P<home>.*)\/)?',
        r'(?P<dataset>(?P<date>[0-9]{8}).*?)\/',
        r'(?:(?P<expt>.*)\/)',
        r'(?:(?P<plate>.*)\/)',
        r'(MAX_)?(?P<mag>[0-9]+[xX]).',
        r'(?:(?P<cycle>[^_\.]*)\/)',
        r'(?:Well(?P<well>[A-H][0-9]*))?',
        r'(?:_Channel(?P<channel>.*)\_.*)?',
        r'(?:\.(?P<tag>.*))*\.(?P<ext>.*)']

TIF_FILE_PATTERN = [
        r'((?P<home>.*)\/)?',
        r'(?P<dataset>(?P<date>[0-9]{8}).*?)\/',
        r'(?:(?P<subdir>.*)\/)*',
        r'(MAX_)?(?P<mag>[0-9]+[xX]).',
        r'(?:(?P<cycle>[^_\.]*))',
        r'(?:\_(?P<well>[A-H][0-9]*))?',
        r'(?:[-_]Channel[-_](?P<channel>.*))?',
        r'(?:\.(?P<tag>.*))*\.(?P<ext>.*)']


FILE_PATTERN_ABS = ''.join(FILE_PATTERN)
FILE_PATTERN_REL = ''.join(FILE_PATTERN[2:])

FOLDER_PATTERN_ABS = ''.join(FILE_PATTERN[:2] + FOLDER_PATTERN)
FOLDER_PATTERN_REL = ''.join(FOLDER_PATTERN)

ND2_FILE_PATTERN_ABS = ''.join(ND2_FILE_PATTERN)
ND2_FILE_PATTERN_REL = ''.join(ND2_FILE_PATTERN[2:])

TIF_FILE_PATTERN_ABS = ''.join(TIF_FILE_PATTERN)
TIF_FILE_PATTERN_REL = ''.join(TIF_FILE_PATTERN[2:])


def parse_filename(filename, custom_patterns=None):
    """Parse filename into dictionary. 

    Some entries in the dictionary are optional, e.g., cycle and tile.

    Examples:
        >>> parse('example_data/input/10X_c1-SBS-1/10X_c1-SBS-1_A1_Tile-107.max.tif')

        {'subdir': 'example_data/input/10X_c1-SBS-1',
         'mag': '10X',
         'cycle': 'c1-SBS-1',
         'well': 'A1',
         'tile': '107',
         'tag': 'max',
         'ext': 'tif',
         'file': 'example_data/input/10X_c1-SBS-1/10X_c1-SBS-1_A1_Tile-107.max.tif'}
    """
    filename = os.path.normpath(filename)
    # windows
    filename = filename.replace('\\', '/')

    patterns = [FILE_PATTERN_ABS, FILE_PATTERN_REL, 
                FOLDER_PATTERN_ABS, FOLDER_PATTERN_REL,
                ND2_FILE_PATTERN_ABS, ND2_FILE_PATTERN_REL,
                TIF_FILE_PATTERN_ABS, TIF_FILE_PATTERN_REL]

    if custom_patterns is not None:
        patterns += list(custom_patterns)

    for pattern in patterns:
        match = re.match(pattern, filename)
        try:
            result = {k:v for k,v in match.groupdict().items() if v is not None}
            # result[FILE] = filename  # convenience, not used by name_file
            result['file'] = filename
            return result
        except AttributeError:
            continue
    
    raise ValueError('failed to parse filename: %s' % filename)


def name_file(description, **more_description):
    """Name a file from a dictionary of filename parts. 

    Can override dictionary with keyword arguments.
    """
    d = dict(description)
    d.update(more_description)
    # if value is None, key is removed
    d = {k: v for k,v in d.items() if v is not None}

    if 'cycle' in d:
        d['first'] = '{mag}_{cycle}_{well}'.format(**d)
    else:
        d['first'] = '{mag}_{well}'.format(**d)

    # positions can be called either tile or site (e.g., tiles are in physical order
    # and sites are in acquisition order)
    if 'tile' in d:
        d['pos'] = 'Tile-{0}'.format(d['tile'])
    elif 'site' in d:
        d['pos'] = 'Site-{0}'.format(d['site'])
    # else:
    #     d['pos'] = None

    # for fast-mode acquisition nd2 files with separated channels
    if 'channel' in d:
        d['ch'] = 'Channel-{0}'.format(d['channel'])

    formats = [
        '{first}_{pos}_{ch}.{tag}.{ext}',
        '{first}_{pos}_{ch}.{ext}',
        '{first}_{pos}.{tag}.{ext}',
        '{first}_{pos}.{ext}',
        '{first}_{ch}.{tag}.{ext}',
        '{first}_{ch}.{ext}',
        '{first}.{tag}.{ext}',
        '{first}.{ext}',
    ]

    for fmt in formats:
        try:
            basename = fmt.format(**d)
            break
        except KeyError:
            continue
    else:
        raise ValueError('extension missing')
    
    optional = lambda x: d.get(x, '')
    filename = os.path.join(optional('home'), optional('dataset'), optional('subdir'), basename)
    return os.path.normpath(filename)


def normpath(filename):
    if not urlparse(filename).scheme: # leave remote urls alone
        filename = os.path.normpath(filename)
    return filename



def guess_filename(row, tag, **override_fields):
    description = {'subdir': 'process', 'mag': '10X', 
                    'tag': tag, 'ext': 'tif'}
    description.update(row.to_dict())
    description.update(override_fields)
    return name_file(description)


def make_filename(df, base_description, **kwargs):
    d = base_description.copy()
    arr = []
    for _, row in df.iterrows():
        d.update(row.to_dict())
        d.update(kwargs)
        arr.append(name_file(d))
    return arr


def make_filename_pipe(df, output_col, template_or_description=None, **kwargs):

    try:
        description = parse_filename(template_or_description)
    except TypeError:
        description = template_or_description.copy()

    arr = []
    for _, row in df.iterrows():
        description.update(row.to_dict())
        description.update(kwargs)
        arr.append(name_file(description))

    return df.assign(**{output_col: arr})


def timestamp(filename='', fmt='%Y%m%d_%H%M%S', sep='.'):
    stamp = time.strftime(fmt)
    pat= r'(.*)\.(.*)'
    match = re.findall(pat, filename)
    if match:
        return sep.join([match[0][0], stamp, match[0][1]])
    elif filename:
        return sep.join([filename, stamp])
    else:
        return stamp


def file_frame(files_or_search, **kwargs):
    """Convenience function, pass either a list of files or a 
    glob wildcard search term. Extra arguments passed to `parse_filename`.
    """
    from natsort import natsorted
    import pandas as pd

    if isinstance(files_or_search, str):
        files = natsorted(glob(files_or_search))
    else:
        files = files_or_search

    return pd.DataFrame([parse_filename(f, **kwargs) for f in files])