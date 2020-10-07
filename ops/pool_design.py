from collections import defaultdict, Counter
import scipy.sparse
import numpy as np
import pandas as pd
import os
from Levenshtein import distance
from natsort import natsorted
from functools import partial

from tqdm import tqdm_notebook as tqdn

import ops.utils
from ops.constants import *


# LOAD TABLES

def validate_design(df_design):
    for group, df in df_design.groupby('group'):
        x = df.drop_duplicates(['prefix_length', 'edit_distance'])
        if len(x) > 1:
            cols = ['group', DESIGN, 'prefix_length', 'edit_distance']
            error = 'multiple prefix specifications for group {0}:\n{1}'
            raise ValueError(error.format(group, df[cols]))
            
    return df_design


def load_gene_list(filename,gene_id=GENE_ID,dtype=None):
    return (pd.read_csv(filename, header=None, dtype=dtype)
     .assign(design=os.path.splitext(filename)[0])
     .rename(columns={0: gene_id})
    )


def validate_genes(df_genes, df_sgRNAs,gene_id=GENE_ID):
    missing = set(df_genes[gene_id]) - set(df_sgRNAs[gene_id])
    if missing:
        error = '{0} gene ids missing from sgRNA table: {1}'
        missing_ids = ', '.join(map(str, missing))
        raise ValueError(error.format(len(missing), missing_ids))

    duplicates = df_genes[[SUBPOOL, gene_id]].duplicated(keep=False)
    if duplicates.any():
        error = 'duplicate genes for the same subpool: {0}'
        xs = df_genes.loc[duplicates, [SUBPOOL, gene_id]].values
        raise ValueError(error.format(xs))

    return df_genes

def design_gene_symbol(df_design_gene_id,df_gene_symbol=pd.DataFrame()):
    if df_gene_symbol.empty:
      df_gene_symbol = df_design_gene_id
    df_gene_symbol = (df_gene_symbol
      .drop_duplicates('gene_id')
      [['gene_id','gene_symbol']]
      )

    def parse_gene_id(design_gene_id):
      return natsorted([int(id) for id in design_gene_id.split('&')])

    df_design_gene_id['design_gene_symbol'] = (df_design_gene_id['design_gene_id']
      .apply(lambda x: '&'.join(df_gene_symbol
        .query('gene_id == {}'.format(str(parse_gene_id(x))))
        ['gene_symbol']
        .tolist()
        ))
      )

    return df_design_gene_id

def multiple_targets(df_sgRNAs):
    return df_sgRNAs.duplicated(subset=['sgRNA'],keep=False).astype('int')

# SELECT GUIDES

def select_prefix_group(df_genes, df_sgRNAs, priority=[RANK], n_cores=-2, gene_id=GENE_ID):
    """priority is for priority within gene id
    """
    # doesn't shortcut if some genes need less guides
    prefix_length, edit_distance = (df_genes[[PREFIX_LENGTH, EDIT_DISTANCE]].values[0])
    
    print('selecting guides')
    return (df_sgRNAs
        .join(df_genes.set_index(gene_id), on=gene_id, how='inner')
        .assign(sgRNAs_per_gene_exist=lambda x: x.groupby(gene_id)[SGRNA].transform(len))
        .pipe(select_guides, prefix_length, edit_distance, gene_id, priority, n_cores)
        .drop(columns=['sgRNAs_per_gene_exist'])
    )


def select_guides(df_input, prefix_length, edit_distance, gene_id=GENE_ID, priority=[RANK], n_cores=-2):
    """`df_input` has gene_id, sgRNAs_per_gene
    priority is for priority within a gene id
    """

    df_deduped = (df_input.copy()
          .assign(prefix=lambda x: x['sgRNA'].str[:prefix_length])
          # guides from genes with more guides more likely to have lower rank, this sort order seems safe:
          .sort_values([RANK,'sgRNAs_per_gene_exist'])
          .drop_duplicates('prefix')
          )

    if edit_distance != 1:
        print('edit distance detected')
        # selected_guides = 
        return (df_deduped
          .assign(sgRNAs_per_gene_exist=lambda x: 
              x.groupby(gene_id)[SGRNA].transform(len))
          .sort_values(priority)
          .pipe(find_group_cliques,prefix_length,edit_distance,
            gene_id, n_cores)
          # [SGRNA].pipe(list)
          )
        # return df_input.query(loc('{SGRNA} == @selected_guides'))

    else:
        return (df_deduped
         .sort_values([SUBPOOL,gene_id]+priority)
         .assign(selected_rank=lambda x: 
            ops.utils.rank_by_order(x, [SUBPOOL, gene_id]))
         .query('selected_rank <= sgRNAs_per_gene')
         .sort_values([SUBPOOL, gene_id, 'selected_rank'])
         .drop(['selected_rank'], axis=1)
         # [SGRNA].pipe(list)
         )
        # return df_input.query(loc('{SGRNA} == @selected_guides'))

def find_group_cliques(df_input, prefix_length=12, edit_distance=2,gene_id=GENE_ID, n_cores=-2):

    prefixes = df_input['sgRNA'].str[:prefix_length+1].pipe(list)

    hash_buckets = build_khash(tqdn(prefixes,'hash'), edit_distance)
    
    # for parallel distance calculation
    arr = [[x] for x in hash_buckets]

    print('hashed')

    # f = partial(sparse_dist, threshold=edit_distance
    #             ,distance_func=distance_prefix
    #            )

    # print('sparse_dist function initialized')
    # import multiprocessing
    # with multiprocessing.Pool(n_cores) as p:
    #     r = list(tqdn(p.imap(f, arr), 'distance',total=len(arr)))

    from joblib import Parallel, delayed
    results = Parallel(n_cores)(delayed(sparse_dist)(bucket,threshold=edit_distance,
                                                     distance_func=distance_prefix) 
                                for bucket
                                in tqdn(arr,'distance'))

    print('distanced')

    Distance = dict()
    for x in results:
        Distance.update(x)

    sparse_distance = sparse_view(prefixes, Distance)

    selected = maxy_clique_groups(sparse_distance, df_input[gene_id].pipe(list), df_input['sgRNAs_per_gene'].pipe(list))
    # xs = [prefixes[i] for i in selected]

    return df_input.iloc[selected]

def build_khash(prefixes, k, return_dict=False):
    """builds prefix substring hash and groups prefixes for prioritization of
    levenshtein distance calculation
    """
    D = defaultdict(list)
    # makes dictionary of {(position,substring) hash:prefix(es)}
    # prefixes sharing one hash are less than levenshtein distance k apart
    for x in prefixes:
        for h in khash(x, k):
             D[h].append(x)

    # remove duplicate prefixes and sort
    D = {k: sorted(set(v)) for k,v in D.items()}
    if return_dict:
        return D
    else:
        hash_buckets = list(D.values())
        return hash_buckets

def khash(s, k):
    """Divide a string into substrings suitable for checking edit distance of 
    `k`. The substrings are marked by position in the string. Two strings of 
    the same length with Levenshtein edit distance less than `k` will share 
    at least one (position, substring) pair.
    """
    n = len(s)
    window = int(np.ceil((n - k) / float(k)))
    s = s + s
    arr = []
    for i in range(n):
        # arr += [s[i:i+window]]
        # for including single insertions? (LF comment)
        for j in (0, 1):
            # builds tuples of (position,slice of size window)
            arr += [((i + j) % n, s[i:i+window])]
    return arr

def sparse_dist(hash_buckets, threshold,distance_func=None):
    """Calculates levenshtein distance between prefixes in each 
    hash bucket; entries only kept if less than threshold.
    """
    if distance_func is None:
        distance_func = distance
    D = defaultdict(int)
    # for each hash bucket
    for prefixes in hash_buckets:
    # calculate distance between all prefixes within bucket
      for i, a in enumerate(prefixes):
          for b in prefixes[i+1:]:
              d = distance_func(a,b)
              # only keep if less than threshold
              if d < threshold:
                  key = tuple(sorted((a,b)))
                  D[key] = d
    return D

def distance_prefix(a, b):
    """Hack to get edit distance of string prefixes to include insertions/deletions that
    shift characters into/out of the prefix. Must pass prefixes of length n+1;
    only works for single character insertion/deletion/substitution. Should be equivalent
    to Levenshtein distance, ignoring the n + 1 position.
    """
    compare = [
        # substitution
        (a[:-1], b[:-1]),
        # deletion
        (a[:-1], b),
        (a, b[:-1]),
        # insertion
        (a[:-1], b[:-1] + a[-1]),
        (b[:-1], a[:-1] + b[-1]),
    ]
    return min(distance(x1, x2) for x1, x2 in compare)

def sparse_view(prefixes, D, symmetric=True):
    """string barcodes
    returns sparse matrix marking where levenshtein distance is less than threshold in sparse_dist
    """
    # give each prefix a number id
    mapper = {x: i for i, x in enumerate(prefixes)}
    f = lambda x: mapper[x]
    i,j,data = zip(*[(f(a), f(b), v) for (a,b),v in D.items()])
    # levenshtein distance greater than 0
    data = np.array(data) > 0
    # prefix ids
    i = np.array(i)
    j = np.array(j)

    # generate sparse matrix with entries = 1 where distance < threshold from sparse_dist()
    n = len(prefixes)
    cm = scipy.sparse.coo_matrix((data, (i, j)), shape=(n, n))

    if symmetric:
        cm = (cm + cm.T).tocsr()
        
    return cm

def maxy_clique_groups(cm, group_ids, prefixes_per_group):
    """
    Selects prefixes from all possible prefix indices
    Take up to `m`. Prioritizes groups with the fewest selected barcodes.
    Prioritizing groups with the fewest remaining barcodes could give
    better results.
    """

    # TODO: m as a list

    # make dictionary to map prefix counts => group_ids
    d_available_counts = defaultdict(set)
    for group_id_, counts in Counter(group_ids).items():
        d_available_counts[counts] |= {group_id_}

    # make dictionaries to map group_id => prefix indices of sparse matrix and desired # of prefixes
    d_group_indices = defaultdict(list)
    d_num_to_select = dict()
    for i_prefix, (group_id_,prefixes_per_group) in enumerate(zip(group_ids,prefixes_per_group)):
        d_group_indices[group_id_] += [i_prefix]
        d_num_to_select[group_id_] = prefixes_per_group
    # reverse list because .pop() takes from the end of the list later
    d_group_indices = {k: v[::-1] for k,v in d_group_indices.items()}

    # make dictionary to map group_id => # selected
    d_selected_counts = Counter()

    selected = []
    available = np.array(range(len(group_ids)))

    while d_available_counts:
        if (len(selected) % 100) == 0:
            print(str(len(selected))+' prefixes selected')
    #     assert cm[selected, :][:, selected].sum() == 0

        # pick a group_id from the lowest prefix count bin
        count = min(d_available_counts.keys())
        group_id_ = d_available_counts[count].pop()

        # remove bin if empty
        if len(d_available_counts[count]) == 0:
            d_available_counts.pop(count)

        # only take up to m prefixes per group
        if d_selected_counts[group_id_] == d_num_to_select[group_id_]:
            # group_id is destroyed
            continue

        # discard indices until we find a new one
        prefix_index = None
        while d_group_indices[group_id_]:
            prefix_index = d_group_indices[group_id_].pop()
            # approach 1: check for conflict every time
            # cm[index, selected].sum() == 0
            # approach 2: keep an array of available indices
            if prefix_index in available:
                break
        else:
            prefix_index = None

        # keep index
        if prefix_index:
            selected.append(prefix_index)
            d_selected_counts[group_id_] += 1
            available = available[available != prefix_index]
            # get rid of incompatible barcodes
            remove = cm[prefix_index, available].indices
            mask = np.ones(len(available), dtype=bool)
            mask[remove] = False
            available = available[mask]


        # move group_id to new available_counts bin
        n = len(d_group_indices[group_id_])
        if n > 0:
            d_available_counts[n] |= {group_id_}

    return selected

def parallel_levenshtein_group(group, dist_func=None, n_cores=-2):
    remainders = [group[i+1:] for i,_ in enumerate(group)]
    if not dist_func:
        dist_func = Levenshtein.distance
        
    def measure_distances(string,remainder):
        arr = []
        for test_string in remainder:
            d = dist_func(string,test_string)
            if d<2:
                print(string,test_string)
            arr.append(d)
        return arr
    
    from joblib import Parallel, delayed
    results = Parallel(n_cores)(delayed(measure_distances)(*subset) 
                              for subset 
                              in tqdn(zip(group,remainders),total=len(group)))
    distances = []
    for result in results:
        distances.extend(result)
        
    return distances

# FILTER SGRNAS

def filter_sgRNAs(df_sgRNAs, homopolymer=5):
    cut = [has_homopolymer(x, homopolymer) or has_BsmBI_site(x) 
            for x in df_sgRNAs[SGRNA]]
    return df_sgRNAs[~np.array(cut)]

def has_homopolymer(x, n):
    a = 'A'*n in x
    t = 'T'*n in x
    g = 'G'*n in x
    c = 'C'*n in x
    return a | t | g | c
   
def has_BsmBI_site(x):
    x = 'CACCG' + x.upper() + 'GTTT'
    return 'CGTCTC' in x or 'GAGACG' in x


# OLIGOS

def build_sgRNA_oligos(df, dialout_primers, 
                        left='CGTCTCgCACCg', right='GTTTcGAGACG'):

    template = '{fwd}{left}{sgRNA}{right}{rev}'
    arr = []
    for s, d in df[[SGRNA, DIALOUT]].values:
        fwd, rev = dialout_primers[d]
        rev = reverse_complement(rev)
        oligo = template.format(fwd=fwd, rev=rev, sgRNA=s, 
                                left=left, right=right)
        arr += [oligo]
    return arr


def build_test(df_oligos, dialout_primers):
    """Pattern-match sgRNA cloning and dialout primers.
    """
    sites = 'CGTCTC', reverse_complement('CGTCTC')
    pat = ('(?P<dialout_fwd>.*){fwd}.CACCG'
           '(?P<sgRNA_cloned>.*)'
           'GTTT.{rev}(?P<dialout_rev>.*)')
    pat = pat.format(fwd=sites[0], rev=sites[1])

    kosuri = {}
    for i, (fwd, rev) in enumerate(dialout_primers):
        kosuri[fwd] = 'fwd_{0}'.format(i)
        kosuri[rev] = 'rev_{0}'.format(i)

    def validate_design(df):
        if not (df[VECTOR] == 'CROPseq').all():
            raise ValueError('can only validate CROPseq design')
        return df

    return (df_oligos
     .pipe(validate_design)
     .assign(sgRNA=lambda x: x['sgRNA'].str.upper())
     .assign(oligo=lambda x: x['oligo'].str.upper())
     .pipe(lambda x: pd.concat([x, x['oligo'].str.extract(pat)], axis=1))
     .assign(dialout_rev=lambda x: x['dialout_rev'].apply(reverse_complement))
     .assign(dialout_fwd_ix=lambda x: x['dialout_fwd'].apply(kosuri.get))      
     .assign(dialout_rev_ix=lambda x: x['dialout_rev'].apply(kosuri.get))            
     .assign(dialout_ix=lambda x: 
             x['dialout_fwd_ix'].str.split('_').str[1].astype(int))
    )


def validate_test(df_test):
    """Check sgRNA cloning and identiy of dialout primers.
    """
    assert df_test.eval('sgRNA_cloned == sgRNA').all()

    assert (df_test['dialout_fwd_ix'].str[-1] == 
            df_test['dialout_rev_ix'].str[-1]).all()

    assert df_test.eval('dialout_ix== dialout').all()

    print('Looking good!')

    return df_test


def reverse_complement(seq):
    watson_crick = {'A': 'T',
                'T': 'A',
                'C': 'G',
                'G': 'C',
                'U': 'A',
                'N': 'N'}

    watson_crick.update({k.lower(): v.lower() 
        for k, v in watson_crick.items()})

    return ''.join(watson_crick[x] for x in seq)[::-1]


# EXTERNAL

def import_brunello(filename):
    """Import "Brunello Library Target Genes", which can be found at:
    https://www.addgene.org/pooled-library/broadgpp-human-knockout-brunello/
    """
    columns = {'Target Gene ID': GENE_ID
              ,'Target Gene Symbol': GENE_SYMBOL
              ,'sgRNA Target Sequence': SGRNA
              , 'Rule Set 2 score': SGRNA_SCORE
              }

    def reassign_nontargeting(df):
        """Given non-targeting sgRNAs a gene ID of -1.
        """
        new_ids = []
        new_symbols = []
        for i, s in df[[GENE_ID, GENE_SYMBOL]].values:
            if s == 'Non-Targeting Control':
                new_ids.append(-1)
                new_symbols.append('nontargeting')
            else:
                new_ids.append(i)
                new_symbols.append(s)

        return df.assign(**{GENE_ID: new_ids, GENE_SYMBOL: new_symbols})

    df_brunello = (pd.read_csv(filename, sep='\t')
        .rename(columns=columns)
        .pipe(reassign_nontargeting)
        .pipe(ops.utils.cast_cols, int_cols=[GENE_ID])
        .assign(**{SGRNA_SCORE: lambda x: x[SGRNA_SCORE].fillna(0)})
        .assign(**{RANK: lambda x: 
            x.groupby(GENE_ID)[SGRNA_SCORE]
             .rank(ascending=False, method='first').astype(int)})
        [[GENE_ID, GENE_SYMBOL, RANK, SGRNA]]
        .sort_values([GENE_ID, RANK])
        )

    df_brunello.loc[df_brunello.gene_symbol=="AKAP2",'gene_id']=445815
    df_brunello.loc[df_brunello.gene_symbol=="PALM2",'gene_id']=445815
    df_brunello.loc[df_brunello.gene_symbol=="C10orf12",'gene_id']=84458
    df_brunello.loc[df_brunello.gene_symbol=="C10orf131",'gene_id']=387707
    df_brunello.loc[df_brunello.gene_symbol=="C16orf47",'gene_id']=463
    df_brunello.loc[df_brunello.gene_symbol=="C17orf47",'gene_id']=5414
    df_brunello.loc[df_brunello.gene_symbol=="C7orf76",'gene_id']=7979
    df_brunello.loc[df_brunello.gene_symbol=="MIA2",'gene_id']=4253
    df_brunello.loc[df_brunello.gene_symbol=="NARR",'gene_id']=83871
    df_brunello.loc[df_brunello.gene_symbol=="TMEM133",'gene_id']=143872
    df_brunello.loc[df_brunello.gene_symbol=="XAGE1B",'gene_id']=653067


    df_brunello = df_brunello.query('gene_symbol != ["C2orf48","TMEM257","TXNRD3NB"]').copy()

    return df_brunello

def import_brunello_dump(filename):
    df_brunello_dump = pd.read_csv(filename)

    df_brunello_dump.loc[df_brunello_dump.gene_symbol=="AKAP2",'gene_id']=445815
    df_brunello_dump.loc[df_brunello_dump.gene_symbol=="PALM2",'gene_id']=445815
    df_brunello_dump.loc[df_brunello_dump.gene_symbol=="C16orf47",'gene_id']=463
    df_brunello_dump.loc[df_brunello_dump.gene_symbol=="C17orf47",'gene_id']=5414

    df_brunello_dump = df_brunello_dump.query('gene_symbol != ["C2orf48","TMEM257","TXNRD3NB"]').copy()

    return df_brunello_dump


def import_tkov3(filename, df_ncbi):    
    columns = {'GENE': GENE_SYMBOL, 'SEQUENCE': SGRNA}
    symbols_to_ids = df_ncbi.set_index(GENE_SYMBOL)[GENE_ID]
    # symbols_to_ids.index = symbols_to_ids.index.str.upper()
    df_tkov3 = (pd.read_excel(filename)
      .rename(columns=columns)
      [[GENE_SYMBOL, SGRNA]]
      )

    df_tkov3 = df_tkov3.query('gene_symbol!=["C2orf48","TMEM257","TXNRD3NB"]').copy()

    df_tkov3.loc[df_tkov3.gene_symbol=="ADC",'gene_symbol']="AZIN2"
    df_tkov3.loc[df_tkov3.gene_symbol=="AGPAT9",'gene_symbol']="GPAT3"
    df_tkov3.loc[df_tkov3.gene_symbol=="AIM1",'gene_symbol']="CRYBG1"
    df_tkov3.loc[df_tkov3.gene_symbol=="B3GNT1",'gene_symbol']="B4GAT1"
    df_tkov3.loc[df_tkov3.gene_symbol=="C11orf48",'gene_symbol']="LBHD1"
    df_tkov3.loc[df_tkov3.gene_symbol=="C15orf38",'gene_symbol']="ARPIN"
    df_tkov3.loc[df_tkov3.gene_symbol=="C2ORF15",'gene_symbol']="C2orf15"
    df_tkov3.loc[df_tkov3.gene_symbol=="C2orf47",'gene_symbol']="MAIP1"
    df_tkov3.loc[df_tkov3.gene_symbol=="C6ORF165",'gene_symbol']="C6orf165"
    df_tkov3.loc[df_tkov3.gene_symbol=="C7orf55",'gene_symbol']="FMC1"
    df_tkov3.loc[df_tkov3.gene_symbol=="CD97",'gene_symbol']="ADGRE5"
    df_tkov3.loc[df_tkov3.gene_symbol=="CXXC11",'gene_symbol']="RTP5"
    df_tkov3.loc[df_tkov3.gene_symbol=="FLJ27365",'gene_symbol']="MIRLET7BHG"
    df_tkov3.loc[df_tkov3.gene_symbol=="GIF",'gene_symbol']="CBLIF"
    df_tkov3.loc[df_tkov3.gene_symbol=="HN1L",'gene_symbol']="JPT2"
    df_tkov3.loc[df_tkov3.gene_symbol=="HN1",'gene_symbol']="JPT1"
    df_tkov3.loc[df_tkov3.gene_symbol=="KIAA1045",'gene_symbol']="PHF24"
    df_tkov3.loc[df_tkov3.gene_symbol=="NAT6",'gene_symbol']="NAA80"
    df_tkov3.loc[df_tkov3.gene_symbol=="NOV",'gene_symbol']="CCN3"
    df_tkov3.loc[df_tkov3.gene_symbol=="STRA13",'gene_symbol']="CENPX"
    df_tkov3.loc[df_tkov3.gene_symbol=="ZHX1-C8ORF76",'gene_symbol']="ZHX1-C8orf76"
    df_tkov3.loc[df_tkov3.gene_symbol=="MUM1",'gene_symbol']="PWWP3A"
    df_tkov3.loc[df_tkov3.gene_symbol=='CSRP2BP','gene_symbol'] = 'KAT14'
    df_tkov3.loc[df_tkov3.gene_symbol=='C10orf2','gene_symbol'] = 'TWNK'
    df_tkov3.loc[df_tkov3.gene_symbol=='AZI1','gene_symbol'] = 'CEP131'
    df_tkov3.loc[df_tkov3.gene_symbol=='EFTUD1','gene_symbol'] = 'EFL1'
    # df_tkov3[GENE_SYMBOL]=df_tkov3[GENE_SYMBOL].str.upper()
    return (df_tkov3
     .join(symbols_to_ids, on=GENE_SYMBOL, how='left')
     .assign(**{RANK: lambda x: ops.utils.rank_by_order(x, GENE_ID)})
    )

def import_CRISPRfocus(filename,df_ncbi):
    df_CRISPRfocus = pd.read_csv(filename)

    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="ZHX1-C8ORF76",'gene_symbol']="ZHX1-C8orf76"
    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="TGIF2-C20ORF24",'gene_symbol']="TGIF2-C20orf24"
    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="RPL17-C18ORF32",'gene_symbol']="RPL17-C18orf32"
    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="ANXA8L1",'gene_id']=np.nan
    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="MUM1",'gene_symbol']="PWWP3A"
    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="NAT6",'gene_symbol']="NAA80"
    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="SLC35E2",'gene_symbol']="SLC35E2A"
    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="GIF",'gene_symbol']="CBLIF"
    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="NOV",'gene_symbol']="CCN3"


    # df_CRISPRfocus_id = df_CRISPRfocus.dropna(subset=['gene_id'])
    # df_CRISPRfocus_na = df_CRISPRfocus[df_CRISPRfocus.gene_id.isna()]
    symbols_to_ids = df_ncbi.set_index(GENE_SYMBOL)[GENE_ID]

    # symbols_to_ids.index = symbols_to_ids.index.str.upper()
    # df_CRISPRfocus_na = (df_CRISPRfocus_na
    #                      .drop(columns=['gene_id'])
    #                      .join(symbols_to_ids, on=GENE_SYMBOL, how='left')
    #                     )
    df_CRISPRfocus = (df_CRISPRfocus
                         .drop(columns=['gene_id'])
                         .join(symbols_to_ids, on=GENE_SYMBOL, how='left')
                        )

    df_CRISPRfocus= df_CRISPRfocus.query('gene_symbol != ["FAM231C","C2orf48","TMEM257","TXNRD3NB","FAM25B"]').copy()

    # return pd.concat([df_CRISPRfocus_id,df_CRISPRfocus_na],sort=True)[['gene_symbol','gene_id','sgRNA','rank']]
    return df_CRISPRfocus[['gene_symbol','gene_id','sgRNA','rank']]

def import_wang2017(filename,df_ncbi):
    df_wang2017 = (pd.read_excel(filename)
                   .rename(columns={'sgRNA ID':'sgRNA_ID','sgRNA location':'sgRNA_location',
                    'Genomic strand targeted':'Genomic_strand','sgRNA sequence':'sgRNA',
                    'Other genes hits':'Other_gene_hits','Symbol':'gene_symbol'})
                  )

    def group_controls(s):
      if s.sgRNA_ID.startswith('CTRL'):
          s.gene_symbol = 'nontargeting'
      elif 'INTERGENIC' in s.sgRNA_ID:
          s.gene_symbol = 'INTERGENIC'
      return s

    df_wang2017 = df_wang2017.apply(lambda x: group_controls(x),axis=1)
    df_wang2017 = df_wang2017.query('gene_symbol != "INTERGENIC"').copy()

    df_wang2017.loc[df_wang2017.gene_symbol=="NOV",'gene_symbol']="CCN3"
    df_wang2017.loc[df_wang2017.gene_symbol=="GIF",'gene_symbol']="CBLIF"
    df_wang2017.loc[df_wang2017.gene_symbol=="B3GNT1",'gene_symbol']="B4GAT1"
    df_wang2017.loc[df_wang2017.gene_symbol=="C7orf55",'gene_symbol']="FMC1"
    df_wang2017.loc[df_wang2017.gene_symbol=="CXXC11",'gene_symbol']="RTP5"
    df_wang2017.loc[df_wang2017.gene_symbol=="AGPAT9",'gene_symbol']="GPAT3"
    df_wang2017.loc[df_wang2017.gene_symbol=="ZHX1-C8ORF76",'gene_symbol']="ZHX1-C8orf76"
    df_wang2017.loc[df_wang2017.gene_symbol=="AIM1",'gene_symbol']="CRYBG1"
    df_wang2017.loc[df_wang2017.gene_symbol=="NAT6",'gene_symbol']="NAA80"
    df_wang2017.loc[df_wang2017.gene_symbol=="CD97",'gene_symbol']="ADGRE5"
    df_wang2017.loc[df_wang2017.gene_symbol=="C15orf38",'gene_symbol']="ARPIN"
    df_wang2017.loc[df_wang2017.gene_symbol=="C2orf47",'gene_symbol']="MAIP1"
    df_wang2017.loc[df_wang2017.gene_symbol=="STRA13",'gene_symbol']="CENPX"
    df_wang2017.loc[df_wang2017.gene_symbol=="C11orf48",'gene_symbol']="LBHD1"
    df_wang2017.loc[df_wang2017.gene_symbol=="MUM1",'gene_symbol']="PWWP3A"
    df_wang2017.loc[df_wang2017.gene_symbol=="HN1L",'gene_symbol']="JPT2"
    df_wang2017.loc[df_wang2017.gene_symbol=="HN1",'gene_symbol']="JPT1"
    df_wang2017.loc[df_wang2017.gene_symbol=="ADC",'gene_symbol']="AZIN2"
    df_wang2017.loc[df_wang2017.gene_symbol=="TRIM49D2P",'gene_symbol']="TRIM49D2"
    df_wang2017.loc[df_wang2017.gene_symbol=="FAM21A",'gene_symbol']="WASHC2A"
    df_wang2017.loc[df_wang2017.gene_symbol=="SLC35E2",'gene_symbol']="SLC35E2A"
    df_wang2017.loc[df_wang2017.gene_symbol=="APITD1",'gene_symbol']="CENPS"
    df_wang2017.loc[df_wang2017.gene_symbol=="LIMS3L",'gene_symbol']="LIMS4"
    df_wang2017.loc[df_wang2017.gene_symbol=='CSRP2BP','gene_symbol'] = 'KAT14'
    df_wang2017.loc[df_wang2017.gene_symbol=='AZI1','gene_symbol'] = 'CEP131'
    df_wang2017.loc[df_wang2017.gene_symbol=='TCEB3C','gene_symbol'] = 'ELOA3'
    df_wang2017.loc[df_wang2017.gene_symbol=='TCEB3CL','gene_symbol'] = 'ELOA3B'
    df_wang2017.loc[df_wang2017.gene_symbol=='EFTUD1','gene_symbol'] = 'EFL1'
    df_wang2017.loc[df_wang2017.gene_symbol=='CGB','gene_symbol'] = 'CGB3'
    df_wang2017.loc[df_wang2017.gene_symbol=='C10orf2','gene_symbol'] = 'TWNK'

    df_wang2017 = df_wang2017.query(('gene_symbol != '
                                     '["CT45A4","SMCR9","PRAMEF3",'
                                     '"SPANXE","PRAMEF16","C2orf48",'
                                     '"TMEM257","TXNRD3NB","FOXD4L2","FAM25B"]'
                                    )
                                   ).copy()

    symbols_to_ids = df_ncbi.set_index('gene_symbol')['gene_id']
    # symbols_to_ids.index = symbols_to_ids.index.str.upper()
    # df_wang2017['gene_symbol']=df_wang2017['gene_symbol'].str.upper()

    df_wang2017 = (df_wang2017
                   .join(symbols_to_ids, on=['gene_symbol'], how='left')
                   .assign(**{RANK: lambda x: ops.utils.rank_by_order(x, 'gene_symbol')})
                  )

    def LOC_to_ID(s):
      if s.gene_symbol.startswith('LOC') & np.isnan(s.gene_id):
          s.gene_id = s.gene_symbol[3:]
      return s

    df_wang2017 = df_wang2017.apply(lambda x: LOC_to_ID(x),axis=1)
    df_wang2017.loc[df_wang2017.gene_symbol=="CRIPAK",'gene_id'] = 285464
    df_wang2017.loc[df_wang2017.gene_symbol=="FAM231A",'gene_id'] = 729574
    df_wang2017.loc[df_wang2017.gene_symbol=="KIAA1804",'gene_id'] = 84451
    df_wang2017.loc[df_wang2017.gene_symbol=="KIAA1045",'gene_id'] = 23349

    df_wang2017.loc[df_wang2017.gene_symbol == "nontargeting",'gene_id']=-1

    return df_wang2017[['gene_symbol','gene_id','sgRNA','rank']]


def import_hugo_ncbi(filename):
    columns = {'Approved symbol': GENE_SYMBOL,
               'NCBI Gene ID(supplied by NCBI)': GENE_ID}
    return (pd.read_csv(filename, comment='#', sep='\t')
         .rename(columns=columns)
         .dropna()
         .pipe(ops.utils.cast_cols, int_cols=[GENE_ID]))


def import_ncbi_synonyms(filename):
    return (pd.read_csv(filename,index_col=[0])
         .pipe(ops.utils.cast_cols, int_cols=[GENE_ID]))

def import_dialout_primers(filename):
    """Returns an array of (forward, reverse) primer pairs.
    """
    return pd.read_csv('kosuri_dialout_primers.csv').values