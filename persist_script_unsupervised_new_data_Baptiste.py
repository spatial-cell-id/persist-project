import anndata as ad
import scanpy as sc

import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import torch
from scipy.sparse import csr_matrix

import sys

from persist import PERSIST, ExpressionDataset


sc.settings.verbosity = 0

#Filtering genes and cells
data_folder=sys.argv[1]
saving_folder=sys.argv[2]

adata_11_1 = sc.read_10x_mtx('%s/filtered_feature_bc_matrix_111' %data_folder, var_names = 'gene_symbols', cache = True)
adata_6_3 = sc.read_10x_mtx('%s/filtered_feature_bc_matrix_63' %data_folder, var_names = 'gene_symbols', cache = True)
adata_8_1 = sc.read_10x_mtx('%s/filtered_feature_bc_matrix_81' %data_folder, var_names = 'gene_symbols', cache = True)

adata_6_3.obs['sample'] = '6_3'
adata_8_1.obs['sample'] = '8_1'
adata_11_1.obs['sample'] = '11_1'

adata_6_3.var_names_make_unique()
adata_8_1.var_names_make_unique()
adata_11_1.var_names_make_unique()

adata = adata_6_3.concatenate(adata_8_1, adata_11_1)
print('Loaded and concatenated the datasets...')


#Extract a matrix
gene_symbols = [i for i in adata.var['gene_ids'].values]
gene_ids = [i for i in adata.var.index]

cells = [i for i  in adata.obs.index]

expr_df = pd.DataFrame.sparse.from_spmatrix(adata.X, columns = gene_symbols, index = cells).transpose()

adata.X = csr_matrix(adata.X)

#Data normalization
adata.layers['log1pcpm'] = sc.pp.normalize_total(adata, target_sum=1e6, inplace=False)['X']
sc.pp.log1p(adata, layer='log1pcpm')

sc.pp.highly_variable_genes(adata, layer='log1pcpm', n_top_genes=10000, inplace=True)

sc.pp.filter_cells(adata, min_genes=200) 
sc.pp.filter_genes(adata, min_cells=3)

print('preprocessed the single cell data...')

# restrict to 10k highly variable genes
adata = adata[:,adata.var['highly_variable']]
adata.layers['bin'] = (adata.X>0).astype(np.float32)

print('start training...')

train_ind, val_ind = sk.model_selection.train_test_split(np.arange(adata.shape[0]), train_size=0.8)

print(f'{adata.shape[0]} total samples')
print(f'{np.size(train_ind)} in training set')
print(f'{np.size(val_ind)} in validation set')

adata_train = adata[train_ind,:]
adata_val = adata[val_ind,:]

train_dataset = ExpressionDataset(adata_train.layers['bin'], adata_train.layers['log1pcpm'])
val_dataset = ExpressionDataset(adata_val.layers['bin'], adata_val.layers['log1pcpm'])


# Use GPU device if available -- we highly recommend using a GPU!
device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')

print(' ')
print(' ')
print(' ')
print('Device: %s' %device)
print(' ')
print(' ')
print(' ')


# Number of genes to select within the current selection process.
#num_genes = (128, 256, 512)
num_genes = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000,     3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000)

persist_results = {}

# Set up the PERSIST selector
selector = PERSIST(train_dataset,
                   val_dataset,
                   loss_fn=torch.nn.CrossEntropyLoss(),
                   device=device)

# Coarse removal of genes
print('Starting initial elimination...')
candidates, model = selector.eliminate(target=2000, max_nepochs=1000, verbose=False)
print('Completed initial elimination.')

print('Selecting specific number of genes...')
for num in num_genes:
    inds, model = selector.select(num_genes=num, max_nepochs=2000, verbose=False)
    persist_results[num] = inds
print('Done')

df = adata.var.copy()

# set a boolean = True for genes selected in any of the rounds
for num in num_genes:
    df[f'persist_set_{num}'] = False
    ind = df.iloc[persist_results[num]].index
    df.loc[ind,f'persist_set_{num}'] = True

df = df[df[[f'persist_set_{num}' for num in num_genes]].any(axis=1)]

df['gene_names'] = df.index

temp_cols=df.columns.tolist()
new_cols=temp_cols[-1:] + temp_cols[:-1]
df=df[new_cols]

#df.to_csv("%s/persist_unsupervised_results_new_data_Baptiste.tsv" %saving_folder, sep = "\t", index = None)
df.to_csv("%s/persist_unsupervised_results_new_data_Baptiste_different_panels.tsv" %saving_folder, sep = "\t", index = None)


##Example of running without a job:
##python persist_script_unsupervised_new_data_Baptiste.py /scratch/ssarnata/persist_project/sc_data_baptiste /scratch/ssarnata/persist_project/sc_data_baptiste/analysis_output/
