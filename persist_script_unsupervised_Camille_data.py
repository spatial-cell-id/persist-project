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
sample_name=sys.argv[2]
saving_folder=sys.argv[3]


#adata = sc.read_h5ad('%s/adata_%s.h5ad' %(data_folder, sample_name), var_names = 'gene_symbols', cache = True)
adata = sc.read_h5ad('%s/adata_%s.h5ad' %(data_folder, sample_name))


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
#num_genes = (256, 512)
#num_genes = [512]
num_genes = [128, 256, 512]
persist_results = {}

# Set up the PERSIST selector
selector = PERSIST(train_dataset,
                   val_dataset,
                   loss_fn=torch.nn.CrossEntropyLoss(),
                   device=device)

# Coarse removal of genes
print('Starting initial elimination...')
candidates, model = selector.eliminate(target=2000, max_nepochs=10000, verbose=False)
print('Completed initial elimination.')

print('Selecting specific number of genes...')
for num in num_genes:
    inds, model = selector.select(num_genes=num, max_nepochs=10000, verbose=False)
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
#df.to_csv("%s/persist_unsupervised_results_camille_data_%s.tsv" %(saving_folder, sample_name), sep = "\t", index = None)
df.to_csv("%s/persist_unsupervised_results_camille_data_%s_%i_genes.tsv" %(saving_folder, sample_name, num_genes[0]), sep = "\t", index = None)


##Example of running without a job:
##python persist_script_unsupervised_Camille_data.py sc_data_camille/data_for_PERSIST 9693 /scratch/ssarnata/persist_project/sc_data_camille/analysis_output
##python persist_script_unsupervised_Camille_data.py /scratch/ssarnata/persist_project/sc_data_camille/data_for_PERSIST WP /scratch/ssarnata/persist_project/sc_data_camille/analysis_output























