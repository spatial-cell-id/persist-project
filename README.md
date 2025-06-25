The scope of the repository is to provide instructions on how to use PERSIST (https://www.nature.com/articles/s41467-023-37392-1) on the machines at CBP.

### Prepare your computational environment
First of all, you need a CBP account. To do that, go here: https://www.cbp.ens-lyon.fr/doku.php?id=contact:compte

Once you get a CBP account, you need to connect to one of the machines which have GPU, since PERSIST requires a GPU for running properly (you can use the CPU for very small datasets, though).
The list of the available machine at CBP is here: https://www.cbp.ens-lyon.fr/python/forms/CloudCBP

You need to create a conda environment with _Python 3.11_.
Then, download and install persist, as described on the official repo of the tool https://github.com/iancovert/persist.

At this point, all the dependencies should be downloaded as well.

### Examples

You should now be able to run your _persist_.
An example is provided in scripts/persist_script_unsupervised_new_data_Baptiste.py

Moreover, scripts/persist_results_comparison.ipynb shows some metrics that could be used to compare the different gene panels obtained by PERSIST.
In particular: 

1 - Procrustes disparity: compares UMAP from panel vs. full data after alignment; lower means better global structure preservation.

2 - KNN preservation: measures overlap of nearest neighbors per cell between full and panel data; higher means better local structure retention.

3 - PCA distance correlation: computes correlation between pairwise cell distances in PCA space; higher means better global geometry retention.

### How to run the script:
python persist_script_unsupervised_new_data_Baptiste.py /path/to/raw/data /path/to/analysis/output