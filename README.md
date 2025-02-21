# MosquitoOlfactionMutualInformation

Code for the manuscript "When non-canonical olfaction is optimal", by Caitlin Lienkaemper, Meg A. Younger, and Gabriel Koch Ocker. 

To install project package, run:

```bash
conda env create --name mosquito_MI --file environment.yml
conda activate mosquito_MI
pip install -e .
```

To make Figure 1, panels D and E, run:

```bash
cd scripts 
python two_receptor_basic.py
```

To make Figure 2, panels B and C, run:

```bash
cd scripts 
python two_receptor_unequal_variance.py
```


To make Figure 3, run 

```bash
cd scripts 
python sensing_matrix.py
```

To make Figure 4, run 

```bash
cd scc
python non_gaussian_fig.py
```

Note: this is slow! 

To plot Figure 5 from pre-existing simulations, run 


```bash
cd scripts 
python large_linear_network_plot.py
```

To re-run mutual information optimization (~24 hours), run 

```bash
cd scripts 
python large_linear_network_sim.py
python large_linear_network_analysis.py
python large_linear_network_plot.py
```

To plot Figure 6, panel B from pre-computed simulation results , run 

```bash
cd scc
python plot_sparse_compressed.py
```


To re-run network training, run 

```bash
cd scc
python sparse_compressed_analysis.py
python sparse_compressed_sim.py
python plot_sparse_compressed.py
```
To make Figure 7, run

```bash
cd scc
python natural_odor_cors.py
```

To make Figure 8 panels B, C, D, E, F, G, run 

```bash
cd scc
python perceptual_subspaces.py
```