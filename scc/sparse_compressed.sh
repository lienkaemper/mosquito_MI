#!/bin/bash -l
 
module load miniconda/23.11.0

conda activate mosquito
# program name or command and its options and arguments
python sparse_compressed_sim.py
python sparse_compressed_analysis.py
python plot_sparse_compressed.py
