#!/bin/bash -l
 
module load miniconda/23.11.0

conda activate mosquito
# program name or command and its options and arguments
python non_gaussian_fig.py

