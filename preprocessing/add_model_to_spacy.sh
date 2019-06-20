#!/bin/bash

cd ~/anaconda3/envs/trace-link-recovery-study/data/mozilla_firefox_v2/firefoxDataset/wv_embeddings/

conda activate trace-link-recovery-study

gzip cust_wv_model.txt

python -m spacy init-model en ./cust_wv_model --vectors-loc cust_wv_model.txt.gz

