#!/bin/bash

#source activate trace-link-recovery-study
#~/anaconda3/envs/trace-link-recovery-study/bin/python ~/anaconda3/envs/trace-link-recovery-study/data/mozilla_firefox_v2/firefoxDataset/br_feat_recovery_empirical_study/pybossa-apps/tasks/create_tasks.py 

pbs delete_tasks
pbs add_tasks --tasks-file ~/anaconda3/envs/trace-link-recovery-study/data/mozilla_firefox_v2/firefoxDataset/br_feat_recovery_empirical_study/pybossa-apps/tasks/tasks.json --redundancy 1