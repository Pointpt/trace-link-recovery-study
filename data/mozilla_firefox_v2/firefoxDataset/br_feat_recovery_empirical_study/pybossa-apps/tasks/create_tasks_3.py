# create tasks relative to remaining jobs for last participant of empirical study

import pandas as pd
import json
import os

# read bug_reports_final dataset
bugreports_final = pd.read_csv('~/anaconda3/envs/trace-link-recovery-study/data/mozilla_firefox_v2/firefoxDataset/docs_english/BR/bugreports_final.csv')
print(bugreports_final.shape)

brs_versions = ['48 Branch', '49 Branch', '50 Branch', '51 Branch']
brs_status = ['RESOLVED','VERIFIED']
brs_priority = ['P1', 'P2', 'P3']
brs_resolutions = ['FIXED']
brs_severities = ['major', 'normal', 'blocker', 'critical']
brs_isconfirmed = [True]
selected_bugs = bugreports_final[(bugreports_final.Version.isin(brs_versions)) &
                                 (bugreports_final.Status.isin(brs_status)) &
                                 (bugreports_final.Priority.isin(brs_priority)) &
                                 (bugreports_final.Resolution.isin(brs_resolutions)) &
                                 (bugreports_final.Severity.isin(brs_severities)) &
                                 (bugreports_final.Is_Confirmed.isin(brs_isconfirmed))
                                ]
print(selected_bugs.shape)

remaining_bugs = [1315514, 1316126, 1335538, 1357458, 1365887, 1408361, 1430603, 1432915, 1449700, 1451475]
selected_bugs = selected_bugs[selected_bugs.Bug_Number.isin(remaining_bugs)]

TASKS_FILE_PATH = 'mozilla_firefox_v2/firefoxDataset/br_feat_recovery_empirical_study/pybossa-apps/tasks/tasks_3.json'

if os.path.exists(TASKS_FILE_PATH):
    os.remove(TASKS_FILE_PATH)
    print("Tasks File Removed")

def get_feats():
    feats = []
    features = pd.read_csv('/home/guilherme/anaconda3/envs/trace-link-recovery-study/data/mozilla_firefox_v2/firefoxDataset/docs_english/Features/features.csv')
    features.fillna("", inplace=True)
    for idx, f in features.iterrows():
        feats.append({
            "feature_id" : f['Feature_Shortname'],
            "feature_name": f['Firefox_Feature'],
            "feature_description" : f['Feature_Description'],
            "feature_reference": f['Reference']
        })
    return feats

BUGZILLA_URL = "https://bugzilla.mozilla.org/show_bug.cgi?id={}"

with open(TASKS_FILE_PATH, 'a+') as tasks_file:
    tasks = []
    for idx,bug in selected_bugs.iterrows():
        print("Bug ID: {}".format(bug['Bug_Number']))
        tasks.append(
            {
                "bug_id" : bug['Bug_Number'],
                "bug_summary": bug['Summary'],
                "bug_first_comment": bug['First_Comment_Text'],
                "bug_link": BUGZILLA_URL.format(bug['Bug_Number']),
                "bug_f_version" : bug['Version'],
                "features" : get_feats()
            })
    
    json.dump(tasks, tasks_file)

print("Finished Tasks Creation")
