# script to create tasks relative to days 01/06/2016 and 02/06/2016 needed for analysis

import pandas as pd
import json
import os

# read bug_reports_final dataset
bugreports_final = pd.read_csv('~/anaconda3/envs/trace-link-recovery-study/data/mozilla_firefox_v2/firefoxDataset/docs_english/BR/bugreports_final.csv')
print(bugreports_final.shape)

brs_ids = [ '882753',  '945665', '1127927', '1154922', '1223550', '1265967', '1266270',
       '1271395', '1271766', '1271774', '1274459', '1274712', '1276070', '1276152',
       '1276447', '1276656', '1276818', '1276884', '1276966', '1277114', '1277151',
       '1277257']
brs_versions = ['48 Branch', '49 Branch', '50 Branch', '51 Branch']
brs_status = ['RESOLVED','VERIFIED']
brs_priority = ['P1', 'P2', 'P3']
brs_resolutions = ['FIXED']
brs_severities = ['major', 'normal', 'blocker', 'critical']
brs_isconfirmed = [True]
selected_bugs = bugreports_final[(bugreports_final.Bug_Number.isin(brs_ids)) &
                                 (bugreports_final.Version.isin(brs_versions)) &
                                 (bugreports_final.Status.isin(brs_status)) &
                                 (bugreports_final.Priority.isin(brs_priority)) &
                                 (bugreports_final.Resolution.isin(brs_resolutions)) &
                                 (bugreports_final.Severity.isin(brs_severities)) &
                                 (bugreports_final.Is_Confirmed.isin(brs_isconfirmed))
                                ]
print(selected_bugs.shape)



TASKS_FILE_PATH = 'mozilla_firefox_v2/firefoxDataset/br_feat_recovery_empirical_study/pybossa-apps/tasks/tasks_2.json'

#if os.path.exists(TASKS_FILE_PATH):
#    os.remove(TASKS_FILE_PATH)
#    print("Tasks File Removed")

def get_feats():
    feats = []
    features = pd.read_csv('~/anaconda3/envs/trace-link-recovery-study/data/mozilla_firefox_v2/firefoxDataset/docs_english/Features/features.csv')
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
