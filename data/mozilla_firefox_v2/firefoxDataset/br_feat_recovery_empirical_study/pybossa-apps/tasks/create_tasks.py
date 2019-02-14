import pandas as pd
import json
import os

# read bug_reports_final dataset
bugreports_final = pd.read_csv('~/anaconda3/envs/trace-link-recovery-study/data/mozilla_firefox_v2/firefoxDataset/docs_english/BR/bugreports_final.csv')
print(bugreports_final.shape)

brs_versions = ['48 Branch', '49 Branch', '50 Branch', '51 Branch']
brs_status = ['NEW','RESOLVED','VERIFIED','REOPENED','ASSIGNED']
brs_priority = ['P1', 'P2', 'P3']
brs_resolutions = ['FIXED']
brs_severities = ['major', 'normal', 'blocker', 'critical','enhancement']
brs_isconfirmed = [True]
selected_bugs = bugreports_final[(bugreports_final.Version.isin(brs_versions)) &
                                 (bugreports_final.Status.isin(brs_status)) &
                                 (bugreports_final.Priority.isin(brs_priority)) &
                                 (bugreports_final.Resolution.isin(brs_resolutions)) &
                                 (bugreports_final.Severity.isin(brs_severities)) &
                                 (bugreports_final.Is_Confirmed.isin(brs_isconfirmed))
                                ]
print(selected_bugs.shape)



TASKS_FILE_PATH = 'mozilla_firefox_v2/firefoxDataset/br_feat_recovery_empirical_study/pybossa-apps/tasks/tasks.json'

if os.path.exists(TASKS_FILE_PATH):
    os.remove(TASKS_FILE_PATH)
    print("Tasks File Removed")

FIREFOX_FEATS = ['new_awesome_bar',
                 'windows_child_mode',
                 'apz_async_scrolling',
                 'browser_customization',
                 'pdf_viewer',
                 'context_menu',
                 'w10_comp',
                 'tts_in_desktop',
                 'tts_in_rm', 'webgl_comp', 'video_and_canvas_render',
                 'pointer_lock_api', 'webm_eme', 'zoom_indicator',
                 'downloads_dropmaker', 'webgl2', 'flac_support',
                 'indicator_device_perm', 'flash_support',  
                 'notificationbox',
                 'update_directory']

def get_feats():
    feats = []
    features = pd.read_csv('/home/guilherme/anaconda3/envs/trace-link-recovery-study/data/mozilla_firefox_v2/firefoxDataset/docs_english/Features/features.csv')
    features.fillna("", inplace=True)
    for idx, f in features.iterrows():
        feats.append({
            "feature_id" : FIREFOX_FEATS[idx],
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
