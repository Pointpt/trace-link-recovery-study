import csv
import requests
import os

PYB_PROJECT_ID = 7

taskruns_url = "http://localhost:8081/api/taskrun?project_id={0}".format(PYB_PROJECT_ID)

FIRST_TASK_ID = 1628
LAST_TASK_ID = 1720
RANGE_TASK_IDS = range(FIRST_TASK_ID, LAST_TASK_ID+1, 1)

FIELDS = ['bug_number',
            'new_awesome_bar',
            'windows_child_mode',
            'apz_async_scrolling',
            'browser_customization',
            'pdf_viewer',
            'context_menu',
            'w10_comp', 
            'tts_in_desktop', 
            'tts_in_rm', 
            'webgl_comp',
            'video_and_canvas_render', 
            'pointer_lock_api',
            'webm_eme', 
            'zoom_indicator',
            'downloads_dropmaker',
            'webgl2', 
            'flac_support', 
            'indicator_device_perm',
            'flash_support',  
            'notificationbox',          
            'update_directory']

csv_line = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21}" + "\n"

header = csv_line.format(*FIELDS)

# file with the answers for each task
BR_2_FEATURE_FILE_PATH = 'mozilla_firefox_v2/firefoxDataset/br_feat_recovery_empirical_study/pybossa-apps/recover_taskruns/br_2_feature_matrix_volunteers.csv'

if os.path.exists(BR_2_FEATURE_FILE_PATH):
    os.remove(BR_2_FEATURE_FILE_PATH)

with open(BR_2_FEATURE_FILE_PATH, 'w') as br_2_feature_file:
    br_2_feature_file.write(header)


with open(BR_2_FEATURE_FILE_PATH, 'a') as br_2_feature_file:
    for task_id in RANGE_TASK_IDS:
        taskruns = requests.get(taskruns_url + "&task_id={0}".format(task_id))
        taskruns_json = taskruns.json()
        print("Task Id: " + str(task_id))

        for tr in taskruns_json:
            br_2_feature_file.write(csv_line.format(
                                        tr['info']['bug_id'],
                                        tr['info']['links']['new_awesome_bar'],
                                        tr['info']['links']['windows_child_mode'],
                                        tr['info']['links']['apz_async_scrolling'],
                                        tr['info']['links']['browser_customization'],
                                        tr['info']['links']['pdf_viewer'],
                                        tr['info']['links']['context_menu'],
                                        tr['info']['links']['w10_comp'],
                                        tr["info"]['links']['tts_in_desktop'],
                                        tr['info']['links']['tts_in_rm'],
                                        tr['info']['links']['webgl_comp'],
                                        tr['info']['links']['video_and_canvas_render'],
                                        tr["info"]['links']['pointer_lock_api'],
                                        tr['info']['links']['webm_eme'],
                                        tr['info']['links']['zoom_indicator'],
                                        tr['info']['links']['downloads_dropmaker'],
                                        tr["info"]['links']['webgl2'],
                                        tr["info"]['links']['flac_support'],
                                        tr["info"]['links']['indicator_device_perm'],
                                        tr["info"]['links']['flash_support'],
                                        tr["info"]['links']['notificationbox'],
                                        tr["info"]['links']['update_directory'])
                                )                                      