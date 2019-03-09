import csv
import requests
import os

BASE_PATH = 'mozilla_firefox_v2/firefoxDataset/docs_english/taskruns/'

FIELDS_TASKRUNS = ['bug_id',
                    'user_id',
                    'task_id',
                    'created',
                    'finish_time',
                    'user_ip',
                    'link',
                    'timeout',
                    'project_id',
                    'id',
                    'answers']

FIELDS_ANSWER = ['new_awesome_bar',
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

def get_taskruns(project_id, project_shortname, first_task_id, last_task_id, output_file_path):
    taskruns_url = "http://localhost:8081/api/taskrun?project_id={0}".format(project_id)

    range_task_ids = range(first_task_id, last_task_id+1, 1)   

    csv_line_taskruns = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}" + "\n"

    header_taskruns = csv_line_taskruns.format(*FIELDS_TASKRUNS)

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    with open(output_file_path, 'w') as taskruns_file:
        taskruns_file.write(header_taskruns)

    with open(output_file_path, 'a') as taskruns_file:
        for task_id in range_task_ids:
            taskruns = requests.get(taskruns_url + "&task_id={0}".format(task_id))
            taskruns_json = taskruns.json()
            print("Task Id: " + str(task_id))

            for tr in taskruns_json:
                answers = [tr['info']['links'][f_ans] for f_ans in FIELDS_ANSWER]
                answers = " ".join([str(ans) for ans in answers])
                taskruns_file.write(csv_line_taskruns.format(
                                        tr['info']['bug_id'],
                                        tr['user_id'],
                                        tr['task_id'],
                                        tr['created'],
                                        tr['finish_time'],
                                        tr['user_ip'],
                                        tr['link'],
                                        tr['timeout'],
                                        tr['project_id'],
                                        tr['id'],
                                        answers)
                                )

LIST_APPS = [
        (2,  'br_feature_expert_app',    1721, 1813, BASE_PATH+'taskruns_expert.csv'),
        (10, 'br_feature_expert_app_2',  1928, 2020, BASE_PATH+'taskruns_expert_2.csv'),
        (7,  'br_feature_volunteer_app', 1628, 1720, BASE_PATH+'taskruns_volunteers.csv'),
        (9,  'br_feature_aux_app',       1918, 1927, BASE_PATH+'taskruns_volunteers_2.csv'),
        (11, 'br_feature_aux_app_2',     2114, 2206, BASE_PATH+'taskruns_aux_2.csv')
    ]

if __name__ == '__main__':
    for app in LIST_APPS:
        print('Init App: {}\n'.format(app[1]))
        get_taskruns(*app)
        print('Finish App: {}\n'.format(app[1]))
