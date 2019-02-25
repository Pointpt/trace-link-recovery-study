import csv
import requests
import os

PYB_PROJECT_ID = 7

taskruns_url = "http://localhost:8081/api/taskrun?project_id={0}".format(PYB_PROJECT_ID)

FIRST_TASK_ID = 1628
LAST_TASK_ID = 1720
RANGE_TASK_IDS = range(FIRST_TASK_ID, LAST_TASK_ID+1, 1)

FIELDS_TASKRUNS = ['bug_id',
    'user_id',
    'task_id',
    'created',
    'finish_time',
    'user_ip',
    'link',
    'timeout',
    'project_id',
    'id']


csv_line_taskruns = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}" + "\n"

header_taskruns = csv_line_taskruns.format(*FIELDS_TASKRUNS)

# file with general info about each task
TASKRUNS_FILE_PATH = 'mozilla_firefox_v2/firefoxDataset/br_feat_recovery_empirical_study/pybossa-apps/recover_taskruns/taskruns.csv'

if os.path.exists(TASKRUNS_FILE_PATH):
    os.remove(TASKRUNS_FILE_PATH)

with open(TASKRUNS_FILE_PATH, 'w') as taskruns_file:
    taskruns_file.write(header_taskruns)

with open(TASKRUNS_FILE_PATH, 'a') as taskruns_file:
    for task_id in RANGE_TASK_IDS:
        taskruns = requests.get(taskruns_url + "&task_id={0}".format(task_id))
        taskruns_json = taskruns.json()
        print("Task Id: " + str(task_id))

        for tr in taskruns_json:
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
                                        tr['id'])
                                )
                                        