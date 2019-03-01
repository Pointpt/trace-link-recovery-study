# some tasks need to be rerunned so the amount of solved tasks be approximatelly the same for every user

import requests
import json

API_KEY = "64b8fd71-dc52-4e20-adb2-4e4b48bd383d"
PROJECT_ID = 7  # volunteers app
TASK_IDS = [1639,1640,1641,1642,1654,1655,1656,1657,1668,1669,1681,1682,1683,1684,1685,1686,1687,1688,1689,1690,1691,1692,1703,1714,1715,1716,1717,1718,1719,1720]

URL = "http://localhost:8081/api/task/{}?project_id={}&api_key={}"

headers = {"Content-Type" : "application/json"}

for task_id in TASK_IDS:
    r = requests.put(URL.format(task_id, PROJECT_ID, API_KEY), data=json.dumps({'n_answers':'2'}), headers=headers)
    print('TaskId:{} - Reponse: {}'.format(task_id, r.status_code))
