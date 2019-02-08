import requests
import os

INIT_DATE = "2016-06-06"
END_DATE = "2018-12-31"
PRODUCT = "firefox"
QUERY_FORMAT = "advanced"
QUERY_BASED_ON = ""
J_TOP = "OR"
LIMIT = 500

BASE_URL = "https://bugzilla.mozilla.org/rest/bug?chfieldfrom={}&chfieldto={}&product={}&query_format={}&limit={}&query_based_on={}&j_top={}&offset={}"
BASE_URL_COMMENT = "https://bugzilla.mozilla.org/rest/bug/"
BUGS_FILE_PATH = '/home/guilherme/anaconda3/envs/trace-link-recovery-study/data/mozilla_firefox_v2/firefoxDataset/docs_english/BR/all_bugs_add_info.csv'

header = "Bug_Number|Status|Product|Priority|Resolution|Severity|Is_Confirmed\n"
line = "{0}|{1}|{2}|{3}|{4}|{5}|{6}\n"

args = [INIT_DATE, END_DATE, PRODUCT, QUERY_FORMAT, LIMIT, QUERY_BASED_ON, J_TOP]

if not os.path.exists(BUGS_FILE_PATH):
    f = open(BUGS_FILE_PATH, 'w+')
    f.write(header)
    f.close()
    
for offset_ in range(0, 37750, LIMIT): # 0, 7000, 1000
    if len(args) < 8:
        args = args + [offset_]
    else:
        args[7] = offset_

    bugs = requests.get(BASE_URL.format(*args))
    bugs_json = bugs.json()

    print('OFFSET: {}'.format(offset_))
    
    for bug in bugs_json['bugs']:
        bug_id = bug['id']
                
        print("Bug Id: " + str(bug_id))

        with open(BUGS_FILE_PATH, 'a+') as bug_file:
            bug_file.write(line.format(
                bug['id'],
                bug['status'],
                bug['product'],
                bug['priority'],
                bug['resolution'],
                bug['severity'],
                bug['is_confirmed']
            )
        )
        
        