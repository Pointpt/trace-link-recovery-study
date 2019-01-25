import requests
import sqlite3
import os

INIT_DATE = "2016-06-06"
END_DATE = "2016-12-31"
PRODUCT = "firefox"
QUERY_FORMAT = "advanced"
QUERY_BASED_ON = ""
J_TOP = "OR"
LIMIT = 50

BASE_URL = "https://bugzilla.mozilla.org/rest/bug?chfieldfrom={}&chfieldto={}&product={}&query_format={}&limit={}&query_based_on={}&j_top={}&offset={}"
BASE_URL_COMMENT = "https://bugzilla.mozilla.org/rest/bug/"
BUGS_FILE_PATH = '/home/guilherme/anaconda3/envs/trace-link-recovery-study/data/mozilla_firefox/firefoxDataset/docs_english/BR_2/all_bugs.csv'

header = "Bug_Number|Summary|Platform|Component|Creation_Time|Whiteboard|QA_Whiteboard|First_Comment_Text|First_Comment_Creation_Time\n"
line = "{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}\n"

args = [INIT_DATE, END_DATE, PRODUCT, QUERY_FORMAT, LIMIT, QUERY_BASED_ON, J_TOP]

#conn = sqlite3.connect('data.db')
#cursor = conn.cursor()

#if os.path.exists(BUGS_FILE_PATH):
#    f = open(BUGS_FILE_PATH, 'w+')
#    f.write(header)
#    f.close()
    
for offset_ in range(2100, 7000, LIMIT): # 0, 7000, 1000
    if len(args) < 8:
        args = args + [offset_]
    else:
        args[7] = offset_

    bugs = requests.get(BASE_URL.format(*args))
    bugs_json = bugs.json()

    print('OFFSET: {}'.format(offset_))
    
    for bug in bugs_json['bugs']:
        bug_id = bug['id']
                
        bug_first_comment = requests.get(BASE_URL_COMMENT + str(bug_id) + '/comment')
        bug_fcomment = bug_first_comment.json()

        print("Bug Id: " + str(bug_id))

        with open(BUGS_FILE_PATH, 'a+') as bug_file:
            bug_file.write(line.format(
                bug['id'],
                bug['summary'],
                bug['platform'],
                bug['component'],
                bug['creation_time'],
                bug['whiteboard'],
                bug['cf_qa_whiteboard'],
                bug_fcomment['bugs'][str(bug_id)]['comments'][0]['text'].replace("\n"," "),
                bug_fcomment['bugs'][str(bug_id)]['comments'][0]['creation_time']
            )
        )
        
        #list_of_tuples = [(bug['id'],
        #        bug['summary'],
        #        bug['platform'],
        #        bug['component'],
        #        bug['creation_time'],
        #        bug['whiteboard'],
        #        bug['cf_qa_whiteboard'],
        #        bug['first_comment'],
        #        bug['first_comment_creation_time']) for bug in bugs_list]

        #for i in range(len(list_of_tuples)):
        #    for j in range(len(list_of_tuples[i])):
        #        print('{}{}: {}'.format(i, j, list_of_tuples[i][j]))

        #cursor.executemany(
        #   """INSERT OR REPLACE INTO bugs (id, summary, platform, component, creation_time, whiteboard, cf_qa_whiteboard, first_comment, first_comment_creation_time)
        #           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", list_of_tuples)
        #conn.commit()

        #print("Bugs inserted in the database.")

# desconectando...
#conn.close()