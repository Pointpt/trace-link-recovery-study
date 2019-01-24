import requests
import sqlite3

INIT_DATE = "2016-06-06"
END_DATE = "2016-12-31"
PRODUCT = "firefox"
QUERY_FORMAT = "advanced"
QUERY_BASED_ON = ""
J_TOP = "OR"
LIMIT = 1000

BASE_URL = "https://bugzilla.mozilla.org/rest/bug?chfieldfrom={}&chfieldto={}&product={}&query_format={}&limit={}&query_based_on={}&j_top={}&offset={}"
BASE_URL_COMMENT = "https://bugzilla.mozilla.org/rest/bug/"

args = [INIT_DATE, END_DATE, PRODUCT, QUERY_FORMAT, LIMIT, QUERY_BASED_ON, J_TOP]

conn = sqlite3.connect('firefox_bugs.db')

cursor = conn.cursor()

for offset_ in range(0, 7000, 1000):
    if len(args) < 8:
        args = args + [offset_]
    else:
        args[7] = offset_

    bugs = requests.get(BASE_URL.format(*args))
    bugs_json = bugs.json()
    
    with open('f_{}.json'.format(offset_), 'w') as ff:
            ff.write(str(bugs_json))

    """
    for bug in bugs_json['bugs']:
        bug_id = bug['id']
        #with open(BASE_FILE_PATH.format(bug_id) , 'w') as bug_file:
        
        bug_first_comment = requests.get(BASE_URL_COMMENT + str(bug_id) + '/comment')
        bug_fcomment = bug_first_comment.json()

        #print("Bug Id: " + str(bug_id))
        
        cursor.execute(
                #INSERT OR REPLACE INTO bugs (id, summary, platform, component, creation_time, whiteboard, cf_qa_whiteboard, first_comment, first_comment_creation_time)
                #VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                , (bug['id'], 
                        bug['summary'], 
                        bug['platform'], 
                        bug['component'],
                        bug['creation_time'],
                        bug['whiteboard'],
                        bug['cf_qa_whiteboard'],
                        bug_fcomment['bugs'][str(bug_id)]['comments'][0]['text'],
                        bug_fcomment['bugs'][str(bug_id)]['comments'][0]['creation_time']))

        conn.commit()

        #print("Bug inserted in the database.")
        """

# desconectando...
conn.close()