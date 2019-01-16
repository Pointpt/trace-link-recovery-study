import requests

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

BASE_FILE_PATH = "data/mozilla_firefox/firefoxDataset/docs_english/BR/{0}.txt"

line = "Bug Number: {0}\nSummary: {1}\nPlatform: {2}\nComponent: {3}\nCreation Time: {4}\nWhiteboard: {5}\nQA Whiteboard:{6}\nFirst Comment Text: {7}\nFirst Comment Creation Time: {8}"

for offset_ in range(0, 7000, 1000):
    args = args + [offset_]
    bugs = requests.get(BASE_URL.format(*args))
    bugs_json = bugs.json()
    
    for bug in bugs_json['bugs']:
        bug_id = bug['id']
        with open(BASE_FILE_PATH.format(bug_id) , 'w') as bug_file:
            bug_first_comment = requests.get(BASE_URL_COMMENT + str(bug_id) + '/comment')
            bug_fcomment = bug_first_comment.json()

            print("Bug Id: " + str(bug_id))
            
            bug_file.write(line.format(
                bug['id'],
                bug['summary'],
                bug['platform'],
                bug['component'],
                bug['creation_time'],
                bug['whiteboard'],
                bug['cf_qa_whiteboard'],
                bug_fcomment['bugs'][str(bug_id)]['comments'][0]['text'],
                bug_fcomment['bugs'][str(bug_id)]['comments'][0]['creation_time'],
                )
            )