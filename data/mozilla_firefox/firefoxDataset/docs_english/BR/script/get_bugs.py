import requests

BASE_URL = "https://bugzilla.mozilla.org/rest/bug/"

BASE_FILE_PATH = "data/mozilla_firefox/firefoxDataset/docs_english/BR/{0}.txt"

bug_ids = [1437359,1436284,1436872,1436980,1437320,1437354,1437397]

line = "Bug Number: {0}\nSummary: {1}\nPlatform: {2}\nComponent: {3}\nCreation Time: {4}\nWhiteboard: {5}\nQA Whiteboard:{6}\nFirst Comment Text: {7}\nFirst Comment Creation Time: {8}"

for bug_id in bug_ids:
    with open(BASE_FILE_PATH.format(bug_id) , 'w') as bug_file:
        bug = requests.get(BASE_URL + str(bug_id))
        bug_json = bug.json()
        
        bug_first_comment = requests.get(BASE_URL + str(bug_id) + '/comment')
        bug_fcomment = bug_first_comment.json()

        print("Bug Id: " + str(bug_id))
        
        bug_file.write(line.format(
            bug_json['bugs'][0]['id'],
            bug_json['bugs'][0]['summary'],
            bug_json['bugs'][0]['platform'],
            bug_json['bugs'][0]['component'],
            bug_json['bugs'][0]['creation_time'],
            bug_json['bugs'][0]['whiteboard'],
            bug_json['bugs'][0]['cf_qa_whiteboard'],
            bug_fcomment['bugs'][str(bug_id)]['comments'][0]['text'],
            bug_fcomment['bugs'][str(bug_id)]['comments'][0]['creation_time'],
            )
        )