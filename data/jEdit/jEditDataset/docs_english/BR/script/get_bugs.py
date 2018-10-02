import requests

BASE_URL = "https://sourceforge.net/rest/p/jedit/bugs/"

BASE_FILE_PATH = "data/jEdit/jEditDataset/docs_english/BR/BR_{0}.txt"

tickets_ids = [4067]
line = "Bug Number: {0}\nSummary: {1}\nDescription: {2}"

for bug_id in tickets_ids:
    with open(BASE_FILE_PATH.format(bug_id) , 'w') as bug_file:
        bug = requests.get(BASE_URL + str(bug_id))
        bug_json = bug.json()
        
        print("Bug Id: " + str(bug_id))
        
        bug_file.write(line.format(
            bug_json['ticket']['ticket_num'],
            bug_json['ticket']['summary'],
            bug_json['ticket']['description'])
        )