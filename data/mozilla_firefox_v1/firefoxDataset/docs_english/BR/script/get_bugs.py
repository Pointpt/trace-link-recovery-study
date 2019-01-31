import requests

BASE_URL = "https://bugzilla.mozilla.org/rest/bug/"

BASE_FILE_PATH = "data/mozilla_firefox/firefoxDataset/docs_english/BR/{0}.txt"

bug_ids = [
    779465, 604710,   #  20160603
    542990,           #  20160624
    1285719,          #  20160708
    1288951,          #  20160722
    1255261, 1294761, 
    1294765, 1294961, 
    1294981, 1294711, 
    1294974, 1294981, 
    1294976, 1294970, 
    1294962,          #  20160812
    1298395, 1298575, #  20160826
                      #  20160909
    1307346, 1298027, #  20160930
                      #  20161014
    1313805,          #  20161028
    1320617, 1320615,
    1316294, 1320658,
    1320419, 1320345,
    1320500, 1310247, 
    1320485,          #  20161125
                      #  20170106
                      #  20170120
                      #  20170203
                      #  20170217
                      #  20170303
                      #  20170317
                      #  20170331
                      #  20170428
                      #  20170512
                      #  20170623
                      #  20170707
                      #  20170721
                      #  20170818
                      #  20170901
                      #  20170915
                      #  20171013
                      #  20171027
                      #  20171117
                      #  20171222
                      #  20180202
                      #  20180216
                      #  20180302
                      #  20180323
                      #  20180420
                      #  20180518
                      #  20180615
                      #  20180713
                      #  20180803
                      #  20180817
                      #  20180928
                      #  20181012
                      #  20181109
                      #  20181123
                      #  20181221
                      #  20190111
]

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