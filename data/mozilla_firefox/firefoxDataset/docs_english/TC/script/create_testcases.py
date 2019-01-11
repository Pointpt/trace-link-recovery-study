import csv

BASE_FILE_PATH = 'data/mozilla_firefox/firefoxDataset/docs_english/TC/{0}.txt'
TESTCASES_FILE_PATH = "data/mozilla_firefox/firefoxDataset/docs_english/TC/script/testcases.csv"

line = "TC Number: {0}\nTest Day: {1}\nGeneric Title: {2}\nCrt. Nr. {3}\nTitle: {4}\nPreconditions: {5}\nSteps: {6}\nExpected Result: {7}"

with(open(TESTCASES_FILE_PATH, 'r')) as testcases_file:
    reader = csv.reader(testcases_file, delimiter=',')
    next(reader, None)
    for row in reader:
        with(open(BASE_FILE_PATH.format(row[0]), 'w')) as tc_file:
            tc_file.write(line.format(*row))