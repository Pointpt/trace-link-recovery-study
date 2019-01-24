import csv
import sqlite3

TESTCASES_FILE_PATH = "data/mozilla_firefox/firefoxDataset/docs_english/TC/script/testcases.csv"


# conectando...
conn = sqlite3.connect('firefox_bugs.db')

# definindo um cursor
cursor = conn.cursor()

with(open(TESTCASES_FILE_PATH, 'r')) as testcases_file:
    reader = csv.reader(testcases_file, delimiter=',')
    next(reader, None)
    for row in reader:
        cursor.execute("""
                INSERT OR REPLACE INTO testcases (
                    id, 
                    testcases, 
                    gentitle, 
                    ctr_num, 
                    title, 
                    preconditions, 
                    steps, 
                    expected_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (*row))

        conn.commit()

        print("TestCase inserted in the database.")

# desconectando...
conn.close()