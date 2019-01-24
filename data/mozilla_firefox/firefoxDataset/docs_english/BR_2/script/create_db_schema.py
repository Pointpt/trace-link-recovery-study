import sqlite3

conn = sqlite3.connect('firefox_bugs.db')

cursor = conn.cursor()

cursor.execute("""
        CREATE TABLE IF NOT EXISTS bugs (
                id INTEGER NOT NULL PRIMARY KEY,
                summary TEXT NOT NULL,
                platform TEXT NOT NULL,
                component TEXT NOT NULL,
                creation_time DATE NOT NULL,
                whiteboard TEXT NOT NULL,
                cf_qa_whiteboard TEXT NOT NULL,
                first_comment TEXT NOT NULL,
                first_comment_creation_time DATE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS testcases (
                id INTEGER NOT NULL PRIMARY KEY,
                testday TEXT NOT NULL,
                gentitle TEXT NOT NULL,
                ctr_num INTEGER NOT NULL,
                title TEXT NOT NULL,
                preconditions TEXT NOT NULL,
                steps TEXT NOT NULL,
                expected_results TEXT NOT NULL
        );   
""")

print('Tables created with success.')

conn.close()