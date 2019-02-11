import pandas as pd

# read bug_reports_final dataset
bugreports_final = pd.read_csv('~/anaconda3/envs/trace-link-recovery-study/data/mozilla_firefox_v2/firefoxDataset/docs_english/BR/bugreports_final.csv')
print(bugreports_final.shape)

brs_versions = ['48 Branch', '49 Branch', '50 Branch', '51 Branch']
brs_status = ['NEW','RESOLVED']
brs_priority = ['P1', 'P2', 'P3']
brs_resolutions = ['FIXED']
brs_severities = ['major', 'normal', 'blocker', 'critical']
brs_isconfirmed = [True]
selected_bugs = bugreports_final[(bugreports_final.Version.isin(brs_versions)) &
                                 (bugreports_final.Status.isin(brs_status)) &
                                 (bugreports_final.Priority.isin(brs_priority)) &
                                 (bugreports_final.Resolution.isin(brs_resolutions)) &
                                 (bugreports_final.Severity.isin(brs_severities)) &
                                 (bugreports_final.Is_Confirmed.isin(brs_isconfirmed))
                                ]
print(selected_bugs.shape)
selected_bugs.head(50)