
files = ['CC_CC', 'CC_TC', 
            'ID_TC', 'ID_ID', 'ID_CC', 'ID_UC',
            'TC_CC', 'TC_TC', 
            'UC_CC', 'UC_ID', 'UC_TC', 'UC_UC']

trace_matrix = {}

def read_matrixes():
    path = 'data/EasyClinic/EasyClinicDataset/oracle/'
    for f_name in files:
        fl = open(path + f_name + '.txt')
        lines = fl.readlines()
        source_abv = f_name.split('_')[0] + '_'
        target_abv = f_name.split('_')[1] + '_'
        for l in lines:
            source, targets = l.split(':')[0], l.split(':')[1]
            targets_list = targets.split(' ')
            
            trace_matrix[source_abv + source.replace('.txt', '')] = [target_abv  + t.replace('.txt', '') for t in targets_list[:-1]]

if __name__ == '__main__':
    read_matrixes()
    
    for source, targets_list in trace_matrix.items():
        print(source + ' : ' + str(targets_list))
