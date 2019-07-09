import os
def copyfile(source, dest):
    with open(source, 'r') as src, open(dest, 'w') as dst: dst.write(src.read())
if not os.path.isdir('./txt_collected/'):
    os.mkdir('./txt_collected/')

run_folder_set_dict = {'IM' + 'Combined':[], 
                  'IM' + 'Separate':[],
                  'PMSM' + 'Combined':[],
                  'PMSM' + 'Separate':[]}

run_folder_set_dict['IM' + 'Combined']   += [ ('Y730',  r'run#550/'),    ('Severson01', r'run#550010/') ]
run_folder_set_dict['IM' + 'Separate']   += [ ('T440p', r'run#550040/'), ('Severson02', r'run#550020/') ]
run_folder_set_dict['PMSM' + 'Combined'] += [ ('T440p', r'run#603010/'), ('Severson02', r'run#603020/') ]
run_folder_set_dict['PMSM' + 'Separate'] += [                            ('Severson01', r'run#604010/') ]

for key, val in run_folder_set_dict.items():
    # print(key, val)
    for pair in val:
        path_to = '../' + pair[1]
        if os.path.isdir(path_to):
            if not os.path.isdir('./txt_collected/' + pair[1]):
                os.mkdir('./txt_collected/' + pair[1])
            try:
                copyfile(path_to + 'swarm_survivor.txt', './txt_collected/' + pair[1] + 'swarm_survivor.txt')
            except Exception as e:
                print(e)
            try:
                copyfile(path_to + 'swarm_MOO_log.txt',  './txt_collected/' + pair[1] + 'swarm_MOO_log.txt')
            except Exception as e:
                print(e)
            try:
                copyfile(path_to + 'swarm_data.txt',     './txt_collected/' + pair[1] + 'swarm_data.txt')
            except Exception as e:
                print(e)

