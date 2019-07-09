import shutil
from utility import my_execfile
import utility_moo
bool_post_processing = False # solve or post-processing

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 0. FEA Setting / General Information & Packages Loading
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
my_execfile('./default_setting.py', g=globals(), l=locals())
fea_config_dict
fea_config_dict['local_sensitivity_analysis'] = False
fea_config_dict['bool_refined_bounds'] = False
fea_config_dict['use_weights'] = 'O2' # this is not working

run_folder_set_dict = {'IM' + 'Combined':[], 
                  'IM' + 'Separate':[],
                  'PMSM' + 'Combined':[],
                  'PMSM' + 'Separate':[]}

run_folder_set_dict['IM' + 'Combined']   += [ ('Y730',  r'run#550/'),    ('Severson01', r'run#550010/') ]
run_folder_set_dict['IM' + 'Separate']   += [ ('T440p', r'run#550040/'), ('Severson02', r'run#550020/') ]
run_folder_set_dict['PMSM' + 'Combined'] += [ ('T440p', r'run#603010/'), ('Severson02', r'run#603020') ]
run_folder_set_dict['PMSM' + 'Separate'] += [                            ('Severson01', r'run#604010') ]

# fea_config_dict['run_folder'] = run_folder
# spec's
my_execfile('./spec_TIA_ITEC_.py', g=globals(), l=locals())
spec

import acm_designer
global ad
ad = acm_designer.acm_designer(fea_config_dict, spec)


def get_swarm_data(run_folder):
    ad.solver.output_dir = ad.solver.fea_config_dict['dir_parent'] + run_folder # severson01
    number_of_chromosome = ad.solver.read_swarm_data()
    return ad.solver.swarm_data

data_dict = {}
markers = ['s', '^', 'o', '*']
colors = ['tomato', '#C766A1', '#3064FD', '#6F50E2']
index = 0
for key, val in run_folder_set_dict.items():
    print('-'*20, key)
    swarm_data = []
    for run_folder_set in val:
        print(run_folder_set[0], run_folder_set[1])
        swarm_data += get_swarm_data(run_folder_set[1])
    print('Count of chromosomes:', len(swarm_data))
    data_dict[key] = (swarm_data, markers[index], colors[index])
    index += 1

# Combine all data 
# swarm_data_severson01 = get_swarm_data(r'run#540???/')
# swarm_data_severson02 = get_swarm_data(r'run#540???/')
# swarm_data_Y730       = get_swarm_data(r'run#540???/')
# swarm_data_T440p      = get_swarm_data(r'run#540???/')

# print('Sizes of the 4 populations (in order):', len(swarm_data_severson01), len(swarm_data_severson02), len(swarm_data_Y730), len(swarm_data_T440p))
# ad.solver.swarm_data = swarm_data_severson01 + swarm_data_severson02 + swarm_data_Y730 + swarm_data_T440p # list add

# Learn Pareto front rank and plot
for el in ad.solver.swarm_data:
    print('\t', el)
print('count:', len(ad.solver.swarm_data))

udp = Problem_BearinglessInductionDesign()
ad.flag_do_not_evaluate_when_init_pop = True
counter_fitness_called, counter_fitness_return = 0, 0
prob = pg.problem(udp)




# swarm_data_on_pareto_front = utility_moo.learn_about_the_archive(prob, ad.solver.swarm_data, len(ad.solver.swarm_data), fea_config_dict)
if True:
    print('Plot for key %s'%(key))
    swarm_data = data_dict[key][0]
    marker = data_dict[key][1]
    color = data_dict[key][2]

    number_of_chromosome = len(swarm_data)
    print('Archive size:', number_of_chromosome)
    # for el in swarm_data:
    #     print('\t', el)

    pop_archive = pg.population(prob, size=number_of_chromosome)
    for i in range(number_of_chromosome):
        pop_archive.set_xf(i, swarm_data[i][:-3], swarm_data[i][-3:])

    sorted_index = pg.sort_population_mo(points=pop_archive.get_f())
    print('Sorted by domination rank and crowding distance:', len(sorted_index))
    print('\t', sorted_index)

    # 这段代码对于重建种群来说不是必须的，单单sort_population_mo（包含fast_non_dominated_sorting和crowding_distance）就够了，
    # 只是我想看看，具体的crowding_distance是多少，然后我想知道排在前面的多少个是属于domination rank 1的。
    if True:
        fits, vectors = pop_archive.get_f(), pop_archive.get_x()
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)

        ind1, ind2 = 0, 0
        for rank_minus_1, front in enumerate(ndf):

            ind2 += len(front)
            sorted_index_at_this_front = sorted_index[ind1:ind2]
            fits_at_this_front = [fits[point] for point in sorted_index_at_this_front]

            # Rank 1 Pareto Front
            if ind1 == 0:
                rank1_ParetoPoints = fits_at_this_front
                if len(front) < popsize:
                    print('There are not enough chromosomes (%d) belonging to domination rank 1 (the best Pareto front).\nWill use rank 2 or lower to reach popsize of %d.'%(len(front), popsize))

            # this crwdsit should be already sorted as well
            if len(fits_at_this_front) >= 2: # or else error:  A non dominated front must contain at least two points: 1 detected.
                crwdst = pg.crowding_distance(fits_at_this_front)
            else:
                print('A non dominated front must contain at least two points: 1 detected.')
                crwdst = [999999]


            print('\nRank/Tier', rank_minus_1+1, 'chromosome count:', len(front), len(sorted_index_at_this_front))
            # print('\t', sorted_index_at_this_front.tolist())
            # print('\t', crwdst)
            print('\tindex in pop\t|\tcrowding distance')
            for index, cd in zip(sorted_index_at_this_front, crwdst):
                print('\t', index, '\t\t\t|', cd)
            ind1 = ind2

    sorted_vectors = [vectors[index].tolist() for index in sorted_index]
    sorted_fits    = [fits[index].tolist() for index in sorted_index]

    my_plot(pop_archive.get_f(), pop_archive.get_x(), ndf)

    swarm_data_on_pareto_front = [design_parameters_denorm + fits for design_parameters_denorm, fits in zip(sorted_vectors, sorted_fits)]

plt.show()
quit()


