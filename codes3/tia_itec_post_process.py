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

run_folder_set_dict['IM' + 'Combined']   += [ ('Y730',  r'run#550/'),    ('Severson01Local', r'run#550010/') ]
run_folder_set_dict['IM' + 'Separate']   += [                            ('Severson02', r'run#550020/'), ('T440p', r'run#550040/') ]
run_folder_set_dict['PMSM' + 'Combined'] += [                            ('Severson02', r'run#603020/'), ('T440p', r'run#603010/') ]
run_folder_set_dict['PMSM' + 'Separate'] += [                            ('Severson01', r'run#604010/') ]

run_folder_set_dict['IM' + 'Combined']   += [ ('Y730',  r'run#550/'),    ('Severson01Local', r'run#550010/') ]
run_folder_set_dict['IM' + 'Separate']   += [                            ('Severson02', r'run#550020/') ]
run_folder_set_dict['PMSM' + 'Combined'] += [                            ('Severson02', r'run#603020/') ]
run_folder_set_dict['PMSM' + 'Separate'] += [                            ('Severson01', r'run#604010/') ]

fea_config_dict['run_folder'] = 'None'
# spec's
my_execfile('./spec_TIA_ITEC_.py', g=globals(), l=locals())
spec

import acm_designer
global ad
ad = acm_designer.acm_designer(fea_config_dict, spec)





import pygmo as pg
global DIMENSION

class Problem_Dummy(object):

    # Define objectives
    def fitness(self, x):
        return [0, 0, 0]

    # Return number of objectives
    def get_nobj(self):
        return 3

    # Return bounds of decision variables (a.k.a. chromosome)
    def get_bounds(self):
        global DIMENSION
        return ([0]*DIMENSION, [1]*DIMENSION)

















# from pylab import plt, np

def get_swarm_data(run_folder):
    ad.solver.swarm_data_container = None
    ad.solver.swarm_data = []
    ad.solver.output_dir = './txt_collected/' + run_folder # severson01
    number_of_chromosome = ad.solver.read_swarm_data()
    print('\tnumber_of_chromosome:', number_of_chromosome)
    return ad.solver.swarm_data

data_dict = {}
markers = ['s', '^', 'o', '*']
colors = ['tomato', '#C766A1', '#3064FD', '#6F50E2']
index = 0
plt.figure()
for key, val in run_folder_set_dict.items():
    print('-'*20, key)
    SWARM_DATA = []
    for run_folder_set in val:
        print(run_folder_set[0], run_folder_set[1])
        SWARM_DATA += get_swarm_data(run_folder_set[1])

        if ad.solver.swarm_data_container is not None:
            # if 'PMSMC' in key or 'IMS' in key:
            # if 'Com' in key:
            # if 'IMS' not in key:
            ll = ad.solver.swarm_data_container.get_list_y_data()
            print('\t index:', index)
            plt.scatter(ll[0], ll[1], marker=markers[index], color=colors[index], alpha=0.6, label=key)

        if ad.solver.swarm_data_container is not None:
            DIMENSION = len(ad.solver.swarm_data[0][:-3])
            udp = Problem_Dummy()
            prob = pg.problem(udp)
            pop_archive = pg.population(prob, size=len(ad.solver.swarm_data))
            for i in range(len(ad.solver.swarm_data)):
                pop_archive.set_xf(i, ad.solver.swarm_data[i][:-3], ad.solver.swarm_data[i][-3:])
            fits, vectors = pop_archive.get_f(), pop_archive.get_x()
            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)
            utility_moo.my_plot(pop_archive.get_f(), pop_archive.get_x(), ndf)


    print('Count of chromosomes:', len(SWARM_DATA))
    data_dict[key] = (SWARM_DATA, markers[index], colors[index])
    index += 1
plt.xlabel('stack length [mm]')
plt.ylabel('-$\\eta$ [%] (efficiency)')
plt.grid()
from collections import OrderedDict
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper left')
plt.show()
quit()


# Learn Pareto front rank and plot
print('count:', len(ad.solver.swarm_data))

popsize = 78
for key, val in run_folder_set_dict.items():
    swarm_data = data_dict[key][0]
    marker = data_dict[key][1]
    color = data_dict[key][2]

    number_of_chromosome = len(swarm_data)
    print('Archive size:', number_of_chromosome)
    # for el in swarm_data:
    #     print('\t', el)

    DIMENSION = len(swarm_data[0][:-3])
    udp = Problem_Dummy()
    prob = pg.problem(udp)
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


    utility_moo.my_plot(pop_archive.get_f(), pop_archive.get_x(), ndf)

    swarm_data_on_pareto_front = [design_parameters_denorm + fits for design_parameters_denorm, fits in zip(sorted_vectors, sorted_fits)]

    print('Plot for key %s'%(key))
    print('-'*20, key)
    plt.show()


