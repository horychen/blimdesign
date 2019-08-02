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

run_folder_set_dict = {'IM' + ' Combined':[], 
                  'IM' + ' Separate':[],
                  'PM' + ' Combined':[],
                  'PM' + ' Separate':[],
                  'IM' + ' No Ripple Constraint':[], # wo/ $E_a$ Constraint
                  'PM' + ' No Ripple Constraint':[]} # wo/ $E_a$ Constraint
                  # 'IM' + ' Combined' + ' No Ripple Constraint':[], # wo/ $E_a$ Constraint
                  # 'PM' + ' Combined' + ' No Ripple Constraint':[]} # wo/ $E_a$ Constraint

    # JMAG 狂甩Exception，如果你试图在一台电脑上同时调用两个jmag实例，就算是Hide状态也不行，当然有一定的概率你可以继续运行，但是出现exception的概率翻了好几倍！
    # run_folder_set_dict['IM' + ' Combined']   += [ ('Y730',  r'run#550/'),    ('Severson01Local', r'run#550010/') ]
    # run_folder_set_dict['IM' + ' Separate']   += [                            ('Severson02', r'run#550020/'), ('T440p', r'run#550040/') ]
    # run_folder_set_dict['PM' + ' Combined'] += [                            ('Severson02', r'run#603020/'), ('T440p', r'run#603010/') ]
    # run_folder_set_dict['PM' + ' Separate'] += [                            ('Severson01', r'run#604010/') ]

run_folder_set_dict['IM' + ' Combined']   += [ ('Y730',  r'run#550/'),               ]
run_folder_set_dict['IM' + ' Separate']   += [ ('Severson02', r'run#550020/')        ]
run_folder_set_dict['PM' + ' Combined'] += [ ('Severson02', r'run#603020/')        ]
run_folder_set_dict['PM' + ' Separate'] += [ ('Severson01', r'run#604010/')        ]

run_folder_set_dict['IM' + ' No Ripple Constraint'] += [ ('Severson02', r'run#550019/')        ]
run_folder_set_dict['PM' + ' No Ripple Constraint'] += [ ('Severson01', r'run#603029/')        ]
# run_folder_set_dict['IM' + ' Combined' + ' No Ripple Constraint'] += [ ('Severson02', r'run#550019/')        ]
# run_folder_set_dict['PM' + ' Combined' + ' No Ripple Constraint'] += [ ('Severson01', r'run#603029/')        ]

fea_config_dict['run_folder'] = 'None'
# spec's
my_execfile('./spec_TIA_ITEC_.py', g=globals(), l=locals())
spec

import acm_designer
global ad
ad = acm_designer.acm_designer(fea_config_dict, spec)



from pylab import mpl

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

# mpl.style.use('classic')
mpl.rcParams['legend.fontsize'] = 12.5
# mpl.rcParams['legend.family'] = 'Times New Roman'
mpl.rcParams['font.family'] = ['Times New Roman']
mpl.rcParams['font.size'] = 14.0
font = {'family' : 'Times New Roman', #'serif',
        'color' : 'darkblue',
        'weight' : 'normal',
        'size' : 14,}
textfont = {'family' : 'Times New Roman', #'serif',
            'color' : 'darkblue',
            'weight' : 'normal',
            'size' : 11.5,}

# mpl.rcParams['mathtext.fontset'] = 'custom'
# mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

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
markers = ['s', '^', 'o', 'd', 's', 'o']
# markers = ['s', '$I$', 'o', '$P$']
# markers = ['$IC$', '$IS$', '$PC$', '$PS$']
colors = ['tomato', '#C766A1', '#BBCD49', '#3064FD', 'tomato', '#BBCD49'] # blue:#3064FD, purple:6F50E2
edgecolors = ['white', 'white', 'white', 'white', 'black', 'black']
zorders = [13,16,11,15, 6,5]
index = 0

# fig = plt.figure()
fig, ax = plt.subplots(1, 1, figsize=(6,3), sharex=True, facecolor='w', edgecolor='k', constrained_layout=True)
for key, val in run_folder_set_dict.items():
    print('-'*20, key)
    SWARM_DATA = []
    if len(val) == 0:
        continue
    for run_folder_set in val:
        print(run_folder_set[0], run_folder_set[1])
        SWARM_DATA += get_swarm_data(run_folder_set[1])

        if ad.solver.swarm_data_container is not None:
            # if 'PMSMC' in key or 'IMS' in key:
            # if 'Com' in key:
            # if 'IMS' not in key:
            ll = ad.solver.swarm_data_container.get_list_y_data()
            print('\t index:', index)
            if 'Constraint' in key:
                x = [1e-3*ll[0][idx] for idx, el in enumerate(ll[2]) ]
                y = [-ll[1][idx] for idx, el in enumerate(ll[2]) ]
            else:
                # x = [1e-3*ll[0][idx] for idx, el in enumerate(ll[2]) ]
                # y = [-ll[1][idx] for idx, el in enumerate(ll[2]) ]
                x = [1e-3*ll[0][idx] for idx, el in enumerate(ll[2]) if el < 5]
                y = [-ll[1][idx] for idx, el in enumerate(ll[2]) if el < 5]
            plt.scatter(x, y, marker=markers[index], color=colors[index], edgecolor=edgecolors[index], alpha=1.0, label=key, zorder=zorders[index])
            # plt.scatter(x, y, marker=markers[index], color=colors[index], edgecolor='black', alpha=0.8, label=key)
            # plt.scatter(x, y, marker=markers[index], color=colors[index], edgecolor=None, alpha=0.8, label=key)

        # if ad.solver.swarm_data_container is not None:
        #     DIMENSION = len(ad.solver.swarm_data[0][:-3])
        #     udp = Problem_Dummy()
        #     prob = pg.problem(udp)
        #     pop_archive = pg.population(prob, size=len(ad.solver.swarm_data))
        #     for i in range(len(ad.solver.swarm_data)):
        #         pop_archive.set_xf(i, ad.solver.swarm_data[i][:-3], ad.solver.swarm_data[i][-3:])
        #     fits, vectors = pop_archive.get_f(), pop_archive.get_x()
        #     ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)
        #     utility_moo.my_plot(pop_archive.get_f(), pop_archive.get_x(), ndf)


    print('Count of chromosomes:', len(SWARM_DATA))
    data_dict[key] = (SWARM_DATA, markers[index], colors[index])
    index += 1
# plt.xlabel('Motor Stack Length [mm]')
plt.xlabel('Torque per Rotor Volume [${\\rm kNm/m^3}$]')
plt.ylabel('Efficiency [%]')
from collections import OrderedDict
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc=(0.50,0.0)).set_zorder(2)
plt.grid(zorder=1)
# plt.xticks(range(40,241,40))
plt.xlim([2,57])
plt.ylim([90,98.5])
fig.savefig(r'C:\Users\horyc\Desktop\IM_PMSM_COMPARISON.eps', format='eps', dpi=1000)
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


