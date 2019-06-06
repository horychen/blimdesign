# coding:u8
import shutil
from utility import my_execfile
bool_post_processing = False # solve or post-processing

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 0. FEA Setting / General Information & Packages Loading
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
my_execfile('./default_setting.py', g=globals(), l=locals())
fea_config_dict
if True:
    # ECCE
    my_execfile('./spec_ECCE_4pole32Qr1000Hz.py', g=globals(), l=locals())
    spec

    fea_config_dict['local_sensitivity_analysis'] = True
    fea_config_dict['bool_refined_bounds'] = True
    fea_config_dict['use_weights'] = 'O2'

    # run_folder = r'run#530/' # 圆形槽JMAG绘制失败BUG

    # fea_config_dict['local_sensitivity_analysis_number_of_variants'] = 2
    # run_folder = r'run#531/' # number_of_variant = 2

    # fea_config_dict['local_sensitivity_analysis_number_of_variants'] = 10
    # run_folder = r'run#533/' # number_of_variant = 10

    fea_config_dict['local_sensitivity_analysis_number_of_variants'] = 20
    run_folder = r'run#534/' # number_of_variant = 20


    fea_config_dict['local_sensitivity_analysis'] = False
    fea_config_dict['bool_refined_bounds'] = False
    fea_config_dict['use_weights'] = 'O2'
    run_folder = r'run#535/' # test with pygmo


    fea_config_dict['local_sensitivity_analysis'] = False
    fea_config_dict['bool_refined_bounds'] = False
    fea_config_dict['use_weights'] = 'O2' # this is not working
    run_folder = r'run#537/' # test with pygmo

    run_folder = r'run#538/' # test with pygmo with new fitness and constraints, Rotor current density Jr is 6.5 Arms/mm^2

else:
    # Prototype
    pass
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # Severson02
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # my_execfile('./spec_Prototype2poleOD150mm500Hz_SpecifyTipSpeed.py', g=globals(), l=locals()) # define spec
    my_execfile('./spec_ECCE_4pole32Qr1000Hz.py', g=globals(), l=locals())
    fea_config_dict['local_sensitivity_analysis'] = False
    fea_config_dict['bool_refined_bounds'] = False
    fea_config_dict['use_weights'] = 'O2'
    run_folder = r'run#538021/'

    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # Severson01
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # my_execfile('./spec_Prototype4poleOD150mm1000Hz_SpecifyTipSpeed.py', g=globals(), l=locals()) # define spec
    my_execfile('./spec_ECCE_4pole32Qr1000Hz.py', g=globals(), l=locals())
    fea_config_dict['local_sensitivity_analysis'] = False
    fea_config_dict['bool_refined_bounds'] = False
    fea_config_dict['use_weights'] = 'O2'
    run_folder = r'run#538011/'

fea_config_dict['run_folder'] = run_folder
fea_config_dict['Active_Qr'] = spec.Qr
fea_config_dict['use_drop_shape_rotor_bar'] = spec.use_drop_shape_rotor_bar
build_model_name_prefix(fea_config_dict) # rebuild the model name for fea_config_dict

import acm_designer
global ad
ad = acm_designer.acm_designer(fea_config_dict, spec)
# if 'Y730' in fea_config_dict['pc_name']:
#     ad.build_oneReport() # require LaTeX
#     # ad.talk_to_mysql_database() # require MySQL
ad.init_logger()

ad.bounds_denorm = ad.get_classic_bounds()
print('classic_bounds and original_bounds')
for A, B in zip(ad.bounds_denorm, ad.original_bounds):
    print(A, B)
# quit()

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Optimization
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~

import pygmo as pg
class Problem_BearinglessInductionDesign(object):

    # Define objectives
    def fitness(self, x):
        global ad, counter_fitness_called, counter_fitness_return

        if ad.flag_do_not_evaluate_when_init_pop == True:
            return [0, 0, 0]

        ad, counter_fitness_called, counter_fitness_return
        if counter_fitness_called == counter_fitness_return:
            counter_fitness_called += 1
        else:
            # This is not reachable
            raise
        print('Call fitness: %d, %d'%(counter_fitness_called, counter_fitness_return))

        # 不要标幺化了！统一用真的bounds，见get_bounds()
        x_denorm = x

        # evaluate x_denorm via FEA tools
        counter_loop = 0
        while True:
            counter_loop += 1
            if counter_loop > 3:
                raise Exception('Abort the optimization. Three attemps to evaluate the design have all failed for individual #%d'%(counter_fitness_called))

            try:
                cost_function, f1, f2, f3, \
                normalized_torque_ripple, \
                normalized_force_error_magnitude, \
                force_error_angle = \
                    ad.evaluate_design(ad.spec.im_template, x_denorm, counter_fitness_called)

                # remove folder .jfiles to save space (we have to generate it first in JMAG Designer to have field data and voltage profiles)
                if ad.solver.folder_to_be_deleted is not None:
                    shutil.rmtree(ad.solver.folder_to_be_deleted) # .jfiles directory
                # update to be deleted when JMAG releases the use
                ad.solver.folder_to_be_deleted = ad.solver.expected_project_file[:-5]+'jfiles'

            except Exception as e:
                # raise e
                print(e)

                msg = 'FEA tool failed for individual #%d: attemp #%d.'%(counter_fitness_called, counter_loop)
                logger = logging.getLogger(__name__)
                logger.error(msg)
                print(msg)

                msg = 'Removing all files for individual #%d and try again...'%(counter_fitness_called)
                logger.error(msg)
                print(msg)

                # turn off JMAG Designer
                ad.solver.app.Quit()
                ad.solver.app = None

                # os.remove(ad.solver.expected_project_file) # .jproj
                # shutil.rmtree(ad.solver.expected_project_file[:-5]+'jfiles') # .jfiles directory # .jplot file in this folder will be used by JSOL softwares even JMAG Designer is closed.
                os.remove(ad.solver.femm_output_file_path) # . csv
                os.remove(ad.solver.femm_output_file_path[:-3]+'fem') # .fem
                for file in os.listdir(ad.solver.dir_femm_temp):
                    if 'femm_temp_' in file:
                        os.remove(ad.solver.dir_femm_temp + file)
            else:
                break

        # - Torque per Rotor Volume
        f1 #= - ad.spec.required_torque / rated_rotor_volume
        # - Efficiency @ Rated Power
        f2 #= - rated_efficiency
        # Ripple Performance (Weighted Sum)
        f3 #= sum(list_weighted_ripples)

        # Constraints (Em<0.2 and Ea<10 deg):
        if normalized_torque_ripple>=0.2 or normalized_force_error_magnitude >= 0.2 or force_error_angle > 10:
            f1 = 0
            f2 = 0

        counter_fitness_return += 1
        print('Fitness: %d, %d\n----------------'%(counter_fitness_called, counter_fitness_return))
        return [f1, f2, f3]

    # Return number of objectives
    def get_nobj(self):
        return 3

    # Return bounds of decision variables (a.k.a. chromosome)
    def get_bounds(self):

        # denormalize the normalized chromosome x to x_denorm
        min_b, max_b = np.asarray(ad.bounds_denorm).T 
        # diff = np.fabs(min_b - max_b)
        # x_denorm = min_b + x * diff

        # print(min_b.tolist(), max_b.tolist())
        # print(([0]*7, [1]*7))
        # quit()
        return ( min_b.tolist(), max_b.tolist() )
        return ([0]*7, [1]*7)

    # Return function name
    def get_name(self):
        return "Bearingless Induction Motor Design"

def my_plot(fits, vectors, ndf):
    if True:
        for fit in fits:
            print(fit)
        # ax = pg.plot_non_dominated_fronts(fits)
        ax = pg.plot_non_dominated_fronts(fits, comp=[0,1], marker='o')
        # ax = pg.plot_non_dominated_fronts(fits, comp=[0,2], marker='o')
        # ax = pg.plot_non_dominated_fronts(fits, comp=[1,2], marker='o')
    else:
        print('Valid for 2D objective function space only for now.')
        fig, axes = plt.subplots(ncols=2, nrows=1, dpi=150, facecolor='w', edgecolor='k');
        ax = axes[0]

        for index, xy in enumerate(vectors):
            ax.plot(*xy, 's', color='0')
            ax.text(*xy, '#%d'%(index), color='0')
        ax.title.set_text("Decision Vector Space")

        ax = axes[1]

        for index, front in enumerate(ndf):
            print('Rank/Tier', index, front)
            the_color = '%g'%(index/len(ndf))
            for individual in front: # individual is integer here
                ax.plot(*fits[individual], 'o',                 color=the_color)
                ax.text(*fits[individual], '#%d'%(individual),  color=the_color)

            front = front.tolist()
            front.sort(key = lambda individual: fits[individual][1]) # sort by y-axis value # individual is integer here
            for individual_A, individual_B in zip(front[:-1], front[1:]): # front is already sorted
                fits_A = fits[individual_A]
                fits_B = fits[individual_B]
                ax.plot([fits_A[0], fits_B[0]], 
                        [fits_B[1], fits_B[1]], color=the_color, lw=0.25) # 取高的点的Y
                ax.plot([fits_A[0], fits_A[0]], 
                        [fits_A[1], fits_B[1]], color=the_color, lw=0.25) # 取低的点的X
        ax.title.set_text("Pareto Front | Objective Function Space")
def my_print(pop, _):
    # ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)
    # extract and print non-dominated fronts
    # - ndf (list of 1D NumPy int array): the non dominated fronts
    # - dl (list of 1D NumPy int array): the domination list
    # - dc (1D NumPy int array): the domination count
    # - ndr (1D NumPy int array): the non domination ranks
    fits, vectors = pop.get_f(), pop.get_x()
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)

    with open(ad.solver.output_dir+'MOO_log.txt', 'a', encoding='utf-8') as fname:
        print('-'*40, 'Generation:', _, file=fname)
        for rank_minus_1, front in enumerate(ndf):
            print('Rank/Tier', rank_minus_1+1, front, file=fname)
        count = 0
        for domination_list, domination_count, non_domination_rank in zip(dl, dc, ndr):
            print('Individual #%d\t'%(count), 'Belong to Rank #%d\t'%(non_domination_rank), 'Dominating', domination_count, 'and they are', domination_list, file=fname)
            count += 1

        # print(fits, vectors, ndf)
        print(pop, file=fname)


def learn_about_the_archive(swarm_data, popsize):
    number_of_chromosome = len(swarm_data)
    print('Archive size:', number_of_chromosome)
    # for el in swarm_data:
    #     print('\t', el)

    pop_archive = pg.population(prob, size=number_of_chromosome)
    for i in range(number_of_chromosome):
        pop_archive.set_xf(i, swarm_data[i][:7], swarm_data[i][-3:])

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

            if ind2 == 0:
                if len(front) < popsize:
                    print('There are not enough chromosomes (%d) on the domination rank 1 (the best Pareto front).\nWill use rank 2 or lower to reach popsize of %d.'%(len(front), popsize))

            ind2 += len(front)
            sorted_index_at_this_front = sorted_index[ind1:ind2]

            fits_at_this_front = [fits[point] for point in sorted_index_at_this_front]

            # this crwdsit should be already sorted as well
            crwdst = pg.crowding_distance(fits_at_this_front)

            print('\nRank/Tier', rank_minus_1+1, 'chromosome count:', len(front), len(sorted_index_at_this_front))
            # print('\t', sorted_index_at_this_front.tolist())
            # print('\t', crwdst)
            print('\tindex in pop\t|\tcrowding distance')
            for index, cd in zip(sorted_index_at_this_front, crwdst):
                print('\t', index, '\t\t\t|', cd)
            ind1 = ind2

    sorted_vectors = [vectors[index].tolist() for index in sorted_index]
    sorted_fits    = [fits[index].tolist() for index in sorted_index]

    swarm_data_on_pareto_front = [design_parameters_denorm + fits for design_parameters_denorm, fits in zip(sorted_vectors, sorted_fits)]

    return swarm_data_on_pareto_front

# if bool_post_processing:
#     plot pareto plot for three objectives...
#     swda = ad.best_design_by_weights(fea_config_dict['use_weights'])
#     from pylab import show
#     show()
#     quit()
#     run_static_structural_fea(swda.best_design_denorm)


#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Multi-Objective Optimization
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
if True:
################################################################
# MOO Step 1:
#   Create UserDefinedProblem and create population
#   The magic method __init__ cannot be fined for UDP class
################################################################
    udp = Problem_BearinglessInductionDesign()
    global counter_fitness_called, counter_fitness_return
    counter_fitness_called, counter_fitness_return = 0, 0
    prob = pg.problem(udp)

        # Traceback (most recent call last):
        #   File "D:\OneDrive - UW-Madison\c\codes3\one_script.py", line 1189, in <module>
        #     pop = algo.evolve(pop)
        # ValueError: 
        # function: decomposition_weights
        # where: C:\bld\pygmo_1557474762576\_h_env\Library\include\pagmo/utils/multi_objective.hpp, 642
        # what: Population size of 72 is detected, but not supported by the 'grid' weight generation method selected. A size of 66 or 78 is possible.
    popsize = 78
    print('-'*40 + '\nPop size is', popsize)

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Add Restarting Feature
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~

    # 检查swarm_data.txt，如果有至少一个数据，返回就不是None。
    print('Check swarm_data.txt...')
    number_of_chromosome = ad.solver.read_swarm_data()
    if number_of_chromosome is not None:
        # 禁止在初始化pop时运行有限元
        ad.flag_do_not_evaluate_when_init_pop               = True

        number_of_finished_iterations                       = number_of_chromosome // popsize
        number_of_finished_chromosome_in_current_generation = number_of_chromosome % popsize

        # 如果刚好整除，把余数0改为popsize
        if number_of_finished_chromosome_in_current_generation == 0:
            number_of_finished_chromosome_in_current_generation = popsize
            print(f'\tThere are {number_of_chromosome} chromosomes found in swarm_data.txt.')
            print('\tWhat is the odds! The script just stopped when the evaluation of the whole pop is finished.')
            print('\tSet number_of_finished_chromosome_in_current_generation to popsize %d'%(number_of_finished_chromosome_in_current_generation))

        print('This is a restart of '+ fea_config_dict['run_folder'][:-1])
        print('\tNumber of finished iterations is %d'%(number_of_finished_iterations))
        # print('This means the initialization of the population class is interrupted. So the pop in swarm_data.txt is used as the survivor.')

        # 继续从swarm_survivor.txt中读取数据，注意，survivor总是是完整的一代的，除非popsize被修改了。
        print('\tCheck swarm_survivor.txt...', end='')
        ad.solver.survivor = ad.solver.read_swarm_survivor(popsize)

        # 如果发现ad.solver.survivor是None，那就说明是初始化pop的时候被中断了，此时就用swarm_data来生成pop。
        if ad.solver.survivor is not None:
            print('Found survivor!\nRestart the optimization based on the swarm_survivor.txt.')
        else:
            if len(ad.solver.survivor) != popsize:
                print('popsize is reduced') # 如果popsize增大了，read_swarm_survivor(popsize)就会报错了，因为-----不能被split后转为float
                raise Exception('This is a feature not tested. However, you can cheat to change popsize by manually modify swarm_data.txt or swarm_survivor.txt.')
            print('Gotta make do with swarm_data to generate survivor.')

        # 这些计数器的值永远都是评估过的chromosome的个数。
        counter_fitness_called = counter_fitness_return = number_of_chromosome
        print('counter_fitness_called = counter_fitness_return = number_of_chromosome = %d'%(number_of_chromosome))

    else:
        print('Nothing exists in swarm_data.txt.\nThis is a whole new run.')
        ad.flag_do_not_evaluate_when_init_pop = False
        number_of_finished_chromosome_in_current_generation = None
        number_of_finished_iterations = 0 # 实际上跑起来它不是零，而是一，因为我们认为初始化的一代也是一代。或者，我们定义number_of_finished_iterations = number_of_chromosome // popsize

    # 初始化population，如果ad.flag_do_not_evaluate_when_init_pop是False，那么就说明是 new run，否则，整代个体的fitness都是[0,0,0]。
    pop = pg.population(prob, size=popsize) 

    # 如果整代个体的fitness都是[0,0,0]，那就需要调用set_xf，把txt文件中的数据写入pop。如果发现数据的个数不够，那就调用set_x()来产生数据，形成初代个体。
    if ad.flag_do_not_evaluate_when_init_pop == True:
        pop_array = pop.get_x()
        if number_of_chromosome <= popsize:
            for i in range(popsize):
                if i < number_of_chromosome: #number_of_finished_chromosome_in_current_generation:
                    pop.set_xf(i, ad.solver.swarm_data[i][:7], ad.solver.swarm_data[i][-3:])
                else:
                    print(i, prob.get_fevals())
                    pop.set_x(i, pop_array[i]) # evaluate this guy
        else:
            # 新办法，直接从swarm_data.txt（相当于archive）中判断出当前最棒的群体
            swarm_data_on_pareto_front = learn_about_the_archive(ad.solver.swarm_data, popsize)
            # print(swarm_data_on_pareto_front)
            for i in range(popsize):
                pop.set_xf(i, swarm_data_on_pareto_front[i][:7], swarm_data_on_pareto_front[i][-3:])

            # if False:
                # # 老办法（蠢办法），依赖于swarm_survivor.txt
                # for i in range(popsize):
                #     pop.set_xf(i, ad.solver.survivor[i][:7], ad.solver.survivor[i][-3:])

                # # 小测试，该式子应该永远成立，如果不成立，说明分析完一代以后，write survivor没有被正常调用。
                # if ad.solver.survivor is not None:
                #     if ad.solver.survivor_title_number // popsize == number_of_finished_iterations:
                #         print('survivor_title_number is', ad.solver.survivor_title_number, 'number_of_chromosome is', number_of_chromosome)
                #         if ad.solver.survivor_title_number == number_of_chromosome:
                #             # 刚好swarm_data也是完整的一代
                #             print('\t刚刚好swarm_data也是完整的一代！！！')
                #     else:
                #         raise

                # # 手动把当前pop和swarm_data中的最新个体进行dominance比较
                #     # 经常出现的情况，survivor和swarm_data都存在，前者总是完整的一代，也就是说，
                #     # 搞到非初始化的某一代的中间的某个个体的时候断了，PyGMO不支持从中间搞起，那么只能我自己来根据swarm_data.txt中最后的数据来判断是否产生了值得留在种群中的个体了。
                # if ad.solver.survivor is not None and number_of_chromosome % popsize != 0: # number_of_finished_chromosome_in_current_generation < popsize: # recall if number_of_finished_chromosome_in_current_generation == 0, it is set to popsize.
                #     pop_array = pop.get_x()
                #     fits_array = pop.get_f()

                #     # number_of_finished_iterations = number_of_chromosome // popsize
                #     base = number_of_finished_iterations*popsize
                #     for i in range(number_of_chromosome % popsize):
                #         obj_challenger = ad.solver.swarm_data[i+base][-3:]
                #         if pg.pareto_dominance(obj_challenger, fits_array[i]):
                #             pop.set_xf(i, ad.solver.swarm_data[i+base][:7] , obj_challenger)
                #             print(i+base, '\t', obj_challenger, '\n\t', fits_array[i].tolist())
                #             print('\t', pop.get_x()[i], pop.get_f()[i].tolist())
                #         else:
                #             print(i+base)

        # 必须放到这个if的最后，因为在 learn_about_the_archive 中是有初始化一个 pop_archive 的，会调用fitness方法。
        ad.flag_do_not_evaluate_when_init_pop = False

    print('-'*40, '\nPop is initialized:\n', pop)
    # print('?????????')
    # raise KeyboardInterrupt

    # 初始化以后，pop.problem.get_fevals()就是popsize，但是如果大于popsize，说明“pop.set_x(i, pop_array[i]) # evaluate this guy”被调用了，说明还没输出过 survivors 数据，那么就写一下。
    if pop.problem.get_fevals() > popsize: 
        print('Write survivors.')
        ad.solver.write_swarm_survivor(pop, counter_fitness_return)

################################################################
# MOO Step 2:
#   Select algorithm (another option is pg.nsga2())
################################################################
    # Don't forget to change neighbours to be below popsize (default is 20) decomposition="bi"
    algo = pg.algorithm(pg.moead(gen=1, weight_generation="grid", decomposition="tchebycheff", 
                                 neighbours=20, 
                                 CR=1, F=0.5, eta_m=20, 
                                 realb=0.9, 
                                 limit=2, preserve_diversity=True)) # https://esa.github.io/pagmo2/docs/python/algorithms/py_algorithms.html#pygmo.moead
    print('-'*40, '\n', algo)

################################################################
# MOO Step 3:
#   Begin optimization
################################################################
    number_of_chromosome = ad.solver.read_swarm_data()
    number_of_finished_iterations = number_of_chromosome // popsize
    number_of_iterations = 50
    logger = logging.getLogger(__name__)
    try:
        for _ in range(number_of_finished_iterations, number_of_iterations):
            msg = 'This is iteration #%d. '%(_)
            print(msg)
            logger.info(msg)
            pop = algo.evolve(pop)

            msg += 'Write survivors to file. '
            ad.solver.write_swarm_survivor(pop, counter_fitness_return)

            hv = pg.hypervolume(pop)
            quality_measure = hv.compute(ref_point=[0.,0.,100.]) # ref_point must be dominated by the pop's pareto front
            msg += 'Quality measure by hyper-volume: %g'% (quality_measure)
            print(msg)
            logger.info(msg)
            
            my_print(pop, _)
            # my_plot(fits, vectors, ndf)
    except Exception as e:
        print(pop.get_x())
        print(pop.get_f().tolist())
        raise e

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Weighted objective function optimmization
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
else:
    # 微分进化的配置
    ad.get_de_config()
    print('Run: ' + run_folder + '\nThe auto generated bounds are:', ad.de_config_dict['original_bounds'])
    # quit()

    # 如果需要局部敏感性分析，那就先跑了再说
    if fea_config_dict['local_sensitivity_analysis'] == True:
        if not ad.check_results_of_local_sensitivity_analysis():
            ad.run_local_sensitivity_analysis(ad.de_config_dict['original_bounds'], design_denorm=None)
        else:
            ad.de_config_dict['bounds'] = ad.de_config_dict['original_bounds']
            ad.init_swarm() # define ad.sw
            ad.sw.generate_pop(specified_initial_design_denorm=None)
        ad.collect_results_of_local_sensitivity_analysis() # need sw to work
        fea_config_dict['local_sensitivity_analysis'] = False # turn off lsa mode

    # Build the final bounds
    if fea_config_dict['bool_refined_bounds'] == -1:
        ad.build_local_bounds_from_best_design(None)
    elif fea_config_dict['bool_refined_bounds'] == True:
        ad.build_refined_bounds(ad.de_config_dict['original_bounds'])
    elif fea_config_dict['bool_refined_bounds'] == False:
        ad.de_config_dict['bounds'] = ad.de_config_dict['original_bounds']
    else:
        raise
    print('The final bounds are:')
    for el in ad.de_config_dict['bounds']:
        print('\t', el)

    if False == bool_post_processing:
        if fea_config_dict['flag_optimization'] == True:
            ad.init_swarm() # define ad.sw
            ad.run_de()
        else:
            print('Do something.')

    elif True == bool_post_processing:
        ad.init_swarm()
        swda = ad.best_design_by_weights(fea_config_dict['use_weights'])
        from pylab import show
        show()
        quit()
        run_static_structural_fea(swda.best_design_denorm)


