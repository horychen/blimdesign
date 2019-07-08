# coding:u8
import shutil
import utility
from utility import my_execfile
import utility_moo
from win32com.client import pywintypes
bool_post_processing = False # solve or post-processing

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 0. FEA Setting / General Information & Packages Loading
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# FEA setting
my_execfile('./default_setting.py', g=globals(), l=locals())
fea_config_dict
fea_config_dict['Active_Qr'] = 16 # obsolete 
fea_config_dict['local_sensitivity_analysis'] = False
fea_config_dict['bool_refined_bounds'] = False
fea_config_dict['use_weights'] = 'O2' # this is not used
if 'Y730' in fea_config_dict['pc_name']:
    ################################################################
    # Y730
    ################################################################
    # run_folder = r'run#600/' # FRW constraint is removed and sleeve_length is 3 (not varying)
    # run_folder = r'run#601/' # FRW constraint is removed and sleeve_length is 2.5 (not varying)

    # # Combined winding PMSM
    # fea_config_dict['TORQUE_CURRENT_RATIO'] = 0.95
    # fea_config_dict['SUSPENSION_CURRENT_RATIO'] = 0.05
    # run_folder = r'run#603/' # FRW constraint is added and sleeve_length is 3 (not varying). Excitation ratio is 95%:5% between Torque and Suspension windings.

    # # Separate winding PMSM
    # fea_config_dict['TORQUE_CURRENT_RATIO'] = 0.60
    # fea_config_dict['SUSPENSION_CURRENT_RATIO'] = 0.05
    # run_folder = r'run#604/'
    raise
elif 'Severson01' in fea_config_dict['pc_name']:
    ################################################################
    # Severson01
    ################################################################
    print('Severson01')
    # Separate winding PMSM
    fea_config_dict['TORQUE_CURRENT_RATIO'] = 0.60
    fea_config_dict['SUSPENSION_CURRENT_RATIO'] = 0.05
    run_folder = r'run#604010/'

elif 'Severson02' in fea_config_dict['pc_name']:
    ################################################################
    # Severson02
    ################################################################
    print('Severson02')
    # Combined winding PMSM
    fea_config_dict['TORQUE_CURRENT_RATIO'] = 0.95
    fea_config_dict['SUSPENSION_CURRENT_RATIO'] = 0.05
    run_folder = r'run#603020/'
else:
    ################################################################
    # T440p
    ################################################################
    print('T440p')
    # Combined winding PMSM
    fea_config_dict['TORQUE_CURRENT_RATIO'] = 0.95
    fea_config_dict['SUSPENSION_CURRENT_RATIO'] = 0.05
    run_folder = r'run#603010/' # continued run from severson01

fea_config_dict['run_folder'] = run_folder

# spec's
my_execfile('./spec_TIA_ITEC_.py', g=globals(), l=locals())
spec.build_im_template(fea_config_dict)
spec.build_pmsm_template(fea_config_dict, im_template=spec.im_template)

# select motor type ehere
print('Build ACM template...')
spec.acm_template = spec.pmsm_template

import acm_designer
global ad
ad = acm_designer.acm_designer(fea_config_dict, spec)
ad.init_logger(prefix='bpmsm')

ad.bounds_denorm = spec.acm_template.get_classic_bounds(which_filter='FixedSleeveLength') # ad.get_classic_bounds()
ad.bound_filter  = spec.acm_template.bound_filter
print('---------------------\nBounds:')
idx_ad = 0
for idx, f in enumerate(ad.bound_filter):
    if f == True:
        print(idx, f, '[%g,%g]'%tuple(spec.acm_template.original_template_neighbor_bounds[idx]), '[%g,%g]'%tuple(ad.bounds_denorm[idx_ad]))
        idx_ad += 1
    else:
        print(idx, f, '[%g,%g]'%tuple(spec.acm_template.original_template_neighbor_bounds[idx]))
# quit()

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Optimization
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
import pygmo as pg
global counter_fitness_called, counter_fitness_return
def get_bad_fintess_values(machine_type=None, ref=False):
    if ref == False:
        if 'IM' in machine_type:
            return 0, 0, 99
        elif 'PMSM' in machine_type:
            return 9999, 0, 999
    else:
        if 'IM' in machine_type:
            return 1, 1, 100
        elif 'PMSM' in machine_type:
            return 10000, 1, 1000        
class Problem_BearinglessSynchronousDesign(object):

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
            raise Exception('counter_fitness_called')
        print('Call fitness: %d, %d'%(counter_fitness_called, counter_fitness_return))

        # 不要标幺化了！统一用真的bounds，见get_bounds()
        x_denorm = x

        # evaluate x_denorm via FEA tools
        counter_loop = 0
        stuck_at = 0
        while True:
            if stuck_at < counter_fitness_called:
                stuck_at = counter_fitness_called
                counter_loop = 0 # reset
            if stuck_at == counter_fitness_called:
                counter_loop += 1

            try:
                cost_function, f1, f2, f3, FRW, \
                normalized_torque_ripple, \
                normalized_force_error_magnitude, \
                force_error_angle = \
                    ad.evaluate_design(ad.spec.acm_template, x_denorm, counter_fitness_called, counter_loop=counter_loop)

                # remove folder .jfiles to save space (we have to generate it first in JMAG Designer to have field data and voltage profiles)
                if ad.solver.folder_to_be_deleted is not None:
                    try:
                        shutil.rmtree(ad.solver.folder_to_be_deleted) # .jfiles directory
                    except PermissionError as error:
                        print(error)
                        print('Skip deleting this folder...')
                # update to be deleted when JMAG releases the use
                ad.solver.folder_to_be_deleted = ad.solver.expected_project_file[:-5]+'jfiles'

            except utility.ExceptionBadNumberOfParts as error:
                print(str(error)) 
                print("Detail: {}".format(error.payload))
                f1, f2, f3 = get_bad_fintess_values(machine_type='PMSM')
                utility.send_notification(ad.solver.fea_config_dict['pc_name'] + '\n\nExceptionBadNumberOfParts:' + str(error) + '\n'*3 + "Detail: {}".format(error.payload))
                break

            except (utility.ExceptionReTry, pywintypes.com_error) as error:
                print(error)

                msg = 'FEA tool failed for individual #%d: attemp #%d.'%(counter_fitness_called, counter_loop)
                logger = logging.getLogger(__name__)
                logger.error(msg)
                print(msg)

                if counter_loop > 2:
                    print(error)
                    raise Exception('Abort the optimization. Three attemps to evaluate the design have all failed for individual #%d'%(counter_fitness_called))
                else:
                    continue

            except AttributeError as error:
                print(str(error)) 
                print("Detail: {}".format(error.payload))

                msg = 'FEA tool failed for individual #%d: attemp #%d.'%(counter_fitness_called, counter_loop)
                logger = logging.getLogger(__name__)
                logger.error(msg)
                print(msg)

                if 'designer.Application' in str(error):
                    if counter_loop > 2:
                        print(error)
                        raise Exception('Abort the optimization. Three attemps to evaluate the design have all failed for individual #%d'%(counter_fitness_called))
                    else:
                        continue
                else:
                    raise error

            except Exception as e: # raise and need human inspection

                print('-'*40 + 'Unexpected error is caught.')
                print(str(e)) 
                utility.send_notification(ad.solver.fea_config_dict['pc_name'] + '\n\nUnexpected expection:' + str(e))
                raise e

            else:
                break


        # - Price
        f1 
        # - Efficiency @ Rated Power
        f2 
        # Ripple Performance (Weighted Sum)
        f3 
        print('f1,f2,f3:',f1,f2,f3)

        # Constraints (Em<0.2 and Ea<10 deg):
        # if abs(normalized_torque_ripple)>=0.2 or abs(normalized_force_error_magnitude) >= 0.2 or abs(force_error_angle) > 10 or SafetyFactor < 1.5:
        # if abs(normalized_torque_ripple)>=0.2 or abs(normalized_force_error_magnitude) >= 0.2 or abs(force_error_angle) > 10 or FRW < 1:
        # if abs(normalized_torque_ripple)>=0.2 or abs(normalized_force_error_magnitude) >= 0.2 or abs(force_error_angle) > 10:
        if abs(normalized_torque_ripple)>=0.3 or abs(normalized_force_error_magnitude) >= 0.3 or abs(force_error_angle) > 10 or FRW < 0.75:
            print('Constraints are violated:')
            if abs(normalized_torque_ripple)>=0.3:
                print('\tabs(normalized_torque_ripple)>=0.3')
            if abs(normalized_force_error_magnitude) >= 0.3:
                print('\tabs(normalized_force_error_magnitude) >= 0.3')
            if abs(force_error_angle) > 10:
                print('\tabs(force_error_angle) > 10')
            if FRW < 0.75:
                print('\tFRW < 0.75')
            f1, f2, f3 = get_bad_fintess_values(machine_type='PMSM')
        print('f1,f2,f3:',f1,f2,f3)

        counter_fitness_return += 1
        print('Fitness: %d, %d\n----------------'%(counter_fitness_called, counter_fitness_return))
        # raise KeyboardInterrupt
        return [f1, f2, f3]

    # Return number of objectives
    def get_nobj(self):
        return 3

    # Return bounds of decision variables (a.k.a. chromosome)
    def get_bounds(self):
        print('Problem_BearinglessSynchronousDesign.get_bounds:', ad.bounds_denorm)
        min_b, max_b = np.asarray(ad.bounds_denorm).T 
        return ( min_b.tolist(), max_b.tolist() )

    # Return function name
    def get_name(self):
        return "Bearingless PMSM Design"

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Multi-Objective Optimization
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
if True:
################################################################
# MOO Step 1:
#   Create UserDefinedProblem and create population
#   The magic method __init__ cannot be fined for UDP class
################################################################
    udp = Problem_BearinglessSynchronousDesign()
    counter_fitness_called, counter_fitness_return = 0, 0
    prob = pg.problem(udp)

    popsize = 78
    print('-'*40 + '\nPop size is', popsize)

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Add Restarting Feature
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # 检查swarm_data.txt，如果有至少一个数据，返回就不是None。
    print('Check swarm_data.txt...')
    number_of_chromosome = ad.solver.read_swarm_data(ad.bound_filter)
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

            if len(ad.solver.survivor) != popsize:
                print('popsize is reduced') # 如果popsize增大了，read_swarm_survivor(popsize)就会报错了，因为-----不能被split后转为float
                raise Exception('This is a feature not tested. However, you can cheat to change popsize by manually modify swarm_data.txt or swarm_survivor.txt.')
        else:
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
                    pop.set_xf(i, ad.solver.swarm_data[i][:-3], ad.solver.swarm_data[i][-3:])
                else:
                    print('Set "ad.flag_do_not_evaluate_when_init_pop" to False...')
                    ad.flag_do_not_evaluate_when_init_pop = False
                    print('Calling pop.set_x()---this is a restart for individual#%d during pop initialization.'%(i))
                    print(i, 'get_fevals:', prob.get_fevals())
                    pop.set_x(i, pop_array[i]) # evaluate this guy
        else:
            # 新办法，直接从swarm_data.txt（相当于archive）中判断出当前最棒的群体
            swarm_data_on_pareto_front = utility_moo.learn_about_the_archive(prob, ad.solver.swarm_data, popsize, fea_config_dict)
            # print(swarm_data_on_pareto_front)
            for i in range(popsize):
                pop.set_xf(i, swarm_data_on_pareto_front[i][:-3], swarm_data_on_pareto_front[i][-3:])

        # 必须放到这个if的最后，因为在 learn_about_the_archive 中是有初始化一个 pop_archive 的，会调用fitness方法。
        ad.flag_do_not_evaluate_when_init_pop = False

    print('-'*40, '\nPop is initialized:\n', pop)
    hv = pg.hypervolume(pop)
    quality_measure = hv.compute(ref_point=get_bad_fintess_values(machine_type='PMSM', ref=True)) # ref_point must be dominated by the pop's pareto front
    print('quality_measure: %g'%(quality_measure))
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
    # quit()

################################################################
# MOO Step 3:
#   Begin optimization
################################################################
    number_of_chromosome = ad.solver.read_swarm_data(ad.bound_filter)
    number_of_finished_iterations = number_of_chromosome // popsize
    number_of_iterations = 50
    logger = logging.getLogger(__name__)
    # try:
    if True:
        for _ in range(number_of_finished_iterations, number_of_iterations):
            msg = 'This is iteration #%d. '%(_)
            print(msg)
            logger.info(msg)
            pop = algo.evolve(pop)

            msg += 'Write survivors to file. '
            ad.solver.write_swarm_survivor(pop, counter_fitness_return)

            hv = pg.hypervolume(pop)
            quality_measure = hv.compute(ref_point=get_bad_fintess_values(machine_type='PMSM', ref=True)) # ref_point must be dominated by the pop's pareto front
            msg += 'Quality measure by hyper-volume: %g'% (quality_measure)
            print(msg)
            logger.info(msg)
            
            utility_moo.my_print(ad, pop, _)
            # my_plot(fits, vectors, ndf)
    # except Exception as e:
    #     print(pop.get_x())
    #     print(pop.get_f().tolist())
    #     raise e
