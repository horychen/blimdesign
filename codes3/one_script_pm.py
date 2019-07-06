# coding:u8
import shutil
import utility
from utility import my_execfile
bool_post_processing = False # solve or post-processing

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 0. FEA Setting / General Information & Packages Loading
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
my_execfile('./default_setting.py', g=globals(), l=locals())
fea_config_dict
fea_config_dict['Active_Qr'] = 16
if True:
    my_execfile('./spec_TIA_ITEC_.py', g=globals(), l=locals())
    spec.build_im_template(fea_config_dict)
    spec.build_pmsm_template(fea_config_dict, im_template=spec.im_template)
    print('Build ACM template...')
    # select motor type ehere
    spec.acm_template = spec.pmsm_template


    fea_config_dict['local_sensitivity_analysis'] = False
    fea_config_dict['bool_refined_bounds'] = False
    fea_config_dict['use_weights'] = 'O2' # this is not used
    run_folder = r'run#600/'
else:
    pass
fea_config_dict['run_folder'] = run_folder

import acm_designer
global ad
ad = acm_designer.acm_designer(fea_config_dict, spec)
ad.init_logger(prefix='bpmsm')



ad.bounds_denorm = ad.get_classic_bounds()
print('classic_bounds and original_bounds')
for A, B in zip(ad.bounds_denorm, ad.original_bounds):
    print(A, B)
# quit()

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Optimization
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
import pygmo as pg
global counter_fitness_called, counter_fitness_return
def get_bad_fintess_values():
    return 0, 0, 99
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
                if counter_loop > 5:
                    raise Exception('Abort the optimization. Five attemps to evaluate the design have all failed for individual #%d'%(counter_fitness_called))

            try:
                cost_function, f1, f2, f3, \
                normalized_torque_ripple, \
                normalized_force_error_magnitude, \
                force_error_angle = \
                    ad.evaluate_design(ad.spec.acm_template, x_denorm, counter_fitness_called, counter_loop=counter_loop)

                # remove folder .jfiles to save space (we have to generate it first in JMAG Designer to have field data and voltage profiles)
                if ad.solver.folder_to_be_deleted is not None:
                    shutil.rmtree(ad.solver.folder_to_be_deleted) # .jfiles directory
                # update to be deleted when JMAG releases the use
                ad.solver.folder_to_be_deleted = ad.solver.expected_project_file[:-5]+'jfiles'

            except Exception as e: # debugging

                print(e)
                raise e

            except utility.ExceptionBadNumberOfParts as error:
                print(str(error)) 
                print("Detail: {}".format(error.payload))
                f1, f2, f3 = get_bad_fintess_values()
                utility.send_notification(ad.solver.fea_config_dict['pc_name'] + '\n\nExceptionBadNumberOfParts:' + str(error) + '\n'*3 + "Detail: {}".format(error.payload))
                break

            except Exception as e: # retry

                print(e)
                raise e

                msg = 'FEA tool failed for individual #%d: attemp #%d.'%(counter_fitness_called, counter_loop)
                logger = logging.getLogger(__name__)
                logger.error(msg)
                print(msg)

                msg = 'Removing all files for individual #%d and try again...'%(counter_fitness_called)
                logger.error(msg)
                print(msg)

                try:
                        # turn off JMAG Designer
                        # try:
                        #     ad.solver.app.Quit()
                        # except:
                        #     print('I think there is no need to Quit the app')
                    ad.solver.app = None

                    # JMAG files
                    # os.remove(ad.solver.expected_project_file) # .jproj
                    # shutil.rmtree(ad.solver.expected_project_file[:-5]+'jfiles') # .jfiles directory # .jplot file in this folder will be used by JSOL softwares even JMAG Designer is closed.

                except Exception as e2:
                    utility.send_notification(ad.solver.fea_config_dict['pc_name'] + '\n\nException 1:' + str(e) + '\n'*3 + 'Exception 2:' + str(e2))
                    raise e2

                else:
                    if 'Number of Parts is unexpected' in str(e):
                        print('Shitty design is found as:\n'+str(e))
                        print('\nEmail has been sent.\nThis design is punished by specifying f1=0, f2=0, f3=99.')
                        f1, f2, f3 = get_bad_fintess_values()

                    utility.send_notification(ad.solver.fea_config_dict['pc_name'] + '\n\nException 1:' + str(e))
                    print('This is Obselete can will not be reached anymore. An exclusive exception is built for number of parts unexpected exception.')
                    break
            else:
                break

        # - Price
        f1 
        # - Efficiency @ Rated Power
        f2 
        # Ripple Performance (Weighted Sum)
        f3 
        print(f1,f2,f3)

        # Constraints (Em<0.2 and Ea<10 deg):
        # if abs(normalized_torque_ripple)>=0.2 or abs(normalized_force_error_magnitude) >= 0.2 or abs(force_error_angle) > 10 or SafetyFactor < 1.5:
        # if abs(normalized_torque_ripple)>=0.2 or abs(normalized_force_error_magnitude) >= 0.2 or abs(force_error_angle) > 10 or FRW < 1:
        if abs(normalized_torque_ripple)>=0.2 or abs(normalized_force_error_magnitude) >= 0.2 or abs(force_error_angle) > 10:
            f1 = 0
            f2 = 0

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
                    pop.set_xf(i, ad.solver.swarm_data[i][:7], ad.solver.swarm_data[i][-3:])
                else:
                    print('Set ad.flag_do_not_evaluate_when_init_pop to False...')
                    ad.flag_do_not_evaluate_when_init_pop = False
                    print('Calling pop.set_x()---this is a restart for individual#%d during pop initialization.'%(i))
                    print(i, 'get_fevals:', prob.get_fevals())
                    pop.set_x(i, pop_array[i]) # evaluate this guy
        else:
            # 新办法，直接从swarm_data.txt（相当于archive）中判断出当前最棒的群体
            swarm_data_on_pareto_front = learn_about_the_archive(prob, ad.solver.swarm_data, popsize)
            # print(swarm_data_on_pareto_front)
            for i in range(popsize):
                pop.set_xf(i, swarm_data_on_pareto_front[i][:7], swarm_data_on_pareto_front[i][-3:])

        # 必须放到这个if的最后，因为在 learn_about_the_archive 中是有初始化一个 pop_archive 的，会调用fitness方法。
        ad.flag_do_not_evaluate_when_init_pop = False

    print('-'*40, '\nPop is initialized:\n', pop)
    hv = pg.hypervolume(pop)
    quality_measure = hv.compute(ref_point=[0.,0.,100.]) # ref_point must be dominated by the pop's pareto front
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