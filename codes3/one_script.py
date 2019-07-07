# https://scipy-lectures.org/packages/3d_plotting/index.html#figure-management

# coding:u8
import shutil
from utility import my_execfile
bool_post_processing = True # solve or post-processing

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
    fea_config_dict['use_weights'] = 'O2' # this is not working

    # run_folder = r'run#535/' # test with pygmo
    # run_folder = r'run#537/' # test with pygmo
    run_folder = r'run#538/' # test with pygmo with new fitness and constraints, Rotor current density Jr is 6.5 Arms/mm^2

    # run_folder = r'run#539/' # New copper loss formula from Bolognani 2006
    run_folder = r'run#540/' # New copper loss formula from Bolognani 2006 # Fix small bugs


    # fea_config_dict['local_sensitivity_analysis'] = True

    # # fea_config_dict['local_sensitivity_analysis_number_of_variants'] = 3 # =2 would waste 1/3 of pop to evaluate the same reference design
    # # fea_config_dict['local_sensitivity_analysis_percent'] = 0.05
    # # run_folder = r'run#5409/' # LSA of high torque density design

    # fea_config_dict['local_sensitivity_analysis_number_of_variants'] = 19 # =20 is also bad idea!
    # fea_config_dict['local_sensitivity_analysis_percent'] = 0.2
    # # run_folder = r'run#54099/' # LSA of high torque density design
    # # run_folder = r'run#54088/' # LSA of high efficiency design
    # run_folder = r'run#54077/' # LSA of low ripple performance design
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
    # run_folder = r'run#538021/'
    run_folder = r'run#540021/'

    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # Severson01
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # my_execfile('./spec_Prototype4poleOD150mm1000Hz_SpecifyTipSpeed.py', g=globals(), l=locals()) # define spec
    my_execfile('./spec_ECCE_4pole32Qr1000Hz.py', g=globals(), l=locals())
    fea_config_dict['local_sensitivity_analysis'] = False
    fea_config_dict['bool_refined_bounds'] = False
    fea_config_dict['use_weights'] = 'O2'
    # run_folder = r'run#538011/'
    run_folder = r'run#540011/'

fea_config_dict['run_folder'] = run_folder
fea_config_dict['Active_Qr'] = spec.Qr
fea_config_dict['use_drop_shape_rotor_bar'] = spec.use_drop_shape_rotor_bar
build_model_name_prefix(fea_config_dict) # rebuild the model name for fea_config_dict

import acm_designer
global ad
ad = acm_designer.acm_designer(fea_config_dict, spec)
# if 'Y730' in fea_config_dict['pc_name']:
#     ad.build_oneReport() # require LaTeX
#     ad.talk_to_mysql_database() # require MySQL
#     quit()
ad.init_logger()

ad.bounds_denorm = ad.get_classic_bounds()
print('classic_bounds and original_bounds')
for A, B in zip(ad.bounds_denorm, ad.original_bounds):
    print(A, B)
# quit()

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Optimization
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
def get_bad_fintess_values(machine_type='IM'):
    if 'IM' in machine_type:
        return 0, 0, 99
    elif 'PMSM' in machine_type:
        return 99999999999999999, 0, 99
import pygmo as pg
global counter_fitness_called, counter_fitness_return
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
                    ad.evaluate_design(ad.spec.im_template, x_denorm, counter_fitness_called, counter_loop=counter_loop)

                # remove folder .jfiles to save space (we have to generate it first in JMAG Designer to have field data and voltage profiles)
                if ad.solver.folder_to_be_deleted is not None:
                    shutil.rmtree(ad.solver.folder_to_be_deleted) # .jfiles directory
                # update to be deleted when JMAG releases the use
                ad.solver.folder_to_be_deleted = ad.solver.expected_project_file[:-5]+'jfiles'

            except utility.ExceptionBadNumberOfParts as error:
                print(str(error)) 
                print("Detail: {}".format(error.payload))
                f1, f2, f3 = get_bad_fintess_values()
                utility.send_notification(ad.solver.fea_config_dict['pc_name'] + '\n\nExceptionBadNumberOfParts:' + str(error) + '\n'*3 + "Detail: {}".format(error.payload))
                break

            except Exception as e: # retry

                # raise e
                print(e)

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

                    # FEMM files
                    if os.path.exists(ad.solver.femm_output_file_path):
                        os.remove(ad.solver.femm_output_file_path) # .csv
                    if os.path.exists(ad.solver.femm_output_file_path[:-3]+'fem'):
                        os.remove(ad.solver.femm_output_file_path[:-3]+'fem') # .fem
                    for file in os.listdir(ad.solver.dir_femm_temp):
                        if 'femm_temp_' in file or 'femm_found' in file:
                            os.remove(ad.solver.dir_femm_temp + file)

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

        # - Torque per Rotor Volume
        f1 #= - ad.spec.required_torque / rated_rotor_volume
        # - Efficiency @ Rated Power
        f2 #= - rated_efficiency
        # Ripple Performance (Weighted Sum)
        f3 #= sum(list_weighted_ripples)

        # Constraints (Em<0.2 and Ea<10 deg):
        if abs(normalized_torque_ripple)>=0.2 or abs(normalized_force_error_magnitude) >= 0.2 or abs(force_error_angle) > 10:
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
        # return ([0]*7, [1]*7)

    # Return function name
    def get_name(self):
        return "Bearingless Induction Motor Design"

def my_plot_non_dominated_fronts(points, marker='o', comp=[0, 1], up_to_rank_no=None):
    # We plot
    fronts, _, _, _ = pg.fast_non_dominated_sorting(points)

    # We define the colors of the fronts (grayscale from black to white)
    if up_to_rank_no is None:
        cl = list(zip(np.linspace(0.1, 0.9, len(fronts)),
                      np.linspace(0.1, 0.9, len(fronts)),
                      np.linspace(0.1, 0.9, len(fronts))))
    else:
        cl = list(zip(np.linspace(0.1, 0.9, up_to_rank_no),
                      np.linspace(0.1, 0.9, up_to_rank_no),
                      np.linspace(0.1, 0.9, up_to_rank_no)))

    fig, ax = plt.subplots()

    count = 0
    for ndr, front in enumerate(fronts):
        count += 1
        # We plot the points
        for idx in front:
            ax.plot(points[idx][comp[0]], points[idx][
                comp[1]], marker=marker, color=cl[ndr])
        # We plot the fronts
        # Frist compute the points coordinates
        x = [points[idx][comp[0]] for idx in front]
        y = [points[idx][comp[1]] for idx in front]
        # Then sort them by the first objective
        tmp = [(a, b) for a, b in zip(x, y)]
        tmp = sorted(tmp, key=lambda k: k[0])
        # Now plot using step
        ax.step([c[0] for c in tmp], [c[1]
                                      for c in tmp], color=cl[ndr], where='post')
        if up_to_rank_no is None:
            pass
        else:
            if count >= up_to_rank_no:
                break

    return ax
def my_2p5d_plot_non_dominated_fronts(points, marker='o', comp=[0, 1], up_to_rank_no=1, no_text=True):
    # from pylab import mpl
    # mpl.rcParams['font.family'] = ['Times New Roman']
    # mpl.rcParams['font.size'] = 16.0

    full_comp = [0, 1, 2]
    full_comp.remove(comp[0])
    full_comp.remove(comp[1])
    z_comp = full_comp[0]

    # We plot
    # fronts, dl, dc, ndr = pg.fast_non_dominated_sorting(points)
    fronts, _, _, _= pg.fast_non_dominated_sorting(points)

    # We define the colors of the fronts (grayscale from black to white)
    cl = list(zip(np.linspace(0.9, 0.1, len(fronts)),
                  np.linspace(0.9, 0.1, len(fronts)),
                  np.linspace(0.9, 0.1, len(fronts))))

    fig, ax = plt.subplots(constrained_layout=False)
    plt.subplots_adjust(left=None, bottom=None, right=0.85, top=None, wspace=None, hspace=None)

    count = 0
    for ndr, front in enumerate(fronts):
        count += 1

        # Frist compute the points coordinates
        x_scale = 1
        y_scale = 1
        z_scale = 1
        if comp[0] == 1: # efficency
            x_scale = 100
        if comp[1] == 1: # efficency
            y_scale = 100
        if z_comp == 1: # efficency
            z_scale = 100
        x = [points[idx][comp[0]]*x_scale for idx in front]
        y = [points[idx][comp[1]]*y_scale for idx in front]
        z = [points[idx][z_comp] *z_scale for idx in front]

        # # We plot the points
        # for idx in front:
        #     ax.plot(points[idx][comp[0]], points[idx][comp[1]], marker=marker, color=cl[ndr])

        # Then sort them by the first objective
        tmp = [(a, b, c) for a, b, c in zip(x, y, z)]
        tmp = sorted(tmp, key=lambda k: k[0])
        # Now plot using step
        ax.step([coords[0] for coords in tmp], 
                [coords[1] for coords in tmp], color=cl[ndr], where='post')

        # Now add color according to the value of the z-axis variable usign scatter
        scatter_handle = ax.scatter(x, y, c=z, alpha=0.5, cmap='Spectral', marker=marker, zorder=99) #'viridis'
        # color bar
        cbar_ax = fig.add_axes([0.875, 0.15, 0.02, 0.7])
        cbar_ax.get_yaxis().labelpad = 10
        clb = fig.colorbar(scatter_handle, cax=cbar_ax)
        if z_comp == 0:
            z_label = r'$-\rm {TRV}$ [Nm/m^3]'
            z_text  = '%.0f'
        elif z_comp == 1:
            z_label = r'$-\eta$ [%]'
            z_text  = '%.1f'
        elif z_comp == 2:
            z_label = r'$O_C$ [1]'
            z_text  = '%.1f'
        clb.ax.set_ylabel(z_label, rotation=270)


        if z_comp == 2: # when OC as z-axis
            print('-----------------------------------------------------')
            print('-----------------------------------------------------')
            print('-----------------------------------------------------')
            # Add index next to the points
            for x_coord, y_coord, z_coord, idx in zip(x, y, z, front):
                if no_text:
                    pass
                else:
                    ax.annotate( z_text%(z_coord) + ' #%d'%(idx), (x_coord, y_coord) )
        else:
            # text next scatter showing the value of the 3rd objective
            for i, val in enumerate(z):
                if no_text:
                    pass
                else:
                    ax.annotate( z_text%(val), (x[i], y[i]) )

        # refine the plotting
        if comp[0] == 0:
            ax.set_xlabel(r'$-\rm {TRV}$ [Nm/m^3]')
        elif comp[0] == 1:
            ax.set_xlabel(r'$-\eta$ [%]')
        elif comp[0] == 2:
            ax.set_xlabel(r'$O_C$ [1]')

        if comp[1] == 0:
            ax.set_ylabel(r'$-\rm {TRV}$ [Nm/m^3]')
        elif comp[1] == 1:
            ax.set_ylabel(r'$-\eta$ [%]')
        elif comp[1] == 2:
            ax.set_ylabel(r'$O_C$ [1]')
        ax.grid()

        # plot up to which domination rank?
        if up_to_rank_no is None:
            pass
        else:
            if count >= up_to_rank_no:
                break
    fig.savefig(r'C:\Users\horyc\Desktop/'+ '2p5D-%d%d.png'%(comp[0],comp[1]), dpi=300)
    return ax
def my_3d_plot_non_dominated_fronts(pop, paretoPoints, az=180, comp=[0, 1, 2], plot_option=1):
    """
    Plots solutions to the DTLZ problems in three dimensions. The Pareto Front is also
    visualized if the problem id is 2,3 or 4.
    Args:
        pop (:class:`~pygmo.population`): population of solutions to a dtlz problem
        az (``float``): angle of view on which the 3d-plot is created
        comp (``list``): indexes the fitness dimension for x,y and z axis in that order
    Returns:
        ``matplotlib.axes.Axes``: the current ``matplotlib.axes.Axes`` instance on the current figure
    Raises:
        ValueError: if *pop* does not contain a DTLZ problem (veryfied by its name only) or if *comp* is not of length 3
    Examples:
        >>> import pygmo as pg
        >>> udp = pg.dtlz(prob_id = 1, fdim =3, dim = 5)
        >>> pop = pg.population(udp, 40)
        >>> udp.plot(pop) # doctest: +SKIP
    """
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    import numpy as np
    # from pylab import mpl
    # mpl.rcParams['font.family'] = ['Times New Roman']
    # mpl.rcParams['font.size'] = 16.0

    # if (pop.problem.get_name()[:-1] != "DTLZ"):
    #     raise(ValueError, "The problem seems not to be from the DTLZ suite")

    if (len(comp) != 3):
        raise(ValueError, "The kwarg *comp* needs to contain exactly 3 elements (ids for the x,y and z axis)")

    # Create a new figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    # plot the points
    fits = np.transpose(pop.get_f())
    try:
        pass
        # ax.plot(fits[comp[0]], fits[comp[1]], fits[comp[2]], 'ro')
    except IndexError:
        print('Error. Please choose correct fitness dimensions for printing!')

    if False:
        # Plot pareto front for dtlz 1
        if plot_option==1: # (pop.problem.get_name()[-1] in ["1"]):

            X, Y = np.meshgrid(np.linspace(0, 0.5, 100), np.linspace(0, 0.5, 100))
            Z = - X - Y + 0.5
            # remove points not in the simplex
            for i in range(100):
                for j in range(100):
                    if X[i, j] < 0 or Y[i, j] < 0 or Z[i, j] < 0:
                        Z[i, j] = float('nan')

            ax.set_xlim(0, 1.)
            ax.set_ylim(0, 1.)
            ax.set_zlim(0, 1.)

            ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
            plt.plot([0, 0.5], [0.5, 0], [0, 0])

        # Plot pareto fronts for dtlz 2,3,4
        if plot_option == 2: # (pop.problem.get_name()[-1] in ["2", "3", "4"]):
            # plot the wireframe of the known optimal pareto front
            thetas = np.linspace(0, (np.pi / 2.0), 30)
            # gammas = np.linspace(-np.pi / 4, np.pi / 4, 30)
            gammas = np.linspace(0, (np.pi / 2.0), 30)

            x_frame = np.outer(np.cos(thetas), np.cos(gammas))
            y_frame = np.outer(np.cos(thetas), np.sin(gammas))
            z_frame = np.outer(np.sin(thetas), np.ones(np.size(gammas)))

            ax.set_autoscalex_on(False)
            ax.set_autoscaley_on(False)
            ax.set_autoscalez_on(False)

            ax.set_xlim(0, 1.8)
            ax.set_ylim(0, 1.8)
            ax.set_zlim(0, 1.8)

            ax.plot_wireframe(x_frame, y_frame, z_frame)

    # https://stackoverflow.com/questions/37000488/how-to-plot-multi-objectives-pareto-frontier-with-deap-in-python
    # def simple_cull(inputPoints, dominates):
    #     paretoPoints = set()
    #     candidateRowNr = 0
    #     dominatedPoints = set()
    #     while True:
    #         candidateRow = inputPoints[candidateRowNr]
    #         inputPoints.remove(candidateRow)
    #         rowNr = 0
    #         nonDominated = True
    #         while len(inputPoints) != 0 and rowNr < len(inputPoints):
    #             row = inputPoints[rowNr]
    #             if dominates(candidateRow, row):
    #                 # If it is worse on all features remove the row from the array
    #                 inputPoints.remove(row)
    #                 dominatedPoints.add(tuple(row))
    #             elif dominates(row, candidateRow):
    #                 nonDominated = False
    #                 dominatedPoints.add(tuple(candidateRow))
    #                 rowNr += 1
    #             else:
    #                 rowNr += 1

    #         if nonDominated:
    #             # add the non-dominated point to the Pareto frontier
    #             paretoPoints.add(tuple(candidateRow))

    #         if len(inputPoints) == 0:
    #             break
    #     return paretoPoints, dominatedPoints
    # def dominates(row, candidateRow):
    #     return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row)  
    # import random
    # print(inputPoints)
    # inputPoints = [[random.randint(70,100) for i in range(3)] for j in range(500)]
    # print(inputPoints)
    # quit()
    # inputPoints = [(x,y,z) for x,y,z in zip(fits[comp[0]], fits[comp[1]], fits[comp[2]])]
    # paretoPoints, dominatedPoints = simple_cull(inputPoints, dominates)
    x = [coords[0]/1000 for coords in paretoPoints]
    y = [coords[1] for coords in paretoPoints]
    z = [coords[2] for coords in paretoPoints]

    # from surface_fitting import surface_fitting
    # surface_fitting(x,y,z)
    # quit()

    if False:
        pass
    else:
        import pandas as pd
        from pylab import cm
        print(dir(cm))
        df = pd.DataFrame({'x': x, 'y': y, 'z': z})

        # 只有plot_trisurf这一个函数，输入是三个以为序列的，其他都要meshgrid得到二维数组的(即ndim=2的数组) 
        # # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
            # surf = ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.magma, linewidth=0.1, edgecolor='none')
            # surf = ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
        # surf = ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.magma, linewidth=0.1)
        surf = ax.plot_trisurf(df.x, df.y*100, df.z, cmap=cm.Spectral, linewidth=0.1)        

        with open('./%s_PF_points.txt'%(fea_config_dict['run_folder'][:-1]), 'w') as f:
            f.write('TRV,eta,OC\n')
            f.writelines(['%g,%g,%g\n'%(a,b,c) for a,b,c in zip(df.x, df.y*100, df.z)])
        quit()

        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_xlabel(' \n$\\rm -TRV$ [$\\rm kNm/m^3$]')
        ax.set_ylabel(' \n$-\\eta$ [%]')
        ax.set_yticks(np.arange(-96,-93.5,0.5))
        ax.set_zlabel(r'$O_C$ [1]')

        # Try to export data from plot_trisurf # https://github.com/WoLpH/numpy-stl/issues/19
        # print(surf.get_vector())

        # plt.savefig('./plots/avgErrs_vs_C_andgamma_type_%s.png'%(k))
        # plt.show()

        # # rotate the axes and update
        # for angle in range(0, 360):
        #     ax.view_init(30, angle)
        #     plt.draw()
        #     plt.pause(.001)

    ax.view_init(azim=245, elev=15) 
    # fig.tight_layout()
    fig.savefig(r'C:\Users\horyc\Desktop/3D-plot.png', dpi=300, layout='tight')
    # ax.view_init(azim=az)
    # ax.set_xlim(0, 1.)
    # ax.set_ylim(0, 1.)
    # ax.set_zlim(0, 10.)
    return ax
def my_plot(fits, vectors, ndf):
    plt.rcParams['mathtext.fontset'] = 'stix' # 'cm'
    plt.rcParams["font.family"] = "Times New Roman"
    if True:
        # for fit in fits:
        #     print(fit)
        # ax = pg.plot_non_dominated_fronts(fits)

        # ax = my_plot_non_dominated_fronts(fits, comp=[0,1], marker='o', up_to_rank_no=3)        
        # ax = my_plot_non_dominated_fronts(fits, comp=[0,2], marker='o', up_to_rank_no=3)
        # ax = my_plot_non_dominated_fronts(fits, comp=[1,2], marker='o', up_to_rank_no=3)

        pass

        ax = my_2p5d_plot_non_dominated_fronts(fits, comp=[0,1], marker='o', up_to_rank_no=1)
        # # for studying LSA population (whether or not the optimal is on Rank 1 Pareto Front)
        # x = fits[0][0]/1e3
        # y = fits[0][1]
        # z = fits[0][2]
        # ax.plot(x, y, color='k', marker='s')
        # ax.annotate(r'$x_{\rm optm}$', xy=(x, y), xytext=(x+1, y+0.0005), arrowprops=dict(facecolor='black', shrink=0.05),)
        ax = my_2p5d_plot_non_dominated_fronts(fits, comp=[0,2], marker='o', up_to_rank_no=1)
        ax = my_2p5d_plot_non_dominated_fronts(fits, comp=[1,2], marker='o', up_to_rank_no=1)

    else:
        # Obselete. Use ax.step instead to plot
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
        index = 0
        for domination_list, domination_count, non_domination_rank in zip(dl, dc, ndr):
            print('Individual #%d\t'%(index), 'Belong to Rank #%d\t'%(non_domination_rank), 'Dominating', domination_count, 'and they are', domination_list, file=fname)
            index += 1

        # print(fits, vectors, ndf)
        print(pop, file=fname)


def learn_about_the_archive(prob, swarm_data, popsize, len_s01=None, len_s02=None):
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
    my_3d_plot_non_dominated_fronts(pop_archive, rank1_ParetoPoints, plot_option=1)
    plt.show()

    swarm_data_on_pareto_front = [design_parameters_denorm + fits for design_parameters_denorm, fits in zip(sorted_vectors, sorted_fits)]
    return swarm_data_on_pareto_front

def pyx_draw_model(im):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"

    myfontsize = 13.5
    plt.rcParams.update({'font.size': myfontsize})

    # # 示意图而已，改改尺寸吧
    # im.Radius_OuterStatorYoke -= 37
    # im.Radius_InnerStatorYoke -= 20
    # im.Radius_Shaft += 20
    # # im.Location_RotorBarCenter2 += 5 # this will change the shape of rotor slot

    import VanGogh
    vg = VanGogh.VanGogh_pyPlotter(im, VanGogh.CUSTOM)
    vg.draw_model()

    # PyX
    import pyx
    vg.tikz.c = pyx.canvas.canvas() # clear the canvas because we want to redraw 90 deg with the data vg.tikz.track_path
    from copy import deepcopy
    def pyx_draw_path(vg, path, sign=1):
        if len(path) == 4:
            vg.tikz.draw_line(path[:2], path[2:4], untrack=True)
        else:
            vg.tikz.draw_arc(path[:2], path[2:4], path[4:6], relangle=sign*path[6], untrack=True)
    def rotate(_, x, y):
        return np.cos(_)*x + np.sin(_)*y, -np.sin(_)*x + np.cos(_)*y
    def is_at_stator(im, path):
        return np.sqrt(path[0]**2 + path[1]**2) > im.Radius_OuterRotor + 0.5*im.Length_AirGap
    
    for path in (vg.tikz.track_path): # track_path is passed by reference and is changed by mirror
        path_mirror = deepcopy(path)
        # for mirror copy (along x-axis)
        path_mirror[1] = path[1]*-1
        path_mirror[3] = path[3]*-1

        # rotate path and plot
        if is_at_stator(im, path):
            Q = im.Qs
        else:
            Q = im.Qr
        _ = 2*np.pi/Q
        path[0], path[1] = rotate(0.5*np.pi - 0.5*_, path[0], path[1])
        path[2], path[3] = rotate(0.5*np.pi - 0.5*_, path[2], path[3])
        pyx_draw_path(vg, path, sign=1)

        path_mirror[0], path_mirror[1] = rotate(0.5*np.pi - 0.5*_, path_mirror[0], path_mirror[1])
        path_mirror[2], path_mirror[3] = rotate(0.5*np.pi - 0.5*_, path_mirror[2], path_mirror[3])
        pyx_draw_path(vg, path_mirror, sign=-1)

        # 注意，所有 tack_path 中的 path 都已经转动了90度了！
        # for mirror copy (along y-axis)
        path[0] *= -1
        path[2] *= -1
        pyx_draw_path(vg, path, sign=-1)

        path_mirror[0] *= -1
        path_mirror[2] *= -1
        pyx_draw_path(vg, path_mirror, sign=1)


    # # 整体转动90度。
    # for path in vg.tikz.track_path:
    #     if is_at_stator(im, path):
    #         Q = im.Qs
    #     else:
    #         Q = im.Qr
    #     _ = 2*np.pi/Q
    #     path[0], path[1] = rotate(0.5*np.pi - _, path[0], path[1])
    #     path[2], path[3] = rotate(0.5*np.pi - _, path[2], path[3])
    #     pyx_draw_path(vg, path)
    # track_path_backup = deepcopy(vg.tikz.track_path)

    # # Rotate Copy
    # for path in deepcopy(vg.tikz.track_path):
    #     if is_at_stator(im, path):
    #         Q = im.Qs
    #     else:
    #         Q = im.Qr
    #     _ = 2*np.pi/Q
    #     path[0], path[1] = rotate(_, path[0], path[1])
    #     path[2], path[3] = rotate(_, path[2], path[3])
    #     pyx_draw_path(vg, path)

    # # Rotate Copy
    # for path in (vg.tikz.track_path):
    #     # if np.sqrt(path[0]**2 + path[1]**2) > im.Radius_OuterRotor + 0.5*im.Length_AirGap:
    #     if is_at_stator(im, path):
    #         Q = im.Qs
    #     else:
    #         Q = im.Qr
    #     _ = 2*np.pi/Q
    #     path[0], path[1] = rotate(_, path[0], path[1])
    #     path[2], path[3] = rotate(_, path[2], path[3])
    #     pyx_draw_path(vg, path, sign=-1)

    vg.tikz.c.writePDFfile("selected_otimal_design%s"%(im.ID))
    # vg.tikz.c.writeEPSfile("pyx_output")
    print('Write to pdf file: selected_otimal_design%s.pdf.'%(im.ID))
    quit()

if bool_post_processing == True:
    # Combine all data 

    # Select optimal design by user-defined criteria
    if r'run#540' in fea_config_dict['run_folder']:
        ad.solver.output_dir = ad.solver.fea_config_dict['dir_parent'] + r'run#540011/' # severson01
        number_of_chromosome = ad.solver.read_swarm_data()
        swarm_data_severson01 = ad.solver.swarm_data

        def selection_criteria(swarm_data_):
            global best_idx, best_chromosome
            # HIGH TORQUE DENSITY
            # Severson01
            #       L_g,    w_st,   w_rt,   theta_so,   w_ro,    d_so,    d_ro,    -TRV,    -eta,    OC.
            # 1624 [1.15021, 8.97302, 8.33786, 3.22996, 0.759612, 2.81857, 1.11651, -22668.7, -0.953807, 4.79617]
            # Y730
            #       L_g,    w_st,   w_rt,   theta_so,   w_ro,    d_so,    d_ro,    -TRV,    -eta,    OC.
            # 794 [1.16163, 9.00566, 8.34039, 2.91474, 0.786231, 2.76114, 1.2485, -22666.7, -0.953681, 4.89779]
            # ----------------------------------------
            # HIGH EFFICIENCY
            # Y730
            #       L_g,    w_st,   w_rt,   theta_so,   w_ro,    d_so,    d_ro,    -TRV,    -eta,    OC.
            # 186 [1.25979, 6.80075, 5.64435, 5.8548, 1.59461, 2.11656, 2.58401, -17633.3, -0.958828, 5.53104]
            # 615 [1.2725, 5.6206, 4.60947, 3.56502, 2.27635, 0.506179, 2.78758, -17888.9, -0.958846, 8.56211]
            # ----------------------------------------
            # LOW RIPPLE PERFORMANCE
            # Severson02
            #       L_g,    w_st,   w_rt,   theta_so,   w_ro,    d_so,    d_ro,    -TRV,    -eta,    OC.
            # 1043 [1.38278, 8.91078, 7.43896, 2.66259, 0.611812, 1.50521, 1.51402, -19125.4, -0.953987, 2.91096]
            # 1129 [1.38878, 8.68378, 7.97301, 2.82904, 0.586374, 1.97867, 1.45825, -19169.0, -0.954226, 2.99944]
            # 1178 [1.36258, 8.9625, 7.49621, 2.5878, 0.503512, 0.678909, 1.74283, -19134.3, -0.952751, 2.90795]

            for idx, chromosome in enumerate(swarm_data_):
                # if chromosome[-1] < 5 and chromosome[-2] < -0.95 and chromosome[-3] < -22500: # best Y730     #1625, 0.000702091 * 8050 * 9.8 = 55.38795899 N.  FRW = 223.257 / 55.38795899 = 4.0
                # if chromosome[-1] < 10 and chromosome[-2] < -0.9585 and chromosome[-3] < -17500: # best Y730  #187, 0.000902584 * 8050 * 9.8 = 71.204851760 N. FRW = 151.246 / 71.204851760 = 2.124
                if chromosome[-1] < 3 and chromosome[-2] < -0.95 and chromosome[-3] < -19000: # best severson02 #1130, 0.000830274 * 8050 * 9.8 = 65.50031586 N.  FRW = 177.418 / 65.5 = 2.7
                    print(idx, chromosome)

                    def pyx_script():
                        # Plot cross section view
                        import population
                        im_best = population.bearingless_induction_motor_design.local_design_variant(ad.spec.im_template, 99, 999, best_chromosome[:7])
                        im_best.ID = str(best_idx)
                        pyx_draw_model(im_best)
                        quit()


                    # # Take high torque density design for LSA
                    # if idx == 1625 - 1:
                    #     best_idx = idx
                    #     best_chromosome = chromosome
                    #     pyx_script()

                    # # Take high efficiency design for LSA
                    # if idx == 187 - 1:
                    #     best_idx = idx
                    #     best_chromosome = chromosome
                    #     pyx_script()

                    # Take low ripple performance design for LSA
                    if idx == 1130 - 1:
                        best_idx = idx
                        best_chromosome = chromosome
                        # pyx_script()


        print('-'*40+'\nSeverson01' + '\n      L_g,    w_st,   w_rt,   theta_so,   w_ro,    d_so,    d_ro,    -TRV,    -eta,    OC.')
        selection_criteria(swarm_data_severson01)

        ad.solver.output_dir = ad.solver.fea_config_dict['dir_parent'] + r'run#540021/' # severson02
        number_of_chromosome = ad.solver.read_swarm_data()
        swarm_data_severson02 = ad.solver.swarm_data

        print('-'*40+'\nSeverson02' + '\n      L_g,    w_st,   w_rt,   theta_so,   w_ro,    d_so,    d_ro,    -TRV,    -eta,    OC.')
        selection_criteria(swarm_data_severson02)

        # swarm_data_severson01 = []
        # swarm_data_severson02 = []
        ad.solver.output_dir = ad.solver.fea_config_dict['dir_parent'] + r'run#540/' # ad.solver.fea_config_dict['run_folder'] 
        number_of_chromosome = ad.solver.read_swarm_data()
        swarm_data_Y730 = ad.solver.swarm_data

        print('-'*40+'\nY730' + '\n      L_g,    w_st,   w_rt,   theta_so,   w_ro,    d_so,    d_ro,    -TRV,    -eta,    OC.')
        selection_criteria(swarm_data_Y730)

        # Set the output_dir back!
        ad.solver.output_dir = ad.solver.fea_config_dict['dir_parent'] + ad.solver.fea_config_dict['run_folder']
        # quit()

    elif fea_config_dict['run_folder'] == r'run#538/':
        ad.solver.output_dir = ad.solver.fea_config_dict['dir_parent'] + r'run#538011/' # severson01
        number_of_chromosome = ad.solver.read_swarm_data()
        swarm_data_severson01 = ad.solver.swarm_data

        ad.solver.output_dir = ad.solver.fea_config_dict['dir_parent'] + r'run#538021/' # severson02
        number_of_chromosome = ad.solver.read_swarm_data()
        swarm_data_severson02 = ad.solver.swarm_data

        ad.solver.output_dir = ad.solver.fea_config_dict['dir_parent'] + r'run#538/' # ad.solver.fea_config_dict['run_folder'] 
        number_of_chromosome = ad.solver.read_swarm_data()
        swarm_data_Y730 = ad.solver.swarm_data

    print('Sizes of the 3 populations (in order):', len(swarm_data_severson01), len(swarm_data_severson02), len(swarm_data_Y730))
    ad.solver.swarm_data = swarm_data_severson01 + swarm_data_severson02 + swarm_data_Y730 # list add

    udp = Problem_BearinglessInductionDesign()
    ad.flag_do_not_evaluate_when_init_pop = True
    counter_fitness_called, counter_fitness_return = 0, 0
    prob = pg.problem(udp)

    # LSA
    if fea_config_dict['local_sensitivity_analysis'] == True:

        number_of_chromosome = ad.solver.read_swarm_data()

        if number_of_chromosome is not None:
            ad.solver.swarm_data

            # Learn Pareto front rank and plot
            for el in ad.solver.swarm_data:
                print('\t', el)
            print('count:', len(ad.solver.swarm_data))
            swarm_data_on_pareto_front = learn_about_the_archive(prob, ad.solver.swarm_data, len(ad.solver.swarm_data))

            # plot LSA
            ad.solver.swarm_data_container.sensitivity_bar_charts()

            plt.show()
            quit()


        else:
            def local_sensitivity_analysis(reference_design_denorm, percent=0.2):
                # 敏感性检查：以基本设计为准，检查不同的参数取极值时的电机性能变化！这是最简单有效的办法。七个设计参数，那么就有14种极值设计。

                if False:
                    # Within the original bounds
                    min_b, max_b = udp.get_bounds()
                    min_b, max_b = np.array(min_b), np.array(max_b)
                    diff = np.fabs(min_b - max_b)
                else:
                    # near the reference design
                    min_b = [el*(1.0-percent) for el in reference_design_denorm]
                    max_b = [el*(1.0+percent) for el in reference_design_denorm]
                    min_b, max_b = np.array(min_b), np.array(max_b)
                    diff = np.fabs(min_b - max_b)

                reference_design = (np.array(reference_design_denorm) - min_b) / diff
                print('reference_design_denorm:', reference_design_denorm)
                print('reference_design:\t\t', reference_design.tolist())
                base_design = reference_design.tolist()
                # quit()
                number_of_variants = fea_config_dict['local_sensitivity_analysis_number_of_variants']
                lsa_swarm = [base_design] # include reference design!
                for i in range(len(base_design)): # 7 design parameters
                    for j in range(number_of_variants+1): # 21 variants interval
                        # copy list
                        design_variant = base_design[::]
                        design_variant[i] = j * 1./number_of_variants
                        lsa_swarm.append(design_variant)

                lsa_swarm_denorm = min_b + lsa_swarm * diff 
                print(lsa_swarm)
                print(lsa_swarm_denorm)
                return lsa_swarm, lsa_swarm_denorm

            print('Best index', best_idx, '#%d'%(best_idx+1), 'Best chromosome', best_chromosome)
            _, lsa_swarm_denorm = local_sensitivity_analysis( reference_design_denorm=best_chromosome[:7],
                                                              percent=fea_config_dict['local_sensitivity_analysis_percent'] )
            lsa_popsize = len(lsa_swarm_denorm)

            # quit()
            lsa_pop = pg.population(prob, size=lsa_popsize)
            print('Set ad.flag_do_not_evaluate_when_init_pop to False...')
            ad.flag_do_not_evaluate_when_init_pop = False
            for i, design_denorm in enumerate(lsa_swarm_denorm):
                print('Evaluate', i)
                lsa_pop.set_x(i, design_denorm)
            print('LSA is done for a pop size of %d'%(lsa_popsize))
        quit()

    # plot pareto plot for three objectives...
    else:
        swarm_data_on_pareto_front = learn_about_the_archive(prob, ad.solver.swarm_data, len(ad.solver.swarm_data), len_s01=len(swarm_data_severson01), len_s02=len(swarm_data_severson02))
        plt.show()    
        quit()

        # # Reproduce a design 
        # cost_function, f1, f2, f3, \
        # normalized_torque_ripple, \
        # normalized_force_error_magnitude, \
        # force_error_angle = \
        #     ad.evaluate_design(ad.spec.im_template, best_chromosome[:7], 1130)
        quit()
        run_static_structural_fea(swda.best_design_denorm)


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
    counter_fitness_called, counter_fitness_return = 0, 0
    prob = pg.problem(udp)

    popsize = 78
        # Traceback (most recent call last):
        #   File "D:\OneDrive - UW-Madison\c\codes3\one_script.py", line 1189, in <module>
        #     pop = algo.evolve(pop)
        # ValueError: 
        # function: decomposition_weights
        # where: C:\bld\pygmo_1557474762576\_h_env\Library\include\pagmo/utils/multi_objective.hpp, 642
        # what: Population size of 72 is detected, but not supported by the 'grid' weight generation method selected. A size of 66 or 78 is possible.
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
        raise Exception('bool_refined_bounds')
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


