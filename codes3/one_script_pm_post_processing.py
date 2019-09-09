import numpy as np

def pyx_script(ad, best_chromosome, best_idx):
    # Plot cross section view
    import bearingless_spmsm_design
    spmsm_best = bearingless_spmsm_design.bearingless_spmsm_design(
                                        spmsm_template=ad.spec.acm_template,
                                        x_denorm=best_chromosome[:-3],
                                        counter=999,
                                        counter_loop=1
                                        )
    spmsm_best.ID = str(best_idx)
    import VanGogh
    tool_tikz = VanGogh.VanGogh_TikZPlotter()
    spmsm_best.draw_spmsm(tool_tikz, bool_pyx=True) # collecting track_path list for tool_tikz

    def redraw_cross_section_with_pyx(tikz, no_repeat_stator, no_repeat_rotor, mm_rotor_outer_radius, mm_air_gap_length):
        # PyX
        import pyx
        tikz.c = pyx.canvas.canvas() # clear the canvas because we want to redraw 90 deg with the data tikz.track_path
        from copy import deepcopy
        def pyx_draw_path(path, sign=1):
            if len(path) == 4: # line
                tikz.draw_line(path[:2], path[2:4], untrack=True)
            else: # == 6 for arc
                tikz.draw_arc(path[:2], path[2:4], path[4:6], relangle=sign*path[6], untrack=True)
        def rotate(_, x, y):
            return np.cos(_)*x + np.sin(_)*y, -np.sin(_)*x + np.cos(_)*y
        def is_at_stator(path):
            return np.sqrt(path[0]**2 + path[1]**2) > mm_rotor_outer_radius + 0.5*mm_air_gap_length

        print('Index   | Path data')
        for index, path in enumerate(tikz.track_path): # track_path is passed by reference and is changed by mirror
            path_mirror = deepcopy(path)
            # for mirror copy (along x-axis)
            path_mirror[1] = path[1]*-1
            path_mirror[3] = path[3]*-1
            # for mirror copy (along y-axis)
            # path_mirror[0] = path[0]*-1
            # path_mirror[2] = path[2]*-1

            # rotate path and plot
            if is_at_stator(path):
                Q = no_repeat_stator
            else:
                Q = no_repeat_rotor

            _ = 2*np.pi/Q

            path[0], path[1] = rotate(0.5*np.pi - 0.5*_, path[0], path[1])
            path[2], path[3] = rotate(0.5*np.pi - 0.5*_, path[2], path[3])
            pyx_draw_path(path, sign=1)

            # print(index, '\t|', ',\t'.join(['%g'%(el) for el in path]))

            if is_at_stator(path): # For PM rotor you don't need to mirror the rotor.
                path_mirror[0], path_mirror[1] = rotate(0.5*np.pi - 0.5*_, path_mirror[0], path_mirror[1])
                path_mirror[2], path_mirror[3] = rotate(0.5*np.pi - 0.5*_, path_mirror[2], path_mirror[3])
                pyx_draw_path(path_mirror, sign=-1)

                # 注意，所有 tack_path 中的 path 都已经转动了90度了！
                # for mirror copy (along y-axis)
                path[0] *= -1
                path[2] *= -1
                pyx_draw_path(path, sign=-1)

                path_mirror[0] *= -1
                path_mirror[2] *= -1
                pyx_draw_path(path_mirror, sign=1)

    redraw_cross_section_with_pyx(tool_tikz, spmsm_best.Q, spmsm_best.p, spmsm_best.Radius_OuterRotor, spmsm_best.Length_AirGap)
    tool_tikz.c.writePDFfile("selected_optimal_design_%s"%(best_idx))
    # tool_tikz.c.writeEPSfile("pyx_output")
    print('Write to pdf file: selected_optimal_design_%s.pdf.'%(best_idx))
    # os.system('start %s'%("selected_optimal_design%s.pdf"%(spmsm_best.name)))
    # quit()

def selection_criteria(ad, _swarm_data, _swarm_project_names, upper_bound_objectives=[20, -0.92, 200], best_idx=None):

    for idx, chromosome in enumerate(_swarm_data):
        OC = upper_bound_objectives[0] # ripple performance
        OB = upper_bound_objectives[1] # -efficiency
        OA = upper_bound_objectives[2] # cost
        if chromosome[-1] < OC and chromosome[-2] < OB and chromosome[-3] < OA:
            print(idx, _swarm_project_names[idx], chromosome[::-1])

            if idx == best_idx:
                best_chromosome = chromosome
                pyx_script(ad, best_chromosome, best_idx)

                return best_chromosome

def post_processing(ad, fea_config_dict):

    ad.solver.output_dir = ad.solver.fea_config_dict['dir_parent'] + ad.solver.fea_config_dict['run_folder'] 
    number_of_chromosome = ad.solver.read_swarm_data(ad.bound_filter)
    print('number_of_chromosome =', number_of_chromosome)

    _swarm_data          = ad.solver.swarm_data
    _swarm_project_names = ad.solver.swarm_data_container.project_names

    # print('-'*40+'\nY730' + '\n      L_g,    w_st,   w_rt,   theta_so,   w_ro,    d_so,    d_ro,    -TRV,    -eta,    OC.')
    print('-'*40+'\n%s | %s'%(fea_config_dict['pc_name'], fea_config_dict['run_folder']) + '\n      _____________________________________________________________________________________')

    # Select optimal design by user-defined criteria
    if r'run#62399' in fea_config_dict['run_folder']:
        selection_criteria(ad, _swarm_data, _swarm_project_names, upper_bound_objectives=[22, -0.94, 200], best_idx=2908)
    elif r'run#62499' in fea_config_dict['run_folder']:
        selection_criteria(ad, _swarm_data, _swarm_project_names, upper_bound_objectives=[22, -0.94, 200], best_idx=4235)
    elif r'run#62599' in fea_config_dict['run_folder']:
        selection_criteria(ad, _swarm_data, _swarm_project_names, upper_bound_objectives=[13, -0.94, 200], best_idx=3741)
    elif r'run#62699' in fea_config_dict['run_folder']:
        selection_criteria(ad, _swarm_data, _swarm_project_names, upper_bound_objectives=[3, -0.946, 80], best_idx=57)
        quit()
    elif r'run#62799' in fea_config_dict['run_folder']:
        selection_criteria(ad, _swarm_data, _swarm_project_names, upper_bound_objectives=[6, -0.94, 200], best_idx=17)
    else:
        raise Exception('Not implmented in post_processing.py')

    # Set the output_dir back! Obsolete?
    ad.solver.output_dir = ad.solver.fea_config_dict['dir_parent'] + ad.solver.fea_config_dict['run_folder']

if __name__ == '__main__':
    # PEMD 2020

    from utility import my_execfile
    bool_post_processing = True # solve or post-processing

    my_execfile('./default_setting.py', g=globals(), l=locals())
    fea_config_dict

    # Combined winding PMSM
    fea_config_dict['TORQUE_CURRENT_RATIO'] = 0.95
    fea_config_dict['SUSPENSION_CURRENT_RATIO'] = 0.05

    # Collect all data 
    for run_folder, spec_file in zip(
                                      [ r'run#62399/',  # spec_ECCE_PMSM_ (Q6p2)
                                        r'run#62499/',  # spec_PEMD_BPMSM_Q12p2
                                        r'run#62599/',  # spec_PEMD_BPMSM_Q6p1)
                                        r'run#62699/',  # spec_PEMD_BPMSM_Q12p4)
                                        r'run#62799/'], # spec_PEMD_BPMSM_Q24p1
                                      [ './spec_ECCE_PMSM_Q6p2.py',
                                        './spec_PEMD_BPMSM_Q12p2.py',
                                        './spec_PEMD_BPMSM_Q6p1.py',
                                        './spec_PEMD_BPMSM_Q12p4.py',
                                        './spec_PEMD_BPMSM_Q24p1.py']
                                    ): 

        fea_config_dict['run_folder'] = run_folder

        my_execfile(spec_file, g=globals(), l=locals()) # Q=24, p=1, ps=2
        spec

        # Adopt Bianchi 2006 for a SPM motor template
        spec.build_pmsm_template(fea_config_dict, im_template=None)

        # select motor type here
        spec.acm_template = spec.pmsm_template
        print('Build ACM template...')

        import acm_designer
        global ad
        ad = acm_designer.acm_designer(fea_config_dict, spec)
        __builtins__.ad = ad # share global variable between modules
        print(__builtins__.ad)

        global counter_fitness_called, counter_fitness_return
        counter_fitness_called, counter_fitness_return = 0, 0
        __builtins__.counter_fitness_called = counter_fitness_called
        __builtins__.counter_fitness_return = counter_fitness_return

        ad.bounds_denorm = spec.acm_template.get_classic_bounds(which_filter='VariableSleeveLength')
        # ad.bounds_denorm = spec.acm_template.get_classic_bounds(which_filter='FixedSleeveLength') # ad.get_classic_bounds() <- obsolete
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


        if bool_post_processing == True:
            # select the optimal design and draw its cross section sketch
            post_processing(ad, fea_config_dict)

        if bool_post_processing == False:
            # plot the Pareto front for the archive
            import utility_moo

            from Problem_BearinglessSynchronousDesign import Problem_BearinglessSynchronousDesign
            udp = Problem_BearinglessSynchronousDesign()
            import pygmo as pg
            prob = pg.problem(udp)
            popsize = 78

            swarm_data_on_pareto_front = utility_moo.learn_about_the_archive(prob, ad.solver.swarm_data, popsize, fea_config_dict, bool_plot_and_show=True)



