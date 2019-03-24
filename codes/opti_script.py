# coding:utf-8
#execfile("D:/OneDrive - UW-Madison/c/codes/opti_script.py")
#execfile(r'K:\jchen782\JMAG\c\codes/opti_script.py')
#execfile('C:/Users/Hory Chen/OneDrive - UW-Madison/c/codes/opti_script.py')

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 1. General Information & Packages Loading
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
execfile('D:/OneDrive - UW-Madison/c/codes/default_setting.py') # Absolute path is needed for running in JMAG

if False: # ECCE
    # fea_config_dict['Active_Qr'] = 16
    fea_config_dict['use_weights'] = 'O1'

    fea_config_dict['local_sensitivity_analysis'] = True
    run_folder = r'run#400/' # Sensitivity analysis for Qr=16, T440p
    run_folder = r'run#140/' # Sensitivity analysis for Qr=16, Y730

    fea_config_dict['local_sensitivity_analysis'] = False
    run_folder = r'run#141/' # run for reference
    run_folder = r'run#142/' # optimize Qr=16 for O2
    run_folder = r'run#143/' # test for shitty design (not true)
    run_folder = r'run#144/' # optimize Qr=16 for O1
else: # NineSigma

    # fea_config_dict['Active_Qr'] = 16
    fea_config_dict['use_weights'] = 'O2'

    fea_config_dict['local_sensitivity_analysis'] = True
    # Separate winding
    run_folder = r'run#180/' # Demo run for Qr=32 (too many end ring layers)
    run_folder = r'run#181/' # Demo run for Qr=16 (flag_optimization should be true)
    run_folder = r'run#182/' # Sensitivity analysis for Qr=16 (forget to update generate_pop function)
    run_folder = r'run#183/' # Sensitivity analysis for Qr=16
    run_folder = r'run#184/' # Bounds are not adjusted accordingly

    # combined DPNV winding
    run_folder = r'run#185/' # Correct bounds
    run_folder = r'run#186/' # New Initial Design but the force profile seems wrong (JMAG only run_list)
    run_folder = r'run#187/' # Run a sensitivity analysis anyway. (new codes in add_study, while wrong codes in add_Tran2TSS)

    run_folder = r'run#188/' # Run a sensitivity analysis (bug exists)

    run_folder = r'run#189/' # Test the generated four pole field and update the new implemenation of CurrentSource in circuit of JMAG.
    run_folder = r'run#190/' # Success: local sensitivity analysis for NineSigma, but...
                             # the initial design [4.484966, 1.60056, 1.130973, 6.238951254340352, 1.0, 3.0, 1.0] is not within the bounds:
                             #         de_config_dict = {  'original_bounds':[ [   3, 5.6],#--# stator_tooth_width_b_ds
                             #                                                 [ 0.8,   3],   # air_gap_length_delta
                             #                                                 [5e-1,   3],   # Width_RotorSlotOpen 
                             #                                                 [ 3.6,5.45],#--# rotor_tooth_width_b_dr 
                             #                                                 [5e-1,   3],   # Length_HeadNeckRotorSlot
                             #                                                 [   1,  10],   # Angle_StatorSlotOpen
                             #                                                 [5e-1,   3] ], # Width_StatorTeethHeadThickness

    run_folder = r'run#191/' # run again for the correct bounds

    fea_config_dict['local_sensitivity_analysis'] = False
    fea_config_dict['bool_refined_bounds'] = True
    run_folder = r'run#192/' # optimize Qr=16 for O2 : 跑了一代了才发现悬浮绕组电流都加错了，悬浮力900多牛。

    run_folder = r'run#193/' # optimize Qr=16 for O2：优化的结果使得电机在效率没有明显提高的情况下，额定轴向长度变长了。。。

    fea_config_dict['use_weights'] = 'O3'
    run_folder = r'run#194/' # optimize Qr=16 for O3

    fea_config_dict['mimic_separate_winding_with_DPNV_winding'] == True:
    run_folder = r'run#195/' # optimize Qr=16 for O3 - separate winding (with a reduced torque current down to 60%)

fea_config_dict['run_folder'] = run_folder
logger = utility.myLogger(fea_config_dict['dir_codes'], prefix='ecce_'+run_folder[:-1])

# Exact Approach: compute the tangent points of the two circles （未修正）



#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 2. Initilize Swarm and Initial Pyrhonen's Design (Run this part in JMAG)
#    Bounds: 1e-1也还是太小了（第三次报错），至少0.5mm长吧 # 1e-1 is the least geometry value. a 1e-2 will leads to：转子闭口槽极限，会导致edge过小，从而报错：small arc entity exists.png
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
if fea_config_dict['flag_optimization'] == True:
    if False: # 4 pole motor
        if False: # intuitive bounds
            pass
            # de_config_dict = {  'bounds':     [ [   3, 9],   # stator_tooth_width_b_ds
            #                                     [ 0.8, 4],   # air_gap_length_delta
            #                                     [5e-1, 3],   # Width_RotorSlotOpen 
            #                                     [ 2.5, 6],   # rotor_tooth_width_b_dr # 8 is too large, 6 is almost too large
            #                                     [5e-1, 3],   # Length_HeadNeckRotorSlot
            #                                     [   1, 10],  # Angle_StatorSlotOpen
            #                                     [5e-1, 3] ], # Width_StatorTeethHeadThickness
            #                     'mut':        0.8,
            #                     'crossp':     0.7,
            #                     'popsize':    20,
            #                     'iterations': 48 } # begin at 5     
        else: # based on Pyrhonen09
            # see Tran2TSS_PS_Opti_Qr=16_Btooth=1.5T.xlsx
            # de_config_dict = {  'bounds':     [ [   4, 7.2],#--# stator_tooth_width_b_ds
            #                                     [ 0.8,   4],   # air_gap_length_delta
            #                                     [5e-1,   3],   # Width_RotorSlotOpen 
            #                                     [ 2.5, 5.2],#--# rotor_tooth_width_b_dr # 8 is too large, 6 is almost too large
            #                                     [5e-1,   3],   # Length_HeadNeckRotorSlot
            #                                     [   1,  10],   # Angle_StatorSlotOpen
            #                                     [5e-1,   3] ], # Width_StatorTeethHeadThickness
            #                     'mut':        0.8,
            #                     'crossp':     0.7,
            #                     'popsize':    14, # 50, # 100,
            #                     'iterations': 1 } # 148

            # see Tran2TSS_PS_Opti_Qr=32_Btooth=1.1T.xlsx
            if fea_config_dict['Active_Qr'] == 32:
                de_config_dict = {  'original_bounds':[ [ 4.9,   9],#--# stator_tooth_width_b_ds
                                                        [ 0.8,   3],   # air_gap_length_delta
                                                        [5e-1,   3],   # Width_RotorSlotOpen 
                                                        [ 2.7,   5],#--# rotor_tooth_width_b_dr # 8 is too large, 6 is almost too large
                                                        [5e-1,   3],   # Length_HeadNeckRotorSlot
                                                        [   1,  10],   # Angle_StatorSlotOpen
                                                        [5e-1,   3] ], # Width_StatorTeethHeadThickness
                                    'mut':        0.8,
                                    'crossp':     0.7,
                                    'popsize':    30, # 50, # 100,
                                    'iterations': 100,
                                    'narrow_bounds_normalized':[[],
                                                                [],
                                                                [],
                                                                [],
                                                                [],
                                                                [],
                                                                [] ],
                                    'bounds':[]}

            # see Tran2TSS_PS_Opti_Qr=16.xlsx
            if fea_config_dict['Active_Qr'] == 16:
                de_config_dict = {  'original_bounds':[ [ 4.9,   9],#--# stator_tooth_width_b_ds
                                                        [ 0.8,   3],   # air_gap_length_delta
                                                        [5e-1,   3],   # Width_RotorSlotOpen 
                                                        [ 6.5, 9.9],#--# rotor_tooth_width_b_dr 
                                                        [5e-1,   3],   # Length_HeadNeckRotorSlot
                                                        [   1,  10],   # Angle_StatorSlotOpen
                                                        [5e-1,   3] ], # Width_StatorTeethHeadThickness
                                    'mut':        0.8,
                                    'crossp':     0.7,
                                    'popsize':    30, # 21*7,  # 50, # 100,
                                    'iterations': 100,
                                    'narrow_bounds_normalized':[[],
                                                                [],
                                                                [],
                                                                [],
                                                                [],
                                                                [],
                                                                [] ],
                                    'bounds':[]}
    else: # 2 pole motor
        de_config_dict = {  'original_bounds':[ [ 2.7, 5.8],#--# stator_tooth_width_b_ds
                                                [ 0.8,   3],   # air_gap_length_delta
                                                [5e-1,   3],   # Width_RotorSlotOpen 
                                                [ 3.5, 6.3],#--# rotor_tooth_width_b_dr # It allows for large rotor tooth because we raise Jr and recall this is for less slots---Qr=16.
                                                [5e-1,   3],   # Length_HeadNeckRotorSlot
                                                [   1,  10],   # Angle_StatorSlotOpen
                                                [5e-1,   3] ], # Width_StatorTeethHeadThickness
                            'mut':        0.8,
                            'crossp':     0.7,
                            'popsize':    30, # 21*7,  # 50, # 100,
                            'iterations': 100,
                            'narrow_bounds_normalized':[[],
                                                        [],
                                                        [],
                                                        [],
                                                        [],
                                                        [],
                                                        [] ],
                            'bounds':[]}

    # Sensitivity Analysis based narrowing bounds
    if fea_config_dict['bool_refined_bounds'] == True:
        # data acquired from run#116
        numver_of_variants = 20.0
        if fea_config_dict['Active_Qr'] == 32: # O1 is already best
            # from utility.py O2
            raw_narrow_bounds = [   [9, 10],
                                    [5, 6, 7], #[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                    [0, 1, 2, 3],
                                    [20],
                                    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                                    [5, 6, 7],
                                    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]

        if fea_config_dict['Active_Qr'] == 16: 
            # from utility.py:run140 O2
            raw_narrow_bounds = [   [3, 4, 7, 9, 12, 14, 15, 16, 19, 20],
                                    [1, 2, 7, 8, 9, 12, 13, 14, 15, 16, 18],
                                    [1, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 19],
                                    [1, 4, 5, 9, 11, 12, 13, 16, 17],
                                    [1, 2, 3, 6, 7, 10, 11, 12, 13, 15, 18],
                                    [2, 4, 5, 6, 8, 10, 11, 12, 13, 16, 18, 19],
                                    [0, 1, 3, 4, 6, 7, 8, 9, 10, 12, 15, 17, 18, 20]]

            # from utility.py:run191 O2
            raw_narrow_bounds = [   [12, 13, 14, 15, 16, 17, 18, 19, 20],
                                    [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                    [19, 20],
                                    [19, 20],
                                    [12, 13],
                                    [11, 13, 14, 15, 16, 17, 18],
                                    [9, 10] ]

            # from utility.py:run191 O3
            raw_narrow_bounds = [   [9, 10, 11, 12, 13],
                                    [0, 1, 2, 3, 4, 5, 6, 7],
                                    [0, 1, 2, 3, 4, 5, 6, 7],
                                    [16, 17, 19, 20],
                                    [0, 1, 2, 3, 4, 5, 6],
                                    [0, 1, 2, 3, 4, 5],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]

        for ind, bound in enumerate(raw_narrow_bounds):
            de_config_dict['narrow_bounds_normalized'][ind].append(bound[0] /numver_of_variants)
            de_config_dict['narrow_bounds_normalized'][ind].append(bound[-1]/numver_of_variants)

            if de_config_dict['narrow_bounds_normalized'][ind][0] == de_config_dict['narrow_bounds_normalized'][ind][1]:
                raise Exception('Take a check here.')
                print 'ind=',ind, '---manually set the proper bounds based on the initial design: 7.00075,1.26943,0.924664,4.93052,1,3,1'
                de_config_dict['narrow_bounds_normalized'][ind][0] = 4.93052 / 5

        print 'narrow_bounds_normalized:', de_config_dict['narrow_bounds_normalized']

        for bnd1, bnd2 in zip(de_config_dict['original_bounds'], de_config_dict['narrow_bounds_normalized']):
            diff = bnd1[1] - bnd1[0]
            de_config_dict['bounds'].append( [ bnd1[0]+diff*bnd2[0] , bnd1[0]+diff*bnd2[1] ]) # 注意，都是乘以original_bounds的上限哦！

        print 'bounds:', de_config_dict['bounds']
        print 'original_bounds:', de_config_dict['original_bounds']
    else:
        de_config_dict['bounds'] = de_config_dict['original_bounds']
else:
    raise
    de_config_dict = {  'bounds':     [ [   3, 9],   # stator_tooth_width_b_ds
                                        [ 0.8, 4],   # air_gap_length_delta
                                        [5e-1, 3],   # Width_RotorSlotOpen 
                                        [ 2.5, 6],   # rotor_tooth_width_b_dr # 8 is too large, 6 is almost too large
                                        [5e-1, 3],   # Length_HeadNeckRotorSlot
                                        [   1, 10],  # Angle_StatorSlotOpen
                                        [5e-1, 3] ], # Width_StatorTeethHeadThickness
                        'mut':        0.8,
                        'crossp':     0.7,
                        'popsize':    20,
                        'iterations': 1 }

# init the swarm
sw = population.swarm(fea_config_dict, de_config_dict=de_config_dict)
# sw.show(which='all')
# print sw.im.show(toString=True)
# quit()


#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 3. Initialize FEMM Solver (if required)
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
if fea_config_dict['jmag_run_list'][0] == 0:
    # and let jmag know about it
    sw.femm_solver = FEMM_Solver.FEMM_Solver(sw.im, flag_read_from_jmag=False, freq=2.23) # eddy+static


#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 4. Run DE Optimization
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
count_abort = 0
# while True:
if True:
    
    logger.debug('count_abort=%d' % (count_abort))

    # if optimization_flat == True:
    # generate the initial generation
    sw.generate_pop()

    # if count_abort == 0:
    #     # add initial_design of Pyrhonen09 to the initial generation
    #     utility.add_Pyrhonen_design_to_first_generation(sw, de_config_dict, logger)

    # write FEA config to disk
    sw.write_to_file_fea_config_dict()

    if True:
        try:
            de_generator = sw.de()
            # run
            # result = list(de_generator)
            for result in de_generator:
                print result
        except Exception as e:
            print 'See log file for the error msg.'
            logger.error(u'Optimization aborted.', exc_info=True)

            raise e

            # # 避免死循环
            # count_abort+1
            # if count_abort > 10:
            #     quit()

            # # quit()
            # try:
            #     # reload for changed codes
            #     reload(population) # relaod for JMAG's python environment
            #     reload(FEMM_Solver)
            #     reload(utility)

            #     # notification via email
            #     # utility.send_notification(u'Optimization aborted.')
            
            #     # msg = 'Pop status report\n------------------------\n'
            #     # msg += '\n'.join('%.16f'%(x) for x in sw.fitness) + '\n'
            #     # msg += '\n'.join(','.join('%.16f'%(x) for x in y) for y in sw.pop_denorm)    
            #     # logger.debug(msg)

            #     # sw.bool_auto_recovered_run = True
            # except:
            #     pass
        else:
            logger.info('Done.')
            utility.send_notification('Done.')

