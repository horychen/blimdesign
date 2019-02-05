# coding:utf-8
#execfile('D:/OneDrive - UW-Madison/c/codes/opti_script.py')
#execfile(r'K:\jchen782\JMAG\c\codes/opti_script.py')
#execfile('C:/Users/Hory Chen/OneDrive - UW-Madison/c/codes/opti_script.py')

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 1. General Information & Packages Loading
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
import os
def where_am_i(fea_config_dict):
    dir_interpreter = os.path.abspath('')
    print dir_interpreter
    if os.path.exists('D:/'):
        print 'you are on Legion Y730'
        dir_parent = 'D:/OneDrive - UW-Madison/c/'
        dir_codes = dir_parent + 'codes/'
        dir_lib = dir_parent + 'codes/'
        # dir_initial_design = dir_parent + 'pop/'
        # dir_csv_output_folder = dir_parent + 'csv_opti/'
        dir_femm_files = 'D:/femm42/' # .ans files are too large to store on OneDrive anymore
        dir_project_files = 'D:/JMAG_Files/'
        pc_name = 'Y730'
    elif os.path.exists('I:/'):
        print 'you are on Severson02'
        dir_parent = 'I:/jchen782/c/'
        dir_codes = dir_parent + 'codes/'
        dir_lib = dir_parent + 'codes/'
        dir_femm_files = 'I:/jchen782/FEMM/'
        dir_project_files = 'I:/jchen782/JMAG/'
        pc_name = 'Seversion02'
    elif os.path.exists('K:/'):
        print 'you are on Severson01'
        dir_parent = 'K:/jchen782/c/'
        dir_codes = dir_parent + 'codes/'
        dir_lib = dir_parent + 'codes/'
        dir_femm_files = 'K:/jchen782/FEMM/'
        dir_project_files = 'K:/jchen782/JMAG/'
        pc_name = 'Seversion01'
    else:
        # elif 'Chen' in 'dir_interpreter':
        print 'you are on T440p'
        dir_parent = 'C:/Users/Hory Chen/OneDrive - UW-Madison/c/'
        dir_codes = dir_parent + 'codes/'
        dir_lib = dir_parent + 'codes/'
        # dir_initial_design = dir_parent + 'pop/'
        # dir_csv_output_folder = dir_parent + 'csv_opti/'
        dir_femm_files = 'C:/femm42/'
        dir_project_files = 'C:/JMAG_Files/'
        pc_name = 'T440p'

    os.chdir(dir_codes)

    fea_config_dict['dir_parent']            = dir_parent
    fea_config_dict['dir_lib']               = dir_lib
    fea_config_dict['dir_codes']             = dir_codes
    fea_config_dict['dir_femm_files']        = dir_femm_files
    fea_config_dict['dir_project_files']     = dir_project_files
    # fea_config_dict['dir_initial_design']    = dir_initial_design
    # fea_config_dict['dir_csv_output_folder'] = dir_csv_output_folder
    fea_config_dict['pc_name']               = pc_name
    fea_config_dict['dir_interpreter']       = dir_interpreter

    if pc_name == 'Y730':
        fea_config_dict['delete_results_after_calculation'] = False # save disk space
        if fea_config_dict['Restart'] == False:
            fea_config_dict['OnlyTableResults'] = True  # save disk space for my PC
    # However, we need field data for iron loss calculation
    fea_config_dict['OnlyTableResults'] = False 

fea_config_dict = {
    ##########################
    # Sysetm Control
    ##########################
    'Active_Qr':32, # 36
    'TranRef-StepPerCycle':40,
    'OnlyTableResults':False, # modified later according to pc_name
        # multiple cpu (SMP=2)
        # directSolver over ICCG Solver
    'Restart':False, # restart from frequency analysis is not needed, because SSATA is checked and JMAG 17103l version is used.
    'flag_optimization':True,
    'FEMM_Coarse_Mesh':True,
    'local_sensitivity_analysis':False,

    ##########################
    # Optimization
    ##########################
    # 'FrequencyRange':range(1,6), # the first generation for PSO
    'number_of_steps_2ndTTS':32, # use a multiples of 4! # 8*32 # steps for half period (0.5). That is, we implement two time sections, the 1st section lasts half slip period and the 2nd section lasts half fandamental period.
    'JMAG_Scheduler':False, # multi-cores run
    'delete_results_after_calculation': False, # check if True can we still export Terminal Voltage? 如果是True，那么就得不到Terminal Voltage了！

    ##########################
    # Design Specifications
    ##########################
    'DPNV': True,
    'End_Ring_Resistance':0, # 0 for consistency with FEMM with pre-determined currents # 9.69e-6, # this is still too small for Chiba's winding

    'Steel': 'M19Gauge29', 
    # 'Steel': 'M15',
    # 'Steel': 'Arnon5', 
                                # If you modify the temperature here, you should update the initial design (the im.DriveW_Rs should be updated and it used in JMAG FEM Coil)
    'Bar_Conductivity':1/((3.76*75+873)*1e-9/55.), # 1/((3.76*100+873)*1e-9/55.) for Copper, where temperature is 25 or 100 deg Celsius.
    # 'Bar_Conductivity':40e6, # 40e6 for Aluminium; 
}
where_am_i(fea_config_dict)
from sys import path as sys_path
sys_path.append(fea_config_dict['dir_lib'])
import population
import FEMM_Solver
import utility
reload(population) # relaod for JMAG's python environment
reload(FEMM_Solver)
reload(utility)

# run_list = [1,1,0,0,0] 
run_list = [0,1,0,0,0] 

# run_folder = r'run#100/' # no iron loss csv data but there are field data!
# run_folder = r'run#101/' # 75 deg Celsius, iron loss csv data, delete field data after calculation.
# run_folder = r'run#102/' # the efficiency is added to objective function，原来没有考虑效率的那些设计必须重新评估，否则就不会进化了，都是旧的好！
# run_folder = r'run#103/' # From this run, write denormalized pop data to disk!
# run_folder = r'run#104/' # Make sure all the gen#xxxx file uses denormalized values.

# run_folder = r'run#105/' # Femm is used for breakdown torque and frequency! 
# run_folder = r'run#106/' # You need to initialize femm_solver every calling of fobj
# run_folder = r'run#107/' # There is no slide mesh if you add new study for Tran2TSS
# run_folder = r'run#108/' # Efficiency is added. femm_found.fem feature is added.
# run_folder = r'run#109/' # test the effectiveness of de algorithm
# run_folder = r'run#110/' # Truly recovable!
# run_folder = r'run#111/' # new living pop and its fitness and its id
# run_folder = r'run#112/' # test shitty design
# run_folder = r'run#113/' # never lose any design data again, you can generate initial pop from the IM design database!
# run_folder = r'run#114/' # femm-mesh-size-sensitivity study

# run_folder = r'run#115/' # 敏感性检查：以基本设计为准，检查不同的参数取极值时的电机性能变化！这是最简单有效的办法。七个设计参数，那么就有14种极值设计。
# run_folder = r'run#116/' # more variants! 20 of them!
# run_folder = r'run#117/' # run for the candidate design only

# run_folder = r'run#118/' # for plot in TpX (VanGogh prints P1-P8)

# run_folder = r'run#120/' # minimize O1 with narrowed bounds according to sensitivity analysis (run#116)
# run_folder = r'run#121/' # minimize O2 with narrowed bounds according to sensitivity analysis (run#116)

# run_list = [1,1,0,0,0] 
# fea_config_dict['flag_optimization'] = False
# fea_config_dict['End_Ring_Resistance'] = 9.69e-6
# run_folder = r'run#122/' # O2 best: with end ring and fine step size transient FEA

fea_config_dict['Active_Qr'] = 16
fea_config_dict['local_sensitivity_analysis'] = True
run_folder = r'run#400/' # Sensitivity analysis for Qr=16, T440p
run_folder = r'run#140/' # Sensitivity analysis for Qr=16, Y730

# Exact Approach: compute the tangent points of the two circles （未修正）

fea_config_dict['local_sensitivity_analysis'] = False
run_folder = r'run#141/' # run for reference
run_folder = r'run#142/' # optimize Qr=16 for O2
run_folder = r'run#143/' # test for shitty design (not true)
run_folder = r'run#144/' # optimize Qr=16 for O1

fea_config_dict['run_folder'] = run_folder
fea_config_dict['jmag_run_list'] = run_list
def build_model_name_prefix(fea_config_dict):
    if fea_config_dict['flag_optimization'] == True:
        fea_config_dict['model_name_prefix'] = 'OP_PS_Qr%d_%s' % (fea_config_dict['Active_Qr'], fea_config_dict['Steel'])
    else:
        fea_config_dict['model_name_prefix'] = 'PS_Qr%d_%s' % (fea_config_dict['Active_Qr'], fea_config_dict['Steel'])
    if fea_config_dict['DPNV'] == True:
        fea_config_dict['model_name_prefix'] += '_DPNV'
    if fea_config_dict['End_Ring_Resistance'] == 0:
        fea_config_dict['model_name_prefix'] += '_NoEndRing'
    if fea_config_dict['Restart'] == True:
        fea_config_dict['model_name_prefix'] += '_Restart'
    return fea_config_dict['model_name_prefix']
print build_model_name_prefix(fea_config_dict)

# FEMM Static FEA
fea_config_dict['femm_deg_per_step'] = 0.25 * (360/4) / utility.lcm(24/4., fea_config_dict['Active_Qr']/4.) # at least half period
# fea_config_dict['femm_deg_per_step'] = 1 * (360/4) / utility.lcm(24/4., fea_config_dict['Active_Qr']/4.) # at least half period
# fea_config_dict['femm_deg_per_step'] = 0.1 #0.5 # deg
print 'femm_deg_per_step is', fea_config_dict['femm_deg_per_step'], 'deg (Qs=24, p=2)'

logger = utility.myLogger(fea_config_dict['dir_codes'], prefix='ecce_'+run_folder[:-1])

# Debug
# if os.path.exists('d:/femm42/PS_Qr32_NoEndRing_M19Gauge29_DPNV_1e3Hz'):
#     os.system('bash -c "rm -r /mnt/d/femm42/PS_Qr32_NoEndRing_M19Gauge29_DPNV_1e3Hz"')
# if os.path.exists('d:/OneDrive - UW-Madison/c/pop/Tran2TSS_PS_Opti.txt'):
#     os.system('bash -c "mv /mnt/d/OneDrive\ -\ UW-Madison/c/pop/Tran2TSS_PS_Opti.txt /mnt/d/OneDrive\ -\ UW-Madison/c/pop/initial_design.txt"')



#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 2. Initilize Swarm and Initial Pyrhonen's Design (Run this part in JMAG)
#    Bounds: 1e-1也还是太小了（第三次报错），至少0.5mm长吧 # 1e-1 is the least geometry value. a 1e-2 will leads to：转子闭口槽极限，会导致edge过小，从而报错：small arc entity exists.png
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
if fea_config_dict['flag_optimization'] == True:
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

    # Sensitivity Analysis based narrowing bounds
    if True:
        # data acquired from run#116
        numver_of_variants = 20.0
        if fea_config_dict['Active_Qr'] == 32: # O1 is already from utility.py
            raw_narrow_bounds = [   [9, 10],
                                    [5, 6, 7], #[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                    [0, 1, 2, 3],
                                    [20],
                                    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                                    [5, 6, 7],
                                    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]

        if fea_config_dict['Active_Qr'] == 16: # O2 is already from utility_run140.py
            raw_narrow_bounds = [   [3, 4, 7, 9, 12, 14, 15, 16, 19, 20],
                                    [1, 2, 7, 8, 9, 12, 13, 14, 15, 16, 18],
                                    [1, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 19],
                                    [1, 4, 5, 9, 11, 12, 13, 16, 17],
                                    [1, 2, 3, 6, 7, 10, 11, 12, 13, 15, 18],
                                    [2, 4, 5, 6, 8, 10, 11, 12, 13, 16, 18, 19],
                                    [0, 1, 3, 4, 6, 7, 8, 9, 10, 12, 15, 17, 18, 20]]

        for ind, bound in enumerate(raw_narrow_bounds):
            de_config_dict['narrow_bounds_normalized'][ind].append(bound[0] /numver_of_variants)
            de_config_dict['narrow_bounds_normalized'][ind].append(bound[-1]/numver_of_variants)
            if de_config_dict['narrow_bounds_normalized'][ind][0] == de_config_dict['narrow_bounds_normalized'][ind][1]:
                print 'ind=',ind, '---manually set the proper bounds based on the initial design: 7.00075,1.26943,0.924664,4.93052,1,3,1'
                de_config_dict['narrow_bounds_normalized'][ind][0] = 4.93052 / 5
        print de_config_dict['narrow_bounds_normalized']

        for bnd1, bnd2 in zip(de_config_dict['original_bounds'], de_config_dict['narrow_bounds_normalized']):
            diff = bnd1[1] - bnd1[0]
            de_config_dict['bounds'].append( [ bnd1[0]+diff*bnd2[0] , bnd1[0]+diff*bnd2[1] ]) # 注意，都是乘以original_bounds的上限哦！

        print de_config_dict['bounds']
        print de_config_dict['original_bounds']

    else:
        de_config_dict['bounds'] = de_config_dict['original_bounds']

else:
    de_config_dict = None

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


################################################################
# Check the shitty design (that fails to draw) or the best design
################################################################
if True:
    if False:
        # Now with this redraw from im.show(toString=True) feature, you can see actually sometimes jmag fails to draw because of PC system level interference, rather than bug in my codes.
        # debug for shitty design that failed to draw
        shitty_design = population.bearingless_induction_motor_design.reproduce_the_problematic_design(r'D:\OneDrive - UW-Madison\c\codes/'+'shitty_design.txt')
        shitty_design.show()
        sw.run(shitty_design)
    elif True:
        # this is not a shitty design. just due to something going wrong calling JMAG remote method
        shitty_design = population.bearingless_induction_motor_design.reproduce_the_problematic_design(r'D:\OneDrive - UW-Madison\c\codes/'+'shitty_design_Qr16_statorCore.txt')        
        sw.number_current_generation = 0
        sw.fobj(99, utility.Pyrhonen_design(shitty_design).design_parameters_denorm)
    else:
        best_design = population.bearingless_induction_motor_design.reproduce_the_problematic_design(r'D:\OneDrive - UW-Madison\c\codes/'+'Qr32_O2Best_Design.txt')
        individual_denorm = utility.Pyrhonen_design(best_design).design_parameters_denorm
        print individual_denorm
        # os.path.mkdirs(sw.dir_parent + 'pop/' + sw.run_folder)
        # open(sw.dir_parent + 'pop/' + sw.run_folder + 'thermal_penalty_individuals.txt', 'w').close()
        sw.number_current_generation = 0
        sw.fobj(99, individual_denorm)
    raise


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




















# Run JCF from command linie instead of GUI
# if not sw.has_results(im_initial, study_type='Tran2TSS'):
#     os.system(r'set InsDir=D:\Program Files\JMAG-Designer17.1/')
#     os.system(r'set WorkDir=D:\JMAG_Files\JCF/')
#     os.system(r'cd /d "%InsDir%"')
#     for jcf_file in os.listdir(sw.dir_jcf):
#         if 'Tran2TSS' in jcf_file and 'Mesh' in jcf_file:
#             os.system('ExecSolver "%WorkDir%' + jcf_file + '"')

'''
Two Shared process - Max CPU occupation 32.6%
Freq: 2:16 (13 Steps)
Tran2TSS: 3:38 (81 Steps)
Freq-FFVRC: 1:33 (10 Steps)
TranRef: 1:49:02 (3354 Steps) -> 3:18:19 if no MultiCPU
StaticJMAG: 
StaticFEMM: 15 sec one eddy current solve
            7 sec one static solve
'''

# # de version 2 is called by this
# for el in result:
#     pop, fit, idx = el
#     print '---------------------------'
#     print 'pop:', pop
#     print 'fit:', fit
#     print 'idx:', idx
#     # for ind in pop:
#     #     data = fmodel(x, ind)
#     #     ax.plot(x, data, alpha=0.3)
