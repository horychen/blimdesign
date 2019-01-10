# coding:utf-8
#execfile('D:/OneDrive - UW-Madison/c/codes/pyfemm_script.py')
#execfile(r'K:\jchen782\JMAG\c\codes/pyfemm_script.py')

''' 0. Configuration
'''
# run_folder = r'run#13/'; deg_per_step = 6; run_list = [1,0,0,0,1]; # test StaticFEA in JMAG

# 端环有的：TranRef 10 Steps Per Cycle
# run_folder = r'run#12/'; deg_per_step = 0.5; run_list = [1,0,0,0,0]; # dense run: 0.1 deg
# run_folder = r'run#14/'; deg_per_step = 0.5; run_list = [1,0,0,1,0]; # Qr=32

# 验证FEMM和JMAG的结果匹配
# run_folder = r'run#15/'; deg_per_step = 0.5; run_list = [1,0,0,0,0]; # Qr=32

# 端环有的：TranRef 100 Steps Per Cycle
# run_folder = r'run#12/'; deg_per_step = 0.5; run_list = [1,0,0,0,0]; # Qr=36 TranRef with 100 Steps per cycle
# run_folder = r'run#16/'; deg_per_step = 6; run_list = [1,0,0,0,0]; # Qr=32 TranRef with 100 Steps per cycle

# 端环没的：TranRef 40 Steps Per Cycle

# No TranRef40 or 400 is ever needed again - 20180104

''' 1. General Information & Packages Loading
'''
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
        dir_parent = 'I:/jchen782/JMAG/c/'
        dir_codes = dir_parent + 'codes/'
        dir_lib = dir_parent + 'codes/'
        dir_femm_files = 'I:/jchen782/FEMM/'
        dir_project_files = 'I:/jchen782/JMAG/'
        pc_name = 'Seversion02'
    elif os.path.exists('K:/'):
        print 'you are on Severson01'
        dir_parent = 'K:/jchen782/JMAG/c/'
        dir_codes = dir_parent + 'codes/'
        dir_lib = dir_parent + 'codes/'
        dir_femm_files = 'K:/jchen782/FEMM/'
        dir_project_files = 'K:/jchen782/JMAG/'
        pc_name = 'Seversion01'
    # elif 'chen' in 'dir_interpreter':
    #     print 'you are on T440p'
    #     dir_parent = 'C:/Users/Hory Chen/OneDrive - UW-Madison/'
    #     dir_lib = dir_parent + 'codes2/'
    #     dir_initial_design = dir_parent + 'pop/'
    #     dir_csv_output_folder = dir_parent + 'csv_opti/'
    #     dir_femm_files = 'C:/femm42/'
    #     dir_project_files = 'C:/JMAG_Files/'
    #     pc_name = 't440p'
    else:
        print 'where are you???'
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
        if fea_config_dict['Restart'] == False:
            fea_config_dict['OnlyTableResults'] = True  # save disk space for my PC
    # However, we need field data for iron loss calculation
    fea_config_dict['OnlyTableResults'] = False 

fea_config_dict = {
    ##########################
    # Sysetm Controlc
    ##########################
    'Active_Qr':36, # 36
    'TranRef-StepPerCycle':40,
    'OnlyTableResults':False, # modified later according to pc_name
        # multiple cpu (SMP=2)
        # directSolver over ICCG Solver
    'Restart':False, # restart from frequency analysis is not needed, because SSATA is checked and JMAG 17103l version is used.
    'flag_optimization':True,

    ##########################
    # Design Specifications
    ##########################
    'DPNV': True,
    'End_Ring_Resistance':0, # 0 for consistency with FEMM with pre-determined currents # 9.69e-6, # this is still too small for Chiba's winding

    'Steel': 'M19Gauge29', 
    # 'Steel': 'M15',
    # 'Steel': 'Arnon5', 
    'Bar_Conductivity':1/((3.76*100+873)*1e-9/55.), # 40e6 for Aluminium; 1/((3.76*100+873)*1e-9/55.) for Copper
    # 'Bar_Conductivity':40e6, # 40e6 for Aluminium; 1/((3.76*100+873)*1e-9/55.) for Copper
}
where_am_i(fea_config_dict)
from sys import path as sys_path
sys_path.append(fea_config_dict['dir_lib'])
import population
import FEMM_Solver
import utility
reload(population) 
reload(FEMM_Solver)
logger = utility.myLogger(fea_config_dict['dir_codes'], prefix='iemdc_')

run_list = [1,1,0,0,0] 
run_folder = r'run#100/'
fea_config_dict['run_folder'] = run_folder
fea_config_dict['jmag_run_list'] = run_list
if fea_config_dict['End_Ring_Resistance'] == 0:
    fea_config_dict['model_name_prefix'] = 'PS_Qr%d_NoEndRing_%s'%(fea_config_dict['Active_Qr'], fea_config_dict['Steel'])
if fea_config_dict['DPNV'] == True:
    fea_config_dict['model_name_prefix'] += '_DPNV'
if fea_config_dict['Restart'] == True:
    fea_config_dict['model_name_prefix'] += '_Restart'
print fea_config_dict['model_name_prefix']

# fea_config_dict['femm_deg_per_step'] = 0.25 * (360/4) / utility.lcm(24/4., fea_config_dict['Active_Qr']/4.) # at least half period
fea_config_dict['femm_deg_per_step'] = 1 * (360/4) / utility.lcm(24/4., fea_config_dict['Active_Qr']/4.) # at least half period
fea_config_dict['femm_deg_per_step'] = 0.1 #0.5 # deg
print 'femm_deg_per_step is', fea_config_dict['femm_deg_per_step'], 'deg (Qs=24, p=2)'


''' 2. Initilize Swarm and Initial Pyrhonen's Design (Run this part in JMAG)
''' # 1e-1也还是太小了（第三次报错），至少0.5mm长吧 # 1e-1 is the least geometry value. a 1e-2 will leads to：转子闭口槽极限，会导致edge过小，从而报错：small arc entity exists.png
de_config_dict = {  'bounds':     [[3,9], [0.5,4], [5e-1,3], [1.5,8], [5e-1,3], [1,10], [5e-1,3]], 
                    'mut':        0.8,
                    'crossp':     0.7,
                    'popsize':    100,
                    'iterations': 20 } # begin at 5
# get initial design as im
sw = population.swarm(fea_config_dict, de_config_dict=None)
# sw.show(which='all')
# print sw.im.show()


# generate the initial generation
# sw.generate_pop()

im_initial = sw.im
print im_initial.l21
print im_initial.l22

''' 3. Initialize FEMM Solver
'''
# define problem
logger.info('Running Script for FEMM with %s'%(run_folder))
solver_jmag = FEMM_Solver.FEMM_Solver(im_initial, flag_read_from_jmag=True, freq=0) # static
solver_femm = FEMM_Solver.FEMM_Solver(im_initial, flag_read_from_jmag=False, freq=2.23) # eddy+static



# # debug
# if not solver_femm.has_results(solver_femm.dir_run + 'sweeping/'):
#     solver_femm.run_frequency_sweeping(range(1,6))
#     data_femm = solver_femm.show_results(bool_plot=False) # this write rotor currents to file, which will be used later for static FEA
# else:
#     solver_femm.show_results_eddycurrent(True) # get right slip
# print solver_femm.get_iron_loss(MAX_FREQUENCY=50e3)
# print solver_femm.get_iron_loss(MAX_FREQUENCY=30e3)
# print solver_femm.get_iron_loss(MAX_FREQUENCY=10e3)
# print solver_femm.get_iron_loss(MAX_FREQUENCY=5e3)
# from pylab import show; show()
# quit()
    # solver_femm.read_current_conditions_from_FEMM()
    # solver_femm.get_copper_loss()
    # quit()

    # solver_femm.keep_collecting_static_results_for_optimization()
    # # # # quit()
    # # # # # load Arnon5 from file
    # # # # sw.print_array()
    # # # # quit()



























''' 4. Show results, if not exist then produce it
'''
if False: # this generate plots for iemdc19
    data = solver_jmag.show_results(bool_plot=False)
    # sw.show_results(femm_solver_data=data)
    sw.show_results_iemdc19(femm_solver_data=data, femm_rotor_current_function=solver_jmag.get_rotor_current_function())
    # sw.timeStepSensitivity()
else:
    # FEMM Static Solver with pre-determined rotor currents from FEMM
    if not solver_femm.has_results():
        solver_femm.run_frequency_sweeping(range(1,6))
        data_femm = solver_femm.show_results(bool_plot=False) # this write rotor currents to file, which will be used later for static FEA

        solver_femm.run_rotating_static_FEA()
        solver_femm.parallel_solve()

    # JMAG 
    if not sw.has_results(im_initial, 'Freq'):
        sw.run(im_initial, run_list=sw.fea_config_dict['jmag_run_list'])
        if not sw.has_results(im_initial, 'Freq'):
            raise Exception('Something went south with JMAG.')

    print 'Rotor current solved by FEMM is shown here:'
    solver_femm.read_current_conditions_from_FEMM()

    print 'Rotor current solved by JMAG is shown here:'
    solver_jmag.read_current_from_EC_FEA()
    for key, item in solver_jmag.dict_rotor_current_from_EC_FEA.iteritems():
        if '1' in key:
            from math import sqrt
            print key, sqrt(item[0]**2+item[1]**2)

    # FEMM Static Solver with pre-determined rotor currents from JMAG
    if not solver_jmag.has_results():
        solver_jmag.run_rotating_static_FEA()
        solver_jmag.parallel_solve()


    # Compare JMAG and FEMM's rotor currents to see DPNV is implemented correctly in bot JMAG and FEMM    
    data_femm = solver_femm.show_results_static(bool_plot=False)
    data_jmag = solver_jmag.show_results_static(bool_plot=False)

    if data_femm is None or data_jmag is None:
        if data_femm is None:
            print 'data_femm is', data_femm, 'try run again'
        if data_jmag is None:
            print 'data_jmag is', data_jmag, 'try run again'
    else:
        from pylab import subplots
        fig, axes = subplots(3, 1, sharex=True)
        for data in [data_femm, data_jmag]:
            ax = axes[0]; ax.plot(data[0]*0.1, data[1], alpha=0.5, label='torque'); ax.legend(); ax.grid()
            ax = axes[1]; ax.plot(data[0]*0.1, data[2], alpha=0.5, label='Fx'); ax.legend(); ax.grid()
            ax = axes[2]; ax.plot(data[0]*0.1, data[3], alpha=0.5, label='Fy'); ax.legend(); ax.grid()



sw.write_to_file_fea_config_dict()
from pylab import show; show()


# if not sw.has_results(im_initial, study_type='Tran2TSS'):
#     os.system(r'set InsDir=D:\Program Files\JMAG-Designer17.1/')
#     os.system(r'set WorkDir=D:\JMAG_Files\JCF/')
#     os.system(r'cd /d "%InsDir%"')
#     for jcf_file in os.listdir(sw.dir_jcf):
#         if 'Tran2TSS' in jcf_file and 'Mesh' in jcf_file:
#             os.system('ExecSolver "%WorkDir%' + jcf_file + '"')


if False: # Test and Debug

    ''' 3. Eddy Current FEA with FEMM
    '''
    from numpy import arange
    solver = FEMM_Solver.FEMM_Solver(deg_per_step, im_initial, dir_codes, dir_femm_files + run_folder)

    solver.read_current_from_EC_FEA()
    for key, item in solver.dict_rotor_current_from_EC_FEA.iteritems():
        if '1' in key:
            from math import sqrt
            print key, sqrt(item[0]**2+item[1]**2)

    if solver.has_results():
        solver.show_results()
    else:
        solver.run_frequency_sweeping(arange(2, 5.1, 0.25), fraction=2)
        solver.parallel_solve(6, bool_watchdog_postproc='JMAG' not in dir_interpreter)


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

a=''' Loss Study of Transient FEA
# -*- coding: utf-8 -*-
app = designer.GetApplication()
app.GetModel(u"PS_ID32").GetStudy(u"Loss_Tran2TSS-Prolong").GetCondition(u"P1").SetValue(u"BasicFrequencyType", 1)
app.GetModel(u"PS_ID32").GetStudy(u"Loss_Tran2TSS-Prolong").GetCondition(u"P1").SetValue(u"RevolutionSpeed", 15000)
app.GetModel(u"PS_ID32").GetStudy(u"Loss_Tran2TSS-Prolong").GetCondition(u"P1").SetValue(u"StressType", 0)
app.GetModel(u"PS_ID32").GetStudy(u"Loss_Tran2TSS-Prolong").GetCondition(u"P1").ClearParts()
sel = app.GetModel(u"PS_ID32").GetStudy(u"Loss_Tran2TSS-Prolong").GetCondition(u"P1").GetSelection()
sel.SelectPart(35)
app.GetModel(u"PS_ID32").GetStudy(u"Loss_Tran2TSS-Prolong").GetCondition(u"P1").AddSelected(sel)
app.SetCurrentStudy(u"Loss_Tran2TSS-Prolong")
app.GetModel(u"PS_ID32").GetStudy(u"Loss_Tran2TSS-Prolong").CreateCondition(u"Ironloss", u"P2")
app.GetModel(u"PS_ID32").GetStudy(u"Loss_Tran2TSS-Prolong").GetCondition(u"P2").SetValue(u"BasicFrequencyType", 2)
app.GetModel(u"PS_ID32").GetStudy(u"Loss_Tran2TSS-Prolong").GetCondition(u"P2").SetValue(u"BasicFrequency", 3)
app.GetModel(u"PS_ID32").GetStudy(u"Loss_Tran2TSS-Prolong").GetCondition(u"P2").SetValue(u"StressType", 0)
app.GetModel(u"PS_ID32").GetStudy(u"Loss_Tran2TSS-Prolong").GetCondition(u"P2").ClearParts()
sel = app.GetModel(u"PS_ID32").GetStudy(u"Loss_Tran2TSS-Prolong").GetCondition(u"P2").GetSelection()
sel.SelectPart(2)
app.GetModel(u"PS_ID32").GetStudy(u"Loss_Tran2TSS-Prolong").GetCondition(u"P2").AddSelected(sel)
app.SetCurrentStudy(u"Loss_Tran2TSS-Prolong")
'''

