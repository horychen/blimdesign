# coding:utf-8
#execfile('D:/OneDrive - UW-Madison/c/codes/opti_script.py')
#execfile(r'K:\jchen782\JMAG\c\codes/opti_script.py')

''' 1. General Information & Packages Loading
'''
import os
# Debug
# if os.path.exists('d:/femm42/PS_Qr32_NoEndRing_M19Gauge29_DPNV_1e3Hz'):
#     os.system('bash -c "rm -r /mnt/d/femm42/PS_Qr32_NoEndRing_M19Gauge29_DPNV_1e3Hz"')
# if os.path.exists('d:/OneDrive - UW-Madison/c/pop/Tran2TSS_PS_Opti.txt'):
#     os.system('bash -c "mv /mnt/d/OneDrive\ -\ UW-Madison/c/pop/Tran2TSS_PS_Opti.txt /mnt/d/OneDrive\ -\ UW-Madison/c/pop/initial_design.txt"')
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
        # T440p
        print 'where are you? T440p?'
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
    'Active_Qr':32, # 36
    'TranRef-StepPerCycle':40,
    'OnlyTableResults':False, # modified later according to pc_name
        # multiple cpu (SMP=2)
        # directSolver over ICCG Solver
    'Restart':False, # restart from frequency analysis is not needed, because SSATA is checked and JMAG 17103l version is used.
    'flag_optimization':True,

    ##########################
    # Optimization
    ##########################
    'FrequencyRange':range(1,6), # first generation for PSO

    ##########################
    # Design Specifications
    ##########################
    'DPNV': True,
    'End_Ring_Resistance':0, # 0 for consistency with FEMM with pre-determined currents # 9.69e-6, # this is still too small for Chiba's winding

    'Steel': 'M19Gauge29', 
    # 'Steel': 'M15',
    # 'Steel': 'Arnon5', 
    'Bar_Conductivity':1/((3.76*25+873)*1e-9/55.), # 1/((3.76*100+873)*1e-9/55.) for Copper, where temperature is 25 or 100 deg Celsius.
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
logger = utility.myLogger(fea_config_dict['dir_codes'], prefix='ecce_')

run_list = [1,1,0,0,0] 
run_folder = r'run#100/'
fea_config_dict['run_folder'] = run_folder
fea_config_dict['jmag_run_list'] = run_list
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
print fea_config_dict['model_name_prefix']

fea_config_dict['femm_deg_per_step'] = 0.25 * (360/4) / utility.lcm(24/4., fea_config_dict['Active_Qr']/4.) # at least half period
# fea_config_dict['femm_deg_per_step'] = 1 * (360/4) / utility.lcm(24/4., fea_config_dict['Active_Qr']/4.) # at least half period
# fea_config_dict['femm_deg_per_step'] = 0.1 #0.5 # deg
print 'femm_deg_per_step is', fea_config_dict['femm_deg_per_step'], 'deg (Qs=24, p=2)'


''' 2. Initilize Swarm and Initial Pyrhonen's Design (Run this part in JMAG)
''' # 1e-1也还是太小了（第三次报错），至少0.5mm长吧 # 1e-1 is the least geometry value. a 1e-2 will leads to：转子闭口槽极限，会导致edge过小，从而报错：small arc entity exists.png
if fea_config_dict['flag_optimization'] == True:
    de_config_dict = {  'bounds':     [[3,9], [0.5,4], [5e-1,3], [1.5,8], [5e-1,3], [1,10], [5e-1,3]], 
                        'mut':        0.8,
                        'crossp':     0.7,
                        'popsize':    20,
                        'iterations': 48 } # begin at 5
else:
    de_config_dict = None

# init the swarm
sw = population.swarm(fea_config_dict, de_config_dict=de_config_dict)
# sw.show(which='all')

# generate the initial generation
sw.generate_pop()
logger.info('Initial pop generated.')



''' 3. Initialize FEMM Solver
'''
if fea_config_dict['jmag_run_list'][0] == 0:
    # and let jmag know about it
    sw.solver_femm = FEMM_Solver.FEMM_Solver(sw.im, flag_read_from_jmag=False, freq=2.23) # eddy+static



''' 4. Run DE Optimization
'''

# run
de_generator = sw.de()
result = list(de_generator)

# de version 2 is called by this
for el in result:
    pop, fit, idx = el
    logger.info('some information')
    print 'pop:' + pop + 'fit & index:' %(fit, idx)
    # for ind in pop:
    #     data = fmodel(x, ind)
    #     ax.plot(x, data, alpha=0.3)

sw.write_to_file_fea_config_dict()


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
