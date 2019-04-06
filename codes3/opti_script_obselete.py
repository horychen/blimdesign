import imp
# coding:u8
# execfile(r'D:\Users\horyc\OneDrive - UW-Madison\codes\opti_script.py')
# execfile(r'C:\Users\Hory Chen\OneDrive - UW-Madison\codes\opti_script.py')


def where_am_i():
    from os import path
    pc_name = None
    if path.isdir("D:"): 
        # print 'you are on Legion Y730'
        dir_project_files = 'D:/JMAG_Files/'
        dir_initial_design = 'D:/Users/horyc/OneDrive - UW-Madison/pop/'
        dir_csv_output_folder = "D:/Users/horyc/OneDrive - UW-Madison/csv_opti/"
        pc_name = 'Y730'
    elif path.isdir("I:"):
        # print 'you are on Severson02'
        dir_project_files = 'I:/jchen782/JMAG/'
        dir_initial_design = 'I:/jchen782/JMAG/pop/'
        dir_csv_output_folder = "I:/jchen782/JMAG/csv_opti/"
    elif path.isdir("K:"):
        # print 'you are on Severson01'
        dir_project_files = 'K:/jchen782/JMAG/'
        dir_initial_design = 'K:/jchen782/JMAG/pop/'
        dir_csv_output_folder = "K:/jchen782/JMAG/csv_opti/"
    else:
        print('you are on T440p')
        dir_project_files = 'C:/JMAG_Files/'
        dir_initial_design = 'C:/Users/hory chen/OneDrive - UW-Madison/pop/'
        dir_csv_output_folder = "C:/Users/hory chen/OneDrive - UW-Madison/csv_opti/"

    return dir_project_files, dir_initial_design, dir_csv_output_folder, pc_name


# add Pyrhonen deisng to the swarm.
# select the best five individuals from the first solve
class Pyrhonen_design(object):
    def __init__(self):
        ''' Determine bounds for these parameters:
            stator_tooth_width_b_ds       = design_parameters[0]*1e-3 # m                       # stator tooth width [mm]
            air_gap_length_delta          = design_parameters[1]*1e-3 # m                       # air gap length [mm]
            b1                            = design_parameters[2]*1e-3 # m                       # rotor slot opening [mm]
            rotor_tooth_width_b_dr        = design_parameters[3]*1e-3 # m                       # rotor tooth width [mm]
            self.Length_HeadNeckRotorSlot        = design_parameters[4]             # [4]       # rotor tooth head & neck length [mm]
            self.Angle_StatorSlotOpen            = design_parameters[5]             # [5]       # stator slot opening [deg]
            self.Width_StatorTeethHeadThickness  = design_parameters[6]             # [6]       # stator tooth head length [mm]
        '''
        self.stator_tooth_width_b_ds = 0.007457249477380651* 1e3
        self.air_gap_length_delta = 0.00126942993991* 1e3
        self.b1 = 0.000959197921278* 1e3
        self.rotor_tooth_width_b_dr = 0.005250074634166455* 1e3
        self.Length_HeadNeckRotorSlot = 1.0
        self.Angle_StatorSlotOpen = 3.0
        self.Width_StatorTeethHeadThickness = 1

        self.design_parameters_denorm = [  self.stator_tooth_width_b_ds,
                                    self.air_gap_length_delta,
                                    self.b1,
                                    self.rotor_tooth_width_b_dr,
                                    self.Length_HeadNeckRotorSlot,
                                    self.Angle_StatorSlotOpen,
                                    self.Width_StatorTeethHeadThickness]

    def show_denorm(self):
        print(self.design_parameters_denorm)

    def show_norm(self, bounds):
        import numpy as np
        min_b, max_b = np.asarray(bounds).T 
        diff = np.fabs(min_b - max_b)
        self.design_parameters_norm = (self.design_parameters_denorm - min_b)/diff #= pop
        print(type(self.design_parameters_norm))
        for el in self.design_parameters_norm:
            print(el, ',', end=' ')
        print() 

        pop = self.design_parameters_norm
        min_b, max_b = np.asarray(bounds).T 
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        print(pop_denorm)
        print(self.design_parameters_denorm)


if __name__ == '__main__':
    dir_project_files, dir_initial_design, dir_csv_output_folder, pc_name = where_am_i()
    dir_codes = dir_initial_design[:-4] + r'codes/'
    from sys import path as sys_path
    sys_path.append(dir_codes)
    import population
    import utility
    imp.reload(population) 

    # testing
    try:
        if os.path.exists(dir_codes + r'opti_script.log'):
            os.remove(dir_codes + r'opti_script.log')
    except Exception as e:
        print('Repeated run of script in JMAG Designer is detected. As a result, you will see duplicated logging entries because more than one logger is activated.')
    # logger = logger_init()
    logger = utility.myLogger(dir_codes)

    ''' Determine bounds for these parameters:
        stator_tooth_width_b_ds       = design_parameters[0]*1e-3 # m                       # stator tooth width [mm]
        air_gap_length_delta          = design_parameters[1]*1e-3 # m                       # air gap length [mm]
        b1                            = design_parameters[2]*1e-3 # m                       # rotor slot opening [mm]
        rotor_tooth_width_b_dr        = design_parameters[3]*1e-3 # m                       # rotor tooth width [mm]
        self.Length_HeadNeckRotorSlot        = design_parameters[4]             # [4]       # rotor tooth head & neck length [mm]
        self.Angle_StatorSlotOpen            = design_parameters[5]             # [5]       # stator slot opening [deg]
        self.Width_StatorTeethHeadThickness  = design_parameters[6]             # [6]       # stator tooth head length [mm]
    '''
    # 1e-1也还是太小了（第三次报错），至少0.5mm长吧
    de_config_dict = { 'bounds':     [[3,9], [0.5,4], [5e-1,3], [1.5,8], [5e-1,3], [1,10], [5e-1,3]], # 1e-1 is the least geometry value. a 1e-2 will leads to：转子闭口槽极限，会导致edge过小，从而报错：small arc entity exists.png
                    'mut':        0.8,
                    'crossp':     0.7,
                    'popsize':    100,
                    'iterations': 20 } # begin at 5

    ''' Remember to check the file 'initial_design.txt'.
    '''
    # generate the initial generation
    sw = population.swarm(dir_initial_design, r'run#1/', dir_csv_output_folder, dir_project_files, model_name_prefix='BLIM_PS', de_config_dict=de_config_dict, pc_name=pc_name)
    # logger.info(sw.show('all', toString=True))

    # the_initial_design = Pyrhonen_design()
    # the_initial_design.show_norm(de_config_dict['bounds'])



    # de_config_dict = { 'bounds':     [[3,9], [0.5,4], [5e-1,3], [1.5,8], [5e-1,3], [1,10], [5e-1,3]], # 1e-1 is the least geometry value. a 1e-2 will leads to：转子闭口槽极限，会导致edge过小，从而报错：small arc entity exists.png
    #                 'mut':        0.8,
    #                 'crossp':     0.7,
    #                 'popsize':    4,
    #                 'iterations': 1 } # begin at 5
    # sw = population.swarm(dir_initial_design, r'run#36/', dir_csv_output_folder, dir_project_files, de_config_dict, pc_name=pc_name)



    if False:
        # run
        de_generator = sw.de()
        result = list(de_generator)

        # de version 2 is called by this
        for el in result:
            pop, fit, idx = el
            logger.info('some information')
            print('pop:' + pop + 'fit & index:' %(fit, idx))
            # for ind in pop:
            #     data = fmodel(x, ind)
            #     ax.plot(x, data, alpha=0.3)
        
        # de version 1 is called by this:
        # result = list(de(fobj,bounds))
        # print(result[-1])
    else:
        sw.plot_csv_results_for_all()



