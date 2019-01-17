# -*- coding: utf-8 -*-
# execfile(r'D:\Users\horyc\OneDrive - UW-Madison\ec_rotate.py') # , {'__name__': 'load'})
from __future__ import division
from math import cos, sin, pi
from csv import reader as csv_reader
import logging
import numpy as np  # for de
import os
from pylab import plot, legend, grid, figure, subplots, array, mpl, show
import utility
from VanGogh import VanGogh

from time import time as clock_time

EPS = 1e-2

class suspension_force_vector(object):
    """docstring for suspension_force_vector"""
    def __init__(self, force_x, force_y, range_ss=None): # range_ss means range_steadystate
        super(suspension_force_vector, self).__init__()
        self.force_x = force_x
        self.force_y = force_y
        self.force_ang = np.arctan2(force_y, force_x) / pi * 180 # [deg]
        self.force_abs = np.sqrt(np.array(force_x)**2 + np.array(force_y)**2 )

        if range_ss == None:
            range_ss = len(force_x)
        self.range_ss = range_ss

        self.ss_avg_force_vector    = np.array([sum(force_x[-range_ss:]), sum(force_y[-range_ss:])]) / range_ss #len(force_x[-range_ss:])
        self.ss_avg_force_angle     = np.arctan2(self.ss_avg_force_vector[1], self.ss_avg_force_vector[0]) / pi * 180
        self.ss_avg_force_magnitude = np.sqrt(self.ss_avg_force_vector[0]**2 + self.ss_avg_force_vector[1]**2)

        self.force_err_ang = self.force_ang - self.ss_avg_force_angle
        self.force_err_abs = self.force_abs - self.ss_avg_force_magnitude

        self.ss_max_force_err_ang = max(self.force_err_ang[-range_ss:]), min(self.force_err_ang[-range_ss:])
        self.ss_max_force_err_abs = max(self.force_err_abs[-range_ss:]), min(self.force_err_abs[-range_ss:])

class data_manager(object):

    def __init__(self):
        self.basic_info = []
        self.time_list = []
        self.TorCon_list = []
        self.ForConX_list = []
        self.ForConY_list = []
        self.ForConAbs_list = []

        self.jmag_loss_list = None
        self.femm_loss_list = None

    def unpack(self):
        return self.basic_info, self.time_list, self.TorCon_list, self.ForConX_list, self.ForConY_list, self.ForConAbs_list

    def terminal_voltage(self, which='4C'): # 2A 2B 2C 4A 4B 4C
        return self.Current_dict['Terminal%s [Case 1]'%(which)]
        # 端点电压是相电压吗？应该是，我们在中性点设置了地电位

    def circuit_current(self, which='4C'): # 2A 2B 2C 4A 4B 4C
        return self.Current_dict['Coil%s'%(which)]

    def power_factor(self, number_of_steps_2ndTTS, targetFreq=1e3, numPeriodicalExtension=1000):
        # for key, val in self.Current_dict.iteritems():
        #     if 'Terminal' in key:
        #         print key, val
        # quit()

        # 4C
        mytime  = self.Current_dict['Time(s)'][-number_of_steps_2ndTTS:]
        voltage =      self.terminal_voltage()[-number_of_steps_2ndTTS:]
        current =       self.circuit_current()[-number_of_steps_2ndTTS:]
        # from pylab import *
        # print len(mytime), len(voltage), len(current)
        # figure()
        # plot(mytime, voltage)
        # plot(mytime, current)
        # show()
        power_factor = utility.compute_power_factor_from_half_period(voltage, current, mytime, targetFreq=targetFreq, numPeriodicalExtension=numPeriodicalExtension)
        return power_factor

class swarm(object):

    def __init__(self, fea_config_dict, de_config_dict=None):
        # directories part I
        self.dir_parent             = fea_config_dict['dir_parent']
        self.initial_design_file    = self.dir_parent + 'pop/' + r'initial_design.txt'

        # load initial design using the obsolete class bearingless_induction_motor_design
        self.im_list = []
        with open(self.initial_design_file, 'r') as f: 
            read_iterator = csv_reader(f, skipinitialspace=True)
            for row in self.whole_row_reader(read_iterator):
                im = bearingless_induction_motor_design([row[0]]+[float(el) for el in row[1:]], fea_config_dict, model_name_prefix=fea_config_dict['model_name_prefix'])
                self.im_list.append(im)
        for im in self.im_list:
            if im.Qr == fea_config_dict['Active_Qr']:
                self.im = im
        try: 
            self.im
        except:
            print 'There is no design matching Active_Qr.'
            msg = 'Please activate one initial design. Refer %s.' % (self.initial_design_file)
            logger = logging.getLogger(__name__)
            logger.warn(msg)
            raise Exception('no match for Active_Qr')

        # directories part II        
        if im.DriveW_Freq == 1000: # New design for 1000 Hz machine. some patch for my scrappy codes (lot of bugs are fixed, we need a new name).
            im.model_name_prefix += '_1e3Hz'
            self.model_name_prefix = im.model_name_prefix
            fea_config_dict['model_name_prefix'] = im.model_name_prefix
            print '[Updated] New model_name_prefix is', self.model_name_prefix
        else:
            # design objective
            self.model_name_prefix = fea_config_dict['model_name_prefix']

        # csv output folder
        if fea_config_dict['flag_optimization'] == False:
            self.dir_csv_output_folder  = self.dir_parent + 'csv/' + self.model_name_prefix + '/'
        else:
            self.dir_csv_output_folder  = self.dir_parent + 'csv_opti/' + self.model_name_prefix + '/'
        if not os.path.exists(self.dir_csv_output_folder):
            os.makedirs(self.dir_csv_output_folder)

        self.run_folder             = fea_config_dict['run_folder']
        self.dir_run                = self.dir_parent + 'pop/' + self.run_folder
        self.dir_project_files      = fea_config_dict['dir_project_files']
        self.dir_jcf                = self.dir_project_files + 'jcf/'
        self.pc_name                = fea_config_dict['pc_name']
        self.fea_config_dict        = fea_config_dict

        # dict of optimization
        self.de_config_dict = de_config_dict
        self.bool_first_time_call_de = True

        # post-process feature
        self.fig_main, self.axeses = subplots(2, 2, sharex=True, dpi=150, figsize=(16, 8), facecolor='w', edgecolor='k')
        self.pyplot_clear()

    def pyplot_clear(self):
        # self.fig_main.clf()
        axeses = self.axeses
        for ax in [axeses[0][0],axeses[0][1],axeses[1][0],axeses[1][1]]:
            ax.cla()
            ax.grid()
        ax = axeses[0][0]; ax.set_xlabel('(a)',fontsize=14.5); ax.set_ylabel('Torque [Nm]',fontsize=14.5)
        ax = axeses[0][1]; ax.set_xlabel('(b)',fontsize=14.5); ax.set_ylabel('Force Amplitude [N]',fontsize=14.5)
        ax = axeses[1][0]; ax.set_xlabel('Time [s]\n(c)',fontsize=14.5); ax.set_ylabel('Normalized Force Error Magnitude [%]',fontsize=14.5)
        ax = axeses[1][1]; ax.set_xlabel('Time [s]\n(d)',fontsize=14.5); ax.set_ylabel('Force Error Angle [deg]',fontsize=14.5)

    def write_to_file_fea_config_dict(self):
        with open(self.dir_run + '../FEA_CONFIG-%s.txt'%(self.fea_config_dict['model_name_prefix']), 'w') as f:
            for key, val in self.fea_config_dict.iteritems():
                # print key, val
                f.write('%s:%s\n' % (key, str(val)) )

    def generate_pop(self):
        # csv_output folder is used for optimziation
        self.dir_csv_output_folder  = self.fea_config_dict['dir_parent'] + 'csv_opti/' + self.run_folder

        # check if it is a new run
        if not os.path.exists(self.dir_run):
            logger = logging.getLogger(__name__)
            logger.debug('There is no run yet. Generate the run folder for pop as %s...', self.run_folder)
            os.makedirs(self.dir_run)
        if not os.path.exists(self.dir_csv_output_folder):
            try:  
                os.makedirs(self.dir_csv_output_folder)
            except OSError, e:
                logging.getLogger(__name__).error("Creation of the directory %s failed" % self.dir_csv_output_folder, exc_info=True)
                raise e
            else:
                logging.getLogger(__name__).info("Successfully created the directory %s " % self.dir_csv_output_folder)

        # the run folder has been created. check the pop data files
        self.index_interrupt_beginning = 0 # 不出意外，就从第一个个体开始迭代。但是很多时候跑到第150个个体的时候跑断了，你总不会想要把这一代都删了，重新跑吧？
        for file in os.listdir(self.dir_run):
            if 'ongoing' in file:
                # remove gen and fit files
                # os.remove(self.dir_run + file) 

                if 'gen' in file:
                    print file

                    self.interrupt_pop_denorm = []
                    with open(self.dir_run + file, 'r') as f:
                        read_iterator = csv_reader(f, skipinitialspace=True)
                        for row in self.whole_row_reader(read_iterator):
                            if len(row)>0: # there could be empty row, since we use append mode and write down f.write('\n')
                                self.interrupt_pop_denorm.append([float(el) for el in row])

                    self.interrupt_pop_denorm = np.asarray(self.interrupt_pop_denorm)
                    self.index_interrupt_beginning = len(self.interrupt_pop_denorm)

                    logger = logging.getLogger(__name__)
                    logger.warn('Unfinished iteration is found with ongoing files in run folder.') # Make sure the project is not opened in JMAG, and we are going to remove the project files and ongoing files.')
                    # logger.warn(u'不出意外，就从第一个个体开始迭代。但是很多时候跑到第150个个体的时候跑断了，你总不会想要把这一代都删了，重新跑吧？')
                    # os.remove(u"D:/JMAG_Files/" + run_folder[:-1] + file[:-12] + ".jproj")
                    print 'List interrupt_pop_denorm here:', self.interrupt_pop_denorm.tolist()

                if 'fit' in file:
                    self.interrupt_fitness = []
                    with open(self.dir_run + file, 'r') as f: 
                        read_iterator = csv_reader(f, skipinitialspace=True)
                        for row in self.whole_row_reader(read_iterator):
                            if len(row)>0: # there could be empty row, since we use append mode and write down f.write('\n')
                                self.interrupt_fitness.append(float(row[0])) # not neccessary to be an array. a list is enough
                    print 'List interrupt_fitness here:', self.interrupt_fitness
                    # interrupt_pop (yes) and interrupt_fitness will be directly used in de.
                    
        # search for complete generation files
        generations = [file[4:8] for file in os.listdir(self.dir_run) if 'gen' in file and not 'ongoing' in file]
        popsize = self.de_config_dict['popsize']
        # the least popsize is 4
        if popsize<=3:
            logger = logging.getLogger(__name__)
            logger.error('The popsize must be greater than 3 so the choice function can pick up three other individuals different from the individual under mutation among the population')
            raise Exception('Specify a popsize larger than 3.')
        bounds = self.de_config_dict['bounds']
        dimensions = len(bounds)
        min_b, max_b = np.asarray(bounds).T 
        diff = np.fabs(min_b - max_b)
        if self.index_interrupt_beginning != 0:
                  # pop_denorm = min_b + pop * diff =>
            self.interrupt_pop = (self.interrupt_pop_denorm - min_b) / diff

        # check for number of generations
        if len(generations) == 0:
            logger = logging.getLogger(__name__)
            logger.debug('There is no swarm yet. Generate the initial random swarm...')
            
            # generate the initial random swarm from the initial design
            if self.de_config_dict == None:
                logger.error(u'ちゃんとキーを設定して下さい。', exc_info=True)
                raise Exception('unexpected de_config_dict')
            self.init_pop = np.random.rand(popsize, dimensions) # normalized design parameters between 0 and 1
            self.init_pop_denorm = min_b + self.init_pop * diff

            # and save to file named gen#0000.txt
            self.number_current_generation = 0
            self.write_population_data(self.init_pop_denorm)
            logger = logging.getLogger(__name__)
            logger.debug('Initial pop (de-normalized) is saved as %s', self.dir_run + 'gen#0000.txt')
        else:
            # get the latest generation of swarm data
            self.number_current_generation = max([int(el) for el in generations])

            logger = logging.getLogger(__name__)
            logger.debug('The latest generation is gen#%d', self.number_current_generation)

            # restore the living pop from file liv#xxxx.txt
            self.init_pop_denorm = self.read_living_pop(self.number_current_generation)
                # 在2019年1月16日以前，每次断了以后，就丢失了真正活着的那组pop，真正的做法是，从第零代开始，根据fit和gen数据当前代重构活着的那组pop，但是这样太麻烦了，我决定加一个文件叫liv#xxx.txt。
                # 下面这个，直接取上一代gen#xxxx.txt的那组pop来作为当前代，显然是错的，因为我们是从fobj返回以后直接写入文件的，不管它是更好的还是更差了。
                # 也就是说，只有当number_current_generation=0的时候，这才是对的。
                # with open(self.get_gen_file(self.number_current_generation), 'r') as f:
                #     read_iterator = csv_reader(f, skipinitialspace=True)
                #     for row in self.whole_row_reader(read_iterator):
                #         if len(row)>0: # there could be empty row, since we use append mode and write down f.write('\n')
                #             self.init_pop_denorm.append([float(el) for el in row])
            self.init_pop = (np.array(self.init_pop_denorm) - min_b) / diff

            # TODO: 文件完整性检查
            solved_popsize = len(self.init_pop)
            if popsize > solved_popsize:
                print 'popsize is changed from last run.'
                self.init_pop = self.init_pop.tolist()
                self.init_pop += np.random.rand(popsize-solved_popsize, dimensions).tolist() # normalized design parameters between 0 and 1
                self.init_pop_denorm = min_b + self.init_pop * diff
                self.append_population_data(self.init_pop_denorm[solved_popsize:])
                # print self.init_pop

        # pop: make sure the data type is array
        self.init_pop = np.asarray(self.init_pop)

        # app for jmag
        self.app = None
        self.jmag_control_state = False # indicating that by default, the jmag designer is already opened but the project file is not yet loaded or created.

        logger = logging.getLogger(__name__)
        logger.info('Swarm is generated.')

    def designer_init(self):
        try:
            if self.app is None:
                import designer
                self.app = designer.GetApplication() 
        except:
            import designer
            self.app = designer.GetApplication() 

        def add_steel(self):
            if 'M15' in self.fea_config_dict['Steel']:
                add_M1xSteel(self.app, self.dir_parent, steel_name=u"M-15 Steel")
            elif 'M19' in self.fea_config_dict['Steel']:
                add_M1xSteel(self.app, self.dir_parent)
            elif 'Arnon5' == self.fea_config_dict['Steel']:
                add_Arnon5(self.app, self.dir_parent)            

        # too avoid tons of the same material in JAMG's material library
        if not os.path.exists(self.dir_parent + '.jmag_state.txt'):
            with open(self.dir_parent + '.jmag_state.txt', 'w') as f:
                f.write(self.fea_config_dict['Steel'] + '\n')
            add_steel(self)
        else:
            with open(self.dir_parent + '.jmag_state.txt', 'r') as f:
                for line in f.readlines():
                    if self.fea_config_dict['Steel'] not in line:
                        add_steel(self)

    def get_gen_file(self, no_generation, ongoing=False):
        if ongoing == True:
            return self.dir_run + 'gen#%04d-ongoing.txt'%(int(no_generation))
        else:
            return self.dir_run + 'gen#%04d.txt'%(int(no_generation))

    def get_fit_file(self, no_generation, ongoing=False):
        if ongoing == True:
            return self.dir_run + 'fit#%04d-ongoing.txt'%(int(no_generation))
        else:
            return self.dir_run + 'fit#%04d.txt'%(int(no_generation))

    def rename_onging_files(self, no_generation):
        os.rename(  self.dir_run + 'fit#%04d-ongoing.txt'%(int(no_generation)),
                    self.dir_run + 'fit#%04d.txt'%(int(no_generation)))
        os.rename(  self.dir_run + 'gen#%04d-ongoing.txt'%(int(no_generation)),
                    self.dir_run + 'gen#%04d.txt'%(int(no_generation)))

    def whole_row_reader(self, reader):
        for row in reader:
            yield row[:]

    def show(self, which=0, toString=False):
        out_string = ''

        if which == 'all':
            for el in self.im_list:
                out_string += el.show(toString)
        else:
            self.im_list[which].show(toString)

        return out_string

    @staticmethod
    def add_plot(axeses, title=None, label=None, zorder=None, time_list=None, sfv=None, torque=None, range_ss=None, alpha=0.7):

        results = '%s' % (title)
        torque_average = sum(torque[-range_ss:])/len(torque[-range_ss:])
        results += '\nAverage Torque: %g Nm' % (torque_average)
        # torque error = torque - avg. torque
        torque_error = np.array(torque) - torque_average
        ss_max_torque_error = max(torque_error[-range_ss:]), min(torque_error[-range_ss:])
        # we use peak value to compute error rather than use peak-to-peak value
        normalized_torque_ripple   = 0.5*(ss_max_torque_error[0] - ss_max_torque_error[1]) / torque_average
        results += '\nNormalized Torque Ripple: %g %%' % (normalized_torque_ripple*100)

        results += '\nAverage Force Mag: %g N'% (sfv.ss_avg_force_magnitude)
        # we use peak value to compute error rather than use peak-to-peak value
        normalized_force_error_magnitude = 0.5*(sfv.ss_max_force_err_abs[0]-sfv.ss_max_force_err_abs[1])/sfv.ss_avg_force_magnitude
        results += '\nNormalized Force Error Mag: %g%%, (+)%g%% (-)%g%%' % (normalized_force_error_magnitude*100,
                                                                      sfv.ss_max_force_err_abs[0]/sfv.ss_avg_force_magnitude*100,
                                                                      sfv.ss_max_force_err_abs[1]/sfv.ss_avg_force_magnitude*100)
        # we use peak value to compute error rather than use peak-to-peak value
        force_error_angle= 0.5*(sfv.ss_max_force_err_ang[0]-sfv.ss_max_force_err_ang[1])
        results += '\nMaximum Force Error Angle: %g [deg], (+)%g deg (-)%g deg' % (force_error_angle,
                                                                     sfv.ss_max_force_err_ang[0],
                                                                     sfv.ss_max_force_err_ang[1])
        results += '\nExtra Info:'
        results += '\n\tAverage Force Vecotr: (%g, %g) N' % (sfv.ss_avg_force_vector[0], sfv.ss_avg_force_vector[1])
        results += '\n\tTorque Ripple (Peak-to-Peak) %g Nm'% ( max(torque[-range_ss:]) - min(torque[-range_ss:]))
        results += '\n\tForce Mag Ripple (Peak-to-Peak) %g N'% (sfv.ss_max_force_err_abs[0] - sfv.ss_max_force_err_abs[1])

        ax = axeses[0][0]; ax.plot(time_list, torque, alpha=alpha, label=label, zorder=zorder)
        ax = axeses[0][1]; ax.plot(time_list, sfv.force_abs, alpha=alpha, label=label, zorder=zorder)
        ax = axeses[1][0]; ax.plot(time_list, 100*sfv.force_err_abs/sfv.ss_avg_force_magnitude, label=label, alpha=alpha, zorder=zorder)
        ax = axeses[1][1]; ax.plot(time_list, np.arctan2(sfv.force_y, sfv.force_x)/pi*180. - sfv.ss_avg_force_angle, label=label, alpha=alpha, zorder=zorder)

        return results, torque_average, normalized_torque_ripple, sfv.ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle


    # @blockPrinting
    def fobj(self, individual_index, individual):
        # based on the individual data, create design variant of the initial design of Pyrhonen09
        logger = logging.getLogger(__name__)
        mylatch = self.im
        self.im = None # to avoid to use this reference by mistake
        im_variant = bearingless_induction_motor_design.local_design_variant(mylatch, \
                        self.number_current_generation, individual_index, individual) # due to compatability issues: a new child class is used instead
        self.im_variant = im_variant # for command line access debug purpose
        im = im_variant # for Tran2TSS (应该给它弄个函数调用的)
        im_variant.individual_name = im_variant.get_individual_name() 

        self.project_name = self.run_folder[:-1]+'gen#%04dind#%04d' % (self.number_current_generation, individual_index)
        self.jmag_control_state = False

        # local scripts
        def open_jmag():
            #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            # Initialize JMAG Designer
            #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            # every model is a new project, so as to avoid the situation of 100 models in one project (occupy RAM and slow). add individual_index to project_name
            self.designer_init()
            app = self.app
            if self.jmag_control_state == False: # initilize JMAG Designer
                expected_project_file = self.dir_project_files + "%s.jproj"%(self.project_name)
                if not os.path.exists(expected_project_file):
                    app.NewProject(u"Untitled")
                    app.SaveAs(expected_project_file)
                    logger.debug('Create JMAG project file: %s'%(expected_project_file))
                else:
                    app.Load(expected_project_file)
                    logger.debug('Load JMAG project file: %s'%(expected_project_file))
                    logger.debug('Existing models of %d are found in %s', app.NumModels(), app.GetDefaultModelFolderPath())

                    # this `if' is obselete. it is used when a project contains 100 models.
                    # if app.NumModels() <= individual_index:
                    #     logger.warn('Some models are not plotted because of bad bounds (some lower bound is too small)! individual_index=%d, NumModels()=%d. See also the fit#%04d.txt file for 99999. There will be no .png file for these individuals either.', individual_index, app.NumModels(), self.number_current_generation)

                    # print app.NumStudies()
                    # print app.NumAnalysisGroups()
                    # app.SubmitAllModelsLocal() # we'd better do it one by one for easing the programing?

            self.jmag_control_state = True # indicating that the jmag project is already created
            return app

        def draw_jmag():
            #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            # Draw the model in JMAG Designer
            #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            DRAW_SUCCESS = self.draw_jmag_model(individual_index, 
                                                im_variant,
                                                im_variant.individual_name)
            if DRAW_SUCCESS == 0:
                # TODO: skip this model and its evaluation
                cost_function = 99999 # penalty
                logging.getLogger(__name__).warn('Draw Failed for %s-%s\nCost function penalty = %g.%s', self.project_name, im_variant.individual_name, cost_function, self.im_variant.show(toString=True))
                raise Exception('Draw Failed: Are you working on the PC? Sometime you by mistake operate in the JMAG Geometry Editor, then it fails to draw.')
                return None
            elif DRAW_SUCCESS == -1:
                # The model already exists
                print 'Model Already Exists'
                logging.getLogger(__name__).debug('Model Already Exists')
            # Tip: 在JMAG Designer中DEBUG的时候，删掉模型，必须要手动save一下，否则再运行脚本重新load project的话，是没有删除成功的，只是删掉了model的name，新导入进来的model name与project name一致。
            
            # JMAG
            if app.NumModels()>=1:
                model = app.GetModel(im_variant.individual_name)
            else:
                logger.error('there is no model yet!')
                raise Exception('why is there no model yet?')
            return model

        def exe_frequency():
            if True:
                # Freq Sweeping for break-down Torque Slip
                if model.NumStudies() == 0:
                    study = im_variant.add_study(app, model, self.dir_csv_output_folder, choose_study_type='frequency')
                else:
                    # there is already a study. then get the first study.
                    study = model.GetStudy(0)

                self.mesh_study(im_variant, app, model, study)
                self.run_study(im_variant, app, study, clock_time())

                # evaluation based on the csv results
                try:
                    slip_freq_breakdown_torque, breakdown_torque, breakdown_force = self.check_csv_results(study.GetName())
                except IOError, e:
                    msg = 'CJH: The solver did not exit with results, so reading the csv files reports an IO error. It is highly possible that some lower bound is too small.'
                    logger.error(msg + self.im_variant.show(toString=True))
                    print msg
                    # raise e
                    breakdown_torque = 0
                    breakdown_force = 0
                # self.fitness_in_physics_data # torque density, torque ripple, force density, force magnitude error, force angle error, efficiency, material cost 
            return slip_freq_breakdown_torque, breakdown_torque, breakdown_force

        def load_transeint():
            #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            # Load Results for Tran2TSS
            #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            try:
                dm = self.read_csv_results_4_optimization(tran2tss_study_name)
                basic_info, time_list, TorCon_list, ForConX_list, ForConY_list, ForConAbs_list = dm.unpack()
                sfv = suspension_force_vector(ForConX_list, ForConY_list, range_ss=self.fea_config_dict['number_of_steps_2ndTTS']) # samples in the tail that are in steady state
                str_results, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle = \
                    self.add_plot( self.axeses,
                                  title=tran2tss_study_name,
                                  label='Transient FEA w/ 2 Time Step Sections',
                                  zorder=8,
                                  time_list=time_list,
                                  sfv=sfv,
                                  torque=TorCon_list,
                                  range_ss=sfv.range_ss)
                str_results += '\n\tbasic info:' +   ''.join(  [str(el) for el in basic_info])

                if dm.jmag_loss_list is None:
                    raise Exception('Loss data is not loaded?')
                else:
                    str_results += '\n\tjmag loss info:'  + ', '.join(['%g'%(el) for el in dm.jmag_loss_list]) # dm.jmag_loss_list = [stator_copper_loss, rotor_copper_loss, stator_iron_loss, stator_eddycurrent_loss, stator_hysteresis_loss]

                if self.fea_config_dict['jmag_run_list'][0] == 0:
                    str_results += '\n\tfemm loss info:'  + ', '.join(['%g'%(el) for el in dm.femm_loss_list])

                if self.fea_config_dict['delete_results_after_calculation'] == False:
                    str_results += '\n\tPF: %g' % (dm.power_factor(self.fea_config_dict['number_of_steps_2ndTTS'], targetFreq=im_variant.DriveW_Freq))

                self.fig_main.savefig(self.dir_run + im_variant.individual_name + 'results.png', dpi=150)
                self.pyplot_clear()
            except Exception, e:
                logger.error(u'Error when loading csv results for Tran2TSS.', exc_info=True)
                raise Exception('Error: see log file.')
            # show()
            return str_results, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle, dm.jmag_loss_list, dm.femm_loss_list

        ################################################################
        # Begin from where left: Frequency Study
        ################################################################
        # Freq Study: you can choose to not use JMAG to find the breakdown slip.
        original_study_name = im_variant.individual_name + u"Freq"
        slip_freq_breakdown_torque = None
        #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        # Eddy Current Solver for Breakdown Torque and Slip
        #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        if self.fea_config_dict['jmag_run_list'][0] == 0:
            # FEMM # In this case, you have to set im_variant.slip_freq_breakdown_torque by FEMM Solver
            # check for existing results
            self.dir_femm_temp = self.dir_csv_output_folder + 'femm_temp/'
            output_file_path = self.dir_femm_temp + original_study_name + '.csv'
            if os.path.exists(output_file_path):
                with open(output_file_path, 'r') as f:
                    data = f.readlines()

                    slip_freq_breakdown_torque                  = float(data[0][:-1])
                    breakdown_torque                            = float(data[1][:-1])

                    self.femm_solver.stator_slot_area           = float(data[2][:-1])
                    self.femm_solver.rotor_slot_area            = float(data[3][:-1])

                    self.femm_solver.vals_results_rotor_current = []
                    for row in data[4:]:
                        index = row.find(',')
                        self.femm_solver.vals_results_rotor_current.append(float(row[:index]) + 1j*float(row[index+1:-1]))
                    # self.femm_solver.list_rotor_current_amp = [abs(el) for el in vals_results_rotor_current]
                    # print 'debug,'
                    # print self.femm_solver.vals_results_rotor_current
            else:

                # no direct returning of results, wait for it later when you need it.
                femm_tic = clock_time()
                self.femm_solver.__init__(im_variant, flag_read_from_jmag=False, freq=2.23)
                self.femm_solver.greedy_search_for_breakdown_slip( self.dir_femm_temp, original_study_name )

                # this is the only if path that no slip_freq_breakdown_torque is assigned!
                # this is the only if path that no slip_freq_breakdown_torque is assigned!
                # this is the only if path that no slip_freq_breakdown_torque is assigned!
        else:
            # check for existing results
            temp = self.check_csv_results(original_study_name)
            if temp is None:
                app = open_jmag()
                model = draw_jmag()
                slip_freq_breakdown_torque, breakdown_torque, breakdown_force = exe_frequency()
            else:
                slip_freq_breakdown_torque, breakdown_torque, breakdown_force = temp
                toc = clock_time()



        ################################################################
        # Begin from where left: Transient Study
        ################################################################
        tran2tss_study_name = im_variant.individual_name + 'Tran2TSS'
        bool_skip_transient = False
        if self.jmag_control_state == False: # means that no jmag project is loaded because the eddy current problem is already solved.
            # check whether or not the transient problem is also solved.
            if self.check_csv_results(tran2tss_study_name, returnBoolean=True):
                bool_skip_transient = True # because the csv files already exist.

            # debug 1
            # yes, leave this here: jmag_control_state == False
            if bool_skip_transient == False:
                app = open_jmag() # will set self.jmag_control_state to True
                model = draw_jmag()

        if bool_skip_transient == False:
            #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            # TranFEAwi2TSS for ripples and iron loss
            #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            # add or duplicate study for transient FEA 
            if self.fea_config_dict['jmag_run_list'][0] == 0:
                # wait for femm to finish, and get your slip of breakdown
                if slip_freq_breakdown_torque is None:
                    slip_freq_breakdown_torque, breakdown_torque, breakdown_force = self.femm_solver.wait_greedy_search(femm_tic)
                # debug 2
                # print slip_freq_breakdown_torque, breakdown_torque
                # quit()
                # FEMM+JMAG
                im_variant.update_mechanical_parameters(slip_freq_breakdown_torque)
                study = im_variant.add_TranFEAwi2TSS_study( slip_freq_breakdown_torque, app, model, self.dir_csv_output_folder, tran2tss_study_name, logger)
                self.mesh_study(im_variant, app, model, study)
                self.run_study(im_variant, app, study, clock_time())
            else:
                # JMAG+JMAG
                # model = app.GetCurrentModel()
                im_variant.update_mechanical_parameters(slip_freq_breakdown_torque)
                self.duplicate_TranFEAwi2TSS_from_frequency_study(im_variant, slip_freq_breakdown_torque, app, model, original_study_name, tran2tss_study_name, logger, clock_time())

            # export Voltage if field data exists.
            if self.fea_config_dict['delete_results_after_calculation'] == False:
                # Export Circuit Voltage
                ref1 = app.GetDataManager().GetDataSet(u"Circuit Voltage")
                app.GetDataManager().CreateGraphModel(ref1)
                app.GetDataManager().GetGraphModel(u"Circuit Voltage").WriteTable(self.dir_csv_output_folder + im_variant.individual_name + "_EXPORT_CIRCUIT_VOLTAGE.csv")



        ################################################################
        # Load data for cost function evaluation
        ################################################################
        str_results, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle, jmag_loss_list, femm_loss_list = load_transeint()

        # compute the fitness 
        rotor_volume = pi*(im_variant.Radius_OuterRotor*1e-3)**2 * (im_variant.stack_length*1e-3)
        rotor_weight = 9.8 * rotor_volume * 8050 # steel 8,050 kg/m3. Copper/Density 8.96 g/cm³
        shaft_power  = im_variant.Omega * torque_average
        if jmag_loss_list is None:        
            copper_loss  = 0.0
            iron_loss    = 0.0
        else:
            if False:
                # by JMAG only
                copper_loss  = jmag_loss_list[0] + jmag_loss_list[1] 
                iron_loss    = jmag_loss_list[2] 
            else:
                # by JMAG for iron loss and FEMM for copper loss
                copper_loss  = femm_loss_list[0] + femm_loss_list[1]
                iron_loss    = jmag_loss_list[2] 
            # some factor to account for rotor iron loss?
            # iron_loss *= 1

        total_loss   = copper_loss + iron_loss
        efficiency   = shaft_power / (total_loss + shaft_power)  # 效率计算：机械功率/(损耗+机械功率)
        str_results  += '\n\teta: %g' % (efficiency)

        # The weight is [TpRV=30e3, FpRW=1, Trip=50%, FEmag=50%, FEang=50deg, eta=sqrt(10)=3.16]
        # which means the FEang must be up to 50deg so so be the same level as TpRV=30e3 or FpRW=1 or eta=316%
        list_weighted_cost = [  30e3 / ( torque_average/rotor_volume ),
                                1.0 / ( ss_avg_force_magnitude/rotor_weight ),
                                normalized_torque_ripple         *   2, #       / 0.05 * 0.1
                                normalized_force_error_magnitude *   2, #       / 0.05 * 0.1
                                force_error_angle * 0.2          * 0.1, # [deg] /5 deg * 0.1 is reported to be the base line (Yegu Kang)
                                10 / efficiency**2,
                                im_variant.thermal_penalty ]            # force_error_angle is not consistent with Yegu Kang 2018-060-case of TFE
        cost_function = sum(list_weighted_cost)
            # this will lead to lower bound of air gap length
            # cost_function = 30e3 / ( breakdown_torque/rotor_volume ) \
            #                 + 1.0 / ( breakdown_force/rotor_weight )

        with open(self.dir_run + 'swarm_data.txt', 'a') as f:
            f.write('\n-------\n%s-%s\n%d,%d,%g\n%s\n%s\n' % (
                        self.project_name, im_variant.individual_name, 
                        self.number_current_generation, individual_index, cost_function, 
                        ','.join(['%g'%(el) for el in list_weighted_cost]),
                        ','.join(['%g'%(el) for el in individual]) ) + str_results)

        self.im = mylatch
        # raise Exception(u'确认能继续跑！')
        return cost_function

    def fobj_test(self, individual_index, individual):

        def fmodel(x, w):
            return w[0] + w[1]*x + w[2] * x**2 + w[3] * x**3 + w[4] * x**4 + w[5] * x**5 + w[6] * x**6
        def rmse(y, individual):
            y_pred = fmodel(x, individual)
            return np.sqrt(sum((y - y_pred)**2) / len(y))

        try:
            self.weights
        except:
            self.weights = [0.5*sum(el) for el in self.bounds]
            print self.bounds
            print self.weights
    
        x = np.linspace(0, 6.28, 50) 
        y = fmodel(x, w=self.weights)

        return rmse(y, individual)
        # plt.scatter(x, y)
        # plt.plot(x, np.cos(x), label='cos(x)')
        # plt.legend()
        # plt.show()

    def de(self):
        fobj = self.fobj
        fobj = self.fobj_test
        if self.bool_first_time_call_de == True:
            self.bool_first_time_call_de = False

        if self.fea_config_dict['flag_optimization'] == False:
            raise Exception('Please set flag_optimization before calling de.')
        # ''' 
        #     https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/
        #     我对于DE的感觉，就是如果受体本身就是原最优个体的时候，如果试验体优于受体，不应该采用greedy selection把它抛弃了。
        #     read also https://nathanrooy.github.io/posts/2017-08-27/simple-differential-evolution-with-python/
        #     Notes: In step #3, this implementation differs from most DE algorithms in that we cycle through each member of the swarm, generate a donor vector, then perform selection. In this setup, every member of the swarm becomes a target vector at some point which means that every individual has the possibility of being replaced. In most DE implementations, the target vector is randomly chosen. In my experience using DE (both in aerodynamic shape optimization, as well as in deep learning applications), I have found that the current implementation works much better. The standard DE implementation might be slightly more stochastic in nature, but whatever… If, you’re interested in the “standard” DE implementation swap out the lines in the above code with the following:
        # '''
        bounds     = self.de_config_dict['bounds']
        mut        = self.de_config_dict['mut']
        crossp     = self.de_config_dict['crossp']
        popsize    = self.de_config_dict['popsize']
        iterations = self.de_config_dict['iterations']
        iterations -= self.number_current_generation # make this iterations the total number of iterations seen from the user
        logger = logging.getLogger(__name__)
        logger.debug('DE Configuration:\n\t' + '\n\t'.join('%.4f,%.4f'%tuple(el) for el in bounds) + '\n\t%.4f, %.4f, %d, %d' % (mut,crossp,popsize,iterations))

        self.bounds = np.array(bounds) # for debug purpose in fobj_test

        # mut \in  [0.5, 2.0]
        if mut < 0.5 or mut > 2.0:
            logger = logging.getLogger(__name__)
            logger.warn('Coefficient mut is generally between 0.5 and 2.0.')

        pop = self.init_pop # modification #1
        dimensions = len(bounds)
        min_b, max_b = np.asarray(bounds).T 
        # [[-5 -5 -5 -5]
        #  [ 5  5  5  5]]
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        
        # 判断：如果是第一次，那就需要对现有pop进行生成fitness；如果续上一次的运行，则读入fitness。
        fitness_file = self.get_fit_file(self.number_current_generation)
        if not os.path.exists(fitness_file):
            # there is no fitness file yet. run evaluation for the initial pop            
            logger = logging.getLogger(__name__)
            logger.debug('Generating fitness data for the initial population: %s', fitness_file)

            self.jmag_control_state = False # demand to initialize the jamg designer
            fitness = np.asarray( [fobj(index, individual) for index, individual in enumerate(pop_denorm)] ) # modification #2

            print 'DEBUG fitness:', fitness.tolist()
            # write fitness results to file for the initial pop
            try:
                with open(fitness_file, 'w') as f:
                    f.write('\n'.join('%.16f'%(x) for x in fitness)) 
                    # TODO: also write self.fitness_in_physics_data
            except Exception as e:
                raise e # fitness

            print 'Write the 1st generation (gen#%4d) of living pop to file.' % (self.number_current_generation)
            for pop_j_denorm in pop_denorm:
                self.write_living_individual(pop_j_denorm)
        else:
            # this is a continued run. load the latest complete fitness data (the first digit of every row is fitness for an individual)
            fitness = []
            with open(fitness_file, 'r') as f: 
                read_iterator = csv_reader(f, skipinitialspace=True)
                for row in self.whole_row_reader(read_iterator):
                    if len(row)>0: # there could be empty row, since we use append mode and write down f.write('\n')
                        fitness.append(float(row[0]))

            # TODO: 文件完整性检查
            # in case the last iteration is not done or the popsize is incread by user after last run of optimization
            solved_popsize = len(fitness)
            if popsize > solved_popsize:
                self.jmag_control_state = False # demand to initialize the jamg designer
                fitness_part2 = np.asarray( [fobj(index+solved_popsize, individual) for index, individual in enumerate(pop_denorm[solved_popsize:])] ) # modification #2
                
                print 'DEBUG fitness_part2:', fitness_part2.tolist()
                try:
                    # write fitness_part2 results to file 
                    with open(fitness_file, 'a') as f:
                        f.write('\n')
                        f.write('\n'.join('%.16f'%(x) for x in fitness_part2)) 
                        # TODO: also write self.fitness_in_physics_data
                    fitness += fitness_part2.tolist()
                    # print fitness
                except Exception as e:
                    raise e

        # make sure fitness is an array
        fitness = np.asarray(fitness)
        best_idx = np.argmin(fitness)
        best = pop_denorm[best_idx]
        # return min_b + pop * diff, fitness, best_idx

        # Begin DE
        for i in range(iterations):

            self.number_current_generation += 1 # modification #3
            logger = logging.getLogger(__name__)
            logger.debug('iteration #%d for this run. total iteration %d.', i, self.number_current_generation) 
            # demand to initialize the jamg designer because number_current_generation has changed and a new jmag project is required.
            self.jmag_control_state = False

            for j in range(popsize): # j is the index of individual
                logger.debug('de individual #%d', j) 

                idxs = [idx for idx in range(popsize) if idx != j]
                # print 'idxs', idxs
                a, b, c = pop[np.random.choice(idxs, 3, replace = False)] # we select three other vectors that are not the current best one, let’s call them a, b and c
                mutant = np.clip(a + mut * (b - c), 0, 1)

                cross_points = np.random.rand(dimensions) < crossp
                if not np.any(cross_points): # 如果运气不好，全都不更新也不行，至少找一个位置赋值为True。
                    cross_points[np.random.randint(0, dimensions)] = True

                # get trial individual
                if i==0 and j < self.index_interrupt_beginning:
                    # legacy ongoing is found during an interrupted run, so the the first iteration should continue from legacy results.
                    trial = self.interrupt_pop[j]
                    trial_denorm = min_b + trial * diff
                    f = self.interrupt_fitness[j]
                else:
                    # normal run
                    trial = np.where(cross_points, mutant, pop[j])
                    trial_denorm = min_b + trial * diff

                    # quit()
                    # get fitness value for the trial individual 
                    f = fobj(j, trial_denorm)

                    # write ongoing results
                    self.write_individual_fitness(f)
                    self.write_individual_data(trial_denorm) # we write individual data after fitness is evaluated in order to make sure the two files are synchronized
                                                             # this means that the pop data file on disk does not necessarily correspondes to the current generation of pop.
                if f < fitness[j]:
                    fitness[j] = f
                    pop[j] = trial # greedy selection
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm

                pop_j_denorm = min_b + pop[j] * diff
                self.write_living_individual(pop_j_denorm)

            # one generation is finished
            self.rename_onging_files(self.number_current_generation)

            yield best, fitness[best_idx] # de verion 1
            # yield min_b + pop * diff, fitness, best_idx # de verion 2

        # TODO: 跑完一轮优化以后，必须把de_config_dict和当前的代数存在文件里，否则gen文件里面存的normalized data就没有物理意义了。

    def write_living_individual(self, pop_j_denorm):
        fname = self.dir_run + 'liv#%04d.txt'%(int(self.number_current_generation))
        with open(fname, 'a') as f:
            f.write('\n' + ','.join('%.16f'%(y) for y in pop_j_denorm)) # convert 1d array to string

    def read_living_pop(self, no_current_generation):
        fname = self.dir_run + 'liv#%04d.txt'%(no_current_generation)
        living_pop_denorm = []
        with open(fname, 'r') as f:
            for row in self.whole_row_reader(csv_reader(f, skipinitialspace=True)):
                if len(row)>0: # there could be empty row, since we use append mode and write down f.write('\n')
                    living_pop_denorm.append([float(el) for el in row])
        return living_pop_denorm

    def write_population_data(self, pop):
        with open(self.get_gen_file(self.number_current_generation), 'w') as f:
            f.write('\n'.join(','.join('%.16f'%(x) for x in y) for y in pop)) # convert 2d array to string

    def append_population_data(self, pop): # for increased popsize from last run
        with open(self.get_gen_file(self.number_current_generation), 'a') as f:
            f.write('\n')
            f.write('\n'.join(','.join('%.16f'%(x) for x in y) for y in pop)) # convert 2d array to string

    def write_individual_data(self, trial_individual):
        with open(self.get_gen_file(self.number_current_generation, ongoing=True), 'a') as f:
            f.write('\n' + ','.join('%.16f'%(y) for y in trial_individual)) # convert 1d array to string

    def write_individual_fitness(self, fitness_scalar):
        with open(self.get_fit_file(self.number_current_generation, ongoing=True), 'a') as f:
            f.write('\n%.16f'%(fitness_scalar))
            # TODO: also write self.fitness_in_physics_data



    def draw_jmag_model(self, individual_index, im_variant, model_name):

        if individual_index+1 <= self.app.NumModels():
            logger = logging.getLogger(__name__)
            logger.debug('The model already exists for individual with index=%d. Skip it.', individual_index)
            return -1 # the model is already drawn

        # open JMAG Geometry Editor
        self.app.LaunchGeometryEditor()
        geomApp = self.app.CreateGeometryEditor()
        # geomApp.Show()
        geomApp.NewDocument()
        doc = geomApp.GetDocument()
        ass = doc.GetAssembly()

        # draw parts
        if True:
            d = TrimDrawer(im_variant) # 传递的是地址哦
        else:
            d = VanGogh_JMAG(im_variant) # 传递的是地址哦
        self.d = d # for debug
        d.doc = doc
        d.ass = ass
        try:
            if True:
                d.plot_shaft(u"Shaft")

                d.plot_rotorCore(u"Rotor Core")
                d.plot_cage(u"Cage")

                d.plot_statorCore(u"Stator Core")
                d.plot_coil(u"Coil")
                # d.plot_airWithinRotorSlots(u"Air Within Rotor Slots")
            else:
                d.draw_model()
        except Exception, e:
            print 'See log file to plotting error.'
            logger = logging.getLogger(__name__)
            logger.error(u'The drawing is terminated. Please check whether the specified bounds are proper.', exc_info=True)

            # print 'Draw Failed'
            # if self.pc_name == 'Y730':
            #     # and send the email to hory chen
            #     raise e

            # or you can skip this model and continue the optimization!
            return False # indicating the model cannot be drawn with the script.

        # Import Model into Designer
        doc.SaveModel(True) # True=on : Project is also saved. 
        model = self.app.GetCurrentModel() # model = self.app.GetModel(u"IM_DEMO_1")
        model.SetName(model_name)
        model.SetDescription(im_variant.model_name_prefix + '\n' + im_variant.show(toString=True))
        im_variant.pre_process(self.app)

        model.CloseCadLink() # this is essential if you want to create a series of models
        return True

    def check_csv_results(self, study_name, returnBoolean=False):
        if not os.path.exists(self.dir_csv_output_folder + study_name + '_torque.csv'):
            if returnBoolean == False:
                return None
            else:
                return False
        else:
            if returnBoolean == True:
                return True

        try:
            # check csv results 
            l_slip_freq = []
            l_TorCon    = []
            l_ForCon_X  = []
            l_ForCon_Y  = []

            self.fitness_in_physics_data = []
            with open(self.dir_csv_output_folder + study_name + '_torque.csv', 'r') as f: 
                read_iterator = csv_reader(f, skipinitialspace=True)
                for ind, row in enumerate(self.whole_row_reader(read_iterator)):
                    if ind >= 5:
                        try:
                            float(row[0])
                        except:
                            continue
                        l_slip_freq.append(float(row[0]))
                        l_TorCon.append(float(row[1]))

            with open(self.dir_csv_output_folder + study_name + '_force.csv', 'r') as f: 
                read_iterator = csv_reader(f, skipinitialspace=True)
                for ind, row in enumerate(self.whole_row_reader(read_iterator)):
                    if ind >= 5:
                        try:
                            float(row[0])
                        except:
                            continue
                        # l_slip_freq.append(float(row[0]))
                        l_ForCon_X.append(float(row[1]))
                        l_ForCon_Y.append(float(row[2]))

            # self.fitness_in_physics_data.append(l_slip_freq)
            # self.fitness_in_physics_data.append(l_TorCon)

            breakdown_force = max(np.sqrt(np.array(l_ForCon_X)**2 + np.array(l_ForCon_Y)**2))

            index, breakdown_torque = utility.get_max_and_index(l_TorCon)
            slip_freq_breakdown_torque = l_slip_freq[index]
            return slip_freq_breakdown_torque, breakdown_torque, breakdown_force

        except NameError, e:
            logger = logging.getLogger(__name__)
            logger.error(u'No CSV File Found.', exc_info=True)
            raise e

    def plot_csv_results_for_all(self):
        # 想要观察所有解的滑差分布情况，合理规划滑差步长的选择以及寻找范围！
        from pylab import plot, show, legend, grid, figure
        max_breakdown_torque = 0.0
        max_breakdown_torque_file = None
        max_breakdown_force_amp = 0.0
        max_breakdown_force_amp_file = None
        # self.fitness_in_physics_data = []
        for file in os.listdir(self.dir_csv_output_folder):
            if 'lock' not in file and self.model_name_prefix in file: # check model_name_prefix in file because there is D:/Users/horyc/OneDrive - UW-Madison/csv_opti/run#1/Freq_#32-0-0-FFVRC-RCR_force.csv
                if '_torque' in file:
                    l_slip_freq, l_TorCon = self.read_csv_results(self.dir_csv_output_folder + file)
                    figure(1)
                    plot(l_slip_freq, l_TorCon, label=file)
                    temp = max(l_TorCon)
                    if temp > max_breakdown_torque:
                        max_breakdown_torque = temp
                        max_breakdown_torque_file = file
                elif '_force' in file:
                    l_slip_freq_2, l_ForCon_XY = self.read_csv_results(self.dir_csv_output_folder + file)
                    l_ForCon_X = [el[0] for el in l_ForCon_XY]
                    l_ForCon_Y = [el[1] for el in l_ForCon_XY]
                    figure(2)
                    plot(l_slip_freq_2, l_ForCon_X, label=file)
                    figure(3)
                    plot(l_slip_freq_2, l_ForCon_Y, label=file)
                    temp = max(np.sqrt(np.array(l_ForCon_X)**2 + np.array(l_ForCon_Y)**2))
                    if temp > max_breakdown_force_amp:
                        max_breakdown_force_amp = temp
                        max_breakdown_force_amp_file = file
        print max_breakdown_torque, max_breakdown_torque_file, type(max_breakdown_torque)
        print max_breakdown_force_amp, max_breakdown_force_amp_file, type(max_breakdown_force_amp)
        for i in range(1, 3+1):
            figure(i)
            legend()
            grid()
        show()

        ''' the slip? the individual? I guess max torque and force are produced with the smallest air gap!
        '''

    def read_csv_results(self, file_location):
        l_slip_freq = [] # or other x-axis variable such as time and angle.
        l_data    = []
        with open(file_location, 'r') as f: 
            read_iterator = csv_reader(f, skipinitialspace=True)
            for ind, row in enumerate(self.whole_row_reader(read_iterator)):
                if ind >= 5:
                    # print file_location
                    # print row
                    l_slip_freq.append(float(row[0]))
                    l_data.append([float(el) for el in row[1:]])
        return l_slip_freq, l_data



    def logging_1d_array_for_debug(self, a, a_name):
        # self.logging_1d_array_for_debug(trial)
        logger = logging.getLogger(__name__)
        logger.debug(a_name + ','.join('%.16f'%(y) for y in a)) # convert 1d array to string

    ''' API for FEMM Solver
    '''
    def get_breakdown_results(self, im, study_type=1):
        if study_type == 1:
            study_name = u"Freq"
        else:
            raise Exception('not supported study_type.')

        return self.check_csv_results(study_name)

    def has_results(self, im, study_type='Freq'):
        # short study_name because of one model one project convension
        if   study_type == 'Freq': 
            study_name =  u"Freq"
        elif study_type == 'Tran2TSS':
            study_name =  u"Tran2TSS"
        elif study_type == 'Freq-FFVRC':
            study_name =  u"Freq-FFVRC"
        else:
            raise Exception('not supported study_type.')

        im.csv_previous_solve = self.dir_csv_output_folder + study_name + '_circuit_current.csv'
        bool_temp = os.path.exists(im.csv_previous_solve)

        if study_type == 'Freq':
            if bool_temp == True:
                # this is no longer needed, because the TranFEAwi2TSS will do this for ya.
                slip_freq_breakdown_torque, _, _ = self.get_breakdown_results(im)
                im.update_mechanical_parameters(slip_freq_breakdown_torque)
                # print 'slip_freq_breakdown_torque:', slip_freq_breakdown_torque, 'Hz'

        return bool_temp

    def run(self, im, run_list=[1,1,0,0,0]): 
        ''' run_list: toggle solver for Freq, Tran2TSS, Freq-FFVRC, TranRef, Static'''

        # Settings 
        self.jmag_control_state = False # new one project one model convension

        # initialize JMAG Designer
        self.designer_init()
        app = self.app
        self.project_name = self.fea_config_dict['model_name_prefix']
        self.im.model_name = self.im.get_individual_name() 
        if self.jmag_control_state == False: # initilize JMAG Designer
            expected_project_file = self.dir_project_files + "%s.jproj"%(self.project_name)
            if not os.path.exists(expected_project_file):
                app.NewProject(u"Untitled")
                app.SaveAs(expected_project_file)
                logger = logging.getLogger(__name__)
                logger.debug('Create JMAG project file: %s'%(expected_project_file))
            else:
                app.Load(expected_project_file)
                logger = logging.getLogger(__name__)
                logger.debug('Load JMAG project file: %s'%(expected_project_file))
                logger.debug('Existing models of %d are found in %s', app.NumModels(), app.GetDefaultModelFolderPath())

                # if app.NumModels() <= individual_index:
                #     logger.warn('Some models are not plotted because of bad bounds (some lower bound is too small)! individual_index=%d, NumModels()=%d. See also the fit#%04d.txt file for 99999. There will be no .png file for these individuals either.', individual_index, app.NumModels(), self.number_current_generation)

                # print app.NumStudies()
                # print app.NumAnalysisGroups()
                # app.SubmitAllModelsLocal() # we'd better do it one by one for easing the programing?

        # draw the model in JMAG Designer
        DRAW_SUCCESS = self.draw_jmag_model( 0, 
                                        im,
                                        self.im.model_name)
        print 'TEST VanGogh for JMAG.'
        return
        self.jmag_control_state = True # indicating that the jmag project is already created
        if DRAW_SUCCESS == 0:
            # TODO: skip this model and its evaluation
            cost_function = 99999
            logging.getLogger(__name__).warn('Draw Failed for'+'%s-%s: %g', self.project_name, self.im.model_name, cost_function)
            return cost_function
        elif DRAW_SUCCESS == -1:
            # The model already exists
            print 'Model Already Exists'
            logging.getLogger(__name__).debug('Model Already Exists')
        # Tip: 在JMAG Designer中DEBUG的时候，删掉模型，必须要手动save一下，否则再运行脚本重新load project的话，是没有删除成功的，只是删掉了model的name，新导入进来的model name与project name一致。
        if app.NumModels()>=1:
            model = app.GetModel(self.im.model_name)
        else:
            logging.getLogger(__name__).error('there is no model yet!')
            print 'why is there no model yet?'

        # Freq Sweeping for break-down Torque Slip
        # remember to export the B data using subroutine 
        # and check export table results only
        if model.NumStudies() == 0:
            study = im.add_study(app, model, self.dir_csv_output_folder, choose_study_type='frequency')
        else:
            # there is already a study. then get the first study.
            study = model.GetStudy(0)

        # Freq Study: you can choose to not use JMAG to find the breakdown slip.
        # In this case, you have to set im.slip_freq_breakdown_torque by FEMM Solver
        print 'debug: run_list[0]', run_list[0]
        if run_list[0] == 0:
            # Does femm has already done the frequency sweeping for breakdown torque?
            if im.slip_freq_breakdown_torque is None:
                raise Exception('run_list[0] is 0, so you have to run FEMM solver first to get slip_freq_breakdown_torque')
            print "FEMM's slip_freq_breakdown_torque is used for Tran2TSS."
            slip_freq_breakdown_torque = im.slip_freq_breakdown_torque
        else:
            # Use JMAG to sweeping the frequency
            # Does study has results?
            if study.AnyCaseHasResult():
                slip_freq_breakdown_torque, breakdown_torque, breakdown_force = self.check_csv_results(study.GetName())
            else:
                # mesh
                im.add_mesh(study, model)

                # Export Image
                    # for i in range(app.NumModels()):
                    #     app.SetCurrentModel(i)
                    #     model = app.GetCurrentModel()
                    #     app.ExportImage(r'D:\Users\horyc\OneDrive - UW-Madison\pop\run#10/' + model.GetName() + '.png')
                app.View().ShowAllAirRegions()
                # app.View().ShowMeshGeometry() # 2nd btn
                app.View().ShowMesh() # 3rn btn
                app.View().Zoom(3)
                app.View().Pan(-im_variant.Radius_OuterRotor, 0)
                app.ExportImageWithSize(self.dir_run + model.GetName() + '.png', 2000, 2000)
                app.View().ShowModel() # 1st btn. close mesh view, and note that mesh data will be deleted because only ouput table results are selected.

                # run
                study.RunAllCases()
                app.Save()

                # evaluation based on the csv results
                slip_freq_breakdown_torque, breakdown_torque, breakdown_force = self.check_csv_results(study.GetName())
                # self.fitness_in_physics_data # torque density, torque ripple, force density, force magnitude error, force angle error, efficiency, material cost 

        # this will be used for other duplicated studies
        original_study_name = study.GetName()
        self.im.update_mechanical_parameters(slip_freq_breakdown_torque, syn_freq=im.DriveW_Freq)



        # Transient FEA wi 2 Time Step Section
        tran2tss_study_name = u"Tran2TSS"
        if model.NumStudies()<2:
            model.DuplicateStudyWithType(original_study_name, u"Transient2D", tran2tss_study_name)
            app.SetCurrentStudy(tran2tss_study_name)
            study = app.GetCurrentStudy()

            # 上一步的铁磁材料的状态作为下一步的初值，挺好，但是如果每一个转子的位置转过很大的话，反而会减慢非线性迭代。
            # 我们的情况是：0.33 sec 分成了32步，每步的时间大概在0.01秒，0.01秒乘以0.5*497 Hz = 2.485 revolution...
            study.GetStudyProperties().SetValue(u"NonlinearSpeedup", 0) 

            # 2 sections of different time step
            number_cycles_prolonged = 1 # 50
            DM = app.GetDataManager()
            DM.CreatePointArray(u"point_array/timevsdivision", u"SectionStepTable")
            refarray = [[0 for i in range(3)] for j in range(4)]
            refarray[0][0] = 0
            refarray[0][1] =    1
            refarray[0][2] =        50
            refarray[1][0] = 1.0/slip_freq_breakdown_torque
            refarray[1][1] =    32 
            refarray[1][2] =        50
            refarray[2][0] = refarray[1][0] + 1.0/self.im.DriveW_Freq
            refarray[2][1] =    48 # don't forget to modify below!
            refarray[2][2] =        50
            refarray[3][0] = refarray[2][0] + number_cycles_prolonged/self.im.DriveW_Freq # =50*0.002 sec = 0.1 sec is needed to converge to TranRef
            refarray[3][1] =    number_cycles_prolonged*self.fea_config_dict['TranRef-StepPerCycle'] # =50*40, every 0.002 sec takes 40 steps 
            refarray[3][2] =        50
            DM.GetDataSet(u"SectionStepTable").SetTable(refarray)
            study.GetStep().SetValue(u"Step", 1 + 32 + 48 + number_cycles_prolonged*self.fea_config_dict['TranRef-StepPerCycle']) # [Double Check] don't forget to modify here!
            study.GetStep().SetValue(u"StepType", 3)
            study.GetStep().SetTableProperty(u"Division", DM.GetDataSet(u"SectionStepTable"))

            # add equations
            study.GetDesignTable().AddEquation(u"freq")
            study.GetDesignTable().AddEquation(u"slip")
            study.GetDesignTable().AddEquation(u"speed")
            study.GetDesignTable().GetEquation(u"freq").SetType(0)
            study.GetDesignTable().GetEquation(u"freq").SetExpression("%g"%((self.im.DriveW_Freq)))
            study.GetDesignTable().GetEquation(u"freq").SetDescription(u"Excitation Frequency")
            study.GetDesignTable().GetEquation(u"slip").SetType(0)
            study.GetDesignTable().GetEquation(u"slip").SetExpression("%g"%(self.im.the_slip))
            study.GetDesignTable().GetEquation(u"slip").SetDescription(u"Slip [1]")
            study.GetDesignTable().GetEquation(u"speed").SetType(1)
            study.GetDesignTable().GetEquation(u"speed").SetExpression(u"freq * (1 - slip) * 30")
            study.GetDesignTable().GetEquation(u"speed").SetDescription(u"mechanical speed of four pole")

            # speed, freq, slip
            study.GetCondition(u"RotCon").SetValue(u"AngularVelocity", u'speed')
            app.ShowCircuitGrid(True)
            study.GetCircuit().GetComponent(u"CS4").SetValue(u"Frequency", u"freq")
            study.GetCircuit().GetComponent(u"CS2").SetValue(u"Frequency", u"freq")

            # max_nonlinear_iteration = 50
            # study.GetStudyProperties().SetValue(u"NonlinearMaxIteration", max_nonlinear_iteration)
            study.GetStudyProperties().SetValue(u"ApproximateTransientAnalysis", 1) # psuedo steady state freq is for PWM drive to use
            study.GetStudyProperties().SetValue(u"SpecifySlip", 1)
            study.GetStudyProperties().SetValue(u"OutputSteadyResultAs1stStep", 0)
            study.GetStudyProperties().SetValue(u"Slip", u"slip")

            # # add other excitation frequencies other than 500 Hz as cases
            # for case_no, DriveW_Freq in enumerate([50.0, slip_freq_breakdown_torque]):
            #     slip = slip_freq_breakdown_torque / DriveW_Freq
            #     study.GetDesignTable().AddCase()
            #     study.GetDesignTable().SetValue(case_no+1, 0, DriveW_Freq)
            #     study.GetDesignTable().SetValue(case_no+1, 1, slip)

            # Iron Loss Calculation Condition
            # Stator 
            study.CreateCondition(u"Ironloss", u"IronLossConStator")
            cond.SetValue(u"RevolutionSpeed", u"freq*60/%d"%(0.5*(self.im.DriveW_poles)))
            cond.ClearParts()
            sel = cond.GetSelection()
            sel.SelectPartByPosition(-im.Radius_OuterStatorYoke+1e-2, 0 ,0)
            cond.AddSelected(sel)
            # Rotor
            study.CreateCondition(u"Ironloss", u"IronLossConRotor")
            study.GetCondition(u"IronLossConRotor").SetValue(u"BasicFrequencyType", 2)
            study.GetCondition(u"IronLossConRotor").SetValue(u"BasicFrequency", u"slip*freq")
            study.GetCondition(u"IronLossConRotor").ClearParts()
            sel = study.GetCondition(u"IronLossConRotor").GetSelection()
            sel.SelectPartByPosition(-im.Radius_Shaft-1e-2, 0 ,0)
            study.GetCondition(u"IronLossConRotor").AddSelected(sel)
            # Use FFT for hysteresis to be consistent with FEMM's results
            cond.SetValue(u"HysteresisLossCalcType", 1)
            cond.SetValue(u"PresetType", 3)
            study.GetCondition(u"IronLossConRotor").SetValue(u"HysteresisLossCalcType", 1)
            study.GetCondition(u"IronLossConRotor").SetValue(u"PresetType", 3)


            # https://www2.jmag-international.com/support/en/pdf/JMAG-Designer_Ver.17.1_ENv3.pdf
            study.GetStudyProperties().SetValue(u"DirectSolverType", 1)

            if run_list[1] == True:
                study.RunAllCases()
                app.Save()
            else:
                pass # if the jcf file already exists, it pops a msg window
                # study.WriteAllSolidJcf(self.dir_jcf, self.im.model_name+study.GetName()+'Solid', True) # True : Outputs cases that do not have results 
                # study.WriteAllMeshJcf(self.dir_jcf, self.im.model_name+study.GetName()+'Mesh', True)


        # These two studies are not needed in optimization
        if self.fea_config_dict['flag_optimization'] == False:
            # These two studies are no longer needed after iemdc digest 
            # EC Rotate
            ecrot_study_name = original_study_name + u"-FFVRC"
            if model.NumStudies()<3:
                # EC Rotate: Rotate the rotor to find the ripples in force and torque # 不关掉这些云图，跑第二个study的时候，JMAG就挂了：app.View().SetVectorView(False); app.View().SetFluxLineView(False); app.View().SetContourView(False)
                casearray = [0 for i in range(1)]
                casearray[0] = 1
                model.DuplicateStudyWithCases(original_study_name, ecrot_study_name, casearray)

                app.SetCurrentStudy(ecrot_study_name)
                study = app.GetCurrentStudy()
                divisions_per_slot_pitch = 24
                study.GetStep().SetValue(u"Step", divisions_per_slot_pitch) 
                study.GetStep().SetValue(u"StepType", 0)
                study.GetStep().SetValue(u"FrequencyStep", 0)
                study.GetStep().SetValue(u"Initialfrequency", slip_freq_breakdown_torque)

                    # study.GetCondition(u"RotCon").SetValue(u"MotionGroupType", 1)
                study.GetCondition(u"RotCon").SetValue(u"Displacement", + 360.0/im.Qr/divisions_per_slot_pitch)

                # https://www2.jmag-international.com/support/en/pdf/JMAG-Designer_Ver.17.1_ENv3.pdf
                study.GetStudyProperties().SetValue(u"DirectSolverType", 1)

                if run_list[2] == True:
                    # model.RestoreCadLink()
                    study.Run()
                    app.Save()
                    # model.CloseCadLink()
                else:
                    pass # if the jcf file already exists, it pops a msg window
                    # study.WriteAllSolidJcf(self.dir_jcf, self.im.model_name+study.GetName()+'Solid', True) # True : Outputs cases that do not have results 
                    # study.WriteAllMeshJcf(self.dir_jcf, self.im.model_name+study.GetName()+'Mesh', True)

            # Transient Reference
            tranRef_study_name = u"TranRef"
            if model.NumStudies()<4:
                model.DuplicateStudyWithType(tran2tss_study_name, u"Transient2D", tranRef_study_name)
                app.SetCurrentStudy(tranRef_study_name)
                study = app.GetCurrentStudy()

                # 将一个滑差周期和十个同步周期，分成 400 * end_point / (1.0/self.im.DriveW_Freq) 份。
                end_point = 1.0/slip_freq_breakdown_torque + 10.0/self.im.DriveW_Freq
                # Pavel Ponomarev 推荐每个电周期400~600个点来捕捉槽效应。
                division = self.fea_config_dict['TranRef-StepPerCycle'] * end_point / (1.0/self.im.DriveW_Freq)  # int(end_point * 1e4)
                                                                        # end_point = division * 1e-4
                study.GetStep().SetValue(u"Step", division + 1) 
                study.GetStep().SetValue(u"StepType", 1) # regular inverval
                study.GetStep().SetValue(u"StepDivision", division)
                study.GetStep().SetValue(u"EndPoint", end_point)

                # https://www2.jmag-international.com/support/en/pdf/JMAG-Designer_Ver.17.1_ENv3.pdf
                study.GetStudyProperties().SetValue(u"DirectSolverType", 1)

                if run_list[3] == True:
                    study.RunAllCases()
                    app.Save()
                else:
                    pass # if the jcf file already exists, it pops a msg window
                    # study.WriteAllSolidJcf(self.dir_jcf, self.im.model_name+study.GetName()+'Solid', True) # True : Outputs cases that do not have results 
                    # study.WriteAllMeshJcf(self.dir_jcf, self.im.model_name+study.GetName()+'Mesh', True)

            # Rotating Static FEA (This can be done in FEMM)
            if run_list[4] == True:

                im.MODEL_ROTATE = True # this is used in Rotating Static FEA
                im.total_number_of_cases = 1 # 这个值取24的话，就可以得到24个不同位置下，电机的转矩-滑差曲线了，这里取1，真正的cases在StaticFEA中添加。

                # draw another model with MODEL_ROTATE as True
                if True:
                    DRAW_SUCCESS = self.draw_jmag_model( 1, # +1
                                                    im,
                                                    self.im.model_name + 'MODEL_ROTATE')
                    self.jmag_control_state = True # indicating that the jmag project is already created
                    if DRAW_SUCCESS == 0:
                        # TODO: skip this model and its evaluation
                        cost_function = 99999
                        return cost_function
                    elif DRAW_SUCCESS == -1:
                        # The model already exists
                        print 'Model Already Exists'
                        logging.getLogger(__name__).debug('Model Already Exists')
                    # Tip: 在JMAG Designer中DEBUG的时候，删掉模型，必须要手动save一下，否则再运行脚本重新load project的话，是没有删除成功的，只是删掉了model的name，新导入进来的model name与project name一致。
                    if app.NumModels()>=2: # +1
                        model = app.GetModel(self.im.model_name + 'MODEL_ROTATE')
                    else:
                        logging.getLogger(__name__).error('there is no model yet!')
                        print 'why is there no model yet?'
                        raise

                    if model.NumStudies() == 0:
                        study = im.add_study(app, model, self.dir_csv_output_folder, choose_study_type='static')
                    else:
                        # there is already a study. then get the first study.
                        study = model.GetStudy(0)

                im.theta = 6./180.0*pi # 5 deg
                total_number_of_cases = 2 #12 #36

                # add equations
                    # DriveW_Freq = self.im.DriveW_Freq
                    # slip = slip_freq_breakdown_torque / DriveW_Freq
                    # im.DriveW_Freq = DriveW_Freq
                    # im.the_speed = DriveW_Freq * (1 - slip) * 30
                    # im.the_slip = slip
                study.GetDesignTable().AddEquation(u"freq")
                study.GetDesignTable().AddEquation(u"slip")
                study.GetDesignTable().AddEquation(u"speed")
                study.GetDesignTable().GetEquation(u"freq").SetType(0)
                study.GetDesignTable().GetEquation(u"freq").SetExpression("%g"%((im.DriveW_Freq)))
                study.GetDesignTable().GetEquation(u"freq").SetDescription(u"Excitation Frequency")
                study.GetDesignTable().GetEquation(u"slip").SetType(0)
                study.GetDesignTable().GetEquation(u"slip").SetExpression("%g"%(im.the_slip))
                study.GetDesignTable().GetEquation(u"slip").SetDescription(u"Slip [1]")
                study.GetDesignTable().GetEquation(u"speed").SetType(1)
                study.GetDesignTable().GetEquation(u"speed").SetExpression(u"freq * (1 - slip) * 30")
                study.GetDesignTable().GetEquation(u"speed").SetDescription(u"mechanical speed of four pole")

                # rotate the rotor by cad parameters via Park Transformation
                # cad parameters cannot be duplicated! even you have a list of cad paramters after duplicating, but they cannot be used to create cases! So you must set total_number_of_cases to 1 in the first place if you want to do Rotating Static FEA in JMAG
                im.add_cad_parameters(study)
                im.add_cases_rotate_rotor(study, total_number_of_cases) 
                    # print study.GetDesignTable().NumParameters()

                # set rotor current conditions
                im.slip_freq_breakdown_torque = slip_freq_breakdown_torque
                im.add_rotor_current_condition(app, model, study, total_number_of_cases, 
                                              self.dir_csv_output_folder + original_study_name + '_circuit_current.csv')
                print self.dir_csv_output_folder + original_study_name + '_circuit_current.csv'
                print self.dir_csv_output_folder + original_study_name + '_circuit_current.csv'
                print self.dir_csv_output_folder + original_study_name + '_circuit_current.csv'
                    # print self.dir_csv_output_folder + im.get_individual_name() + original_study_name + '_circuit_current.csv'
                    # print self.dir_csv_output_folder + im.get_individual_name() + original_study_name + '_circuit_current.csv'
                    # print self.dir_csv_output_folder + im.get_individual_name() + original_study_name + '_circuit_current.csv'
                # set stator current conditions
                im.add_stator_current_condition(app, model, study, total_number_of_cases, 
                                              self.dir_csv_output_folder + original_study_name + '_circuit_current.csv')

                # https://www2.jmag-international.com/support/en/pdf/JMAG-Designer_Ver.17.1_ENv3.pdf
                study.GetStudyProperties().SetValue(u"DirectSolverType", 1)

                model.RestoreCadLink()

                study.RunAllCases()
                app.Save()
                model.CloseCadLink()

        # Loss Study
        pass



        # compute the fitness 
        rotor_volume = pi*(im.Radius_OuterRotor*1e-3)**2 * (im.stack_length*1e-3)
        rotor_weight = rotor_volume * 8050 # steel 8,050 kg/m3. Copper/Density 8.96 g/cm³
        cost_function = 30e3 / ( breakdown_torque/rotor_volume ) \
                        + 1.0 / ( breakdown_force/rotor_weight )
        logger = logging.getLogger(__name__)
        logger.debug('%s-%s: %g', self.project_name, self.im.model_name, cost_function)

        return cost_function

    def get_csv(self, data_name):
        r'D:\Users\horyc\OneDrive - UW-Madison\csv_opti\run#36\BLIM_PS_ID36-0-0Freq_#36-0-0_circuit_current.csv'
        pass
        pass

    def read_current_from_EC_FEA(self):

        # read from eddy current results
        dict_circuit_current_complex = {}
        
        with open(self.im.get_csv('circuit_current'), 'r') as f:
            read_iterator = csv_reader(f, skipinitialspace=True)
            for row in self.whole_row_reader(read_iterator):
                try: 
                    float(row[0])
                except:
                    continue
                else:
                    if '%g'%(self.slip_freq_breakdown_torque) in row[0]:
                        beginning_column = 1 + 2*3*2 # title + drive/bearing * 3 phase * real/imag
                        for i in range(0, int(self.no_slot_per_pole)):
                            natural_i = i+1
                            current_phase_column = beginning_column + i * int(self.im.DriveW_poles) * 2
                            for j in range(int(self.im.DriveW_poles)):
                                natural_j = j+1
                                re = float(row[current_phase_column+2*j])
                                im = float(row[current_phase_column+2*j+1])
                                dict_circuit_current_complex["%s%d"%(rotor_phase_name_list[i], natural_j)] = (re, im)
        dict_circuit_current_amp_and_phase = {}
        for key, item in dict_circuit_current_complex.iteritems():
            amp = np.sqrt(item[1]**2 + item[0]**2)
            phase = np.arctan2(item[0], -item[1]) # atan2(y, x), y=a, x=-b
            dict_circuit_current_amp_and_phase[key] = (amp, phase)

    def show_results(self, femm_solver_data=None):
        print 'show results!'
        # from pylab import *
        # plot style
        # plt.style.use('ggplot') 
        # plt.style.use('grayscale') # print plt.style.available # get [u'dark_background', u'bmh', u'grayscale', u'ggplot', u'fivethirtyeight']
        mpl.rcParams['legend.fontsize'] = 15
        font = {'family' : 'Times New Roman', #'serif',
                'weight' : 'normal',
                'size'   : 15}
        mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
        mpl.rcParams['font.serif'] = ['Times New Roman']

        # color and alpha
        # Freq-FFVRC
        # rotor current

        fig, axes = subplots(2, 1, sharex=True)

        ''' TranRef '''
        study_name = 'TranRef'
        dm = self.read_csv_results_4_comparison__transient(study_name)
        basic_info, time_list, TorCon_list, ForConX_list, ForConY_list, ForConAbs_list = dm.unpack()
        end_time = time_list[-1]

        ax = axes[0]; ax.plot(time_list, TorCon_list, alpha=0.7, label=study_name); ax.set_xlabel('Time [s]'); ax.set_ylabel('Torque [Nm]')
        ax = axes[1]; ax.plot(time_list, ForConX_list, alpha=0.7, label=study_name+'-X'); ax.plot(time_list, ForConY_list, alpha=0.7, label=study_name+'Y'); ax.set_xlabel('Time [s]'); ax.set_ylabel('Force [N]')
        ax.plot(time_list, ForConAbs_list, alpha=0.7, label=study_name+'-Abs')


        Avg_ForCon_Vector, _, Max_ForCon_Err_Angle = self.get_force_error_angle(ForConX_list[-48:], ForConY_list[-48:])
        print '---------------\nTranRef-FEA \nForce Average Vecotr:', Avg_ForCon_Vector, '[N]'
        # print ForCon_Angle_List, 'deg'
        print 'Maximum Force Angle Error', Max_ForCon_Err_Angle, '[deg]'
        print '\tbasic info:', basic_info




        ''' Tran2TSS '''
        study_name = 'Tran2TSS'
        dm = self.read_csv_results_4_comparison__transient(study_name)
        basic_info, time_list, TorCon_list, ForConX_list, ForConY_list, ForConAbs_list = dm.unpack()

        ax = axes[0]; ax.plot(time_list, TorCon_list, alpha=0.7, label=study_name); ax.set_xlabel('Time [s]'); ax.set_ylabel('Torque [Nm]')
        ax = axes[1]; ax.plot(time_list, ForConX_list, alpha=0.7, label=study_name+'-X'); ax.plot(time_list, ForConY_list, alpha=0.7, label=study_name+'Y'); ax.set_xlabel('Time [s]'); ax.set_ylabel('Force [N]')
        ax.plot(time_list, ForConAbs_list, alpha=0.7, label=study_name+'-Abs')


        Avg_ForCon_Vector, _, Max_ForCon_Err_Angle = self.get_force_error_angle(ForConX_list[-48:], ForConY_list[-48:])
        print '---------------\nTran2TSS-FEA \nForce Average Vecotr:', Avg_ForCon_Vector, '[N]'
        # print ForCon_Angle_List, 'deg'
        print 'Maximum Force Angle Error', Max_ForCon_Err_Angle, '[deg]'
        print '\tbasic info:', basic_info




        ''' Static FEA with FEMM '''
        rotor_position_in_deg = femm_solver_data[0]*0.1 
        time_list = rotor_position_in_deg/180.*pi / self.im.Omega
        number_of_repeat = int(end_time / time_list[-1])
        # print number_of_repeat, end_time, time_list[-1]
        femm_force_x = femm_solver_data[2].tolist()
        femm_force_y = femm_solver_data[3].tolist()        
        femm_force_abs = np.sqrt(np.array(femm_force_x)**2 + np.array(femm_force_y)**2 )

        # Vector plot
        ax = figure().gca()
        ax.text(0,0,'FEMM')
        for x, y in zip(femm_force_x, femm_force_y):
            ax.arrow(0,0, x,y)
        xlim([0,220])
        ylim([0,220])

        # Force Error Angle
        Avg_ForCon_Vector, _, Max_ForCon_Err_Angle = self.get_force_error_angle(femm_force_x, femm_force_y)
        print '---------------\nFEMM-Static-FEA \nForce Average Vecotr:', Avg_ForCon_Vector, '[N]'
        # print ForCon_Angle_List, 'deg'
        print 'Maximum Force Angle Error', Max_ForCon_Err_Angle, '[deg]'

        femm_torque  = number_of_repeat * femm_solver_data[1].tolist()
        time_one_step = time_list[1]
        time_list    = [i*time_one_step for i in range(len(femm_torque))]
        femm_force_x = number_of_repeat * femm_solver_data[2].tolist()
        femm_force_y = number_of_repeat * femm_solver_data[3].tolist()
        femm_force_abs = number_of_repeat * femm_force_abs.tolist()
        # print len(time_list), len(femm_force_abs)
        if not femm_solver_data is None:
            ax = axes[0]; ax.plot(time_list, femm_torque,label='FEMM', zorder=1)
            ax = axes[1]; 
            ax.plot(time_list, femm_force_x,label='FEMM-X', zorder=1)
            ax.plot(time_list, femm_force_y,label='FEMM-Y', zorder=1)
            ax.plot(time_list, femm_force_abs,label='FEMM-Abs', zorder=1)





        axes[0].grid()
        axes[0].legend()
        axes[1].grid()
        axes[1].legend()
        # return basic_info

    def show_results_iemdc19(self, femm_solver_data=None, femm_rotor_current_function=None):
        print 'show results!'
        mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
        mpl.rcParams['font.serif'] = ['Times New Roman']
        
        Fs = 500.*400.
        def basefreqFFT(x, Fs, base_freq=500, ax=None, ax_time_domain=None): #频域横坐标除以基频，即以基频为单位
            def nextpow2(L):
                n = 0
                while 2**n < L:
                    n += 1
                return n
            L = len(x)
            Ts = 1.0/Fs
            t = [el*Ts for el in range(0,L)]
            if ax_time_domain != None:
                ax_time_domain.plot(t, x)

            # NFFT = 2**nextpow2(L) # this causes incorrect dc bin (too large)
            NFFT = L
            y = np.fft.fft(x,NFFT) # y is a COMPLEX defined in numpy
            Y = [2 * el.__abs__() / L for el in y] # /L for spectrum aplitude consistent with actual signal. 2* for single-sided. abs for amplitude of complem number.
            Y[0] *= 0.5 # DC does not need to be times 2
            if base_freq==None:
                # f = np.fft.fftfreq(NFFT, t[1]-t[0]) # for double-sided
                f = Fs/2.0*np.linspace(0,1,NFFT/2+1) # unit is Hz
            else:
                f = Fs/2.0/base_freq*np.linspace(0,1,NFFT/2+1) # unit is base_freq Hz

            if ax == None:
                fig, ax = subplots()
            # ax.bar(f,Y[0:int(NFFT/2)+1], width=1.5)
            ax.plot(f,Y[0:int(NFFT/2)+1])

            # fig.title('Single-Sided Amplitude Spectrum of x(t)')
            # ax.xlabel('Frequency divided by base_freq / base freq * Hz')
            #ylabel('|Y(f)|')
            # ax.ylabel('Amplitude / 1')

            # # 计算频谱
            # fft_parameters = np.fft.fft(y_data) / len(y_data)
            # # 计算各个频率的振幅
            # fft_data = np.clip(20*np.log10(np.abs(fft_parameters))[:self.fftsize/2+1], -120, 120)
        def FFT_another_implementation(TorCon_list, Fs):
            # 这个FFT的结果和上面那个差不多，但是会偏小一点！不知道为什么！

            # Number of samplepoints
            N = len(TorCon_list)
            # Sample spacing
            T = 1.0 / Fs
            yf = np.fft.fft(TorCon_list)
            yf[0] *= 0.5
            xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
            fig, ax = subplots()
            ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
        global count_plot
        count_plot = 0
        def add_plot(axeses, title=None, label=None, zorder=None, time_list=None, sfv=None, torque=None, range_ss=None, alpha=0.7):

            # # Avg_ForCon_Vector, Avg_ForCon_Magnitude, Avg_ForCon_Angle, ForCon_Angle_List, Max_ForCon_Err_Angle = self.get_force_error_angle(force_x[-range_ss:], force_y[-range_ss:])
            # print '\n\n---------------%s' % (title)
            # print 'Average Force Mag:', sfv.ss_avg_force_magnitude, '[N]'
            # print 'Average Torque:', sum(torque[-range_ss:])/len(torque[-range_ss:]), '[Nm]'
            # print 'Normalized Force Error Mag: %g%%, (+)%g%% (-)%g%%' % (0.5*(sfv.ss_max_force_err_abs[0]-sfv.ss_max_force_err_abs[1])/sfv.ss_avg_force_magnitude*100,
            #                                                               sfv.ss_max_force_err_abs[0]/sfv.ss_avg_force_magnitude*100,
            #                                                               sfv.ss_max_force_err_abs[1]/sfv.ss_avg_force_magnitude*100)
            # print 'Maximum Force Error Angle: %g [deg], (+)%g deg (-)%g deg' % (0.5*(sfv.ss_max_force_err_ang[0]-sfv.ss_max_force_err_ang[1]),
            #                                                              sfv.ss_max_force_err_ang[0],
            #                                                              sfv.ss_max_force_err_ang[1])
            # print 'Extra Information:'
            # print '\tAverage Force Vecotr:', sfv.ss_avg_force_vector, '[N]'
            # print '\tTorque Ripple (Peak-to-Peak)', max(torque[-range_ss:]) - min(torque[-range_ss:]), 'Nm'
            # print '\tForce Mag Ripple (Peak-to-Peak)', sfv.ss_max_force_err_abs[0] - sfv.ss_max_force_err_abs[1], 'N'

            ax = axeses[0][0]; ax.plot(time_list, torque, alpha=alpha, label=label, zorder=zorder)
            ax = axeses[0][1]; ax.plot(time_list, sfv.force_abs, alpha=alpha, label=label, zorder=zorder)
            ax = axeses[1][0]; ax.plot(time_list, 100*sfv.force_err_abs/sfv.ss_avg_force_magnitude, label=label, alpha=alpha, zorder=zorder)
            ax = axeses[1][1]; ax.plot(time_list, np.arctan2(sfv.force_y, sfv.force_x)/pi*180. - sfv.ss_avg_force_angle, label=label, alpha=alpha, zorder=zorder)

            global count_plot
            count_plot += 1
            # This is used for table in latex
            print '''
            \\newcommand\\torqueAvg%s{%s}
            \\newcommand\\torqueRipple%s{%s}
            \\newcommand\\forceAvg%s{%s}
            \\newcommand\\forceErrMag%s{%s}
            \\newcommand\\forceErrAng%s{%s}
            ''' % (chr(64+count_plot), utility.to_precision(sum(torque[-range_ss:])/len(torque[-range_ss:])),
                   chr(64+count_plot), utility.to_precision(0.5*(max(torque[-range_ss:]) - min(torque[-range_ss:]))),
                   chr(64+count_plot), utility.to_precision(sfv.ss_avg_force_magnitude),
                   chr(64+count_plot), utility.to_precision(0.5*(sfv.ss_max_force_err_abs[0]-sfv.ss_max_force_err_abs[1])/sfv.ss_avg_force_magnitude*100),
                   chr(64+count_plot), utility.to_precision(0.5*(sfv.ss_max_force_err_ang[0]-sfv.ss_max_force_err_ang[1])))

        fig_main, axeses = subplots(2, 2, sharex=True, dpi=150, figsize=(16, 8), facecolor='w', edgecolor='k')
        ax = axeses[0][0]; ax.set_xlabel('(a)',fontsize=14.5); ax.set_ylabel('Torque [Nm]',fontsize=14.5)
        ax = axeses[0][1]; ax.set_xlabel('(b)',fontsize=14.5); ax.set_ylabel('Force Amplitude [N]',fontsize=14.5)
        ax = axeses[1][0]; ax.set_xlabel('Time [s]\n(c)',fontsize=14.5); ax.set_ylabel('Normalized Force Error Magnitude [%]',fontsize=14.5)
        ax = axeses[1][1]; ax.set_xlabel('Time [s]\n(d)',fontsize=14.5); ax.set_ylabel('Force Error Angle [deg]',fontsize=14.5)

        #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        # TranRef400
        #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        study_name = 'TranRef'
        dm = self.read_csv_results_4_comparison__transient(study_name, path_prefix=r'D:\JMAG_Files\TimeStepSensitivity/'+'PS_Qr%d_NoEndRing_M15_17303l/'%(int(self.im.Qr)))
        basic_info, time_list, TorCon_list, ForConX_list, ForConY_list, ForConAbs_list = dm.unpack()
        sfv = suspension_force_vector(ForConX_list, ForConY_list, range_ss=400*10) # samples in the tail that are in steady state
        add_plot( axeses,
                  title=study_name,
                  label='Transient FEA', #'TranFEARef', #400',
                  zorder=1,
                  time_list=time_list,
                  sfv=sfv,
                  torque=TorCon_list,
                  range_ss=sfv.range_ss) 
        print '\tbasic info:', basic_info

        # Current of TranRef
        fig_cur, axes_cur = subplots(2,1)
        ax_cur = axes_cur[0]
        # for key in dm.key_list:
        #     if 'A1' in key: # e.g., ConductorA1
        #         ax_cur.plot(dm.Current_dict['Time(s)'], 
        #                 dm.Current_dict[key], 
        #                 label=study_name, #+'40',
        #                 alpha=0.7)
        # Current of TranRef400
        for key in dm.key_list:
            # if 'Coil' in key:
            #     ax_cur.plot(dm.Current_dict['Time(s)'], 
            #             dm.Current_dict[key], 
            #             label=key,
            #             alpha=0.7)
            if 'A1' in key: # e.g., ConductorA1
                ax_cur.plot(dm.Current_dict['Time(s)'], 
                        dm.Current_dict[key], 
                        label=study_name+'400',
                        alpha=0.7,
                        color='blue') #'purple')
                basefreqFFT(dm.Current_dict[key], Fs, ax=axes_cur[1])
                print '\tcheck time step', time_list[1], '=',  1./Fs
                break

        # basefreqFFT(np.sin(2*pi*1500*np.arange(0,0.1,1./Fs)), Fs)
        fig, axes = subplots(2,1)
        basefreqFFT(TorCon_list[int(len(TorCon_list)/2):], Fs, base_freq=500., ax=axes[1], ax_time_domain=axes[0])
        fig, axes = subplots(2,1)
        basefreqFFT(ForConAbs_list[int(len(TorCon_list)/2):], Fs, ax=axes[1], ax_time_domain=axes[0])


        # ''' TranRef '''
        # study_name = 'TranRef'
        # dm = self.read_csv_results_4_comparison__transient(study_name)        
        # basic_info, time_list, TorCon_list, ForConX_list, ForConY_list, ForConAbs_list = dm.unpack()
        # add_plot( axeses,
        #           title=study_name,
        #           label='TranFEARef40',
        #           zorder=5,
        #           time_list=time_list,
        #           force_x=ForConX_list,
        #           force_y=ForConY_list,
        #           force_abs=ForConAbs_list,
        #           torque=TorCon_list,
        #           range_ss=480) # samples in the tail that are in steady state
        # print '\tbasic info:', basic_info


        #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        # Tran2TSS
        #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        study_name = 'Tran2TSS'
        dm = self.read_csv_results_4_comparison__transient(study_name)
        basic_info, time_list, TorCon_list, ForConX_list, ForConY_list, ForConAbs_list = dm.unpack()
        end_time = time_list[-1]
        sfv = suspension_force_vector(ForConX_list, ForConY_list, range_ss=48) # samples in the tail that are in steady state
        add_plot( axeses,
                  title=study_name,
                  label='Transient FEA w/ 2 Time Step Sections',
                  zorder=8,
                  time_list=time_list,
                  sfv=sfv,
                  torque=TorCon_list,
                  range_ss=sfv.range_ss)
        print '\tbasic info:', basic_info

        # Current of Tran2TSS
        for key in dm.key_list:
            if 'A1' in key: # e.g., ConductorA1
                ax_cur.plot(dm.Current_dict['Time(s)'], 
                        dm.Current_dict[key], 
                        label=study_name,
                        alpha=0.7)
                break

        # Current of FEMM
        if femm_rotor_current_function!=None:
            ax_cur.plot(dm.Current_dict['Time(s)'], 
                    [femm_rotor_current_function(t) for t in dm.Current_dict['Time(s)']], 
                    label='FEMM',
                    alpha=1,
                    c='r')

        #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        # Static FEA with FEMM
        #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        study_name = 'FEMM'
        rotor_position_in_deg = femm_solver_data[0]*0.1 
        time_list = rotor_position_in_deg/180.*pi / self.im.Omega
        number_of_repeat = int(end_time / time_list[-1]) + 2
        femm_force_x = femm_solver_data[2].tolist()
        femm_force_y = femm_solver_data[3].tolist()        
        femm_force_abs = np.sqrt(np.array(femm_force_x)**2 + np.array(femm_force_y)**2 )

        # 延拓
        femm_torque  = number_of_repeat * femm_solver_data[1].tolist()
        time_one_step = time_list[1]
        time_list    = [i*time_one_step for i in range(len(femm_torque))]
        femm_force_x = number_of_repeat * femm_solver_data[2].tolist()
        femm_force_y = number_of_repeat * femm_solver_data[3].tolist()
        femm_force_abs = number_of_repeat * femm_force_abs.tolist()

        sfv = suspension_force_vector(femm_force_x, femm_force_y, range_ss=len(rotor_position_in_deg)) # samples in the tail that are in steady state
        add_plot( axeses,
                  title=study_name,
                  label='Static FEA', #'StaticFEAwiRR',
                  zorder=3,
                  time_list=time_list,
                  sfv=sfv,
                  torque=femm_torque,
                  range_ss=sfv.range_ss,
                  alpha=0.5) 

        #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        # EddyCurrent
        #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        study_name = 'Freq-FFVRC'
        dm = self.read_csv_results_4_comparison_eddycurrent(study_name)
        _, _, TorCon_list, ForConX_list, ForConY_list, ForConAbs_list = dm.unpack()

        rotor_position_in_deg = 360./self.im.Qr / len(TorCon_list) * np.arange(0, len(TorCon_list))
        # print rotor_position_in_deg
        time_list = rotor_position_in_deg/180.*pi / self.im.Omega
        number_of_repeat = int(end_time / time_list[-1])

        # 延拓
        ec_torque        = number_of_repeat*TorCon_list
        time_one_step = time_list[1]
        time_list     = [i*time_one_step for i in range(len(ec_torque))]
        ec_force_abs     = number_of_repeat*ForConAbs_list.tolist()
        ec_force_x       = number_of_repeat*ForConX_list
        ec_force_y       = number_of_repeat*ForConY_list

        sfv = suspension_force_vector(ec_force_x, ec_force_y, range_ss=len(rotor_position_in_deg))
        add_plot( axeses,
                  title=study_name,
                  label='Eddy Current FEA', #'EddyCurFEAwiRR',
                  zorder=2,
                  time_list=time_list,
                  sfv=sfv,
                  torque=ec_torque,
                  range_ss=sfv.range_ss) # samples in the tail that are in steady state

        # # Force Vector plot
        # ax = figure().gca()
        # # ax.text(0,0,'FEMM')
        # for x, y in zip(femm_force_x, femm_force_y):
        #     ax.arrow(0,0, x,y)
        # xlim([0,220])
        # ylim([0,220])
        # ax.grid()
        # ax.set_xlabel('Force X (FEMM) [N]'); 
        # ax.set_ylabel('Force Y (FEMM) [N]')

        for ax in [axeses[0][0],axeses[0][1],axeses[1][0],axeses[1][1]]:
            ax.grid()
            ax.legend(loc='lower center')
            ax.set_xlim([0,0.35335])


        axeses[0][1].set_ylim([260, 335])
        axeses[1][0].set_ylim([-0.06, 0.06])

        ax_cur.legend()
        ax_cur.grid()
        ax_cur.set_xlabel('Time [s]'); 
        ax_cur.set_ylabel('Rotor Current (of One Bar) [A]')
        # plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)


        fig_main.tight_layout()
        if int(self.im.Qr) == 36:
            fig_main.savefig('FEA_Model_Comparisons.png', dpi=150)
            fig_main.savefig(r'D:\OneDrive\[00]GetWorking\31 BlessIMDesign\p2019_iemdc_bearingless_induction full paper\images\FEA_Model_Comparisons.png', dpi=150)

    # ECCE19
    def read_csv_results_4_optimization(self, study_name, path_prefix=None):
        if path_prefix == None:
            path_prefix = self.dir_csv_output_folder
        # print 'look into:', path_prefix

        # Torque
        basic_info = []
        time_list = []
        TorCon_list = []
        with open(path_prefix + study_name + '_torque.csv', 'r') as f:
            read_iterator = csv_reader(f, skipinitialspace=True)
            count = 0
            for row in self.whole_row_reader(read_iterator):
                count +=1
                if count<=8:
                    try:
                        float(row[1])
                    except:
                        continue
                    else:
                        basic_info.append((row[0], float(row[1])))
                else:
                    time_list.append(float(row[0]))
                    TorCon_list.append(float(row[1]))

        # Force
        basic_info = []
        # time_list = []
        ForConX_list = []
        ForConY_list = []
        with open(path_prefix + study_name + '_force.csv', 'r') as f:
            read_iterator = csv_reader(f, skipinitialspace=True)
            count = 0
            for row in self.whole_row_reader(read_iterator):
                count +=1
                if count<=8:
                    try:
                        float(row[1])
                    except:
                        continue
                    else:
                        basic_info.append((row[0], float(row[1])))
                else:
                    # time_list.append(float(row[0]))
                    ForConX_list.append(float(row[1]))
                    ForConY_list.append(float(row[2]))
        ForConAbs_list = np.sqrt(np.array(ForConX_list)**2 + np.array(ForConY_list)**2 )

        # Current
        key_list = []
        Current_dict = {}
        with open(path_prefix + study_name + '_circuit_current.csv', 'r') as f:
            read_iterator = csv_reader(f, skipinitialspace=True)
            count = 0
            for row in self.whole_row_reader(read_iterator):
                count +=1
                if count<=8:
                    if 'Time' in row[0]: # Time(s)
                        for key in row:
                            key_list.append(key)
                            Current_dict[key] = []
                    else:
                        continue
                else:
                    for ind, val in enumerate(row):
                        Current_dict[key_list[ind]].append(float(val))

        # Terminal Voltage 
        new_key_list = []
        if self.fea_config_dict['delete_results_after_calculation'] == False:
            # file name is by individual_name like ID32-2-4_EXPORT_CIRCUIT_VOLTAGE.csv rather than ID32-2-4Tran2TSS_circuit_current.csv
            with open(path_prefix + study_name[:-8] + "_EXPORT_CIRCUIT_VOLTAGE.csv", 'r') as f:
                read_iterator = csv_reader(f, skipinitialspace=True)
                count = 0
                for row in self.whole_row_reader(read_iterator):
                    count +=1
                    if count==1: # Time | Terminal1 | Terminal2 | ... | Termial6
                        if 'Time' in row[0]: # Time, s
                            for key in row:
                                new_key_list.append(key) # Yes, you have to use a new key list, because the ind below bgeins at 0.
                                Current_dict[key] = []
                        else:
                            raise Exception('Problem with csv file for terminal voltage.')
                    else:
                        for ind, val in enumerate(row):
                            Current_dict[new_key_list[ind]].append(float(val))
        key_list += new_key_list

        # Loss
        # Iron Loss
        with open(path_prefix + study_name + '_iron_loss_loss.csv', 'r') as f:
            read_iterator = csv_reader(f, skipinitialspace=True)
            count = 0
            for row in self.whole_row_reader(read_iterator):
                count +=1
                if count>8:
                    stator_iron_loss = float(row[3]) # Stator Core
                    break
        with open(path_prefix + study_name + '_joule_loss_loss.csv', 'r') as f:
            read_iterator = csv_reader(f, skipinitialspace=True)
            count = 0
            for row in self.whole_row_reader(read_iterator):
                count +=1
                if count>8:
                    stator_eddycurrent_loss = float(row[3]) # Stator Core
                    break
        with open(path_prefix + study_name + '_hysteresis_loss_loss.csv', 'r') as f:
            read_iterator = csv_reader(f, skipinitialspace=True)
            count = 0
            for row in self.whole_row_reader(read_iterator):
                count +=1
                if count>8:
                    stator_hysteresis_loss = float(row[3]) # Stator Core
                    break
        # Copper Loss
        rotor_copper_loss_list = []
        with open(path_prefix + study_name + '_joule_loss.csv', 'r') as f:
            read_iterator = csv_reader(f, skipinitialspace=True)
            count = 0
            for row in self.whole_row_reader(read_iterator):
                count +=1
                if count>8:
                    if count==9:
                        stator_copper_loss = float(row[8]) # Coil # it is the same over time, this value does not account for end coil

                    rotor_copper_loss_list.append(float(row[7])) # Cage
        
        # use the last 1/4 period data to compute average copper loss of Tran2TSS rather than use that of Freq study
        effective_part = rotor_copper_loss_list[:int(0.5*self.fea_config_dict['number_of_steps_2ndTTS'])] # number_of_steps_2ndTTS = steps for half peirod
        rotor_copper_loss = sum(effective_part) / len(effective_part)

        if self.fea_config_dict['jmag_run_list'][0] == 0:
            utility.blockPrint()
            try:
                # convert rotor current results (complex number) into its amplitude
                self.femm_solver.list_rotor_current_amp = [abs(el) for el in self.femm_solver.vals_results_rotor_current] # el is complex number
                # settings not necessarily be consistent with Pyrhonen09's design: , STATOR_SLOT_FILL_FACTOR=0.5, ROTOR_SLOT_FILL_FACTOR=1., TEMPERATURE_OF_COIL=75
                s, r = self.femm_solver.get_copper_loss(self.femm_solver.stator_slot_area, self.femm_solver.rotor_slot_area)
            except Exception as e:
                raise e
            utility.enablePrint()
        else:
            s, r = None, None

        dm = data_manager()
        dm.basic_info     = basic_info
        dm.time_list      = time_list
        dm.TorCon_list    = TorCon_list
        dm.ForConX_list   = ForConX_list
        dm.ForConY_list   = ForConY_list
        dm.ForConAbs_list = ForConAbs_list
        dm.Current_dict   = Current_dict
        dm.key_list       = key_list
        dm.jmag_loss_list    = [stator_copper_loss, rotor_copper_loss, stator_iron_loss, stator_eddycurrent_loss, stator_hysteresis_loss]
        dm.femm_loss_list = [s, r]
        return dm

    # IEMDC19
    def read_csv_results_4_comparison__transient(self, study_name, path_prefix=None):
        if path_prefix == None:
            path_prefix = self.dir_csv_output_folder
        # print 'look into:', path_prefix

        # Torque
        basic_info = []
        time_list = []
        TorCon_list = []
        with open(path_prefix + study_name + '_torque.csv', 'r') as f:
            read_iterator = csv_reader(f, skipinitialspace=True)
            count = 0
            for row in self.whole_row_reader(read_iterator):
                count +=1
                if count<=8:
                    try:
                        float(row[1])
                    except:
                        continue
                    else:
                        basic_info.append((row[0], float(row[1])))
                else:
                    time_list.append(float(row[0]))
                    TorCon_list.append(float(row[1]))

        # Force
        basic_info = []
        # time_list = []
        ForConX_list = []
        ForConY_list = []
        with open(path_prefix + study_name + '_force.csv', 'r') as f:
            read_iterator = csv_reader(f, skipinitialspace=True)
            count = 0
            for row in self.whole_row_reader(read_iterator):
                count +=1
                if count<=8:
                    try:
                        float(row[1])
                    except:
                        continue
                    else:
                        basic_info.append((row[0], float(row[1])))
                else:
                    # time_list.append(float(row[0]))
                    ForConX_list.append(float(row[1]))
                    ForConY_list.append(float(row[2]))
        ForConAbs_list = np.sqrt(np.array(ForConX_list)**2 + np.array(ForConY_list)**2 )

        # Current
        key_list = []
        Current_dict = {}
        with open(path_prefix + study_name + '_circuit_current.csv', 'r') as f:
            read_iterator = csv_reader(f, skipinitialspace=True)
            count = 0
            for row in self.whole_row_reader(read_iterator):
                count +=1
                if count<=8:
                    if 'Time' in row[0]:
                        for key in row:
                            key_list.append(key)
                            Current_dict[key] = []
                    else:
                        continue
                else:
                    for ind, val in enumerate(row):
                        Current_dict[key_list[ind]].append(float(val))

        dm = data_manager()
        dm.basic_info     = basic_info
        dm.time_list      = time_list
        dm.TorCon_list    = TorCon_list
        dm.ForConX_list   = ForConX_list
        dm.ForConY_list   = ForConY_list
        dm.ForConAbs_list = ForConAbs_list
        dm.Current_dict   = Current_dict
        dm.key_list       = key_list
        return dm

    def read_csv_results_4_comparison_eddycurrent(self, study_name):
        path_prefix = self.dir_csv_output_folder
        # Torque
        TorCon_list = []
        with open(path_prefix + study_name + '_torque.csv', 'r') as f:
            read_iterator = csv_reader(f, skipinitialspace=True)
            count = 0
            for row in self.whole_row_reader(read_iterator):
                count +=1
                if count<=5:
                    continue
                else:
                    TorCon_list.append(float(row[1]))

        # Force
        ForConX_list = []
        ForConY_list = []
        with open(path_prefix + study_name + '_force.csv', 'r') as f:
            read_iterator = csv_reader(f, skipinitialspace=True)
            count = 0
            for row in self.whole_row_reader(read_iterator):
                count +=1
                if count<=5:
                    continue
                else:
                    ForConX_list.append(float(row[1]))
                    ForConY_list.append(float(row[2]))
        ForConAbs_list = np.sqrt(np.array(ForConX_list)**2 + np.array(ForConY_list)**2 )

        dm = data_manager()
        dm.basic_info     = None
        dm.time_list      = None
        dm.TorCon_list    = TorCon_list
        dm.ForConX_list   = ForConX_list
        dm.ForConY_list   = ForConY_list
        dm.ForConAbs_list = ForConAbs_list
        return dm

    def timeStepSensitivity(self):
        from pylab import figure, show, subplots, xlim, ylim
        print '\n\n\n-----------timeStepSensitivity------------'

        fig, axes = subplots(2, 1, sharex=True)

        ''' Super TranRef '''
        path_prefix = r'D:\JMAG_Files\TimeStepSensitivity/'
        dm = self.read_csv_results_4_comparison__transient('TranRef', path_prefix=path_prefix+self.run_folder)
        basic_info, time_list, TorCon_list, ForConX_list, ForConY_list, ForConAbs_list = dm.unpack()

        study_name = 'SuperTranRef'
        ax = axes[0]; ax.plot(time_list, TorCon_list, alpha=0.7, label=study_name); ax.set_xlabel('Time [s]'); ax.set_ylabel('Torque [Nm]')
        ax = axes[1]; ax.plot(time_list, ForConX_list, alpha=0.7, label=study_name+'-X'); ax.plot(time_list, ForConY_list, alpha=0.7, label=study_name+'Y'); ax.set_xlabel('Time [s]'); ax.set_ylabel('Force [N]')
        ax.plot(time_list, ForConAbs_list, alpha=0.7, label=study_name+'-Abs')


        Avg_ForCon_Vector, _, Max_ForCon_Err_Angle = self.get_force_error_angle(ForConX_list[-48:], ForConY_list[-48:])
        print '---------------\nTranRef-FEA \nForce Average Vecotr:', Avg_ForCon_Vector, '[N]'
        # print ForCon_Angle_List, 'deg'
        print 'Maximum Force Angle Error', Max_ForCon_Err_Angle, '[deg]'
        print 'basic info:', basic_info


        ''' TranRef '''
        study_name = 'TranRef'
        dm = self.read_csv_results_4_comparison__transient(study_name)
        basic_info, time_list, TorCon_list, ForConX_list, ForConY_list, ForConAbs_list = dm.unpack()
        end_time = time_list[-1]

        ax = axes[0]; ax.plot(time_list, TorCon_list, alpha=0.7, label=study_name); ax.set_xlabel('Time [s]'); ax.set_ylabel('Torque [Nm]')
        ax = axes[1]; ax.plot(time_list, ForConX_list, alpha=0.7, label=study_name+'-X'); ax.plot(time_list, ForConY_list, alpha=0.7, label=study_name+'Y'); ax.set_xlabel('Time [s]'); ax.set_ylabel('Force [N]')
        ax.plot(time_list, ForConAbs_list, alpha=0.7, label=study_name+'-Abs')


        Avg_ForCon_Vector, _, Max_ForCon_Err_Angle = self.get_force_error_angle(ForConX_list[-48:], ForConY_list[-48:])
        print '---------------\nTranRef-FEA \nForce Average Vecotr:', Avg_ForCon_Vector, '[N]'
        # print ForCon_Angle_List, 'deg'
        print 'Maximum Force Angle Error', Max_ForCon_Err_Angle, '[deg]'
        print 'basic info:', basic_info


        axes[0].grid()
        axes[0].legend()
        axes[1].grid()
        axes[1].legend()

    def duplicate_TranFEAwi2TSS_from_frequency_study(self, im_variant, slip_freq_breakdown_torque, app, model, original_study_name, tran2tss_study_name, logger):
        if model.NumStudies()<2:
            model.DuplicateStudyWithType(original_study_name, u"Transient2D", tran2tss_study_name)
            app.SetCurrentStudy(tran2tss_study_name)
            study = app.GetCurrentStudy()
            self.study = study

            # 上一步的铁磁材料的状态作为下一步的初值，挺好，但是如果每一个转子的位置转过很大的话，反而会减慢非线性迭代。
            # 我们的情况是：0.33 sec 分成了32步，每步的时间大概在0.01秒，0.01秒乘以0.5*497 Hz = 2.485 revolution...
            # study.GetStudyProperties().SetValue(u"NonlinearSpeedup", 0) # JMAG17.1以后默认使用。现在后面密集的步长还多一点（32步），前面16步慢一点就慢一点呗！

            # two sections of different time step
            if True: # ECCE19
                number_of_steps_2ndTTS = self.fea_config_dict['number_of_steps_2ndTTS'] 
                DM = app.GetDataManager()
                DM.CreatePointArray(u"point_array/timevsdivision", u"SectionStepTable")
                refarray = [[0 for i in range(3)] for j in range(3)]
                refarray[0][0] = 0
                refarray[0][1] =    1
                refarray[0][2] =        50
                refarray[1][0] = 0.5/slip_freq_breakdown_torque #0.5 for 17.1.03l # 1 for 17.1.02y
                refarray[1][1] =    16                          # 16 for 17.1.03l #32 for 17.1.02y
                refarray[1][2] =        50
                refarray[2][0] = refarray[1][0] + 0.5/im_variant.DriveW_Freq #0.5 for 17.1.03l 
                refarray[2][1] =    number_of_steps_2ndTTS  # also modify range_ss! # don't forget to modify below!
                refarray[2][2] =        50
                DM.GetDataSet(u"SectionStepTable").SetTable(refarray)
                number_of_total_steps = 1 + 16 + number_of_steps_2ndTTS # [Double Check] don't forget to modify here!
                study.GetStep().SetValue(u"Step", number_of_total_steps)
                study.GetStep().SetValue(u"StepType", 3)
                study.GetStep().SetTableProperty(u"Division", DM.GetDataSet(u"SectionStepTable"))

            else: # IEMDC19
                number_cycles_prolonged = 1 # 50
                DM = app.GetDataManager()
                DM.CreatePointArray(u"point_array/timevsdivision", u"SectionStepTable")
                refarray = [[0 for i in range(3)] for j in range(4)]
                refarray[0][0] = 0
                refarray[0][1] =    1
                refarray[0][2] =        50
                refarray[1][0] = 1.0/slip_freq_breakdown_torque
                refarray[1][1] =    32 
                refarray[1][2] =        50
                refarray[2][0] = refarray[1][0] + 1.0/im_variant.DriveW_Freq
                refarray[2][1] =    48 # don't forget to modify below!
                refarray[2][2] =        50
                refarray[3][0] = refarray[2][0] + number_cycles_prolonged/im_variant.DriveW_Freq # =50*0.002 sec = 0.1 sec is needed to converge to TranRef
                refarray[3][1] =    number_cycles_prolonged*self.fea_config_dict['TranRef-StepPerCycle'] # =50*40, every 0.002 sec takes 40 steps 
                refarray[3][2] =        50
                DM.GetDataSet(u"SectionStepTable").SetTable(refarray)
                study.GetStep().SetValue(u"Step", 1 + 32 + 48 + number_cycles_prolonged*self.fea_config_dict['TranRef-StepPerCycle']) # [Double Check] don't forget to modify here!
                study.GetStep().SetValue(u"StepType", 3)
                study.GetStep().SetTableProperty(u"Division", DM.GetDataSet(u"SectionStepTable"))

            # add equations
            study.GetDesignTable().AddEquation(u"freq")
            study.GetDesignTable().AddEquation(u"slip")
            study.GetDesignTable().AddEquation(u"speed")
            study.GetDesignTable().GetEquation(u"freq").SetType(0)
            study.GetDesignTable().GetEquation(u"freq").SetExpression("%g"%((im_variant.DriveW_Freq)))
            study.GetDesignTable().GetEquation(u"freq").SetDescription(u"Excitation Frequency")
            study.GetDesignTable().GetEquation(u"slip").SetType(0)
            study.GetDesignTable().GetEquation(u"slip").SetExpression("%g"%(im_variant.the_slip))
            study.GetDesignTable().GetEquation(u"slip").SetDescription(u"Slip [1]")
            study.GetDesignTable().GetEquation(u"speed").SetType(1)
            study.GetDesignTable().GetEquation(u"speed").SetExpression(u"freq * (1 - slip) * 30")
            study.GetDesignTable().GetEquation(u"speed").SetDescription(u"mechanical speed of four pole")

            # speed, freq, slip
            study.GetCondition(u"RotCon").SetValue(u"AngularVelocity", u'speed')
            app.ShowCircuitGrid(True)
            study.GetCircuit().GetComponent(u"CS4").SetValue(u"Frequency", u"freq")
            study.GetCircuit().GetComponent(u"CS2").SetValue(u"Frequency", u"freq")

            # max_nonlinear_iteration = 50
            # study.GetStudyProperties().SetValue(u"NonlinearMaxIteration", max_nonlinear_iteration)
            study.GetStudyProperties().SetValue(u"ApproximateTransientAnalysis", 1) # psuedo steady state freq is for PWM drive to use
            study.GetStudyProperties().SetValue(u"SpecifySlip", 1)
            study.GetStudyProperties().SetValue(u"OutputSteadyResultAs1stStep", 0)
            study.GetStudyProperties().SetValue(u"Slip", u"slip")

            # # add other excitation frequencies other than 500 Hz as cases
            # for case_no, DriveW_Freq in enumerate([50.0, slip_freq_breakdown_torque]):
            #     slip = slip_freq_breakdown_torque / DriveW_Freq
            #     study.GetDesignTable().AddCase()
            #     study.GetDesignTable().SetValue(case_no+1, 0, DriveW_Freq)
            #     study.GetDesignTable().SetValue(case_no+1, 1, slip)

            # 你把Tran2TSS计算周期减半！
            # 也要在计算铁耗的时候选择1/4或1/2的数据！（建议1/4）
            # 然后，手动添加end step 和 start step，这样靠谱！2019-01-09：注意设置铁耗条件（iron loss condition）的Reference Start Step和End Step。

            # Iron Loss Calculation Condition
            # Stator 
            cond = study.CreateCondition(u"Ironloss", u"IronLossConStator")
            cond.SetValue(u"RevolutionSpeed", u"freq*60/%d"%(0.5*(im_variant.DriveW_poles)))
            cond.ClearParts()
            sel = cond.GetSelection()
            sel.SelectPartByPosition(-im_variant.Radius_OuterStatorYoke+1e-2, 0 ,0)
            cond.AddSelected(sel)
            # Use FFT for hysteresis to be consistent with FEMM's results and to have a FFT plot
            cond.SetValue(u"HysteresisLossCalcType", 1)
            cond.SetValue(u"PresetType", 3) # 3:Custom
            # Specify the reference steps yourself because you don't really know what JMAG is doing behind you
            cond.SetValue(u"StartReferenceStep", number_of_total_steps+1-number_of_steps_2ndTTS*0.5) # 1/4 period <=> number_of_steps_2ndTTS*0.5
            cond.SetValue(u"EndReferenceStep", number_of_total_steps)
            cond.SetValue(u"UseStartReferenceStep", 1)
            cond.SetValue(u"UseEndReferenceStep", 1)
            cond.SetValue(u"Cyclicity", 4) # specify reference steps for 1/4 period and extend it to whole period
            cond.SetValue(u"UseFrequencyOrder", 1)
            cond.SetValue(u"FrequencyOrder", u"1-50") # Harmonics up to 50th orders 
            # Check CSV reults for iron loss (You cannot check this for Freq study)
            study.GetStudyProperties().SetValue(u"CsvResultTypes", u"Torque;Force;LineCurrent;TerminalVoltage;JouleLoss;TotalDisplacementAngle;JouleLoss_IronLoss;IronLoss_IronLoss;HysteresisLoss_IronLoss")
            # Terminal Voltage/Circuit Voltage: Check for outputing CSV results 
            study.GetCircuit().CreateTerminalLabel(u"Terminal4A", 8, -13)
            study.GetCircuit().CreateTerminalLabel(u"Terminal4B", 8, -11)
            study.GetCircuit().CreateTerminalLabel(u"Terminal4C", 8, -9)
            study.GetCircuit().CreateTerminalLabel(u"Terminal2A", 23, -13)
            study.GetCircuit().CreateTerminalLabel(u"Terminal2B", 23, -11)
            study.GetCircuit().CreateTerminalLabel(u"Terminal2C", 23, -9)
            # Export Stator Core's field results only for iron loss calculation (the csv file of iron loss will be clean with this setting)
                # study.GetMaterial(u"Rotor Core").SetValue(u"OutputResult", 0) # at least one part on the rotor should be output or else a warning "the jplot file does not contains displacement results when you try to calc. iron loss on the moving part." will pop up, even though I don't add iron loss condition on the rotor.
            study.GetMeshControl().SetValue(u"AirRegionOutputResult", 0)
            study.GetMaterial(u"Shaft").SetValue(u"OutputResult", 0)
            study.GetMaterial(u"Cage").SetValue(u"OutputResult", 0)
            study.GetMaterial(u"Coil").SetValue(u"OutputResult", 0)
            # Rotor
                # study.CreateCondition(u"Ironloss", u"IronLossConRotor")
                # study.GetCondition(u"IronLossConRotor").SetValue(u"BasicFrequencyType", 2)
                # study.GetCondition(u"IronLossConRotor").SetValue(u"BasicFrequency", u"slip*freq")
                # study.GetCondition(u"IronLossConRotor").ClearParts()
                # sel = study.GetCondition(u"IronLossConRotor").GetSelection()
                # sel.SelectPartByPosition(-im.Radius_Shaft-1e-2, 0 ,0)
                # study.GetCondition(u"IronLossConRotor").AddSelected(sel)
                # # Use FFT for hysteresis to be consistent with FEMM's results
                # study.GetCondition(u"IronLossConRotor").SetValue(u"HysteresisLossCalcType", 1)
                # study.GetCondition(u"IronLossConRotor").SetValue(u"PresetType", 3)

            # https://www2.jmag-international.com/support/en/pdf/JMAG-Designer_Ver.17.1_ENv3.pdf
            study.GetStudyProperties().SetValue(u"DirectSolverType", 1)

            # it is duplicated study, so no need to set up mesh 
            # run
            self.run_study(im_variant, app, study, clock_time())
        else:
            # the results exist already?
            return 
    
    def run_study(self, im_variant, app, study, toc):
        logger = logging.getLogger(__name__)
        if self.fea_config_dict['JMAG_Scheduler'] == False:
            # if run_list[1] == True:
            study.RunAllCases()
            logger.debug('Time spent on %s is %g s.'%(study.GetName() , clock_time() - toc))
        else:
            job = study.CreateJob()
            job.SetValue(u"Title", study.GetName())
            job.SetValue(u"Queued", True)
            job.Submit(False) # Fallse:CurrentCase, True:AllCases
            logger.debug('Submit %s to queue (Tran2TSS).'%(im_variant.individual_name))
            # wait and check
            # study.CheckForCaseResults()
        app.Save()
        # if the jcf file already exists, it pops a msg window
        # study.WriteAllSolidJcf(self.dir_jcf, im_variant.model_name+study.GetName()+'Solid', True) # True : Outputs cases that do not have results 
        # study.WriteAllMeshJcf(self.dir_jcf, im_variant.model_name+study.GetName()+'Mesh', True)

        # # run
        # if self.fea_config_dict['JMAG_Scheduler'] == False:
        #     study.RunAllCases()
        #     app.Save()
        # else:
        #     job = study.CreateJob()
        #     job.SetValue(u"Title", study.GetName())
        #     job.SetValue(u"Queued", True)
        #     job.Submit(True)
        #     logger.debug('Submit %s to queue (Freq).'%(im_variant.individual_name))
        #     # wait and check
        #     # study.CheckForCaseResults()

    def mesh_study(self, im_variant, app, model, study):

        # this `if' judgment is effective only if JMAG-DeleteResultFiles is False 
        # if not study.AnyCaseHasResult(): 
        # mesh
        im_variant.add_mesh(study, model)

        # Export Image
        app.View().ShowAllAirRegions()
        # app.View().ShowMeshGeometry() # 2nd btn
        app.View().ShowMesh() # 3rn btn
        app.View().Zoom(3)
        app.View().Pan(-im_variant.Radius_OuterRotor, 0)
        app.ExportImageWithSize(self.dir_run + model.GetName() + '.png', 2000, 2000)
        app.View().ShowModel() # 1st btn. close mesh view, and note that mesh data will be deleted if only ouput table results are selected.


class bearingless_induction_motor_design(object):

    def update_mechanical_parameters(self, slip_freq=None, syn_freq=None):
        # This function is first introduced to derive the new slip for different fundamental frequencies.
        if syn_freq is None:
            syn_freq = self.DriveW_Freq
        else:
            raise Exception('I do not recommend to modify synchronous speed at instance level. Go update the initial design.')

        if syn_freq == 0.0:
            # lock rotor
            self.the_slip = 0. # this is not totally correct
            if slip_freq == None:            
                self.DriveW_Freq = self.slip_freq_breakdown_torque
                self.BeariW_Freq = self.slip_freq_breakdown_torque
            else:
                self.DriveW_Freq = slip_freq
                self.BeariW_Freq = slip_freq
        else:
            if slip_freq != None:
                # change slip
                self.the_slip = slip_freq / syn_freq
                self.slip_freq_breakdown_torque = slip_freq
            else:
                # change syn_freq
                self.the_slip = self.slip_freq_breakdown_torque / syn_freq

            self.DriveW_Freq = syn_freq
            self.BeariW_Freq = syn_freq

        self.the_speed = self.DriveW_Freq*60. / (0.5*self.DriveW_poles) * (1 - self.the_slip) # rpm

        self.Omega = + self.the_speed / 60. * 2*pi
        self.omega = None # This variable name is devil! you can't tell its electrical or mechanical! #+ self.DriveW_Freq * (1-self.the_slip) * 2*pi
        # self.the_speed = + self.the_speed

        if self.fea_config_dict['flag_optimization'] == False: # or else it becomes annoying
            print '[Update %s]'%(self.ID), self.slip_freq_breakdown_torque, self.the_slip, self.the_speed, self.Omega, self.DriveW_Freq, self.BeariW_Freq

    def __init__(self, row=None, fea_config_dict=None, model_name_prefix='PS'):

        # introspection (settings that may differ for initial design and variant designs)
        self.bool_initial_design = True
        self.fea_config_dict = fea_config_dict
        self.slip_freq_breakdown_torque = None
        self.MODEL_ROTATE = False 
        
        #01 Model Name
        self.model_name_prefix = model_name_prefix # do include 'PS' here

        #02 Pyrhonen Data
        if row != None:
            self.ID = str(row[0])
            self.Qs = row[1]
            self.Qr = row[2]
            self.Angle_StatorSlotSpan = 360. / self.Qs # in deg.
            self.Angle_RotorSlotSpan  = 360. / self.Qr # in deg.

            self.Radius_OuterStatorYoke = row[3]
            self.Radius_InnerStatorYoke = row[4]
            self.Length_AirGap          = row[5]
            self.Radius_OuterRotor      = row[6]
            self.Radius_Shaft           = row[7]

            self.Length_HeadNeckRotorSlot = row[8]
            self.Radius_of_RotorSlot      = row[9]
            self.Location_RotorBarCenter  = row[10]
            self.Width_RotorSlotOpen      = row[11]

            self.Radius_of_RotorSlot2     = row[12]
            self.Location_RotorBarCenter2 = row[13]

            self.Angle_StatorSlotOpen            = row[14]
            self.Width_StatorTeethBody           = row[15]
            self.Width_StatorTeethHeadThickness  = row[16]
            self.Width_StatorTeethNeck           = row[17]

            self.DriveW_poles       = row[18]
            self.DriveW_turns       = row[19] # per slot
            self.DriveW_Rs          = row[20]
            self.DriveW_CurrentAmp  = row[21]
            self.DriveW_Freq        = row[22]

            self.stack_length       = row[23]


            # inferred design parameters
            # self.Radius_InnerStator = self.Length_AirGap + self.Radius_OuterRotor
            try:
                self.parameters_for_imposing_constraints_among_design_parameters = row[24:]
            except IndexError, e:
                logger.error(u'The initial design file you provided is not for the puporse of optimization.', exc_info=True)
        else:
            # this is called from shitty_design re-producer.
            return None # __init__ is required to return None. You cannot (or at least shouldn't) return something else.

        #03 Mechanical Parameters
        self.update_mechanical_parameters(slip_freq=2.75) #, syn_freq=500.)

        #04 Material Condutivity Properties
        self.End_Ring_Resistance = fea_config_dict['End_Ring_Resistance']
        self.Bar_Conductivity = fea_config_dict['Bar_Conductivity']
        self.Copper_Loss = self.DriveW_CurrentAmp**2 / 2 * self.DriveW_Rs * 3
        # self.Resistance_per_Turn = 0.01 # TODO


        #05 Windings & Excitation
        self.l41=[ 'C', 'C', 'A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B', ]
        self.l42=[ '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', ]
        if self.fea_config_dict['DPNV'] == True:
            # DPNV style for one phase: -- oo ++ oo
            self.l21=[  'A', 'A', 'C', 'C', 'B', 'B', 
                        'A', 'A', 'C', 'C', 'B', 'B', 
                        'A', 'A', 'C', 'C', 'B', 'B', 
                        'A', 'A', 'C', 'C', 'B', 'B']
            self.l22=[  '-', '-', 'o', 'o', '+', '+', # 横着读和竖着读都是负零正零。 
                        'o', 'o', '-', '-', 'o', 'o', 
                        '+', '+', 'o', 'o', '-', '-', 
                        'o', 'o', '+', '+', 'o', 'o']
        else:
            # separate style for one phase: ---- ++++
            self.l21=[ 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'A', 'A', ]
            self.l22=[ '-', '-', '+', '+', '+', '+', '-', '-', '-', '-', '+', '+', '+', '+', '-', '-', '-', '-', '+', '+', '+', '+', '-', '-', ]


        if self.DriveW_poles == 2:
            self.BeariW_poles = self.DriveW_poles+2; 
        else:
            self.BeariW_poles = 2; 
        self.BeariW_turns= self.DriveW_turns; # TODO Revision here. 20181117
        self.BeariW_Rs = self.BeariW_turns / self.DriveW_turns * self.DriveW_Rs; 
        self.BeariW_CurrentAmp = 0.025 * self.DriveW_CurrentAmp/0.975; 
        self.BeariW_Freq =self.DriveW_Freq;
        self.dict_coil_connection = {41:self.l41, 42:self.l42, 21:self.l21, 22:self.l22}

        #06 Meshing & Solver Properties
        self.max_nonlinear_iteration = 50 # 30 for transient solve
        self.meshSize_Rotor = 1.8 #1.2 0.6 # mm


        #07: Some Checking
        if abs(self.Location_RotorBarCenter2 - self.Location_RotorBarCenter) < 0.1*(self.Radius_of_RotorSlot + self.Radius_of_RotorSlot2):
            print 'Warning: There is no need to use a drop shape rotor, because the required rotor bar height is not high.'
            self.use_drop_shape_rotor_bar = False
        else:
            self.use_drop_shape_rotor_bar = True

        if abs(self.Qs-self.Qr)<1:
            print 'Warning: Must not use a same Qs and Qr, to avoid synchronous torques created by slot harmonics. - (7.111)'

        self.no_slot_per_pole = self.Qr/self.DriveW_poles
        if self.no_slot_per_pole.is_integer() == False:
            print 'This slot-pole combination will not be applied with pole-specific rotor winding'

        self.RSH = []
        for pm in [-1, +1]:
            for v in [-1, +1]:
                    self.RSH.append( (pm * self.Qr*(1-self.the_slip)/(0.5*self.DriveW_poles) + v)*self.DriveW_Freq )
        # print self.Qr, ', '.join("%g" % (rsh/self.DriveW_Freq) for rsh in self.RSH), '\n'

    @staticmethod
    def get_stator_yoke_diameter_Dsyi(stator_tooth_width_b_ds, area_stator_slot_Sus, stator_inner_radius_r_is, Qs, Width_StatorTeethHeadThickness, Width_StatorTeethNeck):
        stator_inner_radius_r_is_eff = stator_inner_radius_r_is + ( Width_StatorTeethHeadThickness + Width_StatorTeethNeck )*1e-3
        temp = (2*pi*stator_inner_radius_r_is_eff - Qs*stator_tooth_width_b_ds)
        stator_tooth_height_h_ds = ( np.sqrt(temp**2 + 4*pi*area_stator_slot_Sus*Qs) - temp ) / (2*pi)
        stator_yoke_diameter_Dsyi = 2*stator_inner_radius_r_is + 2*stator_tooth_height_h_ds
        return stator_yoke_diameter_Dsyi

    @staticmethod
    def get_rotor_tooth_height_h_dr(rotor_tooth_width_b_dr, area_rotor_slot_Sur, rotor_outer_radius_r_or, Qr, Length_HeadNeckRotorSlot, minimum__area_rotor_slot_Sur):
        rotor_outer_radius_r_or_eff = rotor_outer_radius_r_or - Length_HeadNeckRotorSlot*1e-3

        new__rotor_tooth_width_b_dr = rotor_tooth_width_b_dr
        new__area_rotor_slot_Sur = area_rotor_slot_Sur
        logger = logging.getLogger(__name__)
        thermal_penalty = 0.0
        while True:
            temp = (2*pi*rotor_outer_radius_r_or_eff - Qr*new__rotor_tooth_width_b_dr)
            # 注意，这里用的是numpy的sqrt函数，根号下为负号不会像math.sqrt那样raise Exception，而是返回一个nan。
            operand_in_sqrt = temp**2 - 4*pi*new__area_rotor_slot_Sur*Qr
            if operand_in_sqrt < 0: # if np.isnan(rotor_tooth_height_h_dr) == True:
                # 这里应该能自动选择一个最大的可行转子槽才行！
                # modify rotor geometry to make this work and loop back.

                new__area_rotor_slot_Sur -= area_rotor_slot_Sur*0.05 # decrease by 5% every loop
                logger.warn('Sur=%g too large. Try new value=%g.'%(area_rotor_slot_Sur, new__area_rotor_slot_Sur))
                if new__area_rotor_slot_Sur < minimum__area_rotor_slot_Sur: # minimum__area_rotor_slot_Sur corresponds to 8 MA/m^2 current density
                    new__area_rotor_slot_Sur += area_rotor_slot_Sur*0.05 # don't reduce new__area_rotor_slot_Sur any further
                    new__rotor_tooth_width_b_dr -= rotor_tooth_width_b_dr*0.05 # instead, decrease new__rotor_tooth_width_b_dr
                    logger.warn('Reach minimum__area_rotor_slot_Sur. Bad bound on b_dr.\n\tIn other words, b_dr=%g too wide. Try new value=%g.'%(rotor_tooth_width_b_dr, new__rotor_tooth_width_b_dr))
                thermal_penalty += 0.1
                # raise Exception('There is not enough space for rotor slot or the required rotor current density will not be fulfilled.')
            else:
                rotor_tooth_height_h_dr = ( -np.sqrt(operand_in_sqrt) + temp ) / (2*pi)
                break
        return rotor_tooth_height_h_dr, new__rotor_tooth_width_b_dr, thermal_penalty, new__area_rotor_slot_Sur

    @classmethod
    def local_design_variant(cls, im, number_current_generation, individual_index, design_parameters):
        # Never assign anything to im, you can build your self after calling cls and assign stuff to self

        # unpack design_parameters
        stator_tooth_width_b_ds       = design_parameters[0]*1e-3 # m
        air_gap_length_delta          = design_parameters[1]*1e-3 # m
        Width_RotorSlotOpen           = design_parameters[2]      # mm, rotor slot opening
        rotor_tooth_width_b_dr        = design_parameters[3]*1e-3 # m
        Length_HeadNeckRotorSlot      = design_parameters[4]

        # Constranint #2
        # stator_tooth_width_b_ds imposes constraint on stator slot height
        Width_StatorTeethHeadThickness = design_parameters[6]
        Width_StatorTeethNeck = 0.5 * Width_StatorTeethHeadThickness

        area_stator_slot_Sus    = im.parameters_for_imposing_constraints_among_design_parameters[0]
        stator_inner_radius_r_is = im.Radius_OuterRotor*1e-3 + air_gap_length_delta 
        stator_yoke_diameter_Dsyi = cls.get_stator_yoke_diameter_Dsyi(  stator_tooth_width_b_ds, 
                                                                    area_stator_slot_Sus, 
                                                                    stator_inner_radius_r_is,
                                                                    im.Qs,
                                                                    Width_StatorTeethHeadThickness,
                                                                    Width_StatorTeethNeck)

        # Constranint #3
        # rotor_tooth_width_b_dr imposes constraint on rotor slot height
        area_rotor_slot_Sur = im.parameters_for_imposing_constraints_among_design_parameters[1]
        rotor_outer_radius_r_or = im.Radius_OuterRotor*1e-3
        # overwrite rotor_tooth_width_b_dr if there is not enough space for rotor slot
        rotor_tooth_height_h_dr, rotor_tooth_width_b_dr, thermal_penalty, new__area_rotor_slot_Sur = cls.get_rotor_tooth_height_h_dr(  rotor_tooth_width_b_dr,
                                                                area_rotor_slot_Sur,
                                                                rotor_outer_radius_r_or,
                                                                im.Qr,
                                                                Length_HeadNeckRotorSlot,
                                                                im.parameters_for_imposing_constraints_among_design_parameters[2])
        rotor_slot_height_h_sr = rotor_tooth_height_h_dr

        # radius of outer rotor slot
        Radius_of_RotorSlot = 1e3 * (2*pi*(im.Radius_OuterRotor - Length_HeadNeckRotorSlot)*1e-3 - rotor_tooth_width_b_dr*im.Qr) / (2*im.Qr+2*pi)
        Location_RotorBarCenter = im.Radius_OuterRotor - Length_HeadNeckRotorSlot - Radius_of_RotorSlot

        # Constraint #1: Rotor slot opening cannot be larger than rotor slot width.
        punishment = 0.0
        if Width_RotorSlotOpen>2*Radius_of_RotorSlot:
            logger = logging.getLogger(__name__)
            logger.warn('Constraint #1: Rotor slot opening cannot be larger than rotor slot width. Gen#%04d. Individual index=%d.', number_current_generation, individual_index)
            # we will plot a model with b1 = rotor_tooth_width_b_dr instead, and apply a punishment for this model
            Width_RotorSlotOpen = 0.95 * 2*Radius_of_RotorSlot # 确保相交
            punishment = 0.0

        # new method for radius of inner rotor slot 2
        Radius_of_RotorSlot2 = 1e3 * (2*pi*(im.Radius_OuterRotor - Length_HeadNeckRotorSlot - rotor_slot_height_h_sr*1e3)*1e-3 - rotor_tooth_width_b_dr*im.Qr) / (2*im.Qr-2*pi)
        Location_RotorBarCenter2 = im.Radius_OuterRotor - Length_HeadNeckRotorSlot - rotor_slot_height_h_sr*1e3 + Radius_of_RotorSlot2 

        # translate design_parameters into row variable
        row_translated_from_design_paramters = \
            [   im.ID + '-' + str(number_current_generation) + '-' + str(individual_index), # the ID is str
                im.Qs,
                im.Qr,
                im.Radius_OuterStatorYoke,
                0.5*stator_yoke_diameter_Dsyi * 1e3, # 定子内轭部处的半径由需要的定子槽面积和定子齿宽来决定。
                design_parameters[1],                # [1] # Length_AirGap
                im.Radius_OuterRotor,
                im.Radius_Shaft, 
                Length_HeadNeckRotorSlot,            # [4]
                Radius_of_RotorSlot,                 # inferred from [3]
                Location_RotorBarCenter, 
                Width_RotorSlotOpen,                 # [2]
                Radius_of_RotorSlot2,
                Location_RotorBarCenter2,
                design_parameters[5],                # [5] # Angle_StatorSlotOpen
                design_parameters[0],                # [0] # Width_StatorTeethBody
                Width_StatorTeethHeadThickness,      # [6]
                Width_StatorTeethNeck,
                im.DriveW_poles,     
                im.DriveW_turns, # turns per slot
                im.DriveW_Rs,        
                im.DriveW_CurrentAmp,
                im.DriveW_Freq,
                im.stack_length
            ] + im.parameters_for_imposing_constraints_among_design_parameters
                # im.parameters_for_imposing_constraints_among_design_parameters[0], # area_stator_slot_Sus
                # im.parameters_for_imposing_constraints_among_design_parameters[1], # area_rotor_slot_Sur 
                # im.parameters_for_imposing_constraints_among_design_parameters[2]  # minimum__area_rotor_slot_Sur
        # initialze with the class's __init__ method
        self = cls( row_translated_from_design_paramters, 
                    im.fea_config_dict, 
                    im.model_name_prefix)

        # introspection (settings that may differ for initial design and variant designs)
        self.bool_initial_design = False # no, this is a variant design of the initial design
        self.fea_config_dict = im.fea_config_dict # important FEA configuration data
        self.slip_freq_breakdown_torque = None # initialize this slip freq for FEMM or JMAG
        self.MODEL_ROTATE = False # during optimization, rotate at model level is not needed.
        logger = logging.getLogger(__name__) 
        logger.info('im_variant ID %s is initialized.', self.ID)

        # thermal penalty for reduced rotor slot area (copper hot) and rotor tooth width (iron hot)
        # try:
            # thermal_penalty
        # except:
            # thermal_penalty = 0.0
        if thermal_penalty != 0:
            with open(self.fea_config_dict['dir_parent'] + 'pop/' + self.fea_config_dict['run_folder'] + 'thermal_penalty_individuals.txt', 'a') as f:
                f.write(self.get_individual_name() + ',%g,%g,%g\n'%(thermal_penalty, rotor_tooth_width_b_dr, new__area_rotor_slot_Sur))
        self.thermal_penalty = thermal_penalty
        return self

    @classmethod
    def reproduce_the_problematic_design(cls, path_to_shitty_design_file):
        self = cls()
        with open(path_to_shitty_design_file, 'r') as f:
            buf = f.readlines()
            while True:
                title = buf.pop(0) # pop out the first item
                if 'Bearingless' in title: # Have we removed the title line?
                    break
            while True:
                # the last line does not end in ', \n', so remove it first
                last_line = buf.pop()
                if len(last_line) > 1:
                    exec('self.' + last_line[8:])
                    break

        for ind, line in enumerate(buf):
            index_equal_symbol = line.find('=') 
            # the member variable that is a string is problematic
            if 'ID' in line[:index_equal_symbol] or 'name' in line[:index_equal_symbol]: # 在等号之前出现了ID或者name，说明这个变量很可能是字符串！
                index_comma_symbol = line.find(',')
                line = line[:index_equal_symbol+2] + '"' + line[index_equal_symbol+2:index_comma_symbol] + '"' + line[index_comma_symbol:]

            # the leading 8 char are space, while the ending 3 char are ', \n'
            exec('self.' + line[8:-3]) 
        return self

    def whole_row_reader(self, reader):
        for row in reader:
            yield row[:]

    def show(self, toString=False):
        attrs = vars(self).items()
        key_list = [el[0] for el in attrs]
        val_list = [el[1] for el in attrs]
        the_dict = dict(zip(key_list, val_list))
        sorted_key = sorted(key_list, key=lambda item: (int(item.partition(' ')[0]) if item[0].isdigit() else float('inf'), item)) # this is also useful for string beginning with digiterations '15 Steel'.
        tuple_list = [(key, the_dict[key]) for key in sorted_key]
        if toString==False:
            print '- Bearingless Induction Motor Individual #%s\n\t' % (self.ID),
            print ', \n\t'.join("%s = %s" % item for item in tuple_list)
            return ''
        else:
            return u'\n- Bearingless Induction Motor Individual #%s\n\t' % (self.ID) + ', \n\t'.join("%s = %s" % item for item in tuple_list)

    def pre_process(self, app):
        # pre-process : you can select part by coordinate!
        ''' Group '''
        def group(name, id_list):
            model.GetGroupList().CreateGroup(name)
            for the_id in id_list:
                model.GetGroupList().AddPartToGroup(name, the_id)

        model = app.GetCurrentModel() # model = app.GetModel(u"IM_DEMO_1")
        part_ID_list = model.GetPartIDs()
        # print part_ID_list

        # view = app.View()
        # view.ClearSelect()
        # sel = view.GetCurrentSelection()
        # sel.SelectPart(123)
        # sel.SetBlockUpdateView(False)

        global partIDRange_Cage

        # if len(part_ID_list) != int(1 + 1 + 1 + self.Qr + self.Qs*2 + self.Qr + 1): the last +1 is for the air hug rotor
        # if len(part_ID_list) != int(1 + 1 + 1 + self.Qr + self.Qs*2 + self.Qr):
        if len(part_ID_list) != int(1 + 1 + 1 + self.Qr + self.Qs*2):
            msg = 'Number of Parts is unexpected.\n' + self.show(toString=True)
            utility.send_notification(text=msg)
            raise Exception(msg)

        id_shaft = part_ID_list[0]
        id_rotorCore = part_ID_list[1]
        partIDRange_Cage = part_ID_list[2 : 2+int(self.Qr)]
        id_statorCore = part_ID_list[3+int(self.Qr)]
        partIDRange_Coil = part_ID_list[3+int(self.Qr) : 3+int(self.Qr) + int(self.Qs*2)]
        # partIDRange_AirWithinRotorSlots = part_ID_list[3+int(self.Qr) + int(self.Qs*2) : 3+int(self.Qr) + int(self.Qs*2) + int(self.Qr)]

        # print part_ID_list
        # print partIDRange_Cage
        # print partIDRange_Coil
        # print partIDRange_AirWithinRotorSlots
        group(u"Cage", partIDRange_Cage) # 59-44 = 15 = self.Qr - 1
        group(u"Coil", partIDRange_Coil) # 107-60 = 47 = 48-1 = self.Qs*2 - 1
        # group(u"AirWithinRotorSlots", partIDRange_AirWithinRotorSlots) # 123-108 = 15 = self.Qr - 1


        ''' Add Part to Set for later references '''
        def part_set(name, x, y):
            model.GetSetList().CreatePartSet(name)
            model.GetSetList().GetSet(name).SetMatcherType(u"Selection")
            model.GetSetList().GetSet(name).ClearParts()
            sel = model.GetSetList().GetSet(name).GetSelection()
            # print x,y
            sel.SelectPartByPosition(x,y,0) # z=0 for 2D
            model.GetSetList().GetSet(name).AddSelected(sel)

        # def edge_set(name,x,y):
        #     model.GetSetList().CreateEdgeSet(name)
        #     model.GetSetList().GetSet(name).SetMatcherType(u"Selection")
        #     model.GetSetList().GetSet(name).ClearParts()
        #     sel = model.GetSetList().GetSet(name).GetSelection()
        #     sel.SelectEdgeByPosition(x,y,0) # sel.SelectEdge(741)
        #     model.GetSetList().GetSet(name).AddSelected(sel)
        # edge_set(u"AirGapCoast", 0, self.Radius_OuterRotor+0.5*self.Length_AirGap)

        # Create Set for Shaft
        part_set(u"ShaftSet", 0.0, 0.0)

        # Create Set for 4 poles Winding
        R = 0.5*(self.Radius_InnerStatorYoke  +  (self.Radius_OuterRotor+self.Width_StatorTeethHeadThickness+self.Width_StatorTeethNeck)) 
            # THETA = (0.5*(self.Angle_StatorSlotSpan) -  0.05*(self.Angle_StatorSlotSpan-self.Angle_StatorSlotOpen))/180.*pi
        THETA = (0.5*(self.Angle_StatorSlotSpan) -  0.05*(self.Angle_StatorSlotSpan-self.Width_StatorTeethBody))/180.*pi
        X = R*cos(THETA)
        Y = R*sin(THETA)
        # l41=[ 'C', 'C', 'A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B', ]
        # l42=[ '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', ]
        count = 0
        for UVW, UpDown in zip(self.l41,self.l42):
            count += 1 
            part_set(u"Coil4%s%s %d"%(UVW,UpDown,count), X, Y)

            THETA += self.Angle_StatorSlotSpan/180.*pi
            X = R*cos(THETA)
            Y = R*sin(THETA)

        # Create Set for 2 poles Winding
            # THETA = (0.5*(self.Angle_StatorSlotSpan) +  0.05*(self.Angle_StatorSlotSpan-self.Angle_StatorSlotOpen))/180.*pi
        THETA = (0.5*(self.Angle_StatorSlotSpan) +  0.05*(self.Angle_StatorSlotSpan-self.Width_StatorTeethBody))/180.*pi
        X = R*cos(THETA)
        Y = R*sin(THETA)
        # l21=[ 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'A', 'A', ]
        # l22=[ '-', '-', '+', '+', '+', '+', '-', '-', '-', '-', '+', '+', '+', '+', '-', '-', '-', '-', '+', '+', '+', '+', '-', '-', ]
        count = 0
        for UVW, UpDown in zip(self.l21,self.l22):
            count += 1 
            part_set(u"Coil2%s%s %d"%(UVW,UpDown,count), X, Y)

            THETA += self.Angle_StatorSlotSpan/180.*pi
            X = R*cos(THETA)
            Y = R*sin(THETA)



        # Create Set for Bars and Air within rotor slots
        R = self.Location_RotorBarCenter
                                                                                # Another BUG:对于这种槽型，part_set在将AirWithin添加为set的时候，会错误选择到转子导条上！实际上，AirWithinRotorSlots对于JMAG来说完全是没有必要的！
        # R_airR = self.Radius_OuterRotor - 0.1*self.Length_HeadNeckRotorSlot # if the airWithin is too big, minus 1e-2 is not enough anymore
        THETA = pi # it is very important (when Qr is odd) to begin the part set assignment from the first bar you plot.
        X = R*cos(THETA)
        Y = R*sin(THETA)
        list_xy_bars = []
        # list_xy_airWithinRotorSlot = []
        for ind in range(int(self.Qr)):
            natural_ind = ind + 1
            # print THETA / pi *180
            part_set(u"Bar %d"%(natural_ind), X, Y)
            list_xy_bars.append([X,Y])
            # # # part_set(u"AirWithin %d"%(natural_ind), R_airR*cos(THETA),R_airR*sin(THETA))
            # list_xy_airWithinRotorSlot.append([R_airR*cos(THETA),R_airR*sin(THETA)])

            THETA += self.Angle_RotorSlotSpan/180.*pi
            X = R*cos(THETA)
            Y = R*sin(THETA)

        # Create Set for Motion Region
        def part_list_set(name, list_xy, prefix=None):
            model.GetSetList().CreatePartSet(name)
            model.GetSetList().GetSet(name).SetMatcherType(u"Selection")
            model.GetSetList().GetSet(name).ClearParts()
            sel = model.GetSetList().GetSet(name).GetSelection() 
            for xy in list_xy:
                sel.SelectPartByPosition(xy[0],xy[1],0) # z=0 for 2D
                model.GetSetList().GetSet(name).AddSelected(sel)
        # part_list_set(u'Motion_Region', [[0,0],[0,self.Radius_Shaft+1e-2]] + list_xy_bars + list_xy_airWithinRotorSlot) 
        part_list_set(u'Motion_Region', [[0,0],[0,self.Radius_Shaft+1e-2]] + list_xy_bars) 

        # Create Set for Cage
        model.GetSetList().CreatePartSet(u"CageSet")
        model.GetSetList().GetSet(u"CageSet").SetMatcherType(u"MatchNames")
        model.GetSetList().GetSet(u"CageSet").SetParameter(u"style", u"prefix")
        model.GetSetList().GetSet(u"CageSet").SetParameter(u"text", u"Cage")
        model.GetSetList().GetSet(u"CageSet").Rebuild()

    def add_study(self, app, model, dir_csv_output_folder, choose_study_type='frequency'):

        self.choose_study_type = choose_study_type # Transient solves for different Qr, skewing angles and short pitches.

        if self.choose_study_type=='transient':
            self.minimal_time_interval = 1 / 16. / self.DriveW_Freq
            self.end_time = 0.2  #20 / self.DriveW_Freq
            self.no_divisiton = int(self.end_time/self.minimal_time_interval)
            self.no_steps = int(self.no_divisiton + 1)
            # print self.minimal_time_interval, end_time, no_divisiton, no_steps
            # quit()
        elif self.choose_study_type=='frequency': # freq analysis
            self.table_freq_division_refarray = [[0 for i in range(3)] for j in range(4)]
            self.table_freq_division_refarray[0][0] = 2
            self.table_freq_division_refarray[0][1] =   1
            self.table_freq_division_refarray[0][2] =    self.max_nonlinear_iteration
            self.table_freq_division_refarray[1][0] = 10
            self.table_freq_division_refarray[1][1] =   8
            self.table_freq_division_refarray[1][2] =    self.max_nonlinear_iteration
            self.table_freq_division_refarray[2][0] = 16
            self.table_freq_division_refarray[2][1] =   2
            self.table_freq_division_refarray[2][2] =    self.max_nonlinear_iteration
            self.table_freq_division_refarray[3][0] = 24
            self.table_freq_division_refarray[3][1] =   2
            self.table_freq_division_refarray[3][2] =    self.max_nonlinear_iteration

            # self.table_freq_division_refarray = [[0 for i in range(3)] for j in range(2)]
            # self.table_freq_division_refarray[0][0] = 2
            # self.table_freq_division_refarray[0][1] =   1
            # self.table_freq_division_refarray[0][2] =    self.max_nonlinear_iteration
            # self.table_freq_division_refarray[1][0] = 18
            # self.table_freq_division_refarray[1][1] =   4 # for testing the script # 16 
            # self.table_freq_division_refarray[1][2] =    self.max_nonlinear_iteration

            self.no_steps = sum([el[1] for el in self.table_freq_division_refarray])
        else: # static analysis        
            pass
        # print no_steps


        if self.choose_study_type == 'transient':
                # study_name = model.GetName() + u"Tran"
            study_name = self.get_individual_name() + u"Tran"
            model.CreateStudy(u"Transient2D", study_name)
            app.SetCurrentStudy(study_name)
            study = model.GetStudy(study_name)

            study.GetStudyProperties().SetValue(u"ModelThickness", self.stack_length) # Stack Length
            study.GetStudyProperties().SetValue(u"ConversionType", 0)
            study.GetStudyProperties().SetValue(u"ApproximateTransientAnalysis", 1) # psuedo steady state freq is for PWM drive to use
            study.GetStudyProperties().SetValue(u"SpecifySlip", 1)
            study.GetStudyProperties().SetValue(u"Slip", self.the_slip)
            study.GetStudyProperties().SetValue(u"OutputSteadyResultAs1stStep", 0)
            study.GetStudyProperties().SetValue(u"NonlinearMaxIteration", self.max_nonlinear_iteration)
            study.GetStudyProperties().SetValue(u"CsvOutputPath", dir_csv_output_folder) # it's folder rather than file!
            # study.GetStudyProperties().SetValue(u"CsvResultTypes", u"Torque;Force;FEMCoilFlux;LineCurrent;ElectricPower;TerminalVoltage;JouleLoss;TotalDisplacementAngle")
            study.GetStudyProperties().SetValue(u"CsvResultTypes", u"Torque;Force;LineCurrent;TerminalVoltage;JouleLoss;TotalDisplacementAngle")
            study.GetStudyProperties().SetValue(u"TimePeriodicType", 2) # This is for TP-EEC but is not effective
            study.GetStep().SetValue(u"StepType", 1)
            study.GetStep().SetValue(u"Step", self.no_steps)
            study.GetStep().SetValue(u"StepDivision", self.no_divisiton)
            study.GetStep().SetValue(u"EndPoint", self.end_time)
            # study.GetStep().SetValue(u"Step", 501)
            # study.GetStep().SetValue(u"StepDivision", 500)
            # study.GetStep().SetValue(u"EndPoint", 0.5)
            # app.View().SetCurrentCase(1)
        elif self.choose_study_type=='frequency': # freq analysis
            study_name = self.get_individual_name() + u"Freq"
            model.CreateStudy(u"Frequency2D", study_name)
            app.SetCurrentStudy(study_name)
            study = model.GetStudy(study_name)

            study.GetStudyProperties().SetValue(u"NonlinearMaxIteration", self.max_nonlinear_iteration)
            study.GetStudyProperties().SetValue(u"ModelThickness", self.stack_length) # Stack Length
            study.GetStudyProperties().SetValue(u"ConversionType", 0)
            study.GetStudyProperties().SetValue(u"CsvOutputPath", dir_csv_output_folder) # it's folder rather than file!
                # study.GetStudyProperties().SetValue(u"CsvResultTypes", u"Torque;Force;FEMCoilFlux;LineCurrent;ElectricPower;TerminalVoltage;JouleLoss;TotalDisplacementAngle")
            study.GetStudyProperties().SetValue(u"CsvResultTypes", u"Torque;Force;LineCurrent;JouleLoss")
            study.GetStudyProperties().SetValue(u"DeleteResultFiles", self.fea_config_dict['delete_results_after_calculation'])


            DM = app.GetDataManager()
            DM.CreatePointArray(u"point_array/frequency_vs_division", u"table_freq_division")
            # DM.GetDataSet(u"").SetName(u"table_freq_division")
            DM.GetDataSet(u"table_freq_division").SetTable(self.table_freq_division_refarray)
            study.GetStep().SetValue(u"Step", self.no_steps)
            study.GetStep().SetValue(u"StepType", 3)
            study.GetStep().SetTableProperty(u"Division", DM.GetDataSet(u"table_freq_division"))
            # app.View().SetCurrentCase(1)
            # print 'BHCorrection for nonlinear time harmonic analysis is turned ON.'
            study.GetStudyProperties().SetValue(u"BHCorrection", 1)
        else:
            study_name = u"Static"
            model.CreateStudy(u"Static2D", study_name)
            app.SetCurrentStudy(study_name)
            study = model.GetStudy(study_name)

            study.GetStudyProperties().SetValue(u"NonlinearMaxIteration", self.max_nonlinear_iteration)
            study.GetStudyProperties().SetValue(u"ModelThickness", self.stack_length) # Stack Length
            study.GetStudyProperties().SetValue(u"ConversionType", 0)
            study.GetStudyProperties().SetValue(u"CsvOutputPath", dir_csv_output_folder) # it's folder rather than file!
                # study.GetStudyProperties().SetValue(u"CsvResultTypes", u"Torque;Force;FEMCoilFlux;LineCurrent;ElectricPower;TerminalVoltage;JouleLoss;TotalDisplacementAngle")
            study.GetStudyProperties().SetValue(u"CsvResultTypes", u"Torque;Force;LineCurrent;JouleLoss")
            study.GetStudyProperties().SetValue(u"DeleteResultFiles", self.fea_config_dict['delete_results_after_calculation'])

        # Material
        if 'M19' in self.fea_config_dict['Steel']:
            study.SetMaterialByName(u"Stator Core", u"M-19 Steel Gauge-29")
            # study.GetMaterial(u"Stator Core").SetValue(u"UserConductivityValue", 1900000)
            study.GetMaterial(u"Stator Core").SetValue(u"Laminated", 1)
            study.GetMaterial(u"Stator Core").SetValue(u"LaminationFactor", 95)

            study.SetMaterialByName(u"Rotor Core", u"M-19 Steel Gauge-29")
            # study.GetMaterial(u"Rotor Core").SetValue(u"UserConductivityValue", 1900000)
            study.GetMaterial(u"Rotor Core").SetValue(u"Laminated", 1)
            study.GetMaterial(u"Rotor Core").SetValue(u"LaminationFactor", 95)
        elif 'M15' in self.fea_config_dict['Steel']:
            study.SetMaterialByName(u"Stator Core", u"M-15 Steel")
            # study.GetMaterial(u"Stator Core").SetValue(u"UserConductivityValue", 1900000)
            study.GetMaterial(u"Stator Core").SetValue(u"Laminated", 1)
            study.GetMaterial(u"Stator Core").SetValue(u"LaminationFactor", 98)

            study.SetMaterialByName(u"Rotor Core", u"M-15 Steel")
            # study.GetMaterial(u"Rotor Core").SetValue(u"UserConductivityValue", 1900000)
            study.GetMaterial(u"Rotor Core").SetValue(u"Laminated", 1)
            study.GetMaterial(u"Rotor Core").SetValue(u"LaminationFactor", 98)
        elif self.fea_config_dict['Steel'] == 'Arnon5':
            study.SetMaterialByName(u"Stator Core", u"Arnon5-final")
            # study.GetMaterial(u"Stator Core").SetValue(u"UserConductivityValue", 1900000)
            study.GetMaterial(u"Stator Core").SetValue(u"Laminated", 1)
            study.GetMaterial(u"Stator Core").SetValue(u"LaminationFactor", 96)

            study.SetMaterialByName(u"Rotor Core", u"Arnon5-final")
            # study.GetMaterial(u"Rotor Core").SetValue(u"UserConductivityValue", 1900000)
            study.GetMaterial(u"Rotor Core").SetValue(u"Laminated", 1)
            study.GetMaterial(u"Rotor Core").SetValue(u"LaminationFactor", 96)
            # study.SetMaterialByName(u"Rotor Core", u"DCMagnetic Type/50A1000")
            # study.GetMaterial(u"Rotor Core").SetValue(u"UserConductivityType", 1)
            # study.SetMaterialByName(u"Stator Core", u"DCMagnetic Type/50A1000")
            # study.GetMaterial(u"Stator Core").SetValue(u"UserConductivityType", 1)

        study.SetMaterialByName(u"Coil", u"Copper")
        study.GetMaterial(u"Coil").SetValue(u"UserConductivityType", 1)

        study.SetMaterialByName(u"Cage", u"Aluminium")
        study.GetMaterial(u"Cage").SetValue(u"EddyCurrentCalculation", 1)
        study.GetMaterial(u"Cage").SetValue(u"UserConductivityType", 1)
        study.GetMaterial(u"Cage").SetValue(u"UserConductivityValue", self.Bar_Conductivity)

        # Conditions - Motion
        if self.choose_study_type == 'transient':
            study.CreateCondition(u"RotationMotion", u"RotCon")
            # study.GetCondition(u"RotCon").SetXYZPoint(u"", 0, 0, 1) # megbox warning
            study.GetCondition(u"RotCon").SetValue(u"AngularVelocity", int(self.the_speed))
            study.GetCondition(u"RotCon").ClearParts()
            study.GetCondition(u"RotCon").AddSet(model.GetSetList().GetSet(u"Motion_Region"), 0)

            study.CreateCondition(u"Torque", u"TorCon")
            # study.GetCondition(u"TorCon").SetXYZPoint(u"", 0, 0, 0) # megbox warning
            study.GetCondition(u"TorCon").SetValue(u"TargetType", 1)
            study.GetCondition(u"TorCon").SetLinkWithType(u"LinkedMotion", u"RotCon")
            study.GetCondition(u"TorCon").ClearParts()

            study.CreateCondition(u"Force", u"ForCon")
            study.GetCondition(u"ForCon").SetValue(u"TargetType", 1)
            study.GetCondition(u"ForCon").SetLinkWithType(u"LinkedMotion", u"RotCon")
            study.GetCondition(u"ForCon").ClearParts()
        elif self.choose_study_type=='frequency': # freq analysis
            study.CreateCondition(u"FQRotationMotion", u"RotCon")
            # study.GetCondition(u"RotCon").SetXYZPoint(u"", 0, 0, 0)
            study.GetCondition(u"RotCon").ClearParts()
            study.GetCondition(u"RotCon").AddSet(model.GetSetList().GetSet(u"Motion_Region"), 0)

            study.CreateCondition(u"Torque", u"TorCon")
            study.GetCondition(u"TorCon").SetValue(u"TargetType", 1)
            study.GetCondition(u"TorCon").SetLinkWithType(u"LinkedMotion", u"RotCon")
            study.GetCondition(u"TorCon").ClearParts()

            study.CreateCondition(u"Force", u"ForCon")
            study.GetCondition(u"ForCon").SetValue(u"TargetType", 1)
            study.GetCondition(u"ForCon").SetLinkWithType(u"LinkedMotion", u"RotCon")
            study.GetCondition(u"ForCon").ClearParts()
        else: # Static
                # duplicating study can fail if the im instance is destroyed.
                # model.DuplicateStudyWithType(original_study_name, u"Static2D", "Static")
                # study = app.GetCurrentStudy()
            study.CreateCondition(u"Torque", u"TorCon")
            study.GetCondition(u"TorCon").SetValue(u"TargetType", 1)
            study.GetCondition(u"TorCon").ClearParts()
            study.GetCondition(u"TorCon").AddSet(model.GetSetList().GetSet(u"Motion_Region"), 0)

            study.CreateCondition(u"Force", u"ForCon")
            study.GetCondition(u"ForCon").SetValue(u"TargetType", 1)
            study.GetCondition(u"ForCon").ClearParts()
            study.GetCondition(u"ForCon").AddSet(model.GetSetList().GetSet(u"Motion_Region"), 0)

            # 静态场不需要用到电路和FEM Coil/Conductor，这里设置完直接返回了
            # no mesh results are needed
            study.GetStudyProperties().SetValue(u"OnlyTableResults", self.fea_config_dict['OnlyTableResults'])
            study.GetStudyProperties().SetValue(u"Magnetization", 0)
            study.GetStudyProperties().SetValue(u"PermeanceFactor", 0)
            study.GetStudyProperties().SetValue(u"DifferentialPermeability", 0)
            study.GetStudyProperties().SetValue(u"LossDensity", 0)
            study.GetStudyProperties().SetValue(u"SurfaceForceDensity", 0)
            study.GetStudyProperties().SetValue(u"LorentzForceDensity", 0)
            study.GetStudyProperties().SetValue(u"Stress", 0)
            study.GetStudyProperties().SetValue(u"HysteresisLossDensity", 0)
            study.GetStudyProperties().SetValue(u"RestartFile", 0)
            study.GetStudyProperties().SetValue(u"JCGMonitor", 0)
            study.GetStudyProperties().SetValue(u"CoerciveForceNormal", 0)
            study.GetStudyProperties().SetValue(u"Temperature", 0)
            study.GetStudyProperties().SetValue(u"IronLossDensity", 0) # 我们要铁耗作为后处理，而不是和磁场同时求解。（[搜Iron Loss Formulas] Use one of the following methods to calculate iron loss and iron loss density generated in magnetic materials in JMAG. • Calculating Iron Loss Using Only the Magnetic Field Analysis Solver (page 6): It is a method to run magnetic field analysis considering the effect of iron loss. In this method, iron loss condition is not used. • Calculating Iron Loss Using the Iron Loss Analysis Solver (page 8): It is a method to run iron loss analysis using the results data of magnetic field analysis. It will be one of the following procedures. • Run magnetic field analysis study with iron loss condition • Run iron loss analysis study with reference to the result file of magnetic field analysis This chapter describes these two methods.）


            # Linear Solver
            if False:
                # sometime nonlinear iteration is reported to fail and recommend to increase the accerlation rate of ICCG solver
                study.GetStudyProperties().SetValue(u"IccgAccel", 1.2) 
                study.GetStudyProperties().SetValue(u"AutoAccel", 0)
            else:
                # https://www2.jmag-international.com/support/en/pdf/JMAG-Designer_Ver.17.1_ENv3.pdf
                study.GetStudyProperties().SetValue(u"DirectSolverType", 1)

            # too many threads will in turn make them compete with each other and slow down the solve. 2 is good enough for eddy current solve. 6~8 is enough for transient solve.
            study.GetStudyProperties().SetValue(u"UseMultiCPU", True)
            study.GetStudyProperties().SetValue(u"MultiCPU", 2) # this is effective for Transient Solver and 2 is enough!

            self.study_name = study_name
            return study



        # Conditions - FEM Coils (i.e. stator winding)
        # Circuit - Current Source
        app.ShowCircuitGrid(True)
        study.CreateCircuit()
        ''' We can model the circuit in JMAG according to the parallel structure of the winding.
            However, in this case, we do not know how will the CS (Current Source) module behave.
            As required by DPNV, torque current must flow against the suspension inverter.
            If the suspension inverter is a CS, this could not be possible.
            As a result, we model the DPNV as a separate winding with half of slots not exploited by suspension winding.
        '''
            # if self.im.fea_config_dict['DPNV'] == True: # or 'DPNV' in self.model_name_prefix:
            #     # Torque winding
            #     poles = self.DriveW_poles
            #     turns = self.DriveW_turns
            #     Rs=self.DriveW_Rs
            #     amp=self.DriveW_CurrentAmp
            #     freq=self.DriveW_Freq
            #     phase=0

            #     study.GetCircuit().CreateComponent(u"3PhaseCurrentSource", u"CS%d"%(poles))
            #     study.GetCircuit().CreateInstance(u"CS%d"%(poles), -31, 13)
            #     study.GetCircuit().GetComponent(u"CS%d"%(poles)).SetValue(u"Amplitude", amp)
            #     study.GetCircuit().GetComponent(u"CS%d"%(poles)).SetValue(u"Frequency", freq) # this is not needed for freq analysis
            #     study.GetCircuit().GetComponent(u"CS%d"%(poles)).SetValue(u"PhaseU", phase)
            #     study.GetCircuit().GetComponent(u"CS%d"%(poles)).SetValue(u"CommutatingSequence", 0) # this is essencial for the direction of the field to be consistent with speed: UVW rather than UWV

            #     study.GetCircuit().CreateSubCircuit(u"Star Connection", u"Star Connection %d"%(poles), -8, 12) # è¿™äº›æ•°å­—æŒ‡çš„æ˜¯gridçš„ä¸ªæ•°ï¼Œç¬¬å‡ è¡Œç¬¬å‡ åˆ—çš„æ ¼ç‚¹å¤„
            #     study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil1").SetValue(u"Turn", turns)
            #     study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil1").SetValue(u"Resistance", Rs)
            #     study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil2").SetValue(u"Turn", turns)
            #     study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil2").SetValue(u"Resistance", Rs)
            #     study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil3").SetValue(u"Turn", turns)
            #     study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil3").SetValue(u"Resistance", Rs)
            #     study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil1").SetName(u"Coil%dA"%(poles))
            #     study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil2").SetName(u"Coil%dB"%(poles))
            #     study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil3").SetName(u"Coil%dC"%(poles))

            #     # Suspension winding
            #     poles = self.BeariW_poles
            #     turns = self.BeariW_turns
            #     Rs=self.BeariW_Rs
            #     amp=self.BeariW_CurrentAmp
            #     freq=self.BeariW_Freq
            #     phase=0

            #     study.GetCircuit().CreateComponent(u"3PhaseCurrentSource", u"CS%d"%(poles))
            #     study.GetCircuit().CreateInstance(u"CS%d"%(poles),  -31, 1)
            #     study.GetCircuit().GetComponent(u"CS%d"%(poles)).SetValue(u"Amplitude", amp)
            #     study.GetCircuit().GetComponent(u"CS%d"%(poles)).SetValue(u"Frequency", freq) # this is not needed for freq analysis
            #     study.GetCircuit().GetComponent(u"CS%d"%(poles)).SetValue(u"PhaseU", phase)
            #     study.GetCircuit().GetComponent(u"CS%d"%(poles)).SetValue(u"CommutatingSequence", 0) # this is essencial for the direction of the field to be consistent with speed: UVW rather than UWV

            #     study.GetCircuit().CreateComponent(u"Coil", u"Coil10")
            #     study.GetCircuit().CreateInstance(u"Coil10", -18, 15)
            #     study.GetCircuit().GetComponent(u"Coil10").SetName(u"Coil%dA"%(poles))
            #     study.GetCircuit().CreateComponent(u"Coil", u"Coil11")
            #     study.GetCircuit().CreateInstance(u"Coil11", -16, 13)
            #     study.GetCircuit().GetComponent(u"Coil11").SetName(u"Coil%dA"%(poles))
            #     study.GetCircuit().CreateComponent(u"Coil", u"Coil12")
            #     study.GetCircuit().CreateInstance(u"Coil12", -18, 11)
            #     study.GetCircuit().GetComponent(u"Coil12").SetName(u"Coil%dA"%(poles))


            #     # Wires to Connect Circuits 
            #     study.GetCircuit().CreateWire(-10, 15, -16, 15)
            #     study.GetCircuit().CreateWire(-10, 13, -14, 13)
            #     study.GetCircuit().CreateWire(-10, 11, -16, 11)
            #     study.GetCircuit().CreateWire(-10, 15, -10, 19)
            #     study.GetCircuit().CreateWire(-10, 19, -29, 19)
            #     study.GetCircuit().CreateWire(-29, 19, -29, 15)
            #     study.GetCircuit().CreateWire(-29, 13, -27, 13)
            #     study.GetCircuit().CreateWire(-27, 13, -27, 17)
            #     study.GetCircuit().CreateWire(-27, 17, -14, 17)
            #     study.GetCircuit().CreateWire(-14, 17, -14, 13)
            #     study.GetCircuit().CreateWire(-29, 11, -29, 9)
            #     study.GetCircuit().CreateWire(-29, 9, -10, 9)
            #     study.GetCircuit().CreateWire(-10, 9, -10, 11)
            #     study.GetCircuit().CreateWire(-20, 15, -25, 15)
            #     study.GetCircuit().CreateWire(-25, 15, -25, 3)
            #     study.GetCircuit().CreateWire(-25, 3, -29, 3)
            #     study.GetCircuit().CreateWire(-18, 13, -23, 13)
            #     study.GetCircuit().CreateWire(-23, 13, -23, 1)
            #     study.GetCircuit().CreateWire(-23, 1, -29, 1)
            #     study.GetCircuit().CreateWire(-20, 11, -21, 11)
            #     study.GetCircuit().CreateWire(-21, 11, -21, -1)
            #     study.GetCircuit().CreateWire(-21, -1, -29, -1)
            # else:
        def circuit(poles,turns,Rs,amp,freq,phase=0, x=10,y=10):
            study.GetCircuit().CreateSubCircuit(u"Star Connection", u"Star Connection %d"%(poles), x, y) # è¿™äº›æ•°å­—æŒ‡çš„æ˜¯gridçš„ä¸ªæ•°ï¼Œç¬¬å‡ è¡Œç¬¬å‡ åˆ—çš„æ ¼ç‚¹å¤„
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil1").SetValue(u"Turn", turns)
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil1").SetValue(u"Resistance", Rs)
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil2").SetValue(u"Turn", turns)
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil2").SetValue(u"Resistance", Rs)
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil3").SetValue(u"Turn", turns)
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil3").SetValue(u"Resistance", Rs)
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil1").SetName(u"Coil%dA"%(poles))
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil2").SetName(u"Coil%dB"%(poles))
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil3").SetName(u"Coil%dC"%(poles))
            study.GetCircuit().CreateComponent(u"3PhaseCurrentSource", u"CS%d"%(poles))
            study.GetCircuit().CreateInstance(u"CS%d"%(poles), x-4, y+1)
            study.GetCircuit().GetComponent(u"CS%d"%(poles)).SetValue(u"Amplitude", amp)
            study.GetCircuit().GetComponent(u"CS%d"%(poles)).SetValue(u"Frequency", freq) # this is not needed for freq analysis
            study.GetCircuit().GetComponent(u"CS%d"%(poles)).SetValue(u"PhaseU", phase)
            study.GetCircuit().GetComponent(u"CS%d"%(poles)).SetValue(u"CommutatingSequence", 0) # this is essencial for the direction of the field to be consistent with speed: UVW rather than UWV
            study.GetCircuit().CreateComponent(u"Ground", u"Ground")
            study.GetCircuit().CreateInstance(u"Ground", x+2, y+1)
        circuit(self.DriveW_poles, self.DriveW_turns, Rs=self.DriveW_Rs,amp=self.DriveW_CurrentAmp,freq=self.DriveW_Freq,phase=0)
        circuit(self.BeariW_poles, self.BeariW_turns, Rs=self.BeariW_Rs,amp=self.BeariW_CurrentAmp,freq=self.BeariW_Freq,phase=0,x=25)


        # Link FEM Coils to Coil Set 
        def link_FEMCoils_2_CoilSet(poles,l1,l2):
            # link between FEM Coil Condition and Circuit FEM Coil
            for ABC in [u'A',u'B',u'C']:
                which_phase = u"%d%s-Phase"%(poles,ABC)
                study.CreateCondition(u"FEMCoil", which_phase)
                condition = study.GetCondition(which_phase)
                condition.SetLink(u"Coil%d%s"%(poles,ABC))
                condition.GetSubCondition(u"untitled").SetName(u"Coil Set 1")
                condition.GetSubCondition(u"Coil Set 1").SetName(u"delete")
            count = 0
            dict_dir = {'+':1, '-':0, 'o':None}
            # select the part to assign the FEM Coil condition
            for ABC, UpDown in zip(l1,l2):
                count += 1 
                if dict_dir[UpDown] is None:
                    # print 'Skip', ABC, UpDown
                    continue
                which_phase = u"%d%s-Phase"%(poles,ABC)
                condition = study.GetCondition(which_phase)
                condition.CreateSubCondition(u"FEMCoilData", u"Coil Set %d"%(count))
                subcondition = condition.GetSubCondition(u"Coil Set %d"%(count))
                subcondition.ClearParts()
                subcondition.AddSet(model.GetSetList().GetSet(u"Coil%d%s%s %d"%(poles,ABC,UpDown,count)), 0)
                subcondition.SetValue(u"Direction2D", dict_dir[UpDown])
            # clean up
            for ABC in [u'A',u'B',u'C']:
                which_phase = u"%d%s-Phase"%(poles,ABC)
                condition = study.GetCondition(which_phase)
                condition.RemoveSubCondition(u"delete")
        link_FEMCoils_2_CoilSet(self.DriveW_poles, 
                                self.dict_coil_connection[int(self.DriveW_poles*10+1)], # 40 for 4 poles, 1 for ABD, 2 for up or down,
                                self.dict_coil_connection[int(self.DriveW_poles*10+2)])
        link_FEMCoils_2_CoilSet(self.BeariW_poles, 
                                self.dict_coil_connection[int(self.BeariW_poles*10+1)], # 20 for 2 poles.
                                self.dict_coil_connection[int(self.BeariW_poles*10+2)])




        # Condition - Conductor (i.e. rotor winding)
        for ind in range(int(self.Qr)):
            natural_ind = ind + 1
            study.CreateCondition(u"FEMConductor", u"CdctCon %d"%(natural_ind))
            study.GetCondition(u"CdctCon %d"%(natural_ind)).GetSubCondition(u"untitled").SetName(u"Conductor Set 1")
            study.GetCondition(u"CdctCon %d"%(natural_ind)).GetSubCondition(u"Conductor Set 1").ClearParts()
            study.GetCondition(u"CdctCon %d"%(natural_ind)).GetSubCondition(u"Conductor Set 1").AddSet(model.GetSetList().GetSet(u"Bar %d"%(natural_ind)), 0)

        # Condition - Conductor - Grouping
        study.CreateCondition(u"GroupFEMConductor", u"CdctCon_Group")
        for ind in range(int(self.Qr)):
            natural_ind = ind + 1
            study.GetCondition(u"CdctCon_Group").AddSubCondition(u"CdctCon %d"%(natural_ind), ind)

        # Link Conductors to Circuit
        if 'PS' in self.model_name_prefix: # Pole-Specific Rotor Winding
            def place_conductor(x,y,name):
                study.GetCircuit().CreateComponent(u"FEMConductor", name)
                study.GetCircuit().CreateInstance(name, x, y)
            def place_resistor(x,y,name,end_ring_resistance):
                study.GetCircuit().CreateComponent(u"Resistor", name)
                study.GetCircuit().CreateInstance(name, x, y)
                study.GetCircuit().GetComponent(name).SetValue(u"Resistance", end_ring_resistance)

            rotor_phase_name_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            X = 40; Y = 40;
            for i in range(int(self.no_slot_per_pole)):
                Y += -12
                place_conductor(X,   Y, u"Conductor%s1"%(rotor_phase_name_list[i]))
                place_conductor(X, Y-3, u"Conductor%s2"%(rotor_phase_name_list[i]))
                place_conductor(X, Y-6, u"Conductor%s3"%(rotor_phase_name_list[i]))
                place_conductor(X, Y-9, u"Conductor%s4"%(rotor_phase_name_list[i]))

                if self.End_Ring_Resistance == 0: # setting a small value to End_Ring_Resistance is a bad idea (slow down the solver). Instead, don't model it
                    # no end ring resistors to behave like FEMM model
                    study.GetCircuit().CreateWire(X+2,   Y, X+2, Y-3)
                    study.GetCircuit().CreateWire(X-2, Y-3, X-2, Y-6)
                    study.GetCircuit().CreateWire(X+2, Y-6, X+2, Y-9)
                    study.GetCircuit().CreateInstance(u"Ground", X-5, Y-2)
                    study.GetCircuit().CreateWire(X-2,   Y, X-5, Y)
                    study.GetCircuit().CreateWire(X-5,   Y, X-2, Y-9)
                else:
                    place_resistor(X+4,   Y, u"R_%s1"%(rotor_phase_name_list[i]), self.End_Ring_Resistance)
                    place_resistor(X-4, Y-3, u"R_%s2"%(rotor_phase_name_list[i]), self.End_Ring_Resistance)
                    place_resistor(X+4, Y-6, u"R_%s3"%(rotor_phase_name_list[i]), self.End_Ring_Resistance)
                    place_resistor(X-4, Y-9, u"R_%s4"%(rotor_phase_name_list[i]), self.End_Ring_Resistance)
        
                    study.GetCircuit().CreateWire(X+6,   Y, X+2, Y-3)
                    study.GetCircuit().CreateWire(X-6, Y-3, X-2, Y-6)
                    study.GetCircuit().CreateWire(X+6, Y-6, X+2, Y-9)
                    study.GetCircuit().CreateWire(X-6, Y-9, X-7, Y-9)
                    study.GetCircuit().CreateWire(X-2, Y, X-7, Y)
                    study.GetCircuit().CreateInstance(u"Ground", X-7, Y-2)
                        #study.GetCircuit().GetInstance(u"Ground", ini_ground_no+i).RotateTo(90)
                    study.GetCircuit().CreateWire(X-7, Y, X-6, Y-9)

            for i in range(0, int(self.no_slot_per_pole)):
                natural_i = i+1
                study.GetCondition(u"CdctCon %d"%(natural_i)).SetLink(u"Conductor%s1"%(rotor_phase_name_list[i]))
                study.GetCondition(u"CdctCon %d"%(natural_i+self.no_slot_per_pole)).SetLink(u"Conductor%s2"%(rotor_phase_name_list[i]))
                study.GetCondition(u"CdctCon %d"%(natural_i+2*self.no_slot_per_pole)).SetLink(u"Conductor%s3"%(rotor_phase_name_list[i]))
                study.GetCondition(u"CdctCon %d"%(natural_i+3*self.no_slot_per_pole)).SetLink(u"Conductor%s4"%(rotor_phase_name_list[i]))
        else: # Cage
            dyn_circuit = study.GetCircuit().CreateDynamicCircuit(u"Cage")
            dyn_circuit.SetValue(u"AntiPeriodic", False)
            dyn_circuit.SetValue(u"Bars", int(self.Qr))
            dyn_circuit.SetValue(u"EndringResistance", self.End_Ring_Resistance)
            dyn_circuit.SetValue(u"GroupCondition", True)
            dyn_circuit.SetValue(u"GroupName", u"CdctCon_Group")
            dyn_circuit.SetValue(u"UseInductance", False)
            dyn_circuit.Submit(u"Cage1", 23, 2)
            study.GetCircuit().CreateInstance(u"Ground", 25, 1)

        # True: no mesh or field results are needed
        study.GetStudyProperties().SetValue(u"OnlyTableResults", self.fea_config_dict['OnlyTableResults'])

        # Linear Solver
        if False:
            # sometime nonlinear iteration is reported to fail and recommend to increase the accerlation rate of ICCG solver
            study.GetStudyProperties().SetValue(u"IccgAccel", 1.2) 
            study.GetStudyProperties().SetValue(u"AutoAccel", 0)
        else:
            # this can be said to be super fast over ICCG solver.
            # https://www2.jmag-international.com/support/en/pdf/JMAG-Designer_Ver.17.1_ENv3.pdf
            study.GetStudyProperties().SetValue(u"DirectSolverType", 1)

        # This SMP is effective only if there are tons of elements. e.g., over 100,000.
        # too many threads will in turn make them compete with each other and slow down the solve. 2 is good enough for eddy current solve. 6~8 is enough for transient solve.
        study.GetStudyProperties().SetValue(u"UseMultiCPU", True)
        study.GetStudyProperties().SetValue(u"MultiCPU", 2) 

        # # this is for the CAD parameters to rotate the rotor. the order matters for param_no to begin at 0.
        # if self.MODEL_ROTATE:
        #     self.add_cad_parameters(study)

        self.study_name = study_name
        return study

    def add_mesh(self, study, model):
        # this is for multi slide planes, which we will not be using
        refarray = [[0 for i in range(2)] for j in range(1)]
        refarray[0][0] = 3
        refarray[0][1] = 1
        study.GetMeshControl().GetTable("SlideTable2D").SetTable(refarray) 

        study.GetMeshControl().SetValue(u"MeshType", 1) # make sure this has been exe'd: study.GetCondition(u"RotCon").AddSet(model.GetSetList().GetSet(u"Motion_Region"), 0)
        study.GetMeshControl().SetValue(u"RadialDivision", 4) # for air region near which motion occurs
        study.GetMeshControl().SetValue(u"CircumferentialDivision", 720) #1440) # for air region near which motion occurs 这个数足够大，sliding mesh才准确。
        study.GetMeshControl().SetValue(u"AirRegionScale", 1.05) # [Model Length]: Specify a value within the following area. (1.05 <= value < 1000)
        study.GetMeshControl().SetValue(u"MeshSize", 4) # mm
        study.GetMeshControl().SetValue(u"AutoAirMeshSize", 0)
        study.GetMeshControl().SetValue(u"AirMeshSize", 4) # mm
        study.GetMeshControl().SetValue(u"Adaptive", 0)

        study.GetMeshControl().CreateCondition(u"RotationPeriodicMeshAutomatic", u"autoRotMesh") # with this you can choose to set CircumferentialDivision automatically

        study.GetMeshControl().CreateCondition(u"Part", u"CageMeshCtrl")
        study.GetMeshControl().GetCondition(u"CageMeshCtrl").SetValue(u"Size", self.meshSize_Rotor)
        study.GetMeshControl().GetCondition(u"CageMeshCtrl").ClearParts()
        study.GetMeshControl().GetCondition(u"CageMeshCtrl").AddSet(model.GetSetList().GetSet(u"CageSet"), 0)

        study.GetMeshControl().CreateCondition(u"Part", u"ShaftMeshCtrl")
        study.GetMeshControl().GetCondition(u"ShaftMeshCtrl").SetValue(u"Size", 10) # 10 mm
        study.GetMeshControl().GetCondition(u"ShaftMeshCtrl").ClearParts()
        study.GetMeshControl().GetCondition(u"ShaftMeshCtrl").AddSet(model.GetSetList().GetSet(u"ShaftSet"), 0)

        def mesh_all_cases(study):
            numCase = study.GetDesignTable().NumCases()
            for case in range(0, numCase):
                study.SetCurrentCase(case)
                if study.HasMesh() == False:
                    study.CreateMesh()
                # if case == 0:
                #     app.View().ShowAllAirRegions()
                #     app.View().ShowMeshGeometry()
                #     app.View().ShowMesh()
        if self.MODEL_ROTATE:
            if self.total_number_of_cases>1: # just to make sure
                model.RestoreCadLink()
                study.ApplyAllCasesCadParameters()

        mesh_all_cases(study)

    def get_individual_name(self):
        if self.fea_config_dict['flag_optimization'] == True:
            return u"ID%s" % (self.ID)
        else:
            return u"%s_ID%s" % (self.model_name_prefix, self.ID)


    # TranFEAwi2TSS
    def add_TranFEAwi2TSS_study(self, slip_freq_breakdown_torque, app, model, dir_csv_output_folder, tran2tss_study_name, logger):
        im_variant = self
        # logger.debug('Slip frequency: %g = ' % (self.the_slip))
        self.the_slip = slip_freq_breakdown_torque / self.DriveW_Freq
        # logger.debug('Slip frequency:    = %g???' % (self.the_slip))
        study_name = tran2tss_study_name

        model.CreateStudy(u"Transient2D", study_name)
        app.SetCurrentStudy(study_name)
        study = model.GetStudy(study_name)

        # SS-ATA
        study.GetStudyProperties().SetValue(u"ApproximateTransientAnalysis", 1) # psuedo steady state freq is for PWM drive to use
        study.GetStudyProperties().SetValue(u"SpecifySlip", 1)
        study.GetStudyProperties().SetValue(u"Slip", self.the_slip)
        study.GetStudyProperties().SetValue(u"OutputSteadyResultAs1stStep", 0)
        # study.GetStudyProperties().SetValue(u"TimePeriodicType", 2) # This is for TP-EEC but is not effective

        # misc
        study.GetStudyProperties().SetValue(u"ConversionType", 0)
        study.GetStudyProperties().SetValue(u"NonlinearMaxIteration", self.max_nonlinear_iteration)
        study.GetStudyProperties().SetValue(u"ModelThickness", self.stack_length) # Stack Length

        # Material
        if 'M19' in self.fea_config_dict['Steel']:
            study.SetMaterialByName(u"Stator Core", u"M-19 Steel Gauge-29")
            study.GetMaterial(u"Stator Core").SetValue(u"Laminated", 1)
            study.GetMaterial(u"Stator Core").SetValue(u"LaminationFactor", 95)

            study.SetMaterialByName(u"Rotor Core", u"M-19 Steel Gauge-29")
            study.GetMaterial(u"Rotor Core").SetValue(u"Laminated", 1)
            study.GetMaterial(u"Rotor Core").SetValue(u"LaminationFactor", 95)

        elif 'M15' in self.fea_config_dict['Steel']:
            study.SetMaterialByName(u"Stator Core", u"M-15 Steel")
            study.GetMaterial(u"Stator Core").SetValue(u"Laminated", 1)
            study.GetMaterial(u"Stator Core").SetValue(u"LaminationFactor", 98)

            study.SetMaterialByName(u"Rotor Core", u"M-15 Steel")
            study.GetMaterial(u"Rotor Core").SetValue(u"Laminated", 1)
            study.GetMaterial(u"Rotor Core").SetValue(u"LaminationFactor", 98)

        elif self.fea_config_dict['Steel'] == 'Arnon5':
            study.SetMaterialByName(u"Stator Core", u"Arnon5-final")
            study.GetMaterial(u"Stator Core").SetValue(u"Laminated", 1)
            study.GetMaterial(u"Stator Core").SetValue(u"LaminationFactor", 96)

            study.SetMaterialByName(u"Rotor Core", u"Arnon5-final")
            study.GetMaterial(u"Rotor Core").SetValue(u"Laminated", 1)
            study.GetMaterial(u"Rotor Core").SetValue(u"LaminationFactor", 96)

        else:
            msg = 'Warning: default material is used: DCMagnetic Type/50A1000.'
            print msg
            logging.getLogger(__name__).warn(msg)
            study.SetMaterialByName(u"Stator Core", u"DCMagnetic Type/50A1000")
            study.GetMaterial(u"Stator Core").SetValue(u"UserConductivityType", 1)
            study.SetMaterialByName(u"Rotor Core", u"DCMagnetic Type/50A1000")
            study.GetMaterial(u"Rotor Core").SetValue(u"UserConductivityType", 1)

        study.SetMaterialByName(u"Coil", u"Copper")
        study.GetMaterial(u"Coil").SetValue(u"UserConductivityType", 1)

        study.SetMaterialByName(u"Cage", u"Aluminium")
        study.GetMaterial(u"Cage").SetValue(u"EddyCurrentCalculation", 1)
        study.GetMaterial(u"Cage").SetValue(u"UserConductivityType", 1)
        study.GetMaterial(u"Cage").SetValue(u"UserConductivityValue", self.Bar_Conductivity)

        # Conditions - Motion
        study.CreateCondition(u"RotationMotion", u"RotCon") # study.GetCondition(u"RotCon").SetXYZPoint(u"", 0, 0, 1) # megbox warning
        study.GetCondition(u"RotCon").SetValue(u"AngularVelocity", int(self.the_speed))
        study.GetCondition(u"RotCon").ClearParts()
        study.GetCondition(u"RotCon").AddSet(model.GetSetList().GetSet(u"Motion_Region"), 0)

        study.CreateCondition(u"Torque", u"TorCon") # study.GetCondition(u"TorCon").SetXYZPoint(u"", 0, 0, 0) # megbox warning
        study.GetCondition(u"TorCon").SetValue(u"TargetType", 1)
        study.GetCondition(u"TorCon").SetLinkWithType(u"LinkedMotion", u"RotCon")
        study.GetCondition(u"TorCon").ClearParts()

        study.CreateCondition(u"Force", u"ForCon")
        study.GetCondition(u"ForCon").SetValue(u"TargetType", 1)
        study.GetCondition(u"ForCon").SetLinkWithType(u"LinkedMotion", u"RotCon")
        study.GetCondition(u"ForCon").ClearParts()


        # Conditions - FEM Coils (i.e. stator winding)
        # Circuit - Current Source
        app.ShowCircuitGrid(True)
        study.CreateCircuit()
        def circuit(poles,turns,Rs,amp,freq,phase=0, x=10,y=10):
            study.GetCircuit().CreateSubCircuit(u"Star Connection", u"Star Connection %d"%(poles), x, y) # è¿™äº›æ•°å­—æŒ‡çš„æ˜¯gridçš„ä¸ªæ•°ï¼Œç¬¬å‡ è¡Œç¬¬å‡ åˆ—çš„æ ¼ç‚¹å¤„
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil1").SetValue(u"Turn", turns)
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil1").SetValue(u"Resistance", Rs)
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil2").SetValue(u"Turn", turns)
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil2").SetValue(u"Resistance", Rs)
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil3").SetValue(u"Turn", turns)
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil3").SetValue(u"Resistance", Rs)
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil1").SetName(u"Coil%dA"%(poles))
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil2").SetName(u"Coil%dB"%(poles))
            study.GetCircuit().GetSubCircuit(u"Star Connection %d"%(poles)).GetComponent(u"Coil3").SetName(u"Coil%dC"%(poles))
            study.GetCircuit().CreateComponent(u"3PhaseCurrentSource", u"CS%d"%(poles))
            study.GetCircuit().CreateInstance(u"CS%d"%(poles), x-4, y+1)
            study.GetCircuit().GetComponent(u"CS%d"%(poles)).SetValue(u"Amplitude", amp)
            study.GetCircuit().GetComponent(u"CS%d"%(poles)).SetValue(u"Frequency", freq) # this is not needed for freq analysis
            study.GetCircuit().GetComponent(u"CS%d"%(poles)).SetValue(u"PhaseU", phase)
            study.GetCircuit().GetComponent(u"CS%d"%(poles)).SetValue(u"CommutatingSequence", 0) # this is essencial for the direction of the field to be consistent with speed: UVW rather than UWV
            study.GetCircuit().CreateComponent(u"Ground", u"Ground")
            study.GetCircuit().CreateInstance(u"Ground", x+2, y+1)
        circuit(self.DriveW_poles, self.DriveW_turns, Rs=self.DriveW_Rs,amp=self.DriveW_CurrentAmp,freq=self.DriveW_Freq,phase=0)
        circuit(self.BeariW_poles, self.BeariW_turns, Rs=self.BeariW_Rs,amp=self.BeariW_CurrentAmp,freq=self.BeariW_Freq,phase=0,x=25)

        # Link FEM Coils to Coil Set 
        def link_FEMCoils_2_CoilSet(poles,l1,l2):
            # link between FEM Coil Condition and Circuit FEM Coil
            for ABC in [u'A',u'B',u'C']:
                which_phase = u"%d%s-Phase"%(poles,ABC)
                study.CreateCondition(u"FEMCoil", which_phase)
                condition = study.GetCondition(which_phase)
                condition.SetLink(u"Coil%d%s"%(poles,ABC))
                condition.GetSubCondition(u"untitled").SetName(u"Coil Set 1")
                condition.GetSubCondition(u"Coil Set 1").SetName(u"delete")
            count = 0
            dict_dir = {'+':1, '-':0, 'o':None}
            # select the part to assign the FEM Coil condition
            for ABC, UpDown in zip(l1,l2):
                count += 1 
                if dict_dir[UpDown] is None:
                    # print 'Skip', ABC, UpDown
                    continue
                which_phase = u"%d%s-Phase"%(poles,ABC)
                condition = study.GetCondition(which_phase)
                condition.CreateSubCondition(u"FEMCoilData", u"Coil Set %d"%(count))
                subcondition = condition.GetSubCondition(u"Coil Set %d"%(count))
                subcondition.ClearParts()
                subcondition.AddSet(model.GetSetList().GetSet(u"Coil%d%s%s %d"%(poles,ABC,UpDown,count)), 0)
                subcondition.SetValue(u"Direction2D", dict_dir[UpDown])
            # clean up
            for ABC in [u'A',u'B',u'C']:
                which_phase = u"%d%s-Phase"%(poles,ABC)
                condition = study.GetCondition(which_phase)
                condition.RemoveSubCondition(u"delete")
        link_FEMCoils_2_CoilSet(self.DriveW_poles, 
                                self.dict_coil_connection[int(self.DriveW_poles*10+1)], # 40 for 4 poles, 1 for ABD, 2 for up or down,
                                self.dict_coil_connection[int(self.DriveW_poles*10+2)])
        link_FEMCoils_2_CoilSet(self.BeariW_poles, 
                                self.dict_coil_connection[int(self.BeariW_poles*10+1)], # 20 for 2 poles.
                                self.dict_coil_connection[int(self.BeariW_poles*10+2)])


        # Condition - Conductor (i.e. rotor winding)
        for ind in range(int(self.Qr)):
            natural_ind = ind + 1
            study.CreateCondition(u"FEMConductor", u"CdctCon %d"%(natural_ind))
            study.GetCondition(u"CdctCon %d"%(natural_ind)).GetSubCondition(u"untitled").SetName(u"Conductor Set 1")
            study.GetCondition(u"CdctCon %d"%(natural_ind)).GetSubCondition(u"Conductor Set 1").ClearParts()
            study.GetCondition(u"CdctCon %d"%(natural_ind)).GetSubCondition(u"Conductor Set 1").AddSet(model.GetSetList().GetSet(u"Bar %d"%(natural_ind)), 0)

        # Condition - Conductor - Grouping
        study.CreateCondition(u"GroupFEMConductor", u"CdctCon_Group")
        for ind in range(int(self.Qr)):
            natural_ind = ind + 1
            study.GetCondition(u"CdctCon_Group").AddSubCondition(u"CdctCon %d"%(natural_ind), ind)

        # Link Conductors to Circuit
        if 'PS' in self.model_name_prefix: # Pole-Specific Rotor Winding
            def place_conductor(x,y,name):
                study.GetCircuit().CreateComponent(u"FEMConductor", name)
                study.GetCircuit().CreateInstance(name, x, y)
            def place_resistor(x,y,name,end_ring_resistance):
                study.GetCircuit().CreateComponent(u"Resistor", name)
                study.GetCircuit().CreateInstance(name, x, y)
                study.GetCircuit().GetComponent(name).SetValue(u"Resistance", end_ring_resistance)

            rotor_phase_name_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            X = 40; Y = 40;
            for i in range(int(self.no_slot_per_pole)):
                Y += -12
                place_conductor(X,   Y, u"Conductor%s1"%(rotor_phase_name_list[i]))
                place_conductor(X, Y-3, u"Conductor%s2"%(rotor_phase_name_list[i]))
                place_conductor(X, Y-6, u"Conductor%s3"%(rotor_phase_name_list[i]))
                place_conductor(X, Y-9, u"Conductor%s4"%(rotor_phase_name_list[i]))

                if self.End_Ring_Resistance == 0: # setting a small value to End_Ring_Resistance is a bad idea (slow down the solver). Instead, don't model it
                    # no end ring resistors to behave like FEMM model
                    study.GetCircuit().CreateWire(X+2,   Y, X+2, Y-3)
                    study.GetCircuit().CreateWire(X-2, Y-3, X-2, Y-6)
                    study.GetCircuit().CreateWire(X+2, Y-6, X+2, Y-9)
                    study.GetCircuit().CreateInstance(u"Ground", X-5, Y-2)
                    study.GetCircuit().CreateWire(X-2,   Y, X-5, Y)
                    study.GetCircuit().CreateWire(X-5,   Y, X-2, Y-9)
                else:
                    place_resistor(X+4,   Y, u"R_%s1"%(rotor_phase_name_list[i]), self.End_Ring_Resistance)
                    place_resistor(X-4, Y-3, u"R_%s2"%(rotor_phase_name_list[i]), self.End_Ring_Resistance)
                    place_resistor(X+4, Y-6, u"R_%s3"%(rotor_phase_name_list[i]), self.End_Ring_Resistance)
                    place_resistor(X-4, Y-9, u"R_%s4"%(rotor_phase_name_list[i]), self.End_Ring_Resistance)
        
                    study.GetCircuit().CreateWire(X+6,   Y, X+2, Y-3)
                    study.GetCircuit().CreateWire(X-6, Y-3, X-2, Y-6)
                    study.GetCircuit().CreateWire(X+6, Y-6, X+2, Y-9)
                    study.GetCircuit().CreateWire(X-6, Y-9, X-7, Y-9)
                    study.GetCircuit().CreateWire(X-2, Y, X-7, Y)
                    study.GetCircuit().CreateInstance(u"Ground", X-7, Y-2)
                        #study.GetCircuit().GetInstance(u"Ground", ini_ground_no+i).RotateTo(90)
                    study.GetCircuit().CreateWire(X-7, Y, X-6, Y-9)

            for i in range(0, int(self.no_slot_per_pole)):
                natural_i = i+1
                study.GetCondition(u"CdctCon %d"%(natural_i)).SetLink(u"Conductor%s1"%(rotor_phase_name_list[i]))
                study.GetCondition(u"CdctCon %d"%(natural_i+self.no_slot_per_pole)).SetLink(u"Conductor%s2"%(rotor_phase_name_list[i]))
                study.GetCondition(u"CdctCon %d"%(natural_i+2*self.no_slot_per_pole)).SetLink(u"Conductor%s3"%(rotor_phase_name_list[i]))
                study.GetCondition(u"CdctCon %d"%(natural_i+3*self.no_slot_per_pole)).SetLink(u"Conductor%s4"%(rotor_phase_name_list[i]))
        else: # Cage
            dyn_circuit = study.GetCircuit().CreateDynamicCircuit(u"Cage")
            dyn_circuit.SetValue(u"AntiPeriodic", False)
            dyn_circuit.SetValue(u"Bars", int(self.Qr))
            dyn_circuit.SetValue(u"EndringResistance", self.End_Ring_Resistance)
            dyn_circuit.SetValue(u"GroupCondition", True)
            dyn_circuit.SetValue(u"GroupName", u"CdctCon_Group")
            dyn_circuit.SetValue(u"UseInductance", False)
            dyn_circuit.Submit(u"Cage1", 23, 2)
            study.GetCircuit().CreateInstance(u"Ground", 25, 1)

        # True: no mesh or field results are needed
        study.GetStudyProperties().SetValue(u"OnlyTableResults", self.fea_config_dict['OnlyTableResults'])

        # Linear Solver
        if False:
            # sometime nonlinear iteration is reported to fail and recommend to increase the accerlation rate of ICCG solver
            study.GetStudyProperties().SetValue(u"IccgAccel", 1.2) 
            study.GetStudyProperties().SetValue(u"AutoAccel", 0)
        else:
            # this can be said to be super fast over ICCG solver.
            # https://www2.jmag-international.com/support/en/pdf/JMAG-Designer_Ver.17.1_ENv3.pdf
            study.GetStudyProperties().SetValue(u"DirectSolverType", 1)

        # This SMP(shared memory process) is effective only if there are tons of elements. e.g., over 100,000.
        # too many threads will in turn make them compete with each other and slow down the solve. 2 is good enough for eddy current solve. 6~8 is enough for transient solve.
        study.GetStudyProperties().SetValue(u"UseMultiCPU", True)
        study.GetStudyProperties().SetValue(u"MultiCPU", 2) 

        # # this is for the CAD parameters to rotate the rotor. the order matters for param_no to begin at 0.
        # if self.MODEL_ROTATE:
        #     self.add_cad_parameters(study)


        # 上一步的铁磁材料的状态作为下一步的初值，挺好，但是如果每一个转子的位置转过很大的话，反而会减慢非线性迭代。
        # 我们的情况是：0.33 sec 分成了32步，每步的时间大概在0.01秒，0.01秒乘以0.5*497 Hz = 2.485 revolution...
        # study.GetStudyProperties().SetValue(u"NonlinearSpeedup", 0) # JMAG17.1以后默认使用。现在后面密集的步长还多一点（32步），前面16步慢一点就慢一点呗！


        # two sections of different time step
        if True: # ECCE19
            number_of_steps_2ndTTS = self.fea_config_dict['number_of_steps_2ndTTS'] 
            DM = app.GetDataManager()
            DM.CreatePointArray(u"point_array/timevsdivision", u"SectionStepTable")
            refarray = [[0 for i in range(3)] for j in range(3)]
            refarray[0][0] = 0
            refarray[0][1] =    1
            refarray[0][2] =        50
            refarray[1][0] = 0.5/slip_freq_breakdown_torque #0.5 for 17.1.03l # 1 for 17.1.02y
            refarray[1][1] =    16                          # 16 for 17.1.03l #32 for 17.1.02y
            refarray[1][2] =        50
            refarray[2][0] = refarray[1][0] + 0.5/im_variant.DriveW_Freq #0.5 for 17.1.03l 
            refarray[2][1] =    number_of_steps_2ndTTS  # also modify range_ss! # don't forget to modify below!
            refarray[2][2] =        50
            DM.GetDataSet(u"SectionStepTable").SetTable(refarray)
            number_of_total_steps = 1 + 16 + number_of_steps_2ndTTS # [Double Check] don't forget to modify here!
            study.GetStep().SetValue(u"Step", number_of_total_steps)
            study.GetStep().SetValue(u"StepType", 3)
            study.GetStep().SetTableProperty(u"Division", DM.GetDataSet(u"SectionStepTable"))

        else: # IEMDC19
            number_cycles_prolonged = 1 # 50
            DM = app.GetDataManager()
            DM.CreatePointArray(u"point_array/timevsdivision", u"SectionStepTable")
            refarray = [[0 for i in range(3)] for j in range(4)]
            refarray[0][0] = 0
            refarray[0][1] =    1
            refarray[0][2] =        50
            refarray[1][0] = 1.0/slip_freq_breakdown_torque
            refarray[1][1] =    32 
            refarray[1][2] =        50
            refarray[2][0] = refarray[1][0] + 1.0/im_variant.DriveW_Freq
            refarray[2][1] =    48 # don't forget to modify below!
            refarray[2][2] =        50
            refarray[3][0] = refarray[2][0] + number_cycles_prolonged/im_variant.DriveW_Freq # =50*0.002 sec = 0.1 sec is needed to converge to TranRef
            refarray[3][1] =    number_cycles_prolonged*self.fea_config_dict['TranRef-StepPerCycle'] # =50*40, every 0.002 sec takes 40 steps 
            refarray[3][2] =        50
            DM.GetDataSet(u"SectionStepTable").SetTable(refarray)
            study.GetStep().SetValue(u"Step", 1 + 32 + 48 + number_cycles_prolonged*self.fea_config_dict['TranRef-StepPerCycle']) # [Double Check] don't forget to modify here!
            study.GetStep().SetValue(u"StepType", 3)
            study.GetStep().SetTableProperty(u"Division", DM.GetDataSet(u"SectionStepTable"))

        # add equations
        study.GetDesignTable().AddEquation(u"freq")
        study.GetDesignTable().AddEquation(u"slip")
        study.GetDesignTable().AddEquation(u"speed")
        study.GetDesignTable().GetEquation(u"freq").SetType(0)
        study.GetDesignTable().GetEquation(u"freq").SetExpression("%g"%((im_variant.DriveW_Freq)))
        study.GetDesignTable().GetEquation(u"freq").SetDescription(u"Excitation Frequency")
        study.GetDesignTable().GetEquation(u"slip").SetType(0)
        study.GetDesignTable().GetEquation(u"slip").SetExpression("%g"%(im_variant.the_slip))
        study.GetDesignTable().GetEquation(u"slip").SetDescription(u"Slip [1]")
        study.GetDesignTable().GetEquation(u"speed").SetType(1)
        study.GetDesignTable().GetEquation(u"speed").SetExpression(u"freq * (1 - slip) * 30")
        study.GetDesignTable().GetEquation(u"speed").SetDescription(u"mechanical speed of four pole")

        # speed, freq, slip
        study.GetCondition(u"RotCon").SetValue(u"AngularVelocity", u'speed')
        app.ShowCircuitGrid(True)
        study.GetCircuit().GetComponent(u"CS4").SetValue(u"Frequency", u"freq")
        study.GetCircuit().GetComponent(u"CS2").SetValue(u"Frequency", u"freq")

        # max_nonlinear_iteration = 50
        # study.GetStudyProperties().SetValue(u"NonlinearMaxIteration", max_nonlinear_iteration)
        study.GetStudyProperties().SetValue(u"ApproximateTransientAnalysis", 1) # psuedo steady state freq is for PWM drive to use
        study.GetStudyProperties().SetValue(u"SpecifySlip", 1)
        study.GetStudyProperties().SetValue(u"OutputSteadyResultAs1stStep", 0)
        study.GetStudyProperties().SetValue(u"Slip", u"slip")

        # # add other excitation frequencies other than 500 Hz as cases
        # for case_no, DriveW_Freq in enumerate([50.0, slip_freq_breakdown_torque]):
        #     slip = slip_freq_breakdown_torque / DriveW_Freq
        #     study.GetDesignTable().AddCase()
        #     study.GetDesignTable().SetValue(case_no+1, 0, DriveW_Freq)
        #     study.GetDesignTable().SetValue(case_no+1, 1, slip)

        # 你把Tran2TSS计算周期减半！
        # 也要在计算铁耗的时候选择1/4或1/2的数据！（建议1/4）
        # 然后，手动添加end step 和 start step，这样靠谱！2019-01-09：注意设置铁耗条件（iron loss condition）的Reference Start Step和End Step。

        # Iron Loss Calculation Condition
        # Stator 
        cond = study.CreateCondition(u"Ironloss", u"IronLossConStator")
        cond.SetValue(u"RevolutionSpeed", u"freq*60/%d"%(0.5*(im_variant.DriveW_poles)))
        cond.ClearParts()
        sel = cond.GetSelection()
        sel.SelectPartByPosition(-im_variant.Radius_OuterStatorYoke+1e-2, 0 ,0)
        cond.AddSelected(sel)
        # Use FFT for hysteresis to be consistent with FEMM's results and to have a FFT plot
        cond.SetValue(u"HysteresisLossCalcType", 1)
        cond.SetValue(u"PresetType", 3) # 3:Custom
        # Specify the reference steps yourself because you don't really know what JMAG is doing behind you
        cond.SetValue(u"StartReferenceStep", number_of_total_steps+1-number_of_steps_2ndTTS*0.5) # 1/4 period <=> number_of_steps_2ndTTS*0.5
        cond.SetValue(u"EndReferenceStep", number_of_total_steps)
        cond.SetValue(u"UseStartReferenceStep", 1)
        cond.SetValue(u"UseEndReferenceStep", 1)
        cond.SetValue(u"Cyclicity", 4) # specify reference steps for 1/4 period and extend it to whole period
        cond.SetValue(u"UseFrequencyOrder", 1)
        cond.SetValue(u"FrequencyOrder", u"1-50") # Harmonics up to 50th orders 
        # Check CSV reults for iron loss (You cannot check this for Freq study) # CSV and save space
        study.GetStudyProperties().SetValue(u"CsvOutputPath", dir_csv_output_folder) # it's folder rather than file!
        study.GetStudyProperties().SetValue(u"CsvResultTypes", u"Torque;Force;LineCurrent;TerminalVoltage;JouleLoss;TotalDisplacementAngle;JouleLoss_IronLoss;IronLoss_IronLoss;HysteresisLoss_IronLoss")
        study.GetStudyProperties().SetValue(u"DeleteResultFiles", self.fea_config_dict['delete_results_after_calculation'])
        # Terminal Voltage/Circuit Voltage: Check for outputing CSV results 
        study.GetCircuit().CreateTerminalLabel(u"Terminal4A", 8, -13)
        study.GetCircuit().CreateTerminalLabel(u"Terminal4B", 8, -11)
        study.GetCircuit().CreateTerminalLabel(u"Terminal4C", 8, -9)
        study.GetCircuit().CreateTerminalLabel(u"Terminal2A", 23, -13)
        study.GetCircuit().CreateTerminalLabel(u"Terminal2B", 23, -11)
        study.GetCircuit().CreateTerminalLabel(u"Terminal2C", 23, -9)
        # Export Stator Core's field results only for iron loss calculation (the csv file of iron loss will be clean with this setting)
            # study.GetMaterial(u"Rotor Core").SetValue(u"OutputResult", 0) # at least one part on the rotor should be output or else a warning "the jplot file does not contains displacement results when you try to calc. iron loss on the moving part." will pop up, even though I don't add iron loss condition on the rotor.
        study.GetMeshControl().SetValue(u"AirRegionOutputResult", 0)
        study.GetMaterial(u"Shaft").SetValue(u"OutputResult", 0)
        study.GetMaterial(u"Cage").SetValue(u"OutputResult", 0)
        study.GetMaterial(u"Coil").SetValue(u"OutputResult", 0)
        # Rotor
            # study.CreateCondition(u"Ironloss", u"IronLossConRotor")
            # study.GetCondition(u"IronLossConRotor").SetValue(u"BasicFrequencyType", 2)
            # study.GetCondition(u"IronLossConRotor").SetValue(u"BasicFrequency", u"slip*freq")
            # study.GetCondition(u"IronLossConRotor").ClearParts()
            # sel = study.GetCondition(u"IronLossConRotor").GetSelection()
            # sel.SelectPartByPosition(-im.Radius_Shaft-1e-2, 0 ,0)
            # study.GetCondition(u"IronLossConRotor").AddSelected(sel)
            # # Use FFT for hysteresis to be consistent with FEMM's results
            # study.GetCondition(u"IronLossConRotor").SetValue(u"HysteresisLossCalcType", 1)
            # study.GetCondition(u"IronLossConRotor").SetValue(u"PresetType", 3)

        self.study_name = study_name
        return study


    # Static FEA
    def add_cad_parameters(self, study):

        ''' CAD Parameters '''
        study.SetCheckForTopologyChanges(False) # 

        # the order matters for param_no to begin at 0.
        study.GetDesignTable().AddEquation(u"RotorPosition")
        study.GetDesignTable().GetEquation(u"RotorPosition").SetType(0) # 0-value. 1-expression which cannot be modified for different cases.
        study.GetDesignTable().GetEquation(u"RotorPosition").SetExpression(u"0.0")
        study.GetDesignTable().GetEquation(u"RotorPosition").SetDescription(u"rotor position")
        # study.GetDesignTable().SetValue(0, 0, 0.0) # case_no, param_no, value

        # app.GetModel(u"EC_Rotate_32").RestoreCadLink()
        def add_vertex_xy_as_cad_param(list_vertex_names, sketch_name):
            for vertex_name in list_vertex_names:
                study.AddCadParameter(u"X@%s@%s"%(vertex_name,sketch_name))
                study.AddCadParameter(u"Y@%s@%s"%(vertex_name,sketch_name))
                study.GetDesignTable().AddCadParameterVariableName(u"X value@%s@%s"%(vertex_name,sketch_name))
                study.GetDesignTable().AddCadParameterVariableName(u"Y value@%s@%s"%(vertex_name,sketch_name))

        #print im.list_rotorCore_vertex_names
        add_vertex_xy_as_cad_param(self.list_rotorCore_vertex_names, 'Rotor Core')
        add_vertex_xy_as_cad_param(self.list_rotorCage_vertex_names, 'Cage')
            # add_vertex_xy_as_cad_param(self.list_rotorAirWithin_vertex_names, 'Air Within Rotor Slots')

        # total_number_of_cad_parameters = len(self.list_rotorCore_vertex_names) + len(self.list_rotorCage_vertex_names)

    def add_cases_rotate_rotor(self, study, total_number_of_cases):
        print 'total_number_of_cases:', total_number_of_cases
        if total_number_of_cases > 1:

            # add case label!
            study.GetDesignTable().AddCases(total_number_of_cases - 1)        

            def rotate_vertex_in_cad_param(theta, case_no_list, param_no):
                for case_no in case_no_list:
                    # print case_no, total_number_of_cases
                    X = study.GetDesignTable().GetValue(0, param_no)
                    Y = study.GetDesignTable().GetValue(0, param_no+1)
                    radian = case_no*theta
                    study.GetDesignTable().SetValue(case_no, begin_here-1, radian/pi*180)
                    # print case_no, radian/pi*180, case_no_list
                    # Park Transformation
                    NEW_X = cos(radian)*X - sin(radian)*Y # 注意转动的方向，要和磁场旋转的方向一致
                    NEW_Y = sin(radian)*X + cos(radian)*Y
                    study.GetDesignTable().SetValue(case_no, param_no, NEW_X)
                    study.GetDesignTable().SetValue(case_no, param_no+1, NEW_Y)

            begin_here = 4 # rotor_position, freq, slip, speed, 
            study.GetDesignTable().SetValue(0, begin_here-1, 0.0) # rotor_position for case#0 is 0.0 # BUG???
            end_here = study.GetDesignTable().NumParameters()
            # print 'This value should be 4+34:', end_here

            # self.theta = 1./180.0*pi

            for param_no in range(begin_here, end_here, 2): # begin at one for the RotorPosition Variable
                rotate_vertex_in_cad_param(self.theta, range(1,total_number_of_cases), param_no) # case_no = 1
            study.ApplyCadParameters()
                    # study.GetDesignTable().SetActive(0, True)

    def add_rotor_current_condition(self, app, model, study, total_number_of_cases, eddy_current_circuit_current_csv_file): # r'D:\Users\horyc\OneDrive - UW-Madison\csv\Freq_#4_circuit_current.csv'

        # study.GetMaterial(u"Cage").SetValue(u"EddyCurrentCalculation", 0)
        rotor_phase_name_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # read from eddy current results
        dict_circuit_current_complex = {}
        with open(eddy_current_circuit_current_csv_file, 'r') as f:
            read_iterator = csv_reader(f, skipinitialspace=True)
            for row in self.whole_row_reader(read_iterator):
                try: 
                    float(row[0])
                except:
                    continue
                else:
                    if np.abs(self.slip_freq_breakdown_torque - float(row[0])) < 1e-3:                    
                        beginning_column = 1 + 2*3*2 # title + drive/bearing * 3 phase * real/imag
                        for i in range(0, int(self.no_slot_per_pole)):
                            natural_i = i+1
                            current_phase_column = beginning_column + i * int(self.DriveW_poles) * 2
                            for j in range(int(self.DriveW_poles)):
                                natural_j = j+1
                                re = float(row[current_phase_column+2*j])
                                im = float(row[current_phase_column+2*j+1])
                                dict_circuit_current_complex["%s%d"%(rotor_phase_name_list[i], natural_j)] = (re, im)
        dict_circuit_current_amp_and_phase = {}
        for key, item in dict_circuit_current_complex.iteritems():
            amp = np.sqrt(item[1]**2 + item[0]**2)
            phase = np.arctan2(item[0], -item[1]) # atan2(y, x), y=a, x=-b
            dict_circuit_current_amp_and_phase[key] = (amp, phase)


        # Link the FEMCoil with the conditions
        begin_parameters = study.GetDesignTable().NumParameters()
        # print 'num param:', begin_parameters
        count_parameters = 0
        for i in range(0, int(self.no_slot_per_pole)):

            # values of CurCon are determined by equations/variables: so buld variables for that
            study.GetDesignTable().AddEquation( 'var' + u"RotorCurCon%s"%(rotor_phase_name_list[i]) )
            study.GetDesignTable().GetEquation( 'var' + u"RotorCurCon%s"%(rotor_phase_name_list[i]) ).SetType(0)
            study.GetDesignTable().GetEquation( 'var' + u"RotorCurCon%s"%(rotor_phase_name_list[i]) ).SetExpression(u"0")
            study.GetDesignTable().GetEquation( 'var' + u"RotorCurCon%s"%(rotor_phase_name_list[i]) ).SetDescription(u"")

            # now, assign different values to these variables w.r.t. different cases 
            # self.theta = 1./180.0*pi
            t = 0.0
            time_one_step = self.theta / (2*pi * self.the_speed) * 60 # sec
            # ScriptComments
            amp, phase = dict_circuit_current_amp_and_phase[rotor_phase_name_list[i]+'1']
            for case_no in range(total_number_of_cases):
                current_value = amp * sin(2*pi*(self.the_slip*self.DriveW_Freq)*t + phase)
                study.GetDesignTable().SetValue(case_no, begin_parameters + count_parameters, current_value)
                t += time_one_step
            count_parameters += 1

            natural_i = i+1
            for index, j in enumerate([ natural_i,
                                        natural_i + self.no_slot_per_pole]):

                # for Static FEA: amp * sin(2*fpi*f*t + phase)
                current_condition = study.CreateCondition(u"Current", u"RotorCurCon%s%d"%(rotor_phase_name_list[i],j))
                current_condition.SetValue(u"Current", 'var' + u"RotorCurCon%s"%(rotor_phase_name_list[i]))
                    # current_condition.SetValue(u"XType", 1)
                if index == 0:
                    current_condition.SetValue(u"Direction2D", 0)
                else:
                    current_condition.SetValue(u"Direction2D", 1) # 结果不正常？可能是转子电流相位差了180度？ Check.

                current_condition.ClearParts()
                current_condition.AddSet(model.GetSetList().GetSet(u"Bar %d"%(j)), 0)
                current_condition.AddSet(model.GetSetList().GetSet(u"Bar %d"%(j+2*self.no_slot_per_pole)), 0)

        # for key, item in dict_circuit_current_complex.iteritems():
        #     print key, dict_circuit_current_complex[key],
        #     print dict_circuit_current_amp_and_phase[key]

    def add_stator_current_condition(self, app, model, study, total_number_of_cases, eddy_current_circuit_current_csv_file):

        # read from eddy current results
        dict_circuit_current_complex = {}
        with open(eddy_current_circuit_current_csv_file, 'r') as f:
            read_iterator = csv_reader(f, skipinitialspace=True)
            for row in self.whole_row_reader(read_iterator):
                try: 
                    float(row[0])
                except:
                    continue
                else:
                    if '%g'%(self.slip_freq_breakdown_torque) in row[0]:
                        beginning_column = 1 # title column is not needed
                        for i, phase in zip(range(0,12,2), ['2A','2B','2C','4A','4B','4C']): # 3 phase
                            natural_i = i+1
                            current_phase_column = beginning_column + i 
                            re = float(row[current_phase_column])
                            im = float(row[current_phase_column+1])
                            dict_circuit_current_complex[phase] = (re, im)
        dict_circuit_current_amp_and_phase = {}
        for key, item in dict_circuit_current_complex.iteritems():
            amp = np.sqrt(item[1]**2 + item[0]**2)
            phase = np.arctan2(item[0], -item[1]) # atan2(y, x), y=a, x=-b
            dict_circuit_current_amp_and_phase[key] = (amp, phase)

        # for key, item in dict_circuit_current_complex.iteritems():
        #     print key, dict_circuit_current_complex[key],
        #     print dict_circuit_current_amp_and_phase[key]


        ''' Create Variables for being used for different cases '''
        begin_parameters = study.GetDesignTable().NumParameters()
        count_parameters = 0
        print 'num param begins at:', begin_parameters
        for phase_name in ['2A','2B','2C','4A','4B','4C']:
            # ScriptComments
            amp, phase = dict_circuit_current_amp_and_phase[phase_name]

            # values of CurCon are determined by equations/variables: so buld variables for that
            study.GetDesignTable().AddEquation( 'var' + phase_name )
            study.GetDesignTable().GetEquation( 'var' + phase_name ).SetType(0)
            study.GetDesignTable().GetEquation( 'var' + phase_name ).SetExpression(u"0")

            # now, assign different values to these variables w.r.t. different cases 
            # self.theta = 1./180.0*pi
            t = 0.0
            time_one_step = self.theta / (2*pi * self.the_speed) * 60 # sec
            for case_no in range(total_number_of_cases):
                current_value = amp * sin(2*pi*self.DriveW_Freq*t + phase)
                study.GetDesignTable().SetValue(case_no, begin_parameters + count_parameters, current_value)
                t += time_one_step 
            count_parameters += 1
        print 'num param ends at:', study.GetDesignTable().NumParameters()



        def create_stator_current_conditions(turns,condition_name_list):

            set_list = model.GetSetList()

            for condition_name in condition_name_list:
                condition = study.CreateCondition(u"Current", condition_name)
                condition.SetValue(u"Turn", turns)
                condition.SetValue(u"Current", 'var' + condition_name[4:6])
                if '+' in condition_name:
                    condition.SetValue(u"Direction2D", 1) # 1=x
                elif '-' in condition_name:
                    condition.SetValue(u"Direction2D", 0) # 0=.
                else:
                    raise Exception('cannot find + or i in current condition name')

                condition.ClearParts()
                set_name_list = [set_list.GetSet(i).GetName() for i in range(set_list.NumSet()) if condition_name[4:7] in set_list.GetSet(i).GetName()]
                print set_name_list
                for set_name in set_name_list:
                    condition.AddSet(model.GetSetList().GetSet(set_name), 0) # 0: group
        create_stator_current_conditions(self.DriveW_turns, [u"Coil4A-",u"Coil4B-",u"Coil4C-",u"Coil4A+",u"Coil4B+",u"Coil4C+"])
        create_stator_current_conditions(self.DriveW_turns, [u"Coil2A-",u"Coil2B-",u"Coil2C-",u"Coil2A+",u"Coil2B+",u"Coil2C+"])

    def add_rotor_current_condition_obsolete_slow_version(self, app, model, study, total_number_of_cases, eddy_current_circuit_current_csv_file): # r'D:\Users\horyc\OneDrive - UW-Madison\csv\Freq_#4_circuit_current.csv'

        # study.GetMaterial(u"Cage").SetValue(u"EddyCurrentCalculation", 0)
        rotor_phase_name_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # read from eddy current results
        dict_circuit_current_complex = {}
        with open(eddy_current_circuit_current_csv_file, 'r') as f:
            read_iterator = csv_reader(f, skipinitialspace=True)
            for row in self.whole_row_reader(read_iterator):
                try: 
                    float(row[0])
                except:
                    continue
                else:
                    if '%g'%(self.slip_freq_breakdown_torque) in row[0]:
                        beginning_column = 1 + 2*3*2 # title + drive/bearing * 3 phase * real/imag
                        for i in range(0, int(self.no_slot_per_pole)):
                            natural_i = i+1
                            current_phase_column = beginning_column + i * int(self.DriveW_poles) * 2
                            for j in range(int(self.DriveW_poles)):
                                natural_j = j+1
                                re = float(row[current_phase_column+2*j])
                                im = float(row[current_phase_column+2*j+1])
                                dict_circuit_current_complex["%s%d"%(rotor_phase_name_list[i], natural_j)] = (re, im)
        dict_circuit_current_amp_and_phase = {}
        for key, item in dict_circuit_current_complex.iteritems():
            amp = np.sqrt(item[1]**2 + item[0]**2)
            phase = np.arctan2(item[0], -item[1]) # atan2(y, x), y=a, x=-b
            dict_circuit_current_amp_and_phase[key] = (amp, phase)


        # Link the FEMCoil with the conditions
        begin_parameters = study.GetDesignTable().NumParameters()
        # print 'num param:', begin_parameters
        count_parameters = 0
        for i in range(0, int(self.no_slot_per_pole)):
            natural_i = i+1
            for index0123, j in enumerate([  natural_i,
                                natural_i +   self.no_slot_per_pole,
                                natural_i + 2*self.no_slot_per_pole,
                                natural_i + 3*self.no_slot_per_pole]):
                # study.GetCondition(u"BarCoilCon %d"%(natural_j)).SetLink(u"ConductorCircuit %d"%(j))

                # ScriptComments
                amp, phase = dict_circuit_current_amp_and_phase["%s%d"%(rotor_phase_name_list[i], index0123+1)]

                # values of CurCon are determined by equations/variables: so buld variables for that
                study.GetDesignTable().AddEquation( 'var' + u"RotorCurCon%s%d"%(rotor_phase_name_list[i],j) )
                study.GetDesignTable().GetEquation( 'var' + u"RotorCurCon%s%d"%(rotor_phase_name_list[i],j) ).SetType(0)
                study.GetDesignTable().GetEquation( 'var' + u"RotorCurCon%s%d"%(rotor_phase_name_list[i],j) ).SetExpression(u"0")
                study.GetDesignTable().GetEquation( 'var' + u"RotorCurCon%s%d"%(rotor_phase_name_list[i],j) ).SetDescription(u"")

                # for Static FEA: amp * sin(2*fpi*f*t + phase)
                current_condition = study.CreateCondition(u"Current", u"RotorCurCon%s%d"%(rotor_phase_name_list[i],j))
                current_condition.SetValue(u"XType", 1)
                current_condition.SetValue(u"Current", 'var' + u"RotorCurCon%s%d"%(rotor_phase_name_list[i],j))
                current_condition.ClearParts()
                current_condition.AddSet(model.GetSetList().GetSet(u"Bar %d"%(j)), 0)

                # now, assign different values to these variables w.r.t. different cases 
                # self.theta = 1./180.0*pi
                t = 0.0
                time_one_step = self.theta / (2*pi * self.the_speed) * 60 # sec
                for case_no in range(total_number_of_cases):
                    t += time_one_step
                    current_value = amp * sin(2*pi*(self.the_slip*self.DriveW_Freq)*t + phase)
                    study.GetDesignTable().SetValue(case_no, begin_parameters + count_parameters, current_value)
                count_parameters += 1


        # for key, item in dict_circuit_current_complex.iteritems():
        #     print key, dict_circuit_current_complex[key],
        #     print dict_circuit_current_amp_and_phase[key]


class VanGogh_JMAG(VanGogh):
    def __init__(self, im, child_index=1):
        super(VanGogh_JMAG, self).__init__(im, child_index)

        self.SketchName = None
        self.dict_count_arc = {}
        self.dict_count_region = {}

        self.count = 0

    def mirror_and_copyrotate(self, Q, Radius, fraction, 
                                edge4ref=None,
                                symmetry_type=None,
                                merge=True,
                                do_you_have_region_in_the_mirror=False
                                ):

        region = self.create_region([art.GetName() for art in self.artist_list]) 

        self.region_mirror_copy(region, edge4ref=edge4ref, symmetry_type=symmetry_type, merge=merge)
        self.count+=1
        # if self.count == 4: # debug
            # raise Exception
            # merge = True # When overlap occurs between regions because of copying, a boolean operation (sum) is executed and they are merged into 1 region.
        self.region_circular_pattern_360_origin(region, float(Q), merge=merge,
                                                do_you_have_region_in_the_mirror=do_you_have_region_in_the_mirror)
        # print self.artist_list
        self.sketch.CloseSketch()

    def draw_arc(self, p1, p2, angle, maxseg=1): # angle in rad
        center = self.find_center_of_a_circle_using_2_points_and_arc_angle(p1, p2, angle) # ordered p1 and p2 are
        art = self.sketch.CreateArc(center[0], center[1], p1[0], p1[1], p2[0], p2[1])
        self.artist_list.append(art)
    
    def add_arc(self, p1, p2, angle, maxseg=1): # angle in rad
        center = self.find_center_of_a_circle_using_2_points_and_arc_angle(p1, p2, angle) # ordered p1 and p2 are
        art = self.sketch.CreateArc(center[0], center[1], p1[0], p1[1], p2[0], p2[1])
        self.artist_list.append(art)

    def draw_line(self, p1, p2):
        # return self.line(p1[0],p1[1],p2[0],p2[1])
        art = self.sketch.CreateLine(p1[0],p1[1],p2[0],p2[1])
        self.artist_list.append(art)

    def add_line(self, p1, p2):
        # return self.line(p1[0],p1[1],p2[0],p2[1])
        art = self.sketch.CreateLine(p1[0],p1[1],p2[0],p2[1])
        self.artist_list.append(art)

    def plot_sketch_shaft(self):
        self.SketchName = u"Shaft"
        SketchName = self.SketchName
        sketch = self.create_sketch(SketchName, u"#D1B894")

        self.circle(0, 0, self.im.Radius_Shaft)

        self.doc.GetSelection().Clear()
        self.doc.GetSelection().Add(sketch.GetItem(u"Circle"))
        sketch.CreateRegions()

        sketch.CloseSketch()
        # sketch.SetProperty(u"Visible", 0)

    def init_sketch_statorCore(self):
        self.SketchName=u"Stator Core"
        sketch = self.create_sketch(self.SketchName, u"#E8B5CE")
        return 

    def init_sketch_coil(self):
        self.SketchName=u"Coil"
        sketch = self.create_sketch(self.SketchName, u"#EC9787")
        return

    def init_sketch_rotorCore(self):
        self.SketchName=u"Rotor Core"
        sketch = self.create_sketch(self.SketchName, u"#FE840E")
        return

    def init_sketch_cage(self):
        self.SketchName=u"Cage"
        sketch = self.create_sketch(self.SketchName, u"#8D9440")
        return



    # Utility wrap function for JMAG
    def create_sketch(self, SketchName, color):
        self.artist_list = []

        try:self.dict_count_arc[SketchName]
        except: self.dict_count_arc[SketchName] = 0
        try:self.dict_count_region[SketchName]
        except: self.dict_count_region[SketchName] = 0
        ref1 = self.ass.GetItem(u"XY Plane")
        ref2 = self.doc.CreateReferenceFromItem(ref1)
        self.sketch = self.ass.CreateSketch(ref2)
        self.sketch.OpenSketch()
        self.sketch.SetProperty(u"Name", SketchName)
        self.sketch.SetProperty(u"Color", color)
        return self.sketch
    def circle(self, x,y,r):
        # SketchName = self.SketchName
        self.sketch.CreateVertex(x, y)
        # return self.circle(x, y, r)
        return self.sketch.CreateCircle(x, y, r)
    def line(self, x1,y1,x2,y2):
        # SketchName = self.SketchName
        self.sketch.CreateVertex(x1,y1)
        self.sketch.CreateVertex(x2,y2)
        # return self.line(x1,y1,x2,y2)
        return self.sketch.CreateLine(x1,y1,x2,y2)
    def create_region(self, l):
        SketchName = self.SketchName
        self.doc.GetSelection().Clear()
        for art_name in l:
            self.doc.GetSelection().Add(self.sketch.GetItem(art_name))
            # self.doc.GetSelection().Add(el)
        self.sketch.CreateRegions() # this returns None
        # self.sketch.CreateRegionsWithCleanup(0.05, True) # mm. difference at stator outter radius is up to 0.09mm! This turns out to be neccessary for shapely to work with JMAG. Shapely has poor 

        self.dict_count_region[SketchName] += 1
        if self.dict_count_region[SketchName]==1:
            return self.sketch.GetItem(u"Region")
        else:
            return self.sketch.GetItem(u"Region.%d"%(self.dict_count_region[SketchName]))
    def region_mirror_copy(self, region, edge4ref=None, symmetry_type=None, merge=True):
        mirror = self.sketch.CreateRegionMirrorCopy()
        mirror.SetProperty(u"Merge", merge)
        ref2 = self.doc.CreateReferenceFromItem(region)
        mirror.SetPropertyByReference(u"Region", ref2)

        # å¯¹ç§°è½´
        if edge4ref == None:
            if symmetry_type == None:
                print "At least give one of edge4ref and symmetry_type"
                raise Exception
            else:
                mirror.SetProperty(u"SymmetryType", symmetry_type)
        else:
            ref1 = self.sketch.GetItem(edge4ref.GetName()) # e.g., u"Line"
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            mirror.SetPropertyByReference(u"Symmetry", ref2)

        # print region
        # print ass.GetItem(u"Region.1")
        if merge == False and region.GetName()==u"Region":
            return self.ass.GetItem(u"Region.1")
    def region_circular_pattern_360_origin(self, region, Q_float, merge=True, do_you_have_region_in_the_mirror=False):
        circular_pattern = self.sketch.CreateRegionCircularPattern()
        circular_pattern.SetProperty(u"Merge", merge)

        ref2 = self.doc.CreateReferenceFromItem(region)
        circular_pattern.SetPropertyByReference(u"Region", ref2)
        face_region_string = circular_pattern.GetProperty("Region")
        face_region_string = face_region_string[0]
        # print circular_pattern.GetProperty("Region") # this will produce faceRegion references!

        if do_you_have_region_in_the_mirror == True:
            # è¿™é‡Œå‡è®¾face_region_stringæœ€åŽä¸¤ä½æ˜¯æ•°å­—
            if face_region_string[-7:-3] == 'Item':
                number_plus_1 = str(int(face_region_string[-3:-1]) + 1)
                refarray = [0 for i in range(2)]
                refarray[0] = u"faceregion(TRegionMirrorPattern%s+%s_2)" % (number_plus_1, face_region_string)
                refarray[1] = face_region_string
                circular_pattern.SetProperty(u"Region", refarray)
                # print refarray[0]
                # print refarray[1]
            elif face_region_string[-6:-2] == 'Item':
                # è¿™é‡Œå‡è®¾face_region_stringæœ€åŽä¸€ä½æ˜¯æ•°å­—
                number_plus_1 = str(int(face_region_string[-2:-1]) + 1)
                refarray = [0 for i in range(2)]
                refarray[0] = u"faceregion(TRegionMirrorPattern%s+%s_2)" % (number_plus_1, face_region_string)
                refarray[1] = face_region_string
                circular_pattern.SetProperty(u"Region", refarray)
            elif face_region_string[-8:-4] == 'Item':
                # è¿™é‡Œå‡è®¾face_region_stringæœ€åŽä¸‰ä½æ˜¯æ•°å­—
                number_plus_1 = str(int(face_region_string[-4:-1]) + 1)
                refarray = [0 for i in range(2)]
                refarray[0] = u"faceregion(TRegionMirrorPattern%s+%s_2)" % (number_plus_1, face_region_string)
                refarray[1] = face_region_string
                circular_pattern.SetProperty(u"Region", refarray)



        circular_pattern.SetProperty(u"CenterType", 2)
        circular_pattern.SetProperty(u"Angle", u"360/%d"%(int(Q_float)))
        circular_pattern.SetProperty(u"Instance", int(Q_float))

# if __name__ == '__main__':
#     vg_jmag = VanGogh_JMAG(None)
#     print vg_jmag.find_center_of_a_circle_using_2_points_and_arc_angle

class TrimDrawer(object):

    def __init__(self, im):
        self.SketchName = None
        self.trim_a = self.trim_l
        self.dict_count_arc = {}
        self.dict_count_region = {}

        self.im = im

    ''' Basic Functions '''
    def create_sketch(self, SketchName, color):
        try:self.dict_count_arc[SketchName]
        except: self.dict_count_arc[SketchName] = 0
        try:self.dict_count_region[SketchName]
        except: self.dict_count_region[SketchName] = 0
        ref1 = self.ass.GetItem(u"XY Plane")
        ref2 = self.doc.CreateReferenceFromItem(ref1)
        self.sketch = self.ass.CreateSketch(ref2)
        self.sketch.OpenSketch()
        self.sketch.SetProperty(u"Name", SketchName)
        self.sketch.SetProperty(u"Color", color)
        return self.sketch

    def circle(self, x,y,r):
        # SketchName = self.SketchName
        # self.sketch.CreateVertex(x, y)
        # return self.circle(x, y, r)
        return self.sketch.CreateCircle(x, y, r)

    def line(self, x1,y1,x2,y2):
        # SketchName = self.SketchName
        # self.sketch.CreateVertex(x1,y1)
        # self.sketch.CreateVertex(x2,y2)
        # return self.line(x1,y1,x2,y2)
        return self.sketch.CreateLine(x1,y1,x2,y2)

    def trim_l(self, who,x,y):
        # SketchName = self.SketchName
        self.doc.GetSelection().Clear()
        ref1 = self.sketch.GetItem(who.GetName())
        self.doc.GetSelection().Add(ref1)
        self.doc.GetSketchManager().SketchTrim(x,y)
        # l1 trim 完以后还是l1，除非你切中间，这样会多生成一个Line，你自己捕捉一下吧


    def trim_c(self, who,x,y):
        SketchName = self.SketchName
        self.doc.GetSelection().Clear()
        ref1 = self.sketch.GetItem(who.GetName())
        self.doc.GetSelection().Add(ref1)
        self.doc.GetSketchManager().SketchTrim(x,y)

        # print who 
        self.dict_count_arc[SketchName] += 1
        if self.dict_count_arc[SketchName]==1:
            return self.sketch.GetItem(u"Arc")
        else:
            return self.sketch.GetItem(u"Arc.%d"%(self.dict_count_arc[SketchName]))

    def create_region(self, l):
        SketchName = self.SketchName
        self.doc.GetSelection().Clear()
        for string in l:
            self.doc.GetSelection().Add(self.sketch.GetItem(string))
            # self.doc.GetSelection().Add(el)
        self.sketch.CreateRegions() # this returns None

        self.dict_count_region[SketchName] += 1
        if self.dict_count_region[SketchName]==1:
            return self.sketch.GetItem(u"Region")
        else:
            return self.sketch.GetItem(u"Region.%d"%(self.dict_count_region[SketchName]))

    def region_mirror_copy(self, region, edge4ref=None, symmetry_type=None, merge=True):
        mirror = self.sketch.CreateRegionMirrorCopy()
        mirror.SetProperty(u"Merge", merge)
        ref2 = self.doc.CreateReferenceFromItem(region)
        mirror.SetPropertyByReference(u"Region", ref2)

        # å¯¹ç§°è½´
        if edge4ref == None:
            if symmetry_type == None:
                print "At least give one of edge4ref and symmetry_type"
            else:
                mirror.SetProperty(u"SymmetryType", symmetry_type)
        else:
            ref1 = self.sketch.GetItem(edge4ref.GetName()) # e.g., u"Line"
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            mirror.SetPropertyByReference(u"Symmetry", ref2)

        # print region
        # print ass.GetItem(u"Region.1")
        if merge == False and region.GetName()==u"Region":
            return self.ass.GetItem(u"Region.1")
            # å…¶ä»–æƒ…å†µè¿˜æ²¡å†™å“¦ï¼Œæœ€å\½ä¿è¯ä½ å°±è¿™ä¸€ä¸ªregion

            # Region_Mirror_Copy_2 = sketch.CreateRegionMirrorCopy()
            # sketch.GetItem(u"Region Mirror Copy").SetProperty(u"Merge", 1)
            # refarray = [0 for i in range(1)]
            # refarray[0] = u"faceregion(TRegionItem68)" # åœ¨å½•è¿™ä¸€æ­\å‰ï¼Œå¿…é¡»æŠŠGeometry Editorç»™å…³äº†ï¼Œé‡æ–°è·‘ä¸€éå‰é¢çš„ä»£ç ï¼Œè¿™ä¸ªæ•°å­—æ‰å¯¹ï¼
            # sketch.GetItem(u"Region Mirror Copy").SetProperty(u"Region", refarray)
            # ref1 = sketch.GetItem(u"Line")
            # ref2 = self.doc.CreateReferenceFromItem(ref1)
            # sketch.GetItem(u"Region Mirror Copy").SetPropertyByReference(u"Symmetry", ref2)

    def region_circular_pattern_360_origin(self, region, Q_float, merge=True, do_you_have_region_in_the_mirror=False):
        circular_pattern = self.sketch.CreateRegionCircularPattern()
        circular_pattern.SetProperty(u"Merge", merge)

        ref2 = self.doc.CreateReferenceFromItem(region)
        circular_pattern.SetPropertyByReference(u"Region", ref2)
        face_region_string = circular_pattern.GetProperty("Region")
        face_region_string = face_region_string[0]
        # print circular_pattern.GetProperty("Region") # this will produce faceRegion references!

        if do_you_have_region_in_the_mirror == True:
            # 这里假设face_region_string最后两位是数字
            # 乱码的原因是因为我有一次手贱，用DOS-CMD把所有文件的大小扫了一遍还是怎么的，中文就乱码了。
            # 20181114 (Designer from scratch) 没有乱码哦（好像是从Onedrive上找回的）
            # 总结JMAG的代码生成规律……
            # è¿™é‡Œå‡è®¾face_region_stringæœ€åŽä¸¤ä½æ˜¯æ•°å­—
            if face_region_string[-7:-3] == 'Item':
                number_plus_1 = str(int(face_region_string[-3:-1]) + 1)
                refarray = [0 for i in range(2)]
                refarray[0] = u"faceregion(TRegionMirrorPattern%s+%s_2)" % (number_plus_1, face_region_string)
                refarray[1] = face_region_string
                circular_pattern.SetProperty(u"Region", refarray)
                # print refarray[0]
                # print refarray[1]
            elif face_region_string[-6:-2] == 'Item':
                # 这里假设face_region_string最后一位是数字
                # è¿™é‡Œå‡è®¾face_region_stringæœ€åŽä¸€ä½æ˜¯æ•°å­—
                number_plus_1 = str(int(face_region_string[-2:-1]) + 1)
                refarray = [0 for i in range(2)]
                refarray[0] = u"faceregion(TRegionMirrorPattern%s+%s_2)" % (number_plus_1, face_region_string)
                refarray[1] = face_region_string
                circular_pattern.SetProperty(u"Region", refarray)
            elif face_region_string[-8:-4] == 'Item':
                # 这里假设face_region_string最后三位是数字
                # è¿™é‡Œå‡è®¾face_region_stringæœ€åŽä¸‰ä½æ˜¯æ•°å­—
                number_plus_1 = str(int(face_region_string[-4:-1]) + 1)
                refarray = [0 for i in range(2)]
                refarray[0] = u"faceregion(TRegionMirrorPattern%s+%s_2)" % (number_plus_1, face_region_string)
                refarray[1] = face_region_string
                circular_pattern.SetProperty(u"Region", refarray)




        circular_pattern.SetProperty(u"CenterType", 2)
        circular_pattern.SetProperty(u"Angle", u"360/%d"%(int(Q_float)))
        circular_pattern.SetProperty(u"Instance", int(Q_float))

        # if merge == False:


    # I add constraint only for rotate the model for static FEA in JMAG
    def constraint_fixture_circle_center(self, c):
        sketch = self.ass.GetItem(self.SketchName) # ass is global
        ref1 = c.GetCenterVertex()
        ref2 = self.doc.CreateReferenceFromItem(ref1)
        constraint = sketch.CreateMonoConstraint(u"fixture", ref2)
        # constraint.SetProperty(u"Name", constraint_name)

    def constraint_radius_arc(self, arc, radius_value, constraint_name):
        sketch = self.ass.GetItem(self.SketchName)
        ref1 = arc
        ref2 = self.doc.CreateReferenceFromItem(ref1)
        constraint = sketch.CreateMonoConstraint(u"radius", ref2)
        constraint.SetProperty(u"Radius", radius_value)
        constraint.SetProperty(u"Name", constraint_name)

    def rigidset(self, vertex_name_list): # this won't work, reason is mystery
        sketch = self.ass.GetItem(self.SketchName)
        sketch.CreateConstraint(u"rigidset")
        # var_list = ['ref%d'%(i+1) for i in range(2*len(vertex_name_list))]
        var_list = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
        for ind, vtx_name in enumerate(vertex_name_list):
            var_list[ind]   = sketch.GetItem(vtx_name)
            var_list[ind+1] = self.doc.CreateReferenceFromItem(ref1)
            sketch.GetItem(u"Relative Fixation").SetPropertyByReference(u"TargetList", var_list[ind+1])


    # global SketchName 
    ''' Parts to Plot '''
    def plot_shaft(self, name=None):
        if name == None:
            self.SketchName=u"Shaft"
        else:
            self.SketchName=name
        SketchName = self.SketchName
        sketch = self.create_sketch(SketchName, u"#D1B894")

        self.circle(0, 0, self.im.Radius_Shaft)

        self.doc.GetSelection().Clear()
        self.doc.GetSelection().Add(sketch.GetItem(u"Circle"))
        sketch.CreateRegions()

        sketch.CloseSketch()
        # sketch.SetProperty(u"Visible", 0)
    def plot_rotorCore(self, name=None):
        if name == None:
            self.SketchName=u"Rotor Core"
        else:
            self.SketchName=name
        SketchName = self.SketchName
        sketch = self.create_sketch(SketchName, u"#FE840E")

        c1=self.circle(0, 0, self.im.Radius_Shaft) # Circle.1
        c2=self.circle(0, 0, self.im.Radius_OuterRotor) # Circle.2

        Vertex_RotorBarCenter = sketch.CreateVertex(-self.im.Location_RotorBarCenter, 0)
        c3=self.circle(-self.im.Location_RotorBarCenter, 0, self.im.Radius_of_RotorSlot) # Circle.3

        l1=self.line(-5.5-self.im.Radius_OuterRotor, 0.5*self.im.Width_RotorSlotOpen, -self.im.Location_RotorBarCenter, 0.5*self.im.Width_RotorSlotOpen) # Line.1 # -5.5 is arbitrary float <0

        ref1 = sketch.GetItem(u"Line")
        ref2 = self.doc.CreateReferenceFromItem(ref1)
        sketch.CreateMonoConstraint(u"horizontality", ref2) # how to set constraint

        if self.im.use_drop_shape_rotor_bar == True:
            l2=self.line(0, 0, -self.im.Location_RotorBarCenter2+self.im.Radius_of_RotorSlot2, 0) # Line.2
            ref1 = sketch.GetItem(u"Line.2")
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            sketch.CreateMonoConstraint(u"horizontality", ref2)            
        else:
            l2=self.line(0, 0, -self.im.Location_RotorBarCenter+self.im.Radius_of_RotorSlot, 0) # Line.2
            ref1 = sketch.GetItem(u"Line.2")
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            sketch.CreateMonoConstraint(u"horizontality", ref2)

        R = self.im.Radius_OuterRotor
        THETA = (180-0.5*360/self.im.Qr)/180.*pi
        X = R*cos(THETA)
        Y = R*sin(THETA)
        l3 = self.line(0, 0, X, Y) # Line.3

        # raise Exception('before trim')

        # trim the lines first, because there is a chance the inner circle of rotor slot can intesect l1
        self.trim_l(l1,-EPS-self.im.Radius_OuterRotor, 0.5*self.im.Width_RotorSlotOpen)
        self.trim_l(l1,-EPS-self.im.Location_RotorBarCenter, 0.5*self.im.Width_RotorSlotOpen)

        if self.im.use_drop_shape_rotor_bar == True:
            # the inner rotor slot for drop shape rotor suggested by Gerada2011
            c4=self.circle(-self.im.Location_RotorBarCenter2, 0, self.im.Radius_of_RotorSlot2) # Circle.4

            l4 = self.line(-self.im.Location_RotorBarCenter-0.5*self.im.Radius_of_RotorSlot, c3.GetRadius(), -self.im.Location_RotorBarCenter2+0.5*self.im.Radius_of_RotorSlot2, c4.GetRadius())

            # Constraint to fix c4's center
            ref1 = c4.GetCenterVertex()
            ref2 = self.doc.CreateReferenceFromItem(ref1)
                # sketch.CreateMonoConstraint(u"distancefromxaxis", ref2)
                # sketch.GetItem(u"Distance From X Axis").SetProperty(u"Distance", 0)
            # Constrants to avoid moving of circle center
            # self.constraint_fixture_circle_center(c1)
            self.constraint_fixture_circle_center(c4) # Fixture constraint


                # # Constraint to fix c3's center
                # ref1 = c3.GetCenterVertex()
                # ref2 = self.doc.CreateReferenceFromItem(ref1)
                # sketch.CreateMonoConstraint(u"distancefromxaxis", ref2)
                # sketch.GetItem(u"Distance From X Axis").SetProperty(u"Distance", 0)

            ref1 = c4
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            ref3 = l4
            ref4 = self.doc.CreateReferenceFromItem(ref3)
            sketch.CreateBiConstraint(u"tangency", ref2, ref4)
            # sketch.GetItem(u"Vertex.15").SetProperty(u"Y", 2.345385)
            # sketch.GetItem(u"Vertex.16").SetProperty(u"Y", 2.345385)


            ref1 = c3
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            ref3 = l4
            ref4 = self.doc.CreateReferenceFromItem(ref3)
            sketch.CreateBiConstraint(u"tangency", ref2, ref4)
            # sketch.GetItem(u"Vertex.7").SetProperty(u"Y", 2.993642)
            # sketch.GetItem(u"Vertex.8").SetProperty(u"Y", 2.993642)

            # we won't need this fixture constraint anymore
            self.doc.GetSelection().Clear()
            self.doc.GetSelection().Add(sketch.GetItem(u"Fixture")) # Fixture.1
            self.doc.GetSelection().Delete()

            if -self.im.Location_RotorBarCenter + self.im.Radius_of_RotorSlot > -self.im.Location_RotorBarCenter2 - self.im.Radius_of_RotorSlot2:
                'Two circles have intersections.'
                # delete c4
                self.doc.GetSelection().Clear()
                self.doc.GetSelection().Add(c4)
                self.doc.GetSelection().Delete()

                # raise Exception('Is c4 still there? If c4 is not deleted, you will get Unexpected Part Number error afterwards. So fix it here now.')



        # Trim the Sketch Object!
        arc2 = self.trim_c(c2,0, c2.GetRadius()) # or self.trim_c(c2,0, self.im.Radius_OuterRotor)



        X = float(c3.GetCenterVertex().GetX()) # the returned type is long, which will cause failing to trim. Convert to float. Or simply add 1e-2 to it, so Python will do the conversion for you.
        arc3 = self.trim_c(c3, X, -c3.GetRadius()+1e-6) # or self.trim_c(c3,-self.im.Location_RotorBarCenter, -self.im.Radius_of_RotorSlot)


        # åˆ°è¿™ä¸€æ­\ï¼Œä¸èƒ½å…ˆåˆ‡Line.3ï¼Œå¦åˆ™Circle.1ä¼šå\½åƒä¸å­˜åœ¨ä¸€æ ·ï¼Œå¯¼è‡´Line.3æ•´æ ¹æ¶ˆå¤±ï¼æ‰€ä»\ï¼Œå…ˆåˆ‡Circle.1
        arc1 = self.trim_c(c1, 0, c1.GetRadius()) # or self.trim_c(c1, 0, self.im.Radius_Shaft)

        self.trim_l(l2,-0.5*self.im.Radius_Shaft, 0)
            # if self.im.use_drop_shape_rotor_bar == True:
            #     self.trim_l(l2,0.5*self.im.Radius_of_RotorSlot2-self.im.Location_RotorBarCenter2, 0)
            # else:
            #     self.trim_l(l2,0.5*self.im.Radius_of_RotorSlot-self.im.Location_RotorBarCenter, 0)

        # 上面说的“导致Line.3整根消失！”的原因是直接剪在了原点(0,0)上，所以把整根线都删掉了，稍微偏一点(-0.1,0.1)操作即可
        # ä¸Šé¢è¯´çš„â€œå¯¼è‡´Line.3æ•´æ ¹æ¶ˆå¤±ï¼â€çš„åŽŸå› æ˜¯ç›´æŽ\å‰ªåœ¨äº†åŽŸç‚¹(0,0)ä¸Šï¼Œæ‰€ä»\æŠŠæ•´æ ¹çº¿éƒ½åˆ æŽ‰äº†ï¼Œç¨å¾®åä¸€ç‚¹(-0.1,0.1)æ“ä½œå³å¯
            # self.doc.GetSketchManager().SketchTrim(0,0) # BUG - delete the whole Line.3
        self.trim_l(l3,-1e-2, 1e-2)

        if self.im.use_drop_shape_rotor_bar == True:

            if -self.im.Location_RotorBarCenter + self.im.Radius_of_RotorSlot > -self.im.Location_RotorBarCenter2 - self.im.Radius_of_RotorSlot2:
                'Two circles have intersections.'
                # re-create c4
                c4=self.circle(-self.im.Location_RotorBarCenter2, 0, self.im.Radius_of_RotorSlot2) # Circle.4

            arc4 = self.trim_c(c4, -self.im.Location_RotorBarCenter2, -c4.GetRadius())

            self.trim_l(l4, l4.GetStartVertex().GetX()+1e-2, l4.GetStartVertex().GetY()+1e-2)
            self.trim_l(l4, l4.GetEndVertex().GetX()-1e-2, l4.GetEndVertex().GetY()-1e-2)

            # Mirror and Duplicate
            region = self.create_region([u"Arc",u"Arc.2",u"Arc.3",u"Arc.4",u"Line",u"Line.2",u"Line.3",u"Line.4"])
        else:
            # Mirror and Duplicate
            region = self.create_region([u"Arc",u"Arc.2",u"Arc.3",u"Line",u"Line.2",u"Line.3"])


        # This is necessary if you want to do MODEL_ROTATE
        if self.im.MODEL_ROTATE:
            self.doc.GetSelection().Clear()
            self.doc.GetSelection().Add(sketch.GetItem(u"Horizontality"))
            self.doc.GetSelection().Add(sketch.GetItem(u"Horizontality.2"))
            self.doc.GetSelection().Add(sketch.GetItem(u"Tangency"))
            self.doc.GetSelection().Add(sketch.GetItem(u"Tangency.2"))
            self.doc.GetSelection().Add(sketch.GetItem(u"Coincident"))
            self.doc.GetSelection().Add(sketch.GetItem(u"Coincident.2"))
            self.doc.GetSelection().Delete()

            self.im.list_rotorCore_vertex_names = [ arc1.GetCenterVertex().GetName(),
                                                    arc2.GetCenterVertex().GetName(),
                                                    arc3.GetCenterVertex().GetName(),
                                                    arc4.GetCenterVertex().GetName(),
                                                    l1.GetStartVertex().GetName(),
                                                    l1.GetEndVertex().GetName(),
                                                    l2.GetStartVertex().GetName(),
                                                    l2.GetEndVertex().GetName(),
                                                    l3.GetStartVertex().GetName(),
                                                    l3.GetEndVertex().GetName(),
                                                    l4.GetStartVertex().GetName(),
                                                    l4.GetEndVertex().GetName()]
            self.im.list_rotorCore_vertex_names = dict.fromkeys(self.im.list_rotorCore_vertex_names).keys() #https://stackoverflow.com/questions/7961363/removing-duplicates-in-lists?page=1&tab=votes#tab-top

        self.region_mirror_copy(region, l3)
        self.region_circular_pattern_360_origin(region, self.im.Qr)


        sketch.CloseSketch()
        # sketch.SetProperty(u"Visible", 0)
    def plot_statorCore(self, name=None):
        if name == None:
            self.SketchName=u"Stator Core"
        else:
            self.SketchName=name
        SketchName = self.SketchName
        sketch = self.create_sketch(SketchName, u"#E8B5CE")

        self.im.Radius_InnerStator = self.im.Length_AirGap + self.im.Radius_OuterRotor
        sketch.CreateVertex(0., 0.)
        c1=self.circle(0., 0., self.im.Radius_InnerStator)
        c2=self.circle(0., 0., self.im.Radius_InnerStatorYoke)
        c3=self.circle(0., 0., self.im.Radius_OuterStatorYoke)
        c4=self.circle(0., 0., self.im.Radius_InnerStator + self.im.Width_StatorTeethHeadThickness)

        l1=self.line(-self.im.Radius_OuterStatorYoke, 0., 0., 0.)
        l2=self.line(-0.5*(self.im.Radius_OuterStatorYoke+self.im.Radius_InnerStatorYoke), 0.5*self.im.Width_StatorTeethBody, \
                -(self.im.Radius_InnerStator + (self.im.Width_StatorTeethHeadThickness+self.im.Width_StatorTeethNeck)), 0.5*self.im.Width_StatorTeethBody) # legacy 1. NEW: add neck here. Approximation is adopted: the accurate results should be (self.im.Width_StatorTeethHeadThickness+self.im.Width_StatorTeethNeck) * cos(6 deg) or so. 6 deg = 360/24/2 - 3/2

        R = self.im.Radius_OuterStatorYoke
        THETA = (180-0.5*360/self.im.Qs)/180.*pi
        X = R*cos(THETA)
        Y = R*sin(THETA)
        l3 = StatorCore_Line_3 = self.line(0., 0., X, Y) # Line.3

        R = self.im.Radius_InnerStator + self.im.Width_StatorTeethHeadThickness
        THETA = (180.-(0.5*360./self.im.Qs-0.5*self.im.Angle_StatorSlotOpen))/180.*pi # BUG is found here, 3 is instead used for Angle_StatorSlotOpen
        X = R*cos(THETA)
        Y = R*sin(THETA)
        l4 = self.line(0., 0., X, Y) # Line.4


        # Trim for Arcs

        # 为了避免数Arc，我们应该在Circle.4时期就把Circle.4上的小片段优先剪了！
        arc4 = self.trim_c(c4, X+1e-6, Y+1e-6) # Arc.1

        self.trim_a(arc4, 0., self.im.Radius_InnerStator + self.im.Width_StatorTeethHeadThickness)
        self.trim_a(arc4, -(self.im.Radius_InnerStator + self.im.Width_StatorTeethHeadThickness), 1e-6)

        arc1 = self.trim_c(c1, 0., self.im.Radius_InnerStator) # self.trim_c(c1,0, self.im.Radius_InnerStator)

        R = self.im.Radius_InnerStator
        THETA = (180-0.5*360./self.im.Qs)/180.*pi
        X = R*cos(THETA)
        Y = R*sin(THETA)
        self.trim_a(arc1, X, Y-1e-6)

        # 为了避免数Arc，我们应该在Circle.2时期就把Circle.2上的小片段优先剪了！
        arc2 = self.trim_c(c2,-self.im.Radius_InnerStatorYoke, 0.25*self.im.Width_StatorTeethBody)

        self.trim_a(arc2, 0., self.im.Radius_InnerStatorYoke)

        self.trim_c(c3,0., self.im.Radius_OuterStatorYoke)

        self.trim_l(l1,-0.1, 0.)


        self.trim_l(l2,1e-6 -0.5*(self.im.Radius_OuterStatorYoke+self.im.Radius_InnerStatorYoke), 0.5*self.im.Width_StatorTeethBody)
            # self.trim_l(l2,1e-6 -1e-6 -(self.im.Radius_InnerStator + 0.5*(self.im.Width_StatorTeethHeadThickness+self.im.Width_StatorTeethNeck)), 0.5*self.im.Width_StatorTeethBody)

        self.trim_l(l3,-1e-6, 1e-6)

        self.trim_l(l4,-1e-6, 1e-6) # float number

        # This error is caused because Y=2 is an integer!
        # 2019-01-15 17:48:55,243 - population - ERROR - The drawing is terminated. Please check whether the specified bounds are proper.
        # Traceback (most recent call last):
        #   File "D:/OneDrive - UW-Madison/c/codes/population.py", line 816, in draw_jmag_model
        #     d.plot_statorCore(u"Stator Core")
        #   File "D:/OneDrive - UW-Madison/c/codes/population.py", line 4066, in plot_statorCore
        #     l5_start_vertex = sketch.CreateVertex(X, Y)
        #   File "<string>", line 43, in CreateVertex
        # ValueError: Error calling the remote method

        # Similaryly, this error is caused because self.l5_start_vertex_y=2 is an integer!
        # 2019-01-15 19:00:33,286 - population - ERROR - The drawing is terminated. Please check whether the specified bounds are proper.
        # Traceback (most recent call last):
        #   File "D:/OneDrive - UW-Madison/c/codes/population.py", line 818, in draw_jmag_model
        #     d.plot_statorCore(u"Stator Core")
        #   File "D:/OneDrive - UW-Madison/c/codes/population.py", line 4106, in plot_statorCore
        #     raise e
        # ValueError: Error calling the remote method

        # we forget to plot the neck of stator tooth
        self.l5_start_vertex_x = l2.GetEndVertex().GetX()        # used later for plot_coil()
        self.l5_start_vertex_y = float(l2.GetEndVertex().GetY()) # used later for plot_coil()                
        X = arc4.GetStartVertex().GetX()
        Y = arc4.GetStartVertex().GetY()
        try:
            l5 = self.line( self.l5_start_vertex_x, self.l5_start_vertex_y, X, Y) # Line.5
        except Exception, e:
            print l2.GetEndVertex().GetX(), l2.GetEndVertex().GetY(), arc4.GetStartVertex().GetX(), arc4.GetStartVertex().GetY()
            logger = logging.getLogger(__name__)
            logger.error(u'Draw Line.5 for Stator Core Failed, because integer cannot be passed as paramters to JMAG remote mathod. It is wierd, but just follow the rule.', exc_info=True)
            raise e


        self.doc.GetSelection().Clear()
        self.doc.GetSelection().Add(sketch.GetItem(arc4.GetName())) # arc4 corresponds to u"Arc".
        self.doc.GetSelection().Delete()

            # self.trim_l(l2, l2_end_vertex.GetX()-1e-3, l2_end_vertex.GetY()+1e-3) # convert to float (it returns long int) # legacy 3


        region = self.create_region([u"Arc.2",u"Arc.3",u"Arc.4",u"Line",u"Line.2",u"Line.3",u"Line.4",u"Line.5"])

        self.region_mirror_copy(region, l1)

        self.region_circular_pattern_360_origin(region, self.im.Qs)

        ''' Constraints Stator Core '''
        if 0:
            ref1 = sketch.GetItem(u"Line.2")
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            ref3 = sketch.GetItem(u"Line")
            ref4 = self.doc.CreateReferenceFromItem(ref3)
            sketch.CreateBiConstraint(u"distance", ref2, ref4)
            sketch.GetItem(u"Distance").SetProperty(u"Distance", 0.5*self.im.Width_StatorTeethBody)

        sketch.CloseSketch()
        # sketch.SetProperty(u"Visible", 0)
    def plot_cage(self, name=None):
        if name == None:
            self.SketchName=u"Cage"
        else:
            self.SketchName=name
        SketchName = self.SketchName
        sketch = self.create_sketch(SketchName, u"#8D9440")

        c3=self.circle(-self.im.Location_RotorBarCenter, 0., self.im.Radius_of_RotorSlot)

        if self.im.use_drop_shape_rotor_bar == True:
            # the inner rotor slot for drop shape rotor suggested by Gerada2011
            c4=self.circle(-self.im.Location_RotorBarCenter2, 0., self.im.Radius_of_RotorSlot2) # Circle.4
            # Constraint to fix c4's center
            ref1 = c4.GetCenterVertex()
            ref2 = self.doc.CreateReferenceFromItem(ref1)
                # sketch.CreateMonoConstraint(u"distancefromxaxis", ref2)
                # sketch.GetItem(u"Distance From X Axis").SetProperty(u"Distance", 0)
            # Constrants to avoid moving of circle center
            self.constraint_fixture_circle_center(c4)


            l41 = self.line(-self.im.Location_RotorBarCenter-0.5*self.im.Radius_of_RotorSlot, c3.GetRadius(), -self.im.Location_RotorBarCenter2+0.5*self.im.Radius_of_RotorSlot2, c4.GetRadius())
            l42 = self.line(-self.im.Location_RotorBarCenter-0.5*self.im.Radius_of_RotorSlot, -c3.GetRadius(), -self.im.Location_RotorBarCenter2+0.5*self.im.Radius_of_RotorSlot2, -c4.GetRadius())

            ref1 = c4
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            ref3 = l41
            ref4 = self.doc.CreateReferenceFromItem(ref3)
            sketch.CreateBiConstraint(u"tangency", ref2, ref4)
            ref1 = c3
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            ref3 = l41
            ref4 = self.doc.CreateReferenceFromItem(ref3)
            sketch.CreateBiConstraint(u"tangency", ref2, ref4)

            ref1 = c4
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            ref3 = l42
            ref4 = self.doc.CreateReferenceFromItem(ref3)
            sketch.CreateBiConstraint(u"tangency", ref2, ref4)
            ref1 = c3
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            ref3 = l42
            ref4 = self.doc.CreateReferenceFromItem(ref3)
            sketch.CreateBiConstraint(u"tangency", ref2, ref4)

            if -self.im.Location_RotorBarCenter + self.im.Radius_of_RotorSlot > -self.im.Location_RotorBarCenter2 - self.im.Radius_of_RotorSlot2:
                'Two circles have intersections.'
                # delete c4
                self.doc.GetSelection().Clear()
                self.doc.GetSelection().Add(c4)
                self.doc.GetSelection().Delete()
            arc3 = self.trim_c(c3, c3.GetCenterVertex().GetX()+1e-2+c3.GetRadius(), 0) # make sure it is float number

            if -self.im.Location_RotorBarCenter + self.im.Radius_of_RotorSlot > -self.im.Location_RotorBarCenter2 - self.im.Radius_of_RotorSlot2:
                'Two circles have intersections.'
                # re-create c4
                c4=self.circle(-self.im.Location_RotorBarCenter2, 0, self.im.Radius_of_RotorSlot2) # Circle.4
            arc4 = self.trim_c(c4, -self.im.Location_RotorBarCenter2-c4.GetRadius(), 0)

            self.trim_l(l41, l41.GetStartVertex().GetX()+1e-2, l41.GetStartVertex().GetY()+1e-2)
            self.trim_l(l41, l41.GetEndVertex().GetX()-1e-2, l41.GetEndVertex().GetY()-1e-2)

            self.trim_l(l42, l42.GetStartVertex().GetX()+1e-2, l42.GetStartVertex().GetY()+1e-2)
            self.trim_l(l42, l42.GetEndVertex().GetX()-1e-2, l42.GetEndVertex().GetY()-1e-2)

            region = self.create_region([u"Arc",u"Arc.2",u"Line",u"Line.2"])
        else:            
            region = self.create_region([u"Circle"])

        if self.im.MODEL_ROTATE:
            self.doc.GetSelection().Clear()
            self.doc.GetSelection().Add(sketch.GetItem(u"Fixture"))
            self.doc.GetSelection().Add(sketch.GetItem(u"Tangency"))
            self.doc.GetSelection().Add(sketch.GetItem(u"Tangency.2"))
            self.doc.GetSelection().Add(sketch.GetItem(u"Tangency.3"))
            self.doc.GetSelection().Add(sketch.GetItem(u"Tangency.4"))
            self.doc.GetSelection().Delete()

            self.im.list_rotorCage_vertex_names = [  arc3.GetCenterVertex().GetName(),
                                                arc4.GetCenterVertex().GetName(),
                                                l41.GetStartVertex().GetName(),
                                                l41.GetEndVertex().GetName(),
                                                l42.GetStartVertex().GetName(),
                                                l42.GetEndVertex().GetName()]
            self.im.list_rotorCage_vertex_names = dict.fromkeys(self.im.list_rotorCage_vertex_names).keys()


        self.region_circular_pattern_360_origin(region, self.im.Qr)        

        sketch.CloseSketch()
    def plot_coil(self, name=None):
        if name == None:
            self.SketchName=u"Coil"
        else:
            self.SketchName=name
        SketchName = self.SketchName
        sketch = self.create_sketch(SketchName, u"#EC9787")


        # 方案一 画中线，然后镜像到对面，得到另外两个点！
        # 方案二 为什么不直接画圆然后Trim哦！
        if 1:
            self.im.Radius_InnerStator = self.im.Length_AirGap + self.im.Radius_OuterRotor
            # sketch.CreateVertex(0, 0)
            c1=self.circle(0, 0, self.im.Radius_InnerStatorYoke)
            c2=self.circle(0, 0, self.im.Radius_InnerStator + self.im.Width_StatorTeethHeadThickness)


            l2l5 = self.line(-0.5*(self.im.Radius_OuterStatorYoke+self.im.Radius_InnerStatorYoke), 0.5*self.im.Width_StatorTeethBody, \
                             self.l5_start_vertex_x, self.l5_start_vertex_y)

            # l2 = self.line(-0.5*(self.im.Radius_OuterStatorYoke+self.im.Radius_InnerStatorYoke), 0.5*self.im.Width_StatorTeethBody, \
            #                -(self.im.Radius_InnerStator + 1.0*self.im.Width_StatorTeethHeadThickness) - self.im.Width_StatorTeethNeck, 0.5*self.im.Width_StatorTeethBody)

            R = self.im.Radius_InnerStatorYoke
            THETA = (180-0.5*360/self.im.Qs)/180.*pi
            X = R*cos(THETA)
            Y = R*sin(THETA)
            l3 = self.line(0, 0, X, Y) # Line.3


            # Trim Lines
            self.trim_l(l2l5,1e-6 -0.5*(self.im.Radius_OuterStatorYoke+self.im.Radius_InnerStatorYoke), 0.5*self.im.Width_StatorTeethBody)
                # self.trim_l(l2l5,-1e-6 -(self.im.Radius_InnerStator + 0.5*self.im.Width_StatorTeethHeadThickness), 0.5*self.im.Width_StatorTeethBody)
            self.trim_l(l3,-1e-6, 1e-6)


            l6 = self.line(self.l5_start_vertex_x, self.l5_start_vertex_y, \
                            l3.GetStartVertex().GetX(), l3.GetStartVertex().GetY())


            # Trim Circles
            self.trim_c(c1,0, self.im.Radius_InnerStatorYoke)
            self.trim_c(c2,0, self.im.Radius_InnerStator + self.im.Width_StatorTeethHeadThickness)


        # Mirror and Duplicate
        region = self.create_region([u"Line",u"Line.2",u"Line.3",u"Arc"])
        
        # region_mirror_pattern_which_is_ItemObject = self.region_mirror_copy(region, l3, merge=False)
        # region_in_the_mirror = RegionItem(region_mirror_pattern_which_is_ItemObject)
        region_in_the_mirror = self.region_mirror_copy(region, l3, merge=False)
        # print region_in_the_mirror # it's ItemObject rather than RegionObject! We are fucked, because we cannot generate reference for ItemObject. We can generate ref for RegionObject.

        self.region_circular_pattern_360_origin(region, self.im.Qs, merge=False, do_you_have_region_in_the_mirror=True)

        ''' Constraints Coil '''
        if 0:
            ref1 = sketch.GetItem(u"Line") # or l1
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            sketch.CreateMonoConstraint(u"horizontality", ref2)

            ref1 = sketch.GetItem(u"Line").GetEndVertex()
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            sketch.CreateMonoConstraint(u"distancefromxaxis", ref2)
            sketch.GetItem(u"Distance From X Axis").SetProperty(u"Distance", 0.5*self.im.Width_StatorTeethBody)

        sketch.CloseSketch()
    def plot_airWithinRotorSlots(self, name=None):
        if name == None:
            self.SketchName=u"Air Within Rotor Slots"
        else:
            self.SketchName=name
        SketchName = self.SketchName
        sketch = self.create_sketch(SketchName, u"#BFD641")


        c1=self.circle(0,0,self.im.Radius_OuterRotor)
        c2=self.circle(-self.im.Location_RotorBarCenter,0,self.im.Radius_of_RotorSlot)

        l1=self.line(-5.5-self.im.Radius_OuterRotor, 0.5*self.im.Width_RotorSlotOpen, -self.im.Location_RotorBarCenter, 0.5*self.im.Width_RotorSlotOpen)
        l2=self.line(-5.5-self.im.Radius_OuterRotor,                       0, -self.im.Location_RotorBarCenter,                       0)

        self.trim_l(l1,-5-self.im.Radius_OuterRotor, 0.5*self.im.Width_RotorSlotOpen)
        self.trim_l(l2,-5-self.im.Radius_OuterRotor, 0)

        self.trim_l(l1,-1e-6-self.im.Location_RotorBarCenter, 0.5*self.im.Width_RotorSlotOpen)
        self.trim_l(l2,-1e-6-self.im.Location_RotorBarCenter, 0)

        arc1 = self.trim_c(c1,0,c1.GetRadius())
        arc2 = self.trim_c(c2,0,c2.GetRadius())


        if self.im.MODEL_ROTATE:
            self.doc.GetSelection().Clear()
            self.doc.GetSelection().Add(sketch.GetItem(u"Coincident"))
            self.doc.GetSelection().Add(sketch.GetItem(u"Fixture"))
            self.doc.GetSelection().Delete()

            self.im.list_rotorAirWithin_vertex_names = [ arc1.GetCenterVertex().GetName(),
                                                    arc2.GetCenterVertex().GetName(),
                                                    l1.GetStartVertex().GetName(),
                                                    l1.GetEndVertex().GetName(),
                                                    l2.GetStartVertex().GetName(),
                                                    l2.GetEndVertex().GetName()]
            self.im.list_rotorAirWithin_vertex_names = dict.fromkeys(self.im.list_rotorAirWithin_vertex_names).keys()

        region = self.create_region([u"Arc.2",u"Arc",u"Line.2",u"Line"])

        self.region_mirror_copy(region, l2)
        self.region_circular_pattern_360_origin(region, self.im.Qr)

        ''' Constraint Air within Rotor Slots '''
        if 0:
            ref1 = sketch.GetItem(u"Vertex.2")
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            sketch.CreateMonoConstraint(u"distancefromyaxis", ref2)
            sketch.GetItem(u"Distance From Y Axis").SetProperty(u"Distance", self.im.Location_RotorBarCenter)
            sketch.GetItem(u"Distance From Y Axis").SetProperty(u"Name", u"_Air_LocationBar")

            ref1 = sketch.GetItem(u"Arc.2")
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            sketch.CreateMonoConstraint(u"radius", ref2)
            sketch.GetItem(u"Radius/Diameter").SetProperty(u"Radius", self.im.Radius_of_RotorSlot)
            sketch.GetItem(u"Radius/Diameter").SetProperty(u"Name", u"_Air_RadiusOfRotorSlot")

            ref1 = sketch.GetItem(u"Vertex")
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            sketch.CreateMonoConstraint(u"fixture", ref2)

            ref1 = sketch.GetItem(u"Arc")
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            sketch.CreateMonoConstraint(u"radius", ref2)
            sketch.GetItem(u"Radius/Diameter.2").SetProperty(u"Radius", self.im.Radius_OuterRotor)
            sketch.GetItem(u"Radius/Diameter.2").SetProperty(u"Name", u"_Air_RadiusOuterRotor")

            ref1 = sketch.GetItem(u"Line")
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            ref3 = sketch.GetItem(u"Line.2")
            ref4 = self.doc.CreateReferenceFromItem(ref3)
            sketch.CreateBiConstraint(u"distance", ref2, ref4)
            sketch.GetItem(u"Distance").SetProperty(u"Distance", 0.5*self.im.Width_RotorSlotOpen)
            sketch.GetItem(u"Distance").SetProperty(u"Name", u"_Air_HalfSlotOpenRotor")

        sketch.CloseSketch()
    def plot_airHugRotor(self, name=None):
        if name == None:
            self.SketchName=u"Air Hug Rotor"
        else:
            self.SketchName=name
        SketchName = self.SketchName
        sketch = self.create_sketch(SketchName, u"#4F84C4")

        R = self.im.Radius_OuterRotor+0.5*self.im.Length_AirGap
        c1=self.circle(0,0,self.im.Radius_OuterRotor)
        c2=self.circle(0,0,R)

        l1=self.line(0,-R,0,R)
        self.trim_l(l1,0,0)

        self.trim_c(c1,c1.GetRadius(),0)
        self.trim_c(c2,c2.GetRadius(),0)


        region = self.create_region([u"Arc",u"Arc.2",u"Line",u"Line.2"])

        self.region_mirror_copy(region, symmetry_type=3) # y-axis æ˜¯å¯¹ç§°è½´
        # sketch.CreateRegionMirrorCopy()
        # sketch.GetItem(u"Region Mirror Copy").SetProperty(u"Merge", 1)
        # refarray = [0 for i in range(1)]
        # refarray[0] = u"faceregion(TRegionItem126)"
        # sketch.GetItem(u"Region Mirror Copy").SetProperty(u"Region", refarray)
        # sketch.GetItem(u"Region Mirror Copy").SetProperty(u"SymmetryType", 3)

        sketch.CloseSketch()

def add_M1xSteel(app, dir_parent, steel_name=u"M-19 Steel Gauge-29"):

    if '19' in steel_name:
        BH = np.loadtxt(dir_parent + 'Arnon5/M-19-Steel-BH-Curve-afterJMAGsmooth.txt', unpack=True, usecols=(0,1)) # after JMAG smooth, it beomces HB rather than BH
    elif '15' in steel_name:
        BH = np.loadtxt(dir_parent + 'Arnon5/M-15-Steel-BH-Curve.txt', unpack=True, usecols=(0,1))


    app.GetMaterialLibrary().CreateCustomMaterial(steel_name, u"Custom Materials")
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"Density", 7.85)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"MagneticSteelPermeabilityType", 2)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"CoerciveForce", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).GetTable("BhTable").SetName(u"Untitled")

    refarray = BH.T.tolist()

    app.GetMaterialLibrary().GetUserMaterial(steel_name).GetTable("BhTable").SetTable(refarray)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"DemagnetizationCoerciveForce", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"MagnetizationSaturated", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"MagnetizationSaturated2", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"YoungModulus", 210000)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"ShearModulus", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"YoungModulusX", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"YoungModulusY", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"YoungModulusZ", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"ShearModulusXY", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"ShearModulusYZ", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"ShearModulusZX", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G11", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G12", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G13", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G14", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G15", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G16", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G22", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G23", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G24", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G25", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G26", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G33", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G34", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G35", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G36", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G44", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G45", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G46", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G55", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G56", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"G66", 0)

    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"MagnetizationSaturated2", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"MagnetizationSaturatedMakerValue", 0)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"Loss_Type", 1)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"LossConstantKhX", 143.)
    app.GetMaterialLibrary().GetUserMaterial(steel_name).SetValue(u"LossConstantKeX", 0.530)

def add_Arnon5(app, dir_parent):
    app.GetMaterialLibrary().CreateCustomMaterial(u"Arnon5-final", u"Custom Materials")
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"Density", 7.85)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"MagneticSteelPermeabilityType", 2)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"CoerciveForce", 0)
    # app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").GetTable("BhTable").SetName(u"SmoothZeroPointOne")

    BH = np.loadtxt(dir_parent + 'Arnon5/Arnon5-final.txt', unpack=True, usecols=(0,1))
    refarray = BH.T.tolist()

    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").GetTable("BhTable").SetTable(refarray)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"DemagnetizationCoerciveForce", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"MagnetizationSaturated", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"MagnetizationSaturated2", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"ExtrapolationMethod", 1)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"YoungModulus", 210000)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"ShearModulus", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"YoungModulusX", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"YoungModulusY", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"YoungModulusZ", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"ShearModulusXY", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"ShearModulusYZ", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"ShearModulusZX", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G11", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G12", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G13", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G14", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G15", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G16", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G22", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G23", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G24", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G25", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G26", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G33", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G34", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G35", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G36", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G44", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G45", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G46", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G55", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G56", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"G66", 0)

    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"MagnetizationSaturated2", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"MagnetizationSaturatedMakerValue", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"Loss_Type", 1)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"LossConstantKhX", 186.6)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5-final").SetValue(u"LossConstantKeX", 0.07324)


if __name__ == '__main__':
    pass
