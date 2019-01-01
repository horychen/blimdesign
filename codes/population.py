# -*- coding: utf-8 -*-
# execfile(r'D:\Users\horyc\OneDrive - UW-Madison\ec_rotate.py') # , {'__name__': 'load'})
from __future__ import division
from math import cos, sin, pi
from csv import reader as csv_reader
import logging
import numpy as np  # for de
import os
from pylab import plot, legend, grid, figure, subplots, array, mpl
import utility

class suspension_force_vector(object):
    """docstring for suspension_force_vector"""
    def __init__(self, force_x, force_y, range_ss=None):
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

    def unpack(self):
        return self.basic_info, self.time_list, self.TorCon_list, self.ForConX_list, self.ForConY_list, self.ForConAbs_list

class swarm(object):

    def __init__(self, fea_config_dict, de_config_dict=None):
        # design objective
        self.model_name_prefix = fea_config_dict['model_name_prefix']

        # directories
        self.dir_parent             = fea_config_dict['dir_parent']
        self.initial_design_file    = self.dir_parent + 'pop/' + r'initial_design.txt'
        self.dir_csv_output_folder  = self.dir_parent + 'csv/' + self.model_name_prefix + '/'
        if not os.path.exists(self.dir_csv_output_folder):
            os.makedirs(self.dir_csv_output_folder)
        self.run_folder             = fea_config_dict['run_folder']
        self.dir_run                = self.dir_parent + 'pop/' + self.run_folder
        self.dir_project_files      = fea_config_dict['dir_project_files']
        self.dir_jcf                = self.dir_project_files + 'jcf/'
        self.pc_name                = fea_config_dict['pc_name']
        self.fea_config_dict        = fea_config_dict

        # optimization
        self.de_config_dict = de_config_dict
        self.fea_config_dict['flag_optimization'] = False if de_config_dict == None else False

        # load initial design using the obsolete class bearingless_induction_motor_design
        self.im_list = []
        with open(self.initial_design_file, 'r') as f: 
            read_iterator = csv_reader(f, skipinitialspace=True)
            for row in self.whole_row_reader(read_iterator):
                im = bearingless_induction_motor_design([float(el) for el in row], fea_config_dict, model_name_prefix=self.model_name_prefix)
                self.im_list.append(im)
        for im in self.im_list:
            if im.Qr == self.fea_config_dict['Active_Qr']:
                self.im = im
        try: 
            self.im
        except:
            print 'There is no design matching Active_Qr.'
            msg = 'Please activate one initial design. Refer %s.' % (self.initial_design_file)
            logger = logging.getLogger(__name__)
            logger.warn(msg)
            raise Exception('no match for Active_Qr')

        # solve for results
        if not self.has_results(self.im, 'Freq'):
            self.run(self.im, run_list=self.fea_config_dict['jmag_run_list'])
            if not self.has_results(self.im, 'Freq'):
                raise Exception('Something went south with JMAG.')

    def generate_pop(self):
        # csv_output folder is used for optimziation
        self.dir_csv_output_folder  = fea_config_dict['dir_parent'] + 'csv_opti/' + self.run_folder

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

                    self.interrupt_pop = []
                    with open(self.dir_run + file, 'r') as f:
                        read_iterator = csv_reader(f, skipinitialspace=True)
                        for row in self.whole_row_reader(read_iterator):
                            if len(row)>0: # there could be empty row, since we use append mode and write down f.write('\n')
                                self.interrupt_pop.append([float(el) for el in row])

                    self.interrupt_pop = np.asarray(self.interrupt_pop)
                    self.index_interrupt_beginning = len(self.interrupt_pop)

                    logger = logging.getLogger(__name__)
                    logger.warn('Unfinished iteration is found with ongoing files in run folder. Make sure the project is not opened in JMAG, and we are going to remove the project files and ongoing files.')
                    logger.warn(u'不出意外，就从第一个个体开始迭代。但是很多时候跑到第150个个体的时候跑断了，你总不会想要把这一代都删了，重新跑吧？')
                    # os.remove(u"D:/JMAG_Files/" + run_folder[:-1] + file[:-12] + ".jproj")
                    print self.interrupt_pop

                if 'fit' in file:
                    self.interrupt_fitness = []
                    with open(self.dir_run + file, 'r') as f: 
                        read_iterator = csv_reader(f, skipinitialspace=True)
                        for row in self.whole_row_reader(read_iterator):
                            if len(row)>0: # there could be empty row, since we use append mode and write down f.write('\n')
                                self.interrupt_fitness.append(float(row[0])) # not neccessary to be an array. a list is enough
                    print self.interrupt_fitness

        # search for generation files
        generations = [file[4:8] for file in os.listdir(self.dir_run) if 'gen' in file and not 'ongoing' in file]
        popsize = de_config_dict['popsize']
        # the least popsize is 4
        if popsize<=3:
            logger = logging.getLogger(__name__)
            logger.error('The popsize must be greater than 3 so the choice function can pick up three other individuals different from the individual under mutation among the population')
            raise Exception('Specify a popsize larger than 3.')
        bounds = de_config_dict['bounds']
        dimensions = len(bounds)
        # check for number of generations
        if len(generations) == 0:
            logger = logging.getLogger(__name__)
            logger.debug('There is no swarm yet. Generate the initial random swarm...')
            
            # generate the initial random swarm from the initial design
            try:
                if de_config_dict == None:
                    raise Exception('unexpected de_config_dict')
                self.init_pop = np.random.rand(popsize, dimensions) # normalized design parameters between 0 and 1
            except KeyError, e:
                logger.error(u'ちゃんとキーを設定して下さい。', exc_info=True)

            # and save to file named gen#0000.txt
            self.number_current_generation = 0
            with open(self.get_gen_file(self.number_current_generation), 'w') as f:
                f.write('\n'.join(','.join('%.16f'%(x) for x in y) for y in self.init_pop)) # convert 2d array to string
            logger = logging.getLogger(__name__)
            logger.debug('Initial Pop is saved as %s', self.dir_run + 'gen#0000.txt')
        else:
            # get the latest generation of swarm data
            self.number_current_generation = max([int(el) for el in generations])

            logger = logging.getLogger(__name__)
            logger.debug('The latest generation is gen#%d', self.number_current_generation)

            # restore the existing swarm from file             
            self.init_pop = []
            with open(self.get_gen_file(self.number_current_generation), 'r') as f:
                read_iterator = csv_reader(f, skipinitialspace=True)
                for row in self.whole_row_reader(read_iterator):
                    if len(row)>0: # there could be empty row, since we use append mode and write down f.write('\n')
                        self.init_pop.append([float(el) for el in row])
            # for el in self.init_pop:
            #     print 
            #     for e in el:
            #         print e,

            # This is useless unless you know the old bounds. In fact, modify the bounds will change all the individuals' design parameters since they are normalized values.
            # # make sure to be consistent with new bounds (maybe)
            # min_b, max_b = np.asarray(bounds).T 
            # diff = np.fabs(min_b - max_b)
            # self.init_pop_denorm = min_b + self.init_pop * diff
            # for index, individual in enumerate(self.init_pop_denorm):
            #     count = 0
            #     for design_parameter, bound in zip(individual, bounds):
            #         if design_parameter < bound[0]:
            #             self.init_pop_denorm[index][count] = bound[0]
            #         elif design_parameter > bound[1]:
            #             self.init_pop_denorm[index][count] = bound[1]
            #         count += 1
            # self.init_pop = (self.init_pop_denorm - min_b)/diff #= pop
            # # print '--------------------------------------'
            # # for el in self.init_pop:
            # #     print 
            # #     for e in el:
            # #         print e,


            # TODO: 文件完整性检查
            solved_popsize = len(self.init_pop)
            if popsize > solved_popsize:
                self.init_pop += np.random.rand(popsize-solved_popsize, dimensions).tolist() # normalized design parameters between 0 and 1
                with open(self.get_gen_file(self.number_current_generation), 'a') as f:
                    f.write('\n')
                    f.write('\n'.join(','.join('%.16f'%(x) for x in y) for y in self.init_pop[solved_popsize:])) # convert 2d array to string
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
            self.app
        except:
            import designer
            self.app = designer.GetApplication() 

        # add_M15Steel(self.app)

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

    def fobj(self, individual_index, individual): 
        # based on the individual data, create design variant of the initial design of Pyrhonen09
        im_variant = local_design_variant(self.im, self.number_current_generation, individual_index, individual) # due to compatability issues: a new child class is used instead
            # im_variant.dir_run = self.dir_run # for export image
        self.im_variant = im_variant # for debug purpose

        # every model is a new project, so as to avoid the situation of 100 models in one project (occupy RAM and slow). add individual_index to project_name
        self.jmag_control_state = False

        # initialize JMAG Designer
        self.designer_init()
        self.project_name = self.run_folder[:-1]+'gen#%04dind#%04d' % (self.number_current_generation, individual_index)
        im_variant.model_name = im_variant.get_model_name() 
        if self.jmag_control_state == False: # initilize JMAG Designer
            expected_project_file = self.dir_project_files + "%s.jproj"%(self.project_name)
            if not os.path.exists(expected_project_file):
                self.app.NewProject(u"Untitled")
                self.app.SaveAs(expected_project_file)
                logger = logging.getLogger(__name__)
                logger.debug('Create JMAG project file: %s'%(expected_project_file))
            else:
                self.app.Load(expected_project_file)
                logger = logging.getLogger(__name__)
                logger.debug('Load JMAG project file: %s'%(expected_project_file))
                logger.debug('Existing models of %d are found in %s', self.app.NumModels(), self.app.GetDefaultModelFolderPath())

                if self.app.NumModels() <= individual_index:
                    logger.warn('Some models are not plotted because of bad bounds (some lower bound is too small)! individual_index=%d, NumModels()=%d. See also the fit#%04d.txt file for 99999. There will be no .png file for these individuals either.', individual_index, app.NumModels(), self.number_current_generation)

                # print self.app.NumStudies()
                # print self.app.NumAnalysisGroups()
                # self.app.SubmitAllModelsLocal() # we'd better do it one by one for easing the programing?

        # draw the model in JMAG Designer
        DRAW_SUCCESS = self.draw_model( individual_index, 
                                        im_variant,
                                        im_variant.model_name)
        self.jmag_control_state = True # indicating that the jmag project is already created
        if DRAW_SUCCESS == 0:
            # TODO: skip this model and its evaluation
            cost_function = 99999
            logging.getLogger(__name__).warn('Draw Failed for'+'%s-%s: %g', self.project_name, im_variant.model_name, cost_function)
            return cost_function

        elif DRAW_SUCCESS == -1:
            # The model already exists
            print 'Model Already Exists'
            logging.getLogger(__name__).debug('Model Already Exists')

        # Tip: 在JMAG Designer中DEBUG的时候，删掉模型，必须要手动save一下，否则再运行脚本重新load project的话，是没有删除成功的，只是删掉了model的name，新导入进来的model name与project name一致。

        # add study
        # remember to export the B data using subroutine 
        # and check export table results only
        if self.app.NumModels()>=1:
            model = self.app.GetModel(im_variant.model_name)
        else:
            logging.getLogger(__name__).error('there is no model yet!')
            print 'why is there no model yet?'

        if model.NumStudies() == 0:
            study = im_variant.add_study(self.app, model, self.dir_csv_output_folder, choose_study_type='frequency')
        else:
            # there is already a study. then get the first study.
            study = model.GetStudy(0)

        if study.AnyCaseHasResult():
            slip_freq_breakdown_torque, breakdown_torque, breakdown_force = self.check_csv_results(study.GetName())
        else:
            # mesh
            im_variant.add_mesh(study, model)

            # Export Image
                # for i in range(self.app.NumModels()):
                #     self.app.SetCurrentModel(i)
                #     model = self.app.GetCurrentModel()
                #     self.app.ExportImage(r'D:\Users\horyc\OneDrive - UW-Madison\pop\run#10/' + model.GetName() + '.png')
            self.app.View().ShowAllAirRegions()
            # self.app.View().ShowMeshGeometry() # 2nd btn
            self.app.View().ShowMesh() # 3rn btn
            self.app.View().Zoom(1.45)
            self.app.ExportImageWithSize(self.dir_run + model.GetName() + '.png', 4000, 4000)
            self.app.View().ShowModel() # 1st btn. close mesh view, and note that mesh data will be deleted because only ouput table results are selected.

            # run
            study.RunAllCases()
            self.app.Save()

            # evaluation based on the csv results
            try:
                slip_freq_breakdown_torque, breakdown_torque, breakdown_force = self.check_csv_results(study.GetName())
            except IOError, e:
                msg = 'CJH: The solver did not exit with results, so reading the csv files reports an IO error. It is highly possible that some lower bound is too small.'
                logger = logging.getLogger(__name__)
                logger.error(msg + self.im_variant.show(toString=True))
                print msg
                # raise e
                breakdown_torque = 0
                breakdown_force = 0

            # self.fitness_in_physics_data # torque density, torque ripple, force density, force magnitude error, force angle error, efficiency, material cost 

            # EC Rotate
            if False:
                # EC Rotate: Rotate the rotor to find the ripples in force and torque # 不关掉这些云图，跑第二个study的时候，JMAG就挂了：app.View().SetVectorView(False); app.View().SetFluxLineView(False); app.View().SetContourView(False)
                study_name = study.GetName() 
                self.app.SetCurrentStudy(study_name)
                casearray = [0 for i in range(1)]
                casearray[0] = 1
                duplicated_study_name = study_name + u"-FixFreqVaryRotCon"
                model.DuplicateStudyWithCases(study_name, duplicated_study_name, casearray)

                self.app.SetCurrentStudy(duplicated_study_name)
                study = self.app.GetCurrentStudy()
                divisions_per_slot_pitch = 10 # 24
                study.GetStep().SetValue(u"Step", divisions_per_slot_pitch) 
                study.GetStep().SetValue(u"StepType", 0)
                study.GetStep().SetValue(u"FrequencyStep", 0)
                study.GetStep().SetValue(u"Initialfrequency", slip_freq_breakdown_torque)

                    # study.GetCondition(u"RotCon").SetValue(u"MotionGroupType", 1)
                study.GetCondition(u"RotCon").SetValue(u"Displacement", + 360.0/im_variant.Qr/divisions_per_slot_pitch)

                # model.RestoreCadLink()
                study.Run()
                self.app.Save()
                # model.CloseCadLink()


        # compute the fitness 
        # self.fobj_test()
        rotor_volume = pi*(im_variant.Radius_OuterRotor*1e-3)**2 * (im_variant.stack_length*1e-3)
        rotor_weight = rotor_volume * 8050 # steel 8,050 kg/m3. Copper/Density 8.96 g/cm³
        cost_function = 30e3 / ( breakdown_torque/rotor_volume ) \
                        + 1.0 / ( breakdown_force/rotor_weight )
        logger = logging.getLogger(__name__)
        logger.debug('%s-%s: %g', self.project_name, im_variant.model_name, cost_function)

        return cost_function

    def fobj_test(self):

        def fmodel(x, w):
            return w[0] + w[1]*x + w[2] * x**2 + w[3] * x**3 + w[4] * x**4 + w[5] * x**5
        def rmse(w):
            y_pred = fmodel(x, w)
            return np.sqrt(sum((y - y_pred)**2) / len(y))

        x = np.linspace(0, 10, 500)
        y = np.cos(x) + np.random.normal(0, 0.2, 500)

        plt.scatter(x, y)
        plt.plot(x, np.cos(x), label='cos(x)')
        plt.legend()
        plt.show()

    def de(self):
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
        iterations -= self.number_current_generation # it is seen as total number of iterations to the user
        logger = logging.getLogger(__name__)
        logger.debug('DE Configuration:\n\t' + '\n\t'.join('%.4f,%.4f'%tuple(el) for el in bounds) + '\n\t%.4f, %.4f, %d, %d' % (mut,crossp,popsize,iterations))


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
            fitness = np.asarray( [self.fobj(index, individual) for index, individual in enumerate(pop_denorm)] ) # modification #2

            # write fitness results to file for the initial pop
            with open(fitness_file, 'w') as f:
                f.write('\n'.join('%.16f'%(x) for x in fitness)) 
                # TODO: also write self.fitness_in_physics_data
        else:
            # this is a continued run. load the fitness data (the first digit of every row is fitness for an individual)
            fitness = []
            with open(fitness_file, 'r') as f: 
                read_iterator = csv_reader(f, skipinitialspace=True)
                for row in self.whole_row_reader(read_iterator):
                    if len(row)>0: # there could be empty row, since we use append mode and write down f.write('\n')
                        fitness.append(float(row[0]))

            # TODO: 文件完整性检查
            # the last iteration is not done or the popsize is incread by user after last run of optimization
            solved_popsize = len(fitness)
            if popsize > solved_popsize:
                self.jmag_control_state = False # demand to initialize the jamg designer
                fitness_part2 = np.asarray( [self.fobj(index+solved_popsize, individual) for index, individual in enumerate(pop_denorm[solved_popsize:])] ) # modification #2

                # write fitness_part2 results to file 
                with open(fitness_file, 'a') as f:
                    f.write('\n')
                    f.write('\n'.join('%.16f'%(x) for x in fitness_part2)) 
                    # TODO: also write self.fitness_in_physics_data
                fitness += fitness_part2.tolist()
                # print fitness

        # make sure fitness is an array
        fitness = np.asarray(fitness)
        best_idx = np.argmin(fitness)
        best = pop_denorm[best_idx]
        # return min_b + pop * diff, fitness, best_idx


        for i in range(iterations):

            self.number_current_generation += 1 # modification #3
            logger = logging.getLogger(__name__)
            logger.debug('iteration #%d for this run. total iteration %d.', i, self.number_current_generation) 
            # demand to initialize the jamg designer because number_current_generation has changed and a new jmag project is required.
            self.jmag_control_state = False

            for j in range(popsize): # j is the index of individual
                logger.debug('individual #%d', j) 

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

                    # get fitness value for the trial individual 
                    f = self.fobj(j, trial_denorm)

                    # write ongoing results
                    self.write_individual_fitness(f)
                    self.write_individual_norm_data(trial) # we write individual data after fitness is evaluated in order to make sure the two files are synchronized

                if f < fitness[j]:
                    fitness[j] = f
                    pop[j] = trial # greedy selection
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm

            # one generation is finished
            self.rename_onging_files(self.number_current_generation)

            # yield best, fitness[best_idx] # de verion 1
            yield min_b + pop * diff, fitness, best_idx # de verion 2

        # TODO: 跑完一轮优化以后，必须把de_config_dict和当前的代数存在文件里，否则gen文件里面存的normalized data就没有物理意义了。

    def draw_model(self, individual_index, im_variant, model_name):

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
        d = draw(im_variant) # 传递的是地址哦
        d.doc = doc
        d.ass = ass
        try:
            d.plot_shaft(u"Shaft")
            d.plot_rotorCore(u"Rotor Core")
            d.plot_statorCore(u"Stator Core")
                # ass.GetItem("Rotor Core").SetProperty(u"Visible", 1)
            d.plot_cage(u"Cage")
                # ass.GetItem("Stator Core").SetProperty(u"Visible", 1)
            d.plot_coil(u"Coil")
            # d.plot_airWithinRotorSlots(u"Air Within Rotor Slots")
        except Exception, e:
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

    def check_csv_results(self, study_name):
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

            index, breakdown_torque = get_max_and_index(l_TorCon)
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

    def write_individual_norm_data(self, trial_individual):
        with open(self.get_gen_file(self.number_current_generation, ongoing=True), 'a') as f:
            f.write('\n' + ','.join('%.16f'%(y) for y in trial_individual)) # convert 1d array to string

    def write_individual_fitness(self, fitness_scalar):
        with open(self.get_fit_file(self.number_current_generation, ongoing=True), 'a') as f:
            f.write('\n%.16f'%(fitness_scalar))
            # TODO: also write self.fitness_in_physics_data

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

    def run(self, im, run_list=[1,0,0,0,0]): # toggle solver for Freq (it is only a place holder), Tran2TSS, Freq-FFVRC, TranRef, Static
        # Settings 
        self.jmag_control_state = False # new one project one model convension

        # initialize JMAG Designer
        self.designer_init()
        app = self.app
        self.project_name = self.fea_config_dict['model_name_prefix']
        self.im.model_name = self.im.get_model_name() 
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
        DRAW_SUCCESS = self.draw_model( 0, 
                                        im,
                                        self.im.model_name)
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
            app.View().Zoom(1.45)
            app.ExportImageWithSize(self.dir_run + model.GetName() + '.png', 4000, 4000)
            app.View().ShowModel() # 1st btn. close mesh view, and note that mesh data will be deleted because only ouput table results are selected.

            # run
            study.RunAllCases()
            app.Save()

            # evaluation based on the csv results
            slip_freq_breakdown_torque, breakdown_torque, breakdown_force = self.check_csv_results(study.GetName())
            # self.fitness_in_physics_data # torque density, torque ripple, force density, force magnitude error, force angle error, efficiency, material cost 

        # this will be used for other duplicated studies
        original_study_name = study.GetName()
        self.im.update_mechanical_parameters(slip_freq_breakdown_torque, syn_freq=500.)

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
            DM = app.GetDataManager()
            DM.CreatePointArray(u"point_array/timevsdivision", u"SectionStepTable")
            refarray = [[0 for i in range(3)] for j in range(3)]
            refarray[0][0] = 0
            refarray[0][1] =    1
            refarray[0][2] =        50
            refarray[1][0] = 1.0/slip_freq_breakdown_torque
            refarray[1][1] =    32 
            refarray[1][2] =        50
            refarray[2][0] = refarray[1][0] + 1.0/self.im.DriveW_Freq
            refarray[2][1] =    48 # don't forget to modify below two places!
            refarray[2][2] =        50
            DM.GetDataSet(u"SectionStepTable").SetTable(refarray)
            study.GetStep().SetValue(u"Step", 1 + 32 + 48) # don't forget to modify here! [first place]
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

            # https://www2.jmag-international.com/support/en/pdf/JMAG-Designer_Ver.17.1_ENv3.pdf
            study.GetStudyProperties().SetValue(u"DirectSolverType", 1)

            if run_list[1] == True:
                study.RunAllCases()
                app.Save()
            else:
                pass # if the jcf file already exists, it pops a msg window
                # study.WriteAllSolidJcf(self.dir_jcf, self.im.model_name+study.GetName()+'Solid', True) # True : Outputs cases that do not have results 
                # study.WriteAllMeshJcf(self.dir_jcf, self.im.model_name+study.GetName()+'Mesh', True)

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
                DRAW_SUCCESS = self.draw_model( 1, # +1
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
                # print self.dir_csv_output_folder + im.get_model_name() + original_study_name + '_circuit_current.csv'
                # print self.dir_csv_output_folder + im.get_model_name() + original_study_name + '_circuit_current.csv'
                # print self.dir_csv_output_folder + im.get_model_name() + original_study_name + '_circuit_current.csv'
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
        app.Save()



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
            Y = [2 * el.__abs__() / L for el in y] # /L for spectrum aplitude consistent with actual signal. 2* for single-sided. abs for amplitude.
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

            # Avg_ForCon_Vector, Avg_ForCon_Magnitude, Avg_ForCon_Angle, ForCon_Angle_List, Max_ForCon_Err_Angle = self.get_force_error_angle(force_x[-range_ss:], force_y[-range_ss:])
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
            ax = axeses[1][0]; ax.plot(time_list, sfv.force_err_abs/sfv.ss_avg_force_magnitude, label=label, alpha=alpha, zorder=zorder)
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
        ax = axeses[1][0]; ax.set_xlabel('Time [s]\n(c)',fontsize=14.5); ax.set_ylabel('Normalized Force Error Magnitude [N]',fontsize=14.5)
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

class bearingless_induction_motor_design(object):

    def update_mechanical_parameters(self, slip_freq=None, syn_freq=500.):
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
        self.the_speed = + self.the_speed

        print '[Update %s]'%(self.ID), self.slip_freq_breakdown_torque, self.the_slip, self.the_speed, self.Omega, self.DriveW_Freq, self.BeariW_Freq

    def __init__(self, row, fea_config_dict, model_name_prefix='PS'):

        # introspection
        self.bool_initial_design = True
        self.flag_optimization = fea_config_dict['flag_optimization']
        self.fea_config_dict = fea_config_dict

        #01 Model Name
        self.model_name_prefix = model_name_prefix # do include 'PS' here

        #02 Pyrhonen Data
        if row != None:
            self.ID = str(int(row[0]))
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
            print 'What are you feeding me?'

        #03 Mechanical Parameters
        self.update_mechanical_parameters(slip_freq=2.75, syn_freq=500.)

        #04 Material Condutivity Properties
        self.End_Ring_Resistance = fea_config_dict['End_Ring_Resistance']
        self.Bar_Conductivity = fea_config_dict['Bar_Conductivity']
        self.Copper_Loss = self.DriveW_CurrentAmp**2 / 2 * self.DriveW_Rs * 3
        # self.Resistance_per_Turn = 0.01 # TODO


        #05 Windings & Excitation
        self.l41=[ 'C', 'C', 'A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B', ]
        self.l42=[ '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', ]
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
        self.max_nonlinear_iteration = 30 # 50 # for transient solve
        self.meshSize_Rotor = 1.2 # mm


        #07: Some Checking
        if abs(self.Location_RotorBarCenter2 - self.Location_RotorBarCenter) < self.Radius_of_RotorSlot*0.25:
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

        # during optimization, rotate at model level is not needed.
        self.MODEL_ROTATE = False

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
            raise Exception('Number of Parts is unexpected.')

        id_shaft = part_ID_list[0]
        id_rotorCore = part_ID_list[1]
        id_statorCore = part_ID_list[2]

        partIDRange_Cage = part_ID_list[3 : 3+int(self.Qr)]
        partIDRange_Coil = part_ID_list[3+int(self.Qr) : 3+int(self.Qr) + int(self.Qs*2)]
        # partIDRange_AirWithinRotorSlots = part_ID_list[3+int(self.Qr) + int(self.Qs*2) : 3+int(self.Qr) + int(self.Qs*2) + int(self.Qr)]

        # print part_ID_list
        # print partIDRange_Cage
        # print partIDRange_Coil
        # print partIDRange_AirWithinRotorSlots
        group(u"Cage", partIDRange_Cage) # 59-44 = 15 = self.Qr - 1
        group(u"Coil", partIDRange_Coil) # 107-60 = 47 = 48-1 = self.Qs*2 - 1
        # group(u"AirWithinRotorSlots", partIDRange_AirWithinRotorSlots) # 123-108 = 15 = self.Qr - 1


        ''' Add to Set for later references '''
        def part_set(name, x, y):
            model.GetSetList().CreatePartSet(name)
            model.GetSetList().GetSet(name).SetMatcherType(u"Selection")
            model.GetSetList().GetSet(name).ClearParts()
            sel = model.GetSetList().GetSet(name).GetSelection()
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

        # 4 poles Winding
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

        # 2 poles Winding
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



        # Bars and Air within rotor slots
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
            part_set(u"Bar %d"%(natural_ind), X, Y)
            list_xy_bars.append([X,Y])
            # # # part_set(u"AirWithin %d"%(natural_ind), R_airR*cos(THETA),R_airR*sin(THETA))
            # list_xy_airWithinRotorSlot.append([R_airR*cos(THETA),R_airR*sin(THETA)])

            THETA += self.Angle_RotorSlotSpan/180.*pi
            X = R*cos(THETA)
            Y = R*sin(THETA)

        # Motion
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

        # Cage
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
            study_name = u"Tran"
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
            study.GetStudyProperties().SetValue(u"CsvResultTypes", u"Torque;Force;FEMCoilFlux;LineCurrent;ElectricPower;TerminalVoltage;JouleLoss;TotalDisplacementAngle")
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
            study_name = u"Freq"
            model.CreateStudy(u"Frequency2D", study_name)
            app.SetCurrentStudy(study_name)
            study = model.GetStudy(study_name)

            study.GetStudyProperties().SetValue(u"ModelThickness", self.stack_length) # Stack Length
            study.GetStudyProperties().SetValue(u"ConversionType", 0)
            study.GetStudyProperties().SetValue(u"CsvOutputPath", dir_csv_output_folder) # it's folder rather than file!
            study.GetStudyProperties().SetValue(u"CsvResultTypes", u"Torque;Force;FEMCoilFlux;LineCurrent;ElectricPower;TerminalVoltage;JouleLoss;TotalDisplacementAngle")
            study.GetStudyProperties().SetValue(u"NonlinearMaxIteration", self.max_nonlinear_iteration)

            DM = app.GetDataManager()
            DM.CreatePointArray(u"point_array/frequency_vs_division", u"table_freq_division")
            # DM.GetDataSet(u"").SetName(u"table_freq_division")
            DM.GetDataSet(u"table_freq_division").SetTable(self.table_freq_division_refarray)
            study.GetStep().SetValue(u"Step", self.no_steps)
            study.GetStep().SetValue(u"StepType", 3)
            study.GetStep().SetTableProperty(u"Division", DM.GetDataSet(u"table_freq_division"))
            # app.View().SetCurrentCase(1)
            print 'BHCorrection for nonlinear time harmonic analysis is turned ON.'
            study.GetStudyProperties().SetValue(u"BHCorrection", 1)
        else:
            study_name = u"Static"
            model.CreateStudy(u"Static2D", study_name)
            app.SetCurrentStudy(study_name)
            study = model.GetStudy(study_name)

            study.GetStudyProperties().SetValue(u"ModelThickness", self.stack_length) # Stack Length
            study.GetStudyProperties().SetValue(u"ConversionType", 0)
            study.GetStudyProperties().SetValue(u"CsvOutputPath", dir_csv_output_folder) # it's folder rather than file!
            study.GetStudyProperties().SetValue(u"CsvResultTypes", u"Torque;Force;FEMCoilFlux;LineCurrent;ElectricPower;TerminalVoltage;JouleLoss;TotalDisplacementAngle")
            study.GetStudyProperties().SetValue(u"NonlinearMaxIteration", self.max_nonlinear_iteration)

        # Material
        # self.add_Arnon5()
        study.SetMaterialByName(u"Stator Core", u"M-15 Steel")
        # study.GetMaterial(u"Stator Core").SetValue(u"UserConductivityValue", 1900000)
        # study.GetMaterial(u"Stator Core").SetValue(u"Laminated", 1)
        # study.GetMaterial(u"Stator Core").SetValue(u"LaminationFactor", 98)

        study.SetMaterialByName(u"Rotor Core", u"M-15 Steel")
        # study.GetMaterial(u"Rotor Core").SetValue(u"UserConductivityValue", 1900000)
        # study.GetMaterial(u"Rotor Core").SetValue(u"Laminated", 1)
        # study.GetMaterial(u"Rotor Core").SetValue(u"LaminationFactor", 98)

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



        # Conditions - FEM Coils
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
            for ABC in [u'A',u'B',u'C']:
                which_phase = u"%d%s-Phase"%(poles,ABC)
                study.CreateCondition(u"FEMCoil", which_phase)
                condition = study.GetCondition(which_phase)
                condition.SetLink(u"Coil%d%s"%(poles,ABC))
                condition.GetSubCondition(u"untitled").SetName(u"Coil Set 1")
                condition.GetSubCondition(u"Coil Set 1").SetName(u"delete")
            count = 0
            d = {'+':1, '-':0}
            for ABC, UpDown in zip(l1,l2):
                which_phase = u"%d%s-Phase"%(poles,ABC)
                count += 1 
                condition = study.GetCondition(which_phase)
                condition.CreateSubCondition(u"FEMCoilData", u"Coil Set %d"%(count))
                subcondition = condition.GetSubCondition(u"Coil Set %d"%(count))
                subcondition.ClearParts()
                subcondition.AddSet(model.GetSetList().GetSet(u"Coil%d%s%s %d"%(poles,ABC,UpDown,count)), 0)
                subcondition.SetValue(u"Direction2D", d[UpDown])
            for ABC in [u'A',u'B',u'C']:
                which_phase = u"%d%s-Phase"%(poles,ABC)
                condition = study.GetCondition(which_phase)
                condition.RemoveSubCondition(u"delete")
        link_FEMCoils_2_CoilSet(self.DriveW_poles, 
                                self.dict_coil_connection[int(self.DriveW_poles*10+1)], 
                                self.dict_coil_connection[int(self.DriveW_poles*10+2)])
        link_FEMCoils_2_CoilSet(self.BeariW_poles, 
                                self.dict_coil_connection[int(self.BeariW_poles*10+1)], 
                                self.dict_coil_connection[int(self.BeariW_poles*10+2)])




        # Condition - Conductor
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

        # Link to Circuit
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

        # no mesh results are needed
        study.GetStudyProperties().SetValue(u"OnlyTableResults", self.fea_config_dict['OnlyTableResults'])

        # Linear Solver
        if False:
            # sometime nonlinear iteration is reported to fail and recommend to increase the accerlation rate of ICCG solver
            study.GetStudyProperties().SetValue(u"IccgAccel", 1.2) 
            study.GetStudyProperties().SetValue(u"AutoAccel", 0)
        else:
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
        study.GetMeshControl().SetValue(u"CircumferentialDivision", 1440) # for air region near which motion occurs 这个数足够大，sliding mesh才准确。
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

    def get_model_name(self):
        return u"%s_ID%s" % (self.model_name_prefix, self.ID)

class local_design_variant(bearingless_induction_motor_design):

    def get_stator_yoke_diameter_Dsyi(self, stator_tooth_width_b_ds, area_stator_slot_Sus, stator_inner_radius_r_s, Qs):
        stator_inner_diameter_Ds = 2*stator_inner_radius_r_s
        temp = (2*pi*stator_inner_radius_r_s - Qs*stator_tooth_width_b_ds)
        stator_tooth_height_h_ds = ( np.sqrt(temp**2 + 4*area_stator_slot_Sus*Qs) - temp ) / (2*pi)
        stator_yoke_diameter_Dsyi = stator_inner_diameter_Ds + 2*stator_tooth_height_h_ds
        return stator_yoke_diameter_Dsyi

    def get_rotor_tooth_height_h_dr(self, rotor_tooth_width_b_dr, area_rotor_slot_Sur, rotor_outer_radius_r_or, Qr):
        temp = (2*pi*rotor_outer_radius_r_or - Qr*rotor_tooth_width_b_dr)
        rotor_tooth_height_h_dr = ( -np.sqrt(temp**2 - 4*area_rotor_slot_Sur*Qr) + temp ) / (2*pi)
        return rotor_tooth_height_h_dr

    def __init__(self, im, number_current_generation, individual_index, design_parameters):

        # introspection
        self.bool_initial_design = False
        self.flag_optimization = im.fea_config_dict['flag_optimization']

        # unpacking
        stator_tooth_width_b_ds       = design_parameters[0]*1e-3 # m
        air_gap_length_delta          = design_parameters[1]*1e-3 # m
        b1                            = design_parameters[2]*1e-3 # m
        rotor_tooth_width_b_dr        = design_parameters[3]*1e-3 # m

        self.Length_HeadNeckRotorSlot = design_parameters[4]
        rotor_slot_radius = (2*pi*(im.Radius_OuterRotor - self.Length_HeadNeckRotorSlot)*1e-3 - rotor_tooth_width_b_dr*im.Qr) / (2*im.Qr+2*pi)

        # Constraint #1: Rotor slot opening cannot be larger than rotor slot width.
        self.punishment = 0.0
        if b1>2*rotor_slot_radius:
            logger = logging.getLogger(__name__)
            logger.warn('Constraint #1: Rotor slot opening cannot be larger than rotor slot width. Gen#%04d. Individual index=%d.', number_current_generation, individual_index)
            # we will plot a model with b1 = rotor_tooth_width_b_dr instead, and apply a punishment for this model
            b1 = 0.95 * 2*rotor_slot_radius # 确保相交
            self.punishment = 0.0

        # stator_tooth_width_b_ds imposes constraint on stator slot height
        area_stator_slot_Sus    = im.parameters_for_imposing_constraints_among_design_parameters[0]
        stator_inner_radius_r_s = im.Radius_OuterRotor*1e-3 + air_gap_length_delta 
        stator_yoke_diameter_Dsyi = self.get_stator_yoke_diameter_Dsyi(  stator_tooth_width_b_ds, 
                                                                    area_stator_slot_Sus, 
                                                                    stator_inner_radius_r_s,
                                                                    im.Qs)

        # rotor_tooth_width_b_dr imposes constraint on rotor slot height
        area_rotor_slot_Sur = im.parameters_for_imposing_constraints_among_design_parameters[1]
        rotor_outer_radius_r_or = im.Radius_OuterRotor*1e-3
        rotor_tooth_height_h_dr = self.get_rotor_tooth_height_h_dr(  rotor_tooth_width_b_dr,
                                                                area_rotor_slot_Sur,
                                                                rotor_outer_radius_r_or,
                                                                im.Qr)

        #01 Model Name
        self.model_name_prefix = im.model_name_prefix

        #02 Pyrhonen Data
        if im != None:
            self.ID = im.ID + '-' + str(number_current_generation) + '-' + str(individual_index) # the ID is str
            self.Qs = im.Qs
            self.Qr = im.Qr
            self.Angle_StatorSlotSpan = 360. / self.Qs # in deg.
            self.Angle_RotorSlotSpan  = 360. / self.Qr # in deg.

            self.Radius_OuterStatorYoke = im.Radius_OuterStatorYoke
            self.Radius_InnerStatorYoke = 0.5*stator_yoke_diameter_Dsyi * 1e3   # [0]            # 定子内轭部处的半径由需要的定子槽面积和定子齿宽来决定。
            self.Length_AirGap          = air_gap_length_delta * 1e3            # [1]
            self.Radius_OuterRotor      = im.Radius_OuterRotor
            self.Radius_Shaft           = im.Radius_Shaft

            self.Length_HeadNeckRotorSlot                                          # [4]
            self.Radius_of_RotorSlot      = rotor_slot_radius*1e3               # [3]
            self.Location_RotorBarCenter  = self.Radius_OuterRotor - self.Length_HeadNeckRotorSlot - self.Radius_of_RotorSlot
            self.Width_RotorSlotOpen      = b1*1e3                              # [2]

            Arc_betweenOuterRotorSlot     = 360./self.Qr*pi/180.*self.Location_RotorBarCenter - 2*self.Radius_of_RotorSlot
            self.Location_RotorBarCenter2 = self.Radius_OuterRotor - self.Length_HeadNeckRotorSlot - rotor_tooth_height_h_dr*1e3 # 本来还应该减去转子内槽半径的，但是这里还不知道，干脆不要了，这样槽会比预计的偏深，
            self.Radius_of_RotorSlot2     = 0.5 * (360./self.Qr*pi/180.*self.Location_RotorBarCenter2 - Arc_betweenOuterRotorSlot) # 应该小于等于这个值，保证转子齿等宽。

            self.Angle_StatorSlotOpen            = design_parameters[5]             # [5]
            self.Width_StatorTeethBody           = stator_tooth_width_b_ds *1e3 # [0]
            self.Width_StatorTeethHeadThickness  = design_parameters[6]             # [6]
            self.Width_StatorTeethNeck           = 0.5 * self.Width_StatorTeethHeadThickness

            self.DriveW_poles       = im.DriveW_poles     
            self.DriveW_turns       = im.DriveW_turns # turns per slot
            self.DriveW_Rs          = im.DriveW_Rs        
            self.DriveW_CurrentAmp  = im.DriveW_CurrentAmp
            self.DriveW_Freq        = im.DriveW_Freq      

            self.stack_length       = im.stack_length

            # inferred design parameters
        else:
            print 'What are you feeding me?'

        #03 Mechanical Parameters
        self.update_mechanical_parameters(slip_freq=2.75, syn_freq=500.)

        #04 Material Condutivity Properties
        self.End_Ring_Resistance = 9.69e-6 # this may too small for Chiba's winding
        self.Bar_Conductivity = 40000000 #
        self.Copper_Loss = self.DriveW_CurrentAmp**2 / 2 * self.DriveW_Rs * 3
        # self.Resistance_per_Turn = 0.01 # TODO


        #05 Windings & Excitation
        self.l41=[ 'C', 'C', 'A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B', ]
        self.l42=[ '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', ]
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
        self.max_nonlinear_iteration = 30 # 50 # for transient solve
        self.meshSize_Rotor = 0.6 # mm


        #07: Some Checking
        if abs(self.Location_RotorBarCenter2 - self.Location_RotorBarCenter) < self.Radius_of_RotorSlot*0.25:
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

        # during optimization, rotate at model level is not needed.
        self.MODEL_ROTATE = False

        logger = logging.getLogger(__name__) 
        logger.info('im_variant ID %s is initialized.', self.ID)

class draw(object):

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
        SketchName = self.SketchName
        self.sketch.CreateVertex(x, y)
        # return self.circle(x, y, r)
        return self.sketch.CreateCircle(x, y, r)

    def line(self, x1,y1,x2,y2):
        SketchName = self.SketchName
        self.sketch.CreateVertex(x1,y1)
        self.sketch.CreateVertex(x2,y2)
        # return self.line(x1,y1,x2,y2)
        return self.sketch.CreateLine(x1,y1,x2,y2)

    def trim_l(self, who,x,y):
        SketchName = self.SketchName
        self.doc.GetSelection().Clear()
        ref1 = self.sketch.GetItem(who.GetName())
        self.doc.GetSelection().Add(ref1)
        self.doc.GetSketchManager().SketchTrim(x,y)
        # l1 trim å®Œä»\åŽè¿˜æ˜¯l1ï¼Œé™¤éžä½ åˆ‡ä¸­é—´ï¼Œè¿™æ ·ä¼šå¤šç”Ÿæˆä¸€ä¸ªLineï¼Œä½ è‡ªå·±æ•æ‰ä¸€ä¸‹å§

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

        # if merge == False:

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
        self.trim_l(l1,-5.-self.im.Radius_OuterRotor, 0.5*self.im.Width_RotorSlotOpen)
        self.trim_l(l1,-self.im.Location_RotorBarCenter, 0.5*self.im.Width_RotorSlotOpen)

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
            self.constraint_fixture_circle_center(c4)


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
        sketch.CreateVertex(0, 0)
        c1=self.circle(0, 0, self.im.Radius_InnerStator)
        c2=self.circle(0, 0, self.im.Radius_InnerStatorYoke)
        c3=self.circle(0, 0, self.im.Radius_OuterStatorYoke)
        c4=self.circle(0, 0, self.im.Radius_InnerStator + self.im.Width_StatorTeethHeadThickness)

        l1=self.line(-self.im.Radius_OuterStatorYoke, 0, 0, 0)
        l2=self.line(-0.5*(self.im.Radius_OuterStatorYoke+self.im.Radius_InnerStatorYoke), 0.5*self.im.Width_StatorTeethBody, \
                -(self.im.Radius_InnerStator + (self.im.Width_StatorTeethHeadThickness+self.im.Width_StatorTeethNeck)), 0.5*self.im.Width_StatorTeethBody) # legacy 1. NEW: add neck here. Approximation is adopted: the accurate results should be (self.im.Width_StatorTeethHeadThickness+self.im.Width_StatorTeethNeck) * cos(6 deg) or so. 6 deg = 360/24/2 - 3/2

        R = self.im.Radius_OuterStatorYoke
        THETA = (180-0.5*360/self.im.Qs)/180.*pi
        X = R*cos(THETA)
        Y = R*sin(THETA)
        l3 = StatorCore_Line_3 = self.line(0, 0, X, Y) # Line.3

        R = self.im.Radius_InnerStator + self.im.Width_StatorTeethHeadThickness
        THETA = (180-(0.5*360/self.im.Qs-0.5*self.im.Angle_StatorSlotOpen))/180.*pi # BUG is found here, 3 is instead used for Angle_StatorSlotOpen
        X = R*cos(THETA)
        Y = R*sin(THETA)
        l4 = self.line(0, 0, X, Y) # Line.4




        # Trim Arcs

        # ä¸ºäº†é¿å…æ•°Arcï¼Œæˆ‘ä»¬åº”è¯\åœ¨Circle.4æ—¶æœŸå°±æŠŠCircle.4ä¸Šçš„å°ç‰‡æ®µä¼˜å…ˆå‰ªäº†ï¼
        arc4 = self.trim_c(c4, X+1e-6, Y+1e-6) # Arc.1

        self.trim_a(arc4, 0, self.im.Radius_InnerStator + self.im.Width_StatorTeethHeadThickness)
        self.trim_a(arc4, -(self.im.Radius_InnerStator + self.im.Width_StatorTeethHeadThickness), 1e-6)

        arc1 = self.trim_c(c1, 0, self.im.Radius_InnerStator) # self.trim_c(c1,0, self.im.Radius_InnerStator)

        R = self.im.Radius_InnerStator
        THETA = (180-0.5*360/self.im.Qs)/180.*pi
        X = R*cos(THETA)
        Y = R*sin(THETA)
        self.trim_a(arc1, X, Y-1e-6)

        # ä¸ºäº†é¿å…æ•°Arcï¼Œæˆ‘ä»¬åº”è¯\åœ¨Circle.2æ—¶æœŸå°±æŠŠCircle.2ä¸Šçš„å°ç‰‡æ®µä¼˜å…ˆå‰ªäº†ï¼
        arc2 = self.trim_c(c2,-self.im.Radius_InnerStatorYoke, 0.25*self.im.Width_StatorTeethBody)

        self.trim_a(arc2, 0, self.im.Radius_InnerStatorYoke)

        self.trim_c(c3,0, self.im.Radius_OuterStatorYoke)

        self.trim_l(l1,-0.1, 0)


        self.trim_l(l2,1e-6 -0.5*(self.im.Radius_OuterStatorYoke+self.im.Radius_InnerStatorYoke), 0.5*self.im.Width_StatorTeethBody)
            # self.trim_l(l2,1e-6 -1e-6 -(self.im.Radius_InnerStator + 0.5*(self.im.Width_StatorTeethHeadThickness+self.im.Width_StatorTeethNeck)), 0.5*self.im.Width_StatorTeethBody)

        self.trim_l(l3,-1e-6, 1e-6)

        self.trim_l(l4,-1e-6, 1e-6) # float number



        # we forget to plot the neck of stator tooth
        l2_end_vertex = l2.GetEndVertex()
        X = l2_end_vertex.GetX()
        Y = l2_end_vertex.GetY()
            # l5_start_vertex = sketch.CreateVertex(X - self.im.Width_StatorTeethNeck, Y) # legacy 2
        l5_start_vertex = sketch.CreateVertex(X, Y) 
        self.l5_start_vertex_x = l5_start_vertex.GetX()
        self.l5_start_vertex_y = l5_start_vertex.GetY()

        arc4_start_vertex = arc4.GetStartVertex()
        X = arc4_start_vertex.GetX()
        Y = arc4_start_vertex.GetY()
        l5_end_vertex = sketch.CreateVertex(X, Y) 

        l5 = self.line(self.l5_start_vertex_x, self.l5_start_vertex_y, l5_end_vertex.GetX(), l5_end_vertex.GetY()) # Line.5

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

        c3=self.circle(-self.im.Location_RotorBarCenter, 0, self.im.Radius_of_RotorSlot)

        if self.im.use_drop_shape_rotor_bar == True:
            # the inner rotor slot for drop shape rotor suggested by Gerada2011
            c4=self.circle(-self.im.Location_RotorBarCenter2, 0, self.im.Radius_of_RotorSlot2) # Circle.4
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


        # æ–¹æ¡ˆä¸€ ç”»ä¸­çº¿ï¼Œç„¶åŽé•œåƒåˆ°å¯¹é¢ï¼Œå¾—åˆ°å¦å¤–ä¸¤ä¸ªç‚¹ï¼
        # æ–¹æ¡ˆäºŒ ä¸ºä»€ä¹ˆä¸ç›´æŽ\ç”»åœ†ç„¶åŽTrimå“¦ï¼
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

import operator
def get_max_and_index(the_list):
    return max(enumerate(the_list), key=operator.itemgetter(1))


def add_M15Steel(app):
    app.GetMaterialLibrary().CreateCustomMaterial(u"M-15 Steel", u"Custom Materials")
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"Density", 7.85)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"MagneticSteelPermeabilityType", 2)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"CoerciveForce", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").GetTable("BhTable").SetName(u"Untitled")
    refarray = [[0 for i in range(2)] for j in range(47)]
    refarray[0][0] = 0
    refarray[0][1] = 0
    refarray[1][0] = 15.120714
    refarray[1][1] = 0.05
    refarray[2][0] = 22.718292
    refarray[2][1] = 0.1
    refarray[3][0] = 27.842733
    refarray[3][1] = 0.15
    refarray[4][0] = 31.871434
    refarray[4][1] = 0.2
    refarray[5][0] = 35.365044
    refarray[5][1] = 0.25
    refarray[6][0] = 38.600588
    refarray[6][1] = 0.3
    refarray[7][0] = 41.736202
    refarray[7][1] = 0.35
    refarray[8][0] = 44.873979
    refarray[8][1] = 0.4
    refarray[9][0] = 48.087807
    refarray[9][1] = 0.45
    refarray[10][0] = 51.437236
    refarray[10][1] = 0.5
    refarray[11][0] = 54.975221
    refarray[11][1] = 0.55
    refarray[12][0] = 58.752993
    refarray[12][1] = 0.6
    refarray[13][0] = 62.823644
    refarray[13][1] = 0.65
    refarray[14][0] = 67.245285
    refarray[14][1] = 0.7
    refarray[15][0] = 72.084406
    refarray[15][1] = 0.75
    refarray[16][0] = 77.4201
    refarray[16][1] = 0.8
    refarray[17][0] = 83.350021
    refarray[17][1] = 0.85
    refarray[18][0] = 89.999612
    refarray[18][1] = 0.9
    refarray[19][0] = 97.537353
    refarray[19][1] = 0.95
    refarray[20][0] = 106.201406
    refarray[20][1] = 1
    refarray[21][0] = 116.348464
    refarray[21][1] = 1.05
    refarray[22][0] = 128.547329
    refarray[22][1] = 1.1
    refarray[23][0] = 143.765431
    refarray[23][1] = 1.15
    refarray[24][0] = 163.754169
    refarray[24][1] = 1.2
    refarray[25][0] = 191.868158
    refarray[25][1] = 1.25
    refarray[26][0] = 234.833507
    refarray[26][1] = 1.3
    refarray[27][0] = 306.509769
    refarray[27][1] = 1.35
    refarray[28][0] = 435.255202
    refarray[28][1] = 1.4
    refarray[29][0] = 674.911968
    refarray[29][1] = 1.45
    refarray[30][0] = 1108.325569
    refarray[30][1] = 1.5
    refarray[31][0] = 1813.085468
    refarray[31][1] = 1.55
    refarray[32][0] = 2801.217421
    refarray[32][1] = 1.6
    refarray[33][0] = 4053.653117
    refarray[33][1] = 1.65
    refarray[34][0] = 5591.10689
    refarray[34][1] = 1.7
    refarray[35][0] = 7448.318413
    refarray[35][1] = 1.75
    refarray[36][0] = 9708.81567
    refarray[36][1] = 1.8
    refarray[37][0] = 12486.931615
    refarray[37][1] = 1.85
    refarray[38][0] = 16041.483644
    refarray[38][1] = 1.9
    refarray[39][0] = 21249.420624
    refarray[39][1] = 1.95
    refarray[40][0] = 31313.495878
    refarray[40][1] = 2
    refarray[41][0] = 53589.446877
    refarray[41][1] = 2.05
    refarray[42][0] = 88477.484601
    refarray[42][1] = 2.1
    refarray[43][0] = 124329.41054
    refarray[43][1] = 2.15
    refarray[44][0] = 159968.5693
    refarray[44][1] = 2.2
    refarray[45][0] = 197751.604272
    refarray[45][1] = 2.25
    refarray[46][0] = 234024.751347
    refarray[46][1] = 2.3
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").GetTable("BhTable").SetTable(refarray)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"DemagnetizationCoerciveForce", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"MagnetizationSaturated", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"MagnetizationSaturated2", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"YoungModulus", 210000)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"ShearModulus", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"YoungModulusX", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"YoungModulusY", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"YoungModulusZ", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"ShearModulusXY", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"ShearModulusYZ", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"ShearModulusZX", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G11", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G12", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G13", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G14", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G15", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G16", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G22", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G23", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G24", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G25", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G26", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G33", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G34", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G35", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G36", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G44", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G45", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G46", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G55", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G56", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"M-15 Steel").SetValue(u"G66", 0)

def add_Arnon5(app):
    app.GetMaterialLibrary().CreateCustomMaterial(u"Arnon5", u"Custom Materials")
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"Density", 7.85)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"MagneticSteelPermeabilityType", 2)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"CoerciveForce", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").GetTable("BhTable").SetName(u"SmoothZeroPointOne")
    refarray = [[0 for i in range(2)] for j in range(225)]
    refarray[0][0] = 0
    refarray[0][1] = 0
    refarray[1][0] = 2.37757675
    refarray[1][1] = 0.01935985516
    refarray[2][0] = 4.7551535
    refarray[2][1] = 0.0387150791541
    refarray[3][0] = 7.13273025
    refarray[3][1] = 0.0580610408164
    refarray[4][0] = 9.510307
    refarray[4][1] = 0.0773931089808
    refarray[5][0] = 9.93584775
    refarray[5][1] = 0.0808513585492
    refarray[6][0] = 10.3613885
    refarray[6][1] = 0.0843089881429
    refarray[7][0] = 10.78692925
    refarray[7][1] = 0.0877659712091
    refarray[8][0] = 11.21247
    refarray[8][1] = 0.0912222811951
    refarray[9][0] = 11.714206
    refarray[9][1] = 0.0952965603864
    refarray[10][0] = 12.215942
    refarray[10][1] = 0.0993698234457
    refarray[11][0] = 12.717678
    refarray[11][1] = 0.103442026851
    refarray[12][0] = 13.219414
    refarray[12][1] = 0.107513127079
    refarray[13][0] = 13.81087375
    refarray[13][1] = 0.112310772421
    refarray[14][0] = 14.4023335
    refarray[14][1] = 0.117106752976
    refarray[15][0] = 14.99379325
    refarray[15][1] = 0.121900997451
    refarray[16][0] = 15.585253
    refarray[16][1] = 0.12669343455
    refarray[17][0] = 16.28175525
    refarray[17][1] = 0.132334588201
    refarray[18][0] = 16.9782575
    refarray[18][1] = 0.137973020198
    refarray[19][0] = 17.67475975
    refarray[19][1] = 0.143608614113
    refarray[20][0] = 18.371262
    refarray[20][1] = 0.149241253519
    refarray[21][0] = 19.19250175
    refarray[21][1] = 0.155878695613
    refarray[22][0] = 20.0137415
    refarray[22][1] = 0.162511677467
    refarray[23][0] = 20.83498125
    refarray[23][1] = 0.169140008229
    refarray[24][0] = 21.656221
    refarray[24][1] = 0.175763497047
    refarray[25][0] = 22.62249075
    refarray[25][1] = 0.18355022757
    refarray[26][0] = 23.5887605
    refarray[26][1] = 0.191329679895
    refarray[27][0] = 24.55503025
    refarray[27][1] = 0.19910154315
    refarray[28][0] = 25.5213
    refarray[28][1] = 0.206865506461
    refarray[29][0] = 26.656473
    refarray[29][1] = 0.215976092655
    refarray[30][0] = 27.791646
    refarray[30][1] = 0.225074842617
    refarray[31][0] = 28.926819
    refarray[31][1] = 0.234161252296
    refarray[32][0] = 30.061992
    refarray[32][1] = 0.243234817645
    refarray[33][0] = 31.38755425
    refarray[33][1] = 0.253813255725
    refarray[34][0] = 32.7131165
    refarray[34][1] = 0.264372689832
    refarray[35][0] = 34.03867875
    refarray[35][1] = 0.274912317386
    refarray[36][0] = 35.364241
    refarray[36][1] = 0.28543133581
    refarray[37][0] = 36.88078925
    refarray[37][1] = 0.297439617704
    refarray[38][0] = 38.3973375
    refarray[38][1] = 0.30941867158
    refarray[39][0] = 39.91388575
    refarray[39][1] = 0.321367295572
    refarray[40][0] = 41.430434
    refarray[40][1] = 0.333284287818
    refarray[41][0] = 43.16940125
    refarray[41][1] = 0.346908559951
    refarray[42][0] = 44.9083685
    refarray[42][1] = 0.360487849383
    refarray[43][0] = 46.64733575
    refarray[43][1] = 0.374020344101
    refarray[44][0] = 48.386303
    refarray[44][1] = 0.387504232095
    refarray[45][0] = 50.41731975
    refarray[45][1] = 0.403188701653
    refarray[46][0] = 52.4483365
    refarray[46][1] = 0.418801508461
    refarray[47][0] = 54.47935325
    refarray[47][1] = 0.434339765649
    refarray[48][0] = 56.51037
    refarray[48][1] = 0.449800586347
    refarray[49][0] = 58.8992865
    refarray[49][1] = 0.467882848007
    refarray[50][0] = 61.288203
    refarray[50][1] = 0.485849285635
    refarray[51][0] = 63.6771195
    refarray[51][1] = 0.503695201484
    refarray[52][0] = 66.066036
    refarray[52][1] = 0.521415897806
    refarray[53][0] = 68.884671
    refarray[53][1] = 0.542156758805
    refarray[54][0] = 71.703306
    refarray[54][1] = 0.562709043563
    refarray[55][0] = 74.521941
    refarray[55][1] = 0.583065035883
    refarray[56][0] = 77.340576
    refarray[56][1] = 0.603217019571
    refarray[57][0] = 80.6531885
    refarray[57][1] = 0.626629511207
    refarray[58][0] = 83.965801
    refarray[58][1] = 0.64973703818
    refarray[59][0] = 87.2784135
    refarray[59][1] = 0.672527074887
    refarray[60][0] = 90.591026
    refarray[60][1] = 0.694987095726
    refarray[61][0] = 94.49629175
    refarray[61][1] = 0.721024488413
    refarray[62][0] = 98.4015575
    refarray[62][1] = 0.746565285739
    refarray[63][0] = 102.30682325
    refarray[63][1] = 0.771588964817
    refarray[64][0] = 106.212089
    refarray[64][1] = 0.796075002757
    refarray[65][0] = 110.80768975
    refarray[65][1] = 0.824173011219
    refarray[66][0] = 115.4032905
    refarray[66][1] = 0.851464637003
    refarray[67][0] = 119.99889125
    refarray[67][1] = 0.877916436465
    refarray[68][0] = 124.594492
    refarray[68][1] = 0.903494965963
    refarray[69][0] = 130.02366675
    refarray[69][1] = 0.932185586732
    refarray[70][0] = 135.4528415
    refarray[70][1] = 0.95878007922
    refarray[71][0] = 140.88201625
    refarray[71][1] = 0.983129665755
    refarray[72][0] = 146.311191
    refarray[72][1] = 1.00508556866
    refarray[73][0] = 152.74901075
    refarray[73][1] = 1.02882602048
    refarray[74][0] = 159.1868305
    refarray[74][1] = 1.05087178237
    refarray[75][0] = 165.62465025
    refarray[75][1] = 1.07113023252
    refarray[76][0] = 172.06247
    refarray[76][1] = 1.08950874915
    refarray[77][0] = 179.67803675
    refarray[77][1] = 1.10935866849
    refarray[78][0] = 187.2936035
    refarray[78][1] = 1.12767240346
    refarray[79][0] = 194.90917025
    refarray[79][1] = 1.14437272738
    refarray[80][0] = 202.524737
    refarray[80][1] = 1.15938241357
    refarray[81][0] = 211.52495225
    refarray[81][1] = 1.17548394314
    refarray[82][0] = 220.5251675
    refarray[82][1] = 1.19032895172
    refarray[83][0] = 229.52538275
    refarray[83][1] = 1.20385592172
    refarray[84][0] = 238.525598
    refarray[84][1] = 1.21600333556
    refarray[85][0] = 249.147205
    refarray[85][1] = 1.22890940527
    refarray[86][0] = 259.768812
    refarray[86][1] = 1.24059318179
    refarray[87][0] = 270.390419
    refarray[87][1] = 1.25099561414
    refarray[88][0] = 281.012026
    refarray[88][1] = 1.2600576513
    refarray[89][0] = 293.52359825
    refarray[89][1] = 1.2696124748
    refarray[90][0] = 306.0351705
    refarray[90][1] = 1.27853736884
    refarray[91][0] = 318.54674275
    refarray[91][1] = 1.28680855779
    refarray[92][0] = 331.058315
    refarray[92][1] = 1.29440226601
    refarray[93][0] = 345.8298885
    refarray[93][1] = 1.30267543351
    refarray[94][0] = 360.601462
    refarray[94][1] = 1.31037112631
    refarray[95][0] = 375.3730355
    refarray[95][1] = 1.31747192026
    refarray[96][0] = 390.144609
    refarray[96][1] = 1.32396039123
    refarray[97][0] = 407.53229275
    refarray[97][1] = 1.33097162343
    refarray[98][0] = 424.9199765
    refarray[98][1] = 1.33745669854
    refarray[99][0] = 442.30766025
    refarray[99][1] = 1.34340384485
    refarray[100][0] = 459.695344
    refarray[100][1] = 1.34880129065
    refarray[101][0] = 480.20445525
    refarray[101][1] = 1.35472028763
    refarray[102][0] = 500.7135665
    refarray[102][1] = 1.36040050502
    refarray[103][0] = 521.22267775
    refarray[103][1] = 1.36583950086
    refarray[104][0] = 541.731789
    refarray[104][1] = 1.37103483318
    refarray[105][0] = 565.90146525
    refarray[105][1] = 1.3769055887
    refarray[106][0] = 590.0711415
    refarray[106][1] = 1.38256086043
    refarray[107][0] = 614.24081775
    refarray[107][1] = 1.38799940845
    refarray[108][0] = 638.410494
    refarray[108][1] = 1.39321999281
    refarray[109][0] = 666.89128125
    refarray[109][1] = 1.39897356572
    refarray[110][0] = 695.3720685
    refarray[110][1] = 1.40418332863
    refarray[111][0] = 723.85285575
    refarray[111][1] = 1.40884144229
    refarray[112][0] = 752.333643
    refarray[112][1] = 1.41294006744
    refarray[113][0] = 785.893464
    refarray[113][1] = 1.41730868351
    refarray[114][0] = 819.453285
    refarray[114][1] = 1.42141930517
    refarray[115][0] = 853.013106
    refarray[115][1] = 1.42526958317
    refarray[116][0] = 886.572927
    refarray[116][1] = 1.42885716832
    refarray[117][0] = 926.1229445
    refarray[117][1] = 1.43286768316
    refarray[118][0] = 965.672962
    refarray[118][1] = 1.4367548748
    refarray[119][0] = 1005.2229795
    refarray[119][1] = 1.44051890579
    refarray[120][0] = 1044.772997
    refarray[120][1] = 1.44415993872
    refarray[121][0] = 1091.38551775
    refarray[121][1] = 1.44827092788
    refarray[122][0] = 1137.9980385
    refarray[122][1] = 1.45216656199
    refarray[123][0] = 1184.61055925
    refarray[123][1] = 1.45584724184
    refarray[124][0] = 1231.22308
    refarray[124][1] = 1.45931336817
    refarray[125][0] = 1286.0519775
    refarray[125][1] = 1.46317213317
    refarray[126][0] = 1340.880875
    refarray[126][1] = 1.46684676875
    refarray[127][0] = 1395.7097725
    refarray[127][1] = 1.47033799649
    refarray[128][0] = 1450.53867
    refarray[128][1] = 1.47364653798
    refarray[129][0] = 1515.19538875
    refarray[129][1] = 1.477372381
    refarray[130][0] = 1579.8521075
    refarray[130][1] = 1.48096223732
    refarray[131][0] = 1644.50882625
    refarray[131][1] = 1.48441704676
    refarray[132][0] = 1709.165545
    refarray[132][1] = 1.48773774913
    refarray[133][0] = 1785.34110675
    refarray[133][1] = 1.4915055672
    refarray[134][0] = 1861.5166685
    refarray[134][1] = 1.49514172383
    refarray[135][0] = 1937.69223025
    refarray[135][1] = 1.49864627657
    refarray[136][0] = 2013.867792
    refarray[136][1] = 1.50201928295
    refarray[137][0] = 2103.53174025
    refarray[137][1] = 1.50582097515
    refarray[138][0] = 2193.1956885
    refarray[138][1] = 1.50943752569
    refarray[139][0] = 2282.85963675
    refarray[139][1] = 1.5128647456
    refarray[140][0] = 2372.523585
    refarray[140][1] = 1.5160984459
    refarray[141][0] = 2478.18261075
    refarray[141][1] = 1.51968298106
    refarray[142][0] = 2583.8416365
    refarray[142][1] = 1.52304002934
    refarray[143][0] = 2689.50066225
    refarray[143][1] = 1.52615942882
    refarray[144][0] = 2795.159688
    refarray[144][1] = 1.52903101759
    refarray[145][0] = 2919.61889775
    refarray[145][1] = 1.53216091022
    refarray[146][0] = 3044.0781075
    refarray[146][1] = 1.53507875823
    refarray[147][0] = 3168.53731725
    refarray[147][1] = 1.53777460422
    refarray[148][0] = 3292.996527
    refarray[148][1] = 1.54023849082
    refarray[149][0] = 3439.47881025
    refarray[149][1] = 1.5429155429
    refarray[150][0] = 3585.9610935
    refarray[150][1] = 1.54542418249
    refarray[151][0] = 3732.44337675
    refarray[151][1] = 1.54775902333
    refarray[152][0] = 3878.92566
    refarray[152][1] = 1.54991467911
    refarray[153][0] = 4051.46957425
    refarray[153][1] = 1.55228512314
    refarray[154][0] = 4224.0134885
    refarray[154][1] = 1.5545328369
    refarray[155][0] = 4396.55740275
    refarray[155][1] = 1.55665699897
    refarray[156][0] = 4569.101317
    refarray[156][1] = 1.55865678795
    refarray[157][0] = 4772.34225225
    refarray[157][1] = 1.56089679992
    refarray[158][0] = 4975.5831875
    refarray[158][1] = 1.56305433602
    refarray[159][0] = 5178.82412275
    refarray[159][1] = 1.5651316339
    refarray[160][0] = 5382.065058
    refarray[160][1] = 1.56713093122
    refarray[161][0] = 5621.47396675
    refarray[161][1] = 1.5694135214
    refarray[162][0] = 5860.8828755
    refarray[162][1] = 1.57164372605
    refarray[163][0] = 6100.29178425
    refarray[163][1] = 1.57382526399
    refarray[164][0] = 6339.700693
    refarray[164][1] = 1.57596185403
    refarray[165][0] = 6621.16631025
    refarray[165][1] = 1.57842254522
    refarray[166][0] = 6902.6319275
    refarray[166][1] = 1.58083458651
    refarray[167][0] = 7184.09754475
    refarray[167][1] = 1.58320380344
    refarray[168][0] = 7465.563162
    refarray[168][1] = 1.58553602155
    refarray[169][0] = 7797.1029215
    refarray[169][1] = 1.58823724568
    refarray[170][0] = 8128.642681
    refarray[170][1] = 1.59089039728
    refarray[171][0] = 8460.1824405
    refarray[171][1] = 1.59350257956
    refarray[172][0] = 8791.7222
    refarray[172][1] = 1.59608089572
    refarray[173][0] = 9181.85089375
    refarray[173][1] = 1.5990767012
    refarray[174][0] = 9571.9795875
    refarray[174][1] = 1.60203077325
    refarray[175][0] = 9962.10828125
    refarray[175][1] = 1.60494352811
    refarray[176][0] = 10352.236975
    refarray[176][1] = 1.60781538201
    refarray[177][0] = 10811.39915
    refarray[177][1] = 1.61114188272
    refarray[178][0] = 11270.561325
    refarray[178][1] = 1.61439842168
    refarray[179][0] = 11729.7235
    refarray[179][1] = 1.61756892626
    refarray[180][0] = 12188.885675
    refarray[180][1] = 1.62063732383
    refarray[181][0] = 12728.6200687
    refarray[181][1] = 1.62411334829
    refarray[182][0] = 13268.3544625
    refarray[182][1] = 1.62744114618
    refarray[183][0] = 13808.0888563
    refarray[183][1] = 1.63059186338
    refarray[184][0] = 14347.82325
    refarray[184][1] = 1.63353664579
    refarray[185][0] = 14982.8517
    refarray[185][1] = 1.63677083558
    refarray[186][0] = 15617.88015
    refarray[186][1] = 1.63979104576
    refarray[187][0] = 16252.9086
    refarray[187][1] = 1.64257046377
    refarray[188][0] = 16887.93705
    refarray[188][1] = 1.64508227704
    refarray[189][0] = 17633.9761125
    refarray[189][1] = 1.64776450728
    refarray[190][0] = 18380.015175
    refarray[190][1] = 1.65023513284
    refarray[191][0] = 19126.0542375
    refarray[191][1] = 1.65247970368
    refarray[192][0] = 19872.0933
    refarray[192][1] = 1.65448376977
    refarray[193][0] = 20749.2362938
    refarray[193][1] = 1.65662253681
    refarray[194][0] = 21626.3792875
    refarray[194][1] = 1.65861946328
    refarray[195][0] = 22503.5222812
    refarray[195][1] = 1.6604730855
    refarray[196][0] = 23380.665275
    refarray[196][1] = 1.66218193978
    refarray[197][0] = 24411.5917875
    refarray[197][1] = 1.664065814
    refarray[198][0] = 25442.5183
    refarray[198][1] = 1.66587452683
    refarray[199][0] = 26473.4448125
    refarray[199][1] = 1.66761213775
    refarray[200][0] = 27504.371325
    refarray[200][1] = 1.66928270626
    refarray[201][0] = 28719.51975
    refarray[201][1] = 1.67118044304
    refarray[202][0] = 29934.668175
    refarray[202][1] = 1.67301613739
    refarray[203][0] = 31149.8166
    refarray[203][1] = 1.67479704105
    refarray[204][0] = 32364.965025
    refarray[204][1] = 1.67653040579
    refarray[205][0] = 33797.5589688
    refarray[205][1] = 1.67851972995
    refarray[206][0] = 35230.1529125
    refarray[206][1] = 1.68045711586
    refarray[207][0] = 36662.7468562
    refarray[207][1] = 1.68235179282
    refarray[208][0] = 38095.3408
    refarray[208][1] = 1.68421299011
    refarray[209][0] = 39783.3785187
    refarray[209][1] = 1.68637324056
    refarray[210][0] = 41471.4162375
    refarray[210][1] = 1.68850396803
    refarray[211][0] = 43159.4539562
    refarray[211][1] = 1.69061103173
    refarray[212][0] = 44847.491675
    refarray[212][1] = 1.69270029089
    refarray[213][0] = 46840.5101625
    refarray[213][1] = 1.69515036214
    refarray[214][0] = 48833.52865
    refarray[214][1] = 1.69758257946
    refarray[215][0] = 50826.5471375
    refarray[215][1] = 1.69999552204
    refarray[216][0] = 52819.565625
    refarray[216][1] = 1.70238776906
    refarray[217][0] = 55171.4786375
    refarray[217][1] = 1.70518387387
    refarray[218][0] = 57523.39165
    refarray[218][1] = 1.7079497416
    refarray[219][0] = 59875.3046625
    refarray[219][1] = 1.71068249094
    refarray[220][0] = 62227.217675
    refarray[220][1] = 1.71337924057
    refarray[221][0] = 65000.6924938
    refarray[221][1] = 1.71651792343
    refarray[222][0] = 67774.1673125
    refarray[222][1] = 1.71962538606
    refarray[223][0] = 70547.6421313
    refarray[223][1] = 1.7227120352
    refarray[224][0] = 73321.11695
    refarray[224][1] = 1.7257882776
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").GetTable("BhTable").SetTable(refarray)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"DemagnetizationCoerciveForce", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"MagnetizationSaturated", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"MagnetizationSaturated2", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"ExtrapolationMethod", 1)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"YoungModulus", 210000)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"ShearModulus", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"YoungModulusX", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"YoungModulusY", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"YoungModulusZ", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"ShearModulusXY", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"ShearModulusYZ", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"ShearModulusZX", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G11", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G12", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G13", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G14", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G15", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G16", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G22", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G23", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G24", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G25", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G26", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G33", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G34", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G35", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G36", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G44", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G45", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G46", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G55", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G56", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"Arnon5").SetValue(u"G66", 0)


# some new code


# now I add some more codes!
