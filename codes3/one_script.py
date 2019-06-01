import shutil
import os
import win32com.client
import logging
from utility import my_execfile
from pylab import plt
from time import time as clock_time
bool_post_processing = False # solve or post-processing

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
    fea_config_dict['use_weights'] = 'O2'
    run_folder = r'run#535/' # test with pygmo


    fea_config_dict['local_sensitivity_analysis'] = False
    fea_config_dict['bool_refined_bounds'] = False
    fea_config_dict['use_weights'] = 'O2' # this is not working
    run_folder = r'run#537/' # test with pygmo

else:
    # Prototype
    pass
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # Severson02
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    my_execfile('./spec_Prototype2poleOD150mm500Hz_SpecifyTipSpeed.py', g=globals(), l=locals()) # define spec
    fea_config_dict['local_sensitivity_analysis'] = False
    fea_config_dict['bool_refined_bounds'] = False
    fea_config_dict['use_weights'] = 'O2'
    run_folder = r'run#54029/'

    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # Severson01
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    my_execfile('./spec_Prototype4poleOD150mm1000Hz_SpecifyTipSpeed.py', g=globals(), l=locals()) # define spec
    fea_config_dict['local_sensitivity_analysis'] = False
    fea_config_dict['bool_refined_bounds'] = False
    fea_config_dict['use_weights'] = 'O2'
    run_folder = r'run#53019/'

fea_config_dict['run_folder'] = run_folder
fea_config_dict['Active_Qr'] = spec.Qr
fea_config_dict['use_drop_shape_rotor_bar'] = spec.use_drop_shape_rotor_bar
class FEA_Solver:
    def __init__(self, fea_config_dict):
        self.fea_config_dict = fea_config_dict
        self.app = None

        self.output_dir = self.fea_config_dict['dir_parent'] + self.fea_config_dict['run_folder']
        self.dir_csv_output_folder = self.output_dir + 'csv/'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.dir_csv_output_folder):
            os.makedirs(self.dir_csv_output_folder)

        # post-process feature
        self.fig_main, self.axeses = plt.subplots(2, 2, sharex=True, dpi=150, figsize=(16, 8), facecolor='w', edgecolor='k')
        utility.pyplot_clear(self.axeses)

        self.folder_to_be_deleted = None

        open(self.output_dir+'MOO_log.txt', 'w').close()

    def fea_bearingless_induction(self, im_template, x_denorm, counter):
        logger = logging.getLogger(__name__)
        print('Run FEA for individual #%d'%(counter))

        # get local design variant
        im_variant = population.bearingless_induction_motor_design.local_design_variant(im_template, 0, counter, x_denorm)
        im_variant.name = 'ind%d'%(counter)
        self.im_variant = im_variant
        self.femm_solver = FEMM_Solver.FEMM_Solver(self.im_variant, flag_read_from_jmag=False, freq=50) # eddy+static
        im = None

        self.project_name          = 'proj%d'%(counter)
        self.expected_project_file = self.output_dir + "%s.jproj"%(self.project_name)

        original_study_name = im_variant.name + "Freq"
        tran2tss_study_name = im_variant.name + 'Tran2TSS'

        self.dir_femm_temp         = self.output_dir + 'femm_temp/'
        self.femm_output_file_path = self.dir_femm_temp + original_study_name + '.csv'

        self.jmag_control_state = False

        # local scripts
        def open_jmag(expected_project_file):
            if self.app is None:
                app = win32com.client.Dispatch('designer.Application.171')
                if self.fea_config_dict['designer.Show'] == True:
                    app.Show()
                else:
                    app.Hide()
                # app.Quit()
                self.app = app # means that the JMAG Designer is turned ON now.
                self.bool_run_in_JMAG_Script_Editor = False

                def add_steel(self):
                    print('[First run on this computer detected]', self.fea_config_dict['Steel'], 'is added to jmag material library.')

                    if 'M15' in self.fea_config_dict['Steel']:
                        add_M1xSteel(self.app, self.dir_parent, steel_name="M-15 Steel")
                    elif 'M19' in self.fea_config_dict['Steel']:
                        add_M1xSteel(self.app, self.dir_parent)
                    elif 'Arnon5' == self.fea_config_dict['Steel']:
                        add_Arnon5(self.app, self.dir_parent)        

                # too avoid tons of the same material in JAMG's material library
                fname = self.fea_config_dict['dir_parent'] + '.jmag_state.txt'
                if not os.path.exists(fnmae):
                    with open(fname, 'w') as f:
                        f.write(self.fea_config_dict['pc_name'] + '/' + self.fea_config_dict['Steel'] + '\n')
                    add_steel(self)
                else:
                    with open(fname, 'r') as f:
                        for line in f.readlines():
                            if self.fea_config_dict['pc_name'] + '/' + self.fea_config_dict['Steel'] not in line:
                                add_steel(self)

            else:
                app = self.app

            print(expected_project_file)
            if os.path.exists(expected_project_file):
                os.remove(expected_project_file)
            if not os.path.exists(expected_project_file):
                app.NewProject("Untitled")
                app.SaveAs(expected_project_file)
                logger.debug('Create JMAG project file: %s'%(expected_project_file))
            else:
                raise 

            return app

        def draw_jmag(app):
            #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            # Draw the model in JMAG Designer
            #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
            DRAW_SUCCESS = self.draw_jmag_model(app,
                                                counter, 
                                                im_variant,
                                                im_variant.name)
            if DRAW_SUCCESS == 0:
                # TODO: skip this model and its evaluation
                cost_function = 99999 # penalty
                logging.getLogger(__name__).warn('Draw Failed for %s-%s\nCost function penalty = %g.%s', self.project_name, im_variant.name, cost_function, self.im_variant.show(toString=True))
                raise Exception('Draw Failed: Are you working on the PC? Sometime you by mistake operate in the JMAG Geometry Editor, then it fails to draw.')
                return None
            elif DRAW_SUCCESS == -1:
                raise

            # JMAG
            if app.NumModels()>=1:
                model = app.GetModel(im_variant.name)
            else:
                logger.error('there is no model yet!')
                raise Exception('why is there no model yet?')
            return model

        # this should be summoned even before initializing femm, and it will decide whether the femm results are reliable
        app = open_jmag(self.expected_project_file) # will set self.jmag_control_state to True

        ################################################################
        # Begin from where left: Frequency Study
        ################################################################
        #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        # Eddy Current Solver for Breakdown Torque and Slip
        #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        if self.fea_config_dict['jmag_run_list'][0] == 0:
            # check for existing results
            if os.path.exists(self.femm_output_file_path):
                raise
            else:
                # no direct returning of results, wait for it later when you need it.
                femm_tic = clock_time()
                # self.femm_solver.__init__(im_variant, flag_read_from_jmag=False, freq=50.0)
                if im_variant.DriveW_poles == 2:
                    self.femm_solver.greedy_search_for_breakdown_slip( self.dir_femm_temp, original_study_name, 
                                                                        bool_run_in_JMAG_Script_Editor=self.bool_run_in_JMAG_Script_Editor, fraction=1) # 转子导条必须形成通路
                else:
                    self.femm_solver.greedy_search_for_breakdown_slip( self.dir_femm_temp, original_study_name, 
                                                                        bool_run_in_JMAG_Script_Editor=self.bool_run_in_JMAG_Script_Editor, fraction=2)
        else:
            raise

        ################################################################
        # Begin from where left: Transient Study
        ################################################################
        model = draw_jmag(app)

        #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        # TranFEAwi2TSS for ripples and iron loss
        #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        # add or duplicate study for transient FEA denpending on jmag_run_list
        if self.fea_config_dict['jmag_run_list'][0] == 0:
            # FEMM+JMAG
            study = im_variant.add_TranFEAwi2TSS_study( 50.0, app, model, self.dir_csv_output_folder, tran2tss_study_name, logger)
            self.mesh_study(im_variant, app, model, study)

            # wait for femm to finish, and get your slip of breakdown
            slip_freq_breakdown_torque, breakdown_torque, breakdown_force = self.femm_solver.wait_greedy_search(femm_tic)

            # Now we have the slip, set it up!
            im_variant.update_mechanical_parameters(slip_freq_breakdown_torque) # do this for records only
            if im_variant.the_slip != slip_freq_breakdown_torque / im_variant.DriveW_Freq:
                raise Exception('Check update_mechanical_parameters().')
            study.GetDesignTable().GetEquation("slip").SetExpression("%g"%(im_variant.the_slip))

            self.run_study(im_variant, app, study, clock_time())
        else:
            raise

        # export Voltage if field data exists.
        if self.fea_config_dict['delete_results_after_calculation'] == False:
            # Export Circuit Voltage
            ref1 = app.GetDataManager().GetDataSet("Circuit Voltage")
            app.GetDataManager().CreateGraphModel(ref1)
            app.GetDataManager().GetGraphModel("Circuit Voltage").WriteTable(self.dir_csv_output_folder + im_variant.name + "_EXPORT_CIRCUIT_VOLTAGE.csv")

        ################################################################
        # Load data for cost function evaluation
        ################################################################
        #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        # Load Results for Tran2TSS
        #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        results_to_be_unpacked = utility.build_str_results(self.axeses, im_variant, self.project_name, tran2tss_study_name, self.dir_csv_output_folder, self.fea_config_dict, self.femm_solver)
        # str_results, 
        if results_to_be_unpacked is not None:
            self.fig_main.savefig(self.output_dir + im_variant.name + 'results.png', dpi=150)
            utility.pyplot_clear(self.axeses)
            # show()

            str_results, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle, jmag_loss_list, femm_loss_list, power_factor, total_loss, cost_function = results_to_be_unpacked

            # write design evaluation data to file
            with open(self.output_dir + 'swarm_data.txt', 'a') as f:
                f.write(str_results)

            return im_variant.stack_length, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle, jmag_loss_list, femm_loss_list, power_factor, total_loss
        else:
            raise
        # winding analysis? 之前的python代码利用起来啊
        # 希望的效果是：设定好一个设计，马上进行运行求解，把我要看的数据都以latex报告的形式呈现出来。
        # OP_PS_Qr36_M19Gauge29_DPNV_NoEndRing.jproj

    def draw_jmag_model(self, app, individual_index, im_variant, model_name, bool_trimDrawer_or_vanGogh=True, doNotRotateCopy=False):

        if individual_index == -1: # 后处理是-1
            print('Draw model for post-processing')
            if individual_index+1 + 1 <= app.NumModels():
                logger = logging.getLogger(__name__)
                logger.debug('The model already exists for individual with index=%d. Skip it.', individual_index)
                return -1 # the model is already drawn

        elif individual_index+1 <= app.NumModels(): # 一般是从零起步
            logger = logging.getLogger(__name__)
            logger.debug('The model already exists for individual with index=%d. Skip it.', individual_index)
            return -1 # the model is already drawn

        # open JMAG Geometry Editor
        app.LaunchGeometryEditor()
        geomApp = app.CreateGeometryEditor()
        # geomApp.Show()
        geomApp.NewDocument()
        doc = geomApp.GetDocument()
        ass = doc.GetAssembly()

        # draw parts
        try:
            if bool_trimDrawer_or_vanGogh:
                d = population.TrimDrawer(im_variant) # 传递的是地址哦
                d.doc, d.ass = doc, ass
                d.plot_shaft("Shaft")

                d.plot_rotorCore("Rotor Core")
                d.plot_cage("Cage")

                d.plot_statorCore("Stator Core")
                d.plot_coil("Coil")
                # d.plot_airWithinRotorSlots(u"Air Within Rotor Slots")
            else:
                d = VanGogh_JMAG(im_variant, doNotRotateCopy=doNotRotateCopy) # 传递的是地址哦
                d.doc, d.ass = doc, ass
                d.draw_model()
            self.d = d
        except Exception as e:
            print('See log file to plotting error.')
            logger = logging.getLogger(__name__)
            logger.error('The drawing is terminated. Please check whether the specified bounds are proper.', exc_info=True)

            raise e

            # print 'Draw Failed'
            # if self.pc_name == 'Y730':
            #     # and send the email to hory chen
            #     raise e

            # or you can skip this model and continue the optimization!
            return False # indicating the model cannot be drawn with the script.

        # Import Model into Designer
        doc.SaveModel(True) # True=on : Project is also saved. 
        model = app.GetCurrentModel() # model = app.GetModel(u"IM_DEMO_1")
        model.SetName(model_name)
        model.SetDescription(im_variant.model_name_prefix + '\n' + im_variant.show(toString=True))

        if doNotRotateCopy:
            im_variant.pre_process_structural(app, d.listKeyPoints)
        else:
            im_variant.pre_process(app)

        model.CloseCadLink() # this is essential if you want to create a series of models
        return True

    def run_study(self, im_variant, app, study, toc):
        logger = logging.getLogger(__name__)
        if self.fea_config_dict['JMAG_Scheduler'] == False:
            print('Run jam.exe...')
            # if run_list[1] == True:
            study.RunAllCases()
            msg = 'Time spent on %s is %g s.'%(study.GetName() , clock_time() - toc)
            logger.debug(msg)
            print(msg)
        else:
            print('Submit to JMAG_Scheduler...')
            job = study.CreateJob()
            job.SetValue("Title", study.GetName())
            job.SetValue("Queued", True)
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
        app.ExportImageWithSize(self.output_dir + model.GetName() + '.png', 2000, 2000)
        app.View().ShowModel() # 1st btn. close mesh view, and note that mesh data will be deleted if only ouput table results are selected.

class acm_designer(object):
    def __init__(self, fea_config_dict, spec):

        build_model_name_prefix(fea_config_dict) # rebuild the model name for fea_config_dict
        spec.build_im_template(fea_config_dict)

        self.spec = spec
        self.solver = FEA_Solver(fea_config_dict)
        self.fea_config_dict = fea_config_dict

    def init_logger(self, prefix='pygmo_'):
        self.logger = utility.myLogger(self.fea_config_dict['dir_codes'], prefix=prefix+self.fea_config_dict['run_folder'][:-1])

    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # Automatic Report Generation
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    def build_oneReport(self):
        if '730' in self.fea_config_dict['pc_name']:
            os.system('cd /d "'+ self.fea_config_dict['dir_parent'] + 'release/OneReport/OneReport_TEX" && z_nul"')

    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # Talk to Database
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    def talk_to_mysql_database(self):
        if self.spec.bool_bad_specifications:
            print('\nThe specifiaction can not be fulfilled. Read script log or OneReport.pdf for information and revise the specifiaction for $J_r$ or else your design name is wrong.')
        else:
            print('\nThe specifiaction is meet. Now check the database of blimuw if on Y730.')
            if '730' in self.fea_config_dict['pc_name']:
                utility.communicate_database(self.spec)

    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # Automatic Performance Evaluation (This is just a wraper)
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    def evaluate_design(self, im_template, x_denorm, counter):
        return self.solver.fea_bearingless_induction(im_template, x_denorm, counter)

    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # 1. Bounds for DE optimiazation
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    def get_original_bounds(self, Jr_max=8e6):

        from math import tan, pi
        定子齿宽最小值 = 1
        定子齿宽最大值 = tan(2*pi/self.spec.Qs*0.5)*self.spec.Radius_OuterRotor * 2 # 圆（半径为Radius_OuterRotor）的外接正多边形（Regular polygon）的边长
        # print(定子齿宽最大值, 2*pi*self.spec.Radius_OuterRotor/self.spec.Qs) # 跟弧长比应该比较接近说明对了

        转子齿宽最小值 = 1
        内接圆的半径 = self.spec.Radius_OuterRotor - (self.spec.d_ro + self.spec.Radius_of_RotorSlot)
        转子齿宽最大值 = tan(2*pi/self.spec.Qr*0.5)*内接圆的半径 * 2 
        # print(转子齿宽最大值, 2*pi*内接圆的半径/self.spec.Qr) # 跟弧长比应该比较接近说明对了

        self.original_bounds = [ [           0.8,              3],          # air_gap_length_delta
                                 [定子齿宽最小值, 定子齿宽最大值],#--# stator_tooth_width_b_ds
                                 [转子齿宽最小值, 转子齿宽最大值],#--# rotor_tooth_width_b_dr
                                 [             1,    360/spec.Qs],           # Angle_StatorSlotOpen
                                 [          5e-1,              3],           # Width_RotorSlotOpen 
                                 [          5e-1,              3],           # Width_StatorTeethHeadThickness
                                 [          5e-1,              3] ]          # Length_HeadNeckRotorSlot
        # 定子齿范围检查
        # 定子齿再宽，都可以无限加长轭部来满足导电面积。
        # stator_inner_radius_r_is_eff = stator_inner_radius_r_is + (width_statorTeethHeadThickness + width_StatorTeethNeck)
        # temp = (2*pi*stator_inner_radius_r_is_eff - self.Qs*stator_tooth_width_b_ds)
        # stator_tooth_height_h_ds = ( sqrt(temp**2 + 4*pi*area_stator_slot_Sus*self.Qs) - temp ) / (2*pi)

        def check_valid_rotor_slot_height(rotor_tooth_width_b_dr, Jr_max):

            area_conductor_rotor_Scr = self.spec.rotor_current_actual / Jr_max
            area_rotor_slot_Sur = area_conductor_rotor_Scr
            
            rotor_outer_radius_r_or_eff = 1e-3*(self.spec.Radius_OuterRotor - self.spec.d_ro)

            slot_height, _, _ = pyrhonen_procedure_as_function.get_parallel_tooth_height(area_rotor_slot_Sur, rotor_tooth_width_b_dr, self.spec.Qr, rotor_outer_radius_r_or_eff)
            return np.isnan(slot_height)


        # 转子齿范围检查
        下界, 上界 = self.original_bounds[2][0], self.original_bounds[2][1]
        步长 = (上界-下界)*0.05
        list_valid_tooth_width = []
        for rotor_tooth_width_b_dr in np.arange(下界, 上界, 步长):
            # print('b_dr =', rotor_tooth_width_b_dr)
            list_valid_tooth_width.append( check_valid_rotor_slot_height(rotor_tooth_width_b_dr, Jr_max) ) # 8e6 from Pyrhonen's book for copper
        # print(list_valid_tooth_width)
        有效上界 = 下界
        for ind, el in enumerate(list_valid_tooth_width):
            if el == True:
                break
            else:
                有效上界 += 步长
        self.original_bounds[2][1] = 有效上界

        return self.original_bounds

    def get_classic_bounds(self):
        self.get_original_bounds()
        self.classic_bounds = [ [self.spec.delta*0.7, self.spec.delta*2  ],          # air_gap_length_delta
                                [self.spec.w_st *0.5, self.spec.w_st *1.5],          #--# stator_tooth_width_b_ds
                                [self.spec.w_rt *0.5, self.spec.w_rt *1.5],          #--# rotor_tooth_width_b_dr
                                [                1.5,                  12],           # Angle_StatorSlotOpen
                                [               5e-1,                   3],           # Width_RotorSlotOpen 
                                [               5e-1,                   3],           # Width_StatorTeethHeadThickness
                                [               5e-1,                   3] ]          # Length_HeadNeckRotorSlot
        # classic_bounds cannot be beyond original_bounds
        index = 0
        for A, B in zip(self.classic_bounds, self.original_bounds):
            if A[0] < B[0]:
                self.classic_bounds[index] = B[0]
            if A[1] > B[1]:
                self.classic_bounds[index] = B[1]
            index += 1
            
        return self.classic_bounds

    def get_de_config(self):

        self.de_config_dict = { 'original_bounds': self.get_original_bounds(),
                                'mut':        0.8,
                                'crossp':     0.7,
                                'popsize':    35, # 5~10 \times number of geometry parameters --JAC223
                                'iterations': 70,
                                'narrow_bounds_normalized':[[],
                                                            [],
                                                            [],
                                                            [],
                                                            [],
                                                            [],
                                                            [] ], # != []*7 （完全是两回事）
                                'bounds':None}
        return self.de_config_dict

    def run_local_sensitivity_analysis(self, the_bounds, design_denorm=None):
        # if design_denorm not in the_bounds: then raise 
        if design_denorm is not None:
            for ind, el in enumerate(design_denorm):
                if el < the_bounds[ind][0] or el > the_bounds[ind][1]:
                    raise Exception('给的设计不在边界的内部')

        self.logger.debug('---------\nBegin Local Sensitivity Analysis')

        # de_config_dict['bounds'] 还没有被赋值
        self.de_config_dict['bounds'] = the_bounds

        self.init_swarm() # define app.sw
        self.sw.generate_pop(specified_initial_design_denorm=design_denorm)
        de_generator = self.sw.de()
        for result in de_generator:
            print(result)

    def check_results_of_local_sensitivity_analysis(self):
        if self.fea_config_dict['local_sensitivity_analysis'] == True:
            run_folder = self.fea_config_dict['run_folder'][:-1] + 'lsa/'
        else:
            run_folder = self.fea_config_dict['run_folder']

        # Sensitivity Bar Charts
        return os.path.exists(fea_config_dict['dir_parent'] + 'pop/' + run_folder + 'swarm_data.txt')

    def collect_results_of_local_sensitivity_analysis(self):
        print('Start to collect results for local sensitivity analysis...')
        try:
            results_for_refining_bounds = utility.build_sensitivity_bar_charts(self.spec, self.sw)
            # quit()
        except Exception as e:
            raise e
            os.remove(fea_config_dict['dir_parent'] + 'pop/' + run_folder + 'swarm_data.txt')
            print('Remove ' + fea_config_dict['dir_parent'] + 'pop/' + run_folder + 'swarm_data.txt')
            print('Continue for sensitivity analysis...')

        self.results_for_refining_bounds = results_for_refining_bounds
        # return results_for_refining_bounds

    def build_refined_bounds(self, the_bounds):

        de_config_dict = self.de_config_dict
        number_of_variants= self.fea_config_dict['local_sensitivity_analysis_number_of_variants'] # 故意不加1的

        print('-'*20, 'results_for_refining_bounds (a.k.a. results_for_refining_bounds):')
        str_to_file = 'When you are done, replace this line with "Done", save file and close all figures.\n'
        for key, val in self.results_for_refining_bounds.items():
            print('---', key)
            for el in val:
                print('\t', el)
            str_to_file += key + '\n' + '\n'.join( ','.join(f'{x}' for x in y) for y in val ) + '\n'

        self.logger.debug('The default refining factors will be for: %s'%(self.fea_config_dict['use_weights']))

        with open('./refining_factors.txt', 'w') as f:
            f.write(str_to_file)
        file_backup_user_input = './refining_factors_%s.txt'%(self.fea_config_dict['run_folder'][:-1])
        if os.path.exists(file_backup_user_input):
            os.system('start %s'%(file_backup_user_input))
        os.system('start ./refining_factors.txt')
        from pylab import show
        show()
        from time import sleep
        while True:
            with open('./refining_factors.txt') as f:
                buf = f.read()
                if buf[:4] == 'Done' or buf[:4] == 'done':
                    print('Done')
                    break
                else:
                    print('.', end='')
                    sleep(1)
        if os.path.exists(file_backup_user_input):
            os.remove(file_backup_user_input)
        os.rename('./refining_factors.txt', file_backup_user_input)

        buf_list = buf.split('\n')
        # for el in buf_list:
        #     print(el)

        print('fea_config_dict - use_weights changed from %s to %s'%(self.fea_config_dict['use_weights'], buf_list[1]))
        self.fea_config_dict['use_weights'] = buf_list[1]

        user_input_for_refining_bounds = [[float(x) for x in y.split(',')] for y in buf_list[2:2+7]]
        print('user_input_for_refining_bounds')
        for el in user_input_for_refining_bounds:
            print('\t', el)

        for ind, bound in enumerate(user_input_for_refining_bounds):
            下界 = bound[0]
            if 下界 != 0:
                下界 -= 1
            上界 = bound[-1]
            if 上界 != number_of_variants:
                上界 += 1
            self.de_config_dict['narrow_bounds_normalized'][ind].append(下界/number_of_variants)
            self.de_config_dict['narrow_bounds_normalized'][ind].append(上界/number_of_variants)

        self.de_config_dict['bounds'] = []
        for bnd1, bnd2 in zip(self.de_config_dict['original_bounds'], self.de_config_dict['narrow_bounds_normalized']):
            diff = bnd1[1] - bnd1[0]
            self.de_config_dict['bounds'].append( [ bnd1[0]+diff*bnd2[0] , bnd1[0]+diff*bnd2[1] ]) # 注意，都是乘以original_bounds的上限哦！

        print('-'*40)
        print('narrow_bounds_normalized:')
        for el in self.de_config_dict['narrow_bounds_normalized']:
            print('\t', el)

        print('original_bounds:')
        for el in self.de_config_dict['original_bounds']:
            print('\t', el)

        print('refined bounds:')
        for el in self.de_config_dict['bounds']:
            print('\t', el)

        return de_config_dict['bounds']

    def build_local_bounds_from_best_design(self, best_design):
        raise
        return local_bounds

    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # 2. Initilize Swarm and Initial Pyrhonen's Design (Run this part in JMAG) and femm solver (if required by run_list)
    #    Bounds: 1e-1也还是太小了（第三次报错），至少0.5mm长吧 
    #    # 1e-1 is the least geometry value. 
    #    A 1e-2 will leads to：转子闭口槽极限，会导致edge过小，从而报错：small arc entity exists.png
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    def init_swarm(self):
        self.sw = population.swarm(self.fea_config_dict, de_config_dict=self.de_config_dict)
        # sw.show(which='all')
        # print sw.im.show(toString=True)
        # quit()

        if self.fea_config_dict['jmag_run_list'][0] == 0:
            self.sw.init_femm_solver() # define app.sw.femm_solver        

    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # 3. Run DE Optimization
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    def run_de(self):
        sw = self.sw
        logger = self.logger

        count_abort = 0
        logger.debug('-------------------------count_abort=%d' % (count_abort))

        # if optimization_flat == True:
        # generate the initial generation
        sw.generate_pop(specified_initial_design_denorm=None)
        # quit()

        # add initial_design of Pyrhonen09 to the initial generation
        if sw.fea_config_dict['local_sensitivity_analysis'] == False:
            if count_abort == 0:
                utility.add_Pyrhonen_design_to_first_generation(sw, self.de_config_dict, logger)

        # write FEA config to disk
        sw.write_to_file_fea_config_dict()

        if True:
            try:
                de_generator = sw.de()
                # run
                # result = list(de_generator)
                for result in de_generator:
                    print(result)
            except Exception as e:
                print('See log file for the error msg.')
                logger.error('Optimization aborted.', exc_info=True)

                raise e
                # 避免死循环
                count_abort+1
                if count_abort > 10:
                    quit()
            else:
                logger.info('Done.')
                utility.send_notification('Done.')

    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # 4. Post-processing
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    def best_design_by_weights(self, use_weights):
        if use_weights != self.fea_config_dict['use_weights']:
            raise Exception('Not implemented')

        # import utility
        swda = utility.build_Pareto_plot(self.spec, self.sw)
        # sw.fobj(999, individual_denorm)

        # Final LaTeX Report
        print('According to Pyrhonen09, 300 MPa is the typical yield stress of iron core.')
        initial_design = utility.Pyrhonen_design(self.sw.im, self.de_config_dict['bounds'])
        print (initial_design.design_parameters_denorm)
        print (swda.best_design_denorm)
        print (self.de_config_dict['bounds'])

        best_report_dir_prefix = '../release/OneReport/BestReport_TEX/contents/'
        file_name = 'final_report'
        file_suffix = '.tex'
        fname = open(best_report_dir_prefix+file_name+'_s01'+file_suffix, 'w', encoding='utf-8')
        def combine_lists_alternating_2(list1, list2):
            if abs(len(list1) - len(list2))<=1:
                result = [None]*(len(list1)+len(list2))
                result[::2] = list1
                result[1::2] = list2
                return result
            else:
                raise Exception('Try this (not tested).') # https://stackoverflow.com/questions/3678869/pythonic-way-to-combine-two-lists-in-an-alternating-fashion
                import itertools 
                return [x for x in itertools.chain.from_iterable(itertools.izip_longest(list1,list2)) if x]
        def combine_lists_alternating_4(list1, list2, list3, list4):
            result = [None]*(len(list1)+len(list2)+len(list3)+len(list4))
            result[::4] = list1
            result[1::4] = list2
            result[2::4] = list3
            result[3::4] = list4
            return result
        lower_bounds = [el[0] for el in self.de_config_dict['bounds']]
        upper_bounds = [el[1] for el in self.de_config_dict['bounds']]
        design_data = combine_lists_alternating_4(  initial_design.design_parameters_denorm, 
                                                    swda.best_design_denorm,
                                                    lower_bounds,
                                                    upper_bounds,)
        latex_table = r'''
            \begin{table*}[!t]
              \caption{Comparison of the key geometry parameters between best design and initial design}
              \centering
                \begin{tabular}{ccccc}
                    \hline
                    \hline
                    \thead{Geometry parameters} &
                    \thead{Initial\\\relax design} &
                    \thead{The best\\\relax design} &
                    \thead{Lower\\\relax bounds} &
                    \thead{Upper\\\relax bounds} \\
                    \hline
                    Air gap length $L_g$                    & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ \\
                    Stator tooth width $w_{st}$             & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ \\
                    Rotor tooth width $w_{rt}$              & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ \\
                    Stator slot open angle $\theta_{so}$    & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ \\
                    Rotor slot open width $w_{ro}$          & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ \\
                    Stator slot open depth $d_{so}$         & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ \\
                    Rotor slot open depth $d_{ro}$          & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ \\
                    \hline
                    \vspace{-2.5ex}
                    \\
                    \multicolumn{3}{l}{*Note: Blah.}
                \end{tabular}
              \label{tab:001}
              \vspace{-3ex}
            \end{table*}
            ''' % tuple(design_data)
        print('''\\subsection{Best Design of Objective Function %s}\n\n
                    ''' % (self.fea_config_dict['use_weights']) + latex_table, file=fname)
        print(swda.str_best_design_details, file=fname)
        fname.close()

        os.system('cd /d '+ r'"D:\OneDrive - UW-Madison\c\release\OneReport\BestReport_TEX" && z_nul"') # 必须先关闭文件！否则编译不起来的
        # import subprocess
        # subprocess.call(r"D:\OneDrive - UW-Madison\c\release\OneReport\OneReport_TEX\z_nul", shell=True)

    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # 5. Check mechanical strength for the best design
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    def run_static_structural_fea(self, design_denorm):
        sw = self.sw
        im_best = population.bearingless_induction_motor_design.local_design_variant(sw.im, \
                     -1, -1, design_denorm)

        # initialize JMAG Designer
        sw.designer_init()
        sw.app.Show()
        project_name = fea_config_dict['run_folder'][:-1]+'_Best' # spec.build_name() is too long...
        expected_project_file = sw.dir_project_files + "%s.jproj"%(project_name)
        print(expected_project_file)
        if not os.path.exists(expected_project_file):
            sw.app.NewProject("Untitled")
            sw.app.SaveAs(expected_project_file)
            logger.debug('Create JMAG project file: %s'%(expected_project_file))
        else:
            sw.app.Load(expected_project_file)
            logger.debug('Load JMAG project file: %s'%(expected_project_file))
            logger.debug('Existing models of %d are found in %s', sw.app.NumModels(), sw.app.GetDefaultModelFolderPath())

        # draw the model in JMAG Designer
        DRAW_SUCCESS = sw.draw_jmag_model( -1, 
                                             im_best,
                                             'Best %s %s'%(fea_config_dict['use_weights'], fea_config_dict['run_folder'][:-1]),
                                             bool_trimDrawer_or_vanGogh=False,
                                             doNotRotateCopy=True)
        if DRAW_SUCCESS == 0:
            raise Exception('Drawing failed')
        elif DRAW_SUCCESS == -1:
            print('Model Already Exists')
        # quit()

        model = sw.app.GetCurrentModel()
        if model.NumStudies() == 0:
            expected_csv_output_dir = sw.dir_csv_output_folder+'structural/'
            if not os.path.isdir(expected_csv_output_dir):
                os.makedirs(expected_csv_output_dir)
            study = im_best.add_structural_study(sw.app, model, expected_csv_output_dir) # 文件夹名应该与jproj同名
        else:
            study = model.GetStudy(0)


        if study.AnyCaseHasResult():
            pass
        else:
            # Add cases 
            study.GetDesignTable().AddParameterVariableName(u"Centrifugal_Force (CentrifugalForce2D): AngularVelocity")
            study.GetDesignTable().AddCase()
            study.GetDesignTable().SetValue(1, 0, 45000) # r/min
            study.GetDesignTable().AddCase()
            study.GetDesignTable().SetValue(2, 0, 15000) # r/min
            study.GetDesignTable().AddParameterVariableName(u"MeshSizeControl (ElementSizeOnPart): Size")
            study.GetDesignTable().AddCase()
            study.GetDesignTable().SetValue(3, 1, 0.5) # mm
            study.GetDesignTable().AddCase()
            study.GetDesignTable().SetValue(4, 1, 0.05) # mm

            # run (mesh is included in add_study_structural)
            study.RunAllCases()
            sw.app.Save()

            # Results-Graphs-Calculations-Add Part Calculation
            study.CreateCalculationDefinition(u"VonMisesStress")
            study.GetCalculationDefinition(u"VonMisesStress").SetResultType(u"MisesStress", u"")
            study.GetCalculationDefinition(u"VonMisesStress").SetResultCoordinate(u"Global Rectangular")
            study.GetCalculationDefinition(u"VonMisesStress").SetCalculationType(u"max")
            study.GetCalculationDefinition(u"VonMisesStress").ClearParts()
            study.GetCalculationDefinition(u"VonMisesStress").AddSet(model.GetSetList().GetSet(u"Motion_Region"), 0)

            # show graph, Tab File|Edit|Calculation, click on Calculation - Response Graph Data to register response data
            parameter = sw.app.CreateResponseDataParameter(u"VonMisesStress")
            parameter.SetCalculationType(u"SingleValue")
            parameter.SetStartValue(u"-1")
            parameter.SetEndValue(u"-1")
            parameter.SetUnit(u"s")
            parameter.SetVariable(u"VonMisesStress")
            parameter.SetAllLine(False)
            parameter.SetCaseRangeType(1)
            parameter.SetLine(u"Maximum Value")
            sw.app.GetDataManager().CreateParametricDataWithParameter(study.GetDataSet(u"VonMisesStress", 4), parameter)

            # Contour results
            study.CreateScaling(u"Scale100")
            study.GetScaling(u"Scale100").SetScalingFactor(100)
            # sw.app.View().SetOriginalModelView(True)
            sw.app.View().ShowMeshGeometry()
            sw.app.View().SetScaledDisplacementView(True)
            study.CreateContour(u"MisesElement")
            study.GetContour(u"MisesElement").SetResultType(u"MisesStress", u"")
            study.GetContour(u"MisesElement").SetResultCoordinate(u"Global Rectangular")
            study.GetContour(u"MisesElement").SetContourType(2)
            study.GetContour(u"MisesElement").SetDigitsNotationType(2)

            study.GetContour(u"MisesElement").SetLogScale(True)
            study.GetContour(u"MisesElement").SetNumLabels(u"11")
            study.GetContour(u"MisesElement").SetPrecision(u"1")
            study.GetContour(u"MisesElement").SetGradient(u"PurpleRed", u"11", False)

            sw.app.View().SetContourView(True)



            # study.CreateContour(u"PrincipleStressElement")
            # study.GetContour(u"PrincipleStressElement").SetResultType(u"PrincipalStress", u"")
            # study.GetContour(u"PrincipleStressElement").SetComponent(u"I")
            # study.GetContour(u"PrincipleStressElement").SetContourType(2)
            # study.GetContour(u"PrincipleStressElement").SetDigitsNotationType(2)

            sw.app.Save()


        from pylab import show
        show()
global ad
ad = acm_designer(fea_config_dict, spec)
if 'Y730' in fea_config_dict['pc_name']:
    ad.build_oneReport() # require LaTeX
    # ad.talk_to_mysql_database() # require MySQL
ad.init_logger()

ad.bounds_denorm = ad.get_classic_bounds()
print('classic_bounds and original_bounds')
for A, B in zip(ad.bounds_denorm, ad.original_bounds):
    print(A, B)
# quit()

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Optimization
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~

import pygmo as pg
class Problem_BearinglessInductionDesign(object):

    # Define objectives
    def fitness(self, x):
        # return [x[0], x[1], x[2]]

        global ad, counter_fitness_called, counter_fitness_return, folder_to_be_deleted
        ad, counter_fitness_called, counter_fitness_return
        if counter_fitness_called == counter_fitness_return:
            counter_fitness_called += 1
        else:
            # This is a re-evaluation
            raise
        print('Fitness: %d, %d'%(counter_fitness_called, counter_fitness_return))

        # denormalize the chromosome
        min_b, max_b = np.asarray(ad.bounds_denorm).T 
        diff = np.fabs(min_b - max_b)
        x_denorm = min_b + x * diff

        # evaluate x_denorm via FEA tools
        counter_loop = 0
        while True:
            counter_loop += 1
            if counter_loop > 3:
                raise Exception('Abort the optimization. Three attemps to evaluate the design have all failed for individual #%d'%(counter_fitness_called))

            try:
                stack_length_mm, torque_average, normalized_torque_ripple, \
                    ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle, \
                    jmag_loss_list, femm_loss_list, power_factor, total_loss = \
                    ad.evaluate_design(ad.spec.im_template, x_denorm, counter_fitness_called)

                # remove folder .jfiles to save space (we have to generate it first in JMAG Designer to have field data and voltage profiles)
                if ad.solver.folder_to_be_deleted is not None:
                    shutil.rmtree(ad.solver.folder_to_be_deleted) # .jfiles directory
                # update to be deleted when JMAG releases the use
                ad.solver.folder_to_be_deleted = ad.solver.expected_project_file[:-5]+'jfiles'

            except Exception as e:
                # raise e
                print(e)

                msg = 'FEA tool failed for individual #%d: attemp #%d.'%(counter_fitness_called, counter_loop)
                logger = logging.getLogger(__name__)
                logger.error(msg)
                print(msg)

                msg = 'Removing all files for individual #%d and try again...'%(counter_fitness_called)
                logger.error(msg)
                print(msg)

                # turn off JMAG Designer
                ad.solver.app.Quit()
                ad.solver.app = None

                os.remove(ad.solver.expected_project_file) # .jproj
                # shutil.rmtree(ad.solver.expected_project_file[:-5]+'jfiles') # .jfiles directory # .jplot file in this folder will be used by JSOL softwares even JMAG Designer is closed.
                os.remove(ad.solver.femm_output_file_path) # . csv
                os.remove(ad.solver.femm_output_file_path[:-3]+'fem') # .fem
                for file in os.listdir(ad.solver.dir_femm_temp):
                    if 'femm_temp_' in file:
                        os.remove(ad.solver.dir_femm_temp + file)
            else:
                break

        # caculate the fitness
        print('-'*40)
        print('caculate the fitness for', ad.solver.im_variant.name)

        # LOSS
        stator_copper_loss_along_stack = femm_loss_list[2]
        rotor_copper_loss_along_stack  = femm_loss_list[3]

        stator_copper_loss_in_end_turn = femm_loss_list[0] - stator_copper_loss_along_stack 
        rotor_copper_loss_in_end_turn  = femm_loss_list[1] - rotor_copper_loss_along_stack

        rated_ratio                          = ad.spec.required_torque / torque_average 
        rated_stack_length_mm                = rated_ratio * stack_length_mm
        rated_stator_copper_loss_along_stack = rated_ratio * stator_copper_loss_along_stack
        rated_rotor_copper_loss_along_stack  = rated_ratio * rotor_copper_loss_along_stack
        rated_iron_loss                      = rated_ratio * jmag_loss_list[2]

        # total_loss   = copper_loss + iron_loss + windage_loss
        rated_total_loss =  rated_stator_copper_loss_along_stack \
                          + rated_rotor_copper_loss_along_stack \
                          + stator_copper_loss_in_end_turn \
                          + rotor_copper_loss_in_end_turn \
                          + rated_iron_loss \
                          + utility.get_windage_loss(ad.solver.im_variant)

        # THERMAL
        stator_current_density = femm_loss_list[4]
        rotor_current_density  = femm_loss_list[5]
        print('Current density [Arms/m^2]:', stator_current_density, rotor_current_density, sep='\n')
        if rotor_current_density > 8e6:
            print('rotor_current_density is over 8e6 Arms/m^2')

        rated_shaft_power  = ad.solver.im_variant.Omega * ad.spec.required_torque
        rated_efficiency   = rated_shaft_power / (rated_total_loss + rated_shaft_power)  # 效率计算：机械功率/(损耗+机械功率)

        rated_rotor_volume = np.pi*(ad.solver.im_variant.Radius_OuterRotor*1e-3)**2 * (rated_stack_length_mm*1e-3)

        # This weighted list suggests that peak-to-peak torque ripple of 5% is comparable with Em of 5% or Ea of 5 deg. Ref: Ye gu ECCE 2018
        list_weighted_ripples = [normalized_torque_ripple/0.05, normalized_force_error_magnitude/0.05, force_error_angle/5.0]

        # - Torque per Rotor Volume
        f1 = - ad.spec.required_torque / rated_rotor_volume
        # - Efficiency @ Rated Power
        f2 = - rated_efficiency
        # Ripple Performance (Weighted Sum)
        f3 = sum(list_weighted_ripples)

        # Constraints (Em<0.2 and Ea<10 deg):
        if normalized_force_error_magnitude >= 0.2 or force_error_angle > 10:
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
        return ([0]*7, [1]*7)

    # Return function name
    def get_name(self):
        return "Bearingless Induction Motor Design"
def my_plot(fits, vectors, ndf):
    if True:
        for fit in fits:
            print(fit)
        # ax = pg.plot_non_dominated_fronts(fits)
        ax = pg.plot_non_dominated_fronts(fits, comp=[0,1], marker='o')
        # ax = pg.plot_non_dominated_fronts(fits, comp=[0,2], marker='o')
        # ax = pg.plot_non_dominated_fronts(fits, comp=[1,2], marker='o')
    else:
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
        for index, front in enumerate(ndf):
            print('Rank/Tier', index, front, file=fname)
        count = 0
        for domination_list, domination_count, non_domination_rank in zip(dl, dc, ndr):
            print('Individual #%d\t'%(count), 'Belong to Rank #%d\t'%(non_domination_rank), 'Dominating', domination_count, 'and they are', domination_list, file=fname)
            count += 1

        # print(fits, vectors, ndf)
        print(pop, file=fname)

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
    global counter_fitness_called, counter_fitness_return
    counter_fitness_called, counter_fitness_return = 0, 0
    prob = pg.problem(udp)
    pop = pg.population(prob, size=36) # don't forget to change neighbours to be below size (default is 20)
    print('-'*40, '\nInitial pop:', pop)

    ################################################################
    # MOO Step 2:
    #   Select algorithm (another option is pg.nsga2())
    ################################################################
    algo = pg.algorithm(pg.moead(gen=1, weight_generation="grid", decomposition="tchebycheff", neighbours=20, CR=1, F=0.5, eta_m=20, realb=0.9, limit=2, preserve_diversity=True)) # https://esa.github.io/pagmo2/docs/python/algorithms/py_algorithms.html#pygmo.moead
    print('-'*40, '\n', algo)

    ################################################################
    # MOO Step 3:
    #   Begin optimization
    ################################################################
    number_of_iterations = 50
    for _ in range(number_of_iterations):
        pop = algo.evolve(pop)
        my_print(pop, _)
        # my_plot(fits, vectors, ndf)

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
        raise
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


