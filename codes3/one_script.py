from utility import my_execfile
bool_post_processing = True # solve or post-processing

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 0. FEA Setting / General Information & Packages Loading
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
my_execfile('./default_setting.py', g=globals(), l=locals()) # define fea_config_dict
if True:
    # ECCE

    my_execfile('./spec_ECCE_4pole32Qr1000Hz.py', g=globals(), l=locals()) # define spec
    fea_config_dict['local_sensitivity_analysis'] = True
    fea_config_dict['bool_refined_bounds'] = True
    fea_config_dict['use_weights'] = 'O2'

    # run_folder = r'run#530/' # 圆形槽JMAG绘制失败BUG

    # fea_config_dict['local_sensitivity_analysis_number_of_variants'] = 2
    # run_folder = r'run#531/' # number_of_variant = 2

    # fea_config_dict['local_sensitivity_analysis'] = False
    # fea_config_dict['bool_refined_bounds'] = False
    # fea_config_dict['use_weights'] = 'O2'
    # run_folder = r'run#532/'

    # fea_config_dict['local_sensitivity_analysis_number_of_variants'] = 10
    # run_folder = r'run#533/' # number_of_variant = 10

    fea_config_dict['local_sensitivity_analysis_number_of_variants'] = 20
    run_folder = r'run#534/' # number_of_variant = 20

else:
    # Prototype

    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # Severson02
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # my_execfile('./spec_Prototype2poleOD150mm500Hz_SpecifyTipSpeed.py', g=globals(), l=locals()) # define spec
    # fea_config_dict['local_sensitivity_analysis'] = False
    # fea_config_dict['bool_refined_bounds'] = False
    # fea_config_dict['use_weights'] = 'O2'
    # run_folder = r'run#53001/'

    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # Severson01
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    my_execfile('./spec_Prototype4poleOD150mm1000Hz_SpecifyTipSpeed.py', g=globals(), l=locals()) # define spec
    fea_config_dict['local_sensitivity_analysis'] = False
    fea_config_dict['bool_refined_bounds'] = False
    fea_config_dict['use_weights'] = 'O2'
    run_folder = r'run#53002/'

fea_config_dict['run_folder'] = run_folder
fea_config_dict['Active_Qr'] = spec.Qr
fea_config_dict['use_drop_shape_rotor_bar'] = spec.use_drop_shape_rotor_bar

class acm_designer(object):
    def __init__(self, fea_config_dict, spec):

        build_model_name_prefix(fea_config_dict) # rebuild the model name for fea_config_dict

        self.fea_config_dict = fea_config_dict
        self.spec = spec

    def init_logger(self, prefix='ones_'):
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
    # Automatic Performance Evaluation
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    def evaluate_design(self, im):
        pass
        # winding analysis? 之前的python代码利用起来啊
        # 希望的效果是：设定好一个设计，马上进行运行求解，把我要看的数据都以latex报告的形式呈现出来。
        # OP_PS_Qr36_M19Gauge29_DPNV_NoEndRing.jproj

    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # 1. Bounds for DE optimiazation
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    def get_original_bounds(self):

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
                                 [             1,             11],           # Angle_StatorSlotOpen
                                 [          5e-1,              3],           # Width_RotorSlotOpen 
                                 [          5e-1,              3],           # Width_StatorTeethHeadThickness
                                 [          5e-1,              3] ]          # Length_HeadNeckRotorSlot
        # 定子齿范围检查
        # 定子齿再宽，都可以无限加长轭部来满足导电面积。
        # stator_inner_radius_r_is_eff = stator_inner_radius_r_is + (width_statorTeethHeadThickness + width_StatorTeethNeck)
        # temp = (2*pi*stator_inner_radius_r_is_eff - self.Qs*stator_tooth_width_b_ds)
        # stator_tooth_height_h_ds = ( sqrt(temp**2 + 4*pi*area_stator_slot_Sus*self.Qs) - temp ) / (2*pi)

        # 转子齿范围检查
        下界, 上界 = self.original_bounds[2][0], self.original_bounds[2][1]
        步长 = (上界-下界)*0.05
        list_valid_tooth_width = []
        for rotor_tooth_width_b_dr in np.arange(下界, 上界, 步长):
            print('b_dr =', rotor_tooth_width_b_dr)
            list_valid_tooth_width.append( self.check_valid_rotor_slot_height(rotor_tooth_width_b_dr, 8e6) ) # 8e6 from Pyrhonen's book for copper
        print(list_valid_tooth_width)
        有效上界 = 下界
        for ind, el in enumerate(list_valid_tooth_width):
            if el == True:
                break
            else:
                有效上界 += 步长
        self.original_bounds[2][1] = 有效上界

        return self.original_bounds

    def check_valid_rotor_slot_height(self, rotor_tooth_width_b_dr, J_max):

        area_conductor_rotor_Scr = self.spec.rotor_current_actual / J_max
        area_rotor_slot_Sur = area_conductor_rotor_Scr
        
        rotor_outer_radius_r_or_eff = 1e-3*(self.spec.Radius_OuterRotor - self.spec.d_ro)

        slot_height, _, _ = pyrhonen_procedure_as_function.get_parallel_tooth_height(area_rotor_slot_Sur, rotor_tooth_width_b_dr, self.spec.Qr, rotor_outer_radius_r_or_eff)
        return np.isnan(slot_height)


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

app = acm_designer(fea_config_dict, spec)
if 'Y730' in fea_config_dict['pc_name']:
    app.build_oneReport()
    # app.talk_to_mysql_database()
# quit()
app.init_logger()
# app.evaluate_design(spec.sw.im)

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Optimization
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
if True:
    # 微分进化的配置
    app.get_de_config()
    print('Run: ' + run_folder + '\nThe auto generated bounds are:', app.de_config_dict['original_bounds'])
    # quit()

    # 如果需要局部敏感性分析，那就先跑了再说
    if fea_config_dict['local_sensitivity_analysis'] == True:
        if not app.check_results_of_local_sensitivity_analysis():
            app.run_local_sensitivity_analysis(app.de_config_dict['original_bounds'], design_denorm=None)
        else:
            app.de_config_dict['bounds'] = app.de_config_dict['original_bounds']
            app.init_swarm() # define app.sw
            app.sw.generate_pop(specified_initial_design_denorm=None)
        app.collect_results_of_local_sensitivity_analysis() # need sw to work
        fea_config_dict['local_sensitivity_analysis'] = False # turn off lsa mode

    # Build the final bounds
    if fea_config_dict['bool_refined_bounds'] == -1:
        app.build_local_bounds_from_best_design(None)
    elif fea_config_dict['bool_refined_bounds'] == True:
        app.build_refined_bounds(app.de_config_dict['original_bounds'])
    elif fea_config_dict['bool_refined_bounds'] == False:
        app.de_config_dict['bounds'] = app.de_config_dict['original_bounds']
    else:
        raise
    print('The final bounds are:')
    for el in app.de_config_dict['bounds']:
        print('\t', el)

    if False == bool_post_processing:
        if fea_config_dict['flag_optimization'] == True:
            app.init_swarm() # define app.sw
            app.run_de()
        else:
            print('Do something.')

    elif True == bool_post_processing:
        app.init_swarm()
        swda = app.best_design_by_weights(fea_config_dict['use_weights'])
        from pylab import show
        show()
        quit()
        run_static_structural_fea(swda.best_design_denorm)


