import pyrhonen_procedure_as_function 
bool_post_processing = False # solve or post-processing

# Situation when default_setting does not match spec may happen. What???
filename = './default_setting.py'
exec(compile(open(filename, "rb").read(), filename, 'exec'), globals(), locals())

文件名 = filename
from math import pi as π
print(文件名)



#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Design Specification
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
p = 2
spec = pyrhonen_procedure_as_function.desgin_specification(
        PS_or_SC = True, # Pole Specific or Squirrel Cage
        DPNV_or_SEPA = True, # Dual purpose no voltage or Separate winding
        p = p,
        ps = 2 if p==1 else 1,
        mec_power = 80e3, # kW
        ExcitationFreq = p*780, # Hz
        ExcitationFreqSimulated = p*500, # Hz This sets to DriveW_Freq that is actually used in FE simulation.
        VoltageRating = 480, # Vrms (line-to-line, Wye-Connect)
        TangentialStress = 12000, # Pa
        Qs = 24,
        Qr = 16,
        Js = 3.7e6, # Arms/m^2
        Jr = 7.5e6,  #7.25e6, #7.5e6, #6.575e6, # Arms/m^2
        Steel = 'M19Gauge29', # Arnon-7
        lamination_stacking_factor_kFe = 0.95, # from http://www.femm.info/wiki/spmloss # 0.91 for Arnon
        Coil = 'Cu',
        space_factor_kCu = 0.5, # Stator slot fill/packign factor
        Conductor = 'Cu',
        space_factor_kAl = 1.0, # Rotor slot fill/packing factor
        Temperature = 75, # deg Celsius
        stator_tooth_flux_density_B_ds = 1.4, # Tesla
        rotor_tooth_flux_density_B_dr  = 1.5, # Tesla
        stator_yoke_flux_density_Bys = 1.1, # Tesla
        rotor_yoke_flux_density_Byr  = 1.1 + 0.3 if p==1 else 1.1, # Tesla
        guess_air_gap_flux_density = 0.8, # 0.8, # Tesla | 0.7 ~ 0.9 | Table 6.3
        guess_efficiency = 0.95,
        guess_power_factor = 0.6, 
        debug_or_release = True, # 如果是debug，数据库里有记录就删掉重新跑；如果release且有记录，那就报错。=debug_or_release = True # 如果是debug，数据库里有记录就删掉重新跑；如果release且有记录，那就报错。
        bool_skew_stator = None,
        bool_skew_rotor = None,
)
# spec.show()
print(spec.build_name())
bool_bad_specifications = spec.pyrhonen_procedure()
print(spec.build_name())

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Automatic Report Generation
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# if not bool_post_processing:
if '730' in fea_config_dict['pc_name']:
    os.system('cd /d '+ r'"D:\OneDrive - UW-Madison\c\release\OneReport\OneReport_TEX" && z_nul"')
# quit()



#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Add to Database
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
if bool_bad_specifications:
    print('\nThe specifiaction can not be fulfilled. Read script log or OneReport.pdf for information and revise the specifiaction for $J_r$ or else your design name is wrong.')
else:
    print('\nThe specifiaction is meet. Now check the database of blimuw if on Y730.')
    if '730' in fea_config_dict['pc_name']:
        utility.communicate_database(spec)

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Automatic Performance Evaluation
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # winding analysis? 之前的python代码利用起来啊
    # 希望的效果是：设定好一个设计，马上进行运行求解，把我要看的数据都以latex报告的形式呈现出来。
    # OP_PS_Qr36_M19Gauge29_DPNV_NoEndRing.jproj
if True:

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 0. FEA Setting / General Information & Packages Loading
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    
    # Run settings
    fea_config_dict['Active_Qr'] = 16
    fea_config_dict['use_weights'] = 'O1' # 'O2' # 'O3'
    if True:
        # global search

        # Prototype OD150 two pole motor
        if False:
            fea_config_dict['local_sensitivity_analysis'] = True
            fea_config_dict['bool_refined_bounds'] = False
            run_folder = r'run#500/' # Sensitivity analysis for Qr=16 and p=1

            fea_config_dict['local_sensitivity_analysis'] = False
            fea_config_dict['bool_refined_bounds'] = True
            run_folder = r'run#501/' # Optimize with refined bounds

        # Prototype OD150 four pole motor
        else:
            # # Prototype OD150 four pole motor
            # fea_config_dict['local_sensitivity_analysis'] = True
            # fea_config_dict['bool_refined_bounds'] = False
            # run_folder = r'run#502/' # Sensitivity analysis for Qr=16 and p=2 (Air gap length is 2*"50Hz delta") ExcitationFreqSimulated = ExcitationFreq = 1500 Hz
            # run_folder = r'run#503/' # Sensitivity analysis for Qr=16 and p=2 (Air gap length is 1.5*"50Hz delta") ExcitationFreqSimulated = ExcitationFreq = 1500 Hz
            # run_folder = r'run#504/' # Sensitivity analysis for Qr=16 and p=2 (Air gap length is 1.5*"50Hz delta") ExcitationFreqSimulated = 1000 Hz



            # fea_config_dict['local_sensitivity_analysis'] = False
            # fea_config_dict['bool_refined_bounds'] = True
            # run_folder = r'run#505/' # Optimize with refined bounds (popsize is 70=7*10 now)

            # fea_config_dict['local_sensitivity_analysis'] = False
            # fea_config_dict['bool_refined_bounds'] = False
            # run_folder = r'run#506/' # Optimize without refined bounds (popsize is 70=7*10 now) the mec_power is changed from 100kW to 75kW.


            # # spec_4poleOD150mm1500Hz82kW.py
            # fea_config_dict['local_sensitivity_analysis'] = False
            # fea_config_dict['bool_refined_bounds'] = False
            # if '730' in fea_config_dict['pc_name']:
            #     fea_config_dict['use_weights'] = 'O1'
            #     run_folder = r'run#50701/'
            # # elif '730' in fea_config_dict['pc_name']:
            # #     fea_config_dict['use_weights'] = 'O2'
            # #     run_folder = r'run#50702/'
            # # elif '730' in fea_config_dict['pc_name']:
            # #     fea_config_dict['use_weights'] = 'O3'
            # #     run_folder = r'run#507/' # Optimize without refined bounds (popsize is 70=7*10 now) the mec_power is changed from 100kW to 75kW. Use weights O3
            # else:
            #    raise

            # # spec_4poleOD150mm1500Hz82kW.py
            # fea_config_dict['local_sensitivity_analysis'] = False
            # fea_config_dict['bool_refined_bounds'] = False
            # fea_config_dict['use_weights'] = 'O1'
            # run_folder = r'run#50901/' # PF is now set to 0.6 but OD is over 150 mm

            # # spec_4poleOD150mm1500Hz82kW.py
            # fea_config_dict['local_sensitivity_analysis'] = False
            # fea_config_dict['bool_refined_bounds'] = False
            # fea_config_dict['use_weights'] = 'O1R'
            # run_folder = r'run#510/' # PF is now set to 0.6 but OD is over 150 mm

            # # spec_4poleOD150mm1500Hz82kW.py
            # fea_config_dict['local_sensitivity_analysis'] = False
            # fea_config_dict['bool_refined_bounds'] = False
            # fea_config_dict['use_weights'] = 'O1'
            # run_folder = r'run#511/' # PF is now set to 0.6 and OD < 150 mm

            # # fea_config_dict['use_weights'] = 'O4'
            # # if which == 'O4':
            # #     return [ 2,2,1,1,1,  0 ]            
            # run_folder = r'run#512/' # Searching for low torque ripple
            # run_folder = r'run#513/' # Searching for low torque ripple
            # run_folder = r'run#514/' # Searching for low torque ripple
            # run_folder = r'run#515/' # Searching for low torque ripple
            # run_folder = r'run#516/' # Searching for low torque ripple

            # run_folder = r'run#517/' # 修改round bar代码
            # run_folder = r'run#518/' # 修改round bar代码（smaller radius of rotor slot is used now)

            fea_config_dict['local_sensitivity_analysis'] = True
            fea_config_dict['bool_refined_bounds'] = False
            fea_config_dict['use_weights'] = 'O2'
            run_folder = r'run#520/' # 修改round bar代码（smaller radius of rotor slot is used now)

    else:
        # local tuning (add 99 suffix)

        # run#511gen#0000ind#0025-ID16-0-25
        # 6.84866,1.07652,1.77721,8.64138,1.05135,2.63571,2.12527        
        # spec_4poleOD150mm1500Hz82kW.py
        fea_config_dict['local_sensitivity_analysis'] = False
        fea_config_dict['bool_refined_bounds'] = -1 # Manually set the bounds
        fea_config_dict['use_weights'] = 'O1'
        # First tune
        run_folder = r'run#51199/'
        # Second tune
        run_folder = r'run#51198/' # Rotor slot open depth d ro 1.00 0.95 0.95 1.16 (low bound is reached)
        # Third tune
        run_folder = r'run#51197/' # Rotor slot open depth d ro 1.00 0.76 0.76 0.95 (low bound is reached again)

        # look for low torque ripple design
        run_folder = r'run#51196/' 

    # run folder
    fea_config_dict['run_folder'] = run_folder
    # logging file
    logger = utility.myLogger(fea_config_dict['dir_codes'], prefix='ones_'+fea_config_dict['run_folder'][:-1])
    # rebuild the name
    build_model_name_prefix(fea_config_dict)

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 1. Bounds for DE optimiazation
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # save some time when degbuging
    loc_txt_file = '../' + 'pop/' + 'loop_for_bounds_%s.txt'%(run_folder[:-1])
    if not os.path.exists(loc_txt_file):
        import copy
        spec_latch = copy.deepcopy(spec) # 后面调用loop_for_bounds的时候回覆盖掉这里的initial_design的结果的。
        # run for tooth width bounds
        loc_txt_file = pyrhonen_procedure_as_function.loop_for_bounds(spec, run_folder)
        spec = spec_latch
    # read in tooth width data
    list_b_dr = []
    list_b_ds = []
    with open(loc_txt_file, 'r') as f:
        buf = f.readlines()
        for row in buf:
            design = [float(el) for el in row.split(',')]
            list_b_ds.append(design[15])
            list_b_dr.append(design[30]*1e3)
    b_ds_max, b_ds_min = max(list_b_ds), min(list_b_ds)
    b_dr_max, b_dr_min = max(list_b_dr), min(list_b_dr)
    print(b_ds_max, b_ds_min, 'initial design:', 1e3*spec.stator_tooth_width_b_ds, 'mm')
    print(b_dr_max, b_dr_min, 'initial design:', 1e3*spec.rotor_tooth_width_b_dr, 'mm')
    if b_ds_min<4:
        print('Too small lower bound b_ds (%g) is detected. Set it to 2.5 mm.'%(b_ds_min))
        b_ds_min = 4
    if b_dr_min<4:
        print('Too small lower bound b_dr (%g) is detected. Set it to 2.5 mm.'%(b_dr_min))
        b_dr_min = 4
    if 1e3*spec.stator_tooth_width_b_ds < b_ds_min  and 1e3*spec.stator_tooth_width_b_ds > b_ds_max:
        raise Exception('The initial design is not within the bounds.')
    if 1e3*spec.rotor_tooth_width_b_dr < b_dr_min  and 1e3*spec.rotor_tooth_width_b_dr > b_dr_max:
        raise Exception('The initial design is not within the bounds.')

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 2. Initilize Swarm and Initial Pyrhonen's Design (Run this part in JMAG)
#    Bounds: 1e-1也还是太小了（第三次报错），至少0.5mm长吧 
#    # 1e-1 is the least geometry value. 
#    A 1e-2 will leads to：转子闭口槽极限，会导致edge过小，从而报错：small arc entity exists.png
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    if fea_config_dict['flag_optimization'] == True:
        de_config_dict = {  'original_bounds':[ [b_ds_min, b_ds_max],#--# stator_tooth_width_b_ds
                                                [     0.8,        3],   # air_gap_length_delta
                                                [    5e-1,        3],   # Width_RotorSlotOpen 
                                                [b_dr_min, b_dr_max],#--# rotor_tooth_width_b_dr # It allows for large rotor tooth because we raise Jr and recall this is for less slots---Qr=16.
                                                [    5e-1,        3],   # Length_HeadNeckRotorSlot
                                                [       1,       11],   # Angle_StatorSlotOpen
                                                [    5e-1,        3] ], # Width_StatorTeethHeadThickness
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
                                                        [] ],
                            'bounds':[]}

        print('the auto bounds are:', de_config_dict['original_bounds'])
        if '511' in run_folder or '508' in run_folder:
            de_config_dict['original_bounds'][0][0] += 2.5
            de_config_dict['original_bounds'][3][0] += 2.5
            de_config_dict['original_bounds'][0][1] += 2
            de_config_dict['original_bounds'][3][1] += 2
            print('the extended bounds are:', de_config_dict['original_bounds'])
        # quit()

        # Sensitivity Analysis based narrowing bounds
        if fea_config_dict['bool_refined_bounds'] == True:
            numver_of_variants = 20.0
            if fea_config_dict['Active_Qr'] == 16: 

                # from utility.py:run500 O1
                raw_narrow_bounds = [   [15, 16, 17, 18, 19, 20],
                                        [6, 7, 8, 9, 10, 1,1, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                        [0, 1, 5, 6, 7, 8],
                                        [19, 28], # 这里出了BUG，由于loop_for_bound没有遍历到initial design的Jr bds bdr组合，导致initial design的转子齿=5.38mm大于该边界上界5.05mm，结果是，所有改变转子齿的design variant都比initial design差，所以这里特地加大上界。
                                        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                        [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                        [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
                    # original: [[2, 4.483629], [0.8, 3], [0.5, 3], [2, 5.042], [0.5, 3], [1, 10], [0.5, 3]], 
                    # refined:  [[3.86, 4.483629], [1.46, 3.0], [0.5, 1.5], [4.89, 6.26], [0.5, 3.0], [2.8, 10.0], [0.5, 3.0]]}

                # from utility.py:run504 O2
                raw_narrow_bounds = [   [13, 15],
                                        [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                        [1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                        [14, 15, 18, 19, 20],
                                        [0, 1, 2, 3, 4, 5],
                                        [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                        [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]

            for ind, bound in enumerate(raw_narrow_bounds):
                de_config_dict['narrow_bounds_normalized'][ind].append(bound[0] /numver_of_variants)
                de_config_dict['narrow_bounds_normalized'][ind].append(bound[-1]/numver_of_variants)

                if de_config_dict['narrow_bounds_normalized'][ind][0] == de_config_dict['narrow_bounds_normalized'][ind][1]:
                    raise Exception('Upper bound equals to lower bound. Take a check here.')
                    print('ind=',ind, '---manually set the proper bounds based on the initial design: 7.00075,1.26943,0.924664,4.93052,1,3,1')
                    de_config_dict['narrow_bounds_normalized'][ind][0] = 4.93052 / 5

            for bnd1, bnd2 in zip(de_config_dict['original_bounds'], de_config_dict['narrow_bounds_normalized']):
                diff = bnd1[1] - bnd1[0]
                de_config_dict['bounds'].append( [ bnd1[0]+diff*bnd2[0] , bnd1[0]+diff*bnd2[1] ]) # 注意，都是乘以original_bounds的上限哦！

            print('original_bounds:', de_config_dict['original_bounds'])
            print('refined bounds:', de_config_dict['bounds'])
            print('narrow_bounds_normalized:', de_config_dict['narrow_bounds_normalized'])
        elif fea_config_dict['bool_refined_bounds'] == -1:
            print('Manually set the bounds... Before:', de_config_dict['original_bounds'])
            if '51199' in fea_config_dict['run_folder']:
                best_design_denorm = [6.84866,1.07652,1.77721,8.64138,1.05135,2.63571,2.12527] # best design from run#511
                for ind, geom_param in enumerate(best_design_denorm):
                    de_config_dict['original_bounds'][ind] = [geom_param*0.9, geom_param*1.1]
            elif '51198' in fea_config_dict['run_folder']:
                best_design_denorm = [6.71879,1.01978,1.76687,8.33909,0.946215,2.47339,2.08366] # best design from run#51199
                de_config_dict['original_bounds'][0] = [best_design_denorm[0]*0.9, best_design_denorm[0]*1.1]
                de_config_dict['original_bounds'][1] = [best_design_denorm[1]*0.9, best_design_denorm[1]*1.1]
                de_config_dict['original_bounds'][2] = [best_design_denorm[2]*0.9, best_design_denorm[2]*1.1]
                de_config_dict['original_bounds'][3] = [best_design_denorm[3]*0.9, best_design_denorm[3]*1.1]
                de_config_dict['original_bounds'][4] = [best_design_denorm[4]*0.8, best_design_denorm[4]*1.0]
                de_config_dict['original_bounds'][5] = [best_design_denorm[5]*0.9, best_design_denorm[5]*1.1]
                de_config_dict['original_bounds'][6] = [best_design_denorm[6]*0.9, best_design_denorm[6]*1.1]
            elif '51197' in fea_config_dict['run_folder']:
                best_design_denorm = [6.66737,0.967267,1.7789,8.86449,0.756972,2.46541,2.08662] # best design from run#51198
                de_config_dict['original_bounds'][0] = [best_design_denorm[0]*0.9, best_design_denorm[0]*1.1]
                de_config_dict['original_bounds'][1] = [best_design_denorm[1]*0.9, best_design_denorm[1]*1.1]
                de_config_dict['original_bounds'][2] = [best_design_denorm[2]*0.9, best_design_denorm[2]*1.1]
                de_config_dict['original_bounds'][3] = [best_design_denorm[3]*0.9, best_design_denorm[3]*1.1]
                de_config_dict['original_bounds'][4] = [best_design_denorm[4]*0.7, best_design_denorm[4]*1.0]
                de_config_dict['original_bounds'][5] = [best_design_denorm[5]*0.9, best_design_denorm[5]*1.1]
                de_config_dict['original_bounds'][6] = [best_design_denorm[6]*0.9, best_design_denorm[6]*1.1]
            elif '51196' in fea_config_dict['run_folder']:
                best_design_denorm = [4.48363,1.46,1.19787,6.25996,0.506877,9.25457,0.5] # best design from run#501 (it is originally a 2 pole motor)
                de_config_dict['original_bounds'][0] = [best_design_denorm[0]*0.9, best_design_denorm[0]*1.1]
                de_config_dict['original_bounds'][1] = [best_design_denorm[1]*0.9, best_design_denorm[1]*1.1]
                de_config_dict['original_bounds'][2] = [best_design_denorm[2]*0.9, best_design_denorm[2]*1.1]
                de_config_dict['original_bounds'][3] = [best_design_denorm[3]*0.9, best_design_denorm[3]*1.1]
                de_config_dict['original_bounds'][4] = [0.5, best_design_denorm[4]*1.1]
                de_config_dict['original_bounds'][5] = [best_design_denorm[5]*0.9, best_design_denorm[5]*1.1]
                de_config_dict['original_bounds'][6] = [0.5, best_design_denorm[6]*1.1]
            else:
                raise
            print('Manually set the bounds... After:', de_config_dict['original_bounds'])
            print(best_design_denorm)
            de_config_dict['bounds'] = de_config_dict['original_bounds']
            # quit()
        else:
            print('No refined bounds are applied.')
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
    print(de_config_dict)

    # init the swarm
    sw = population.swarm(fea_config_dict, de_config_dict=de_config_dict)
    # sw.show(which='all')
    # print sw.im.show(toString=True)
    # quit()

    # Sensitivity Bar Charts
    if fea_config_dict['local_sensitivity_analysis'] == True:
        if os.path.exists(fea_config_dict['dir_parent'] + 'pop/' + run_folder + 'swarm_data.txt'):
            import utility
            try:
                utility.build_sensitivity_bar_charts(spec, sw)
                quit()
            except:
                os.remove(fea_config_dict['dir_parent'] + 'pop/' + run_folder + 'swarm_data.txt')
                print('Remove ' + fea_config_dict['dir_parent'] + 'pop/' + run_folder + 'swarm_data.txt')
                print('Continue for sensitivity analysis...')
        else:
            print('Now begin to generate results for sensitivity analysis...')

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
    if False == bool_post_processing:
        
        logger.debug('-------------------------count_abort=%d' % (count_abort))

        # if optimization_flat == True:
        # generate the initial generation
        sw.generate_pop()
        # quit()

        # add initial_design of Pyrhonen09 to the initial generation
        if sw.fea_config_dict['local_sensitivity_analysis'] == False:
            if count_abort == 0:
                utility.add_Pyrhonen_design_to_first_generation(sw, de_config_dict, logger)

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

    elif True == bool_post_processing:

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 5. Post-processing
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        import utility
        swda = utility.build_Pareto_plot(spec, sw)
        # sw.fobj(999, individual_denorm)

        # Final LaTeX Report
        print('According to Pyrhonen09, 300 MPa is the typical yield stress of iron core.')
        initial_design = utility.Pyrhonen_design(sw.im, de_config_dict['bounds'])
        print (initial_design.design_parameters_denorm)
        print (swda.best_design_denorm)
        print (de_config_dict['bounds'])

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
        lower_bounds = [el[0] for el in de_config_dict['bounds']]
        upper_bounds = [el[1] for el in de_config_dict['bounds']]
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
                    Stator tooth width $w_{st}$             & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ \\
                    Air gap length $L_g$                    & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ \\
                    Rotor slot open width $w_{ro}$          & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ \\
                    Rotor tooth width $w_{rt}$              & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ \\
                    Rotor slot open depth $d_{ro}$          & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ \\
                    Stator slot open angle $\theta_{so}$    & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ \\
                    Stator slot open depth $d_{so}$         & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ \\
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
                    ''' % (fea_config_dict['use_weights']) + latex_table, file=fname)
        print(swda.str_best_design_details, file=fname)
        fname.close()

        os.system('cd /d '+ r'"D:\OneDrive - UW-Madison\c\release\OneReport\BestReport_TEX" && z_nul"') # 必须先关闭文件！否则编译不起来的
        # import subprocess
        # subprocess.call(r"D:\OneDrive - UW-Madison\c\release\OneReport\OneReport_TEX\z_nul", shell=True)

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 6. Check mechanical strength for the best design
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        from pylab import show
        show()
        quit()

        im_best = population.bearingless_induction_motor_design.local_design_variant(sw.im, \
                     -1, -1, swda.best_design_denorm)

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
        quit()

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


