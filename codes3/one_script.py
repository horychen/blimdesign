import pyrhonen_procedure_as_function 

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Design Specification
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
p = 1
spec = pyrhonen_procedure_as_function.desgin_specification(
        PS_or_SC = True, # Pole Specific or Squirrel Cage
        DPNV_or_SEPA = True, # Dual purpose no voltage or Separate winding
        p = p,
        ps = 2 if p==1 else 1,
        mec_power = 100e3, # kW
        ExcitationFreq = 880, # Hz
        VoltageRating = 480, # Vrms (line-to-line, Wye-Connect)
        TangentialStress = 12000, # Pa
        Qs = 24,
        Qr = 16,
        Js = 3.7e6, # Arms/m^2
        Jr = 7.25e6, #7.5e6, #6.575e6, # Arms/m^2
        Steel = 'M19Gauge29', # Arnon-7
        lamination_stacking_factor_kFe = 0.95, # from http://www.femm.info/wiki/spmloss # 0.91 for Arnon
        Coil = 'Cu',
        space_factor_kCu = 0.5, # Stator slot fill/packign factor
        Conductor = 'Cu',
        space_factor_kAl = 1.0, # Rotor slot fill/packing factor
        Temperature = 75, # deg Celsius
        stator_tooth_flux_density_B_ds = 1.4, # Tesla
        rotor_tooth_flux_density_B_dr  = 1.5, # Tesla
        stator_yoke_flux_density_Bys = 1.2, # Tesla
        rotor_yoke_flux_density_Byr  = 1.1 + 0.3 if p==1 else 1.1, # Tesla
        guess_air_gap_flux_density = 0.8, # 0.8, # Tesla | 0.7 ~ 0.9 | Table 6.3
        guess_efficiency = 0.95,
        guess_power_factor = 0.7,
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
import os
os.system('cd /d '+ r'"D:\OneDrive - UW-Madison\c\release\OneReport\OneReport_TEX" && z_nul"')

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Add to Database
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
if bool_bad_specifications:
    print('\nThe specifiaction can not be fulfilled. Read script log or OneReport.pdf for information and revise the specifiaction for $J_r$ or else your design name is wrong.')
else:
    print('\nThe specifiaction is meet. Now check the database of blimuw.')
    try:
        import mysql.connector
    except:
        print('MySQL python connector is not installed. Skip database communication.')
    else:
        db = mysql.connector.connect(
            host ='localhost',
            user ='root',
            passwd ='password123',
            database ='blimuw',
            )
        cursor = db.cursor()
        cursor.execute('SELECT name FROM designs')
        result = cursor.fetchall()
        if spec.build_name() not in [row[0] for row in result]:
            def sql_add_one_record(spec):
                # Add one record
                sql = "INSERT INTO designs " \
                    + "(" \
                        + "name, " \
                            + "PS_or_SC, " \
                            + "DPNV_or_SEPA, " \
                            + "p, " \
                            + "ps, " \
                            + "MecPow, " \
                            + "Freq, " \
                            + "Voltage, " \
                            + "TanStress, " \
                            + "Qs, " \
                            + "Qr, " \
                            + "Js, " \
                            + "Jr, " \
                            + "Coil, " \
                            + "kCu, " \
                            + "Condct, " \
                            + "kAl, " \
                            + "Temp, " \
                            + "Steel, " \
                            + "kFe, " \
                            + "Bds, " \
                            + "Bdr, " \
                            + "Bys, " \
                            + "Byr, " \
                            + "G_b, " \
                            + "G_eta, " \
                            + "G_PF, " \
                            + "debug, " \
                            + "Sskew, " \
                            + "Rskew, " \
                            + "Pitch " \
                    + ") VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                record = (  spec.build_name(), 
                            'PS' if spec.PS_or_SC else 'SC',
                            'DPNV' if spec.DPNV_or_SEPA else 'SEPA',
                            spec.p,
                            spec.ps,
                            spec.mec_power,
                            spec.ExcitationFreq,
                            spec.VoltageRating,
                            spec.TangentialStress,
                            spec.Qs,
                            spec.Qr,
                            spec.Js,
                            spec.Jr,
                            spec.Coil,
                            spec.space_factor_kCu,
                            spec.Conductor,
                            spec.space_factor_kAl,
                            spec.Temperature,
                            spec.Steel,
                            spec.lamination_stacking_factor_kFe,
                            spec.stator_tooth_flux_density_B_ds,
                            spec.rotor_tooth_flux_density_B_dr,
                            spec.stator_yoke_flux_density_Bys,
                            spec.rotor_yoke_flux_density_Byr,
                            spec.guess_air_gap_flux_density,
                            spec.guess_efficiency,
                            spec.guess_power_factor,
                            spec.debug_or_release,
                            spec.bool_skew_stator,
                            spec.bool_skew_rotor,
                            spec.winding_layout.coil_pitch
                        )
                cursor.execute(sql, record)
                db.commit()
            sql_add_one_record(spec)
            'A new record is added to table named designs.'
        else:
            'Record already exists, skip database communication.'
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Automatic Performance Evaluation
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # winding analysis? 之前的python代码利用起来啊
    # 希望的效果是：设定好一个设计，马上进行运行求解，把我要看的数据都以latex报告的形式呈现出来。
    # OP_PS_Qr36_M19Gauge29_DPNV_NoEndRing.jproj
if True:

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 0. Bounds
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # loc_txt_file = pyrhonen_procedure_as_function.loop_for_bounds(spec)
    loc_txt_file = '../' + 'pop/' + r'loop_for_bounds.txt'
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
    print(b_ds_max, b_ds_min)
    print(b_dr_max, b_dr_min)
    if b_ds_min<2:
        print('Too small lower bound b_ds (%g) is detected. Set it to 2 mm.'%(b_ds_min))
        b_ds_min = 2
    if b_dr_min<2:
        print('Too small lower bound b_dr (%g) is detected. Set it to 2 mm.'%(b_dr_min))
        b_dr_min = 2

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 1. FEA Setting / General Information & Packages Loading
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # Situation when default_setting does not match spec may happen
    filename = './default_setting.py'
    exec(compile(open(filename, "rb").read(), filename, 'exec'), globals(), locals())


    if True: # ECCE
        fea_config_dict['Active_Qr'] = 16
        fea_config_dict['use_weights'] = 'O1' # 'O2' # 'O3'

            # fea_config_dict['local_sensitivity_analysis'] = True
            # run_folder = r'run#400/' # Sensitivity analysis for Qr=16, T440p
            # run_folder = r'run#140/' # Sensitivity analysis for Qr=16, Y730

            # fea_config_dict['local_sensitivity_analysis'] = False
            # run_folder = r'run#141/' # run for reference
            # run_folder = r'run#142/' # optimize Qr=16 for O2
            # run_folder = r'run#143/' # test for shitty design (not true)
            # run_folder = r'run#144/' # optimize Qr=16 for O1

        # Prototype OD150
        fea_config_dict['local_sensitivity_analysis'] = True
        fea_config_dict['bool_refined_bounds'] = False
        run_folder = r'run#500/' # Sensitivity analysis for Qr=16

        fea_config_dict['local_sensitivity_analysis'] = False
        fea_config_dict['bool_refined_bounds'] = True
        run_folder = r'run#501/' # Optimize with refined bounds

    # run folder
    fea_config_dict['run_folder'] = run_folder
    # logging file
    logger = utility.myLogger(fea_config_dict['dir_codes'], prefix='ones_'+fea_config_dict['run_folder'][:-1])
    # rebuild the name
    build_model_name_prefix(fea_config_dict)



#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 2. Initilize Swarm and Initial Pyrhonen's Design (Run this part in JMAG)
#    Bounds: 1e-1也还是太小了（第三次报错），至少0.5mm长吧 
#    # 1e-1 is the least geometry value. 
#    A 1e-2 will leads to：转子闭口槽极限，会导致edge过小，从而报错：small arc entity exists.png
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    if fea_config_dict['flag_optimization'] == True:
        if True: # 2 pole motor
            de_config_dict = {  'original_bounds':[ [b_ds_min, b_ds_max],#--# stator_tooth_width_b_ds
                                                    [     0.8,        3],   # air_gap_length_delta
                                                    [    5e-1,        3],   # Width_RotorSlotOpen 
                                                    [b_dr_min, b_dr_max],#--# rotor_tooth_width_b_dr # It allows for large rotor tooth because we raise Jr and recall this is for less slots---Qr=16.
                                                    [    5e-1,        3],   # Length_HeadNeckRotorSlot
                                                    [       1,       10],   # Angle_StatorSlotOpen
                                                    [    5e-1,        3] ], # Width_StatorTeethHeadThickness
                                'mut':        0.8,
                                'crossp':     0.7,
                                'popsize':    35, # 5~10 \times number of geometry parameters --JAC223
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
    print(de_config_dict)

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
        
        logger.debug('-------------------------count_abort=%d' % (count_abort))

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


