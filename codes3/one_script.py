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
    print('\nThe specifiaction is meet. Check database and add record.')
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
    # Radius_OuterRotor = pyrhonen_procedure_as_function.loop_for_bounds(spec)

    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # 1. FEA Setting / General Information & Packages Loading
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    filename = './default_setting.py'
    exec(compile(open(filename, "rb").read(), filename, 'exec'), globals(), locals())

    fea_config_dict['Active_Qr'] = 16
    fea_config_dict['use_weights'] = 'O2'
    fea_config_dict['use_weights'] = 'O1'
    fea_config_dict['run_folder'] = r'run#500/' # 
    logger = utility.myLogger(fea_config_dict['dir_codes'], prefix='ones_'+fea_config_dict['run_folder'][:-1])

    # rebuild the name
    build_model_name_prefix(fea_config_dict)


sw = population.swarm(fea_config_dict, de_config_dict=de_config_dict)



# initial design
# call pyrhonen to given 



