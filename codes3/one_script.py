p = 1
p2 = 2
Qs = 24
Qr = 14
Js
Jr
Steel
lamination_stacking_factor_kFe
Temperature = 75 # deg Celsius
stator_tooth_flux_density_B_ds
rotor_tooth_flux_density_B_dr
stator_yoke_flux_density_Bys = 1.2
rotor_yoke_flux_density_Byr = 1.1 + 0.3 if p==1 else 1.1
VoltageRating
list_guesses = [alpha_i, efficiency, power_factor]
space_factor_kCu
space_factor_kAl

debug_or_release = True # 如果是debug，数据库里有记录就删掉重新跑；如果release且有记录，那就报错。

# winding analysis? 之前的python代码利用起来啊

希望的效果是：设定好一个设计，马上进行运行求解，把我要看的数据都以latex报告的形式呈现出来。


OP_PS_Qr36_M19Gauge29_DPNV_NoEndRing.jproj

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# 1. General Information & Packages Loading
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

if fea_config_dict['pc_name'] == 'Y730':
    # Check the database
    import mysql.connector
    db = mysql.connector.connect(
        host ='localhost',
        user ='root',
        passwd ='password123',
        database ='blimuw',
        )
    cursor = db.cursor()

# initial design
# call pyrhonen to given 

