#coding:u8
filename = './default_setting.py'
exec(compile(open(filename, "rb").read(), filename, 'exec'), globals(), locals())

在最优设计附近进行搜索，比如加大齿宽，如果发现目标函数O1变好了，那就继续加大，直到O1变差，停止，然后以此类推。




fea_config_dict['Active_Qr'] = 16

if fea_config_dict['Active_Qr'] == 32:
    # run_integer = '120' # minimize O1 with narrowed bounds according to sensitivity analysis (run#116)
    run_integer = 121 # minimize O2 with narrowed bounds according to sensitivity analysis (run#116)
    fea_config_dict['use_weights'] = 'O2'

    dir_csv_output_folder = r'D:\OneDrive - UW-Madison\c\csv\PS_Qr32_M19Gauge29_DPNV_1e3Hz/'
    freq_study_name = 'PS_Qr32_M19Gauge29_DPNV_1e3Hz_ID32-0-99Freq'
    tran_study_name = 'PS_Qr32_M19Gauge29_DPNV_1e3Hz_ID32-0-99TranRef2'

elif fea_config_dict['Active_Qr'] == 16:
    run_integer = 142 # optimize Qr=16 for O2
    fea_config_dict['use_weights'] = 'O1'

individual_index = 999
fea_config_dict['run_folder'] = r'run#%d/'%(run_integer) # best individual release
fea_config_dict['jmag_run_list'] = run_list = [1,1,0,0,0] # use JMAG to search for breakdown frequency (serach for "32 # 8" in population.py)
fea_config_dict['flag_optimization'] = False # must be set correct
fea_config_dict['End_Ring_Resistance'] = 9.69e-6 # to release, we have to consider end ring and its effect on the breakdown slip
print(build_model_name_prefix(fea_config_dict)) # rebuild model name prefix (without NoEndRing)


def local_sensitivity_analysis(self):
    # 敏感性检查：以基本设计为准，检查不同的参数取极值时的电机性能变化！这是最简单有效的办法。七个设计参数，那么就有14种极值设计。
    initial_design_denorm = np.array( utility.Pyrhonen_design(self.im).design_parameters_denorm )
    initial_design = (initial_design_denorm - min_b) / diff
    print(initial_design_denorm.tolist())
    print(initial_design.tolist())
    base_design = initial_design.tolist()
    print('base_design:', base_design, '\n-------------')
    # quit()
    number_of_variants = 20
    self.init_pop = [initial_design] # include initial design!
    for i in range(len(base_design)): # 7 design parameters
        for j in range(number_of_variants+1): # 21 variants interval
            # copy list
            design_variant = base_design[::]
            design_variant[i] = j * 1./number_of_variants
            self.init_pop.append(design_variant)
    # for ind, el in enumerate(self.init_pop):
    #     print ind, el

if self.fea_config_dict['local_sensitivity_analysis'] == True:
    local_sensitivity_analysis(self)


