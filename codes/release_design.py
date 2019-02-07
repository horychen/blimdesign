#coding:u8
#execfile('D:/OneDrive - UW-Madison/c/codes/release_design.py')
execfile('D:/OneDrive - UW-Madison/c/codes/default_setting.py')


fea_config_dict['Active_Qr'] = 32

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
print build_model_name_prefix(fea_config_dict) # rebuild model name prefix (without NoEndRing)



################################################################
# find best individual
################################################################ 
swda = utility.SwarmDataAnalyzer(run_integer=run_integer)
gen_best, indices, costs = swda.get_best_generation(popsize=30, returnMore=True)

best_individual_denorm = gen_best[0]
print '------------------------------------'
print best_individual_denorm, indices[0], costs[0]



################################################################
# Run Transient FEA Reference for the best design
################################################################
no_steps_half_period_TranRef = 400
if True:
    sw = population.swarm(fea_config_dict)
    sw.number_current_generation = 0 # no need to call sw.generate_pop()


    # open swarm_data.txt to search for 
    print 'cmd.exe> subl "%s"' % (sw.dir_run+'swarm_data.txt')
    os.system('subl "%s"' % (sw.dir_run+'swarm_data.txt'))

    try:
        # sw.app
        app
        # pass
    except:
        print 'run the code with: app = designer.GetApplication()'
        print 'if you have results already, just go on.s'

        # 这些都是不需要的代码，直接利用fobj将individual_denorm转换成im_variant就可以。
        # 当然，如果你一开始就不需要调用fobj的话，do this here
        im_best_design = population.bearingless_induction_motor_design.local_design_variant(sw.im, sw.number_current_generation, individual_index, best_individual_denorm)
        # individual_denorm = utility.Pyrhonen_design(im_best_design).design_parameters_denorm
        # print 'individual_denorm=', individual_denorm
    else:    
        # build frequency study and TranFEA2TSS study 
        cost_function = sw.fobj(individual_index, best_individual_denorm)
        im_best_design = sw.im_variant
        print 're-evaluated cost_function=', cost_function

        # duplicate study for Transient FEA Reference
        model = app.GetCurrentModel()
        study = app.GetCurrentStudy()
        # print study.GetName()
        # slip_freq_breakdown_torque, _, _ = sw.check_csv_results(study.GetName())
        # print 'slip_freq_breakdown_torque=', slip_freq_breakdown_torque
        slip_freq_breakdown_torque = im_best_design.slip_freq_breakdown_torque
        print 'slip_freq_breakdown_torque=', slip_freq_breakdown_torque


        tranRef_study_name = u"TranRef"
        model.DuplicateStudyName(study.GetName(), tranRef_study_name) # equivalent: # model.DuplicateStudyWithType(study.GetName(), u"Transient2D", tranRef_study_name)
        app.SetCurrentStudy(tranRef_study_name)
        study = app.GetCurrentStudy()


        # 为了让Tran2TSS的平均转矩达到稳态，我们跑一半的滑差周期
        temp_no_steps = sw.fea_config_dict['TranRef-StepPerCycle'] * (1.0-0.5)/slip_freq_breakdown_torque*im_best_design.DriveW_Freq
        # print slip_freq_breakdown_torque
        # print temp_no_steps
        # print (1.0-0.5)/slip_freq_breakdown_torque / (1./sw.im.DriveW_Freq)
        # raise
        DM = app.GetDataManager()
        DM.CreatePointArray(u"point_array/timevsdivision", u"SectionStepTable")
        refarray = [[0 for i in range(3)] for j in range(4)]
        refarray[0][0] = 0
        refarray[0][1] =    1
        refarray[0][2] =        50
        refarray[1][0] = 0.5/slip_freq_breakdown_torque #0.5 for 17.1.03l # 1 for 17.1.02y
        refarray[1][1] =    16                          # 16 for 17.1.03l #32 for 17.1.02y
        refarray[1][2] =        50
        refarray[2][0] = 1.0/slip_freq_breakdown_torque 
        refarray[2][1] =    temp_no_steps
        refarray[2][2] =        50
        refarray[3][0] = refarray[2][0] + 0.5/im_best_design.DriveW_Freq #半周期0.5 for 17.1.03l 
        refarray[3][1] =    no_steps_half_period_TranRef
        refarray[3][2] =        50
        DM.GetDataSet(u"SectionStepTable").SetTable(refarray)
        number_of_total_steps = 1 + 16 + temp_no_steps + no_steps_half_period_TranRef # [Double Check] don't forget to modify here!
        study.GetStep().SetValue(u"Step", number_of_total_steps)
        study.GetStep().SetValue(u"StepType", 3)
        study.GetStep().SetTableProperty(u"Division", DM.GetDataSet(u"SectionStepTable"))

        # study.RunAllCases()
        app.Save()

        # obsolete codes
        # if False:
        #     # Now with this redraw from im.show(toString=True) feature, you can see actually sometimes jmag fails to draw because of PC system level interference, rather than bug in my codes.
        #     # debug for shitty design that failed to draw
        #     shitty_design = population.bearingless_induction_motor_design.reproduce_the_problematic_design(r'D:\OneDrive - UW-Madison\c\codes/'+'shitty_design.txt')
        #     shitty_design.show()
        #     sw.run(shitty_design)
        # elif False:
        #     # this is not a shitty design. just due to something going wrong calling JMAG remote method
        #     shitty_design = population.bearingless_induction_motor_design.reproduce_the_problematic_design(r'D:\OneDrive - UW-Madison\c\codes/'+'shitty_design_Qr16_statorCore.txt')        
        #     sw.number_current_generation = 0
        #     sw.fobj(99, utility.Pyrhonen_design(shitty_design).design_parameters_denorm)
        # else:
        #     best_design = population.bearingless_induction_motor_design.reproduce_the_problematic_design(r'D:\OneDrive - UW-Madison\c\codes/'+'Qr32_O2Best_Design.txt')
        #     individual_denorm = utility.Pyrhonen_design(best_design).design_parameters_denorm
        #     print individual_denorm
        #     # os.path.mkdirs(sw.dir_parent + 'pop/' + sw.run_folder)
        #     # open(sw.dir_parent + 'pop/' + sw.run_folder + 'thermal_penalty_individuals.txt', 'w').close()
        #     sw.number_current_generation = 0
        #     sw.fobj(99, individual_denorm)



################################################################
# Check the results
################################################################
fea_config_dict['number_of_steps_2ndTTS'] = no_steps_half_period_TranRef # 400 # used for computing power factor

slip_freq_breakdown_torque, breakdown_torque, breakdown_force = utility.check_csv_results_4_general_purpose(freq_study_name, dir_csv_output_folder)
print slip_freq_breakdown_torque, breakdown_torque, breakdown_force

from FEMM_Solver import FEMM_Solver
femm_solver = FEMM_Solver(sw.im, flag_read_from_jmag=False, freq=2.23)

# fig_main, axeses = subplots(2, 2, sharex=True, dpi=150, figsize=(16, 8), facecolor='w', edgecolor='k')
axeses = sw.axeses
utility.pyplot_clear(axeses)

str_results, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle, jmag_loss_list, femm_loss_list, power_factor, total_loss, cost_function = \
        utility.build_str_results(axeses, im_best_design, sw.get_project_name(), tran_study_name, dir_csv_output_folder, fea_config_dict, femm_solver)
print str_results

from pylab import show
show()


