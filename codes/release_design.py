#coding:u8
#execfile('D:/OneDrive - UW-Madison/c/codes/release_design.py')
execfile('D:/OneDrive - UW-Madison/c/codes/default_setting.py')

run_integer = 142
fea_config_dict['Active_Qr'] = 16
fea_config_dict['run_folder'] = r'run#%d/'%(run_integer) # best individual release
fea_config_dict['jmag_run_list'] = run_list = [1,1,0,0,0] # use JMAG to search for breakdown frequency (serach for "32 # 8" in population.py)
fea_config_dict['flag_optimization'] = False
fea_config_dict['End_Ring_Resistance'] = 9.69e-6 # to release, we have to consider end ring and its effect on the breakdown slip
print build_model_name_prefix(fea_config_dict) # rebuild model name prefix (without NoEndRing)



from utility import *
################################################################
# find best individual
################################################################ 
swda = SwarmDataAnalyzer(run_integer=run_integer)
gen_best, indices, costs = swda.get_best_generation(popsize=30, returnMore=True)
with open('d:/Qr%d_gen_best.txt'%(fea_config_dict['Active_Qr']), 'w') as f:
    f.write('\n'.join(','.join('%.16f'%(x) for x in y) for y in gen_best)) # convert 2d array to string 

best_individual_denorm = gen_best[0]
print '------------------------------------'
print best_individual_denorm, indices[0], costs[0]



################################################################
# Check the shitty design (that fails to draw) or the best design
################################################################
if True:
    sw = population.swarm(fea_config_dict)
    sw.number_current_generation = 0 # no need to call sw.generate_pop()

    # 这些都是不需要的代码，直接利用fobj将individual_denorm转换成im_variant就可以。
    # im_best_design = population.bearingless_induction_motor_design.local_design_variant(sw.im, sw.number_current_generation, 999, best_individual_denorm)
    # individual_denorm = utility.Pyrhonen_design(im_best_design).design_parameters_denorm
    # print 'individual_denorm=', individual_denorm

    # open swarm_data.txt to search for 
    print 'cmd.exe> subl "%s"' % (sw.dir_run+'swarm_data.txt')
    # os.system('subl "%s"' % (sw.dir_run+'swarm_data.txt'))
    
    cost_function = sw.fobj(999, best_individual_denorm)
    im_best_design = sw.im_variant
    print 're-evaluated cost_function=', cost_function

    # duplicate study for Transient FEA Reference
    try:
        # sw.app
        app
        # pass
    except:
        print 'run the code with: app = designer.GetApplication()'
        raise

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
    refarray[3][1] =    400
    refarray[3][2] =        50
    DM.GetDataSet(u"SectionStepTable").SetTable(refarray)
    number_of_total_steps = 1 + 16 + temp_no_steps + 400 # [Double Check] don't forget to modify here!
    study.GetStep().SetValue(u"Step", number_of_total_steps)
    study.GetStep().SetValue(u"StepType", 3)
    study.GetStep().SetTableProperty(u"Division", DM.GetDataSet(u"SectionStepTable"))

    # study.RunAllCases()
    app.Save()

else: # obsolete codes
    if False:
        # Now with this redraw from im.show(toString=True) feature, you can see actually sometimes jmag fails to draw because of PC system level interference, rather than bug in my codes.
        # debug for shitty design that failed to draw
        shitty_design = population.bearingless_induction_motor_design.reproduce_the_problematic_design(r'D:\OneDrive - UW-Madison\c\codes/'+'shitty_design.txt')
        shitty_design.show()
        sw.run(shitty_design)
    elif False:
        # this is not a shitty design. just due to something going wrong calling JMAG remote method
        shitty_design = population.bearingless_induction_motor_design.reproduce_the_problematic_design(r'D:\OneDrive - UW-Madison\c\codes/'+'shitty_design_Qr16_statorCore.txt')        
        sw.number_current_generation = 0
        sw.fobj(99, utility.Pyrhonen_design(shitty_design).design_parameters_denorm)
    else:
        best_design = population.bearingless_induction_motor_design.reproduce_the_problematic_design(r'D:\OneDrive - UW-Madison\c\codes/'+'Qr32_O2Best_Design.txt')
        individual_denorm = utility.Pyrhonen_design(best_design).design_parameters_denorm
        print individual_denorm
        # os.path.mkdirs(sw.dir_parent + 'pop/' + sw.run_folder)
        # open(sw.dir_parent + 'pop/' + sw.run_folder + 'thermal_penalty_individuals.txt', 'w').close()
        sw.number_current_generation = 0
        sw.fobj(99, individual_denorm)
