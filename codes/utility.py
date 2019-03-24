# coding:u8
import os
import logging
import datetime
import operator

def get_max_and_index(the_list):
    return max(enumerate(the_list), key=operator.itemgetter(1))
    # index, max_value

def get_min_and_index(the_list):
    return min(enumerate(the_list), key=operator.itemgetter(1))
    # index, min_value

def gcd(a,b):
    while b:
        a,b = b, a%b
    return a
# Now find LCM using GCD
def lcm(a,b):
    return a*b // gcd(a,b)
# print lcm(8,7.5)

def myLogger(dir_codes, prefix='opti_script_'): # This works even when the module is reloaded (which is not the case of the other answers) https://stackoverflow.com/questions/7173033/duplicate-log-output-when-using-python-logging-module
    logger=logging.getLogger()
    if not len(logger.handlers):
        logger.setLevel(logging.DEBUG)
        now = datetime.datetime.now()

        if not os.path.isdir(dir_codes + 'log/'):
            os.makedirs(dir_codes + 'log/')

        # create a file handler
        handler=logging.FileHandler(dir_codes + 'log/' + prefix + '-' + now.strftime("%Y-%m-%d") +'.log')
        handler.setLevel(logging.DEBUG)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)
    return logger

def logger_init(): # This will lead to duplicated logging output
    # logger = logging.getLogger(__name__) # this is used in modules 
    logger = logging.getLogger() # use this (root) in the main executable file
    logger.setLevel(logging.DEBUG)

    # create a file handler
    now = datetime.datetime.now()
    if not os.path.isdir(dir_codes + 'log/'):
        os.makedir(dir_codes + 'log/')
    handler = logging.FileHandler(dir_codes + r'opti_script.log')
    handler.setLevel(logging.DEBUG)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    return logger



import sys
# For a better solution for print, see https://stackoverflow.com/questions/4230855/why-am-i-getting-ioerror-9-bad-file-descriptor-error-while-making-print-st/4230866
# decorater used to block function printing to the console
def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__

    return func_wrapper
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__



import math
def to_precision(x,p=4):
    """ http://randlet.com/blog/python-significant-figures-format/
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)



import numpy as np
def singleSidedDFT(signal, samp_freq):
    NFFT = len(signal)
    dft_complex = np.fft.fft(signal,NFFT) # y is a COMPLEX defined in numpy
    dft_single_sided = [2 * abs(dft_bin) / NFFT for dft_bin in dft_complex][0:int(NFFT/2)+1] # /NFFT for spectrum aplitude consistent with actual signal. 2* for single-sided. abs for amplitude of complem number.
    dft_single_sided[0] *= 0.5 # DC bin in single sided spectrem does not need to be times 2
    return np.array(dft_single_sided)
    dft_freq = 0.5*samp_freq*np.linspace(0,1,NFFT/2+1) # unit is Hz # # f = np.fft.fftfreq(NFFT, Ts) # for double-sided

def basefreqDFT(signal, samp_freq, ax_time_domain=None, ax_freq_domain=None, base_freq=1):
    NFFT = len(signal)

    dft_complex = np.fft.fft(signal,NFFT) # y is a COMPLEX defined in numpy
    dft_double_sided = [abs(dft_bin) / NFFT for dft_bin in dft_complex]
    dft_single_sided = [2 * abs(dft_bin) / NFFT for dft_bin in dft_complex][0:int(NFFT/2)+1] # /NFFT for spectrum aplitude consistent with actual signal. 2* for single-sided. abs for amplitude of complem number.
    dft_single_sided[0] *= 0.5 # DC bin in single sided spectrem does not need to be times 2

    if base_freq==None:
        dft_freq = 0.5*samp_freq*np.linspace(0,1,NFFT/2+1) # unit is Hz # # f = np.fft.fftfreq(NFFT, Ts) # for double-sided
    else: # 0.5*samp_freq is the Nyquist frequency
        dft_freq = 0.5*samp_freq/base_freq*np.linspace(0,1,NFFT/2+1) # unit is per base_freq 

    if ax_time_domain != None:
        Ts = 1.0/samp_freq
        t = [el*Ts for el in range(0,NFFT)]
        ax_time_domain.plot(t, signal, '*', alpha=0.4)
        ax_time_domain.set_xlabel('time [s]')
        ax_time_domain.set_ylabel('B [T]')
    if ax_freq_domain != None:
        ax_freq_domain.plot(dft_freq, dft_single_sided, '*',alpha=0.4)
        ax_freq_domain.set_xlabel('per base frequency')
        ax_time_domain.set_ylabel('B [T]')

class Pyrhonen_design(object):
    def __init__(self, im, original_bounds=None):
        ''' Determine original_bounds for these parameters:
            stator_tooth_width_b_ds              = design_parameters[0]*1e-3 # m                       # stator tooth width [mm]
            air_gap_length_delta                 = design_parameters[1]*1e-3 # m                       # air gap length [mm]
            b1                                   = design_parameters[2]*1e-3 # m                       # rotor slot opening [mm]
            rotor_tooth_width_b_dr               = design_parameters[3]*1e-3 # m                       # rotor tooth width [mm]
            self.Length_HeadNeckRotorSlot        = design_parameters[4]      # mm       # rotor tooth head & neck length [mm]
            self.Angle_StatorSlotOpen            = design_parameters[5]      # mm       # stator slot opening [deg]
            self.Width_StatorTeethHeadThickness  = design_parameters[6]      # mm       # stator tooth head length [mm]
        '''
        # rotor_slot_radius = (2*pi*(Radius_OuterRotor - Length_HeadNeckRotorSlot)*1e-3 - rotor_tooth_width_b_dr*Qr) / (2*Qr+2*pi)
        # => rotor_tooth_width_b_dr = ( 2*pi*(Radius_OuterRotor - Length_HeadNeckRotorSlot)*1e-3  - rotor_slot_radius * (2*Qr+2*pi) ) / Qr
        # Verified in Tran2TSS_PS_Opti.xlsx.
            # ( 2*pi*(im.Radius_OuterRotor - im.Length_HeadNeckRotorSlot)  - im.Radius_of_RotorSlot * (2*Qr+2*pi) ) / Qr
            # = ( 2*PI()*(G4 - I4) - J4 * (2*C4+2*PI()) ) / C4
        from math import pi
        Qr = im.Qr


        # unit: mm and deg
        self.stator_tooth_width_b_ds        = im.Width_StatorTeethBody
        self.air_gap_length_delta           = im.Length_AirGap
        self.b1                             = im.Width_RotorSlotOpen
        self.rotor_tooth_width_b_dr         = ( 2*pi*(im.Radius_OuterRotor - im.Length_HeadNeckRotorSlot)  - im.Radius_of_RotorSlot * (2*Qr+2*pi) ) / Qr
        self.Length_HeadNeckRotorSlot       = im.Length_HeadNeckRotorSlot
        self.Angle_StatorSlotOpen           = im.Angle_StatorSlotOpen
        self.Width_StatorTeethHeadThickness = im.Width_StatorTeethHeadThickness

        self.design_parameters_denorm = [   self.stator_tooth_width_b_ds,
                                            self.air_gap_length_delta,
                                            self.b1,
                                            self.rotor_tooth_width_b_dr,
                                            self.Length_HeadNeckRotorSlot,
                                            self.Angle_StatorSlotOpen,
                                            self.Width_StatorTeethHeadThickness]

        if original_bounds is None:
            self.design_parameters_denorm
        else:
            self.show_norm(original_bounds, self.design_parameters_denorm)


    def show_denorm(self, original_bounds, design_parameters_norm):
        import numpy as np
        pop = design_parameters_norm
        min_b, max_b = np.asarray(original_bounds).T 
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        print '[De-normalized]:',
        print pop_denorm.tolist()
        
    def show_norm(self, original_bounds, design_parameters_denorm):
        import numpy as np
        min_b, max_b = np.asarray(original_bounds).T 
        diff = np.fabs(min_b - max_b)
        self.design_parameters_norm = (design_parameters_denorm - min_b)/diff #= pop
        # print type(self.design_parameters_norm)
        print '[Normalized]:',
        print self.design_parameters_norm.tolist()

        # pop = design_parameters_norm
        # min_b, max_b = np.asarray(original_bounds).T 
        # diff = np.fabs(min_b - max_b)
        # pop_denorm = min_b + pop * diff
        # print '[De-normalized:]---------------------------Are these two the same?'
        # print pop_denorm.tolist()
        # print design_parameters_denorm

def add_Pyrhonen_design_to_first_generation(sw, de_config_dict, logger):
    initial_design = Pyrhonen_design(sw.im, de_config_dict['original_bounds'])
    # print 'SWAP!'
    # print initial_design.design_parameters_norm.tolist()
    # print '\nInitial Population:'
    # for index in range(len(sw.init_pop)):
    #     # print sw.init_pop[index].tolist()
    #     print index,
    #     initial_design.show_denorm(de_config_dict['original_bounds'], sw.init_pop[index])
    # print sw.init_pop[0].tolist()
    sw.init_pop_denorm[0] = initial_design.design_parameters_denorm
    # print sw.init_pop[0].tolist()
    with open(sw.get_gen_file(0), 'w') as f:
        f.write('\n'.join(','.join('%.16f'%(x) for x in y) for y in sw.init_pop_denorm)) # convert 2d array to string
    logger.info('Initial design from Pyrhonen09 is added to the first generation of pop (i.e., gen#0000ind#0000).')



from smtplib import SMTP
def send_notification(text='Hello'):
    subject = 'AUTO BLIM OPTIMIZATION NOTIFICATION'
    content = 'Subject: %s\n\n%s' % (subject, text)
    mail = SMTP('smtp.gmail.com', 587)
    mail.ehlo()
    mail.starttls()
    with open('temp.txt', 'r') as f:
        pswd = f.read()
    mail.login('horyontour@gmail.com', pswd)
    mail.sendmail('horyontour@gmail.com', 'jiahao.chen@wisc.edu', content) 
    mail.close()
    print "Notificaiont sent."

# @staticmethod
def add_plots(axeses, dm, title=None, label=None, zorder=None, time_list=None, sfv=None, torque=None, range_ss=None, alpha=0.7):

    info = '%s' % (title)
    torque_average = sum(torque[-range_ss:])/len(torque[-range_ss:])
    info += '\nAverage Torque: %g Nm' % (torque_average)
    # torque error = torque - avg. torque
    torque_error = np.array(torque) - torque_average
    ss_max_torque_error = max(torque_error[-range_ss:]), min(torque_error[-range_ss:])
    # we use peak value to compute error rather than use peak-to-peak value
    normalized_torque_ripple   = 0.5*(ss_max_torque_error[0] - ss_max_torque_error[1]) / torque_average
    info += '\nNormalized Torque Ripple: %g %%' % (normalized_torque_ripple*100)

    info += '\nAverage Force Mag: %g N'% (sfv.ss_avg_force_magnitude)
    # we use peak value to compute error rather than use peak-to-peak value
    normalized_force_error_magnitude = 0.5*(sfv.ss_max_force_err_abs[0]-sfv.ss_max_force_err_abs[1])/sfv.ss_avg_force_magnitude
    info += '\nNormalized Force Error Mag: %g%%, (+)%g%% (-)%g%%' % (normalized_force_error_magnitude*100,
                                                                  sfv.ss_max_force_err_abs[0]/sfv.ss_avg_force_magnitude*100,
                                                                  sfv.ss_max_force_err_abs[1]/sfv.ss_avg_force_magnitude*100)
    # we use peak value to compute error rather than use peak-to-peak value
    force_error_angle= 0.5*(sfv.ss_max_force_err_ang[0]-sfv.ss_max_force_err_ang[1])
    info += '\nMaximum Force Error Angle: %g [deg], (+)%g deg (-)%g deg' % (force_error_angle,
                                                                 sfv.ss_max_force_err_ang[0],
                                                                 sfv.ss_max_force_err_ang[1])
    info += '\nExtra Info:'
    info += '\n\tAverage Force Vecotr: (%g, %g) N' % (sfv.ss_avg_force_vector[0], sfv.ss_avg_force_vector[1])
    info += '\n\tTorque Ripple (Peak-to-Peak) %g Nm'% ( max(torque[-range_ss:]) - min(torque[-range_ss:]))
    info += '\n\tForce Mag Ripple (Peak-to-Peak) %g N'% (sfv.ss_max_force_err_abs[0] - sfv.ss_max_force_err_abs[1])

    # plot for torque and force
    ax = axeses[0][0]; ax.plot(time_list, torque, alpha=alpha, label=label, zorder=zorder)
    ax = axeses[0][1]; ax.plot(time_list, sfv.force_abs, alpha=alpha, label=label, zorder=zorder)
    ax = axeses[1][0]; ax.plot(time_list, 100*sfv.force_err_abs/sfv.ss_avg_force_magnitude, label=label, alpha=alpha, zorder=zorder)
    ax = axeses[1][1]; ax.plot(time_list, np.arctan2(sfv.force_y, sfv.force_x)/np.pi*180. - sfv.ss_avg_force_angle, label=label, alpha=alpha, zorder=zorder)

    # plot for visialization of power factor 
    # dm.get_voltage_and_current(range_ss)
    # ax = axeses[2][0]; ax.plot(dm.mytime, dm.myvoltage, label=label, alpha=alpha, zorder=zorder)
    # ax = axeses[2][0]; ax.plot(dm.mytime, dm.mycurrent, label=label, alpha=alpha, zorder=zorder)

    return info, torque_average, normalized_torque_ripple, sfv.ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle

def build_str_results(axeses, im_variant, project_name, tran_study_name, dir_csv_output_folder, fea_config_dict, femm_solver):
    # originate from fobj

    dm = read_csv_results_4_general_purpose(tran_study_name, dir_csv_output_folder, fea_config_dict, femm_solver)
    basic_info, time_list, TorCon_list, ForConX_list, ForConY_list, ForConAbs_list = dm.unpack()
    sfv = suspension_force_vector(ForConX_list, ForConY_list, range_ss=fea_config_dict['number_of_steps_2ndTTS']) # samples in the tail that are in steady state
    str_results, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle = \
        add_plots( axeses, dm,
                      title=tran_study_name,
                      label='Transient FEA w/ 2 Time Step Sections',
                      zorder=8,
                      time_list=time_list,
                      sfv=sfv,
                      torque=TorCon_list,
                      range_ss=sfv.range_ss)
    str_results += '\n\tbasic info:' +   ''.join(  [str(el) for el in basic_info])

    if dm.jmag_loss_list is None:
        raise Exception('Loss data is not loaded?')
    else:
        str_results += '\n\tjmag loss info:'  + ', '.join(['%g'%(el) for el in dm.jmag_loss_list]) # dm.jmag_loss_list = [stator_copper_loss, rotor_copper_loss, stator_iron_loss, stator_eddycurrent_loss, stator_hysteresis_loss]

    if fea_config_dict['jmag_run_list'][0] == 0:
        str_results += '\n\tfemm loss info:'  + ', '.join(['%g'%(el) for el in dm.femm_loss_list])

    if fea_config_dict['delete_results_after_calculation'] == False:
        power_factor = dm.power_factor(fea_config_dict['number_of_steps_2ndTTS'], targetFreq=im_variant.DriveW_Freq)
        str_results += '\n\tPF: %g' % (power_factor)

    # compute the fitness 
    rotor_volume = im_variant.get_rotor_volume() 
    rotor_weight = im_variant.get_rotor_weights()
    shaft_power  = im_variant.Omega * torque_average # make sure update_mechanical_parameters is called so that Omega corresponds to slip_freq_breakdown_torque
    if dm.jmag_loss_list is None:
        copper_loss  = 0.0
        iron_loss    = 0.0
    else:
        if False:
            # by JMAG only
            copper_loss  = dm.jmag_loss_list[0] + dm.jmag_loss_list[1] 
            iron_loss    = dm.jmag_loss_list[2] 
        else:
            # by JMAG for iron loss and FEMM for copper loss
            if dm.femm_loss_list[0] is None: # this will happen for running release_design.py
                copper_loss  = dm.jmag_loss_list[0] + dm.jmag_loss_list[1] 
            else:
                copper_loss  = dm.femm_loss_list[0] + dm.femm_loss_list[1]
            iron_loss = dm.jmag_loss_list[2] 

        # some factor to account for rotor iron loss?
        # iron_loss *= 1

    # 这样计算效率，输出转矩大的，铁耗大一倍也没关系了，总之就是气隙变得最小。。。要不就不要优化气隙了。。。

    total_loss   = copper_loss + iron_loss
    efficiency   = shaft_power / (total_loss + shaft_power)  # 效率计算：机械功率/(损耗+机械功率)
    str_results  += '\n\teta: %g' % (efficiency)

    # for easy access to codes
    machine_results = [power_factor, efficiency, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle]
    machine_results.extend(dm.jmag_loss_list)
    if dm.femm_loss_list is not None:
        machine_results.extend(dm.femm_loss_list)
    else:
        machine_results.extend([0.0, 0.0])
    str_machine_results = ','.join('%g'%(el) for el in machine_results if el is not None) # note that femm_loss_list can be None called by release_design.py
    
    cost_function, list_cost = compute_list_cost(use_weights(fea_config_dict['use_weights']), rotor_volume, rotor_weight, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle, dm.jmag_loss_list, dm.femm_loss_list, power_factor, total_loss)

    str_results = '\n-------\n%s-%s\n%d,%d,%g\n%s\n%s\n%s\n' % (
                    project_name, im_variant.get_individual_name(), 
                    im_variant.number_current_generation, im_variant.individual_index, cost_function, 
                    str_machine_results,
                    ','.join(['%g'%(el) for el in list_cost]),
                    ','.join(['%g'%(el) for el in im_variant.design_parameters]) ) + str_results


    return str_results, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle, dm.jmag_loss_list, dm.femm_loss_list, power_factor, total_loss, cost_function

class suspension_force_vector(object):
    """docstring for suspension_force_vector"""
    def __init__(self, force_x, force_y, range_ss=None): # range_ss means range_steadystate
        super(suspension_force_vector, self).__init__()
        self.force_x = force_x
        self.force_y = force_y
        self.force_ang = np.arctan2(force_y, force_x) / np.pi * 180 # [deg]
        self.force_abs = np.sqrt(np.array(force_x)**2 + np.array(force_y)**2 )

        if range_ss == None:
            range_ss = len(force_x)
        self.range_ss = range_ss

        self.ss_avg_force_vector    = np.array([sum(force_x[-range_ss:]), sum(force_y[-range_ss:])]) / range_ss #len(force_x[-range_ss:])
        self.ss_avg_force_angle     = np.arctan2(self.ss_avg_force_vector[1], self.ss_avg_force_vector[0]) / np.pi * 180
        self.ss_avg_force_magnitude = np.sqrt(self.ss_avg_force_vector[0]**2 + self.ss_avg_force_vector[1]**2)

        self.force_err_ang = self.force_ang - self.ss_avg_force_angle
        self.force_err_abs = self.force_abs - self.ss_avg_force_magnitude

        self.ss_max_force_err_ang = max(self.force_err_ang[-range_ss:]), min(self.force_err_ang[-range_ss:])
        self.ss_max_force_err_abs = max(self.force_err_abs[-range_ss:]), min(self.force_err_abs[-range_ss:])

def pyplot_clear(axeses):
    # self.fig_main.clf()
    # axeses = self.axeses
    # for ax in [axeses[0][0],axeses[0][1],axeses[1][0],axeses[1][1],axeses[2][0],axeses[2][1]]:
    for ax in [axeses[0][0],axeses[0][1],axeses[1][0],axeses[1][1]]:
        ax.cla()
        ax.grid()
    ax = axeses[0][0]; ax.set_xlabel('(a)',fontsize=14.5); ax.set_ylabel('Torque [Nm]',fontsize=14.5)
    ax = axeses[0][1]; ax.set_xlabel('(b)',fontsize=14.5); ax.set_ylabel('Force Amplitude [N]',fontsize=14.5)
    ax = axeses[1][0]; ax.set_xlabel('Time [s]\n(c)',fontsize=14.5); ax.set_ylabel('Norm. Force Error Mag. [%]',fontsize=14.5)
    ax = axeses[1][1]; ax.set_xlabel('Time [s]\n(d)',fontsize=14.5); ax.set_ylabel('Force Error Angle [deg]',fontsize=14.5)



# IEMDC 2019
def read_csv_results_4_comparison__transient(study_name, path_prefix):
    print 'look into:', path_prefix

    # Torque
    basic_info = []
    time_list = []
    TorCon_list = []
    with open(path_prefix + study_name + '_torque.csv', 'r') as f:
        count = 0
        for row in csv_row_reader(f):
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
        count = 0
        for row in csv_row_reader(f):
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
        count = 0
        for row in csv_row_reader(f):
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

def read_csv_results_4_comparison_eddycurrent(study_name, path_prefix):
    # path_prefix = self.dir_csv_output_folder
    # Torque
    TorCon_list = []
    with open(path_prefix + study_name + '_torque.csv', 'r') as f:
        count = 0
        for row in csv_row_reader(f):
            count +=1
            if count<=5:
                continue
            else:
                TorCon_list.append(float(row[1]))

    # Force
    ForConX_list = []
    ForConY_list = []
    with open(path_prefix + study_name + '_force.csv', 'r') as f:
        count = 0
        for row in csv_row_reader(f):
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

def collect_jmag_Tran2TSSProlong_results(im_variant, path_prefix, fea_config_dict, axeses, femm_solver_data=None):

    data_results = []

    ################################################################
    # TranRef
    ################################################################   
    study_name = im_variant.get_individual_name() + 'Tran2TSS'
    dm = read_csv_results_4_comparison__transient(study_name, path_prefix)
    basic_info, time_list, TorCon_list, ForConX_list, ForConY_list, ForConAbs_list = dm.unpack()
    # t,T,Fx,Fy,Fabs = time_list, TorCon_list, ForConX_list, ForConY_list, ForConAbs_list
    end_time = time_list[-1]

    sfv = suspension_force_vector(ForConX_list, ForConY_list, range_ss=1*fea_config_dict['number_of_steps_2ndTTS']) # samples in the tail that are in steady state
    str_results, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle = \
        add_plots( axeses, dm,
                      title='TranRef',#tran_study_name,
                      label='Transient FEA w/ 2 Time Step Sections',
                      zorder=8,
                      time_list=time_list,
                      sfv=sfv,
                      torque=TorCon_list,
                      range_ss=sfv.range_ss)
    str_results += '\n\tbasic info:' +   ''.join(  [str(el) for el in basic_info])
    # print str_results
    data_results.extend([torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle])

    ################################################################
    # Tran2TSS
    ################################################################
    time_list, TorCon_list, ForConX_list, ForConY_list, ForConAbs_list = time_list[:49], TorCon_list[:49], ForConX_list[:49], ForConY_list[:49], ForConAbs_list[:49]

    sfv = suspension_force_vector(ForConX_list, ForConY_list, range_ss=fea_config_dict['number_of_steps_2ndTTS']) # samples in the tail that are in steady state
    str_results, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle = \
        add_plots( axeses, dm,
                      title='Tran2TSS',#tran_study_name,
                      label='Transient FEA w/ 2 Time Step Sections',
                      zorder=8,
                      time_list=time_list,
                      sfv=sfv,
                      torque=TorCon_list,
                      range_ss=sfv.range_ss)
    str_results += '\n\tbasic info:' +   ''.join(  [str(el) for el in basic_info])
    # print str_results
    data_results.extend([torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle])


    ################################################################
    # EddyCurrent-Rotate
    ################################################################
    study_name = im_variant.get_individual_name() + 'Freq-FFVRC'
    dm = read_csv_results_4_comparison_eddycurrent(study_name, path_prefix)
    _, _, TorCon_list, ForConX_list, ForConY_list, ForConAbs_list = dm.unpack()

    rotor_position_in_deg = 360./im_variant.Qr / len(TorCon_list) * np.arange(0, len(TorCon_list))
    # print rotor_position_in_deg
    time_list = rotor_position_in_deg/180.*math.pi / im_variant.Omega
    number_of_repeat = int(end_time / time_list[-1])

    # 延拓
    ec_torque        = number_of_repeat*TorCon_list
    time_one_step = time_list[1]
    time_list     = [i*time_one_step for i in range(len(ec_torque))]
    ec_force_abs     = number_of_repeat*ForConAbs_list.tolist()
    ec_force_x       = number_of_repeat*ForConX_list
    ec_force_y       = number_of_repeat*ForConY_list

    sfv = suspension_force_vector(ec_force_x, ec_force_y, range_ss=len(rotor_position_in_deg))
    str_results, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle = \
        add_plots( axeses, dm,
              title='Eddy Current Rotate',
              label='Eddy Current FEA', #'EddyCurFEAwiRR',
              zorder=2,
              time_list=time_list,
              sfv=sfv,
              torque=ec_torque,
              range_ss=sfv.range_ss) # samples in the tail that are in steady state
    # print str_results
    data_results.extend([torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle])


    ################################################################
    # FEMM
    ################################################################
    if femm_solver_data is not None:
        study_name = 'FEMM'
        rotor_position_in_deg = femm_solver_data[0]*0.1 
        time_list = rotor_position_in_deg/180.*math.pi / im_variant.Omega
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
        str_results, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle = \
            add_plots( axeses, dm,
                  title=study_name,
                  label='Static FEA', #'StaticFEAwiRR',
                  zorder=3,
                  time_list=time_list,
                  sfv=sfv,
                  torque=femm_torque,
                  range_ss=sfv.range_ss,
                  alpha=0.5) 
        # print str_results
        data_results.extend([torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle])


    # # for easy access to codes
    # machine_results = [torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle]
    # str_machine_results = ','.join('%g'%(el) for el in machine_results if el is not None) # note that femm_loss_list can be None called by release_design.py
    # return str_results, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle

    return data_results


class data_manager(object):

    def __init__(self):
        self.basic_info = []
        self.time_list = []
        self.TorCon_list = []
        self.ForConX_list = []
        self.ForConY_list = []
        self.ForConAbs_list = []

        self.jmag_loss_list = None
        self.femm_loss_list = None

    def unpack(self):
        return self.basic_info, self.time_list, self.TorCon_list, self.ForConX_list, self.ForConY_list, self.ForConAbs_list

    def terminal_voltage(self, which='4W'): # 2A 2B 2C 4A 4B 4C
        return self.Current_dict['Terminal%s [Case 1]'%(which)]
        # 端点电压是相电压吗？应该是，我们在中性点设置了地电位

    def circuit_current(self, which='4W'): # 2A 2B 2C 4A 4B 4C
        return self.Current_dict['CircuitCoil%s'%(which)]

    def get_voltage_and_current(self, number_of_steps_2ndTTS):

        # 4C <- the C-phase of the 4 pole winding
        mytime  = self.Current_dict['Time(s)'][-number_of_steps_2ndTTS:]
        voltage =      self.terminal_voltage()[-number_of_steps_2ndTTS:]
        current =       self.circuit_current()[-number_of_steps_2ndTTS:]

        # if len(mytime) > len(voltage):
        #     mytime = mytime[:len(voltage)]

        # print len(mytime), len(voltage), number_of_steps_2ndTTS
        # print len(mytime), len(voltage)
        # print len(mytime), len(voltage)

        # for access to plot
        self.myvoltage = voltage
        self.mycurrent = current
        self.mytime    = mytime

    def power_factor(self, number_of_steps_2ndTTS, targetFreq=1e3, numPeriodicalExtension=1000):
        # number_of_steps_2ndTTS: steps corresponding to half the period 

        # for key, val in self.Current_dict.iteritems():
        #     if 'Terminal' in key:
        #         print key, val
        # quit()

        self.get_voltage_and_current(number_of_steps_2ndTTS)
        mytime  = self.mytime
        voltage = self.myvoltage
        current = self.mycurrent

        # from pylab import *
        # print len(mytime), len(voltage), len(current)
        # figure()
        # plot(mytime, voltage)
        # plot(mytime, current)
        # show()
        power_factor = compute_power_factor_from_half_period(voltage, current, mytime, targetFreq=targetFreq, numPeriodicalExtension=numPeriodicalExtension)
        return power_factor

from VanGogh import csv_row_reader
def read_csv_results_4_general_purpose(study_name, path_prefix, fea_config_dict, femm_solver):
    # Read TranFEAwi2TSS results
    
    # logging.getLogger(__name__).debug('Look into: ' + path_prefix)

    # Torque
    basic_info = []
    time_list = []
    TorCon_list = []
    with open(path_prefix + study_name + '_torque.csv', 'r') as f:
        count = 0
        for row in csv_row_reader(f):
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
        count = 0
        for row in csv_row_reader(f):
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
        count = 0
        for row in csv_row_reader(f):
            count +=1
            if count<=8:
                if 'Time' in row[0]: # Time(s)
                    for key in row:
                        key_list.append(key)
                        Current_dict[key] = []
                else:
                    continue
            else:
                for ind, val in enumerate(row):
                    Current_dict[key_list[ind]].append(float(val))

    # Terminal Voltage 
    new_key_list = []
    if fea_config_dict['delete_results_after_calculation'] == False:
        # file name is by individual_name like ID32-2-4_EXPORT_CIRCUIT_VOLTAGE.csv rather than ID32-2-4Tran2TSS_circuit_current.csv
        fname = path_prefix + study_name[:-8] + "_EXPORT_CIRCUIT_VOLTAGE.csv"
        # print 'Terminal Voltage - look into:', fname
        with open(fname, 'r') as f:
            count = 0
            for row in csv_row_reader(f):
                count +=1
                if count==1: # Time | Terminal1 | Terminal2 | ... | Termial6
                    if 'Time' in row[0]: # Time, s
                        for key in row:
                            new_key_list.append(key) # Yes, you have to use a new key list, because the ind below bgeins at 0.
                            Current_dict[key] = []
                    else:
                        raise Exception('Problem with csv file for terminal voltage.')
                else:
                    for ind, val in enumerate(row):
                        Current_dict[new_key_list[ind]].append(float(val))
    key_list += new_key_list

    # Loss
    # Iron Loss
    with open(path_prefix + study_name + '_iron_loss_loss.csv', 'r') as f:
        count = 0
        for row in csv_row_reader(f):
            count +=1
            if count>8:
                stator_iron_loss = float(row[3]) # Stator Core
                break
    with open(path_prefix + study_name + '_joule_loss_loss.csv', 'r') as f:
        count = 0
        for row in csv_row_reader(f):
            count +=1
            if count>8:
                stator_eddycurrent_loss = float(row[3]) # Stator Core
                break
    with open(path_prefix + study_name + '_hysteresis_loss_loss.csv', 'r') as f:
        count = 0
        for row in csv_row_reader(f):
            count +=1
            if count>8:
                stator_hysteresis_loss = float(row[3]) # Stator Core
                break
    # Copper Loss
    rotor_copper_loss_list = []
    with open(path_prefix + study_name + '_joule_loss.csv', 'r') as f:
        count = 0
        for row in csv_row_reader(f):
            count +=1
            if count>8:
                if count==9:
                    stator_copper_loss = float(row[8]) # Coil # it is the same over time, this value does not account for end coil

                rotor_copper_loss_list.append(float(row[7])) # Cage
    
    # use the last 1/4 period data to compute average copper loss of Tran2TSS rather than use that of Freq study
    effective_part = rotor_copper_loss_list[:int(0.5*fea_config_dict['number_of_steps_2ndTTS'])] # number_of_steps_2ndTTS = steps for half peirod
    rotor_copper_loss = sum(effective_part) / len(effective_part)

    if fea_config_dict['jmag_run_list'][0] == 0:
        blockPrint()
        try:
            # convert rotor current results (complex number) into its amplitude
            femm_solver.list_rotor_current_amp = [abs(el) for el in femm_solver.vals_results_rotor_current] # el is complex number
            # settings not necessarily be consistent with Pyrhonen09's design: , STATOR_SLOT_FILL_FACTOR=0.5, ROTOR_SLOT_FILL_FACTOR=1., TEMPERATURE_OF_COIL=75
            s, r = femm_solver.get_copper_loss(femm_solver.stator_slot_area, femm_solver.rotor_slot_area)
        except Exception as e:
            raise e
        enablePrint()
    else:
        s, r = None, None

    dm = data_manager()
    dm.basic_info     = basic_info
    dm.time_list      = time_list
    dm.TorCon_list    = TorCon_list
    dm.ForConX_list   = ForConX_list
    dm.ForConY_list   = ForConY_list
    dm.ForConAbs_list = ForConAbs_list
    dm.Current_dict   = Current_dict
    dm.key_list       = key_list
    dm.jmag_loss_list = [stator_copper_loss, rotor_copper_loss, stator_iron_loss, stator_eddycurrent_loss, stator_hysteresis_loss]
    dm.femm_loss_list = [s, r]
    return dm

def check_csv_results_4_general_purpose(study_name, path_prefix, returnBoolean=False):
    # Check frequency analysis results

    print 'Check:', path_prefix + study_name + '_torque.csv'

    if not os.path.exists(path_prefix + study_name + '_torque.csv'):
        if returnBoolean == False:
            return None
        else:
            return False
    else:
        if returnBoolean == True:
            return True

    try:
        # check csv results 
        l_slip_freq = []
        l_TorCon    = []
        l_ForCon_X  = []
        l_ForCon_Y  = []

        # fitness_in_physics_data = []
        with open(path_prefix + study_name + '_torque.csv', 'r') as f: 
            for ind, row in enumerate(csv_row_reader(f)):
                if ind >= 5:
                    try:
                        float(row[0])
                    except:
                        continue
                    l_slip_freq.append(float(row[0]))
                    l_TorCon.append(float(row[1]))

        with open(path_prefix + study_name + '_force.csv', 'r') as f: 
            for ind, row in enumerate(csv_row_reader(f)):
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


# https://dsp.stackexchange.com/questions/11513/estimate-frequency-and-peak-value-of-a-signals-fundamental
#define N_SAMPLE ((long int)(1.0/(0.1*TS))) // Resolution 0.1 Hz = 1 / (N_SAMPLE * TS)
class Goertzel_Data_Struct(object):
    """docstring for Goertzel_Data_Struct"""
    def __init__(self, id=None, ):
        self.id = id    
        self.bool_initialized = False
        self.sine = None
        self.cosine = None
        self.coeff = None
        self.scalingFactor = None
        self.q = None
        self.q2 = None
        self.count = None
        self.k = None #// k is the normalized target frequency 
        self.real = None
        self.imag = None
        self.accumSquaredData = None
        self.ampl = None
        self.phase = None


    # /************************************************
    #  * Real time implementation to avoid the array of input double *data[]
    #  * with Goertzel Struct to store the variables and the output values
    #  *************************************************/
    def goertzel_realtime(gs, targetFreq, numSamples, samplingRate, data):
        # gs is equivalent to self
        try:
            len(data)
        except:
            pass
        else:
            raise Exception('This is for real time implementation of Goertzel, hence data must be a scalar rather than array-like object.')

        if gs.bool_initialized == False:
            gs.bool_initialized = True

            gs.count = 0
            gs.k = (0.5 + ((numSamples * targetFreq) / samplingRate))
            omega = (2.0 * math.pi * gs.k) / numSamples
            gs.sine = sin(omega)
            gs.cosine = cos(omega)
            gs.coeff = 2.0 * gs.cosine
            gs.q1=0
            gs.q2=0
            gs.scalingFactor = 0.5 * numSamples
            gs.accumSquaredData = 0.0

        q0 = gs.coeff * gs.q1 - gs.q2 + data
        gs.q2 = gs.q1
        gs.q1 = q0 # // q1 is the newest output vk[N], while q2 is the last output vk[N-1].

        gs.accumSquaredData += data*data

        gs.count += 1
        if gs.count>=numSamples:
            # // calculate the real and imaginary results with scaling appropriately
            gs.real = (gs.q1 * gs.cosine - gs.q2) / gs.scalingFactor #// inspired by the python script of sebpiq
            gs.imag = (gs.q1 * gs.sine) / gs.scalingFactor

            # // reset
            gs.bool_initialized = False
            return True
        else:
            return False

    def goertzel_offline(gs, targetFreq, samplingRate, data_list):
        # gs is equivalent to self

        numSamples = len(data_list)

        if gs.bool_initialized == False:
            gs.bool_initialized = True

            gs.count = 0
            gs.k = (0.5 + ((numSamples * targetFreq) / samplingRate))
            omega = (2.0 * math.pi * gs.k) / numSamples
            gs.sine = math.sin(omega)
            gs.cosine = math.cos(omega)
            gs.coeff = 2.0 * gs.cosine
            gs.q1=0
            gs.q2=0
            gs.scalingFactor = 0.5 * numSamples
            gs.accumSquaredData = 0.0

        for data in data_list:
            q0 = gs.coeff * gs.q1 - gs.q2 + data
            gs.q2 = gs.q1
            gs.q1 = q0 # // q1 is the newest output vk[N], while q2 is the last output vk[N-1].

            gs.accumSquaredData += data*data

            gs.count += 1
            if gs.count>=numSamples:

                # // calculate the real and imaginary results with scaling appropriately
                gs.real = (gs.q1 * gs.cosine - gs.q2) / gs.scalingFactor #// inspired by the python script of sebpiq
                gs.imag = (gs.q1 * gs.sine) / gs.scalingFactor

                # // reset
                gs.bool_initialized = False
                return True
        print data_list
        print gs.count
        print numSamples
        return None

def compute_power_factor_from_half_period(voltage, current, mytime, targetFreq=1e3, numPeriodicalExtension=1000): # 目标频率默认是1000Hz

    gs_u = Goertzel_Data_Struct("Goertzel Struct for Voltage\n")
    gs_i = Goertzel_Data_Struct("Goertzel Struct for Current\n")

    TS = mytime[-1] - mytime[-2]

    if type(voltage)!=type([]):
        voltage = voltage.tolist() + (-voltage).tolist()
        current = current.tolist() + (-current).tolist()
    else:
        voltage = voltage + [-el for el in voltage]
        current = current + [-el for el in current]

    voltage *= numPeriodicalExtension
    current *= numPeriodicalExtension

    N_SAMPLE = len(voltage)
    gs_u.goertzel_offline(targetFreq, 1./TS, voltage)
    gs_i.goertzel_offline(targetFreq, 1./TS, current)

    gs_u.ampl = math.sqrt(gs_u.real*gs_u.real + gs_u.imag*gs_u.imag) 
    gs_u.phase = math.atan2(gs_u.imag, gs_u.real)

    gs_i.ampl = math.sqrt(gs_i.real*gs_i.real + gs_i.imag*gs_i.imag) 
    gs_i.phase = math.atan2(gs_i.imag, gs_i.real)

    phase_difference_in_deg = ((gs_i.phase-gs_u.phase)/math.pi*180)
    power_factor = math.cos(gs_i.phase-gs_u.phase)
    return power_factor
    




def max_indices_2(arr, k):
    '''
    Returns the indices of the k first largest elements of arr
    (in descending order in values)
    '''
    assert k <= arr.size, 'k should be smaller or equal to the array size'
    arr_ = arr.astype(float)  # make a copy of arr
    max_idxs = []
    for _ in range(k):
        max_element = np.max(arr_)
        if np.isinf(max_element):
            break
        else:
            idx = np.where(arr_ == max_element)
        max_idxs.append(idx)
        arr_[idx] = -np.inf
    return max_idxs

def min_indices(arr, k):
    if type(arr) == type([]):
        arr = np.array(arr)
    indices = np.argsort(arr)[:k]
    items   = arr[indices] # arr and indices must be np.array
    return indices, items

def max_indices(arr, k):
    if type(arr) == type([]):
        arr = np.array(arr)
    arr_copy = arr[::]
    indices = np.argsort(-arr)[:k]
    items   = arr_copy[indices]
    return indices, items


class SwarmDataAnalyzer(object):
    """docstring for SwarmDataAnalyzer"""
    def __init__(self, dir_run=None, run_integer=None, bool_sensitivity_analysis=True):
        if run_integer is not None:
            dir_run = r'D:\OneDrive - UW-Madison\c\pop\run#%d/'%(run_integer)

        with open(dir_run+'swarm_data.txt', 'r') as f:
            self.buf = f.readlines()[1:]

        self.number_of_designs = len(self.buf) / 21 # 此21是指每个个体的结果占21行，非彼20+1哦。

        if bool_sensitivity_analysis:            
            if self.number_of_designs == 21*7:
                print 'These are legacy results without the initial design within the swarm_data.txt. '

            elif self.number_of_designs == 21*7 + 1:
                print 'Initial design is among the pop and used as reference design.'
                self.reference_design = self.buf[:21]
                self.buf = self.buf[21:]
            else:
                raise Exception('Please remove duplicate results in swarm_data.txt.')

        # now we have the actual pop size
        self.buf_length = len(self.buf)
        self.number_of_designs = len(self.buf) / 21

        msg = 'self.buf_length %% 21 = %d, / 21 = %d' % (self.buf_length % 21, self.number_of_designs)
        logger = logging.getLogger(__name__)
        logger.debug(msg)

    def design_display_generator(self):
        for i in range(self.number_of_designs):
            yield ''.join(self.buf[i*21:(1+i)*21])

    def design_parameters_generator(self):
        for i in range(self.number_of_designs):
            yield [float(el) for el in self.buf[i*21:(1+i)*21][5].split(',')]

    def list_generations(self):

        the_dict = {}
        for i in range(self.number_of_designs):
            the_row = [float(el) for el in self.buf[i*21:(1+i)*21][2].split(',')]        
            the_dict[int(the_row[0])] = (the_row[1], the_row[2])
            # the_row[0] # generation
            # the_row[1] # individual
            # the_row[2] # cost_function
        return the_dict #每代只留下最后一个个体的结果，因为字典不能重复key

    def list_cost_function(self):
        l = []
        for i in range(self.number_of_designs):
            l.append( float( self.buf[i*21:(1+i)*21][2].split(',')[2] ) )
        return l

    def find_individual(self, generation_index, individual_index):
        # for i in range(self.number_of_designs):
        for i in range(self.number_of_designs)[::-1]:
            # print i
            the_row = [int(el) for el in self.buf[i*21:(1+i)*21][2].split(',')[:2]]
            if the_row[0] == generation_index:
                if the_row[1] == individual_index:
                    # found it, return the design parameters and cost_function
                    return [float(el) for el in self.buf[i*21:(1+i)*21][5].split(',')], float(self.buf[i*21:(1+i)*21][2].split(',')[2])
        return None, None

    def get_best_generation(self, popsize=50, generator=None, returnMore=False):

        if generator is None:
            generator = self.design_parameters_generator()

        cost = self.list_cost_function()
        indices, items = min_indices(cost, popsize)
        print indices
        print items

        # for ind, el in enumerate(cost):
        #     print ind, el,
        # quit()

        # this is in wrong order as inidices, so use dict instead
        # gen_best = [design for index, design in enumerate(generator) if index in indices]
        gen_best = {}
        for index, design in enumerate(generator):
            if index in indices:
                gen_best[index] = design
        gen_best = [gen_best[index] for index in indices] # now it is in order

        if returnMore == False:
            return gen_best
        else:
            return gen_best, indices, items

    def get_list_objective_function(self):
        for i in range(self.number_of_designs):
            individual = self.buf[i*21:(1+i)*21]
            yield     [float(el) for el in individual[3].split(',')] #\
                    # + [float(el) for el in individual[17][16:].split(',')] \
                    # + [float(el) for el in individual[18][16:].split(',')]

    def get_certain_objective_function(self, which):
        for i in range(self.number_of_designs):
            individual = self.buf[i*21:(1+i)*21]
            try:
                yield [float(el) for el in individual[3].split(',')][which]
            except Exception as e:
                print [float(el) for el in individual[3].split(',')], which, i, individual
                raise e


def autolabel(ax, rects, xpos='center', bias=0.0):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height+bias,
                '%.2f'%(height), ha=ha[xpos], va='bottom', rotation=90)

def efficiency_at_50kW(total_loss):
    return 50e3 / (array(total_loss) + 50e3)  # 用50 kW去算才对，只是这样转子铜耗数值会偏大哦。 效率计算：机械功率/(损耗+机械功率)        

def use_weights(which='O1'):
    if which == 'O1':
        return [ 1, 0.1,   1, 0.1, 0.1,   0 ]
    if which == 'O2':
        return [ 1,1,1,1,1,  0 ]
    if which == 'O3':
        return [ 1,0,1,0,0,  1 ]
    return None

def compute_list_cost(weights, rotor_volume, rotor_weight, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle, jmag_loss_list, femm_loss_list, power_factor, total_loss):
    # O2
    # weights = [ 1, 1.0,   1, 1.0, 1.0,   0 ]
    # O1
    # weights = [ 1, 0.1,   1, 0.1, 0.1,   0 ]
    list_cost = [   30e3 / ( torque_average/rotor_volume ),
                    normalized_torque_ripple         *  20, 
                    1.0 / ( ss_avg_force_magnitude/rotor_weight ),
                    normalized_force_error_magnitude *  20, 
                    force_error_angle                * 0.2, # [deg] 
                    total_loss                       / 2500. ] 
    cost_function = np.dot(np.array(list_cost), np.array(weights))
    return cost_function, list_cost

    # The weight is [TpRV=30e3, FpRW=1, Trip=50%, FEmag=50%, FEang=50deg, eta=sqrt(10)=3.16]
    # which means the FEang must be up to 50deg so so be the same level as TpRV=30e3 or FpRW=1 or eta=316%
    # list_weighted_cost = [  30e3 / ( torque_average/rotor_volume ),
    #                         1.0 / ( ss_avg_force_magnitude/rotor_weight ),
    #                         normalized_torque_ripple         *   2, #       / 0.05 * 0.1
    #                         normalized_force_error_magnitude *   2, #       / 0.05 * 0.1
    #                         force_error_angle * 0.2          * 0.1, # [deg] /5 deg * 0.1 is reported to be the base line (Yegu Kang) # force_error_angle is not consistent with Yegu Kang 2018-060-case of TFE
    #                         2*total_loss/2500., #10 / efficiency**2,
    #                         im_variant.thermal_penalty ] # thermal penalty is evaluated when drawing the model according to the parameters' constraints (if the rotor current and rotor slot size requirement does not suffice)
    # cost_function = sum(list_weighted_cost)


def fobj_scalar(torque_average, ss_avg_force_magnitude, normalized_torque_ripple, normalized_force_error_magnitude, force_error_angle, total_loss, 
                weights=None, rotor_volume=None, rotor_weight=None):

    list_cost = [   30e3 / ( torque_average/rotor_volume ),
                    normalized_torque_ripple         *  20, #       / 0.05 * 0.1
                    1.0 / ( ss_avg_force_magnitude/rotor_weight ),
                    normalized_force_error_magnitude *  20, #       / 0.05 * 0.1
                    force_error_angle                * 0.2, # [deg] /5 deg * 0.1 is reported to be the base line (Yegu Kang) # force_error_angle is not consistent with Yegu Kang 2018-060-case of TFE
                    total_loss                       / 2500. ] #10 / efficiency**2,
    cost_function = np.dot(np.array(list_cost), np.array(weights))
    return cost_function

def fobj_list(l_torque_average, l_ss_avg_force_magnitude, l_normalized_torque_ripple, l_normalized_force_error_magnitude, l_force_error_angle, l_total_loss,
                weights=None, rotor_volume=None, rotor_weight=None):

    l_cost_function = []
    for torque_average, ss_avg_force_magnitude, normalized_torque_ripple, normalized_force_error_magnitude, force_error_angle, total_loss in zip(l_torque_average, l_ss_avg_force_magnitude, l_normalized_torque_ripple, l_normalized_force_error_magnitude, l_force_error_angle, l_total_loss):
        list_cost = [   30e3 / ( torque_average/rotor_volume ),
                        normalized_torque_ripple         *  20,
                        1.0 / ( ss_avg_force_magnitude/rotor_weight ),
                        normalized_force_error_magnitude *  20,
                        force_error_angle                * 0.2,
                        total_loss                       / 2500. ]
        cost_function = np.dot(np.array(list_cost), np.array(weights))
        l_cost_function.append(cost_function)
    return np.array(l_cost_function)

# Basic information
if __name__ == '__main__':
    
    if False: # 4 pole motor
        required_torque = 15.9154943092 #Nm
        Radius_OuterRotor = 47.092753
        stack_length = 93.200295
        Omega = 30000 / 60. * 2*np.pi
        Qr = 32 

    else: # 2 pole motor
        required_torque = 14.8544613552 #Nm
        Radius_OuterRotor = 28.8
        stack_length = 237.525815777
        Omega = 45000 / 60. * 2*np.pi
        Qr = 16

    rotor_volume = math.pi*(Radius_OuterRotor*1e-3)**2 * (stack_length*1e-3)
    rotor_weight = 9.8 * rotor_volume * 8050 # steel 8,050 kg/m3. Copper/Density 8.96 g/cm³. gravity: 9.8 N/kg

    print 'utility.py'
    print 'Qr=%d, rotor_volume='%(Qr), rotor_volume, 'm^3'
    print 'Qr=%d, rotor_weight='%(Qr), rotor_weight, 'N'

    if False: # 4 pole motor
        # run 115 and 116
        de_config_dict = {  'original_bounds':[ [ 4.9,   9],#--# stator_tooth_width_b_ds
                                                [ 0.8,   3],   # air_gap_length_delta
                                                [5e-1,   3],   # Width_RotorSlotOpen 
                                                [ 2.7,   5],#--# rotor_tooth_width_b_dr # 8 is too large, 6 is almost too large
                                                [5e-1,   3],   # Length_HeadNeckRotorSlot
                                                [   1,  10],   # Angle_StatorSlotOpen
                                                [5e-1,   3] ], # Width_StatorTeethHeadThickness
                            'mut':        0.8,
                            'crossp':     0.7,
                            'popsize':    42, # 50, # 100,
                            'iterations': 1 } # 148
        # run 114 and 200
        # de_config_dict = {  'original_bounds':     [ [   4, 7.2],#--# stator_tooth_width_b_ds
        #                                     [ 0.8,   4],   # air_gap_length_delta
        #                                     [5e-1,   3],   # Width_RotorSlotOpen 
        #                                     [ 2.5, 5.2],#--# rotor_tooth_width_b_dr # 8 is too large, 6 is almost too large
        #                                     [5e-1,   3],   # Length_HeadNeckRotorSlot
        #                                     [   1,  10],   # Angle_StatorSlotOpen
        #                                     [5e-1,   3] ], # Width_StatorTeethHeadThickness
        #                     'mut':        0.8,
        #                     'crossp':     0.7,
        #                     'popsize':    50, # 50, # 100,
        #                     'iterations': 2*48 } # 148
    else: # 2 pole motor denorm: [4.484966, 1.60056, 1.130973, 6.238951254340352, 1.0, 3.0, 1.0]
        # # run#190
        # de_config_dict = {  'original_bounds':[ [   3, 5.6],#--# stator_tooth_width_b_ds
        #                                         [ 0.8,   3],   # air_gap_length_delta
        #                                         [5e-1,   3],   # Width_RotorSlotOpen 
        #                                         [ 3.6,5.45],#--# rotor_tooth_width_b_dr 
        #                                         [5e-1,   3],   # Length_HeadNeckRotorSlot
        #                                         [   1,  10],   # Angle_StatorSlotOpen
        #                                         [5e-1,   3] ], # Width_StatorTeethHeadThickness
        #                     'mut':        0.8,
        #                     'crossp':     0.7,
        #                     'popsize':    30, # 21*7,  # 50, # 100,
        #                     'iterations': 100}

        # run#191 # initial design: 4.48497,1.60056,1.13097,6.23895,1,3,1
        de_config_dict = {  'original_bounds':[ [ 2.7, 5.8],#--# stator_tooth_width_b_ds
                                                [ 0.8,   3],   # air_gap_length_delta
                                                [5e-1,   3],   # Width_RotorSlotOpen 
                                                [ 3.5, 6.3],#--# rotor_tooth_width_b_dr # It allows for large rotor tooth because we raise Jr and recall this is for less slots---Qr=16.
                                                [5e-1,   3],   # Length_HeadNeckRotorSlot
                                                [   1,  10],   # Angle_StatorSlotOpen
                                                [5e-1,   3] ], # Width_StatorTeethHeadThickness
                            'mut':        0.8,
                            'crossp':     0.7,
                            'popsize':    30, # 21*7,  # 50, # 100,
                            'iterations': 100}

    
    O1_weights = use_weights(which='O1') # [ 1, 0.1,   1, 0.1, 0.1,   0 ]
    O2_weights = use_weights(which='O2') # [ 1, 1.0,   1, 1.0, 1.0,   0 ]
    # O1_weights = [ 1, 0.1,   1, 0.1, 0.1,   0.1 ]
    # O2_weights = [ 1, 1.0,   1, 1.0, 1.0,   1 ]
    # O2_weights = [ 0, 0,   0, 0, 0,   1.0 ]

    O2_weights = use_weights(which='O3') # run#194

    # In fact, you can run a bounds-check from the swarm_data.txt (whether the initial design falls within given bounds)
    # In fact, you can run a bounds-check from the swarm_data.txt
    # In fact, you can run a bounds-check from the swarm_data.txt

    original_bounds = de_config_dict['original_bounds']
    dimensions = len(original_bounds)
    min_b, max_b = np.asarray(original_bounds).T 
    diff = np.fabs(min_b - max_b)
    # pop_denorm = min_b + pop * diff
    # pop[j] = (pop_denorm[j] - min_b) / diff

def pareto_plot():
    pass


# plot sensitivity bar chart, Oj v.s. geometry parameters, pareto plot for ecce paper
import itertools
if __name__ == '__main__':
    from pylab import *
    plt.rcParams["font.family"] = "Times New Roman"
    # import seaborn as sns

    # style.use('seaborn-poster') #sets the size of the charts
    # style.use('grayscale')
    # ax = sns.barplot(y= "Deaths", x = "Causes", data = deaths_pd, palette=("Blues_d"))
    # sns.set_context("poster")


    # Pareto Plot or Correlation Plot
    if True:
        # swda = SwarmDataAnalyzer(run_integer=121, bool_sensitivity_analysis=False) # 4 pole Qr=32 motor for ecce19 digest
        # torque_average, ss_avg_force_magnitude, normalized_torque_ripple, normalized_force_error_magnitude, force_error_angle, total_loss = 19.1197, 96.9263, 0.0864712, 0.104915, 6.53137, (1817.22+216.216+224.706)
        # O2_ref = fobj_scalar(torque_average, ss_avg_force_magnitude, normalized_torque_ripple, normalized_force_error_magnitude, force_error_angle, total_loss, weights=O2_weights, rotor_volume=rotor_volume, rotor_weight=rotor_weight)

        # swda = SwarmDataAnalyzer(run_integer=193, bool_sensitivity_analysis=False) # 2 pole Qr=16 motor for NineSigma - O2
        swda = SwarmDataAnalyzer(run_integer=194, bool_sensitivity_analysis=False) # 2 pole Qr=16 motor for NineSigma - O3
        torque_average, ss_avg_force_magnitude, normalized_torque_ripple, normalized_force_error_magnitude, force_error_angle, total_loss = 13.8431,107.522,0.046109,0.035044,2.17509, (1388.12+433.332+251.127)
        O2_ref = fobj_scalar(torque_average, ss_avg_force_magnitude, normalized_torque_ripple, normalized_force_error_magnitude, force_error_angle, total_loss, weights=O2_weights, rotor_volume=rotor_volume, rotor_weight=rotor_weight) # reference design is the same from the sensitivty analysis

        print 'O2_ref=', O2_ref
        # print swda.list_cost_function()

        O2 = fobj_list( list(swda.get_certain_objective_function(2)), #torque_average, 
                        list(swda.get_certain_objective_function(4)), #ss_avg_force_magnitude, 
                        list(swda.get_certain_objective_function(3)), #normalized_torque_ripple, 
                        list(swda.get_certain_objective_function(5)), #normalized_force_error_magnitude, 
                        list(swda.get_certain_objective_function(6)), #force_error_angle, 
                        array(list(swda.get_certain_objective_function(9))) + array(list(swda.get_certain_objective_function(12))) + array(list(swda.get_certain_objective_function(13))), #total_loss, 
                        weights=O2_weights, rotor_volume=rotor_volume, rotor_weight=rotor_weight)
        # print O2
        # print array(swda.list_cost_function()) - array(O2) # they are the same
        O2 = O2.tolist()

        def my_scatter_plot(x,y,O,xy_ref,O_ref, fig=None, ax=None, s=15):
            # O is a copy of your list rather than array or the adress of the list
            # O is a copy of your list rather than array or the adress of the list
            # O is a copy of your list rather than array or the adress of the list
            x += [xy_ref[0]]
            y += [xy_ref[1]]
            O += [O_ref]
            if ax is None or fig is None:
                fig = figure()
                ax = fig.gca()
            # O2_mix = np.concatenate([[O_ref], O2], axis=0) # # https://stackoverflow.com/questions/46106912/one-colorbar-for-multiple-scatter-plots
            # min_, max_ = O2_mix.min(), O2_mix.max()
            ax.annotate('Initial design', xytext=(xy_ref[0]*0.95, xy_ref[1]*1.0), xy=xy_ref, xycoords='data', arrowprops=dict(arrowstyle="->"))
            # scatter(*xy_ref, marker='s', c=O_ref, s=20, alpha=0.75, cmap='viridis')
            # clim(min_, max_)
            scatter_handle = ax.scatter(x, y, c=O, s=s, alpha=0.5, cmap='viridis')
            # clim(min_, max_)
            ax.grid()

            if True:
                best_index, best_O = get_min_and_index(O)
                print best_index, best_O
                xy_best = (x[best_index], y[best_index])
                handle_best = ax.scatter(*xy_best, s=s*3, marker='s', facecolors='none', edgecolors='r')
                ax.legend((handle_best,), ('Best',))

            return scatter_handle
        
        fig, axeses = subplots(2, 2, sharex=False, dpi=150, figsize=(12, 6), facecolor='w', edgecolor='k')
        # fig, axeses = subplots(2, 2, sharex=False, dpi=150, figsize=(10, 8), facecolor='w', edgecolor='k')
        fig.subplots_adjust(right=0.9, hspace=0.21, wspace=0.11) # won't work after I did something. just manual adjust!

        # Use FRW and TRV
        if True:
            # TRV vs Torque Ripple
            ax = axeses[0][0]
            xy_ref = (torque_average/rotor_volume/1e3, normalized_torque_ripple) # from run#117
            x, y = array(list(swda.get_certain_objective_function(2)))/rotor_volume/1e3, list(swda.get_certain_objective_function(3))
            x = x.tolist()
            my_scatter_plot(x,y,O2[::],xy_ref,O2_ref, fig=fig, ax=ax)
            ax.set_xlabel('TRV [kNm/m^3]\n(a)')
            ax.set_ylabel(r'$T_{\rm rip}$ [100%]')

            # FRW vs Ea
            ax = axeses[0][1]
            xy_ref = (ss_avg_force_magnitude/rotor_weight, force_error_angle)
            x, y = array(list(swda.get_certain_objective_function(4)))/rotor_weight, list(swda.get_certain_objective_function(6))
            x = x.tolist()
            my_scatter_plot(x,y,O2[::],xy_ref,O2_ref, fig=fig, ax=ax)
            ax.set_xlabel('FRW [1]\n(b)')
            ax.set_ylabel(r'$E_a$ [deg]')

            # FRW vs Em
            ax = axeses[1][0]
            xy_ref = (ss_avg_force_magnitude/rotor_weight, normalized_force_error_magnitude)
            x, y = array(list(swda.get_certain_objective_function(4)))/rotor_weight, list(swda.get_certain_objective_function(5))
            x = x.tolist()
            my_scatter_plot(x,y,O2[::],xy_ref,O2_ref, fig=fig, ax=ax)
            ax.set_xlabel('FRW [1]\n(c)')
            ax.set_ylabel(r'$E_m$ [100%]')

            # Em vs Ea
            ax = axeses[1][1]
            xy_ref = (normalized_force_error_magnitude, force_error_angle)
            x, y = list(swda.get_certain_objective_function(5)), list(swda.get_certain_objective_function(6))
            scatter_handle = my_scatter_plot(x,y,O2[::],xy_ref,O2_ref, fig=fig, ax=ax)
            ax.set_xlabel('$E_m$ [100%]\n(d)')
            ax.set_ylabel(r'$E_a$ [deg]')

            # fig.subplots_adjust(right=0.8)
            # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7]) # left, bottom, width, height
            # fig.subplots_adjust(right=0.9)
            # cbar_ax = fig.add_axes([0.925, 0.15, 0.025, 0.7])
            cbar_ax = fig.add_axes([0.925, 0.15, 0.02, 0.7])
            cbar_ax.get_yaxis().labelpad = 10
            clb = fig.colorbar(scatter_handle, cax=cbar_ax)
            clb.ax.set_ylabel(r'Cost function $O_2$', rotation=270)
            # clb.ax.set_title(r'Cost function $O_2$', rotation=0)

        # Use Torque and Force
        if False:

            # Torque vs Torque Ripple
            ax = axeses[0][0]
            xy_ref = (19.1197, 0.0864712) # from run#117
            x, y = list(swda.get_certain_objective_function(2)), list(swda.get_certain_objective_function(3))
            my_scatter_plot(x,y,O2[::],xy_ref,O2_ref, fig=fig, ax=ax)
            ax.set_xlabel('$T_{em}$ [Nm]\n(a)')
            ax.set_ylabel(r'$T_{\rm rip}$ [100%]')

            # Force vs Ea
            ax = axeses[0][1]
            xy_ref = (96.9263, 6.53137)
            x, y = list(swda.get_certain_objective_function(4)), list(swda.get_certain_objective_function(6))
            my_scatter_plot(x,y,O2[::],xy_ref,O2_ref, fig=fig, ax=ax)
            ax.set_xlabel('$|F|$ [N]\n(b)')
            ax.set_ylabel(r'$E_a$ [deg]')

            # Force vs Em
            ax = axeses[1][0]
            xy_ref = (96.9263, 0.104915)
            x, y = list(swda.get_certain_objective_function(4)), list(swda.get_certain_objective_function(5))
            my_scatter_plot(x,y,O2[::],xy_ref,O2_ref, fig=fig, ax=ax)
            ax.set_xlabel('$|F|$ [N]\n(c)')
            ax.set_ylabel(r'$E_m$ [100%]')

            # Em vs Ea
            ax = axeses[1][1]
            xy_ref = (0.104915, 6.53137)
            x, y = list(swda.get_certain_objective_function(5)), list(swda.get_certain_objective_function(6))
            scatter_handle = my_scatter_plot(x,y,O2[::],xy_ref,O2_ref, fig=fig, ax=ax)
            ax.set_xlabel('$E_m$ [100%]\n(d)')
            ax.set_ylabel(r'$E_a$ [deg]')

            # fig.subplots_adjust(right=0.8)
            # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7]) # left, bottom, width, height
            # fig.subplots_adjust(right=0.9)
            # cbar_ax = fig.add_axes([0.925, 0.15, 0.025, 0.7])
            cbar_ax = fig.add_axes([0.925, 0.15, 0.02, 0.7])
            cbar_ax.get_yaxis().labelpad = 10
            clb = fig.colorbar(scatter_handle, cax=cbar_ax)
            clb.ax.set_ylabel(r'Cost function $O_2$', rotation=270)
            # clb.ax.set_title(r'Cost function $O_2$', rotation=0)


            fig.tight_layout()
            # fig.savefig(r'D:\OneDrive\[00]GetWorking\32 blimopti\p2019_ecce_bearingless_induction\images\pareto_plot.png', dpi=150, bbox_inches='tight')
            show()

            # Loss vs Ea
            xy_ref = ((1817.22+216.216+224.706), 6.53137)
            x, y = array(list(swda.get_certain_objective_function(9))) + array(list(swda.get_certain_objective_function(12))) + array(list(swda.get_certain_objective_function(13))), list(swda.get_certain_objective_function(6))
            x = x.tolist()
            my_scatter_plot(x,y,O2[::],xy_ref,O2_ref)
            xlabel(r'$P_{\rm Cu,Fe}$ [W]')
            ylabel(r'$E_a$ [deg]')

            quit()   

        # Efficiency vs Rated power stack length
        fig, ax = subplots(1, 1, sharex=False, dpi=150, figsize=(12, 6), facecolor='w', edgecolor='k')
        def get_rated_values(l_torque_average, l_total_loss):
            l_rated_stack_length = []
            l_rated_total_loss = []
            for torque_average, total_loss in zip(l_torque_average, l_total_loss):
                rated_stack_length = stack_length / torque_average * required_torque
                l_rated_stack_length.append(rated_stack_length)

                rated_total_loss   = total_loss / stack_length * rated_stack_length
                l_rated_total_loss.append(rated_total_loss)

            return l_rated_stack_length, l_rated_total_loss
        a, b = get_rated_values([torque_average], [total_loss])
        rated_stack_length = a[0]
        rated_total_loss = b[0]
        print 'stack_length=', stack_length, 'mm, rated_stack_length=', rated_stack_length, 'mm'
        print 'total_loss=', total_loss, 'W, rated_total_loss=', rated_total_loss, 'W'
        xy_ref = (rated_stack_length, 1 - rated_total_loss/70e3)

        x, y = get_rated_values(list(swda.get_certain_objective_function(2)), 
                                array(list(swda.get_certain_objective_function(9))) + array(list(swda.get_certain_objective_function(12))) + array(list(swda.get_certain_objective_function(13))))
        y = 1 - array(y)/70e3
        y = y.tolist()
        my_scatter_plot(x,y,O2[::],xy_ref,O2_ref, fig=fig, ax=ax)
        ax.set_xlabel('Stack length [mm]')
        ax.set_ylabel(r'Efficiency at 70 kW [1]')



        fig.tight_layout()
        # fig.savefig(r'D:\OneDrive\[00]GetWorking\32 blimopti\p2019_ecce_bearingless_induction\images\pareto_plot.png', dpi=150, bbox_inches='tight')
        # fig.savefig(r'D:\OneDrive\[00]GetWorking\32 blimopti\p2019_ecce_bearingless_induction_full_paper\images\pareto_plot.png', dpi=150, bbox_inches='tight')
        show()
        quit()


    # ------------------------------------ Sensitivity Analysis Bar Chart Scripts
    # ------------------------------------ Sensitivity Analysis Bar Chart Scripts
    # ------------------------------------ Sensitivity Analysis Bar Chart Scripts
    if False: # 4 pole motor
        # swda = SwarmDataAnalyzer(run_integer=113)
        # swda = SwarmDataAnalyzer(run_integer=200)

        # swda = SwarmDataAnalyzer(run_integer=115)
        # number_of_variant = 5 + 1

        swda = SwarmDataAnalyzer(run_integer=116)
        number_of_variant = 20 + 1

        # swda = SwarmDataAnalyzer(run_integer=117)
        # number_of_variant = 1
            # gives the reference values:
            # 0 [0.635489] <-In population.py   [0.65533] <- from initial_design.txt
            # 1 [0.963698] <-In population.py   [0.967276] <- from initial_design.txt
            # 2 [19.1197]  <-In population.py  [16.9944] <- from initial_design.txt
            # 3 [0.0864712]<-In population.py    [0.0782085] <- from initial_design.txt
            # 4 [96.9263]  <-In population.py  [63.6959] <- from initial_design.txt
            # 5 [0.104915] <-In population.py   [0.159409] <- from initial_design.txt
            # 6 [6.53137]  <-In population.py  [10.1256] <- from initial_design.txt
            # 7 [1817.22]  <-In population.py  [1353.49] <- from initial_design.txt
    else: # 2 pole motor
        # swda = SwarmDataAnalyzer(run_integer=184)
        # swda = SwarmDataAnalyzer(run_integer=190) # Wrong bounds
        swda = SwarmDataAnalyzer(run_integer=191) # Correct bounds
        number_of_variant = 20 + 1

    fi, axeses = subplots(4, 2, sharex=True, dpi=150, figsize=(16, 8), facecolor='w', edgecolor='k')
    ax_list = []
    for i in range(4):
        ax_list.extend(axeses[i].tolist())

    param_list = ['stator_tooth_width_b_ds',
    'air_gap_length_delta',
    'Width_RotorSlotOpen ',
    'rotor_tooth_width_b_dr',
    'Length_HeadNeckRotorSlot',
    'Angle_StatorSlotOpen',
    'Width_StatorTeethHeadThickness']
    # [power_factor, efficiency, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle]
    y_label_list = ['PF', r'$\eta$ [100%]', r'$T_{em} [N]$', r'$T_{rip}$ [100%]', r'$|F|$ [N]', r'$E_m$ [100%]', r'$E_a$ [deg]', r'$P_{Cu,s}$', r'$P_{Cu,r}$', r'$P_{Fe}$ [W]', r'$P_{eddy}$', r'$P_{hyst}$', r'$P_{Cu,s}$', r'$P_{Cu,r}$']
    # print next(swda.get_list_objective_function())
    data_max = []
    data_min = []
    eta_at_50kW_max = []
    eta_at_50kW_min = []
    O1_max   = []
    O1_min   = []
    for ind, i in enumerate(range(7)+[9]):
    # for i in range(14):
        print '\n-----------', y_label_list[i]
        l = list(swda.get_certain_objective_function(i))

        if i == 9: # replace P_Fe with P_Fe,Cu
            l_femm_stator_copper = array(list(swda.get_certain_objective_function(12)))
            l_femm_rotor_copper  = array(list(swda.get_certain_objective_function(13)))
            y = array(l) + l_femm_stator_copper + l_femm_rotor_copper 
            # print l, len(l)
            # print y, len(y)
            # quit()
        else:
            # y = l[:len(l)/2] # 115
            y = l # 116
        print 'ind=', ind, 'i=', i, 'len(y)=', len(y)

        data_max.append([])
        data_min.append([])

        for j in range(len(y)/number_of_variant): # iterate design parameters
            y_vs_design_parameter = y[j*number_of_variant:(j+1)*number_of_variant]

            try:
                # if j == 6:
                ax_list[ind].plot(y_vs_design_parameter, label=str(j)+' '+param_list[j], alpha=0.5)
            except IndexError as e:
                print 'Check the length of y should be 7*21=147, or else you should remove the redundant results in swarm_data.txt (they are produced because of the interrupted/resumed script run.)'
                raise e
            print '\tj=', j, param_list[j], '\t\t', max(y_vs_design_parameter) - min(y_vs_design_parameter)

            data_max[ind].append(max(y_vs_design_parameter))
            data_min[ind].append(min(y_vs_design_parameter))            

        if i==1:
            ax_list[ind].legend()
        ax_list[ind].grid()
        ax_list[ind].set_ylabel(y_label_list[i])

    for ind, el in enumerate(data_max):
        print ind, el
    print
    for ind, el in enumerate(data_min):
        print ind, el

    if swda.reference_design is None:
        # add reference design results manually as follows:
        O2_ref = fobj_scalar(19.1197, 96.9263, 0.0864712, 0.104915, 6.53137, (1817.22+216.216+224.706), weights=O2_weights, rotor_volume=rotor_volume, rotor_weight=rotor_weight)
        O1_ref = fobj_scalar(19.1197, 96.9263, 0.0864712, 0.104915, 6.53137, (1817.22+216.216+224.706), weights=O1_weights, rotor_volume=rotor_volume, rotor_weight=rotor_weight)
    else:
        print '-------------------- Here goes the reference design:'
        for el in swda.reference_design[1:]:
            print el,
        swda.reference_data = [float(el) for el in swda.reference_design[3].split(',')]
        O2_ref = fobj_scalar(swda.reference_data[2],
                             swda.reference_data[4],
                             swda.reference_data[3],
                             swda.reference_data[5],
                             swda.reference_data[6], (swda.reference_data[-5]+swda.reference_data[-1]+swda.reference_data[-2]), 
                             weights=O2_weights, rotor_volume=rotor_volume, rotor_weight=rotor_weight)
        O1_ref = fobj_scalar(swda.reference_data[2],
                             swda.reference_data[4],
                             swda.reference_data[3],
                             swda.reference_data[5],
                             swda.reference_data[6], (swda.reference_data[-5]+swda.reference_data[-1]+swda.reference_data[-2]), 
                             weights=O1_weights, rotor_volume=rotor_volume, rotor_weight=rotor_weight)

    print  'Objective function 1'
    O1 = fobj_list( list(swda.get_certain_objective_function(2)), 
                    list(swda.get_certain_objective_function(4)), 
                    list(swda.get_certain_objective_function(3)), 
                    list(swda.get_certain_objective_function(5)), 
                    list(swda.get_certain_objective_function(6)), 
                    array(list(swda.get_certain_objective_function(9))) + array(list(swda.get_certain_objective_function(12))) + array(list(swda.get_certain_objective_function(13))),
                    weights=O1_weights, rotor_volume=rotor_volume, rotor_weight=rotor_weight)
    O1_max = []
    O1_min = []
    O1_ax  = figure().gca()
    for j in range(len(O1)/number_of_variant): # iterate design parameters
        O1_vs_design_parameter = O1[j*number_of_variant:(j+1)*number_of_variant]

        O1_ax.plot(O1_vs_design_parameter, label=str(j)+' '+param_list[j], alpha=0.5)
        print '\t', j, param_list[j], '\t\t', max(O1_vs_design_parameter) - min(O1_vs_design_parameter)

        O1_max.append(max(O1_vs_design_parameter))
        O1_min.append(min(O1_vs_design_parameter))            
    O1_ax.legend()
    O1_ax.grid()
    O1_ax.set_ylabel('O1 [1]')
    O1_ax.set_xlabel('Count of design variants')

    print  'Objective function 2'
    O2 = fobj_list( list(swda.get_certain_objective_function(2)), 
                    list(swda.get_certain_objective_function(4)), 
                    list(swda.get_certain_objective_function(3)), 
                    list(swda.get_certain_objective_function(5)), 
                    list(swda.get_certain_objective_function(6)), 
                    array(list(swda.get_certain_objective_function(9))) + array(list(swda.get_certain_objective_function(12))) + array(list(swda.get_certain_objective_function(13))),
                    weights=O2_weights, rotor_volume=rotor_volume, rotor_weight=rotor_weight )
    O2_max = []
    O2_min = []
    O2_ax  = figure().gca()
    O2_ecce_data = []
    for j in range(len(O2)/number_of_variant): # iterate design parameters: range(7)
        O2_vs_design_parameter = O2[j*number_of_variant:(j+1)*number_of_variant]
        O2_ecce_data.append(O2_vs_design_parameter)

        # narrow bounds (refine bounds)
        O2_ax.plot(O2_vs_design_parameter, 'o-', label=str(j)+' '+param_list[j], alpha=0.5)
        print '\t', j, param_list[j], '\t\t', max(O2_vs_design_parameter) - min(O2_vs_design_parameter), '\t\t',
        print [ind for ind, el in enumerate(O2_vs_design_parameter) if el < O2_ref*1.005] #'<- to derive new original_bounds.'

        O2_max.append(max(O2_vs_design_parameter))
        O2_min.append(min(O2_vs_design_parameter))            
    O2_ax.legend()
    O2_ax.grid()
    O2_ax.set_ylabel('O2 [1]')
    O2_ax.set_xlabel('Count of design variants')

    # for ecce digest
    fig_ecce = figure(figsize=(10, 5), facecolor='w', edgecolor='k')
    O2_ecce_ax = fig_ecce.gca()
    O2_ecce_ax.plot(range(-1, 22), O2_ref*np.ones(23), 'k--', label='reference design')
    O2_ecce_ax.plot(O2_ecce_data[1], 'o-', lw=0.75, alpha=0.5, label=r'$\delta$'         )
    O2_ecce_ax.plot(O2_ecce_data[0], 'v-', lw=0.75, alpha=0.5, label=r'$b_{\rm tooth,s}$')
    O2_ecce_ax.plot(O2_ecce_data[3], 's-', lw=0.75, alpha=0.5, label=r'$b_{\rm tooth,r}$')
    O2_ecce_ax.plot(O2_ecce_data[5], '^-', lw=0.75, alpha=0.5, label=r'$w_{\rm open,s}$')
    O2_ecce_ax.plot(O2_ecce_data[2], 'd-', lw=0.75, alpha=0.5, label=r'$w_{\rm open,r}$')
    O2_ecce_ax.plot(O2_ecce_data[6], '*-', lw=0.75, alpha=0.5, label=r'$h_{\rm head,s}$')
    O2_ecce_ax.plot(O2_ecce_data[4], 'X-', lw=0.75, alpha=0.5, label=r'$h_{\rm head,r}$')

    myfontsize = 12.5
    rcParams.update({'font.size': myfontsize})


    # Reference candidate design
    ref = zeros(8)
        # ref[0] = 0.635489                                   # PF
        # ref[1] = 0.963698                                   # eta
        # ref[1] = efficiency_at_50kW(1817.22+216.216+224.706)# eta@50kW
    O1_ax.plot(range(-1, 22), O1_ref*np.ones(23), 'k--')
    O2_ax.plot(range(-1, 22), O2_ref*np.ones(23), 'k--')
    O2_ecce_ax.legend()
    O2_ecce_ax.grid()
    O2_ecce_ax.set_xticks(range(21))
    O2_ecce_ax.annotate('Lower bound', xytext=(0.5, 5.5), xy=(0, 4), xycoords='data', arrowprops=dict(arrowstyle="->"))
    O2_ecce_ax.annotate('Upper bound', xytext=(18.0, 5.5),  xy=(20, 4), xycoords='data', arrowprops=dict(arrowstyle="->"))
    O2_ecce_ax.set_xlim((-0.5,20.5))
    O2_ecce_ax.set_ylim((0,14)) # 4,14
    O2_ecce_ax.set_xlabel(r'Number of design variant', fontsize=myfontsize)
    O2_ecce_ax.set_ylabel(r'$O_2(x)$ [1]', fontsize=myfontsize)
    fig_ecce.tight_layout()
    # fig_ecce.savefig(r'D:\OneDrive\[00]GetWorking\32 blimopti\p2019_ecce_bearingless_induction_full_paper\images\O2_vs_params.png', dpi=150)
    show()
    # quit() ###################################

    if swda.reference_design is None:
        list_plotting_weights = [8, 3, required_torque, 0.1, rotor_weight, 0.2, 10, 2500]
        # manually set this up
        ref[0] = O2_ref    / list_plotting_weights[0]
        ref[1] = O1_ref    / list_plotting_weights[1]
        ref[2] = 19.1197   / list_plotting_weights[2]                   # 100%
        ref[3] = 0.0864712 / list_plotting_weights[3]                   # 100%
        ref[4] = 96.9263   / list_plotting_weights[4]                   # 100% = FRW
        ref[5] = 0.104915  / list_plotting_weights[5]                   # 100%
        ref[6] = 6.53137   / list_plotting_weights[6]                   # deg
        ref[7] = (1817.22+216.216+224.706) / list_plotting_weights[7]   # W
    else:
        list_plotting_weights = [8, 3, required_torque, 0.1, rotor_weight, 0.2, 10, 2100]
        ref[0] = O2_ref                                                                   / list_plotting_weights[0] 
        ref[1] = O1_ref                                                                   / list_plotting_weights[1] 
        ref[2] = swda.reference_data[2]                                                   / list_plotting_weights[2]  # 100%
        ref[3] = swda.reference_data[3]                                                   / list_plotting_weights[3]  # 100%
        ref[4] = swda.reference_data[4]                                                   / list_plotting_weights[4]  # 100% = FRW
        ref[5] = swda.reference_data[5]                                                   / list_plotting_weights[5]  # 100%
        ref[6] = swda.reference_data[6]                                                   / list_plotting_weights[6]  # deg
        ref[7] = (swda.reference_data[-5]+swda.reference_data[-1]+swda.reference_data[-2])/ list_plotting_weights[7]  # W

    # Maximum
    data_max = array(data_max)
    O1_max   = array(O1_max)
    O2_max   = array(O2_max)
        # data_max[0] = (data_max[0])                   # PF
        # data_max[1] = (data_max[1])                   # eta
        # data_max[1] = efficiency_at_50kW(data_max[7]) # eta@50kW # should use data_min[7] because less loss, higher efficiency
    data_max[0] = O2_max       / list_plotting_weights[0]  
    data_max[1] = O1_max       / list_plotting_weights[1]  
    data_max[2] = (data_max[2])/ list_plotting_weights[2]  # 100%
    data_max[3] = (data_max[3])/ list_plotting_weights[3]  # 100%
    data_max[4] = (data_max[4])/ list_plotting_weights[4]  # 100% = FRW
    data_max[5] = (data_max[5])/ list_plotting_weights[5]  # 100%
    data_max[6] = (data_max[6])/ list_plotting_weights[6]  # deg
    data_max[7] = (data_max[7])/ list_plotting_weights[7]  # W
    y_max_vs_design_parameter_0 = [el[0] for el in data_max]
    y_max_vs_design_parameter_1 = [el[1] for el in data_max]
    y_max_vs_design_parameter_2 = [el[2] for el in data_max]
    y_max_vs_design_parameter_3 = [el[3] for el in data_max]
    y_max_vs_design_parameter_4 = [el[4] for el in data_max]
    y_max_vs_design_parameter_5 = [el[5] for el in data_max]
    y_max_vs_design_parameter_6 = [el[6] for el in data_max]

    # Minimum
    data_min = array(data_min)
    O1_min   = array(O1_min)
    O2_min   = array(O2_min)
        # data_min[0] = (data_min[0])                    # PF
        # data_min[1] = (data_min[1])                    # eta
        # data_min[1] = efficiency_at_50kW(data_min[7])  # eta@50kW
    data_min[0] = O2_min        / list_plotting_weights[0] 
    data_min[1] = O1_min        / list_plotting_weights[1] 
    data_min[2] = (data_min[2]) / list_plotting_weights[2] # 100%
    data_min[3] = (data_min[3]) / list_plotting_weights[3] # 100%
    data_min[4] = (data_min[4]) / list_plotting_weights[4] # 100% = FRW
    data_min[5] = (data_min[5]) / list_plotting_weights[5] # 100%
    data_min[6] = (data_min[6]) / list_plotting_weights[6] # deg
    data_min[7] = (data_min[7]) / list_plotting_weights[7] # W
    y_min_vs_design_parameter_0 = [el[0] for el in data_min]
    y_min_vs_design_parameter_1 = [el[1] for el in data_min]
    y_min_vs_design_parameter_2 = [el[2] for el in data_min]
    y_min_vs_design_parameter_3 = [el[3] for el in data_min]
    y_min_vs_design_parameter_4 = [el[4] for el in data_min]
    y_min_vs_design_parameter_5 = [el[5] for el in data_min]
    y_min_vs_design_parameter_6 = [el[6] for el in data_min]

    count = np.arange(len(y_max_vs_design_parameter_0))  # the x locations for the groups
    width = 1.0  # the width of the bars

    fig, ax = plt.subplots(dpi=150, figsize=(16, 8), facecolor='w', edgecolor='k')                                      #  #1034A
    rects2 = ax.bar(count - 3*width/8, y_min_vs_design_parameter_1, width/8, alpha=0.5, label=r'$\delta$,           Air gap length', color='#6593F5')
    rects1 = ax.bar(count - 2*width/8, y_min_vs_design_parameter_0, width/8, alpha=0.5, label=r'$b_{\rm tooth,s}$, Stator tooth width', color='#1D2951') # https://digitalsynopsis.com/design/beautiful-color-palettes-combinations-schemes/
    rects4 = ax.bar(count - 1*width/8, y_min_vs_design_parameter_3, width/8, alpha=0.5, label=r'$b_{\rm tooth,r}$, Rotor tooth width', color='#03396c')
    rects6 = ax.bar(count + 0*width/8, y_min_vs_design_parameter_5, width/8, alpha=0.5, label=r'$w_{\rm open,s}$, Stator slot open', color='#6497b1')
    rects3 = ax.bar(count + 1*width/8, y_min_vs_design_parameter_2, width/8, alpha=0.5, label=r'$w_{\rm open,r}$, Rotor slot open',  color='#0E4D92')
    rects5 = ax.bar(count + 2*width/8, y_min_vs_design_parameter_4, width/8, alpha=0.5, label=r'$h_{\rm head,s}$, Stator head height', color='#005b96')
    rects7 = ax.bar(count + 3*width/8, y_min_vs_design_parameter_6, width/8, alpha=0.5, label=r'$h_{\rm head,r}$, Rotor head height', color='#b3cde0') 
    print 'ylim=', ax.get_ylim()
    autolabel(ax, rects1, bias=-0.10)
    autolabel(ax, rects2, bias=-0.10)
    autolabel(ax, rects3, bias=-0.10)
    autolabel(ax, rects4, bias=-0.10)
    autolabel(ax, rects5, bias=-0.10)
    autolabel(ax, rects6, bias=-0.10)
    autolabel(ax, rects7, bias=-0.10)
    one_one = array([1, 1])
    minus_one_one = array([-1, 1])
    ax.plot(rects6[0].get_x() + 0.5*width*minus_one_one, ref[0]*one_one, 'k--', lw=1.0, alpha=0.6, label='Reference design' )
    ax.plot(rects6[1].get_x() + 0.5*width*minus_one_one, ref[1]*one_one, 'k--', lw=1.0, alpha=0.6 )
    ax.plot(rects6[2].get_x() + 0.5*width*minus_one_one, ref[2]*one_one, 'k--', lw=1.0, alpha=0.6 )
    ax.plot(rects6[3].get_x() + 0.5*width*minus_one_one, ref[3]*one_one, 'k--', lw=1.0, alpha=0.6 )
    ax.plot(rects6[4].get_x() + 0.5*width*minus_one_one, ref[4]*one_one, 'k--', lw=1.0, alpha=0.6 )
    ax.plot(rects6[5].get_x() + 0.5*width*minus_one_one, ref[5]*one_one, 'k--', lw=1.0, alpha=0.6 )
    ax.plot(rects6[6].get_x() + 0.5*width*minus_one_one, ref[6]*one_one, 'k--', lw=1.0, alpha=0.6 )
    ax.plot(rects6[7].get_x() + 0.5*width*minus_one_one, ref[7]*one_one, 'k--', lw=1.0, alpha=0.6 )
    ax.legend(loc='upper right') 
    ax.text(rects6[0].get_x() - 3.5/8*width, ref[0]*1.01, '%.2f'%(ref[0]), ha='center', va='bottom', rotation=90)
    ax.text(rects6[1].get_x() - 3.5/8*width, ref[1]*1.01, '%.2f'%(ref[1]), ha='center', va='bottom', rotation=90)
    ax.text(rects6[2].get_x() - 3.5/8*width, ref[2]*1.01, '%.2f'%(ref[2]), ha='center', va='bottom', rotation=90)
    ax.text(rects6[3].get_x() - 3.5/8*width, ref[3]*1.01, '%.2f'%(ref[3]), ha='center', va='bottom', rotation=90)
    ax.text(rects6[4].get_x() - 3.5/8*width, ref[4]*1.01, '%.2f'%(ref[4]), ha='center', va='bottom', rotation=90)
    ax.text(rects6[5].get_x() - 3.5/8*width, ref[5]*1.01, '%.2f'%(ref[5]), ha='center', va='bottom', rotation=90)
    ax.text(rects6[6].get_x() - 3.5/8*width, ref[6]*1.01, '%.2f'%(ref[6]), ha='center', va='bottom', rotation=90)
    ax.text(rects6[7].get_x() - 3.5/8*width, ref[7]*1.01, '%.2f'%(ref[7]), ha='center', va='bottom', rotation=90)

    rects1 = ax.bar(count - 2*width/8, y_max_vs_design_parameter_0, width/8, alpha=0.5, label=r'$b_{\rm tooth,s}$', color='#1D2951') # bottom=y_min_vs_design_parameter_0, 
    rects2 = ax.bar(count - 3*width/8, y_max_vs_design_parameter_1, width/8, alpha=0.5, label=r'$\delta$',          color='#6593F5') # bottom=y_min_vs_design_parameter_1, 
    rects3 = ax.bar(count + 1*width/8, y_max_vs_design_parameter_2, width/8, alpha=0.5, label=r'$w_{\rm open,r}$',  color='#0E4D92') # bottom=y_min_vs_design_parameter_2, 
    rects4 = ax.bar(count - 1*width/8, y_max_vs_design_parameter_3, width/8, alpha=0.5, label=r'$b_{\rm tooth,r}$', color='#03396c') # bottom=y_min_vs_design_parameter_3, 
    rects5 = ax.bar(count + 2*width/8, y_max_vs_design_parameter_4, width/8, alpha=0.5, label=r'$h_{head,s}$',      color='#005b96') # bottom=y_min_vs_design_parameter_4, 
    rects6 = ax.bar(count + 0*width/8, y_max_vs_design_parameter_5, width/8, alpha=0.5, label=r'$w_{\rm open,s}$',  color='#6497b1') # bottom=y_min_vs_design_parameter_5, 
    rects7 = ax.bar(count + 3*width/8, y_max_vs_design_parameter_6, width/8, alpha=0.5, label=r'$h_{head,r}$',      color='#b3cde0') # bottom=y_min_vs_design_parameter_6, 
    autolabel(ax, rects1)
    autolabel(ax, rects2)
    autolabel(ax, rects3)
    autolabel(ax, rects4)
    autolabel(ax, rects5)
    autolabel(ax, rects6)
    autolabel(ax, rects7)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Normalized Objective Functions')
    ax.set_xticks(count)
    # ax.set_xticklabels(('Power Factor [100%]', r'$\eta$@$T_{em}$ [100%]', r'$T_{em}$ [15.9 N]', r'$T_{rip}$ [10%]', r'$|F|$ [51.2 N]', r'    $E_m$ [20%]', r'      $E_a$ [10 deg]', r'$P_{\rm Cu,Fe}$ [2.5 kW]')))
    # ax.set_xticklabels(('Power Factor [100%]', r'$O_1$ [3]', r'$T_{em}$ [15.9 N]', r'$T_{rip}$ [10%]', r'$|F|$ [51.2 N]', r'    $E_m$ [20%]', r'      $E_a$ [10 deg]', r'$P_{\rm Cu,Fe}$ [2.5 kW]'))
    ax.set_xticklabels(('$O_2$ [%g]'               %(list_plotting_weights[0]), 
                        '$O_1$ [%g]'               %(list_plotting_weights[1]), 
                        '$T_{em}$ [%g Nm]'         %(list_plotting_weights[2]), 
                        '$T_{rip}$ [%g%%]'         %(list_plotting_weights[3]*100), 
                        '$|F|$ [%g N]'             %(list_plotting_weights[4]), 
                        '    $E_m$ [%g%%]'         %(list_plotting_weights[5]*100), 
                        '      $E_a$ [%g deg]'     %(list_plotting_weights[6]), 
                        '$P_{\\rm Cu,Fe}$ [%g kW]' %(list_plotting_weights[7]*1e-3) ))
    ax.grid()
    fig.tight_layout()
    # fig.savefig(r'D:\OneDrive\[00]GetWorking\32 blimopti\p2019_ecce_bearingless_induction\images\sensitivity_results.png', dpi=150)

    show()

    quit()


# test for power factor (Goertzel Algorithm with periodic extension 1000), etc.
if __name__ == '__main__':
    swda = SwarmDataAnalyzer(run_integer=121)

    # Pseudo Pareto Optimal Front
    gen_best = swda.get_best_generation()
    with open('d:/gen#0000.txt', 'w') as f:
        f.write('\n'.join(','.join('%.16f'%(x) for x in y) for y in gen_best)) # convert 2d array to string            

    design_parameters_norm = (gen_best - min_b) / diff


    for el in design_parameters_norm:
        print ','.join('%.4f'%(_) for _ in el.tolist())
    print 'airgap length\n', [el[1] for el in gen_best]

    print 'Average value of design parameters'
    avg_design_parameters = []
    for el in design_parameters_norm.T:
        avg_design_parameters.append(sum(el)/len(el))
    print avg_design_parameters
    avg_design_parameters_denorm = min_b + avg_design_parameters * diff
    print avg_design_parameters_denorm

    # for design in swda.get_best_generation(generator=swda.design_display_generator()):
    #     print ''.join(design),
    quit()

    cost = swda.list_cost_function()
    indices, items = min_indices(cost, 50)
    print indices
    print items

    stop = max(indices)
    start = min(indices)
    print start, stop

    gene = swda.design_parameters_generator()
    gen_best = []
    for index, design in enumerate(gene):
        if index in indices:
            gen_best.append(design)
    print 'there are', index, 'designs'

    # print  len(list(gene))
    # for index in indices:
    #     start = index
    # print next(itertools.islice(gene, start, None)) 


    quit()
    # print min_indices([3,5,7,4,2], 5)
    # print max_indices([3,5,7,4,2], 5)
    # quit()



    print ''.join(swda.buf[:21]),

    print swda.find_individual(14, 0)


    for design in swda.design_display_generator():
        print design,
        break

    for design in swda.design_parameters_generator():
        print design
        break

    print 

    # for generation in range(5):
    #     print '----------gen#%d'%(generation)
    #     generation_file_path = r'D:\OneDrive - UW-Madison\c\pop\run#107/' + 'gen#%04d.txt'%(generation)
    #     print generation_file_path
    #     if os.path.exists( generation_file_path ):
    #         with open(generation_file_path, 'r') as f:
    #             for el in f.readlines():
    #                 print el[:-1]

    # read voltage and current to see the power factor!
    # read voltage and current to see the power factor!
    # read voltage and current to see the power factor!
    
    # 绘制损耗图形。
    # 绘制损耗图形。
    # 绘制损耗图形。

    if False:
        from pylab import *
        gs_u = Goertzel_Data_Struct("Goertzel Struct for Voltage\n")
        gs_i = Goertzel_Data_Struct("Goertzel Struct for Current\n")

        phase = arccos(0.6) # PF=0.6
        targetFreq = 1000.
        TS = 1.5625e-5

        if False: # long signal
            time = arange(0./targetFreq, 100.0/targetFreq, TS)
            voltage = 3*sin(targetFreq*2*pi*time + 30/180.*pi)
            current = 2*sin(targetFreq*2*pi*time + 30/180.*pi - phase)
        else: # half period
            time = arange(0.5/targetFreq, 1.0/targetFreq, TS)
            voltage = 3*sin(targetFreq*2*pi*time + 30/180.*pi)
            current = 2*sin(targetFreq*2*pi*time + 30/180.*pi - phase)

        N_SAMPLE = len(voltage)
        noise = ( 2*rand(N_SAMPLE) - 1 ) * 0.233
        voltage += noise

        # test
        print 'PF=', compute_power_factor_from_half_period(voltage, current, time, targetFreq=1e3)
        quit()

        # plot(voltage+0.5)
        # plot(current+0.5)

        # Check for resolution
        end_time = N_SAMPLE * TS
        resolution = 1./end_time
        print resolution, targetFreq
        if resolution > targetFreq:

            if True: # for half period signal
                print 'Data length (N_SAMPLE) too short or sampling frequency too high (1/TS too high).'
                print 'Periodical extension is applied. This will not really increase your resolution. It is a visual trick for Fourier Analysis.'
                voltage = voltage.tolist() + (-voltage).tolist() #[::-1]
                current = current.tolist() + (-current).tolist() #[::-1]

            voltage *= 1000
            current *= 1000

            N_SAMPLE = len(voltage)
            end_time = N_SAMPLE * TS
            resolution = 1./end_time
            print resolution, targetFreq, 'Hz'
            if resolution <= targetFreq:
                print 'Now we are good.'


            # 目前就写了二分之一周期的处理，只有四分之一周期的数据，可以用反对称的方法周期延拓。

        print gs_u.goertzel_offline(targetFreq, 1./TS, voltage)
        print gs_i.goertzel_offline(targetFreq, 1./TS, current)

        gs_u.ampl = sqrt(gs_u.real*gs_u.real + gs_u.imag*gs_u.imag) 
        gs_u.phase = arctan2(gs_u.imag, gs_u.real)

        print 'N_SAMPLE=', N_SAMPLE
        print "\n"
        print gs_u.id
        print "RT:\tAmplitude: %g, %g\n" % (gs_u.ampl, sqrt(2.0*gs_u.accumSquaredData/N_SAMPLE))
        print "RT:\tre=%g, im=%g\n\tampl=%g, phase=%g\n" % (gs_u.real, gs_u.imag, gs_u.ampl, gs_u.phase*180/math.pi)

        # // do the analysis
        gs_i.ampl = sqrt(gs_i.real*gs_i.real + gs_i.imag*gs_i.imag) 
        gs_i.phase = arctan2(gs_i.imag, gs_i.real)
        print gs_i.id
        print "RT:\tAmplitude: %g, %g\n" % (gs_i.ampl, sqrt(2.0*gs_i.accumSquaredData/N_SAMPLE))
        print "RT:\tre=%g, im=%g\n\tampl=%g, phase=%g\n" % (gs_i.real, gs_i.imag, gs_i.ampl, gs_i.phase*180/math.pi)

        print "------------------------"
        print "\tPhase Difference of I (version 1): %g\n" % ((gs_i.phase-gs_u.phase)*180/math.pi), 'PF=', cos(gs_i.phase-gs_u.phase)

        # // Take reference to the voltage phaser
        Ireal = sqrt(2.0*gs_i.accumSquaredData/N_SAMPLE) * cos(gs_i.phase-gs_u.phase)
        Iimag = sqrt(2.0*gs_i.accumSquaredData/N_SAMPLE) * sin(gs_i.phase-gs_u.phase)
        print "\tAmplitude from accumSquaredData->%g\n" % (sqrt(2.0*gs_u.accumSquaredData/N_SAMPLE))
        print "\tPhaser I:\tre=%g, im=%g\n" % (Ireal, Iimag)
        print "\tPhase Difference of I (version 2): %g\n" % (arctan2(Iimag,Ireal)*180/math.pi), 'PF=', cos(arctan2(Iimag,Ireal))

        plot(voltage)
        plot(current)

        show()

        # it works!
        # send_notification('Test email')


# find best individual
if __name__ == '__main__':
    swda = SwarmDataAnalyzer(run_integer=142)
    gen_best = swda.get_best_generation(popsize=30)

    with open('d:/Qr16_gen_best.txt', 'w') as f:
        f.write('\n'.join(','.join('%.16f'%(x) for x in y) for y in gen_best)) # convert 2d array to string            
    quit()

