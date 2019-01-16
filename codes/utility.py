# coding:u8
import os
import logging
import datetime
import operator

def get_max_and_index(the_list):
    return max(enumerate(the_list), key=operator.itemgetter(1))

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
    def __init__(self, im, bounds):
        ''' Determine bounds for these parameters:
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

        self.show_norm(bounds, self.design_parameters_denorm)


    def show_denorm(self, bounds, design_parameters_norm):
        import numpy as np
        pop = design_parameters_norm
        min_b, max_b = np.asarray(bounds).T 
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        print '[De-normalized]:',
        print pop_denorm.tolist()
        
    def show_norm(self, bounds, design_parameters_denorm):
        import numpy as np
        min_b, max_b = np.asarray(bounds).T 
        diff = np.fabs(min_b - max_b)
        self.design_parameters_norm = (design_parameters_denorm - min_b)/diff #= pop
        # print type(self.design_parameters_norm)
        print '[Normalized]:',
        print self.design_parameters_norm.tolist()

        # pop = design_parameters_norm
        # min_b, max_b = np.asarray(bounds).T 
        # diff = np.fabs(min_b - max_b)
        # pop_denorm = min_b + pop * diff
        # print '[De-normalized:]---------------------------Are these two the same?'
        # print pop_denorm.tolist()
        # print design_parameters_denorm

def add_Pyrhonen_design_to_first_generation(sw, de_config_dict, logger):
    initial_design = Pyrhonen_design(sw.im, de_config_dict['bounds'])
    # print 'SWAP!'
    # print initial_design.design_parameters_norm.tolist()
    # print '\nInitial Population:'
    # for index in range(len(sw.init_pop)):
    #     # print sw.init_pop[index].tolist()
    #     print index,
    #     initial_design.show_denorm(de_config_dict['bounds'], sw.init_pop[index])
    # print sw.init_pop[0].tolist()
    sw.init_pop_denorm[0] = initial_design.design_parameters_denorm
    # print sw.init_pop[0].tolist()
    with open(sw.get_gen_file(0), 'w') as f:
        f.write('\n'.join(','.join('%.16f'%(x) for x in y) for y in sw.init_pop_denorm)) # convert 2d array to string
    logger.info('Initial design from Pyrhonen09 is added to the first generation of pop.')



def send_notification(text='Hello'):
    subject = 'AUTO BLIM OPTIMIZATION NOTIFICATION'
    content = 'Subject: %s\n\n%s' % (subject, text)
    from smtplib import SMTP
    mail = SMTP('smtp.gmail.com', 587)
    mail.ehlo()
    mail.starttls()
    with open('temp.txt', 'r') as f:
        pswd = f.read()
    mail.login('horyontour@gmail.com', pswd)
    mail.sendmail('horyontour@gmail.com', 'jiahao.chen@wisc.edu', content) 
    mail.close()
    print "Notificaiont sent."



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

    def goertzel_offline(gs, targetFreq, numSamples, samplingRate, data_list):
        # gs is equivalent to self
        try:
            len(data_list) # = numSamples
        except:
            raise Exception('The input data_list must be array or list.') 

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




if __name__ == '__main__':
    # from pylab import *

    for generation in range(5):
        print '----------gen#%d'%(generation)
        generation_file_path = r'D:\OneDrive - UW-Madison\c\pop\run#107/' + 'gen#%04d.txt'%(generation)
        print generation_file_path
        if os.path.exists( generation_file_path ):
            with open(generation_file_path, 'r') as f:
                for el in f.readlines():
                    print el[:-1]

    # read voltage and current to see the power factor!
    # read voltage and current to see the power factor!
    # read voltage and current to see the power factor!
    
    # 绘制损耗图形。
    # 绘制损耗图形。
    # 绘制损耗图形。

    if False:
        gs_u = Goertzel_Data_Struct("Goertzel Struct for Voltage\n")
        gs_i = Goertzel_Data_Struct("Goertzel Struct for Current\n")

        phase = arccos(0.6) # PF=0.6
        targetFreq = 1000.
        TS = 1.5625e-5

        if False:
            time = arange(0./targetFreq, 100.0/targetFreq, TS)
            voltage = 3*sin(targetFreq*2*pi*time + 30/180.*pi)
            current = 2*sin(targetFreq*2*pi*time + 30/180.*pi - phase)
        else:
            time = arange(0.5/targetFreq, 1.0/targetFreq, TS)
            voltage = 3*sin(targetFreq*2*pi*time + 30/180.*pi)
            current = 2*sin(targetFreq*2*pi*time + 30/180.*pi - phase)

        N_SAMPLE = len(voltage)
        noise = ( 2*rand(N_SAMPLE) - 1 ) * 0.233
        voltage += noise

        # plot(voltage+0.5)
        # plot(current+0.5)

        # Check for resolution
        end_time = N_SAMPLE * TS
        resolution = 1./end_time
        print resolution, targetFreq
        if resolution > targetFreq:

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

        print gs_u.goertzel_offline(targetFreq, N_SAMPLE, 1./TS, voltage)
        print gs_i.goertzel_offline(targetFreq, N_SAMPLE, 1./TS, current)

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


