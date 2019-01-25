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
    def __init__(self, dir_run=None, run_integer=None):
        if run_integer is not None:
            dir_run = r'D:\OneDrive - UW-Madison\c\pop\run#%d/'%(run_integer)

        with open(dir_run+'swarm_data.txt', 'r') as f:
            self.buf = f.readlines()[1:]
            self.buf_length = len(self.buf)

        self.number_of_designs = self.buf_length / 21

        logger = logging.getLogger(__name__)
        logger.debug('self.buf_length %% 21 = %d, / 21 = %d' % (self.buf_length % 21, self.number_of_designs))

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

    def lsit_cost_function(self):
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

    def get_best_generation(self, popsize=50, generator=None):

        if generator is None:
            generator = swda.design_parameters_generator()

        cost = swda.lsit_cost_function()
        indices, items = min_indices(cost, popsize)
        print indices, items

        gen_best = []
        for index, design in enumerate(generator):
            if index in indices:
                gen_best.append(design)
        return gen_best

    def get_list_objective_function(self):
        for i in range(self.number_of_designs):
            individual = self.buf[i*21:(1+i)*21]
            yield     [float(el) for el in individual[3].split(',')] #\
                    # + [float(el) for el in individual[17][16:].split(',')] \
                    # + [float(el) for el in individual[18][16:].split(',')]

    def get_certain_objective_function(self, which):
        for i in range(self.number_of_designs):
            individual = self.buf[i*21:(1+i)*21]
            yield [float(el) for el in individual[3].split(',')][which]

# run 115
de_config_dict = {  'bounds':     [ [ 4.9,   9],#--# stator_tooth_width_b_ds
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
# de_config_dict = {  'bounds':     [ [   4, 7.2],#--# stator_tooth_width_b_ds
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
bounds = de_config_dict['bounds']
dimensions = len(bounds)
min_b, max_b = np.asarray(bounds).T 
diff = np.fabs(min_b - max_b)
# pop_denorm = min_b + pop * diff
# pop[j] = (pop_denorm[j] - min_b) / diff

import itertools
if __name__ == '__main__':
    from pylab import *
    plt.rcParams["font.family"] = "Times New Roman"
    # import seaborn as sns

    # style.use('seaborn-poster') #sets the size of the charts
    # style.use('grayscale')
    # ax = sns.barplot(y= "Deaths", x = "Causes", data = deaths_pd, palette=("Blues_d"))
    # sns.set_context("poster")

    def autolabel(rects, xpos='center', bias=0.0):
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

    def fobj_scalar(torque_average, ss_avg_force_magnitude, normalized_torque_ripple, normalized_force_error_magnitude, force_error_angle, total_loss, 
                    weights=[ 1, 0.1,   1, 0.1, 0.1,   0 ]):

        list_cost = [   30e3 / ( torque_average/rotor_volume ),
                        normalized_torque_ripple         *  20, #       / 0.05 * 0.1
                        1.0 / ( ss_avg_force_magnitude/rotor_weight ),
                        normalized_force_error_magnitude *  20, #       / 0.05 * 0.1
                        force_error_angle                * 0.2, # [deg] /5 deg * 0.1 is reported to be the base line (Yegu Kang) # force_error_angle is not consistent with Yegu Kang 2018-060-case of TFE
                        total_loss                       / 2500. ] #10 / efficiency**2,
        cost_function = np.dot(array(list_cost), array(weights))
        return cost_function

    def fobj_list(l_torque_average, l_ss_avg_force_magnitude, l_normalized_torque_ripple, l_normalized_force_error_magnitude, l_force_error_angle, l_total_loss,
                    weights=[ 1, 0.1,   1, 0.1, 0.1,   0 ]):
    
        l_cost_function = []
        for torque_average, ss_avg_force_magnitude, normalized_torque_ripple, normalized_force_error_magnitude, force_error_angle, total_loss in zip(l_torque_average, l_ss_avg_force_magnitude, l_normalized_torque_ripple, l_normalized_force_error_magnitude, l_force_error_angle, l_total_loss):
            list_cost = [   30e3 / ( torque_average/rotor_volume ),
                            normalized_torque_ripple         *  20,
                            1.0 / ( ss_avg_force_magnitude/rotor_weight ),
                            normalized_force_error_magnitude *  20,
                            force_error_angle                * 0.2,
                            total_loss                     / 2500. ]
            cost_function = np.dot(array(list_cost), array(weights))
            l_cost_function.append(cost_function)
        return array(l_cost_function)


    # def fobj2(l_torque_average, l_ss_avg_force_magnitude, l_normalized_torque_ripple, l_normalized_force_error_magnitude, l_force_error_angle, l_total_loss):
    
    #     weights    = [ 1, 1,   1, 1, 1,   0 ]

    #     if type(l_torque_average) == type(0.0): # not a list
    #         list_weighted_cost = [  30e3 / ( l_torque_average/rotor_volume ),
    #                                 1.0 / ( l_ss_avg_force_magnitude/rotor_weight ),
    #                                 l_normalized_torque_ripple         *  20, #       / 0.05 * 0.1
    #                                 l_normalized_force_error_magnitude *  20, #       / 0.05 * 0.1
    #                                 l_force_error_angle * 0.2          *  10, # [deg] /5 deg * 0.1 is reported to be the base line (Yegu Kang) # force_error_angle is not consistent with Yegu Kang 2018-060-case of TFE
    #                                 2*l_total_loss/2500.*0 ] #10 / efficiency**2,
    #         cost_function = sum(list_weighted_cost)
    #         return cost_function
    #     else:
    #         l_cost_function = []
    #         for torque_average, ss_avg_force_magnitude, normalized_torque_ripple, normalized_force_error_magnitude, force_error_angle, total_loss in zip(l_torque_average, l_ss_avg_force_magnitude, l_normalized_torque_ripple, l_normalized_force_error_magnitude, l_force_error_angle, l_total_loss):
    #             list_weighted_cost = [  30e3 / ( torque_average/rotor_volume ),
    #                                     1.0 / ( ss_avg_force_magnitude/rotor_weight ),
    #                                     normalized_torque_ripple         *  20, #       / 0.05 * 0.1
    #                                     normalized_force_error_magnitude *  20, #       / 0.05 * 0.1
    #                                     force_error_angle * 0.2          *  10, # [deg] /5 deg * 0.1 is reported to be the base line (Yegu Kang) # force_error_angle is not consistent with Yegu Kang 2018-060-case of TFE
    #                                     2*total_loss/2500.*0 ] #10 / efficiency**2,
    #             # print list_weighted_cost
    #             cost_function = sum(list_weighted_cost)
    #             # print cost_function
    #             l_cost_function.append(cost_function)
    #         # print l_cost_function
    #         return array(l_cost_function)

    # fobj1 = fobj2

    # Basic information
    required_torque = 15.9154943092 #Nm

    Radius_OuterRotor = 47.092753
    stack_length = 93.200295
    Omega =  3132.95327379
    rotor_volume = pi*(Radius_OuterRotor*1e-3)**2 * (stack_length*1e-3)
    rotor_weight = 9.8 * rotor_volume * 8050 # steel 8,050 kg/m3. Copper/Density 8.96 g/cm³. gravity: 9.8 N/kg
    print 'rotor_volume=', rotor_volume, 'm^3'
    print 'rotor_weight=', rotor_weight, 'N'


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

        # y = l[:len(l)/2] # 115
        y = l # 116
        print 'len(y)=', len(y)

        if i == 9: # replace P_Fe with P_Fe,Cu
            l_femm_stator_copper = array(list(swda.get_certain_objective_function(12)))
            l_femm_rotor_copper  = array(list(swda.get_certain_objective_function(13)))
            y = array(l) + l_femm_stator_copper + l_femm_rotor_copper 
            # print l, len(l)
            # print y, len(y)
            # quit()

        data_max.append([])
        data_min.append([])
        for j in range(len(y)/number_of_variant): # iterate design parameters
            y_vs_design_parameter = y[j*number_of_variant:(j+1)*number_of_variant]

            # if j == 6:
            ax_list[ind].plot(y_vs_design_parameter, label=str(j)+' '+param_list[j], alpha=0.5)
            print '\t', j, param_list[j], '\t\t', max(y_vs_design_parameter) - min(y_vs_design_parameter)

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

    O2_ref = fobj_scalar(19.1197, 96.9263, 0.0864712, 0.104915, 6.53137, (1817.22+216.216+224.706), weights=[ 1, 1.0,   1, 1.0, 1.0,   0 ])
    O1_ref = fobj_scalar(19.1197, 96.9263, 0.0864712, 0.104915, 6.53137, (1817.22+216.216+224.706), weights=[ 1, 0.1,   1, 0.1, 0.1,   0 ])

    print  'Objective function 1'
    O1 = fobj_list( list(swda.get_certain_objective_function(2)), 
                    list(swda.get_certain_objective_function(4)), 
                    list(swda.get_certain_objective_function(3)), 
                    list(swda.get_certain_objective_function(5)), 
                    list(swda.get_certain_objective_function(6)), 
                    array(list(swda.get_certain_objective_function(9))) + array(list(swda.get_certain_objective_function(12))) + array(list(swda.get_certain_objective_function(13))),
                    weights=[ 1, 0.1,   1, 0.1, 0.1,   0 ] )
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
                    weights=[ 1, 1.0,   1, 1.0, 1.0,   0 ] )
    O2_max = []
    O2_min = []
    O2_ax  = figure().gca()
    for j in range(len(O2)/number_of_variant): # iterate design parameters
        O2_vs_design_parameter = O2[j*number_of_variant:(j+1)*number_of_variant]

        O2_ax.plot(O2_vs_design_parameter, label=str(j)+' '+param_list[j], alpha=0.5)
        print '\t', j, param_list[j], '\t\t', max(O2_vs_design_parameter) - min(O2_vs_design_parameter), '\t\t',
        print [ind for ind, el in enumerate(O2_vs_design_parameter) if el < O2_ref] #'<- to derive new bounds.'

        O2_max.append(max(O2_vs_design_parameter))
        O2_min.append(min(O2_vs_design_parameter))            
    O2_ax.legend()
    O2_ax.grid()
    O2_ax.set_ylabel('O2 [1]')
    O2_ax.set_xlabel('Count of design variants')

    # Reference candidate design
    ref = zeros(8)
        # ref[0] = 0.635489                                   # PF
        # ref[1] = 0.963698                                   # eta
        # ref[1] = efficiency_at_50kW(1817.22+216.216+224.706)# eta@50kW
    O2_ax.plot(range(-1, 22), O2_ref*np.ones(23), 'k--')
    O1_ax.plot(range(-1, 22), O1_ref*np.ones(23), 'k--')
    ref[0] = O2_ref    / 8
    ref[1] = O1_ref    / 3
    ref[2] = 19.1197   / required_torque                # 100%
    ref[3] = 0.0864712 / 0.1                            # 100%
    ref[4] = 96.9263   / rotor_weight                   # 100% = FRW
    ref[5] = 0.104915  / 0.2                            # 100%
    ref[6] = 6.53137   / 10                             # deg
    ref[7] = (1817.22+216.216+224.706) / 2500           # W

    # Maximum
    data_max = array(data_max)
    O1_max   = array(O1_max)
    O2_max   = array(O2_max)
        # data_max[0] = (data_max[0])                   # PF
        # data_max[1] = (data_max[1])                   # eta
        # data_max[1] = efficiency_at_50kW(data_max[7]) # eta@50kW # should use data_min[7] because less loss, higher efficiency
    data_max[0] = O2_max / 8
    data_max[1] = O1_max / 3
    data_max[2] = (data_max[2])/ required_torque  # 100%
    data_max[3] = (data_max[3])/ 0.1              # 100%
    data_max[4] = (data_max[4])/ rotor_weight     # 100% = FRW
    data_max[5] = (data_max[5])/ 0.2              # 100%
    data_max[6] = (data_max[6])/ 10               # deg
    data_max[7] = (data_max[7])/ 2500             # W
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
    data_min[0] = O2_min / 8
    data_min[1] = O1_min / 3
    data_min[2] = (data_min[2]) / required_torque  # 100%
    data_min[3] = (data_min[3]) / 0.1              # 100%
    data_min[4] = (data_min[4]) / rotor_weight     # 100% = FRW
    data_min[5] = (data_min[5]) / 0.2              # 100%
    data_min[6] = (data_min[6]) / 10               # deg
    data_min[7] = (data_min[7]) / 2500             # W
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
    autolabel(rects1, bias=-0.10)
    autolabel(rects2, bias=-0.10)
    autolabel(rects3, bias=-0.10)
    autolabel(rects4, bias=-0.10)
    autolabel(rects5, bias=-0.10)
    autolabel(rects6, bias=-0.10)
    autolabel(rects7, bias=-0.10)
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
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)
    autolabel(rects7)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Normalized Objective Functions')
    ax.set_xticks(count)
    # ax.set_xticklabels(('Power Factor [100%]', r'$\eta$@$T_{em}$ [100%]', r'$T_{em}$ [15.9 N]', r'$T_{rip}$ [10%]', r'$|F|$ [51.2 N]', r'    $E_m$ [20%]', r'      $E_a$ [10 deg]', r'$P_{\rm Cu,Fe}$ [2.5 kW]')))
    # ax.set_xticklabels(('Power Factor [100%]', r'$O_1$ [3]', r'$T_{em}$ [15.9 N]', r'$T_{rip}$ [10%]', r'$|F|$ [51.2 N]', r'    $E_m$ [20%]', r'      $E_a$ [10 deg]', r'$P_{\rm Cu,Fe}$ [2.5 kW]'))
    ax.set_xticklabels((r'$O_2$ [8]', r'$O_1$ [3]', r'$T_{em}$ [15.9 Nm]', r'$T_{rip}$ [10%]', r'$|F|$ [51.2 N]', r'    $E_m$ [20%]', r'      $E_a$ [10 deg]', r'$P_{\rm Cu,Fe}$ [2.5 kW]'))
    ax.grid()
    fig.tight_layout()

    show()

    quit()
























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

    cost = swda.lsit_cost_function()
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

