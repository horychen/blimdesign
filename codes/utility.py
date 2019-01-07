
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

        # create a file handler
        handler=logging.FileHandler(dir_codes + prefix + now.strftime("%Y-%m-%d") +'.log')
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
    handler = logging.FileHandler(dir_codes + r'opti_script.log')
    handler.setLevel(logging.DEBUG)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    return logger

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
