# coding:u8
# parasolve_greedy_search_manager

import os
import sys
import femm
from time import time, sleep
import operator
import subprocess

def savetofile(id_solver, freq, stack_length):
    femm.mi_probdef(freq, 'millimeters', 'planar', 1e-8, # must < 1e-8
                    stack_length, 18, 1) # The acsolver parameter (default: 0) specifies which solver is to be used for AC problems: 0 for successive approximation, 1 for Newton.
    femm.mi_saveas(dir_femm_temp+'femm_temp_%d.fem'%(id_solver))

def remove_files(number_of_instantces, dir_femm_temp, suffix='.txt', id_solver_femm_found=None)
    for id_solver in range(number_of_instantces):
        fname = dir_femm_temp + "femm_temp_%d"%(id_solver) + suffix

        if id_solver == id_solver_femm_found:
            os.rename(fname, dir_femm_temp + "femm_found" + suffix)
            continue
            
        os.remove(fname)


number_of_instantces = int(sys.argv[1])
dir_femm_temp        = sys.argv[2]
stack_length         = float(sys.argv[3])

# print dir_femm_temp
# os.system('pause')
# quit()

#debug 
# number_of_instantces = 5
# dir_femm_temp        = "D:/OneDrive - UW-Madison/c/csv_opti/run#105/femm_temp/"
# stack_length         = 186.4005899999999940

femm.openfemm(True)
femm.opendocument(dir_femm_temp + 'femm_temp.fem')

# here, the only variable for an individual is frequency, so pop = list of frequencies
freq_begin = 1. # hz
freq_end   = freq_begin + number_of_instantces - 1. # 5 Hz

list_torque = []
list_slipfreq = []

while True:
    # freq_step can be negative!
    freq_step  = (freq_end - freq_begin) / (number_of_instantces-1)

    for id_solver in range(number_of_instantces):
        savetofile(id_solver, freq_begin + id_solver*freq_step, stack_length)

    procs = []
    # parasolve
    for i in range(number_of_instantces):
        proc = subprocess.Popen([sys.executable, 'parasolve_greedy_search.py', 
                                 str(i), '"'+dir_femm_temp+'"'], bufsize=-1)
        procs.append(proc)

    # 等了也是白等，一运行返回了
    for proc in procs:
        proc.wait() 

    list_solver_id = []
    count_sec = 0
    while True:
        sleep(1)
        count_sec += 1
        if count_sec > 120: # two min 
            raise Exception('It is highly likely that exception occurs during the solving of FEMM.')
        
        print '\nbegin waiting for eddy current solver...'
        for id_solver in range(number_of_instantces):

            if id_solver in list_solver_id:
                continue

            fname = dir_femm_temp + "femm_temp_%d.txt"%(id_solver)
            if os.path.exists(fname):
                with open(fname, 'r') as f:
                    data = f.readlines()
                    if data == []:
                        sleep(0.1)
                        data = f.readlines()
                        if data == []:
                            raise Exception('What takes you so long to write two float numbers?')
                    list_slipfreq.append( float(data[0][:-1]) )
                    list_torque.append(   float(data[1][:-1]) )
                list_solver_id.append(id_solver)
        if len(list_solver_id) >= number_of_instantces:
            break

    print list_solver_id
    print list_slipfreq
    print list_torque

    # find the max
    list_torque_copy = list_torque[::]
    index_1st, breakdown_torque_1st = max(enumerate(list_torque_copy), key=operator.itemgetter(1))
    breakdown_slipfreq_1st = list_slipfreq[index_1st]

    # find the 2nd max
    list_torque_copy[index_1st] = -999999
    index_2nd, breakdown_torque_2nd = max(enumerate(list_torque_copy), key=operator.itemgetter(1))
    breakdown_slipfreq_2nd = list_slipfreq[index_2nd]

    print 'max slip freq error=', 0.5*(breakdown_slipfreq_1st - breakdown_slipfreq_2nd), 'Hz'

    # find the two slip freq close enough then break.5
    if abs(breakdown_slipfreq_1st - breakdown_slipfreq_2nd) < 0.25: # Hz
        print 'Found it.', breakdown_slipfreq_1st, 'Hz', breakdown_torque_1st, 'Nm'
        remove_files(number_of_instantces, dir_femm_temp, suffix='.fem', id_solver_femm_found=list_solver_id[index_1st])
        remove_files(number_of_instantces, dir_femm_temp, suffix='.ans', id_solver_femm_found=list_solver_id[index_1st])

            # we have breakdown data, but we may like to know more about the corresponding rotor current as well as rotor slot size
            # however, why not leave this task to FEMM_Solver because you need geometry 
            # fname = dir_femm_temp + "femm_temp_%d.ans"%(list_solver_id[index_1st])
            # femm.mi_close()
            # femm.opendocument(fname)

            # # get stator slot area for copper loss calculation
            # femm.mo_groupselectblock(11)
            # Qs_stator_slot_area = femm.mo_blockintegral(5) # / self.im.Qs # unit: m^2 (verified by GUI operation)
            # femm.mo_clearblock()

            # # get rotor slot area for copper loss calculation
            # femm.mo_groupselectblock(101)
            # Qr_rotor_slot_area = femm.mo_blockintegral(5) # / self.im.Qr
            # femm.mo_clearblock()

            # femm.mo_close()

        with open(dir_femm_temp + 'femm_found.csv', 'w') as f:
            f.write('%g\n%g\n%g\n%g\n'%(breakdown_slipfreq_1st, breakdown_torque_1st, Qs_stator_slot_area, Qr_rotor_slot_area))
        break


    else:
        print 'not yet'
        remove_files(number_of_instantces, dir_femm_temp, suffix='.fem')
        remove_files(number_of_instantces, dir_femm_temp, suffix='.ans')

        # not found yet, try new frequencies.
        if breakdown_slipfreq_1st > breakdown_torque_2nd:
            freq_begin = breakdown_slipfreq_1st + 1.
            freq_end   = freq_begin + number_of_instantces - 1.

        elif breakdown_slipfreq_1st < breakdown_torque_2nd:
            freq_begin = breakdown_slipfreq_1st
            freq_end   = breakdown_slipfreq_2nd

            # freq_step can be negative!
            freq_step  = (freq_end - freq_begin) / (2 + number_of_instantces-1)
            freq_begin += freq_step
            freq_end   -= freq_step
        print 'try: freq_begin=%g, freq_end=%g.' % (freq_begin, freq_end)

remove_files(number_of_instantces, dir_femm_temp, suffix='.txt')
os.remove(dir_femm_temp + "femm_temp.fem")

femm.mi_close()
femm.closefemm()
os.system('pause')

