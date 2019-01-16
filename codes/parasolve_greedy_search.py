# coding:u8

import os
import sys
import femm
from time import time
# from itertools import izip
# from numpy import exp, pi # sqrt
# from numpy import savetxt, c_

def write_torque_data_to_file(handle_torque):
    # call this after mi_analyze
    femm.mi_loadsolution()

    # Physical Amount on the Rotor
    femm.mo_groupselectblock(100) # rotor iron
    femm.mo_groupselectblock(101) # rotor bars
    # Fx = femm.mo_blockintegral(18) #-- 18 x (or r) part of steady-state weighted stress tensor force
    # Fy = femm.mo_blockintegral(19) #--19 y (or z) part of steady-state weighted stress tensor force
    torque = femm.mo_blockintegral(22) #-- 22 = Steady-state weighted stress tensor torque
    freq = femm.mo_getprobleminfo()[1]
    femm.mo_clearblock()

    # write results to a data file (write to partial files to avoid compete between parallel instances)
    handle_torque.write("%g\n%g\n" % ( freq, torque))
    femm.mo_close()

id_solver = int(sys.argv[1])
dir_femm_temp = sys.argv[2]
print 'ParaSolve', id_solver

handle_torque = open(dir_femm_temp + "femm_temp_%d.txt"%(id_solver), 'w')

fem_file_list = os.listdir(dir_femm_temp)
fem_file_list = [f for f in fem_file_list if '.fem' in f]

femm.openfemm(True) # bHide
# this is essential to reduce elements counts from >50000 to ~20000.
femm.callfemm_noeval('smartmesh(0)')
print 'smartmesh is off'

tic = time()
fem_file_path = dir_femm_temp + 'temm_temp_%d.fem'%(id_solver)
femm.opendocument(fem_file_path)
try:
    femm.mi_analyze(1) # None for inherited. 1 for a minimized window,
    write_torque_data_to_file(handle_torque)
except Exception as error:
    print error.args
    raise error
femm.mi_close()
toc = time()
print i, fem_file_list[i], toc - tic, 's'
femm.closefemm()
handle_torque.close()

# os.remove(fem_file_path)
# os.remove(fem_file_path[:-4]+'.ans')
