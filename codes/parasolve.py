# coding:u8
# https://stackoverflow.com/questions/19156467/run-multiple-instances-of-python-script-simultaneously
# https://docs.python.org/2/library/subprocess.html#subprocess.Popen

import os
import sys
import femm
from time import time

number_solver = int(sys.argv[1])
number_of_instances = int(sys.argv[2])
dir_run = sys.argv[3]
print 'ParaSolve!'

fem_file_list = os.listdir(dir_run)
fem_file_list = [f for f in fem_file_list if '.fem' in f]

femm.openfemm(True) # bHide
femm.smartmesh(False) # this is essential to reduce elements counts from >50000 to ~20000.

for i in range(number_solver, len(fem_file_list), number_of_instances):

    output_file_name = dir_run + fem_file_list[i][:-4]

    if not os.path.exists(output_file_name + '.ans'):
        tic = time()
        femm.opendocument(output_file_name + '.fem')
        try:
            # femm.mi_createmesh() # [useless] 
            femm.mi_analyze(1) # None for inherited. 1 for a minimized window,
        except:
            print 'Is it: Material properties have not been defined for all regions? Check the following file:'
            print i, fem_file_list[i]
        femm.mi_close()
        # femm.mi_loadsolution()
        toc = time()
    print i, fem_file_list[i], toc - tic, 's'
femm.closefemm()



# True
# slip_freq_breakdown_torque: 3.0 Hz
# ParaSolve!
# 2 36-0Hz  30.ans
# 6 36-0Hz  90.ans
# 10 36-0Hz 180.fem
# Traceback (most recent call last):
#   File "parasolve.py", line 29, in <module>
#     Fx = femm.mo_blockintegral(18) #-- 18 x (or r) part of steady-state weighted stress tensor force
#   File "D:\Users\horyc\Anaconda2\lib\site-packages\femm\__init__.py", line 1722, in mo_blockintegral
#     return callfemm('mo_blockintegral(' + num(ptype) + ')' );
#   File "D:\Users\horyc\Anaconda2\lib\site-packages\femm\__init__.py", line 25, in callfemm
#     x = HandleToFEMM.mlab2femm(myString).replace("[ ","[").replace(" ]","]").replace(" ",",").replace("I","1j");
#   File "<COMObject femm.ActiveFEMM>", line 3, in mlab2femm
# pywintypes.com_error: (-2147023170, 'The remote procedure call failed.', None, None)
