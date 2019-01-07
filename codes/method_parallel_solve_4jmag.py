import sys
import subprocess

dir_run = sys.argv[1]
number_of_instantces = int(sys.argv[2])
str_deg_per_step = sys.argv[3]

procs = []
for i in range(number_of_instantces):
    proc = subprocess.Popen([sys.executable, 'parasolve.py', 
                             str_deg_per_step, str(i), str(number_of_instantces), dir_run], bufsize=-1)
    procs.append(proc)

for proc in procs:
    proc.wait()

