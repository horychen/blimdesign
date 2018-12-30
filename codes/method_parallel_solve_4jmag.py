import sys
import subprocess

dir_run = sys.argv[1]
number_of_instantces = int(sys.argv[2])
# print number_of_instantces

procs = []
for i in range(number_of_instantces):
    # proc = subprocess.Popen([sys.executable, 'parasolve.py', '{}in.csv'.format(i), '{}out.csv'.format(i)], bufsize=-1)
    proc = subprocess.Popen([sys.executable, 'parasolve.py', str(i), str(number_of_instantces), dir_run], bufsize=-1)
    procs.append(proc)

for proc in procs:
    proc.wait()

