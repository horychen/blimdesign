# blimdesign

Requirements:

- anaconda3

- pyx, pyfemm and others (if any) can be installed via pip

- pygmo:
    > conda config --add channels conda-forge
    > conda install pygmo

- latex (if you want pdf report for motor design)


# TODO

- The current circuit excitation will give wrong terminal voltage results (the electromagnetic performance is correct because the current is correct, though).




**[Under development]**

~~Steps:~~

~~1. pyrhonen_blim_design.py~~

~~At design phase 2, specify tangential stress and machine constant.~~
~~The former is used to determine motor size, and the latter is for determining linear current density A according to the air gap B.~~

~~Choose a zQs. It will affect your air gap B.~~

~~At design phase 12, choose the correct magnetic material you want to use: Arnon5 or M19Gauge29~~

~~At design phase 11, Check the current density specified on the design, especially the rotor current density.~~
~~The current density should be chosen according to your cooling method.~~
~~A hot rotor should be avoided.~~

