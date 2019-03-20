#coding:u8
import numpy as np
from csv import reader as csv_reader

# quit()

number_current_generation = 0
run_folder = r'run#36/'
# dir_csv_output_folder = u"D:/Users/horyc/OneDrive - UW-Madison/csv_opti/" # 2018
dir_csv_output_folder = u"D:/OneDrive - UW-Madison/c_obsolete/csv_opti/" # 2019 
dir_csv_output_folder = dir_csv_output_folder + run_folder

dir_project_files = 'D:/JMAG_Files/'
project_name = run_folder[:-1]+'gen#%04d' % (number_current_generation)
expected_project_file = dir_project_files + "%s.jproj"%(project_name)
model_name_prefix = 'BLIM_PS'

def whole_row_reader(reader):
    for row in reader:
        yield row[:]

def read_csv_results(prefix, data_type):
    file_location = prefix + data_type
    print file_location

    l_time_or_angle = [] # or other x-axis variable such as time and angle.
    l_data    = []
    slip_freq = None
    speed = None
    l_current = []
    l_force = []

    if 'Tran' in file_location:
        with open(file_location, 'r') as f: 
            read_iterator = csv_reader(f, skipinitialspace=True)
            for ind, row in enumerate(whole_row_reader(read_iterator)):                
                # print row
                if ind == 4:
                    print row
                    freq = [float(el) for el in row[1:] if el != ''] #  
                elif ind == 5:
                    slip = [float(el) for el in row[1:] if el != '']
                else:
                    try: 
                        float(row[0])
                    except:
                        continue
                    else:
                        l_time_or_angle.append(float(row[0]))
                        l_data.append([float(el) for el in row[1:]])
            slip_freq = np.array(freq)*np.array(slip)
            speed = np.array(freq) * (1 - np.array(slip)) * 30

        file_location = prefix + '_circuit_current.csv'
        with open(file_location, 'r') as f: 
            read_iterator = csv_reader(f, skipinitialspace=True)
            for ind, row in enumerate(whole_row_reader(read_iterator)):
                try: 
                    float(row[0])
                except:
                    continue
                else:
                    l_current.append([float(row[7+i*DriveW_poles]) for i in range(int(imID)/DriveW_poles) ])

        file_location = prefix + '_force.csv'
        with open(file_location, 'r') as f: 
            read_iterator = csv_reader(f, skipinitialspace=True)
            for ind, row in enumerate(whole_row_reader(read_iterator)):
                try: 
                    float(row[0])
                except:
                    continue
                else:
                    l_force.append([float(el) for el in row[1:]])

    elif 'FFVRC' in file_location:
        with open(file_location, 'r') as f: 
            read_iterator = csv_reader(f, skipinitialspace=True)
            for ind, row in enumerate(whole_row_reader(read_iterator)):
                if ind>=5:
                    if slip_freq == None:
                        slip_freq = float(row[0])
                    l_data.append([float(el) for el in row[1:]])

            file_location = prefix + '_total_rotational_displacement.csv'
            with open(file_location, 'r') as f: 
                read_iterator = csv_reader(f, skipinitialspace=True)

                for ind, row in enumerate(whole_row_reader(read_iterator)):
                    if ind>=5:
                        l_time_or_angle.append([float(el) for el in row[1:]])

            file_location = prefix + '_circuit_current.csv'
            with open(file_location, 'r') as f: 
                read_iterator = csv_reader(f, skipinitialspace=True)
                for ind, row in enumerate(whole_row_reader(read_iterator)):
                    if ind>=5:
                        l_current.append([float(el) for el in row[13:]])

            file_location = prefix + '_force.csv'
            with open(file_location, 'r') as f: 
                read_iterator = csv_reader(f, skipinitialspace=True)
                for ind, row in enumerate(whole_row_reader(read_iterator)):
                    if ind>=5:
                        l_force.append([float(el) for el in row[1:]])

    return l_time_or_angle, l_data, slip_freq, speed, l_current, l_force

DriveW_poles = 4
DriveW_Freq = 500.0
Qr = 36.


for individual_index in range(1):
# for individual_index in range(20, 44):

    #1. compare the results of FFVRC and Tran500Hz
    imID = str(36) #str(int(Qr))
    ID = imID + '-' + str(number_current_generation) + '-' + str(individual_index) # the ID is str
    model_name = u"%s_ID%s" % (model_name_prefix, ID)
    original_study_name = model_name + u"Freq_#%s" % (ID)
    tran_study_name = u"Tran_%s-%dHz"%(ID, int(DriveW_Freq))
    ecrot_study_name = u"Freq_#%s" % (ID) + u"-FFVRC" # FixFreqVaryRotCon

    from pylab import * #plot, show, legend, grid, figure, pi, ylim



    # Plot figure with subplots of different sizes
    # fig = plt.figure(99, dpi=150, figsize=(16, 8), facecolor='w', edgecolor='k')
    # set up subplot grid
    # plt.GridSpec(2,3)
    # plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=1)
    # plt.subplot2grid((2,3), (1,0), colspan=1, rowspan=1)
    # plt.subplot2grid((2,3), (0,1), colspan=2, rowspan=1)
    # plt.subplot2grid((2,3), (1,1), colspan=2, rowspan=1)
    # plt.xlim(0,180)
    # plt.xticks(np.arange(0, 181, 30))

    # import Image
    # fig = plt.figure()
    # fig.figimage(Image.open('.png'), 0, 0)










    # EC-Rotate-24cases from Severson02
    f_name1 = r'D:\JMAG_Files\CSV\Severson02_EC_Rotate_PS_24cases-ForConAbs.csv'
    freq_list = []
    ForConAbs_list = []
    with open(f_name1, 'r') as f:
        read_iterator = csv_reader(f, skipinitialspace=True)
        count = 0
        for row in whole_row_reader(read_iterator):
            if count == 0:
                count += 1
                continue
            freq_list.append(float(row[0]))
            ForConAbs_list.append([float(el) for el in row[1:]])
    fig = figure(26, dpi=150, figsize=(16/2, 8/2), facecolor='w', edgecolor='k')
    ax = fig.gca()
    # ax = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=1)
    # plt.subplot2grid((2,3), (1,0), colspan=1, rowspan=1)
    # plt.subplot2grid((2,3), (0,1), colspan=2, rowspan=1)
    # plt.subplot2grid((2,3), (1,1), colspan=2, rowspan=1)
    lines = []
    for i in range(len(ForConAbs_list[0])):
        y = [el[i] for el in ForConAbs_list]
        lines += ax.plot(freq_list, y) # label=str(i*360./Qr/24.)

    ax.grid()
    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    
    ax.set_ylabel('Force [N]')
    ax.set_xlabel('Frequency [Hz]')
    # 高级图例
    if True:
        # specify the lines and labels of the first legend
        ax.legend(lines[:5], ['%.2f deg' % (i*360./Qr/24.) for i in range(5)],
                    frameon=False, loc='upper left')
                    # bbox_to_anchor=(1.05, 1), borderaxespad=0.) # loc=2
        # Create the second legend and add the artist manually.
        from matplotlib.legend import Legend                                              # https://jakevdp.github.io/PythonDataScienceHandbook/04.06-customizing-legends.html
        leg = Legend(ax, lines[5:], ['%.2f deg' % (i*360./Qr/24.) for i in range(5,16)],
                     loc='lower center', frameon=False)
        ax.add_artist(leg);

        # Create the third legend and add the artist manually.
        leg = Legend(ax, lines[16:], ['%.2f deg' % (i*360./Qr/24.) for i in range(16,24)],
                    frameon=False, loc='lower right')
                    # bbox_to_anchor=(0.8, 1), borderaxespad=0.)
        ax.add_artist(leg);
        # ax.text()
    fig.tight_layout()
    fig_name = 'EC-RR-24cases-force.png'
    fig.savefig(fig_name, dpi=150)



    f_name2 = r'D:\JMAG_Files\CSV\Severson02_EC_Rotate_PS_24cases-TorCon.csv'
    freq_list = []
    TorCon_list = []
    with open(f_name2, 'r') as f:
        read_iterator = csv_reader(f, skipinitialspace=True)
        count = 0
        for row in whole_row_reader(read_iterator):
            if count == 0:
                count += 1
                continue
            freq_list.append(float(row[0]))
            TorCon_list.append([float(el) for el in row[1:]])
    fig = figure(27, dpi=150, figsize=(16/2, 8/2), facecolor='w', edgecolor='k')
    ax = fig.gca()
    # plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=1)
    # ax = plt.subplot2grid((2,3), (1,0), colspan=1, rowspan=1)
    # plt.subplot2grid((2,3), (0,1), colspan=2, rowspan=1)
    # plt.subplot2grid((2,3), (1,1), colspan=2, rowspan=1)
    lines = []
    for i in range(len(TorCon_list[0])):
        y = [el[i] for el in TorCon_list]
        lines += ax.plot(freq_list, y) # label=str(i*360./Qr/24.)
    ax.grid()
    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_ylabel('Torque [Nm]')
    ax.set_xlabel('Frequency [Hz]')
    # 高级图例
    if True:
        # specify the lines and labels of the first legend
        ax.legend(lines[:8], ['%.2f deg' % (i*360./Qr/24.) for i in range(8)],
                    loc='lower left', frameon=False)

        # Create the second legend and add the artist manually.
        leg = Legend(ax, lines[8:], ['%.2f deg' % (i*360./Qr/24.) for i in range(8,14)],
                     loc='lower center', frameon=False)
        ax.add_artist(leg);

        # Create the third legend and add the artist manually.
        leg = Legend(ax, lines[14:], ['%.2f deg' % (i*360./Qr/24.) for i in range(14,24)],
                     loc='upper right', frameon=False)
        ax.add_artist(leg);
    fig.tight_layout()
    fig_name = 'EC-RR-24cases-torque.png'
    fig.savefig(fig_name, dpi=150)












    # Transient FEA-ThinStepRun
    slip_freq = 3

    # f_name1 = r'D:\JMAG_Files\CSV\run#36-EC_Rotate_PS-Tran-TinyStepRun-ForConAbs.csv'
    f_name1 = r'D:\JMAG_Files\Qr36-ForConAbs.csv'
    time_list_case123 = []
    time_list_case4 = []
    ForConAbs_list_case1 = []
    ForConAbs_list_case2 = []
    ForConAbs_list_case3 = []
    ForConAbs_list_case4 = []
    with open(f_name1, 'r') as f:
        read_iterator = csv_reader(f, skipinitialspace=True)
        count = 0
        for row in whole_row_reader(read_iterator):
            if count == 0:
                count += 1
                continue
            try: 
                float(row[0])
            except:
                time_list_case4.append(float(row[5]))
                ForConAbs_list_case4.append(float(row[6]))

            else:
                time_list_case123.append(float(row[0]))
                ForConAbs_list_case1.append(float(row[1]))
                ForConAbs_list_case2.append(float(row[2]))
                ForConAbs_list_case3.append(float(row[3]))

                time_list_case4.append(float(row[5]))
                ForConAbs_list_case4.append(float(row[6]))

    fig25 = figure(125, dpi=150, figsize=(16/2, 8/2), facecolor='w', edgecolor='k')
    ax = fig25.gca()
    ax.set_ylabel('Force [N]')
    ax.set_xlabel('rotor position [deg]')    
    rpm = (DriveW_Freq-slip_freq)*30
    omega = rpm /60.*2*pi # mechanical speed: rpm to rad/s (no pole pair number neeeded)

    theta = np.array(time_list_case4)*omega
    angle = theta / pi * 180
    ax.plot(angle, ForConAbs_list_case4, label='TranFEA: %g Hz, %g rpm'%(DriveW_Freq, rpm))

    theta = np.array(time_list_case123)*omega
    angle = theta / pi * 180
    ax.plot(angle, ForConAbs_list_case1, zorder=99, label='TranFEAwi2TSS: %g Hz, %g rpm'%(DriveW_Freq, rpm))
        # ax.plot(angle, ForConAbs_list_case2, label='TranFEAwi2TSS: %g Hz, %g rpm'%(DriveW_Freq, rpm)) # wrong speed!
        # ax.plot(angle, ForConAbs_list_case3, label='TranFEAwi2TSS: %g Hz, %g rpm'%(DriveW_Freq, rpm))

    ax.legend()
    ax.grid()
    plt.xlim(0,angle[-1])


    # f_name2 = r'D:\JMAG_Files\CSV\run#36-EC_Rotate_PS-Tran-TinyStepRun-ForConAbs.csv'
    f_name2 = r'D:\JMAG_Files\Qr36-TorCon.csv'
    time_list_case123 = []
    time_list_case4 = []
    TorCon_list_case1 = []
    TorCon_list_case2 = []
    TorCon_list_case3 = []
    TorCon_list_case4 = []
    with open(f_name2, 'r') as f:
        read_iterator = csv_reader(f, skipinitialspace=True)
        count = 0
        for row in whole_row_reader(read_iterator):
            if count == 0:
                count += 1
                continue
            try: 
                float(row[0])
            except:
                time_list_case4.append(float(row[5]))
                TorCon_list_case4.append(float(row[6]))
            else:
                time_list_case123.append(float(row[0]))
                TorCon_list_case1.append(float(row[1]))
                TorCon_list_case2.append(float(row[2]))
                TorCon_list_case3.append(float(row[3]))

                time_list_case4.append(float(row[5]))
                TorCon_list_case4.append(float(row[6]))


    fig2 = figure(102, dpi=150, figsize=(16/2, 8/2), facecolor='w', edgecolor='k')
    ax = fig2.gca()
    ax.set_ylabel('Torquee [Nm]')
    ax.set_xlabel('rotor position [deg]')        
    speed = (DriveW_Freq-slip_freq)*30
    omega = speed /60.*2*pi # mechanical speed: rpm to rad/s (no pole pair number neeeded)

    theta = np.array(time_list_case4)*omega
    angle = theta / pi * 180
    ax.plot(angle, TorCon_list_case4, label='TranFEA: %g Hz, %g rpm'%(DriveW_Freq, speed))

    theta = np.array(time_list_case123)*omega
    angle = theta / pi * 180
    ax.plot(angle, TorCon_list_case1, zorder=99, label='TranFEAwi2TSS: %g Hz, %g rpm'%(DriveW_Freq, speed))
        # ax.plot(angle, TorCon_list_case2, label='TranFEAwi2TSS: %g Hz, %g rpm'%(DriveW_Freq, speed)) # wrong speed!
        # ax.plot(angle, TorCon_list_case3, label='TranFEAwi2TSS: %g Hz, %g rpm'%(DriveW_Freq, speed))

    ax.legend()
    ax.grid()
    plt.xlim(0,angle[-1])

    # quit()


    tranFEA_end_angle = angle[-1]


    # this part of codes only works with case123, a case4 will ruin the csv file format...
    # Transient FEA-TwoStepSection

    # # read csv data
    # number_of_total_cases = 3
    #     # tran_study_name += '-ori'
    # t, torque, slip_freq, speed, current, force = read_csv_results(dir_csv_output_folder + tran_study_name, '_torque.csv')
    # print slip_freq
    # print speed

    # for i in range(number_of_total_cases):
    #     y = [cases[i] for cases in torque]
    #     omega = speed[i] / 60. * 2*pi # mechanical speed: rpm to rad/s (no pole pair number neeeded)
    #     theta = np.array(t)*omega
    #     angle = theta / pi * 180

    #     # toruqe vs time
    #     ax = figure(1).gca()
    #     ax.plot(t, y, label=str(speed[i]))

    #     # toruqe vs angle
    #     ax = figure(2).gca()
    #     ax.plot(angle, y, zorder=99, label='TranFEAwi2TSS: 500 Hz, %g rpm' % (speed[i]))
    #     if i == 0:
    #         tranFEA_end_angle = angle[-1]

    #     # force vs angle
    #     ForCon_X = [cases[i*3:i*3+3][0] for cases in force]
    #     ForCon_Y = [cases[i*3:i*3+3][1] for cases in force]

    #     # print len(ForCon_X)
    #     # angle    = angle[33:]
    #     # ForCon_X = ForCon_X[33:]
    #     # ForCon_Y = ForCon_Y[33:]

    #     ax = figure(4).gca()
    #     ax.plot(angle, ForCon_X, label='X'+str(speed[i]))
    #     ax = figure(5).gca()
    #     ax.plot(angle, ForCon_Y, label='Y'+str(speed[i]))

    #     # Eric: as the rotor currents build up, the unbalanced air gap field is shifted.
    #     ax = figure(25).gca()
    #     ax.plot(angle, np.sqrt(np.array(ForCon_X)**2+np.array(ForCon_Y)**2), zorder=99, label='TranFEAwi2TSS: 500 Hz, %g rpm' % (speed[i]))


    #     print len(ForCon_X)
    #     angle    = angle[33:]
    #     ForCon_X = ForCon_X[33:]
    #     ForCon_Y = ForCon_Y[33:]


    #     ForCon_Avg_Amplitude = np.array([sum(ForCon_X), sum(ForCon_Y)]) / len(ForCon_Y)
    #     ForCon_Angle_List = arctan2(ForCon_Y, ForCon_X) / pi * 180
    #     ForCon_Avg_Angle = 0.5*(max(ForCon_Angle_List) - min(ForCon_Angle_List))
    #     print 'case:', i, speed[i], 'rpm', ForCon_Avg_Amplitude
    #     # print ForCon_Angle_List, 'deg'
    #     print ForCon_Avg_Angle, 'deg'

    #     # Vector plot
    #     ax = figure(6+i).gca()
    #     for x, y in zip(ForCon_X, ForCon_Y):
    #         ax.arrow(0,0, x,y)

    #     # ax = figure(9).gca()
    #     # ax.arrow(0,0, 0.5, 0.5)
    #     break

    # ax = figure(1).gca()
    # ax.legend()
    # ax.grid()
    # ax.set_ylabel('Torque [Nm]')
    # ax.set_xlabel('Time [s]')

    # ax = figure(2).gca()
    # ax.legend()
    # ax.grid()
    # ax.set_ylabel('Torque [Nm]')
    # ax.set_xlabel('rotor position [deg]')

    # ax = figure(3).gca()
    # for i in range(int(imID)/DriveW_poles):
    #     y = [bars[i] for bars in current]
    #     ax.plot(t,y)
    # ax.legend()
    # ax.grid()
    # ax.set_ylabel('Current [A]')
    # ax.set_xlabel('Time [s]')
    
    # ax = figure(4).gca()
    # ax.legend()
    # ax.grid()
    # ax.set_ylabel('Force X [N]')
    # ax.set_xlabel('Angle [deg]')

    # ax = figure(5).gca()
    # ax.legend()
    # ax.grid()
    # ax.set_ylabel('Force Y [N]')
    # ax.set_xlabel('Angle [deg]')

    # ax = figure(25).gca()
    # ax.legend()
    # ax.grid()
    # ax.set_ylabel('Force [N]')
    # ax.set_xlabel('rotor position [deg]')

    # for i in range(3):
    #     ax = figure(6+i).gca()
    #     ax.legend()
    #     ax.grid()
    #     ax.set_ylabel('Force Y [N]')
    #     ax.set_xlabel('Force X [N]')
    #     plt.xlim([0,200])
    #     plt.ylim([0,200])
    #     break








    # Rotating EC FEA
    angle, torque, slip_freq, _, current, force = read_csv_results(dir_csv_output_folder + ecrot_study_name, '_torque.csv')
    print dir_csv_output_folder + ecrot_study_name
    print slip_freq
    angle = [el[0] for el in angle]
        # angle = angle[1:] + [angle[-1] + angle[1]] # angle[0] is 0, so extending angle leads to bug
    y = [el[0] for el in torque]
    ax = figure(102).gca()
    # print type(angle), type(y)

    data_length_using_angle = int(tranFEA_end_angle / angle[1]) / len(y)
    extended_angle = [angle[1]*i for i in range(len(y)*data_length_using_angle)]
    plot(extended_angle, y*data_length_using_angle, zorder=1, label='EddyCurFEAwiRR: %g Hz'%(slip_freq))
    # ax.set_ylabel('Torque [Nm]')
    # ax.set_xlabel('Angle [deg]')
    ax.legend()

    # Current step by step (rotating rotor)
    # first bar's current amplitude changes as the rotor position changes
    for i, row in enumerate(current):
        print i
        for ind in range(0, len(row)/2, 2):
            print '\t', ind, np.sqrt(row[ind]**2 + row[ind+1]**2)
            break 

    # print force
    print 'EC-Rotate Force Results:'
    ForCon_X = [el[0] for el in force]
    ForCon_Y = [el[1] for el in force]

    ax = figure(4).gca()
    ax.plot(extended_angle, ForCon_X*data_length_using_angle, label='X-EC-Rot')
    ax = figure(5).gca()
    ax.plot(extended_angle, ForCon_Y*data_length_using_angle, label='Y-EC-Rot')

    ax = figure(125).gca()
    ForCon_Abs = sqrt(array(ForCon_X)**2 + array(ForCon_Y)**2)
    print len(ForCon_Abs)*data_length_using_angle
    print len(extended_angle)
    ax.plot(extended_angle, ForCon_Abs.tolist()*data_length_using_angle, zorder=1, label='EddyCurFEAwiRR: %g Hz'%(slip_freq))
    ax.legend()
    ax.grid(True)

    ForCon_Avg_Amplitude = np.array([sum(ForCon_X), sum(ForCon_Y)]) / len(ForCon_Y)
    ForCon_Angle_List = arctan2(ForCon_Y, ForCon_X) / pi * 180
    ForCon_Avg_Angle = 0.5*(max(ForCon_Angle_List) - min(ForCon_Angle_List))
    print 'EC-Rotate:', ForCon_Avg_Amplitude
    # print ForCon_Angle_List, 'deg'
    print ForCon_Avg_Angle, 'deg'

    # Vector plot
    ax = figure(13).gca()
    for x, y in zip(ForCon_X, ForCon_Y):
        ax.arrow(0,0, x,y)
    plt.xlim([0,200])
    plt.ylim([0,200])





    fig2.tight_layout()
    fig_name = 'EC-RR-cmp-torque.png'
    fig2.savefig(fig_name, dpi=150)

    fig25.tight_layout()
    fig_name = 'EC-RR-cmp-force.png'
    fig25.savefig(fig_name, dpi=150)



    # quit()
    show()

    # take a look at current
    #2. compute the force mag error and angle error?
    # add all the force vector together and get its mean force vector





