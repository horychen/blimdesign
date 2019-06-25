
import pandas as pd
from pylab import subplots, mpl, plt, np
# plt.style.use('seaborn')
# plt.style.use('fivethirtyeight')
mpl.style.use('classic')

mpl.rcParams['legend.fontsize'] = 12
# mpl.rcParams['legend.family'] = 'Times New Roman'
mpl.rcParams['font.family'] = ['Times New Roman']
mpl.rcParams['font.size'] = 14.0
font = {'family' : 'Times New Roman', #'serif',
        'color' : 'darkblue',
        'weight' : 'normal',
        'size' : 14,}
textfont = {'family' : 'Times New Roman', #'serif',
            'color' : 'darkblue',
            'weight' : 'normal',
            'size' : 11.5,}

# fig, ax_list = subplots(3, 1, sharex=True, dpi=150, figsize=(16*0.75, 8*0.75), facecolor='w', edgecolor='k')

def my_read_csv(fname):
    header = pd.read_csv(fname, skiprows=2, nrows=7) # na_values = ['no info', '.', '']
    data = pd.read_csv(fname, skiprows=9)
    return header, data 

import utility
def performance_metrics(time, torque, x_force, y_force, number_of_steps_2ndTSS):
    sfv = utility.suspension_force_vector(x_force, y_force, range_ss=number_of_steps_2ndTSS)
    str_results, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle = \
        utility.add_plots( None, None,
                      title='ReleasePerformance',#tran_study_name,
                      label='Fine Step Transient FEA',
                      zorder=8,
                      time_list=time,
                      sfv=sfv,
                      torque=torque,
                      range_ss=sfv.range_ss)
    return str_results, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle

def plot_data(fname_force, fname_torque, fig2, ax2, fig2_title, fig, ax_list, label='', marker='-ko'):

    ################################################################
    # Common
    ################################################################
    number_of_steps_2ndTSS = 300
    index_begin = -number_of_steps_2ndTSS

    stack_length_mm = 92.592593
    rated_rotor_volume = 0.000830274
    rated_stack_length_mm = 1e3 * rated_rotor_volume / (np.pi*(47.746483*1e-3)**2)
    rated_scale = rated_stack_length_mm / stack_length_mm

    # excitation_ratio = [2.5] + list(range(5,100,5))
    excitation_ratio = [0.025] + list(np.arange(0.05,1,0.05))

    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # Force
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    header, data = my_read_csv(fname_force)
    # print(header)
    # print(data)
    time = data['Time(s)']
    time = time.tolist()
    time = [1000*(el-time[index_begin]) for el in time]
    count = 0
    l_x_force = []
    l_y_force = []
    ax = ax2[0]
    for key, val in data.iteritems():
        if 'Time' in key or 'Z' in key:
            continue
        count += 1
        # print('-'*40)
        # print(key, val)
        if 'X' in key:
            x_force = rated_scale*val[index_begin:]
            # ax_list[0].plot(time[index_begin:], x_force)
            l_x_force.append(x_force)
        if 'Y' in key:
            y_force = rated_scale*val[index_begin:]
            # ax_list[1].plot(time[index_begin:], y_force)
            l_y_force.append(y_force)
        if count % 2 == 0:
            if count <= 7*2:
                # ax_list[2].plot(time[index_begin:], np.sqrt(x_force**2+y_force**2), label=str(int(count/2)))
                amp_force = np.sqrt(x_force**2+y_force**2)
                ax.plot(time[index_begin:], amp_force, label=str(count//2))
                t=ax.text(  time[index_begin:][number_of_steps_2ndTSS//2], 
                            amp_force[number_of_steps_2ndTSS//2],
                            '%g'%(excitation_ratio[count//2-1])+' p.u.' )
                t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor=None))

    # ax_list[-1].set_xlabel('Time [s]')
    # ax_list[0].set_ylabel('x-axis Force [N]')
    # ax_list[1].set_ylabel('y-axis Force [N]')
    # ax_list[2].legend()

    ax.set_xlabel('Time [ms]\n%s'%(fig2_title[0]))
    ax.set_ylabel('Force Amplitude [N]')
    ax.grid(True)

    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # Torque
    #~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    # fig, ax = subplots(1, 1, facecolor='w', edgecolor='k')
    ax=ax2[1]
    header, data = my_read_csv(fname_torque)

    time = data['Time(s)']
    time = time.tolist()
    time = [1000*(el-time[index_begin]) for el in time]
    count = 0
    l_torque = []
    for key, val in data.iteritems():
        if 'Time' in key:
            continue
        count += 1

        torque = rated_scale*val[index_begin:]

        if count <=7:
            ax.plot(time[index_begin:], torque, label=str(int(count)))
            t=ax.text(  time[index_begin:][number_of_steps_2ndTSS//2], 
                        torque[number_of_steps_2ndTSS//2],
                        '%g'%(excitation_ratio[count-1])+' p.u.' )
            t.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor=None))

        l_torque.append(torque)

    ax.set_xlabel('Time [ms]\n%s'%(fig2_title[1]))
    ax.set_ylabel('Torque [Nm]')
    ax.grid(True)

    fig2.tight_layout()
    fig2.savefig(r'C:\Users\horyc\Desktop/cases_force_amp.png', dpi=150)

    ################################################################
    # Performance Metrics
    ################################################################
    count = 0
    l_torque_average = []
    l_normalized_torque_ripple = []
    l_ss_avg_force_magnitude = []
    l_normalized_force_error_magnitude = []
    l_force_error_angle = []
    for torque, x_force, y_force in zip(l_torque, l_x_force, l_y_force):
        count += 1
        str_results, torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle = \
            performance_metrics(time, torque, x_force, y_force, number_of_steps_2ndTSS)
        l_torque_average.append(torque_average)
        l_normalized_torque_ripple.append(normalized_torque_ripple)
        l_ss_avg_force_magnitude.append(ss_avg_force_magnitude)
        l_normalized_force_error_magnitude.append(normalized_force_error_magnitude)
        l_force_error_angle.append(force_error_angle)
        print(count, str_results)

    # excitation_ratio = [2.5] + list(range(5,100,5))
    # print(len(l_torque_average))
    # print(len(excitation_ratio))
    # ax.plot(excitation_ratio, l_torque_average)
    # for x, y in zip(excitation_ratio, l_force_error_angle):
    #     print(x,y )
    ax_list[0][0].plot(excitation_ratio, l_torque_average, marker, alpha=0.75, label=label)
    ax_list[0][0].set_ylabel('Torque [Nm]')
    ax_list[0][1].plot(excitation_ratio, l_ss_avg_force_magnitude, marker, alpha=0.75)
    ax_list[0][1].set_ylabel('Force Amplitude [N]')
    ax_list[1][0].plot(excitation_ratio, l_force_error_angle, marker, alpha=0.75)
    ax_list[1][0].set_ylabel('Force Error Angle $E_a$ [deg]')
    ax_list[1][1].plot(excitation_ratio, [100*el for el in l_normalized_force_error_magnitude], marker, alpha=0.75)
    ax_list[1][1].set_ylabel('Force Error Magnitude $E_m$ [%]')


    ax_list[0][0].legend(loc=(0.45,0.7))
    ax_list[0][0].set_xlabel('(a)')
    ax_list[0][1].set_xlabel('(b)')
    ax_list[1][0].set_xlabel('Suspension Current [p.u.]\n(c)')
    ax_list[1][1].set_xlabel('Suspension Current [p.u.]\n(d)')
    for i in range(2):
        for j in range(2):
            ax_list[i][j].grid(True)
    fig.savefig(r'C:\Users\horyc\Desktop/cases_suspension_current_ratio.png', dpi=300)

fname_force = '../article/csv_proj1130/ind1130Tran2TSSProlong-Cases_force.csv'
fname_torque = '../article/csv_proj1130/ind1130Tran2TSSProlong-Cases_torque.csv'
    # fname_force = '../article/csv_proj1130/ind1130Tran2TSSProlong-CasesNoload_force.csv'
    # fname_torque = '../article/csv_proj1130/ind1130Tran2TSSProlong-CasesNoload_torque.csv'
fig2, ax2 = subplots(1, 4, figsize=(16,8), facecolor='w', edgecolor='k', constrained_layout=False, dpi=200)
fig2.subplots_adjust(wspace=0.25)
fig, ax_list = subplots(2, 2, sharex=True, facecolor='w', edgecolor='k', constrained_layout=True)
plot_data(fname_force, fname_torque, fig2, ax2[:2], ('(a)', '(b)'), fig, ax_list, label='Rated Current', marker='--ko')

fname_force = '../article/csv_proj1130/ind1130Tran2TSSProlong-CasesTorqueFixed_force.csv'
fname_torque = '../article/csv_proj1130/ind1130Tran2TSSProlong-CasesTorqueFixed_torque.csv'
plot_data(fname_force, fname_torque, fig2, ax2[2:], ('(c)', '(d)'), fig, ax_list, label='Over Current', marker='--rs')

plt.show()
quit()

