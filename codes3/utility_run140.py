# coding:u8

from utility import *

Qr = 16

# Basic information
# from math import pi
required_torque = 15.9154943092 #Nm

Radius_OuterRotor = 47.092753
stack_length = 93.200295
Omega =  3132.95327379
rotor_volume = math.pi*(Radius_OuterRotor*1e-3)**2 * (stack_length*1e-3)
rotor_weight = 9.8 * rotor_volume * 8050 # steel 8,050 kg/m3. Copper/Density 8.96 g/cm³. gravity: 9.8 N/kg

print('utility_run140.py')
print('Qr=%d, rotor_volume='%(Qr), rotor_volume, 'm^3')
print('Qr=%d, rotor_weight='%(Qr), rotor_weight, 'N')

# run 140
de_config_dict = {  'original_bounds':[ [ 4.9,   9],#--# stator_tooth_width_b_ds
                                        [ 0.8,   3],   # air_gap_length_delta
                                        [5e-1,   3],   # Width_RotorSlotOpen 
                                        [ 6.5, 9.9],#--# rotor_tooth_width_b_dr # 8 is too large, 6 is almost too large
                                        [5e-1,   3],   # Length_HeadNeckRotorSlot
                                        [   1,  10],   # Angle_StatorSlotOpen
                                        [5e-1,   3] ], # Width_StatorTeethHeadThickness
                    'mut':        0.8,
                    'crossp':     0.7,
                    'popsize':    42, # 50, # 100,
                    'iterations': 1 } # 148

original_bounds = de_config_dict['original_bounds']
dimensions = len(original_bounds)
min_b, max_b = np.asarray(original_bounds).T 
diff = np.fabs(min_b - max_b)
# pop_denorm = min_b + pop * diff
# pop[j] = (pop_denorm[j] - min_b) / diff

import itertools
if __name__ == '__main__':
    from pylab import *
    plt.rcParams["font.family"] = "Times New Roman"

    # Pareto Plot
    if False:
        swda = SwarmDataAnalyzer(run_integer=121)
        # print swda.lsit_cost_function()
        # 
        O2 = fobj_list( list(swda.get_certain_objective_function(2)), #torque_average, 
                        list(swda.get_certain_objective_function(4)), #ss_avg_force_magnitude, 
                        list(swda.get_certain_objective_function(3)), #normalized_torque_ripple, 
                        list(swda.get_certain_objective_function(5)), #normalized_force_error_magnitude, 
                        list(swda.get_certain_objective_function(6)), #force_error_angle, 
                        array(list(swda.get_certain_objective_function(9))) + array(list(swda.get_certain_objective_function(12))) + array(list(swda.get_certain_objective_function(13))), #total_loss, 
                        weights=[ 1, 1.0,   1, 1.0, 1.0,   0 ], rotor_volume=rotor_volume, rotor_weight=rotor_weight)
        # print O2
        # print array(swda.lsit_cost_function()) - array(O2) # they are the same
        O2 = O2.tolist()

        O2_ref = fobj_scalar(19.1197, 96.9263, 0.0864712, 0.104915, 6.53137, (1817.22+216.216+224.706), weights=[ 1, 1.0,   1, 1.0, 1.0,   0 ], rotor_volume=rotor_volume, rotor_weight=rotor_weight)
        print('O2_ref=', O2_ref)

        def my_scatter_plot(x,y,O,xy_ref,O_ref, fig=None, ax=None, s=15):
            # O is a copy of your list rather than array or the adress of the list
            # O is a copy of your list rather than array or the adress of the list
            # O is a copy of your list rather than array or the adress of the list
            x += [xy_ref[0]]
            y += [xy_ref[1]]
            O += [O2_ref]
            if ax is None or fig is None:
                fig = figure()
                ax = fig.gca()
            # O2_mix = np.concatenate([[O2_ref], O2], axis=0) # # https://stackoverflow.com/questions/46106912/one-colorbar-for-multiple-scatter-plots
            # min_, max_ = O2_mix.min(), O2_mix.max()
            ax.annotate('Initial design', xytext=(xy_ref[0]*0.95, xy_ref[1]*0.9), xy=xy_ref, xycoords='data', arrowprops=dict(arrowstyle="->"))
            # scatter(*xy_ref, marker='s', c=O2_ref, s=20, alpha=0.75, cmap='viridis')
            # clim(min_, max_)
            scatter_handle = ax.scatter(x, y, c=O, s=s, alpha=0.5, cmap='viridis')
            # clim(min_, max_)
            ax.grid()

            if True:
                best_index, best_O = get_min_and_index(O)
                print(best_index, best_O)
                xy_best = (x[best_index], y[best_index])
                handle_best = ax.scatter(*xy_best, s=s*3, marker='s', facecolors='none', edgecolors='r')
                ax.legend((handle_best,), ('Best',))

            return scatter_handle
        
        fig, axeses = subplots(2, 2, sharex=False, dpi=150, figsize=(12, 6), facecolor='w', edgecolor='k')
        # fig, axeses = subplots(2, 2, sharex=False, dpi=150, figsize=(10, 8), facecolor='w', edgecolor='k')
        fig.subplots_adjust(right=0.9, hspace=0.21, wspace=0.11) # won't work after I did something. just manual adjust!


        if True:
            # TRV vs Torque Ripple
            ax = axeses[0][0]
            xy_ref = (19.1197/rotor_volume/1e3, 0.0864712) # from run#117
            x, y = array(list(swda.get_certain_objective_function(2)))/rotor_volume/1e3, list(swda.get_certain_objective_function(3))
            x = x.tolist()
            my_scatter_plot(x,y,O2[::],xy_ref,O2_ref, fig=fig, ax=ax)
            ax.set_xlabel('TRV [Nm/m^3]\n(a)')
            ax.set_ylabel(r'$T_{\rm rip}$ [100%]')

            # FRW vs Ea
            ax = axeses[0][1]
            xy_ref = (96.9263/rotor_weight, 6.53137)
            x, y = array(list(swda.get_certain_objective_function(4)))/rotor_weight, list(swda.get_certain_objective_function(6))
            x = x.tolist()
            my_scatter_plot(x,y,O2[::],xy_ref,O2_ref, fig=fig, ax=ax)
            ax.set_xlabel('FRW [1]\n(b)')
            ax.set_ylabel(r'$E_a$ [deg]')

            # FRW vs Em
            ax = axeses[1][0]
            xy_ref = (96.9263/rotor_weight, 0.104915)
            x, y = array(list(swda.get_certain_objective_function(4)))/rotor_weight, list(swda.get_certain_objective_function(5))
            x = x.tolist()
            my_scatter_plot(x,y,O2[::],xy_ref,O2_ref, fig=fig, ax=ax)
            ax.set_xlabel('FRW [1]\n(c)')
            ax.set_ylabel(r'$E_m$ [100%]')

            # Em vs Ea
            ax = axeses[1][1]
            xy_ref = (0.104915, 6.53137)
            x, y = list(swda.get_certain_objective_function(5)), list(swda.get_certain_objective_function(6))
            scatter_handle = my_scatter_plot(x,y,O2[::],xy_ref,O2_ref, fig=fig, ax=ax)
            ax.set_xlabel('$E_m$ [100%]\n(d)')
            ax.set_ylabel(r'$E_a$ [deg]')

            # fig.subplots_adjust(right=0.8)
            # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7]) # left, bottom, width, height
            # fig.subplots_adjust(right=0.9)
            # cbar_ax = fig.add_axes([0.925, 0.15, 0.025, 0.7])
            cbar_ax = fig.add_axes([0.925, 0.15, 0.02, 0.7])
            cbar_ax.get_yaxis().labelpad = 10
            clb = fig.colorbar(scatter_handle, cax=cbar_ax)
            clb.ax.set_ylabel(r'Cost function $O_2$', rotation=270)
            # clb.ax.set_title(r'Cost function $O_2$', rotation=0)


            fig.tight_layout()
            # fig.savefig(r'D:\OneDrive\[00]GetWorking\32 blimopti\p2019_ecce_bearingless_induction\images\pareto_plot.png', dpi=150, bbox_inches='tight')
            show()
            quit()

        if False:

            # Torque vs Torque Ripple
            ax = axeses[0][0]
            xy_ref = (19.1197, 0.0864712) # from run#117
            x, y = list(swda.get_certain_objective_function(2)), list(swda.get_certain_objective_function(3))
            my_scatter_plot(x,y,O2[::],xy_ref,O2_ref, fig=fig, ax=ax)
            ax.set_xlabel('$T_{em}$ [Nm]\n(a)')
            ax.set_ylabel(r'$T_{\rm rip}$ [100%]')

            # Force vs Ea
            ax = axeses[0][1]
            xy_ref = (96.9263, 6.53137)
            x, y = list(swda.get_certain_objective_function(4)), list(swda.get_certain_objective_function(6))
            my_scatter_plot(x,y,O2[::],xy_ref,O2_ref, fig=fig, ax=ax)
            ax.set_xlabel('$|F|$ [N]\n(b)')
            ax.set_ylabel(r'$E_a$ [deg]')

            # Force vs Em
            ax = axeses[1][0]
            xy_ref = (96.9263, 0.104915)
            x, y = list(swda.get_certain_objective_function(4)), list(swda.get_certain_objective_function(5))
            my_scatter_plot(x,y,O2[::],xy_ref,O2_ref, fig=fig, ax=ax)
            ax.set_xlabel('$|F|$ [N]\n(c)')
            ax.set_ylabel(r'$E_m$ [100%]')

            # Em vs Ea
            ax = axeses[1][1]
            xy_ref = (0.104915, 6.53137)
            x, y = list(swda.get_certain_objective_function(5)), list(swda.get_certain_objective_function(6))
            scatter_handle = my_scatter_plot(x,y,O2[::],xy_ref,O2_ref, fig=fig, ax=ax)
            ax.set_xlabel('$E_m$ [100%]\n(d)')
            ax.set_ylabel(r'$E_a$ [deg]')

            # fig.subplots_adjust(right=0.8)
            # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7]) # left, bottom, width, height
            # fig.subplots_adjust(right=0.9)
            # cbar_ax = fig.add_axes([0.925, 0.15, 0.025, 0.7])
            cbar_ax = fig.add_axes([0.925, 0.15, 0.02, 0.7])
            cbar_ax.get_yaxis().labelpad = 10
            clb = fig.colorbar(scatter_handle, cax=cbar_ax)
            clb.ax.set_ylabel(r'Cost function $O_2$', rotation=270)
            # clb.ax.set_title(r'Cost function $O_2$', rotation=0)


            fig.tight_layout()
            # fig.savefig(r'D:\OneDrive\[00]GetWorking\32 blimopti\p2019_ecce_bearingless_induction\images\pareto_plot.png', dpi=150, bbox_inches='tight')
            show()

            # Loss vs Ea
            xy_ref = ((1817.22+216.216+224.706), 6.53137)
            x, y = array(list(swda.get_certain_objective_function(9))) + array(list(swda.get_certain_objective_function(12))) + array(list(swda.get_certain_objective_function(13))), list(swda.get_certain_objective_function(6))
            x = x.tolist()
            my_scatter_plot(x,y,O2[::],xy_ref,O2_ref)
            xlabel(r'$P_{\rm Cu,Fe}$ [W]')
            ylabel(r'$E_a$ [deg]')

            quit()   






    # ------------------------------------ Sensitivity Analysis Bar Chart Scripts
    # ------------------------------------ Sensitivity Analysis Bar Chart Scripts
    # ------------------------------------ Sensitivity Analysis Bar Chart Scripts

    # swda = SwarmDataAnalyzer(run_integer=116)
    swda = SwarmDataAnalyzer(run_integer=140)
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
    for ind, i in enumerate(list(range(7))+[9]):
    # for i in range(14):
        print('\n-----------', y_label_list[i])
        l = list(swda.get_certain_objective_function(i))

        # y = l[:len(l)/2] # 115
        y = l # 116
        print('len(y)=', len(y))

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
            print('\t', j, param_list[j], '\t\t', max(y_vs_design_parameter) - min(y_vs_design_parameter))

            data_max[ind].append(max(y_vs_design_parameter))
            data_min[ind].append(min(y_vs_design_parameter))            

        if i==1:
            ax_list[ind].legend()
        ax_list[ind].grid()
        ax_list[ind].set_ylabel(y_label_list[i])

    print('data_max:')
    for ind, el in enumerate(data_max):
        print(ind, el)
    print('data_min')
    for ind, el in enumerate(data_min):
        print(ind, el)


    O2_ref = fobj_scalar(9.89265, 45.7053,  0.11069, 0.0249602, 1.24351, (727.695+230.058+289.72), weights=[ 1, 1.0,   1, 1.0, 1.0,   0 ], rotor_volume=rotor_volume, rotor_weight=rotor_weight)
    O1_ref = fobj_scalar(9.89265, 45.7053,  0.11069, 0.0249602, 1.24351, (727.695+230.058+289.72), weights=[ 1, 0.1,   1, 0.1, 0.1,   0 ], rotor_volume=rotor_volume, rotor_weight=rotor_weight)
    # O2_ref = fobj_scalar(19.1197, 96.9263, 0.0864712, 0.104915, 6.53137, (1817.22+216.216+224.706), weights=[ 1, 1.0,   1, 1.0, 1.0,   0 ], rotor_volume=rotor_volume, rotor_weight=rotor_weight)
    # O1_ref = fobj_scalar(19.1197, 96.9263, 0.0864712, 0.104915, 6.53137, (1817.22+216.216+224.706), weights=[ 1, 0.1,   1, 0.1, 0.1,   0 ], rotor_volume=rotor_volume, rotor_weight=rotor_weight)

    print('Objective function 1')
    O1 = fobj_list( list(swda.get_certain_objective_function(2)), 
                    list(swda.get_certain_objective_function(4)), 
                    list(swda.get_certain_objective_function(3)), 
                    list(swda.get_certain_objective_function(5)), 
                    list(swda.get_certain_objective_function(6)), 
                    array(list(swda.get_certain_objective_function(9))) + array(list(swda.get_certain_objective_function(12))) + array(list(swda.get_certain_objective_function(13))),
                    weights=[ 1, 0.1,   1, 0.1, 0.1,   0 ], rotor_volume=rotor_volume, rotor_weight=rotor_weight)
    O1_max = []
    O1_min = []
    O1_ax  = figure().gca()
    for j in range(len(O1)/number_of_variant): # iterate design parameters
        O1_vs_design_parameter = O1[j*number_of_variant:(j+1)*number_of_variant]

        O1_ax.plot(O1_vs_design_parameter, label=str(j)+' '+param_list[j], alpha=0.5)
        print('\t', j, param_list[j], '\t\t', max(O1_vs_design_parameter) - min(O1_vs_design_parameter), '\t\t', end=' ')
        print([ind for ind, el in enumerate(O1_vs_design_parameter) if el < O1_ref]) #'<- to derive new original_bounds.'

        O1_max.append(max(O1_vs_design_parameter))
        O1_min.append(min(O1_vs_design_parameter))            
    O1_ax.legend()
    O1_ax.grid()
    O1_ax.set_ylabel('O1 [1]')
    O1_ax.set_xlabel('Count of design variants')

    print('Objective function 2')
    O2 = fobj_list( list(swda.get_certain_objective_function(2)), 
                    list(swda.get_certain_objective_function(4)), 
                    list(swda.get_certain_objective_function(3)), 
                    list(swda.get_certain_objective_function(5)), 
                    list(swda.get_certain_objective_function(6)), 
                    array(list(swda.get_certain_objective_function(9))) + array(list(swda.get_certain_objective_function(12))) + array(list(swda.get_certain_objective_function(13))),
                    weights=[ 1, 1.0,   1, 1.0, 1.0,   0 ], rotor_volume=rotor_volume, rotor_weight=rotor_weight)
    O2_max = []
    O2_min = []
    O2_ax  = figure().gca()
    O2_ecce_data = []
    for j in range(len(O2)/number_of_variant): # iterate design parameters: range(7)
        O2_vs_design_parameter = O2[j*number_of_variant:(j+1)*number_of_variant]
        O2_ecce_data.append(O2_vs_design_parameter)

        O2_ax.plot(O2_vs_design_parameter, 'o-', label=str(j)+' '+param_list[j], alpha=0.5)
        print('\t', j, param_list[j], '\t\t', max(O2_vs_design_parameter) - min(O2_vs_design_parameter), '\t\t', end=' ')
        print([ind for ind, el in enumerate(O2_vs_design_parameter) if el < O2_ref]) #'<- to derive new original_bounds.'

        O2_max.append(max(O2_vs_design_parameter))
        O2_min.append(min(O2_vs_design_parameter))            
    O2_ax.legend()
    O2_ax.grid()
    O2_ax.set_ylabel('O2 [1]')
    O2_ax.set_xlabel('Count of design variants')

    # for ecce digest
    fig_ecce = figure(figsize=(10, 5), facecolor='w', edgecolor='k')
    O2_ecce_ax = fig_ecce.gca()
    O2_ecce_ax.plot(list(range(-1, 22)), O2_ref*np.ones(23), 'k--', label='reference design')
    O2_ecce_ax.plot(O2_ecce_data[1], 'o-', lw=0.75, alpha=0.5, label=r'$\delta$'         )
    O2_ecce_ax.plot(O2_ecce_data[0], 'v-', lw=0.75, alpha=0.5, label=r'$b_{\rm tooth,s}$')
    O2_ecce_ax.plot(O2_ecce_data[3], 's-', lw=0.75, alpha=0.5, label=r'$b_{\rm tooth,r}$')
    O2_ecce_ax.plot(O2_ecce_data[5], '^-', lw=0.75, alpha=0.5, label=r'$w_{\rm open,s}$')
    O2_ecce_ax.plot(O2_ecce_data[2], 'd-', lw=0.75, alpha=0.5, label=r'$w_{\rm open,r}$')
    O2_ecce_ax.plot(O2_ecce_data[4], '*-', lw=0.75, alpha=0.5, label=r'$h_{\rm head,s}$')
    O2_ecce_ax.plot(O2_ecce_data[6], 'X-', lw=0.75, alpha=0.5, label=r'$h_{\rm head,r}$')

    myfontsize = 12.5
    rcParams.update({'font.size': myfontsize})


    # Reference candidate design
    ref = zeros(8)
        # ref[0] = 0.635489                                   # PF
        # ref[1] = 0.963698                                   # eta
        # ref[1] = efficiency_at_50kW(1817.22+216.216+224.706)# eta@50kW
    O1_ax.plot(list(range(-1, 22)), O1_ref*np.ones(23), 'k--')
    O2_ax.plot(list(range(-1, 22)), O2_ref*np.ones(23), 'k--')
    O2_ecce_ax.legend()
    O2_ecce_ax.grid()
    O2_ecce_ax.set_xticks(list(range(21)))
    O2_ecce_ax.annotate('Lower bound', xytext=(0.5, 5.5), xy=(0, 4), xycoords='data', arrowprops=dict(arrowstyle="->"))
    O2_ecce_ax.annotate('Upper bound', xytext=(18.0, 5.5),  xy=(20, 4), xycoords='data', arrowprops=dict(arrowstyle="->"))
    # O2_ecce_ax.set_xlim((-0.5,20.5))
    # O2_ecce_ax.set_ylim((4,14))
    O2_ecce_ax.set_xlabel(r'Number of design variant', fontsize=myfontsize)
    O2_ecce_ax.set_ylabel(r'$O_2(x)$ [1]', fontsize=myfontsize)
    fig_ecce.tight_layout()
    # fig_ecce.savefig(r'D:\OneDrive\[00]GetWorking\32 blimopti\p2019_ecce_bearingless_induction_full_paper\images\O2_vs_params.png', dpi=150)
    # show()
    # quit()


    ################################################################
    # Sensitivity Bar Chart
    ################################################################
    ref[0] = O2_ref    / 8
    ref[1] = O1_ref    / 3
    ref[2] = 9.89265   / required_torque                # 100%
    ref[3] = 0.11069 / 0.1                            # 100%
    ref[4] = 45.7053   / rotor_weight                   # 100% = FRW
    ref[5] = 0.0249602  / 0.2                            # 100%
    ref[6] = 1.24351   / 10                             # deg
    ref[7] = (727.695+230.058+289.72) / 2500           # W

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
    print('ylim=', ax.get_ylim())
    autolabel(ax, rects1, bias=-0.10)
    autolabel(ax, rects2, bias=-0.10)
    autolabel(ax, rects3, bias=-0.10)
    autolabel(ax, rects4, bias=-0.10)
    autolabel(ax, rects5, bias=-0.10)
    autolabel(ax, rects6, bias=-0.10)
    autolabel(ax, rects7, bias=-0.10)
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
    autolabel(ax, rects1)
    autolabel(ax, rects2)
    autolabel(ax, rects3)
    autolabel(ax, rects4)
    autolabel(ax, rects5)
    autolabel(ax, rects6)
    autolabel(ax, rects7)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Normalized Objective Functions')
    ax.set_xticks(count)
    # ax.set_xticklabels(('Power Factor [100%]', r'$\eta$@$T_{em}$ [100%]', r'$T_{em}$ [15.9 N]', r'$T_{rip}$ [10%]', r'$|F|$ [51.2 N]', r'    $E_m$ [20%]', r'      $E_a$ [10 deg]', r'$P_{\rm Cu,Fe}$ [2.5 kW]')))
    # ax.set_xticklabels(('Power Factor [100%]', r'$O_1$ [3]', r'$T_{em}$ [15.9 N]', r'$T_{rip}$ [10%]', r'$|F|$ [51.2 N]', r'    $E_m$ [20%]', r'      $E_a$ [10 deg]', r'$P_{\rm Cu,Fe}$ [2.5 kW]'))
    ax.set_xticklabels((r'$O_2$ [8]', r'$O_1$ [3]', r'$T_{em}$ [15.9 Nm]', r'$T_{rip}$ [10%]', r'$|F|$ [51.2 N]', r'    $E_m$ [20%]', r'      $E_a$ [10 deg]', r'$P_{\rm Cu,Fe}$ [2.5 kW]'))
    ax.grid()
    fig.tight_layout()
    # fig.savefig(r'D:\OneDrive\[00]GetWorking\32 blimopti\p2019_ecce_bearingless_induction\images\sensitivity_results.png', dpi=150)

    show()
    quit()
























    # Pseudo Pareto Optimal Front
    gen_best = swda.get_best_generation()
    with open('d:/gen#0000.txt', 'w') as f:
        f.write('\n'.join(','.join('%.16f'%(x) for x in y) for y in gen_best)) # convert 2d array to string            

    design_parameters_norm = (gen_best - min_b) / diff


    for el in design_parameters_norm:
        print(','.join('%.4f'%(_) for _ in el.tolist()))
    print('airgap length\n', [el[1] for el in gen_best])

    print('Average value of design parameters')
    avg_design_parameters = []
    for el in design_parameters_norm.T:
        avg_design_parameters.append(sum(el)/len(el))
    print(avg_design_parameters)
    avg_design_parameters_denorm = min_b + avg_design_parameters * diff
    print(avg_design_parameters_denorm)

    # for design in swda.get_best_generation(generator=swda.design_display_generator()):
    #     print ''.join(design),
    quit()

    cost = swda.lsit_cost_function()
    indices, items = min_indices(cost, 50)
    print(indices)
    print(items)

    stop = max(indices)
    start = min(indices)
    print(start, stop)

    gene = swda.design_parameters_generator()
    gen_best = []
    for index, design in enumerate(gene):
        if index in indices:
            gen_best.append(design)
    print('there are', index, 'designs')

    # print  len(list(gene))
    # for index in indices:
    #     start = index
    # print next(itertools.islice(gene, start, None)) 


    quit()
    # print min_indices([3,5,7,4,2], 5)
    # print max_indices([3,5,7,4,2], 5)
    # quit()



    print(''.join(swda.buf[:21]), end=' ')

    print(swda.find_individual(14, 0))


    for design in swda.design_display_generator():
        print(design, end=' ')
        break

    for design in swda.design_parameters_generator():
        print(design)
        break

    print() 

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
        print('PF=', compute_power_factor_from_half_period(voltage, current, time, targetFreq=1e3))
        quit()

        # plot(voltage+0.5)
        # plot(current+0.5)

        # Check for resolution
        end_time = N_SAMPLE * TS
        resolution = 1./end_time
        print(resolution, targetFreq)
        if resolution > targetFreq:

            if True: # for half period signal
                print('Data length (N_SAMPLE) too short or sampling frequency too high (1/TS too high).')
                print('Periodical extension is applied. This will not really increase your resolution. It is a visual trick for Fourier Analysis.')
                voltage = voltage.tolist() + (-voltage).tolist() #[::-1]
                current = current.tolist() + (-current).tolist() #[::-1]

            voltage *= 1000
            current *= 1000

            N_SAMPLE = len(voltage)
            end_time = N_SAMPLE * TS
            resolution = 1./end_time
            print(resolution, targetFreq, 'Hz')
            if resolution <= targetFreq:
                print('Now we are good.')


            # 目前就写了二分之一周期的处理，只有四分之一周期的数据，可以用反对称的方法周期延拓。

        print(gs_u.goertzel_offline(targetFreq, 1./TS, voltage))
        print(gs_i.goertzel_offline(targetFreq, 1./TS, current))

        gs_u.ampl = sqrt(gs_u.real*gs_u.real + gs_u.imag*gs_u.imag) 
        gs_u.phase = arctan2(gs_u.imag, gs_u.real)

        print('N_SAMPLE=', N_SAMPLE)
        print("\n")
        print(gs_u.id)
        print("RT:\tAmplitude: %g, %g\n" % (gs_u.ampl, sqrt(2.0*gs_u.accumSquaredData/N_SAMPLE)))
        print("RT:\tre=%g, im=%g\n\tampl=%g, phase=%g\n" % (gs_u.real, gs_u.imag, gs_u.ampl, gs_u.phase*180/math.pi))

        # // do the analysis
        gs_i.ampl = sqrt(gs_i.real*gs_i.real + gs_i.imag*gs_i.imag) 
        gs_i.phase = arctan2(gs_i.imag, gs_i.real)
        print(gs_i.id)
        print("RT:\tAmplitude: %g, %g\n" % (gs_i.ampl, sqrt(2.0*gs_i.accumSquaredData/N_SAMPLE)))
        print("RT:\tre=%g, im=%g\n\tampl=%g, phase=%g\n" % (gs_i.real, gs_i.imag, gs_i.ampl, gs_i.phase*180/math.pi))

        print("------------------------")
        print("\tPhase Difference of I (version 1): %g\n" % ((gs_i.phase-gs_u.phase)*180/math.pi), 'PF=', cos(gs_i.phase-gs_u.phase))

        # // Take reference to the voltage phaser
        Ireal = sqrt(2.0*gs_i.accumSquaredData/N_SAMPLE) * cos(gs_i.phase-gs_u.phase)
        Iimag = sqrt(2.0*gs_i.accumSquaredData/N_SAMPLE) * sin(gs_i.phase-gs_u.phase)
        print("\tAmplitude from accumSquaredData->%g\n" % (sqrt(2.0*gs_u.accumSquaredData/N_SAMPLE)))
        print("\tPhaser I:\tre=%g, im=%g\n" % (Ireal, Iimag))
        print("\tPhase Difference of I (version 2): %g\n" % (arctan2(Iimag,Ireal)*180/math.pi), 'PF=', cos(arctan2(Iimag,Ireal)))

        plot(voltage)
        plot(current)

        show()

        # it works!
        # send_notification('Test email')

