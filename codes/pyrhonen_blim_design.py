#coding:u8
from __future__ import division
from pylab import *

print 'Pyrhonen 2009 Chapter 7.'

# class initial_induction_motor_design(object):

print 'Guesses: alpha_i, efficiency, power_factor.'

# model_name_prefix = 'Qr_loop'
# model_name_prefix = 'Qr_loop_B' # longer solve time, less models, better mesh, higher turns for bearing winding.
# model_name_prefix = 'EC_Rotate_PS' # longer solve time, less models, better mesh, higher turns for bearing winding.
# model_name_prefix = 'ECRot_PS_Opti' # longer solve time, less models, better mesh, higher turns for bearing winding.
model_name_prefix = 'StaticFEA_PS_Opti' # Fix Bug for the rotor slot radius as half of rotor tooth width

loc_txt_file = '../pop/%s.txt'%(model_name_prefix)
f=open(loc_txt_file, 'w')
f.close()
# for THE_IM_DESIGN_ID, Qr in enumerate([16,20,28,32,36]): # any Qr>36 will not converge (for alpha_i and k_sat)
for THE_IM_DESIGN_ID, Qr in enumerate([32,36]): # any Qr>36 will not converge (for alpha_i and k_sat)
# for THE_IM_DESIGN_ID, Qr in enumerate([32]):

    THE_IM_DESIGN_ID = Qr

    print '''\n1. Initial Design Parameters '''
    mec_power = 50e3 # W
    rated_frequency = 500 # Hz
    no_pole_pairs = 2
    speed_rpm = rated_frequency * 60 / no_pole_pairs # rpm
    U1_rms = 500 / sqrt(3) # V - Wye-connect
    # U1_rms = 500  # V - Delta-connect
    stator_phase_voltage_rms = U1_rms
    no_phase_m = 3
    print 'speed_rpm=', speed_rpm



    print '''\n2. Machine Constants '''
    tangential_stress = 21500 # 12000 ~ 33000
    if no_pole_pairs == 1:
        machine_constant_Cmec = 150 # kW s / m^3. Figure 6.3
    else:
        machine_constant_Cmec = 250 # kW s / m^3. 这里的电机常数都是按100kW去查的表哦，偏大了。
    print 'tangential_stress=', tangential_stress
    print 'machine_constant_Cmec=', machine_constant_Cmec



    print '''\n3. Machine Sizing '''
    required_torque = mec_power/(2*pi*speed_rpm)*60
    print 'required_torque', required_torque, 'Nm'

    rotor_volume_Vr = required_torque/(2*tangential_stress)

    length_ratio_chi = pi/(2*no_pole_pairs) * no_pole_pairs**(1/3) # Table 6.5
    print 'length_ratio_chi=', length_ratio_chi

    rotor_outer_diameter_Dr = (4/pi*rotor_volume_Vr*length_ratio_chi)**(1/3)
    rotor_outer_radius_r_or = 0.5 * rotor_outer_diameter_Dr
    print 'rotor_outer_diameter_Dr=', rotor_outer_diameter_Dr*1e3, 'mm'
    print 'rotor_outer_radius_r_or=', rotor_outer_radius_r_or*1e3, 'mm'
    print 'Yegu Kang: rotor_outer_diameter_Dr, 95 mm'

    stack_length = rotor_outer_diameter_Dr * length_ratio_chi
    print 'stack_length=', stack_length*1e3, 'mm'




    print '''\n4. Air Gap Length  (看integrated box 硕士论文，3.1.3，Pyrhonen09给的只适用50Hz电机) '''
    if no_pole_pairs == 1:
        air_gap_length_delta = (0.2 + 0.01*mec_power**0.4) / 1000
    else:
        air_gap_length_delta = (0.18 + 0.006*mec_power**0.4) / 1000
    print 'air_gap_length_delta=', air_gap_length_delta*1e3, 'mm', 'this is for 50 Hz line-start IM'
    print 'Kevin S. Campbell: this is too small. 3 mm is good!'
    air_gap_length_delta *= 2
    print 'air_gap_length_delta=', air_gap_length_delta*1e3, 'mm', 'high speed inverted driven IM'


    stack_length_eff = stack_length + 2 * air_gap_length_delta
    print 'stack_length_eff=', stack_length_eff*1e3, 'mm'

    air_gap_diameter_D = 1*air_gap_length_delta + rotor_outer_diameter_Dr
    stator_inner_diameter_Ds = 2*air_gap_length_delta + rotor_outer_diameter_Dr
    stator_inner_radius_r_s = 0.5*stator_inner_diameter_Ds



    print '''\n5. Stator Winding and Slots '''
    '''为了方便散热，应该加大槽数！
    为了获得正弦磁势，应该加大槽数！
    缺点，就是基波绕组系数降低。A high number of slots per pole and phase lead to a situation in which the influence of the fifth and seventh harmonics is almost insignificant. Py09

    The magnitude of the harmonic torques depends greatly on the ratio of the slot numbers of
    the stator and the rotor. The torques can be reduced by skewing the rotor slots with respect to
    the stator. In that case, the slot torques resulting from the slotting are damped. In the design
    of an induction motor, special attention has to be paid to the elimination of harmonic torques. Py09

    注意，其实50kW、30krpm的电机和5kW、3krpm的电机产生的转矩是一样大的，区别就在产生的电压，如果限制电压，那就只能降低匝数（线圈变粗，电阻变小），方便提高额定电流！
    把代码改改，弄成双层绕组的（短距系数）。

    按照公式计算一下短距、分布、斜槽系数啊！
    '''
    print 'Available Qs choice: k * 2*no_pole_pairs * no_phase_m. However, we have to make sure it is integral slot for the bearing winding that has two pole pairs'
    for i in range(1, 5):
        if no_pole_pairs == 1:
            print i * 2*(no_pole_pairs+1)*no_phase_m,
        else:
            print i * 2*(no_pole_pairs)*no_phase_m,
    Qs = 24
    no_winding_layer = 1

    distribution_q = Qs / (2*no_pole_pairs*no_phase_m)
    print 'distribution_q=', distribution_q
    pole_pitch_tau_p = pi*air_gap_diameter_D/(2*no_pole_pairs) # (3/17)
    print 'pole_pitch_tau_p / (pi*air_gap_diameter_D)=', pole_pitch_tau_p / (pi*air_gap_diameter_D)

    if no_winding_layer == 1:
        coil_span_W = pole_pitch_tau_p # full pitch - easy
        print 'coil_span_W=', coil_span_W, '= pole_pitch_tau_p =', pole_pitch_tau_p
    else:
        if no_pole_pairs == 1:
            stator_slot_pitch_tau_us = pi * air_gap_diameter_D / Qs
            coil_span_W = 0.7 * Qs/2 # for 2 pole motor, p76
            print 'coil_span_W counted by slop:', coil_span_W, round(coil_span_W)
            coil_span_W *= stator_slot_pitch_tau_us
            print 'coil_span_W=', coil_span_W
        else:
            raise Exception('TODO: Add codes for short pitched winding for 4 pole motor.')


    kd1 = 2*sin(1/no_phase_m*pi/2)/(Qs/(no_phase_m*no_pole_pairs)*sin(pi*no_pole_pairs/Qs))
    kq1 = sin(coil_span_W/pole_pitch_tau_p*pi/2)
    ksq1 = sin(pi/(2*no_phase_m*distribution_q)) / (pi/(2*no_phase_m*distribution_q)) # (4.77), skew_s = slot_pitch_tau_u
    kw1 = kd1 * kq1 * ksq1
    print 'winding factor:', kw1, kd1, kq1, ksq1


    # Qr = 30 # Table 7.5. If Qs=24, then Qr!=28, see (7.113).



    print '''\n6. Air Gap Density '''
    air_gap_flux_density_B = 0.8 # 0.7 ~ 0.9 Table 6.3
    print 'air_gap_flux_density_B=', air_gap_flux_density_B
    linear_current_density_A = machine_constant_Cmec / (pi**2/sqrt(2)*kw1*air_gap_flux_density_B)
    print 'linear_current_density_A=', linear_current_density_A, 'kA/m'



    print '''\n7. Number of Coil Turns '''
    desired_emf_Em = 0.95 * U1_rms # 0.96~0.98, high speed motor has higher leakage reactance
    flux_linkage_Psi_m = desired_emf_Em / (2*pi*rated_frequency) 

    alpha_i = 2/pi # ideal sinusoidal flux density distribusion, when the saturation happens in teeth, alpha_i becomes higher.
    air_gap_flux_Phi_m = alpha_i * air_gap_flux_density_B * pole_pitch_tau_p * stack_length_eff
    no_series_coil_turns_N = sqrt(2)*desired_emf_Em / (2*pi*rated_frequency * kw1 * air_gap_flux_Phi_m) # p306
    print 'no_series_coil_turns_N=', no_series_coil_turns_N


    print '''\n8. Round Up Number of Coil Turns '''
    no_series_coil_turns_N = round(no_series_coil_turns_N)
    print 'no_series_coil_turns_N=', no_series_coil_turns_N

    print 'no_series_coil_turns_N must be divisible by pq=%d' %(no_pole_pairs*distribution_q)
    print 'remainder is', int(no_series_coil_turns_N) % int(no_pole_pairs*distribution_q)
    no_series_coil_turns_N = min([no_pole_pairs*distribution_q*i for i in range(100)], key=lambda x:abs(x - no_series_coil_turns_N)) # https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
    print 'suggested no_series_coil_turns_N is', no_series_coil_turns_N, 'modify this manually, if necessary'

    no_parallel_path = 1
    print '''In some cases, especially in low-voltage, high-power machines, there may be a need to change the stator slot number, the number of parallel paths or even the main dimensions of the machine in order to find the appropriate number of conductors in a slot.'''
    no_conductors_per_slot_zQ = 2* no_phase_m * no_series_coil_turns_N /Qs * no_parallel_path
    print 'no_conductors_per_slot_zQ=', no_conductors_per_slot_zQ

    loop_count = 0
    global BH_lookup, B_Arnon5, H_Arnon5
    while True:
    # if True:
        loop_count += 1
        print '''\n9. Recalculate Air Gap Flux Density '''
        air_gap_flux_density_B = sqrt(2)*desired_emf_Em / (2*pi*rated_frequency * kw1 *  alpha_i * no_series_coil_turns_N * pole_pitch_tau_p * stack_length_eff) # p306
        print 'air_gap_flux_density_B=', air_gap_flux_density_B, 'T', '变得比0.8T大了，是因为你减少了匝数取整，反之亦然。'



        print '''\n10. Tooth Flux Density '''
        stator_tooth_flux_density_B_ds = 1.4 #1.4–2.1 (stator) 
        rotor_tooth_flux_density_B_dr = 1.5 #1.5–2.2 (rotor)

        stator_slot_pitch_tau_us = pi * air_gap_diameter_D / Qs
        stator_tooth_apparent_flux_over_slot_pitch_Phi_ds = stack_length_eff * stator_slot_pitch_tau_us * air_gap_flux_density_B
        lamination_stacking_factor_kFe = 0.91 # Yegu Kang
        stator_tooth_width_b_ds = stack_length_eff*stator_slot_pitch_tau_us*air_gap_flux_density_B / (lamination_stacking_factor_kFe*stack_length*stator_tooth_flux_density_B_ds) + 0.1e-3
        print 'stator_tooth_width_b_ds=', stator_tooth_width_b_ds*1e3, 'mm', '--- Here, we neglect the flux through the stator slot!'
        print 'stator_slot_pitch_tau_us=', stator_slot_pitch_tau_us*1e3, 'mm'


        rotor_slot_pitch_tau_ur = pi * air_gap_diameter_D / Qr
        rotor_tooth_apparent_flux_over_slot_pitch_Phi_dr = stack_length_eff*rotor_slot_pitch_tau_ur*air_gap_flux_density_B
        rotor_tooth_width_b_dr = stack_length_eff*rotor_slot_pitch_tau_ur*air_gap_flux_density_B / (lamination_stacking_factor_kFe*stack_length*rotor_tooth_flux_density_B_dr) + 0.1e-3
        print 'rotor_tooth_width_b_dr=', rotor_tooth_width_b_dr*1e3, 'mm', '--- Here, we neglect the flux through the rotor slot!'
        print 'rotor_slot_pitch_tau_ur=', rotor_slot_pitch_tau_ur*1e3, 'mm'



        print '''\n11. Dimension of Slots '''
        efficiency = 0.9 # iterative?
        print 'efficiency=', efficiency
        power_factor = 0.85
        print 'power_factor=', power_factor

        stator_phase_current_rms = mec_power / (no_phase_m*efficiency*stator_phase_voltage_rms*power_factor)
        print 'stator_phase_current_rms', stator_phase_current_rms, 'A'

        rotor_current_referred = stator_phase_current_rms * power_factor
        rotor_current_actual = no_conductors_per_slot_zQ / no_parallel_path * Qs / Qr * rotor_current_referred
        print 'rotor current:', rotor_current_referred, rotor_current_actual

        stator_current_density_Js = 3.7e6 # A/m^2
        print 'stator_current_density_Js=', stator_current_density_Js, 'A/m^2'
        area_conductor_stator_Scs = stator_phase_current_rms / (no_parallel_path * stator_current_density_Js)
        print 'area_conductor_stator_Scs=', area_conductor_stator_Scs * 1e6, 'A/mm^2'

        space_factor_kCu = 0.50 # 未计入绝缘的导体填充率 slot packing factor
        area_stator_slot_Sus = no_conductors_per_slot_zQ * area_conductor_stator_Scs / space_factor_kCu 
        print 'area_stator_slot_Sus', area_stator_slot_Sus*1e6, 'A/mm^2'

        rotor_current_density_Js = 4e6 # A/m^2
        print 'rotor_current_density_Js=', rotor_current_density_Js, 'A/m^2'
        area_conductor_rotor_Scr = rotor_current_actual / (1 * rotor_current_density_Js)
        print 'area_conductor_rotor_Scr=', area_conductor_rotor_Scr * 1e6, 'A/mm^2'

        space_factor_kAl = 1 # no clearance for die cast aluminium bar
        area_rotor_slot_Sur = 1 * area_conductor_rotor_Scr / space_factor_kAl 
        print 'area_rotor_slot_Sur', area_rotor_slot_Sur*1e6, 'A/mm^2'




        temp = (2*pi*stator_inner_radius_r_s - Qs*stator_tooth_width_b_ds)
        stator_tooth_height_h_ds = ( sqrt(temp**2 + 4*area_stator_slot_Sus*Qs) - temp ) / (2*pi)
        print 'stator_tooth_height_h_ds', stator_tooth_height_h_ds*1e3, 'mm'


        temp = (2*pi*rotor_outer_radius_r_or - Qr*rotor_tooth_width_b_dr)
        rotor_tooth_height_h_dr = ( +sqrt(temp**2 - 4*area_rotor_slot_Sur*Qr) + temp ) / (2*pi)
        print 'rotor_tooth_height_h_dr(+)=', rotor_tooth_height_h_dr*1e3, 'mm' # too large to be right answer
        rotor_tooth_height_h_dr = ( -sqrt(temp**2 - 4*area_rotor_slot_Sur*Qr) + temp ) / (2*pi)
        print 'rotor_tooth_height_h_dr(-)=', rotor_tooth_height_h_dr*1e3, 'mm'






        print '''\n12. Magnetic Voltage '''
        # Arnon5
        H_Arnon5 = [0, 9.51030700000000, 11.2124700000000, 13.2194140000000, 15.5852530000000, 18.3712620000000, 21.6562210000000, 25.5213000000000, 30.0619920000000, 35.3642410000000, 41.4304340000000, 48.3863030000000, 56.5103700000000, 66.0660360000000, 77.3405760000000, 90.5910260000000, 106.212089000000, 124.594492000000, 146.311191000000, 172.062470000000, 202.524737000000, 238.525598000000, 281.012026000000, 331.058315000000, 390.144609000000, 459.695344000000, 541.731789000000, 638.410494000000, 752.333643000000, 886.572927000000, 1044.77299700000, 1231.22308000000, 1450.53867000000, 1709.16554500000, 2013.86779200000, 2372.52358500000, 2795.15968800000, 3292.99652700000, 3878.92566000000, 4569.10131700000, 5382.06505800000, 6339.70069300000, 7465.56316200000, 8791.72220000000, 10352.2369750000, 12188.8856750000, 14347.8232500000, 16887.9370500000, 19872.0933000000, 23380.6652750000, 27504.3713250000, 32364.9650250000, 38095.3408000000, 44847.4916750000, 52819.5656250000, 62227.2176750000, 73321.1169500000]
        B_Arnon5 = [0, 0.0654248125493027, 0.0748613131259592, 0.0852200097732390, 0.0964406675582732, 0.108404414030963, 0.120978202862830, 0.133981410558774, 0.147324354453074, 0.161128351463696, 0.175902377184132, 0.193526821151857, 0.285794748353625, 0.411139883513949, 0.532912618951425, 0.658948953940289, 0.787463844307836, 0.911019620277348, 1.01134216103736, 1.09097860155578, 1.15946725009315, 1.21577636425715, 1.26636706123955, 1.29966244236095, 1.32941739086224, 1.35630922421149, 1.37375630182574, 1.39003487040401, 1.41548927346395, 1.43257623013269, 1.44423937756642, 1.45969672805890, 1.47405771023894, 1.48651531058339, 1.49890498452922, 1.51343941451204, 1.52867783835158, 1.54216506561365, 1.55323686869400, 1.56223503150867, 1.56963683394210, 1.57600636116484, 1.58332795425880, 1.59306861236599, 1.60529276088440, 1.61939615147952, 1.63357053682375, 1.64622605475232, 1.65658227422276, 1.66426678010510, 1.66992280459884, 1.67585542605930, 1.68316554465867, 1.69199548893857, 1.70235212334602, 1.71387033561736, 1.72578827760282]

        def BH_lookup(B_list, H_list, your_B):
            if your_B<=0:
                print 'positive B only'
                return None
            for ind, B in enumerate(B_list):
                if your_B > B:
                    continue
                elif your_B <= B:
                    return (your_B - B_list[ind-1]) / (B-B_list[ind-1]) * (H_list[ind] - H_list[ind-1]) + H_list[ind-1]

                if ind == len(B_list)-1:
                    slope = (H_list[ind]-H_list[ind-1]) / (B-B_list[ind-1]) 
                    return (your_B - B) * slope + H_list[ind]


        stator_tooth_field_strength_H = BH_lookup(B_Arnon5, H_Arnon5, stator_tooth_flux_density_B_ds)
        rotor_tooth_field_strength_H = BH_lookup(B_Arnon5, H_Arnon5, rotor_tooth_flux_density_B_dr)
        mu0 = 4*pi*10e-7
        air_gap_field_strength_H = air_gap_flux_density_B / mu0
        print 'stator_tooth_field_strength_H=', stator_tooth_field_strength_H, 'A/m'
        print 'rotor_tooth_field_strength_H=', rotor_tooth_field_strength_H, 'A/m'
        print 'air_gap_field_strength_H=', air_gap_field_strength_H, 'A/m'
        # for B, H in zip(B_Arnon5, H_Arnon5):
        #     print B, H

        stator_tooth_magnetic_voltage_Um_ds = stator_tooth_field_strength_H*stator_tooth_height_h_ds
        print 'stator_tooth_magnetic_voltage_Um_ds=', stator_tooth_magnetic_voltage_Um_ds, 'A'
        rotor_tooth_magnetic_voltage_Um_dr = rotor_tooth_field_strength_H*rotor_tooth_height_h_dr
        print 'rotor_tooth_magnetic_voltage_Um_dr=', rotor_tooth_magnetic_voltage_Um_dr, 'A'

        angle_stator_slop_open = 0.2*(360/Qs) / 180 * pi # 参考Chiba的电机槽的开口比例
        b1 = angle_stator_slop_open * stator_inner_radius_r_s
        kappa = b1/air_gap_length_delta / (5 + b1/air_gap_length_delta)
        Carter_factor_kCs = stator_slot_pitch_tau_us / (stator_slot_pitch_tau_us - kappa * b1)

        angle_rotor_slop_open = 0.1*(360/Qr) / 180 * pi # 参考Chiba的电机槽的开口比例。Gerada11建议不要开口。
        b1 = angle_rotor_slop_open * rotor_outer_radius_r_or
        kappa = b1/air_gap_length_delta / (5+b1/air_gap_length_delta) 
        Carter_factor_kCr = rotor_slot_pitch_tau_ur / (rotor_slot_pitch_tau_ur - kappa * b1)

        Carter_factor_kC = Carter_factor_kCs * Carter_factor_kCr
        print 'Carter_factor_kC=', Carter_factor_kC
        air_gap_length_delta_eff = Carter_factor_kC * air_gap_length_delta
        print 'air_gap_length_delta_eff=', air_gap_length_delta_eff*1e3, 'mm (delta=', air_gap_length_delta*1e3, 'mm)'
        air_gap_magnetic_voltage_Um_delta = air_gap_field_strength_H * air_gap_length_delta_eff
        print 'air_gap_magnetic_voltage_Um_delta=', air_gap_magnetic_voltage_Um_delta, 'A'


        #Arnon7
        coef_eddy = 0.07324; # Eddy current coefficient in (Watt/(meter^3 * T^2 * Hz^2)
        coef_hysteresis = 187.6; # Hysteresis coefficient in (Watts/(meter^3 * T^2 * Hz)

        volume_stator_tooth = stator_tooth_width_b_ds * stator_tooth_height_h_ds * stack_length
        volume_rotor_tooth = rotor_tooth_width_b_dr * rotor_tooth_height_h_dr * stack_length

        stator_tooth_core_loss_power_per_meter_cube = 1.8 * coef_hysteresis* 2*pi*rated_frequency * stator_tooth_flux_density_B_ds**2 + coef_eddy* (2*pi*rated_frequency * stator_tooth_flux_density_B_ds) **2  # table 3.2
        stator_tooth_core_loss_power = Qs* volume_stator_tooth * stator_tooth_core_loss_power_per_meter_cube
        print 'stator_tooth_core_loss_power=', stator_tooth_core_loss_power, 'W'

        rotor_tooth_core_loss_power_per_meter_cube = 1.8 * coef_hysteresis* 2*pi*rated_frequency * rotor_tooth_flux_density_B_dr**2 + coef_eddy* (2*pi*rated_frequency * rotor_tooth_flux_density_B_dr) **2 
        rotor_tooth_core_loss_power = Qr* volume_rotor_tooth * rotor_tooth_core_loss_power_per_meter_cube
        print 'rotor_tooth_core_loss_power=', rotor_tooth_core_loss_power, 'W'


        print '''\n13. Saturation Factor '''

        saturation_factor_k_sat = (stator_tooth_magnetic_voltage_Um_ds + rotor_tooth_magnetic_voltage_Um_dr) / air_gap_magnetic_voltage_Um_delta
        print 'saturation_factor_k_sat=', saturation_factor_k_sat

        tick = 1/60 # Figure 7.2
        k_sat_list =   [   0, 2.5*tick, 5.5*tick, 9.5*tick, 14.5*tick, 20*tick, 28*tick, 37.5*tick, 52.5*tick, 70*tick, 100*tick]
        alpha_i_list = [2/pi,     0.66,     0.68,     0.70,      0.72,    0.74,    0.76,      0.78,      0.80,    0.82,     0.84]

        def alpha_i_lookup(k_sat_list, alpha_i_list, your_k_sat):
            if your_k_sat<=0:
                print 'positive k_sat only'
                return None 
            for ind, k_sat in enumerate(k_sat_list):
                if your_k_sat > k_sat:
                    continue
                elif your_k_sat <= k_sat:
                    return (your_k_sat - k_sat_list[ind-1]) / (k_sat-k_sat_list[ind-1]) * (alpha_i_list[ind] - alpha_i_list[ind-1]) + alpha_i_list[ind-1]

                if ind == len(k_sat_list)-1:
                    slope = (alpha_i_list[ind]-alpha_i_list[ind-1]) / (k_sat-k_sat_list[ind-1]) 
                    return (your_k_sat - k_sat) * slope + alpha_i_list[ind]


        alpha_i_next = alpha_i_lookup(k_sat_list, alpha_i_list, saturation_factor_k_sat)
        print 'alpha_i_next=', alpha_i_next, 'alpha_i=', alpha_i

        if abs(alpha_i_next - alpha_i)< 1e-3:
            print 'alpha_i converges for %d loop' % (loop_count)
            break
        else:
            print 'loop for alpha_i_next'
            alpha_i = alpha_i_next
            continue


    print '''\n14. Yoke Geometry (its Magnetic Voltage cannot be caculated because Dse is still unknown) '''

    stator_yoke_flux_density_Bys = 1.2
    rotor_yoke_flux_density_Byr = 1.1

    # compute this again for the new alpha_i and new air_gap_flux_density_B
    air_gap_flux_Phi_m = alpha_i * air_gap_flux_density_B * pole_pitch_tau_p * stack_length_eff
    stator_yoke_height_h_ys = 0.5*air_gap_flux_Phi_m / (lamination_stacking_factor_kFe*stack_length*stator_yoke_flux_density_Bys)
    print 'stator_yoke_height_h_ys=', stator_yoke_height_h_ys*1e3, 'mm'

    rotor_yoke_height_h_yr = 0.5*air_gap_flux_Phi_m / (lamination_stacking_factor_kFe*stack_length*rotor_yoke_flux_density_Byr)
    print 'rotor_yoke_height_h_yr=', rotor_yoke_height_h_yr*1e3, 'mm'





    print '''\n15. Machine Main Geometry and Total Magnetic Voltage '''
    stator_yoke_diameter_Dsyi = stator_inner_diameter_Ds + 2*stator_tooth_height_h_ds
    stator_outer_diameter_Dse = stator_yoke_diameter_Dsyi + 2*stator_yoke_height_h_ys
    print 'stator_outer_diameter_Dse=', stator_outer_diameter_Dse*1e3, 'mm'
    print 'Yegu Kang: stator_outer_diameter fixed at 250 mm '

    rotor_yoke_diameter_Dryi = rotor_outer_diameter_Dr - 2*rotor_tooth_height_h_dr
    rotor_inner_diameter_Dri = rotor_yoke_diameter_Dryi - 2*rotor_yoke_height_h_yr
    print 'rotor_inner_diameter_Dri=', rotor_inner_diameter_Dri*1e3, 'mm'



    By_list     = [    0,  0.5,   0.6, 0.7,   0.8,  0.9, 0.95,  1.0, 1.08,   1.1, 1.2,   1.3,   1.4, 1.5,  1.6,  1.7,   1.8,  1.9,   2.0] # Figure 3.17
    coef_c_list = [ 0.72, 0.72, 0.715, 0.7, 0.676, 0.62,  0.6, 0.56,  0.5, 0.485, 0.4, 0.325, 0.255, 0.2, 0.17, 0.15, 0.142, 0.13, 0.125]


    def coef_c_lookup(By_list, coef_c_list, your_By):
        if your_By<=0:
            print 'positive By only'
            return None 
        for ind, By in enumerate(By_list):
            if your_By > By:
                continue
            elif your_By <= By:
                return (your_By - By_list[ind-1]) / (By-By_list[ind-1]) * (coef_c_list[ind] - coef_c_list[ind-1]) + coef_c_list[ind-1]

            if ind == len(By_list)-1:
                slope = (coef_c_list[ind]-coef_c_list[ind-1]) / (By-By_list[ind-1]) 
                return (your_By - By) * slope + coef_c_list[ind]

    stator_yoke_field_strength_Hys = BH_lookup(B_Arnon5, H_Arnon5, stator_yoke_flux_density_Bys)
    stator_yoke_middle_pole_pitch_tau_ys = pi*(stator_outer_diameter_Dse - stator_yoke_height_h_ys) / (2*no_pole_pairs)
    coef_c = coef_c_lookup(By_list, coef_c_list, stator_yoke_flux_density_Bys)
    print 'coef_c', coef_c
    stator_yoke_magnetic_voltage_Um_ys = coef_c * stator_yoke_field_strength_Hys * stator_yoke_middle_pole_pitch_tau_ys
    print 'stator_yoke_magnetic_voltage_Um_ys=', stator_yoke_magnetic_voltage_Um_ys, 'A'



    rotor_yoke_field_strength_Hys = BH_lookup(B_Arnon5, H_Arnon5, rotor_yoke_flux_density_Byr)
    rotor_yoke_middle_pole_pitch_tau_yr = pi*(rotor_yoke_diameter_Dryi - rotor_yoke_height_h_yr) / (2*no_pole_pairs) # (3.53b)
    coef_c = coef_c_lookup(By_list, coef_c_list, rotor_yoke_flux_density_Byr)
    print 'coef_c', coef_c
    rotor_yoke_magnetic_voltage_Um_yr = coef_c * rotor_yoke_field_strength_Hys * rotor_yoke_middle_pole_pitch_tau_yr
    print 'rotor_yoke_magnetic_voltage_Um_yr=', rotor_yoke_magnetic_voltage_Um_yr, 'A'



    print '''\n16. Magnetizing Current Providing Total Magnetic Voltage '''
    total_magnetic_voltage_Um_tot = air_gap_magnetic_voltage_Um_delta + stator_tooth_magnetic_voltage_Um_ds + rotor_tooth_magnetic_voltage_Um_dr \
                                    + 0.5*stator_yoke_magnetic_voltage_Um_ys + 0.5*rotor_yoke_magnetic_voltage_Um_yr
    print 'total_magnetic_voltage_Um_tot (half magnetic circuit) =', total_magnetic_voltage_Um_tot, 'A'
    stator_magnetizing_current_Is_mag = total_magnetic_voltage_Um_tot * pi* no_pole_pairs / (no_phase_m * kw1 * no_series_coil_turns_N * sqrt(2))
    print 'stator_magnetizing_current_Is_mag=', stator_magnetizing_current_Is_mag, 'A'



    print '''\n17. Losses Computation and Efficiency '''




    # fig, axes = subplots(1,3, dpi=80)
    # ax = axes[0]
    # ax.plot(H_Arnon5, B_Arnon5)
    # ax.set_xlabel(r'$H [A/m]$')
    # ax.set_ylabel(r'$B [T]$')
    # ax.grid()
    # ax = axes[1]
    # ax.plot(k_sat_list, alpha_i_list)
    # ax.set_xlabel(r'$k_{sat}$')
    # ax.set_ylabel(r'$\alpha_i$')
    # ax.grid()
    # ax = axes[2]
    # ax.plot(By_list, coef_c_list)
    # ax.set_xlabel(r'$B_y$ [T]')
    # ax.set_ylabel(r'$c$')
    # ax.grid()
    # show()

    print '''\n[Export to %d-entry ./pop/xxx.txt] Geometry for Plotting and Its Constraints ''' %(THE_IM_DESIGN_ID)

    Qs # number of stator slots
    Qr # number of rotor slots
    Angle_StatorSlotSpan = 360 / Qs # in deg.
    Angle_RotorSlotSpan = 360 / Qr # in deg.

    Radius_OuterStatorYoke  = 0.5*stator_outer_diameter_Dse * 1e3
    Radius_InnerStatorYoke  = 0.5*stator_yoke_diameter_Dsyi * 1e3
    Length_AirGap           = air_gap_length_delta * 1e3
    Radius_OuterRotor       = 0.5*rotor_outer_diameter_Dr * 1e3
    Radius_Shaft            = 0.5*rotor_inner_diameter_Dri * 1e3

    Length_HeadNeckRotorSlot = 1.0 # mm # 这里假设与HeadNeck相对的槽的部分也能放导体了。准确来说应该有：rotor_inner_diameter_Dri = rotor_yoke_diameter_Dryi - 2*rotor_yoke_height_h_yr - 2*1e-3*Length_HeadNeckRotorSlot

    rotor_slot_radius = (2*pi*(Radius_OuterRotor - Length_HeadNeckRotorSlot)*1e-3 - rotor_tooth_width_b_dr*Qr) / (2*Qr+2*pi)
    print 'Rotor Slot Radius:', rotor_slot_radius * 1e3, rotor_tooth_width_b_dr * 1e3

    Radius_of_RotorSlot = rotor_slot_radius*1e3 
    Location_RotorBarCenter = Radius_OuterRotor - Length_HeadNeckRotorSlot - Radius_of_RotorSlot
    Width_RotorSlotOpen = b1*1e3 # 10% of 360/Qr

    Location_RotorBarCenter2 = Radius_OuterRotor - Length_HeadNeckRotorSlot - rotor_tooth_height_h_dr*1e3 # 本来还应该减去转子内槽半径的，但是这里还不知道，干脆不要了，这样槽会比预计的偏深，
    if abs(Location_RotorBarCenter2 - Location_RotorBarCenter) < Radius_of_RotorSlot*0.25:
        print 'There is no need to use a drop shape rotor, because the required rotor bar height is not high.'
    print 'the width of outer rotor slot: %g' % (Radius_of_RotorSlot)
    print 'the height of total rotor slot: %g' % (rotor_tooth_height_h_dr*1e3)
    Arc_betweenOuterRotorSlot = 360/Qr*pi/180*Location_RotorBarCenter - 2*Radius_of_RotorSlot
    Radius_of_RotorSlot2 = 0.5 * (360/Qr*pi/180*Location_RotorBarCenter2 - Arc_betweenOuterRotorSlot) # 应该小于等于这个值，保证转子齿等宽。

    Angle_StatorSlotOpen = angle_stator_slop_open / pi *180 # in deg.
    Width_StatorTeethBody = stator_tooth_width_b_ds*1e3
    Width_StatorTeethHeadThickness = 1 # mm # 这里假设与齿头、齿脖相对的槽的部分也能放导体了。准确来说应该stator_yoke_diameter_Dsyi = stator_inner_diameter_Ds + 2*stator_tooth_height_h_ds + 2*1e-3*(Width_StatorTeethHeadThickness+Width_StatorTeethNeck)
    Width_StatorTeethNeck = 0.5 # mm 



    DriveW_Rs = 0.1 # 按照书上的公式算一下

    DriveW_poles=no_pole_pairs*2
    DriveW_turns=no_conductors_per_slot_zQ
    DriveW_Rs = 0.1
    DriveW_CurrentAmp = stator_phase_current_rms * sqrt(2); 
    DriveW_Freq = rated_frequency


    with open(loc_txt_file, 'a') as f:
        f.write('%d, '%(THE_IM_DESIGN_ID))
        f.write('%d, %d, '%(Qs, Qr))
        f.write('%f, %f, %f, %f, %f, ' % (Radius_OuterStatorYoke, Radius_InnerStatorYoke, Length_AirGap, Radius_OuterRotor, Radius_Shaft))
        f.write('%f, %f, %f, %f, %f, %f, ' % (Length_HeadNeckRotorSlot,Radius_of_RotorSlot, Location_RotorBarCenter, Width_RotorSlotOpen, Radius_of_RotorSlot2, Location_RotorBarCenter2))
        f.write('%f, %f, %f, %f, ' % (Angle_StatorSlotOpen, Width_StatorTeethBody, Width_StatorTeethHeadThickness, Width_StatorTeethNeck))
        f.write('%f, %f, %f, %f, %f, %f,' % (DriveW_poles, DriveW_turns, DriveW_Rs, DriveW_CurrentAmp, DriveW_Freq, stack_length*1000))
        f.write('%f, %f\n' % (area_stator_slot_Sus, area_rotor_slot_Sur)) # this line exports values need to impose constraints among design parameters for the de optimization




    ''' Determine bounds for these parameters:
        stator_tooth_width_b_ds       = design_parameters[0]*1e-3 # m                       # stator tooth width [mm]
        air_gap_length_delta          = design_parameters[1]*1e-3 # m                       # air gap length [mm]
        b1                            = design_parameters[2]*1e-3 # m                       # rotor slot opening [mm]
        rotor_tooth_width_b_dr        = design_parameters[3]*1e-3 # m                       # rotor tooth width [mm]
        self.Length_HeadNeckRotorSlot        = design_parameters[4]             # [4]       # rotor tooth head & neck length [mm]
        self.Angle_StatorSlotOpen            = design_parameters[5]             # [5]       # stator slot opening [deg]
        self.Width_StatorTeethHeadThickness  = design_parameters[6]             # [6]       # stator tooth head length [mm]
    '''
    print 'stator_tooth_width_b_ds =', stator_tooth_width_b_ds
    print 'air_gap_length_delta =', air_gap_length_delta         
    print 'b1 =', b1
    print 'rotor_tooth_width_b_dr =', rotor_tooth_width_b_dr
    print 'Length_HeadNeckRotorSlot =', Length_HeadNeckRotorSlot
    print 'Angle_StatorSlotOpen =', Angle_StatorSlotOpen
    print 'Width_StatorTeethHeadThickness =', Width_StatorTeethHeadThickness











'''
1. 按照这本书Pyrhonen2009根据Eric的要求50kW30000rpm设计一个初始电机。
2. 用瞬态场，遍历短距、斜槽、转子槽数，对转矩脉动和悬浮力脉动的影响。

注意，必须先确定短距和转子槽数才能继续优化槽型啊！

3. 根据确定好的短距和转子槽数，构造涡流场，优化电机的槽型和气隙长度，提高电机的转矩输出性能和效率。
4. 最后再按最终设计，进行瞬态场验证。
'''



'mechanical limits'

if False:
    rotor_radius = 0.075
    speed_rpm = 20000

    Omega = speed_rpm/(60)*2*pi
    modulus_of_elasticity = 190 * 10e9 # Young's modulus
    D_out = rotor_radius * 2
    D_in = 0
    second_moment_of_inertia_of_area_I = pi*(D_out**4 - D_in**4) / 64 
    stack_length_max = sqrt( 1**2 * pi**2 / (1.5*Omega) * sqrt( 200*10e9 * second_moment_of_inertia_of_area_I / (8760* pi*(D_out/2)**2) ) )
    print 'stack_length_max=', stack_length_max

    stack_length_max = sqrt(1*pi**2/(1.5*20000/60*2*pi)*sqrt(200*10e9*pi*0.15**4/64/(8760*pi*0.15**2/4)))
    print 'stack_length_max=', stack_length_max

    speed_rpm = 30000
    Omega = speed_rpm/(60)*2*pi
    C_prime = (3+0.29)/4
    rotor_radius_max = sqrt(300e6/(C_prime*8760*Omega**2))
    print 'rotor_radius_max', rotor_radius_max
    print 'rotor_diameter_max', 2*rotor_radius_max
