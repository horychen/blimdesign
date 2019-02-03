#coding:u8
from __future__ import division
from pylab import *
import sys
import os

print 'Pyrhonen 2009 Chapter 7.'
print 'Guesses: alpha_i, efficiency, power_factor.'

# model_name_prefix = 'Qr_loop'
# model_name_prefix = 'Qr_loop_B' # longer solve time, less models, better mesh, higher turns for bearing winding.
# model_name_prefix = 'EC_Rotate_PS' # longer solve time, less models, better mesh, higher turns for bearing winding.
# model_name_prefix = 'ECRot_PS_Opti' # longer solve time, less models, better mesh, higher turns for bearing winding.
# model_name_prefix = 'StaticFEA_PS_Opti' # Fix Bug for the rotor slot radius as half of rotor tooth width
model_name_prefix = 'Tran2TSS_PS_Opti'
model_name_prefix = 'Tran2TSS_PS_Opti_Qr16'

# delete existing file
loc_txt_file = '../pop/%s.txt'%(model_name_prefix)
open(loc_txt_file, 'w').close()


# @blockPrinting
def pyrhonen_blim_design(rotor_tooth_flux_density_B_dr, stator_tooth_flux_density_B_ds, rotor_current_density_Jr):
    # for tangential_stress in arange(12000, 33001, 3000):
    # tangential_stress - this value will change the motor size, so it is not suited for loop for optimization bounds

    # # debug    
    # if Qr == 32:
    #     continue

    THE_IM_DESIGN_ID = Qr

    print '''\n1. Initial Design Parameters '''
    mec_power = 50e3 # W
    rated_frequency = 1000 # Hz
    no_pole_pairs = 2
    speed_rpm = rated_frequency * 60 / no_pole_pairs # rpm
    U1_rms = 500 / sqrt(3) # V - Wye-connect #480 is standarnd
    # U1_rms = 500  # V - Delta-connect
    stator_phase_voltage_rms = U1_rms
    no_phase_m = 3
    print 'rated_frequency=', rated_frequency
    print 'speed_rpm=', speed_rpm
    bool_we_have_plenty_voltage = True
    print 'The default option for voltage avaliability is plenty. That is, the dc bus can be high enough (higher than the specified U1_rms) for allowing more conductors in stator slots.'
    bool_pole_specific_rotor = True
    if bool_pole_specific_rotor==True:
        print 'Pole-specific wound rotor.'
    else:
        print 'Cage rotor.'
    print 



    print '''\n2. Machine Constants '''
    if bool_pole_specific_rotor == False:
        tangential_stress = 21500 # 12000 ~ 33000
    else: # for wound rotor, the required rotor slot is really too large to find a solution, we must reduce tangential_stress to make a larger rotor.    
        tangential_stress = 12000 # larger rotor to increase rotor slot area and to reduce rotor heating.

    if no_pole_pairs == 1:
        machine_constant_Cmec = 150 # kW s / m^3. Figure 6.3 <- This is what Eric calls the penalty on the 2 pole IM---you loss power density.
    else:
        machine_constant_Cmec = 250 # kW s / m^3. 这里的电机常数都是按100kW去查的表哦，偏大了。不过电机常数在这里只用于检查AJ值的合理性。
    print 'tangential_stress=', tangential_stress
    print 'machine_constant_Cmec=', machine_constant_Cmec
    print 'The former is used to determine motor size, and the latter is for determining linear current density A according to the air gap B.'


    print '''\n3. Machine Sizing '''
    required_torque = mec_power/(2*pi*speed_rpm)*60
    print 'required_torque', required_torque, 'Nm'

    rotor_volume_Vr = required_torque/(2*tangential_stress)

    length_ratio_chi = pi/(2*no_pole_pairs) * no_pole_pairs**(1/3.) # Table 6.5
    print 'length_ratio_chi=', length_ratio_chi

    rotor_outer_diameter_Dr = (4/pi*rotor_volume_Vr*length_ratio_chi)**(1/3.)
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
    print 'air_gap_length_delta=', air_gap_length_delta*1e3, 'mm,', 'but this is for 50 Hz line-start IM.'
    print 'Kevin S. Campbell: this is too small. 3.5 mm is good!'
    air_gap_length_delta *= 2 # *=3 will not converge
    print 'air_gap_length_delta=', air_gap_length_delta*1e3, 'mm.', 'Double it for high speed inverted driven IM.'
    

    stack_length_eff = stack_length + 2 * air_gap_length_delta
    print 'stack_length_eff=', stack_length_eff*1e3, 'mm'

    air_gap_diameter_D = 1*air_gap_length_delta + rotor_outer_diameter_Dr
    stator_inner_diameter_Dis = 2*air_gap_length_delta + rotor_outer_diameter_Dr
    stator_inner_radius_r_is = 0.5*stator_inner_diameter_Dis



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
    print 'Regular available Qs choice: k * 2*no_pole_pairs * no_phase_m. However, we have to make sure it is integral slot for the bearing winding that has two pole pairs'
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
    else: # short pitch (not tested)
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
    print 'winding factor:', kw1, '=', kd1, '*', kq1, '*', ksq1


    # Qr = 30 # Table 7.5. If Qs=24, then Qr!=28, see (7.113).



    print '''\n6. Air Gap Flux Density '''
    air_gap_flux_density_B = 0.8 # 0.7 ~ 0.9 Table 6.3
    print 'air_gap_flux_density_B=', air_gap_flux_density_B
    linear_current_density_A = machine_constant_Cmec / (pi**2/sqrt(2)*kw1*air_gap_flux_density_B)
    
    if linear_current_density_A<65 and linear_current_density_A>30:
        print 'linear_current_density_A=', linear_current_density_A, 'kA/m'
    else:
        raise Exception('Bad linear_current_density_A.')

    print '''\n7. Number of Coil Turns '''
    desired_emf_Em = 0.95 * U1_rms # 0.96~0.98, high speed motor has higher leakage reactance hence 0.95
    flux_linkage_Psi_m = desired_emf_Em / (2*pi*rated_frequency) 

    alpha_i = 2/pi # ideal sinusoidal flux density distribusion, when the saturation happens in teeth, alpha_i becomes higher.
    air_gap_flux_Phi_m = alpha_i * air_gap_flux_density_B * pole_pitch_tau_p * stack_length_eff
    no_series_coil_turns_N = sqrt(2)*desired_emf_Em / (2*pi*rated_frequency * kw1 * air_gap_flux_Phi_m) # p306
    print 'no_series_coil_turns_N=', no_series_coil_turns_N


    print '''\n8. Round Up Number of Coil Turns '''
    no_series_coil_turns_N = round(no_series_coil_turns_N)
    print 'no_series_coil_turns_N=', no_series_coil_turns_N

    print 'The no_series_coil_turns_N must be divisible by pq=%d' %(no_pole_pairs*distribution_q)
    print 'Remainder is', int(no_series_coil_turns_N) % int(no_pole_pairs*distribution_q)
    if bool_we_have_plenty_voltage:
        no_series_coil_turns_N = min([no_pole_pairs*distribution_q*i for i in range(100,0,-1)], key=lambda x:abs(x - no_series_coil_turns_N))
    else:
        no_series_coil_turns_N = min([no_pole_pairs*distribution_q*i for i in range(100)], key=lambda x:abs(x - no_series_coil_turns_N)) # https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
    print 'Suggested no_series_coil_turns_N is', no_series_coil_turns_N, '---modify this manually, if necessary.'

    no_parallel_path = 1
    print '''In some cases, especially in low-voltage, high-power machines, there may be a need to change the stator slot number, the number of parallel paths or even the main dimensions of the machine in order to find the appropriate number of conductors in a slot.'''
    no_conductors_per_slot_zQ = 2* no_phase_m * no_series_coil_turns_N /Qs * no_parallel_path
    print 'no_conductors_per_slot_zQ=', no_conductors_per_slot_zQ

    loop_count = 0
    global BH_lookup, bdata, hdata
    while True:
    # if True:
        if loop_count>25:
            raise Exception("Abort. This script may not converge anyway.")
        loop_count += 1
        print '''\n9. Recalculate Air Gap Flux Density '''
        air_gap_flux_density_B = sqrt(2)*desired_emf_Em / (2*pi*rated_frequency * kw1 *  alpha_i * no_series_coil_turns_N * pole_pitch_tau_p * stack_length_eff) # p306
        print 'air_gap_flux_density_B=', air_gap_flux_density_B, 'T', '变得比0.8T大了，是因为你减少了匝数取整，反之亦然。'



        print '''\n10. Tooth Flux Density '''
        # stator_tooth_flux_density_B_ds = 1.4 #1.4–2.1 (stator) 
        # rotor_tooth_flux_density_B_dr = 1.5 #1.8 #1.5–2.2 (rotor)
        print 'stator_tooth_flux_density_B_ds=' , stator_tooth_flux_density_B_ds
        print 'rotor_tooth_flux_density_B_dr=' , rotor_tooth_flux_density_B_dr
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
        power_factor =0.6  #0.85
        print 'power_factor=', power_factor

        stator_phase_current_rms = mec_power / (no_phase_m*efficiency*stator_phase_voltage_rms*power_factor)
        print 'stator_phase_current_rms', stator_phase_current_rms, 'Arms'

        rotor_current_referred = stator_phase_current_rms * power_factor
        rotor_current_actual = no_conductors_per_slot_zQ / no_parallel_path * Qs / Qr * rotor_current_referred
        print 'rotor current:', rotor_current_referred, '(referred)', rotor_current_actual, '(actual) Arms'

    
        # Current density (Pyrhonen09@Example6.4)

        stator_current_density_Js = 3.7e6 # typical value is 3.7e6 Arms/m^2 = stator_current_density_Js
        my_AJ_value = stator_current_density_Js*(linear_current_density_A*1e3)   # \in [9e10, 52e10]
        print '\nstator_current_density_Js=', stator_current_density_Js, 'A/m^2'
        if my_AJ_value*1e-10 < 52 and my_AJ_value > 9:
            print '\tmy_AJ_value is', my_AJ_value*1e-10, 'e10 A^2/m^3. It is \\in [9e10, 52e10].'
        else:
            print '\tmy_AJ_value is', my_AJ_value*1e-10, 'e10 A^2/m^3.'
            raise Exception('The liear or slot current density is bad.')
        area_conductor_stator_Scs = stator_phase_current_rms / (no_parallel_path * stator_current_density_Js)
        print 'area_conductor_stator_Scs=', area_conductor_stator_Scs * 1e6, 'mm^2'

        # space factor or slot packing factor
        space_factor_kCu = 0.50 # 不计绝缘的导体填充率：也就是说，一般来说槽满率能达到60%-66%，但是，这里用于下式计算的，要求考虑导体，而槽满率已经考虑了细导线的绝缘了，所以space factor会比槽满率更小，一般在0.5-0.6，低压电机则取下限0.5。
        print 'space_factor_kCu=', space_factor_kCu
        area_stator_slot_Sus = no_conductors_per_slot_zQ * area_conductor_stator_Scs / space_factor_kCu 
        print 'area_stator_slot_Sus', area_stator_slot_Sus*1e6, 'mm^2'

        # guess these local design values or adapt from other designs
        width_statorTeethHeadThickness = 1e-3 # m
        width_StatorTeethNeck = 0.5 * width_statorTeethHeadThickness

        # stator slot height
        stator_inner_radius_r_is_eff = stator_inner_radius_r_is + (width_statorTeethHeadThickness + width_StatorTeethNeck)
        temp = (2*pi*stator_inner_radius_r_is_eff - Qs*stator_tooth_width_b_ds)
        stator_tooth_height_h_ds = ( sqrt(temp**2 + 4*pi*area_stator_slot_Sus*Qs) - temp ) / (2*pi)
        print 'stator_tooth_height_h_ds', stator_tooth_height_h_ds*1e3, 'mm\n'
        # The slot depth equals the tooth height hs = hd Pyrhonen09@p309
        stator_slot_height_h_ss = stator_tooth_height_h_ds
        print 'stator_slot_height_h_ss', stator_slot_height_h_ss*1e3, 'mm\n'

        # we move this loop outside
        # for rotor_current_density_Jr in arange(3e6, 8e6+1, 1e6):
        if True:
            # rotor_current_density_Jr = 8e6 # 8e6 for Cu # 6.5e6 for Al # 减少电流密度有效减少转子欧姆热 # 修正4*pi*area_rotor_slot_Sur*Qr处的BUG前，这里设置的转子电流密度都是没用的，和FEA的结果对不上，现在能对上了（用FEMM积分电流和面积验证）！
            print 'rotor_current_density_Jr=', rotor_current_density_Jr, 'A/m^2'
            area_conductor_rotor_Scr = rotor_current_actual / (1 * rotor_current_density_Jr) # no_parallel_path=1
            print 'area_conductor_rotor_Scr=', area_conductor_rotor_Scr * 1e6, 'mm^2'

            if bool_pole_specific_rotor == True:
                # space_factor_kAl = 0.6 # We use wound-rotor to implement Chiba's pole specific rotor. That is, it is coils.
                # space_factor_kAl = 0.95 # (debug)
                space_factor_kAl = 1.0
                print 'space_factor_kAl=', space_factor_kAl
            else:
                space_factor_kAl = 1 # no clearance for die cast aluminium bar
                                     # However, if a cage winding is produced from copper bars by soldering, a clearance of about 0.4mm in width and 1mm in height has to be left in the rotor slot. This clearance also decreases the space factor.
            # no_conductors_per_slot_zQ (z_Qr) for cage rotor is clearly 1. 
            # Even though you use coils in rotor (wound rotor), since all the coils within a slot are parallel connected.
            # This will give give a no_parallel_path equal to z_Qr, so they will cancel out, anyway. 
            # Just let both no_parallel_path and z_Qr to be 1 and don't bother.
            area_rotor_slot_Sur = 1 * area_conductor_rotor_Scr / space_factor_kAl 
            print 'area_rotor_slot_Sur', area_rotor_slot_Sur*1e6, 'mm^2'

            # guess this local design values or adapt from other designs
            length_headNeckRotorSlot = 1e-3 # m

            # rotor slot height depends on rotor_tooth_width_b_dr and rotor current (power factor)
            rotor_outer_radius_r_or_eff = rotor_outer_radius_r_or - length_headNeckRotorSlot
            temp = (2*pi*rotor_outer_radius_r_or_eff - Qr*rotor_tooth_width_b_dr)
            rotor_tooth_height_h_dr = ( +sqrt(temp**2 - 4*pi*area_rotor_slot_Sur*Qr) + temp ) / (2*pi)
            print 'rotor_tooth_height_h_dr(+)=', rotor_tooth_height_h_dr*1e3, 'mm' # too large to be right answer
            rotor_tooth_height_h_dr = ( -sqrt(temp**2 - 4*pi*area_rotor_slot_Sur*Qr) + temp ) / (2*pi)
            print 'rotor_tooth_height_h_dr(-)=', rotor_tooth_height_h_dr*1e3, 'mm'
            if isnan(rotor_tooth_height_h_dr) == True:
                bool_enough_rotor_slot_space = False
                'Loop for working rotor_current_density_Jr, not', rotor_current_density_Jr
                # continue
                # raise Exception('There are no space on the rotor to fulfill the required rotor slot to reach your J_r and space_factor_kAl. Reduce your tangential_stress and run the script again.')
                return
            else:
                bool_enough_rotor_slot_space = True
                # print 'The working rotor_current_density_Jr is', rotor_current_density_Jr
                # break

        if bool_enough_rotor_slot_space == False:
            raise Exception('Lower your tangential_stress. If that is not possible, increase your B in the rotor tooth because rotor iron loss is of slip freqnecy---it is not that 怖い.')

        # The slot depth equals the tooth height hs = hd Pyrhonen09@p309
        rotor_slot_height_h_sr = rotor_tooth_height_h_dr
        print 'rotor_slot_height_h_sr', rotor_slot_height_h_sr*1e3, 'mm\n'


        minimum__area_rotor_slot_Sur = rotor_current_actual / (8e6) / space_factor_kAl
        print 'minimum__area_rotor_slot_Sur', minimum__area_rotor_slot_Sur, '<', area_rotor_slot_Sur
        minimum__rotor_tooth_height_h_dr = ( -sqrt(temp**2 - 4*pi*minimum__area_rotor_slot_Sur*Qr) + temp ) / (2*pi)
        print 'minimum__rotor_tooth_height_h_dr', minimum__rotor_tooth_height_h_dr, '<', rotor_tooth_height_h_dr
        print 'rotor_tooth_width_b_dr', rotor_tooth_width_b_dr*1e3, 'mm'
        # quit()

        print '''\n12. Magnetic Voltage '''
        bool_use_M19 = True
        if bool_use_M19 == True:
            # M19-Gauge29
            hdata, bdata = np.loadtxt('../Arnon5/M-19-Steel-BH-Curve-afterJMAGsmooth.txt', unpack=True, usecols=(0,1))
            print 'The magnetic material is M19-Gauge29.'
        else:
            # Arnon5
            hdata = [0, 9.51030700000000, 11.2124700000000, 13.2194140000000, 15.5852530000000, 18.3712620000000, 21.6562210000000, 25.5213000000000, 30.0619920000000, 35.3642410000000, 41.4304340000000, 48.3863030000000, 56.5103700000000, 66.0660360000000, 77.3405760000000, 90.5910260000000, 106.212089000000, 124.594492000000, 146.311191000000, 172.062470000000, 202.524737000000, 238.525598000000, 281.012026000000, 331.058315000000, 390.144609000000, 459.695344000000, 541.731789000000, 638.410494000000, 752.333643000000, 886.572927000000, 1044.77299700000, 1231.22308000000, 1450.53867000000, 1709.16554500000, 2013.86779200000, 2372.52358500000, 2795.15968800000, 3292.99652700000, 3878.92566000000, 4569.10131700000, 5382.06505800000, 6339.70069300000, 7465.56316200000, 8791.72220000000, 10352.2369750000, 12188.8856750000, 14347.8232500000, 16887.9370500000, 19872.0933000000, 23380.6652750000, 27504.3713250000, 32364.9650250000, 38095.3408000000, 44847.4916750000, 52819.5656250000, 62227.2176750000, 73321.1169500000]
            bdata = [0, 0.0654248125493027, 0.0748613131259592, 0.0852200097732390, 0.0964406675582732, 0.108404414030963, 0.120978202862830, 0.133981410558774, 0.147324354453074, 0.161128351463696, 0.175902377184132, 0.193526821151857, 0.285794748353625, 0.411139883513949, 0.532912618951425, 0.658948953940289, 0.787463844307836, 0.911019620277348, 1.01134216103736, 1.09097860155578, 1.15946725009315, 1.21577636425715, 1.26636706123955, 1.29966244236095, 1.32941739086224, 1.35630922421149, 1.37375630182574, 1.39003487040401, 1.41548927346395, 1.43257623013269, 1.44423937756642, 1.45969672805890, 1.47405771023894, 1.48651531058339, 1.49890498452922, 1.51343941451204, 1.52867783835158, 1.54216506561365, 1.55323686869400, 1.56223503150867, 1.56963683394210, 1.57600636116484, 1.58332795425880, 1.59306861236599, 1.60529276088440, 1.61939615147952, 1.63357053682375, 1.64622605475232, 1.65658227422276, 1.66426678010510, 1.66992280459884, 1.67585542605930, 1.68316554465867, 1.69199548893857, 1.70235212334602, 1.71387033561736, 1.72578827760282]
            print 'The magnetic material is Arnon5.'

     
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


        stator_tooth_field_strength_H = BH_lookup(bdata, hdata, stator_tooth_flux_density_B_ds)
        rotor_tooth_field_strength_H = BH_lookup(bdata, hdata, rotor_tooth_flux_density_B_dr)
        mu0 = 4*pi*1e-7
        air_gap_field_strength_H = air_gap_flux_density_B / mu0
        print 'stator_tooth_field_strength_H=', stator_tooth_field_strength_H, 'A/m'
        print 'rotor_tooth_field_strength_H=', rotor_tooth_field_strength_H, 'A/m'
        print 'air_gap_field_strength_H=', air_gap_field_strength_H, 'A/m'
        # for B, H in zip(bdata, hdata):
        #     print B, H

        stator_tooth_magnetic_voltage_Um_ds = stator_tooth_field_strength_H*stator_tooth_height_h_ds
        print 'stator_tooth_magnetic_voltage_Um_ds=', stator_tooth_magnetic_voltage_Um_ds, 'A'
        rotor_tooth_magnetic_voltage_Um_dr = rotor_tooth_field_strength_H*rotor_tooth_height_h_dr
        print 'rotor_tooth_magnetic_voltage_Um_dr=', rotor_tooth_magnetic_voltage_Um_dr, 'A'

        angle_stator_slop_open = 0.2*(360/Qs) / 180 * pi # 参考Chiba的电机槽的开口比例
        b1 = angle_stator_slop_open * stator_inner_radius_r_is
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

        if bool_use_M19 == True:
            #M19
            coef_eddy = 0.530 # % Eddy current coefficient in (Watt/(meter^3 * T^2 * Hz^2)
            coef_hysteresis = 143.  # % Hysteresis coefficient in (Watts/(meter^3 * T^2 * Hz)
        else:
            #Arnon7
            coef_eddy = 0.07324; # Eddy current coefficient in (Watt/(meter^3 * T^2 * Hz^2)
            coef_hysteresis = 187.6; # Hysteresis coefficient in (Watts/(meter^3 * T^2 * Hz)

        volume_stator_tooth = stator_tooth_width_b_ds * stator_tooth_height_h_ds * stack_length
        volume_rotor_tooth = rotor_tooth_width_b_dr * rotor_tooth_height_h_dr * stack_length

        # 这里在瞎算什么，人家Example 7.4用的是loss table的结果P15，不要混淆了！还有，这个2pi是什么鬼？虽然写的是omega，但是人家是指频率Hz啊！
        # stator_tooth_core_loss_power_per_meter_cube = 1.8 * coef_hysteresis* 2*pi*rated_frequency * stator_tooth_flux_density_B_ds**2 \
        #                                                   + coef_eddy* (2*pi*rated_frequency * stator_tooth_flux_density_B_ds) **2  # table 3.2
        # stator_tooth_core_loss_power = Qs* volume_stator_tooth * stator_tooth_core_loss_power_per_meter_cube
        # print 'stator_tooth_core_loss_power=', stator_tooth_core_loss_power, 'W'

        # rotor_tooth_core_loss_power_per_meter_cube = 1.8 * coef_hysteresis* 2*pi*rated_frequency * rotor_tooth_flux_density_B_dr**2 \
        #                                                  + coef_eddy* (2*pi*rated_frequency * rotor_tooth_flux_density_B_dr) **2 
        # rotor_tooth_core_loss_power = Qr* volume_rotor_tooth * rotor_tooth_core_loss_power_per_meter_cube
        # print 'rotor_tooth_core_loss_power=', rotor_tooth_core_loss_power, 'W'


        print '''\n13. Saturation Factor '''

        saturation_factor_k_sat = (stator_tooth_magnetic_voltage_Um_ds + rotor_tooth_magnetic_voltage_Um_dr) / air_gap_magnetic_voltage_Um_delta
        print 'saturation_factor_k_sat=', saturation_factor_k_sat

        tick = 1/60. # Figure 7.2
        k_sat_list =   [   0, 2.5*tick, 5.5*tick, 9.5*tick, 14.5*tick, 20*tick, 28*tick, 37.5*tick, 52.5*tick, 70*tick, 100*tick]
        alpha_i_list = [2/pi,     0.66,     0.68,     0.70,      0.72,    0.74,    0.76,      0.78,      0.80,    0.82,     0.84]

        def alpha_i_lookup(k_sat_list, alpha_i_list, your_k_sat):
            if your_k_sat<=0:
                print 'positive k_sat only'
                return None 
            for ind, k_sat in enumerate(k_sat_list):
                if your_k_sat > k_sat:
                    if ind == len(k_sat_list)-1: # it is the last one
                        slope = (alpha_i_list[ind]-alpha_i_list[ind-1]) / (k_sat-k_sat_list[ind-1]) 
                        return (your_k_sat - k_sat) * slope + alpha_i_list[ind]
                    else:
                        continue
                elif your_k_sat <= k_sat:
                    return (your_k_sat - k_sat_list[ind-1]) / (k_sat-k_sat_list[ind-1]) * (alpha_i_list[ind] - alpha_i_list[ind-1]) + alpha_i_list[ind-1]

            # these will not be reached
            print 'Reach the end of the curve of k_sat.'
            print 'End of Loop Error.\n'*3
            return None

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
    stator_yoke_diameter_Dsyi = stator_inner_diameter_Dis + 2*stator_tooth_height_h_ds
    stator_outer_diameter_Dse = stator_yoke_diameter_Dsyi + 2*stator_yoke_height_h_ys
    print 'stator_outer_diameter_Dse=', stator_outer_diameter_Dse*1e3, 'mm'
    print 'Yegu Kang: stator_outer_diameter fixed at 250 mm '

    rotor_yoke_diameter_Dryi = rotor_outer_diameter_Dr - 2*rotor_tooth_height_h_dr
    rotor_inner_diameter_Dri = rotor_yoke_diameter_Dryi - 2*rotor_yoke_height_h_yr
    print 'rotor_inner_diameter_Dri=', rotor_inner_diameter_Dri*1e3, 'mm'


    # B@yoke
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

    stator_yoke_field_strength_Hys = BH_lookup(bdata, hdata, stator_yoke_flux_density_Bys)
    stator_yoke_middle_pole_pitch_tau_ys = pi*(stator_outer_diameter_Dse - stator_yoke_height_h_ys) / (2*no_pole_pairs)
    coef_c = coef_c_lookup(By_list, coef_c_list, stator_yoke_flux_density_Bys)
    print 'coef_c', coef_c
    stator_yoke_magnetic_voltage_Um_ys = coef_c * stator_yoke_field_strength_Hys * stator_yoke_middle_pole_pitch_tau_ys
    print 'stator_yoke_magnetic_voltage_Um_ys=', stator_yoke_magnetic_voltage_Um_ys, 'A'



    rotor_yoke_field_strength_Hys = BH_lookup(bdata, hdata, rotor_yoke_flux_density_Byr)
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



    print '''\n17. Losses Computation and Efficiency]\n略。 '''




    # fig, axes = subplots(1,3, dpi=80)
    # ax = axes[0]
    # ax.plot(hdata, bdata)
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

    print '''\n[Export ID-%d ./pop/xxx.txt] Geometry for Plotting and Its Constraints ''' %(THE_IM_DESIGN_ID)

    Qs # number of stator slots
    Qr # number of rotor slots
    Angle_StatorSlotSpan = 360 / Qs # in deg.
    Angle_RotorSlotSpan = 360 / Qr # in deg.

    Radius_OuterStatorYoke  = 0.5*stator_outer_diameter_Dse * 1e3
    Radius_InnerStatorYoke  = 0.5*stator_yoke_diameter_Dsyi * 1e3
    Length_AirGap           = air_gap_length_delta * 1e3
    Radius_OuterRotor       = 0.5*rotor_outer_diameter_Dr * 1e3
    Radius_Shaft            = 0.5*rotor_inner_diameter_Dri * 1e3

    Length_HeadNeckRotorSlot = length_headNeckRotorSlot *1e3 # mm # 这里假设与HeadNeck相对的槽的部分也能放导体了。准确来说应该有：rotor_inner_diameter_Dri = rotor_yoke_diameter_Dryi - 2*rotor_yoke_height_h_yr - 2*1e-3*Length_HeadNeckRotorSlot

    rotor_slot_radius = (2*pi*(Radius_OuterRotor - Length_HeadNeckRotorSlot)*1e-3 - rotor_tooth_width_b_dr*Qr) / (2*Qr+2*pi)
    print 'Rotor Slot Radius=', rotor_slot_radius * 1e3, 'mm'
    print 'rotor_tooth_width_b_dr=', rotor_tooth_width_b_dr * 1e3, 'mm'

    Radius_of_RotorSlot = rotor_slot_radius*1e3 
    Location_RotorBarCenter = Radius_OuterRotor - Length_HeadNeckRotorSlot - Radius_of_RotorSlot
    Width_RotorSlotOpen = b1*1e3 # 10% of 360/Qr


    # new method
    Radius_of_RotorSlot2 = 1e3 * (2*pi*(Radius_OuterRotor - Length_HeadNeckRotorSlot - rotor_slot_height_h_sr*1e3)*1e-3 - rotor_tooth_width_b_dr*Qr) / (2*Qr-2*pi)
    print 'Radius_of_RotorSlot2=', Radius_of_RotorSlot2
    Location_RotorBarCenter2 = Radius_OuterRotor - Length_HeadNeckRotorSlot - rotor_slot_height_h_sr*1e3 + Radius_of_RotorSlot2 

    # old approximate mehtod
    # Location_RotorBarCenter2 = Radius_OuterRotor - Length_HeadNeckRotorSlot - rotor_tooth_height_h_dr*1e3 # 本来还应该减去转子内槽半径的，但是这里还不知道，干脆不要了，这样槽会比预计的偏深，
    # if abs(Location_RotorBarCenter2 - Location_RotorBarCenter) < Radius_of_RotorSlot*0.25:
    #     print 'There is no need to use a drop shape rotor, because the required rotor bar height is not high.'
    # print 'the width of outer rotor slot: %g' % (Radius_of_RotorSlot)
    # print 'the height of total rotor slot: %g' % (rotor_tooth_height_h_dr*1e3)
    # Arc_betweenOuterRotorSlot = 2*pi/Qr*Location_RotorBarCenter - 2*Radius_of_RotorSlot
    # Radius_of_RotorSlot2 = 0.5 * (2*pi/Qr*Location_RotorBarCenter2 - Arc_betweenOuterRotorSlot) # 应该小于等于这个值，保证转子齿等宽。
    # print 'Radius_of_RotorSlot2=', Radius_of_RotorSlot2
    
    Angle_StatorSlotOpen = angle_stator_slop_open / pi *180 # in deg.
    Width_StatorTeethBody = stator_tooth_width_b_ds*1e3
    
    Width_StatorTeethHeadThickness = width_statorTeethHeadThickness*1e3 # mm # 这里假设与齿头、齿脖相对的槽的部分也能放导体了。准确来说应该stator_yoke_diameter_Dsyi = stator_inner_diameter_Dis + 2*stator_tooth_height_h_ds + 2*1e-3*(Width_StatorTeethHeadThickness+Width_StatorTeethNeck)
    Width_StatorTeethNeck = 0.5*Width_StatorTeethHeadThickness # mm 



    DriveW_poles=no_pole_pairs*2
    DriveW_turns=no_conductors_per_slot_zQ
    DriveW_CurrentAmp = stator_phase_current_rms * sqrt(2); 
    DriveW_Freq = rated_frequency

    # 参考FEMM_Solver.py 或 按照书上的公式算一下
    stator_slot_area = area_stator_slot_Sus # FEMM是对槽积分得到的，更准哦
    TEMPERATURE_OF_COIL = 75
    print 'TEMPERATURE_OF_COIL=', TEMPERATURE_OF_COIL, 'deg Celcius'
    rho_Copper = (3.76*TEMPERATURE_OF_COIL+873)*1e-9/55.
    SLOT_FILL_FACTOR = space_factor_kCu
    coil_pitch_slot_count = Qs / DriveW_poles # 整距！        
    length_endArcConductor = coil_pitch_slot_count/Qs * (0.5*(Radius_OuterRotor + Length_AirGap + Radius_InnerStatorYoke)) * 2*pi # [mm] arc length = pi * diameter  
    length_conductor = (stack_length*1e3 + length_endArcConductor) * 1e-3 # mm to m  ## imagine: two conductors + two end conducotors = one loop (in and out)
    area_conductor   = (stator_slot_area) * SLOT_FILL_FACTOR / DriveW_turns # TODO: 这里绝缘用槽满率算进去了，但是没有考虑圆形导体之间的空隙？槽满率就是空隙，这里没有考虑绝缘的面积占用。
    number_parallel_branch = 1
    resistance_per_conductor = rho_Copper * length_conductor / (area_conductor * number_parallel_branch)
    DriveW_Rs = resistance_per_conductor * DriveW_turns * Qs / 3. # resistance per phase
    print 'DriveW_Rs=', DriveW_Rs, 'Ohm'

    with open(loc_txt_file, 'a') as f:
        f.write('%d, ' %(THE_IM_DESIGN_ID))
        f.write('%d, %d, ' %(Qs, Qr))
        f.write('%f, %f, %f, %f, %f, ' % (Radius_OuterStatorYoke, Radius_InnerStatorYoke, Length_AirGap, Radius_OuterRotor, Radius_Shaft))
        f.write('%f, %f, %f, %f, %f, %f, ' % (Length_HeadNeckRotorSlot,Radius_of_RotorSlot, Location_RotorBarCenter, Width_RotorSlotOpen, Radius_of_RotorSlot2, Location_RotorBarCenter2))
        f.write('%f, %f, %f, %f, ' % (Angle_StatorSlotOpen, Width_StatorTeethBody, Width_StatorTeethHeadThickness, Width_StatorTeethNeck))
        f.write('%f, %f, %f, %f, %f, %f,' % (DriveW_poles, DriveW_turns, DriveW_Rs, DriveW_CurrentAmp, DriveW_Freq, stack_length*1000))
        f.write('%.14f, %.14f, %.14f,' % (area_stator_slot_Sus, area_rotor_slot_Sur, minimum__area_rotor_slot_Sur)) # this line exports values need to impose constraints among design parameters for the de optimization
        f.write('%g, %g, %g, %g\n' % (rotor_tooth_flux_density_B_dr, stator_tooth_flux_density_B_ds, rotor_current_density_Jr, rotor_tooth_width_b_dr))

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
    print 'b1 (rotor slot opening) =', b1 
    print 'rotor_tooth_width_b_dr =', rotor_tooth_width_b_dr
    print 'Length_HeadNeckRotorSlot =', Length_HeadNeckRotorSlot
    print 'Angle_StatorSlotOpen =', Angle_StatorSlotOpen
    print 'Width_StatorTeethHeadThickness =', Width_StatorTeethHeadThickness

    return Radius_OuterRotor

# for THE_IM_DESIGN_ID, Qr in enumerate([16,20,28,32,36]): # any Qr>36 will not converge (for alpha_i and k_sat)
# for THE_IM_DESIGN_ID, Qr in enumerate([32,36]): # any Qr>36 will not converge (for alpha_i and k_sat) with Arnon5 at least
# for THE_IM_DESIGN_ID, Qr in enumerate([32]):
Qr = 32
Qr = 16
bool_run_for_bounds = False
for rotor_tooth_flux_density_B_dr in arange(1.1, 2.11, 0.2): #1.5–2.2 (rotor) 
    for stator_tooth_flux_density_B_ds in arange(1.1, 1.81, 0.2): #1.4–2.1 (stator) # too large you will get End of Loop Error (Fixed by extropolating the k_sat vs alpha_i curve.)
        for rotor_current_density_Jr in arange(3e6, 8e6+1, 1e6):

            if not bool_run_for_bounds:
                rotor_tooth_flux_density_B_dr = 1.5
                stator_tooth_flux_density_B_ds = 1.4
                rotor_current_density_Jr = 6.4e6

            Radius_OuterRotor = pyrhonen_blim_design(   rotor_tooth_flux_density_B_dr,
                                                        stator_tooth_flux_density_B_ds,
                                                        rotor_current_density_Jr)

            if not bool_run_for_bounds:
                break
        if not bool_run_for_bounds:
            break
    if not bool_run_for_bounds:
        break






'''
1. 按照这本书Pyrhonen2009根据Eric的要求50kW30000rpm设计一个初始电机。
2. 用瞬态场，遍历短距、斜槽、转子槽数，对转矩脉动和悬浮力脉动的影响。

注意，必须先确定短距和转子槽数才能继续优化槽型啊！

3. 根据确定好的短距和转子槽数，构造涡流场，优化电机的槽型和气隙长度，提高电机的转矩输出性能和效率。
4. 最后再按最终设计，进行瞬态场验证。
'''


print '\n\n\nMechanical Limits Check:'
# quit()

if True:
    print 'Radius_OuterRotor=', Radius_OuterRotor, 'mm'
    rotor_radius = Radius_OuterRotor*1e-3
    speed_rpm = 30000

    Omega = speed_rpm/(60)*2*pi
    modulus_of_elasticity = 190 * 1e9 # Young's modulus
    D_out = rotor_radius * 2
    D_in = 0
    second_moment_of_inertia_of_area_I = pi*(D_out**4 - D_in**4) / 64 
    stack_length_max = sqrt( 1**2 * pi**2 / (1.5*Omega) * sqrt( 200*1e9 * second_moment_of_inertia_of_area_I / (8760* pi*(D_out/2)**2) ) )
    print 'stack_length_max=', stack_length_max*1e3, 'mm'

    # stack_length_max = sqrt(1*pi**2/(1.5*20000/60*2*pi)*sqrt(200*1e9*pi*0.15**4/64/(8760*pi*0.15**2/4)))
    # print 'stack_length_max=', stack_length_max

    # speed_rpm = 30000
    # Omega = speed_rpm/(60)*2*pi
    C_prime = (3+0.29)/4
    rotor_radius_max = sqrt(300e6/(C_prime*8760*Omega**2))
    print 'rotor_radius_max', rotor_radius_max*1e3, 'mm'
    print 'rotor_diameter_max', 2*rotor_radius_max*1e3, 'mm'
