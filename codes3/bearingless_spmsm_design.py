import CrossSectInnerNotchedRotor
import CrossSectStator
import Location2D
import winding_layout
from pyrhonen_procedure_as_function import get_material_data
import numpy as np
import logging
import utility
EPS = 1e-2 # unit: mm

class bearingless_spmsm_template(object):
    def __init__(self, fea_config_dict=None, model_name_prefix='SPMSM'):
        self.model_name_prefix = model_name_prefix
        self.name = model_name_prefix
        self.fea_config_dict = fea_config_dict

    def build_design_parameters_list(self):
        self.design_parameters = [
            self.deg_alpha_st,
            self.deg_alpha_so,
            self.mm_r_si,
            self.mm_d_so,
            self.mm_d_sp,
            self.mm_d_st,
            self.mm_d_sy,
            self.mm_w_st,
            self.mm_r_st,
            self.mm_r_sf,
            self.mm_r_sb,
            self.Q,
            self.sleeve_length,
            self.fixed_air_gap_length,
            self.mm_d_pm,
            self.deg_alpha_rm,
            self.deg_alpha_rs,
            self.mm_d_ri,
            self.mm_r_ri,
            self.mm_d_rp,
            self.mm_d_rs,
            self.p,
            self.s
            ]
        return self.design_parameters

    def get_classic_bounds(self, which_filter='FixedSleeveLength', user_bound_filter=None):
        # bound_filter is used to filter out some free_variables that are not going to be optimized.
        self.bound_filter = [ 1,          # deg_alpha_st        = free_variables[0]
                              1,          # mm_d_so             = free_variables[1]
                              1,          # mm_d_st             = free_variables[2]
                              1,          # stator_outer_radius = free_variables[3]
                              1,          # mm_w_st             = free_variables[4] # STATOR
                              0,          # sleeve_length       = free_variables[5] # AIRGAP  # 0: sleeve_length = 3 mm        # TIA ITEC
                              1,          # mm_d_pm             = free_variables[6] # ROTOR
                              1,          # deg_alpha_rm        = free_variables[7]
                              self.s!=1,  # deg_alpha_rs        = free_variables[8]           # 0: deg_alpha_rs = deg_alpha_rm # s=1
                              1,          # mm_d_ri             = free_variables[9]
                              1,          # rotor_outer_radius  = free_variables[10]                                           # Comment: the outer radius of the rotor without magnet
                              1,          # mm_d_rp             = free_variables[11]
                              self.s!=1]  # mm_d_rs             = free_variables[12]          # 0: mm_d_rs = 0                 # s=1
        if 'FixedSleeveLength' in which_filter:
            pass
        elif 'VariableSleeveLength' in which_filter:
            self.bound_filter[5] = 1
        else:
            if len(user_bound_filter) != 13:
                raise Exception('Invalid bound_filter for bounds. Should be 13 length but %d.'%(len(user_bound_filter)))
            self.bound_filter = user_bound_filter

        Q = self.Q
        s = self.s
        p = self.p
        self.original_template_neighbor_bounds =  [ 
                            [ 0.35*360/Q, 0.9*360/Q],                                           # deg_alpha_st        = free_variables[0]
                            [  0.5,   5],                                                       # mm_d_so             = free_variables[1]
                            [0.8*self.mm_d_st,                1.2*self.mm_d_st],                # mm_d_st             = free_variables[2]
                            [1.0*self.Radius_OuterStatorYoke, 1.2*self.Radius_OuterStatorYoke], # stator_outer_radius = free_variables[3]
                            [0.8*self.mm_w_st,                1.2*self.mm_w_st],                # mm_w_st             = free_variables[4] # STATOR
                            [3,   4],                                                           # sleeve_length       = free_variables[5] # AIRGAP
                            [2.5, 7],                                                           # mm_d_pm             = free_variables[6] # ROTOR
                            [0.6*360/(2*p), 1.0*360/(2*p)],                                     # deg_alpha_rm        = free_variables[7]
                            [0.8*360/(2*p)/s, 0.975*360/(2*p)/s],                               # deg_alpha_rs        = free_variables[8]
                            [0.8*self.mm_d_ri,  1.2*self.mm_d_ri],                              # mm_d_ri             = free_variables[9]
                            [0.8*self.Radius_OuterRotor, 1.2*self.Radius_OuterRotor],           # rotor_outer_radius  = free_variables[10] 
                            [2.5,   6],                                                         # mm_d_rp             = free_variables[11]
                            [2.5,   6] ]                                                        # mm_d_rs             = free_variables[12]
        index_not_included = [idx for idx, el in enumerate(self.bound_filter) if el==0]
        self.filtered_template_neighbor_bounds = [bound for idx, bound in enumerate(self.original_template_neighbor_bounds) if idx not in index_not_included]
        return self.filtered_template_neighbor_bounds

    def get_rotor_volume(self, stack_length=None):
        if stack_length is None:
            return np.pi*(self.Radius_OuterRotor*1e-3)**2 * (self.stack_length*1e-3)
        else:
            return np.pi*(self.Radius_OuterRotor*1e-3)**2 * (stack_length*1e-3)

    def get_rotor_weight(self, gravity=9.8, stack_length=None):
        material_density_rho = get_material_data()[0]
        if stack_length is None:
            return gravity * self.get_rotor_volume() * material_density_rho # steel 7860 or 8050 kg/m^3. Copper/Density 8.96 g/cm³. gravity: 9.8 N/kg
        else:
            return gravity * self.get_rotor_volume(stack_length=stack_length) * material_density_rho # steel 7860 or 8050 kg/m^3. Copper/Density 8.96 g/cm³. gravity: 9.8 N/kg

class bearingless_spmsm_design(bearingless_spmsm_template):

    def __init__(self, spmsm_template=None, x_denorm=None, counter=None, counter_loop=None):
        #00 Settings
        super(bearingless_spmsm_design, self).__init__()
        self.fea_config_dict = spmsm_template.fea_config_dict

        #01 Model ID
        self.model_name_prefix
        self.counter = counter
        self.counter_loop = counter_loop
        if counter is not None:
            if counter_loop == 1:
                self.name = 'ind%d'%(counter)
            else:
                self.name = 'ind%d-redo%d'%(counter, counter_loop)
        else:
            self.name = 'SPMSM Template'

        #02 Geometry Data
        if x_denorm is None:
            free_variables = [0,0,0,0,0, 0,0,0,0,0, 0,0,0]
            free_variables[0]  = spmsm_template.deg_alpha_st    
            free_variables[1]  = spmsm_template.mm_d_so         
            free_variables[2]  = spmsm_template.mm_d_st
            free_variables[3]  = spmsm_template.mm_r_si + spmsm_template.mm_d_sp + spmsm_template.mm_d_st + spmsm_template.mm_d_sy
            free_variables[4]  = spmsm_template.mm_w_st         
            free_variables[5]  = spmsm_template.sleeve_length   
            free_variables[6]  = spmsm_template.mm_d_pm         
            free_variables[7]  = spmsm_template.deg_alpha_rm    
            free_variables[8]  = spmsm_template.deg_alpha_rs    
            free_variables[9]  = spmsm_template.mm_d_ri         
            free_variables[10] = spmsm_template.mm_r_ri + spmsm_template.mm_d_ri + spmsm_template.mm_d_rp
            free_variables[11] = spmsm_template.mm_d_rp         
            free_variables[12] = spmsm_template.mm_d_rs         
            # use bound_filter to get x_denorm from free_variables
            # raise Exception('Not implemented')
        else:
            # build free_variables from x_denorm
            free_variables = [None]*13
            idx_x_denorm = 0
            for idx, boo in enumerate(spmsm_template.bound_filter):
                if boo == 1:
                    free_variables[idx] = x_denorm[idx_x_denorm]
                    idx_x_denorm += 1
                else:
                    if idx == 5:
                        free_variables[5] = spmsm_template.sleeve_length
                        print('TIA ITEC: Sleeve length is fixed to %g mm'%(spmsm_template.sleeve_length))
                    elif idx == 8:
                        free_variables[8] = free_variables[idx-1]
                    elif idx == 12:
                        free_variables[12] = spmsm_template.mm_d_rs
                    else:
                        raise Exception('Not tested feature.')

        deg_alpha_st             = free_variables[0]  # if self.filter[0]  else spmsm_template.deg_alpha_st            
        mm_d_so                  = free_variables[1]  # if self.filter[1]  else spmsm_template.mm_d_so                 
        mm_d_st                  = free_variables[2]  # if self.filter[2]  else spmsm_template.mm_d_st                 
        stator_outer_radius      = free_variables[3]  # if self.filter[3]  else spmsm_template.stator_outer_radius     
        mm_w_st                  = free_variables[4]  # if self.filter[4]  else spmsm_template.mm_w_st                 
        sleeve_length            = free_variables[5]  # if self.filter[5]  else spmsm_template.sleeve_length           
        mm_d_pm                  = free_variables[6]  # if self.filter[6]  else spmsm_template.mm_d_pm                 
        deg_alpha_rm             = free_variables[7]  # if self.filter[7]  else spmsm_template.deg_alpha_rm            
        deg_alpha_rs             = free_variables[8]  # if self.filter[8]  else deg_alpha_rm
        mm_d_ri                  = free_variables[9]  # if self.filter[9]  else spmsm_template.mm_d_ri                 
        rotor_steel_outer_radius = free_variables[10] # if self.filter[10] else spmsm_template.rotor_steel_outer_radius
        mm_d_rp                  = free_variables[11] # if self.filter[11] else spmsm_template.mm_d_rp                 
        mm_d_rs                  = free_variables[12] # if self.filter[12] else spmsm_template.mm_d_rs                 

        if mm_d_rp > mm_d_pm:
            mm_d_rp = mm_d_pm
            free_variables[11] = free_variables[6]
            msg = 'Warning: mm_d_rp cannot be larger than mm_d_pm or else the sleeve cannot really hold or even touch the PM. So mm_d_rp is set to equal to mm_d_pm for this design variant.'
            print(msg)
            logger = logging.getLogger(__name__).warn(msg)

        self.deg_alpha_st = free_variables[0]
        self.deg_alpha_so = spmsm_template.deg_alpha_so
        self.mm_r_si      = rotor_steel_outer_radius + (mm_d_pm - mm_d_rp) + sleeve_length + spmsm_template.fixed_air_gap_length
        self.mm_d_so      = free_variables[1]
        self.mm_d_sp      = 1.5*self.mm_d_so # The neck is 0.5 of the head as IM's geometry.
        self.mm_d_st      = free_variables[2]; stator_outer_radius = free_variables[3]
        self.mm_d_sy      = stator_outer_radius - spmsm_template.mm_d_sp - mm_d_st - self.mm_r_si
        self.mm_w_st      = free_variables[4]
        self.mm_r_st      = spmsm_template.mm_r_st
        self.mm_r_sf      = spmsm_template.mm_r_sf
        self.mm_r_sb      = spmsm_template.mm_r_sb
        self.Q            = spmsm_template.Q
        self.sleeve_length = free_variables[5]
        self.fixed_air_gap_length = spmsm_template.fixed_air_gap_length
        self.mm_d_pm      = free_variables[6]
        self.deg_alpha_rm = free_variables[7]
        self.deg_alpha_rs = free_variables[8]
        self.mm_d_ri      = free_variables[9]; rotor_steel_outer_radius = free_variables[10]
        self.mm_r_ri      = rotor_steel_outer_radius - mm_d_rp - mm_d_ri
        self.mm_d_rp      = free_variables[11]
        self.mm_d_rs      = free_variables[12]
        self.p = spmsm_template.p
        self.s = spmsm_template.s
        if self.s == 1:
            self.deg_alpha_rs = self.deg_alpha_rm # raise Exception('Invalid alpha_rs. Check that it is equal to alpha_rm for s=1')
            self.mm_d_rs = 0 # raise Exception('Invalid d_rs. Check that it is equal to 0 for s =1')
        # This is all we need
        design_parameters = self.build_design_parameters_list()
        # spmsm_template.design_parameters = [
        #                                   0 spmsm_template.deg_alpha_st 
        #                                   1 spmsm_template.deg_alpha_so 
        #                                   2 spmsm_template.mm_r_si      
        #                                   3 spmsm_template.mm_d_so      
        #                                   4 spmsm_template.mm_d_sp      
        #                                   5 spmsm_template.mm_d_st      
        #                                   6 spmsm_template.mm_d_sy      
        #                                   7 spmsm_template.mm_w_st      
        #                                   8 spmsm_template.mm_r_st      
        #                                   9 spmsm_template.mm_r_sf      
        #                                  10 spmsm_template.mm_r_sb      
        #                                  11 spmsm_template.Q            
        #                                  12 spmsm_template.sleeve_length
        #                                  13 spmsm_template.fixed_air_gap_length
        #                                  14 spmsm_template.mm_d_pm      
        #                                  15 spmsm_template.deg_alpha_rm 
        #                                  16 spmsm_template.deg_alpha_rs 
        #                                  17 spmsm_template.mm_d_ri      
        #                                  18 spmsm_template.mm_r_ri      
        #                                  19 spmsm_template.mm_d_rp      
        #                                  20 spmsm_template.mm_d_rs      
        #                                  21 spmsm_template.p
        #                                  22 spmsm_template.s
        #                                 ]

        # obsolete variable from IM
        self.Radius_OuterRotor = rotor_steel_outer_radius  + (mm_d_pm - mm_d_rp) # the outer radius of the rotor with magnet / the magnet
        self.Length_AirGap = self.fixed_air_gap_length
        self.ID = 'p%gs%g'%(self.p,self.s)
        self.number_current_generation = 0
        self.individual_index = counter
        self.Radius_OuterStatorYoke = stator_outer_radius
        self.stator_yoke_diameter_Dsyi = 1e-3 * 2*(stator_outer_radius - self.mm_d_sy)

        # Parts
        self.rotorCore = CrossSectInnerNotchedRotor.CrossSectInnerNotchedRotor(
                            name = 'NotchedRotor',
                            mm_d_pm      = design_parameters[-9],
                            deg_alpha_rm = design_parameters[-8], # angular span of the pole: class type DimAngular
                            deg_alpha_rs = design_parameters[-7], # segment span: class type DimAngular
                            mm_d_ri      = design_parameters[-6], # inner radius of rotor: class type DimLinear
                            mm_r_ri      = design_parameters[-5], # rotor iron thickness: class type DimLinear
                            mm_d_rp      = design_parameters[-4], # interpolar iron thickness: class type DimLinear
                            mm_d_rs      = design_parameters[-3], # inter segment iron thickness: class type DimLinear
                            p = design_parameters[-2], # Set pole-pairs to 2
                            s = design_parameters[-1], # Set magnet segments/pole to 4
                            location = Location2D.Location2D(anchor_xy=[0,0], deg_theta=0))

        self.shaft = CrossSectInnerNotchedRotor.CrossSectShaft(name = 'Shaft',
                                                      notched_rotor = self.rotorCore
                                                    )

        self.rotorMagnet = CrossSectInnerNotchedRotor.CrossSectInnerNotchedMagnet( name = 'RotorMagnet',
                                                      notched_rotor = self.rotorCore
                                                    )

        self.stator_core = CrossSectStator.CrossSectInnerRotorStator( name = 'StatorCore',
                                            deg_alpha_st = design_parameters[0], #40,
                                            deg_alpha_so = design_parameters[1], #20,
                                            mm_r_si = design_parameters[2],
                                            mm_d_so = design_parameters[3],
                                            mm_d_sp = design_parameters[4],
                                            mm_d_st = design_parameters[5],
                                            mm_d_sy = design_parameters[6],
                                            mm_w_st = design_parameters[7],
                                            mm_r_st = design_parameters[8], # =0
                                            mm_r_sf = design_parameters[9], # =0
                                            mm_r_sb = design_parameters[10], # =0
                                            Q = design_parameters[11],
                                            location = Location2D.Location2D(anchor_xy=[0,0], deg_theta=0)
                                            )

        self.coils = CrossSectStator.CrossSectInnerRotorStatorWinding(name = 'Coils',
                                                    stator_core = self.stator_core)
        self.design_parameters = design_parameters

        #03 Inherit properties
        self.DriveW_Freq       = spmsm_template.DriveW_Freq      
        self.DriveW_Rs         = spmsm_template.DriveW_Rs        
        self.DriveW_zQ         = spmsm_template.DriveW_zQ
        print('---DriveW_zQ:', self.DriveW_zQ)
        print('---Template CurrentAmp:', spmsm_template.DriveW_CurrentAmp)
        self.DriveW_CurrentAmp = None ########### will be assisned when drawing the coils
        self.DriveW_poles      = spmsm_template.DriveW_poles

        self.Js                = spmsm_template.Js 
        self.fill_factor       = spmsm_template.fill_factor 

        self.stack_length      = spmsm_template.stack_length 

        # self.TORQUE_CURRENT_RATIO = spmsm_template.TORQUE_CURRENT_RATIO
        # self.SUSPENSION_CURRENT_RATIO = spmsm_template.SUSPENSION_CURRENT_RATIO

        #03 Mechanical Parameters
        self.update_mechanical_parameters()

        # #04 Material Condutivity Properties
        # if self.fea_config_dict is not None:
        #     self.End_Ring_Resistance = fea_config_dict['End_Ring_Resistance']
        #     self.Bar_Conductivity = fea_config_dict['Bar_Conductivity']
        # self.Copper_Loss = self.DriveW_CurrentAmp**2 / 2 * self.DriveW_Rs * 3
        # # self.Resistance_per_Turn = 0.01 # TODO


        #05 Windings & Excitation
        self.wily = spmsm_template.wily

        if self.DriveW_poles == 2:
            self.BeariW_poles = 4
            if self.DriveW_zQ % 2 != 0:
                print('zQ=', self.DriveW_zQ)
                raise Exception('This zQ does not suit for two layer winding.')
        elif self.DriveW_poles == 4:
            self.BeariW_poles = 2;
        else:
            raise Exception('Not implemented error.')
        self.BeariW_zQ      = self.DriveW_zQ
        self.BeariW_Rs         = self.DriveW_Rs * self.BeariW_zQ / self.DriveW_zQ
        if self.DriveW_CurrentAmp is None:
            self.BeariW_CurrentAmp = None
        else:
            self.BeariW_CurrentAmp = self.fea_config_dict['SUSPENSION_CURRENT_RATIO'] * (self.DriveW_CurrentAmp / self.fea_config_dict['TORQUE_CURRENT_RATIO'])
        self.BeariW_Freq       = self.DriveW_Freq

        self.CurrentAmp_per_phase = None # will be used in copper loss calculation
        self.slot_area_utilizing_ratio = self.fea_config_dict['SUSPENSION_CURRENT_RATIO'] + self.fea_config_dict['TORQUE_CURRENT_RATIO']
        print('self.slot_area_utilizing_ratio:', self.slot_area_utilizing_ratio)

        if self.fea_config_dict is not None:
            self.dict_coil_connection = {41:self.wily.l41, 42:self.wily.l42, 21:self.wily.l21, 22:self.wily.l22} # 这里的2和4等价于leftlayer和rightlayer。

        # #06 Meshing & Solver Properties
        self.max_nonlinear_iteration = 50 # 30 for transient solve
        self.meshSize_Magnet = 2 # mm

    def update_mechanical_parameters(self, syn_freq=None):
        if syn_freq is None:
            self.the_speed = self.DriveW_Freq*60. / (0.5*self.DriveW_poles) # rpm
            self.Omega = + self.the_speed / 60. * 2*np.pi
            self.omega = None # This variable name is devil! you can't tell its electrical or mechanical! #+ self.DriveW_Freq * (1-self.the_slip) * 2*pi
        else:
            raise Exception('Not implemented.')

    def draw_spmsm(self, toolJd):

        # Rotor Core
        list_segments = self.rotorCore.draw(toolJd)
        toolJd.bMirror = False
        toolJd.iRotateCopy = self.rotorCore.p*2
        region1 = toolJd.prepareSection(list_segments)

        # Shaft
        list_segments = self.shaft.draw(toolJd)
        toolJd.bMirror = False
        toolJd.iRotateCopy = 1
        region0 = toolJd.prepareSection(list_segments)

        # Rotor Magnet
        list_regions = self.rotorMagnet.draw(toolJd)
        toolJd.bMirror = False
        toolJd.iRotateCopy = self.rotorMagnet.notched_rotor.p*2
        region2 = toolJd.prepareSection(list_regions, bRotateMerge=False)

        # Sleeve
        sleeve = CrossSectInnerNotchedRotor.CrossSectSleeve(
                        name = 'Sleeve',
                        notched_magnet = self.rotorMagnet,
                        d_sleeve = self.sleeve_length
                        )
        list_regions = sleeve.draw(toolJd)
        toolJd.bMirror = False
        toolJd.iRotateCopy = self.rotorMagnet.notched_rotor.p*2
        regionS = toolJd.prepareSection(list_regions)

        # Stator Core
        list_regions = self.stator_core.draw(toolJd)
        toolJd.bMirror = True
        toolJd.iRotateCopy = self.stator_core.Q
        region3 = toolJd.prepareSection(list_regions)

        # Stator Winding
        list_regions = self.coils.draw(toolJd)
        toolJd.bMirror = False
        toolJd.iRotateCopy = self.coils.stator_core.Q
        region4 = toolJd.prepareSection(list_regions)

        CurrentAmp_in_the_slot = self.coils.mm2_slot_area * self.fill_factor * self.Js*1e-6 * np.sqrt(2) #/2.2*2.8
        CurrentAmp_per_conductor = CurrentAmp_in_the_slot / self.DriveW_zQ
        CurrentAmp_per_phase = CurrentAmp_per_conductor * self.wily.number_parallel_branch # 跟几层绕组根本没关系！除以zQ的时候，就已经变成每根导体的电流了。
        variant_DriveW_CurrentAmp = CurrentAmp_per_phase # this current amp value is for non-bearingless motor
        self.CurrentAmp_per_phase = CurrentAmp_per_phase
        self.DriveW_CurrentAmp = self.fea_config_dict['TORQUE_CURRENT_RATIO'] * variant_DriveW_CurrentAmp 
        self.BeariW_CurrentAmp = self.fea_config_dict['SUSPENSION_CURRENT_RATIO'] * variant_DriveW_CurrentAmp

        slot_area_utilizing_ratio = (self.DriveW_CurrentAmp + self.BeariW_CurrentAmp) / self.CurrentAmp_per_phase
        print('---Heads up! slot_area_utilizing_ratio is', slot_area_utilizing_ratio)

        print('---Variant CurrentAmp_in_the_slot =', CurrentAmp_in_the_slot)
        print('---variant_DriveW_CurrentAmp = CurrentAmp_per_phase =', variant_DriveW_CurrentAmp)
        print('---self.DriveW_CurrentAmp =', self.DriveW_CurrentAmp)
        print('---self.BeariW_CurrentAmp =', self.BeariW_CurrentAmp)
        print('---TORQUE_CURRENT_RATIO:', self.fea_config_dict['TORQUE_CURRENT_RATIO'])
        print('---SUSPENSION_CURRENT_RATIO:', self.fea_config_dict['SUSPENSION_CURRENT_RATIO'])

        # Import Model into Designer
        toolJd.doc.SaveModel(False) # True: Project is also saved. 
        model = toolJd.app.GetCurrentModel()
        model.SetName(self.name)
        model.SetDescription(self.show(toString=True))

        return True

    def pre_process(self, app, model):
        # pre-process : you can select part by coordinate!
        ''' Group '''
        def group(name, id_list):
            model.GetGroupList().CreateGroup(name)
            for the_id in id_list:
                model.GetGroupList().AddPartToGroup(name, the_id)
                # model.GetGroupList().AddPartToGroup(name, name) #<- this also works

        part_ID_list = model.GetPartIDs()
        # print(part_ID_list)

        # view = app.View()
        # view.ClearSelect()
        # sel = view.GetCurrentSelection()
        # sel.SelectPart(123)
        # sel.SetBlockUpdateView(False)

        if len(part_ID_list) != int(1 + 1 + self.p*2 + 1 + 1 + self.Q*2):
            msg = 'Number of Parts is unexpected. Should be %d but only %d.\n'%(int(1 + 1 + self.p*2 + 1 + 1 + self.Q*2), len(part_ID_list)) + self.show(toString=True)
            raise utility.ExceptionBadNumberOfParts(msg)

        self.id_backiron = id_backiron = part_ID_list[0]
        id_shaft = part_ID_list[1]
        partIDRange_Magnet = part_ID_list[2:int(2+self.p*2)]
        id_sleeve = part_ID_list[int(2+self.p*2)]
        id_statorCore = part_ID_list[int(2+self.p*2)+1]
        partIDRange_Coil = part_ID_list[int(2+self.p*2)+2 : int(2+self.p*2)+2 + int(self.Q*2)]

        print(id_backiron)
        print(id_shaft)
        print(partIDRange_Magnet)
        print(id_sleeve)
        print(id_statorCore)
        print(partIDRange_Coil)

        model.SuppressPart(id_sleeve, 1)

        group("Magnet", partIDRange_Magnet)
        group("Coils", partIDRange_Coil)

        ''' Add Part to Set for later references '''
        def add_part_to_set(name, x, y, ID=None):
            model.GetSetList().CreatePartSet(name)
            model.GetSetList().GetSet(name).SetMatcherType("Selection")
            model.GetSetList().GetSet(name).ClearParts()
            sel = model.GetSetList().GetSet(name).GetSelection()
            if ID is None:
                # print x,y
                sel.SelectPartByPosition(x,y,0) # z=0 for 2D
            else:
                sel.SelectPart(ID)
            model.GetSetList().GetSet(name).AddSelected(sel)

        # def edge_set(name,x,y):
        #     model.GetSetList().CreateEdgeSet(name)
        #     model.GetSetList().GetSet(name).SetMatcherType(u"Selection")
        #     model.GetSetList().GetSet(name).ClearParts()
        #     sel = model.GetSetList().GetSet(name).GetSelection()
        #     sel.SelectEdgeByPosition(x,y,0) # sel.SelectEdge(741)
        #     model.GetSetList().GetSet(name).AddSelected(sel)
        # edge_set(u"AirGapCoast", 0, self.Radius_OuterRotor+0.5*self.Length_AirGap)

        # Shaft
        add_part_to_set('ShaftSet', 0.0, 0.0, ID=id_shaft) # 坐标没用，不知道为什么，而且都给了浮点数了

        # Create Set for 4 poles Winding
        Angle_StatorSlotSpan = 360/self.Q
        # R = self.mm_r_si + self.mm_d_sp + self.mm_d_st *0.5 # this is not generally working (JMAG selects stator core instead.)
        # THETA = 0.25*(Angle_StatorSlotSpan)/180.*np.pi
        R = np.sqrt(self.coils.PCoil[0]**2 + self.coils.PCoil[1]**2)
        THETA = np.arctan(self.coils.PCoil[1]/self.coils.PCoil[0])
        X = R*np.cos(THETA)
        Y = R*np.sin(THETA)
        count = 0
        for UVW, UpDown in zip(self.wily.l41,self.wily.l42):
            count += 1 
            add_part_to_set("Coil4%s%s %d"%(UVW,UpDown,count), X, Y)

            # print(X, Y, THETA)
            THETA += Angle_StatorSlotSpan/180.*np.pi
            X = R*np.cos(THETA)
            Y = R*np.sin(THETA)

        # Create Set for 2 poles Winding
        # THETA = 0.75*(Angle_StatorSlotSpan)/180.*np.pi # 这里这个角度的选择，决定了悬浮绕组产生悬浮力的方向！！！！！
        THETA = np.arctan(-self.coils.PCoil[1]/self.coils.PCoil[0]) + (2*np.pi)/self.Q
        X = R*np.cos(THETA)
        Y = R*np.sin(THETA)
        count = 0
        for UVW, UpDown in zip(self.wily.l21,self.wily.l22):
            count += 1 
            add_part_to_set("Coil2%s%s %d"%(UVW,UpDown,count), X, Y)

            THETA += Angle_StatorSlotSpan/180.*np.pi
            X = R*np.cos(THETA)
            Y = R*np.sin(THETA)

        # Create Set for Magnets
        R = self.mm_r_si - self.sleeve_length - self.fixed_air_gap_length - 0.5*self.mm_d_pm
        Angle_RotorSlotSpan = 360 / (self.p*2)
        THETA = 0.5*self.deg_alpha_rm / 180*np.pi # initial position
        X = R*np.cos(THETA)
        Y = R*np.sin(THETA)
        list_xy_magnets = []
        # list_xy_airWithinRotorSlot = []
        for ind in range(int(self.p*2)):
            natural_ind = ind + 1
            add_part_to_set("Magnet %d"%(natural_ind), X, Y)
            list_xy_magnets.append([X,Y])

            # print(ind, X, Y, THETA/np.pi*180, Angle_RotorSlotSpan)
            THETA += Angle_RotorSlotSpan/180.*np.pi
            X = R*np.cos(THETA)
            Y = R*np.sin(THETA)

        # Create Set for Motion Region
        def part_list_set(name, list_xy, list_part_id=None, prefix=None):
            model.GetSetList().CreatePartSet(name)
            model.GetSetList().GetSet(name).SetMatcherType("Selection")
            model.GetSetList().GetSet(name).ClearParts()
            sel = model.GetSetList().GetSet(name).GetSelection() 
            for xy in list_xy:
                sel.SelectPartByPosition(xy[0],xy[1],0) # z=0 for 2D
            if list_part_id is not None:
                for ID in list_part_id:
                    sel.SelectPart(ID)
            model.GetSetList().GetSet(name).AddSelected(sel)
        part_list_set('Motion_Region', list_xy_magnets, list_part_id=[id_backiron, id_shaft])

        part_list_set('MagnetSet', list_xy_magnets)
        return True

    def add_magnetic_transient_study(self, app, model, dir_csv_output_folder, study_name):
        logger = logging.getLogger(__name__)
        spmsm_variant = self

        model.CreateStudy("Transient2D", study_name)
        app.SetCurrentStudy(study_name)
        study = model.GetStudy(study_name)

        # SS-ATA
        # study.GetStudyProperties().SetValue("ApproximateTransientAnalysis", 1) # psuedo steady state freq is for PWM drive to use
        # study.GetStudyProperties().SetValue("SpecifySlip", 0)
        # study.GetStudyProperties().SetValue("OutputSteadyResultAs1stStep", 0)
        # study.GetStudyProperties().SetValue(u"TimePeriodicType", 2) # This is for TP-EEC but is not effective

        # misc
        study.GetStudyProperties().SetValue("ConversionType", 0)
        study.GetStudyProperties().SetValue("NonlinearMaxIteration", self.max_nonlinear_iteration)
        study.GetStudyProperties().SetValue("ModelThickness", self.stack_length) # [mm] Stack Length

        # Material
        self.add_material(study)

        # Conditions - Motion
        study.CreateCondition("RotationMotion", "RotCon") # study.GetCondition(u"RotCon").SetXYZPoint(u"", 0, 0, 1) # megbox warning
        print('the_speed:', self.the_speed)
        study.GetCondition("RotCon").SetValue("AngularVelocity", int(self.the_speed))
        study.GetCondition("RotCon").ClearParts()
        study.GetCondition("RotCon").AddSet(model.GetSetList().GetSet("Motion_Region"), 0)
        # Implementation of id=0 control:
        #   d-axis initial position is self.deg_alpha_rm*0.5
        #   The U-phase current is sin(omega_syn*t) = 0 at t=0.
        study.GetCondition("RotCon").SetValue(u"InitialRotationAngle", -self.deg_alpha_rm*0.5 + 90 + self.wily.initial_excitation_bias_compensation_deg + (180/self.p)) # add 360/(2p) deg to reverse the initial magnetizing direction to make torque positive.


        study.CreateCondition("Torque", "TorCon") # study.GetCondition(u"TorCon").SetXYZPoint(u"", 0, 0, 0) # megbox warning
        study.GetCondition("TorCon").SetValue("TargetType", 1)
        study.GetCondition("TorCon").SetLinkWithType("LinkedMotion", "RotCon")
        study.GetCondition("TorCon").ClearParts()

        study.CreateCondition("Force", "ForCon")
        study.GetCondition("ForCon").SetValue("TargetType", 1)
        study.GetCondition("ForCon").SetLinkWithType("LinkedMotion", "RotCon")
        study.GetCondition("ForCon").ClearParts()


        # Conditions - FEM Coils & Conductors (i.e. stator/rotor winding)
        self.add_circuit(app, model, study, bool_3PhaseCurrentSource=self.wily.bool_3PhaseCurrentSource)


        # True: no mesh or field results are needed
        study.GetStudyProperties().SetValue("OnlyTableResults", self.fea_config_dict['OnlyTableResults'])

        # Linear Solver
        if False:
            # sometime nonlinear iteration is reported to fail and recommend to increase the accerlation rate of ICCG solver
            study.GetStudyProperties().SetValue("IccgAccel", 1.2) 
            study.GetStudyProperties().SetValue("AutoAccel", 0)
        else:
            # this can be said to be super fast over ICCG solver.
            # https://www2.jmag-international.com/support/en/pdf/JMAG-Designer_Ver.17.1_ENv3.pdf
            study.GetStudyProperties().SetValue("DirectSolverType", 1)

        if self.fea_config_dict['MultipleCPUs'] == True:
            # This SMP(shared memory process) is effective only if there are tons of elements. e.g., over 100,000.
            # too many threads will in turn make them compete with each other and slow down the solve. 2 is good enough for eddy current solve. 6~8 is enough for transient solve.
            study.GetStudyProperties().SetValue("UseMultiCPU", True)
            study.GetStudyProperties().SetValue("MultiCPU", 2) 

        # 上一步的铁磁材料的状态作为下一步的初值，挺好，但是如果每一个转子的位置转过很大的话，反而会减慢非线性迭代。
        # 我们的情况是：0.33 sec 分成了32步，每步的时间大概在0.01秒，0.01秒乘以0.5*497 Hz = 2.485 revolution...
        # study.GetStudyProperties().SetValue(u"NonlinearSpeedup", 0) # JMAG17.1以后默认使用。现在后面密集的步长还多一点（32步），前面16步慢一点就慢一点呗！

        # two sections of different time step
        if True: # ECCE19
            number_of_steps_2ndTTS = self.fea_config_dict['number_of_steps_2ndTTS'] 
            DM = app.GetDataManager()
            DM.CreatePointArray("point_array/timevsdivision", "SectionStepTable")
            refarray = [[0 for i in range(3)] for j in range(3)]
            refarray[0][0] = 0
            refarray[0][1] =    1
            refarray[0][2] =        50
            refarray[1][0] = 1.0/self.DriveW_Freq
            refarray[1][1] =    0.5 * number_of_steps_2ndTTS
            refarray[1][2] =        50
            refarray[2][0] = 1.5/self.DriveW_Freq
            refarray[2][1] =    1 * number_of_steps_2ndTTS # 最后的number_of_steps_2ndTTS（32）步，必须对应半个周期，从而和后面的铁耗计算相对应。
            refarray[2][2] =        50
            DM.GetDataSet("SectionStepTable").SetTable(refarray)
            number_of_total_steps = 1 + 0.5*number_of_steps_2ndTTS + 1 * number_of_steps_2ndTTS # [Double Check] don't forget to modify here!
            study.GetStep().SetValue("Step", number_of_total_steps)
            study.GetStep().SetValue("StepType", 3)
            study.GetStep().SetTableProperty("Division", DM.GetDataSet("SectionStepTable"))

        # add equations
        study.GetDesignTable().AddEquation("freq")
        study.GetDesignTable().AddEquation("speed")
        study.GetDesignTable().GetEquation("freq").SetType(0)
        study.GetDesignTable().GetEquation("freq").SetExpression("%g"%((spmsm_variant.DriveW_Freq)))
        study.GetDesignTable().GetEquation("freq").SetDescription("Excitation Frequency")
        study.GetDesignTable().GetEquation("speed").SetType(1)
        study.GetDesignTable().GetEquation("speed").SetExpression("freq * %d"%(60/(spmsm_variant.DriveW_poles/2)))
        study.GetDesignTable().GetEquation("speed").SetDescription("mechanical speed of four pole")

        # speed, freq, slip
        study.GetCondition("RotCon").SetValue("AngularVelocity", 'speed')
        if self.fea_config_dict['DPNV']==False:
            app.ShowCircuitGrid(True)
            study.GetCircuit().GetComponent("CS4").SetValue("Frequency", "freq")
            study.GetCircuit().GetComponent("CS2").SetValue("Frequency", "freq")

        # max_nonlinear_iteration = 50
        # study.GetStudyProperties().SetValue(u"NonlinearMaxIteration", max_nonlinear_iteration)
        # study.GetStudyProperties().SetValue("ApproximateTransientAnalysis", 1) # psuedo steady state freq is for PWM drive to use
        # study.GetStudyProperties().SetValue("OutputSteadyResultAs1stStep", 0)

        # # add other excitation frequencies other than 500 Hz as cases
        # for case_no, DriveW_Freq in enumerate([50.0, slip_freq_breakdown_torque]):
        #     slip = slip_freq_breakdown_torque / DriveW_Freq
        #     study.GetDesignTable().AddCase()
        #     study.GetDesignTable().SetValue(case_no+1, 0, DriveW_Freq)
        #     study.GetDesignTable().SetValue(case_no+1, 1, slip)

        # 你把Tran2TSS计算周期减半！
        # 也要在计算铁耗的时候选择1/4或1/2的数据！（建议1/4）
        # 然后，手动添加end step 和 start step，这样靠谱！2019-01-09：注意设置铁耗条件（iron loss condition）的Reference Start Step和End Step。

        # Iron Loss Calculation Condition
        # Stator 
        if True:
            cond = study.CreateCondition("Ironloss", "IronLossConStator")
            cond.SetValue("RevolutionSpeed", "freq*60/%d"%(0.5*(spmsm_variant.DriveW_poles)))
            cond.ClearParts()
            sel = cond.GetSelection()
            sel.SelectPartByPosition(self.mm_r_si + EPS, 0 ,0)
            cond.AddSelected(sel)
            # Use FFT for hysteresis to be consistent with FEMM's results and to have a FFT plot
            cond.SetValue("HysteresisLossCalcType", 1)
            cond.SetValue("PresetType", 3) # 3:Custom
            # Specify the reference steps yourself because you don't really know what JMAG is doing behind you
            cond.SetValue("StartReferenceStep", number_of_total_steps+1-number_of_steps_2ndTTS*0.5) # 1/4 period <=> number_of_steps_2ndTTS*0.5
            cond.SetValue("EndReferenceStep", number_of_total_steps)
            cond.SetValue("UseStartReferenceStep", 1)
            cond.SetValue("UseEndReferenceStep", 1)
            cond.SetValue("Cyclicity", 4) # specify reference steps for 1/4 period and extend it to whole period
            cond.SetValue("UseFrequencyOrder", 1)
            cond.SetValue("FrequencyOrder", "1-50") # Harmonics up to 50th orders 
        # Check CSV reults for iron loss (You cannot check this for Freq study) # CSV and save space
        study.GetStudyProperties().SetValue("CsvOutputPath", dir_csv_output_folder) # it's folder rather than file!
        study.GetStudyProperties().SetValue("CsvResultTypes", "Torque;Force;LineCurrent;TerminalVoltage;JouleLoss;TotalDisplacementAngle;JouleLoss_IronLoss;IronLoss_IronLoss;HysteresisLoss_IronLoss")
        study.GetStudyProperties().SetValue("DeleteResultFiles", self.fea_config_dict['delete_results_after_calculation'])
        # Terminal Voltage/Circuit Voltage: Check for outputing CSV results 
        study.GetCircuit().CreateTerminalLabel("Terminal4U", 8, -13)
        study.GetCircuit().CreateTerminalLabel("Terminal4V", 8, -11)
        study.GetCircuit().CreateTerminalLabel("Terminal4W", 8, -9)
        study.GetCircuit().CreateTerminalLabel("Terminal2U", 23, -13)
        study.GetCircuit().CreateTerminalLabel("Terminal2V", 23, -11)
        study.GetCircuit().CreateTerminalLabel("Terminal2W", 23, -9)
        # Export Stator Core's field results only for iron loss calculation (the csv file of iron loss will be clean with this setting)
            # study.GetMaterial(u"Rotor Core").SetValue(u"OutputResult", 0) # at least one part on the rotor should be output or else a warning "the jplot file does not contains displacement results when you try to calc. iron loss on the moving part." will pop up, even though I don't add iron loss condition on the rotor.
        # study.GetMeshControl().SetValue(u"AirRegionOutputResult", 0)
        # study.GetMaterial("Shaft").SetValue("OutputResult", 0)
        # study.GetMaterial("Cage").SetValue("OutputResult", 0)
        # study.GetMaterial("Coil").SetValue("OutputResult", 0)
        # Rotor
        if True:
            cond = study.CreateCondition("Ironloss", "IronLossConRotor")
            cond.SetValue("BasicFrequencyType", 2)
            cond.SetValue("BasicFrequency", "freq")
                # cond.SetValue(u"BasicFrequency", u"slip*freq") # this require the signal length to be at least 1/4 of slip period, that's too long!
            cond.ClearParts()
            sel = cond.GetSelection()
            # sel.SelectPartByPosition(self.mm_r_ri + EPS, 0 ,0) # Why this is not working??? Because it is integer.... you must use 0.0 instead of 0!!!
            sel.SelectPart(self.id_backiron)

            cond.AddSelected(sel)
            # Use FFT for hysteresis to be consistent with FEMM's results
            cond.SetValue("HysteresisLossCalcType", 1)
            cond.SetValue("PresetType", 3)
            # Specify the reference steps yourself because you don't really know what JMAG is doing behind you
            cond.SetValue("StartReferenceStep", number_of_total_steps+1-number_of_steps_2ndTTS*0.5) # 1/4 period <=> number_of_steps_2ndTTS*0.5
            cond.SetValue("EndReferenceStep", number_of_total_steps)
            cond.SetValue("UseStartReferenceStep", 1)
            cond.SetValue("UseEndReferenceStep", 1)
            cond.SetValue("Cyclicity", 4) # specify reference steps for 1/4 period and extend it to whole period
            cond.SetValue("UseFrequencyOrder", 1)
            cond.SetValue("FrequencyOrder", "1-50") # Harmonics up to 50th orders 
        self.study_name = study_name
        return study

    def add_structural_static_study(self):
        pass

    def add_mesh(self, study, model):
        # this is for multi slide planes, which we will not be using
        refarray = [[0 for i in range(2)] for j in range(1)]
        refarray[0][0] = 3
        refarray[0][1] = 1
        study.GetMeshControl().GetTable("SlideTable2D").SetTable(refarray) 

        study.GetMeshControl().SetValue("MeshType", 1) # make sure this has been exe'd: study.GetCondition(u"RotCon").AddSet(model.GetSetList().GetSet(u"Motion_Region"), 0)
        study.GetMeshControl().SetValue("RadialDivision", 4) # for air region near which motion occurs
        study.GetMeshControl().SetValue("CircumferentialDivision", 720) #1440) # for air region near which motion occurs 这个数足够大，sliding mesh才准确。
        study.GetMeshControl().SetValue("AirRegionScale", 1.05) # [Model Length]: Specify a value within the following area. (1.05 <= value < 1000)
        study.GetMeshControl().SetValue("MeshSize", 4) # mm
        study.GetMeshControl().SetValue("AutoAirMeshSize", 0)
        study.GetMeshControl().SetValue("AirMeshSize", 4) # mm
        study.GetMeshControl().SetValue("Adaptive", 0)

        study.GetMeshControl().CreateCondition("RotationPeriodicMeshAutomatic", "autoRotMesh") # with this you can choose to set CircumferentialDivision automatically

        study.GetMeshControl().CreateCondition("Part", "MagnetMeshCtrl")
        study.GetMeshControl().GetCondition("MagnetMeshCtrl").SetValue("Size", self.meshSize_Magnet)
        study.GetMeshControl().GetCondition("MagnetMeshCtrl").ClearParts()
        study.GetMeshControl().GetCondition("MagnetMeshCtrl").AddSet(model.GetSetList().GetSet("MagnetSet"), 0)

        study.GetMeshControl().CreateCondition("Part", "ShaftMeshCtrl")
        study.GetMeshControl().GetCondition("ShaftMeshCtrl").SetValue("Size", 10) # 10 mm
        study.GetMeshControl().GetCondition("ShaftMeshCtrl").ClearParts()
        study.GetMeshControl().GetCondition("ShaftMeshCtrl").AddSet(model.GetSetList().GetSet("ShaftSet"), 0)

        def mesh_all_cases(study):
            numCase = study.GetDesignTable().NumCases()
            for case in range(0, numCase):
                study.SetCurrentCase(case)
                if study.HasMesh() == False:
                    study.CreateMesh()
                # if case == 0:
                #     app.View().ShowAllAirRegions()
                #     app.View().ShowMeshGeometry()
                #     app.View().ShowMesh()

        mesh_all_cases(study)

    # TranFEAwi2TSS
    def add_material(self, study):
        if 'M19' in self.fea_config_dict['Steel']:
            study.SetMaterialByName("StatorCore", "M-19 Steel Gauge-29")
            study.GetMaterial("StatorCore").SetValue("Laminated", 1)
            study.GetMaterial("StatorCore").SetValue("LaminationFactor", 95)
                # study.GetMaterial(u"Stator Core").SetValue(u"UserConductivityValue", 1900000)

            study.SetMaterialByName("NotchedRotor", "M-19 Steel Gauge-29")
            study.GetMaterial("NotchedRotor").SetValue("Laminated", 1)
            study.GetMaterial("NotchedRotor").SetValue("LaminationFactor", 95)

        elif 'M15' in self.fea_config_dict['Steel']:
            study.SetMaterialByName("StatorCore", "M-15 Steel")
            study.GetMaterial("StatorCore").SetValue("Laminated", 1)
            study.GetMaterial("StatorCore").SetValue("LaminationFactor", 98)

            study.SetMaterialByName("NotchedRotor", "M-15 Steel")
            study.GetMaterial("NotchedRotor").SetValue("Laminated", 1)
            study.GetMaterial("NotchedRotor").SetValue("LaminationFactor", 98)

        elif self.fea_config_dict['Steel'] == 'Arnon5':
            study.SetMaterialByName("StatorCore", "Arnon5-final")
            study.GetMaterial("StatorCore").SetValue("Laminated", 1)
            study.GetMaterial("StatorCore").SetValue("LaminationFactor", 96)

            study.SetMaterialByName("NotchedRotor", "Arnon5-final")
            study.GetMaterial("NotchedRotor").SetValue("Laminated", 1)
            study.GetMaterial("NotchedRotor").SetValue("LaminationFactor", 96)

        else:
            msg = 'Warning: default material is used: DCMagnetic Type/50A1000.'
            print(msg)
            logging.getLogger(__name__).warn(msg)
            study.SetMaterialByName("StatorCore", "DCMagnetic Type/50A1000")
            study.GetMaterial("StatorCore").SetValue("UserConductivityType", 1)
            study.SetMaterialByName("NotchedRotor", "DCMagnetic Type/50A1000")
            study.GetMaterial("NotchedRotor").SetValue("UserConductivityType", 1)

        study.SetMaterialByName("Coils", "Copper")
        study.GetMaterial("Coils").SetValue("UserConductivityType", 1)

        # study.SetMaterialByName("Cage", "Aluminium")
        # study.GetMaterial("Cage").SetValue("EddyCurrentCalculation", 1)
        # study.GetMaterial("Cage").SetValue("UserConductivityType", 1)
        # study.GetMaterial("Cage").SetValue("UserConductivityValue", self.Bar_Conductivity)

        # N40H Reversible
        study.SetMaterialByName(u"Magnet", u"Arnold/Reversible/N40H")
        study.GetMaterial(u"Magnet").SetValue(u"EddyCurrentCalculation", 1)
        study.GetMaterial(u"Magnet").SetValue(u"Temperature", 80) # TEMPERATURE (There is no 75 deg C option)

        study.GetMaterial(u"Magnet").SetValue(u"Poles", self.DriveW_poles)

        study.GetMaterial(u"Magnet").SetDirectionXYZ(1, 0, 0)
        study.GetMaterial(u"Magnet").SetAxisXYZ(0, 0, 1)
        study.GetMaterial(u"Magnet").SetOriginXYZ(0, 0, 0)
        study.GetMaterial(u"Magnet").SetPattern(u"RadialCircular")
        study.GetMaterial(u"Magnet").SetValue(u"StartAngle", 0.5*self.deg_alpha_rm)


        # add_carbon_fiber_material(app)

    def add_circuit(self, app, model, study, bool_3PhaseCurrentSource=True):
        # Circuit - Current Source
        app.ShowCircuitGrid(True)
        study.CreateCircuit()

        # 4 pole motor Qs=24 dpnv implemented by two layer winding (6 coils). In this case, drive winding has the same slot turns as bearing winding
        def circuit(poles,turns,Rs,ampD,ampB,freq,phase=0, CommutatingSequenceD=0, CommutatingSequenceB=0, x=10,y=10, bool_3PhaseCurrentSource=True):
            study.GetCircuit().CreateSubCircuit("Star Connection", "Star Connection %d"%(poles), x, y) # è¿™äº›æ•°å­—æŒ‡çš„æ˜¯gridçš„ä¸ªæ•°ï¼Œç¬¬å‡ è¡Œç¬¬å‡ åˆ—çš„æ ¼ç‚¹å¤„
            study.GetCircuit().GetSubCircuit("Star Connection %d"%(poles)).GetComponent("Coil1").SetValue("Turn", turns)
            study.GetCircuit().GetSubCircuit("Star Connection %d"%(poles)).GetComponent("Coil1").SetValue("Resistance", Rs)
            study.GetCircuit().GetSubCircuit("Star Connection %d"%(poles)).GetComponent("Coil2").SetValue("Turn", turns)
            study.GetCircuit().GetSubCircuit("Star Connection %d"%(poles)).GetComponent("Coil2").SetValue("Resistance", Rs)
            study.GetCircuit().GetSubCircuit("Star Connection %d"%(poles)).GetComponent("Coil3").SetValue("Turn", turns)
            study.GetCircuit().GetSubCircuit("Star Connection %d"%(poles)).GetComponent("Coil3").SetValue("Resistance", Rs)
            study.GetCircuit().GetSubCircuit("Star Connection %d"%(poles)).GetComponent("Coil1").SetName("CircuitCoil%dU"%(poles))
            study.GetCircuit().GetSubCircuit("Star Connection %d"%(poles)).GetComponent("Coil2").SetName("CircuitCoil%dV"%(poles))
            study.GetCircuit().GetSubCircuit("Star Connection %d"%(poles)).GetComponent("Coil3").SetName("CircuitCoil%dW"%(poles))
            # Star Connection_2 is GroupAC
            # Star Connection_4 is GroupBD

            if bool_3PhaseCurrentSource == True: # must use this for frequency analysis

                study.GetCircuit().CreateComponent("3PhaseCurrentSource", "CS%d"%(poles))
                study.GetCircuit().CreateInstance("CS%d"%(poles), x-4, y+1)
                study.GetCircuit().GetComponent("CS%d"%(poles)).SetValue("Amplitude", ampD+ampB)
                study.GetCircuit().GetComponent("CS%d"%(poles)).SetValue("Frequency", "freq") # this is not needed for freq analysis # "freq" is a variable
                study.GetCircuit().GetComponent("CS%d"%(poles)).SetValue("PhaseU", phase)
                # Commutating sequence is essencial for the direction of the field to be consistent with speed: UVW rather than UWV
                study.GetCircuit().GetComponent("CS%d"%(poles)).SetValue("CommutatingSequence", CommutatingSequenceD) 
            else: 
                I1 = "CS%d-1"%(poles)
                I2 = "CS%d-2"%(poles)
                I3 = "CS%d-3"%(poles)
                study.GetCircuit().CreateComponent("CurrentSource", I1)
                study.GetCircuit().CreateInstance(                   I1, x-4, y+3)
                study.GetCircuit().CreateComponent("CurrentSource", I2)
                study.GetCircuit().CreateInstance(                   I2, x-4, y+1)
                study.GetCircuit().CreateComponent("CurrentSource", I3)
                study.GetCircuit().CreateInstance(                   I3, x-4, y-1)

                phase_shift_drive = -120 if CommutatingSequenceD == 1 else 120
                phase_shift_beari = -120 if CommutatingSequenceB == 1 else 120

                func = app.FunctionFactory().Composite()        
                f1 = app.FunctionFactory().Sin(ampD, freq, 0*phase_shift_drive) # "freq" variable cannot be used here. So pay extra attension here when you create new case of a different freq.
                f2 = app.FunctionFactory().Sin(ampB, freq, 0*phase_shift_beari)
                func.AddFunction(f1)
                func.AddFunction(f2)
                study.GetCircuit().GetComponent(I1).SetFunction(func)

                func = app.FunctionFactory().Composite()        
                f1 = app.FunctionFactory().Sin(ampD, freq, 1*phase_shift_drive)
                f2 = app.FunctionFactory().Sin(ampB, freq, 1*phase_shift_beari)
                func.AddFunction(f1)
                func.AddFunction(f2)
                study.GetCircuit().GetComponent(I2).SetFunction(func)

                func = app.FunctionFactory().Composite()
                f1 = app.FunctionFactory().Sin(ampD, freq, 2*phase_shift_drive)
                f2 = app.FunctionFactory().Sin(ampB, freq, 2*phase_shift_beari)
                func.AddFunction(f1)
                func.AddFunction(f2)
                study.GetCircuit().GetComponent(I3).SetFunction(func)

            study.GetCircuit().CreateComponent("Ground", "Ground")
            study.GetCircuit().CreateInstance("Ground", x+2, y+1)
        # 这里电流幅值中的0.5因子源自DPNV导致的等于2的平行支路数。没有考虑到这一点，是否会对initial design的有效性产生影响？
        # 仔细看DPNV的接线，对于转矩逆变器，绕组的并联支路数为2，而对于悬浮逆变器，绕组的并联支路数为1。

        npb = self.wily.number_parallel_branch
        nwl = self.wily.no_winding_layer # number of windign layers 
        # if self.fea_config_dict['DPNV_separate_winding_implementation'] == True or self.fea_config_dict['DPNV'] == False:
        if self.fea_config_dict['DPNV'] == False:
            # either a separate winding or a DPNV winding implemented as a separate winding
            ampD =  0.5 * (self.DriveW_CurrentAmp/npb + self.BeariW_CurrentAmp) # 为了代码能被四极电机和二极电机通用，代入看看就知道啦。
            ampB = -0.5 * (self.DriveW_CurrentAmp/npb - self.BeariW_CurrentAmp) # 关于符号，注意下面的DriveW对应的circuit调用时的ampB前还有个负号！
            if bool_3PhaseCurrentSource != True:
                raise Exception('Logic Error Detected.')
        else:
            # case: DPNV as an actual two layer winding
            ampD = self.DriveW_CurrentAmp/npb
            ampB = self.BeariW_CurrentAmp
            if bool_3PhaseCurrentSource != False:
                raise Exception('Logic Error Detected.')

        circuit(self.DriveW_poles,  self.DriveW_zQ/nwl, bool_3PhaseCurrentSource=bool_3PhaseCurrentSource,
            Rs=self.DriveW_Rs,ampD= ampD,
                              ampB=-ampB, freq=self.DriveW_Freq, phase=0,
                              CommutatingSequenceD=self.wily.CommutatingSequenceD,
                              CommutatingSequenceB=self.wily.CommutatingSequenceB)
        circuit(self.BeariW_poles,  self.BeariW_zQ/nwl, bool_3PhaseCurrentSource=bool_3PhaseCurrentSource,
            Rs=self.BeariW_Rs,ampD= ampD,
                              ampB=+ampB, freq=self.BeariW_Freq, phase=0,
                              CommutatingSequenceD=self.wily.CommutatingSequenceD,
                              CommutatingSequenceB=self.wily.CommutatingSequenceB,x=25) # CS4 corresponds to uauc (conflict with following codes but it does not matter.)

        # Link FEM Coils to Coil Set     
        # if self.fea_config_dict['DPNV_separate_winding_implementation'] == True or self.fea_config_dict['DPNV'] == False:
        if self.fea_config_dict['DPNV'] == False:
            def link_FEMCoils_2_CoilSet(poles,l1,l2):
                # link between FEM Coil Condition and Circuit FEM Coil
                for UVW in ['U','V','W']:
                    which_phase = "%d%s-Phase"%(poles,UVW)
                    study.CreateCondition("FEMCoil", which_phase)
                    condition = study.GetCondition(which_phase)
                    condition.SetLink("CircuitCoil%d%s"%(poles,UVW))
                    condition.GetSubCondition("untitled").SetName("Coil Set 1")
                    condition.GetSubCondition("Coil Set 1").SetName("delete")
                count = 0
                dict_dir = {'+':1, '-':0, 'o':None}
                # select the part to assign the FEM Coil condition
                for UVW, UpDown in zip(l1,l2):
                    count += 1 
                    if dict_dir[UpDown] is None:
                        # print 'Skip', UVW, UpDown
                        continue
                    which_phase = "%d%s-Phase"%(poles,UVW)
                    condition = study.GetCondition(which_phase)
                    condition.CreateSubCondition("FEMCoilData", "Coil Set %d"%(count))
                    subcondition = condition.GetSubCondition("Coil Set %d"%(count))
                    subcondition.ClearParts()
                    subcondition.AddSet(model.GetSetList().GetSet("Coil%d%s%s %d"%(poles,UVW,UpDown,count)), 0)
                    subcondition.SetValue("Direction2D", dict_dir[UpDown])
                # clean up
                for UVW in ['U','V','W']:
                    which_phase = "%d%s-Phase"%(poles,UVW)
                    condition = study.GetCondition(which_phase)
                    condition.RemoveSubCondition("delete")
            link_FEMCoils_2_CoilSet(self.DriveW_poles, 
                                    self.dict_coil_connection[int(self.DriveW_poles*10+1)], # 40 for 4 poles, +1 for UVW, 
                                    self.dict_coil_connection[int(self.DriveW_poles*10+2)])                 # +2 for up or down, 
            link_FEMCoils_2_CoilSet(self.BeariW_poles, 
                                    self.dict_coil_connection[int(self.BeariW_poles*10+1)], # 20 for 2 poles, +1 for UVW, .
                                    self.dict_coil_connection[int(self.BeariW_poles*10+2)])                 # +2 for up or down,  这里的2和4等价于leftlayer和rightlayer。
        else:
            # 两个改变，一个是激励大小的改变（本来是200A 和 5A，现在是205A和195A），
            # 另一个绕组分组的改变，现在的A相是上层加下层为一相，以前是用俩单层绕组等效的。

            # Link FEM Coils to Coil Set as double layer short pitched winding
            # Create FEM Coil Condition
            # here we map circuit component `Coil2A' to FEM Coil Condition 'phaseAuauc
            # here we map circuit component `Coil4A' to FEM Coil Condition 'phaseAubud
            for suffix, poles in zip(['GroupAC', 'GroupBD'], [2, 4]): # 仍然需要考虑poles，是因为为Coil设置Set那里的代码还没有更新。这里的2和4等价于leftlayer和rightlayer。
                for UVW in ['U','V','W']:
                    study.CreateCondition("FEMCoil", 'phase'+UVW+suffix)
                    # link between FEM Coil Condition and Circuit FEM Coil
                    condition = study.GetCondition('phase'+UVW+suffix)
                    condition.SetLink("CircuitCoil%d%s"%(poles,UVW))
                    condition.GetSubCondition("untitled").SetName("delete")
            count = 0 # count indicates which slot the current rightlayer is in.
            index = 0
            dict_dir = {'+':1, '-':0}
            coil_pitch = self.wily.coil_pitch #self.dict_coil_connection[0]
            # select the part (via `Set') to assign the FEM Coil condition
            for UVW, UpDown in zip(self.wily.l_rightlayer1, self.wily.l_rightlayer2):

                count += 1 
                if self.wily.grouping_AC[index] == 1:
                    suffix = 'GroupAC'
                else:
                    suffix = 'GroupBD'
                condition = study.GetCondition('phase'+UVW+suffix)

                # right layer
                # print (count, "Coil Set %d"%(count), end=' ')
                condition.CreateSubCondition("FEMCoilData", "Coil Set Right %d"%(count))
                subcondition = condition.GetSubCondition("Coil Set Right %d"%(count))
                subcondition.ClearParts()
                subcondition.AddSet(model.GetSetList().GetSet("Coil%d%s%s %d"%(4,UVW,UpDown,count)), 0) # poles=4 means right layer, rather than actual poles
                subcondition.SetValue("Direction2D", dict_dir[UpDown])

                # left layer
                if coil_pitch > 0:
                    if count+coil_pitch <= self.Q:
                        count_leftlayer = count+coil_pitch
                        index_leftlayer = index+coil_pitch
                    else:
                        count_leftlayer = int(count+coil_pitch - self.Q)
                        index_leftlayer = int(index+coil_pitch - self.Q)
                else:
                    if count+coil_pitch > 0:
                        count_leftlayer = count+coil_pitch
                        index_leftlayer = index+coil_pitch
                    else:
                        count_leftlayer = int(count+coil_pitch + self.Q)
                        index_leftlayer = int(index+coil_pitch + self.Q)

                # Check if it is a distributed windg???
                if self.wily.distributed_or_concentrated == False:
                    print('Concentrated winding!')
                    UVW    = self.wily.l_leftlayer1[index_leftlayer]
                    UpDown = self.wily.l_leftlayer2[index_leftlayer]
                else:
                    # print('Distributed winding.')
                    if self.wily.l_leftlayer1[index_leftlayer] != UVW:
                        raise Exception('But in winding.')
                    # 右层导体的电流方向是正，那么与其串联的一个coil_pitch之处的左层导体就是负！不需要再检查l_leftlayer2了~
                    if UpDown == '+': 
                        UpDown = '-'
                    else:
                        UpDown = '+'
                # print (count_leftlayer, "Coil Set %d"%(count_leftlayer))
                condition.CreateSubCondition("FEMCoilData", "Coil Set Left %d"%(count_leftlayer))
                subcondition = condition.GetSubCondition("Coil Set Left %d"%(count_leftlayer))
                subcondition.ClearParts()
                subcondition.AddSet(model.GetSetList().GetSet("Coil%d%s%s %d"%(2,UVW,UpDown,count_leftlayer)), 0) # poles=2 means left layer, rather than actual poles
                subcondition.SetValue("Direction2D", dict_dir[UpDown])
                # print 'coil_pitch=', coil_pitch
                # print l_rightlayer1[index], UVW
                # print l_leftlayer1[index_leftlayer]
                # print l_rightlayer1
                # print l_leftlayer1
                index += 1
            # clean up
            for suffix in ['GroupAC', 'GroupBD']:
                for UVW in ['U','V','W']:
                    condition = study.GetCondition('phase'+UVW+suffix)
                    condition.RemoveSubCondition("delete")
            # raise Exception('Test DPNV PE.')

    def show(self, toString=False):
        attrs = list(vars(self).items())
        key_list = [el[0] for el in attrs]
        val_list = [el[1] for el in attrs]
        the_dict = dict(list(zip(key_list, val_list)))
        sorted_key = sorted(key_list, key=lambda item: (int(item.partition(' ')[0]) if item[0].isdigit() else float('inf'), item)) # this is also useful for string beginning with digiterations '15 Steel'.
        tuple_list = [(key, the_dict[key]) for key in sorted_key]
        if toString==False:
            print('- Bearingless PMSM Individual #%s\n\t' % (self.name), end=' ')
            print(', \n\t'.join("%s = %s" % item for item in tuple_list))
            return ''
        else:
            return '\n- Bearingless PMSM Individual #%s\n\t' % (self.name) + ', \n\t'.join("%s = %s" % item for item in tuple_list)

    def get_individual_name(self):
        if self.fea_config_dict['flag_optimization'] == True:
            return "ID%s" % (self.ID)
        else:
            return "%s_ID%s" % (self.model_name_prefix, self.ID)


# circumferential segmented rotor 
if __name__ == '__main__':
    import JMAG
    import Location2D
    import CrossSectInnerNotchedRotor
    import CrossSectStator
    # from pylab import np

    if True:
        from utility import my_execfile
        my_execfile('./default_setting.py', g=globals(), l=locals())
        fea_config_dict
        toolJd = JMAG.JMAG(fea_config_dict)

        project_name          = 'proj%d'%(0)
        expected_project_file_path = './' + "%s.jproj"%(project_name)
        toolJd.open(expected_project_file_path)

    spmsm_template = bearingless_spmsm_template()
    spmsm_template.fea_config_dict = fea_config_dict
    Q = 6

    spmsm_template.deg_alpha_st = 360/Q*0.8   # deg_alpha_st # span angle of tooth: class type DimAngular
    spmsm_template.deg_alpha_so = 0                          # deg_alpha_so # angle of tooth edge: class type DimAngular
    spmsm_template.mm_r_si      = 50     # mm_r_si           # inner radius of stator teeth: class type DimLinear
    spmsm_template.mm_d_so      = 5      # mm_d_so           # tooth edge length: class type DimLinear
    spmsm_template.mm_d_sp      = 1.5*spmsm_template.mm_d_so # mm_d_sp      # tooth tip length: class type DimLinear
    spmsm_template.mm_d_st      = 15     # mm_d_st      # tooth base length: class type DimLinear
    spmsm_template.mm_d_sy      = 15     # mm_d_sy      # back iron thickness: class type DimLinear
    spmsm_template.mm_w_st      = 13     # mm_w_st      # tooth base width: class type DimLinear
    spmsm_template.mm_r_st      = 0         # mm_r_st      # fillet on outter tooth: class type DimLinear
    spmsm_template.mm_r_sf      = 0         # mm_r_sf      # fillet between tooth tip and base: class type DimLinear
    spmsm_template.mm_r_sb      = 0         # mm_r_sb      # fillet at tooth base: class type DimLinear
    spmsm_template.Q            = 6      # number of stator slots (integer)
    spmsm_template.sleeve_length        = 2 # mm
    spmsm_template.fixed_air_gap_length = 0.75 # mm
    spmsm_template.mm_d_pm      = 6      # mm_d_pm          # manget depth
    spmsm_template.deg_alpha_rm = 60     # deg_alpha_rm     # angular span of the pole: class type DimAngular
    spmsm_template.deg_alpha_rs = 10     # deg_alpha_rs     # segment span: class type DimAngular
    spmsm_template.mm_d_ri      = 8      # mm_d_ri          # inner radius of rotor: class type DimLinear
    spmsm_template.mm_r_ri      = 40     # mm_r_ri          # rotor iron thickness: class type DimLinear
    spmsm_template.mm_d_rp      = 5      # mm_d_rp          # interpolar iron thickness: class type DimLinear
    spmsm_template.mm_d_rs      = 3      # mm_d_rs          # inter segment iron thickness: class type DimLinear
    spmsm_template.p = 2     # p     # number of pole pairs
    spmsm_template.s = 3     # s     # number of segments  

    spmsm_template.build_design_parameters_list()

    spmsm_template.DriveWinding_Freq       = 1000
    spmsm_template.DriveWinding_Rs         = 0.1 # TODO
    spmsm_template.DriveWinding_zQ         = 1
    spmsm_template.DriveWinding_CurrentAmp = None # this means it depends on the slot area
    spmsm_template.DriveWinding_poles = 2*spmsm_template.p

    spmsm_template.Js = 4e6 # Arms/m^2
    spmsm_template.fill_factor = 0.45

    spmsm_template.stack_length = 100 # mm

    # logger = logging.getLogger(__name__) 
    # logger.info('spmsm_variant ID %s is initialized.', self.name)

    spmsm = bearingless_spmsm_design(   spmsm_template=spmsm_template,
                                        free_variables=None,
                                        counter=None, 
                                        counter_loop=None
                                        )
    # Rotor Core
    list_segments = spmsm.rotorCore.draw(toolJd)
    toolJd.bMirror = False
    toolJd.iRotateCopy = 0 #spmsm.rotorCore.p*2
    region1 = toolJd.prepareSection(list_segments)

    # Rotor Magnet    
    list_regions = spmsm.rotorMagnet.draw(toolJd)
    toolJd.bMirror = False
    toolJd.iRotateCopy = 0 #spmsm.rotorMagnet.notched_rotor.p*2
    region2 = toolJd.prepareSection(list_regions)

    # Import Model into Designer
    toolJd.doc.SaveModel(False) # True: Project is also saved. 
    model = toolJd.app.GetCurrentModel()
    model.SetName('BPMSM Modeling')
    model.SetDescription('BPMSM Test')


# notched rotor 
if __name__ == '__main__':
    import JMAG
    import Location2D
    import CrossSectInnerNotchedRotor
    import CrossSectStator
    # from pylab import np

    if True:
        from utility import my_execfile
        my_execfile('./default_setting.py', g=globals(), l=locals())
        fea_config_dict
        toolJd = JMAG.JMAG(fea_config_dict)

        project_name          = 'proj%d'%(0)
        expected_project_file_path = './' + "%s.jproj"%(project_name)
        toolJd.open(expected_project_file_path)

    spmsm_template = bearingless_spmsm_template()
    spmsm_template.fea_config_dict = fea_config_dict
    Q = 6

    spmsm_template.deg_alpha_st = 360/Q*0.8   # deg_alpha_st # span angle of tooth: class type DimAngular
    spmsm_template.deg_alpha_so = 0                          # deg_alpha_so # angle of tooth edge: class type DimAngular
    spmsm_template.mm_r_si      = 50     # mm_r_si           # inner radius of stator teeth: class type DimLinear
    spmsm_template.mm_d_so      = 5      # mm_d_so           # tooth edge length: class type DimLinear
    spmsm_template.mm_d_sp      = 1.5*spmsm_template.mm_d_so # mm_d_sp      # tooth tip length: class type DimLinear
    spmsm_template.mm_d_st      = 15     # mm_d_st      # tooth base length: class type DimLinear
    spmsm_template.mm_d_sy      = 15     # mm_d_sy      # back iron thickness: class type DimLinear
    spmsm_template.mm_w_st      = 13     # mm_w_st      # tooth base width: class type DimLinear
    spmsm_template.mm_r_st      = 0         # mm_r_st      # fillet on outter tooth: class type DimLinear
    spmsm_template.mm_r_sf      = 0         # mm_r_sf      # fillet between tooth tip and base: class type DimLinear
    spmsm_template.mm_r_sb      = 0         # mm_r_sb      # fillet at tooth base: class type DimLinear
    spmsm_template.Q            = 6      # number of stator slots (integer)
    spmsm_template.sleeve_length        = 2 # mm
    spmsm_template.fixed_air_gap_length = 0.75 # mm
    spmsm_template.mm_d_pm      = 6      # mm_d_pm          # manget depth
    spmsm_template.deg_alpha_rm = 60     # deg_alpha_rm     # angular span of the pole: class type DimAngular
    spmsm_template.deg_alpha_rs = 60     # deg_alpha_rs     # segment span: class type DimAngular
    spmsm_template.mm_d_ri      = 8      # mm_d_ri          # inner radius of rotor: class type DimLinear
    spmsm_template.mm_r_ri      = 40     # mm_r_ri          # rotor iron thickness: class type DimLinear
    spmsm_template.mm_d_rp      = 5      # mm_d_rp          # interpolar iron thickness: class type DimLinear
    spmsm_template.mm_d_rs      = 0*3      # mm_d_rs          # inter segment iron thickness: class type DimLinear
    spmsm_template.p = 2     # p     # number of pole pairs
    spmsm_template.s = 1     # s     # number of segments  

    spmsm_template.build_design_parameters_list()

    spmsm_template.DriveWinding_Freq       = 1000
    spmsm_template.DriveWinding_Rs         = 0.1 # TODO
    spmsm_template.DriveWinding_zQ         = 1
    spmsm_template.DriveWinding_CurrentAmp = None # this means it depends on the slot area
    spmsm_template.DriveWinding_poles = 2*spmsm_template.p

    spmsm_template.Js = 4e6 # Arms/m^2
    spmsm_template.fill_factor = 0.45

    spmsm_template.stack_length = 100 # mm

    # logger = logging.getLogger(__name__) 
    # logger.info('spmsm_variant ID %s is initialized.', self.name)

    spmsm = bearingless_spmsm_design(   spmsm_template=spmsm_template,
                                        free_variables=None,
                                        counter=None, 
                                        counter_loop=None
                                        )
    # Rotor Core
    list_segments = spmsm.rotorCore.draw(toolJd)
    toolJd.bMirror = False
    toolJd.iRotateCopy = spmsm.rotorCore.p*2
    region1 = toolJd.prepareSection(list_segments)

    # Rotor Magnet    
    list_regions = spmsm.rotorMagnet.draw(toolJd)
    toolJd.bMirror = False
    toolJd.iRotateCopy = spmsm.rotorMagnet.notched_rotor.p*2
    region2 = toolJd.prepareSection(list_regions)

    # Rotor Magnet
    sleeve = CrossSectInnerNotchedRotor.CrossSectSleeve(
                    name = 'Sleeve',
                    notched_magnet = spmsm.rotorMagnet,
                    d_sleeve = spmsm_template.sleeve_length
                    )

    list_regions = sleeve.draw(toolJd)
    toolJd.bMirror = False
    toolJd.iRotateCopy = spmsm.rotorMagnet.notched_rotor.p*2
    regionS = toolJd.prepareSection(list_regions)


    # # Stator Core
    # list_regions = spmsm.stator_core.draw(toolJd)
    # toolJd.bMirror = True
    # toolJd.iRotateCopy = spmsm.stator_core.Q
    # region1 = toolJd.prepareSection(list_regions)

    # # Stator Winding
    # list_regions = spmsm.coils.draw(toolJd)
    # toolJd.bMirror = False
    # toolJd.iRotateCopy = spmsm.coils.stator_core.Q
    # region2 = toolJd.prepareSection(list_regions)

    # Import Model into Designer
    toolJd.doc.SaveModel(False) # True: Project is also saved. 
    model = toolJd.app.GetCurrentModel()
    model.SetName('BPMSM Modeling')
    model.SetDescription('BPMSM Test')


def add_carbon_fiber_material(app):
    app.GetMaterialLibrary().CreateCustomMaterial(u"CarbonFiber", u"Custom Materials")
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"Density", 1.6)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"CoerciveForce", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"DemagnetizationCoerciveForce", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"MagnetizationSaturated", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"MagnetizationSaturated2", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"YoungModulus", 110000)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"ShearModulus", 5000)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"PoissonRatio", 0.1)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"Thermal Expansion", 8.4e-06)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"YoungModulusX", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"YoungModulusY", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"YoungModulusZ", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"ShearModulusXY", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"ShearModulusYZ", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"ShearModulusZX", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G11", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G12", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G13", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G14", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G15", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G16", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G22", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G23", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G24", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G25", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G26", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G33", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G34", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G35", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G36", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G44", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G45", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G46", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G55", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G56", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"G66", 0)





    # -*- coding: utf-8 -*-
    app = designer.GetApplication()
    app.GetMaterialLibrary().CopyMaterial(u"Arnold/Reversible/N40H")
    app.GetMaterialLibrary().GetUserMaterial(u"N40H(reversible) copy").SetValue(u"Name", u"MyN40H(reversible)")
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"Density", 7.5)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"CoerciveForce", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"DemagnetizationCoerciveForce", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"MagnetizationSaturated", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"MagnetizationSaturated2", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"MagnetizationSaturatedMakerValue", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"YoungModulus", 160000)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"PoissonRatio", 0.24)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"YoungModulusX", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"YoungModulusY", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"YoungModulusZ", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"ShearModulusXY", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"ShearModulusYZ", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"ShearModulusZX", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G11", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G12", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G13", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G14", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G15", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G16", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G22", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G23", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G24", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G25", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G26", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G33", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G34", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G35", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G36", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G44", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G45", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G46", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G55", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G56", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"MyN40H(reversible)").SetValue(u"G66", 0)



