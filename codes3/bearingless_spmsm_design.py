import CrossSectInnerNotchedRotor
import CrossSectStator
import Location2D

class bearingless_spmsm_template(object):
    def __init__(self, model_name_prefix='SPMSM', fea_config_dict=None):
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

    def get_bounds(self):
        return
        # <360/Q
        # >= 0
        # r_ri + L_g

class bpmsm_specification(object):
    def __init__(self,
                    PS_or_SC = None,
                    DPNV_or_SEPA = None,
                    p  = None,
                    ps = None,
                    mec_power = None,
                    ExcitationFreq = None,
                    ExcitationFreqSimulated = None,
                    VoltageRating = None,
                    TangentialStress = None,
                    Qs = None,
                    segment = None,
                    Js = None,
                    Jr = None,
                    Coil = None,
                    space_factor_kCu = None,
                    Conductor = None,
                    space_factor_kAl = None,
                    Temperature = None,
                    Steel = None,
                    lamination_stacking_factor_kFe = None,
                    stator_tooth_flux_density_B_ds = None,
                    rotor_tooth_flux_density_B_dr = None,
                    stator_yoke_flux_density_Bys = None,
                    rotor_yoke_flux_density_Byr = None,
                    guess_air_gap_flux_density = None,
                    guess_efficiency = None,
                    guess_power_factor = None,
                    safety_factor_to_yield = None,
                    safety_factor_to_critical_speed = None,
                    use_drop_shape_rotor_bar = None,
                    tip_speed = None,
                    debug_or_release= None,
                    bool_skew_stator = None,
                    bool_skew_rotor = None,
                ):
        self.DPNV_or_SEPA = DPNV_or_SEPA
        self.p = p
        self.ps = ps
        self.mec_power = mec_power
        self.ExcitationFreq = ExcitationFreq
        self.ExcitationFreqSimulated = ExcitationFreqSimulated
        self.VoltageRating = VoltageRating
        self.TangentialStress = TangentialStress
        self.Qs = Qs
        self.segment = segment
        self.Js = Js
        self.Jr = Jr
        self.Jr_backup = self.Jr
        self.Coil = Coil
        self.space_factor_kCu = space_factor_kCu
        self.Conductor = Conductor
        self.space_factor_kAl = space_factor_kAl
        self.Temperature = Temperature
        self.Steel = Steel
        self.lamination_stacking_factor_kFe = lamination_stacking_factor_kFe
        self.stator_tooth_flux_density_B_ds = stator_tooth_flux_density_B_ds
        self.rotor_tooth_flux_density_B_dr = rotor_tooth_flux_density_B_dr
        self.stator_yoke_flux_density_Bys = stator_yoke_flux_density_Bys
        self.rotor_yoke_flux_density_Byr = rotor_yoke_flux_density_Byr
        self.guess_air_gap_flux_density = guess_air_gap_flux_density
        self.guess_efficiency = guess_efficiency
        self.guess_power_factor = guess_power_factor
        self.safety_factor_to_yield = safety_factor_to_yield
        self.safety_factor_to_critical_speed = safety_factor_to_critical_speed
        self.use_drop_shape_rotor_bar = use_drop_shape_rotor_bar
        self.tip_speed = tip_speed
        self.debug_or_release = debug_or_release
        self.bool_skew_stator = bool_skew_stator
        self.bool_skew_rotor  = bool_skew_rotor 

        # self.winding_layout = winding_layout(self.DPNV_or_SEPA, self.Qs, self.p)

        self.bool_high_speed_design = self.tip_speed is not None

        # if not os.path.isdir('../' + 'pop/'):
        #     os.mkdir('../' + 'pop/')
        # self.loc_txt_file = '../' + 'pop/' + r'initial_design.txt'
        # open(self.loc_txt_file, 'w').close() # clean slate to begin with

    def build_acm_template(self, fea_config_dict):
        
        acm_template = bearingless_spmsm_template(fea_config_dict=fea_config_dict)
        Q = 6

        acm_template.deg_alpha_st = 360/Q*0.8   # deg_alpha_st # span angle of tooth: class type DimAngular
        acm_template.deg_alpha_so = 0                          # deg_alpha_so # angle of tooth edge: class type DimAngular
        acm_template.mm_r_si      = 50   # mm_r_si           # inner radius of stator teeth: class type DimLinear
        acm_template.mm_d_so      = 5      # mm_d_so           # tooth edge length: class type DimLinear
        acm_template.mm_d_sp      = 1.5*acm_template.mm_d_so # mm_d_sp      # tooth tip length: class type DimLinear
        acm_template.mm_d_st      = 15     # mm_d_st      # tooth base length: class type DimLinear
        acm_template.mm_d_sy      = 15     # mm_d_sy      # back iron thickness: class type DimLinear
        acm_template.mm_w_st      = 13     # mm_w_st      # tooth base width: class type DimLinear
        acm_template.mm_r_st      = 0         # mm_r_st      # fillet on outter tooth: class type DimLinear
        acm_template.mm_r_sf      = 0         # mm_r_sf      # fillet between tooth tip and base: class type DimLinear
        acm_template.mm_r_sb      = 0         # mm_r_sb      # fillet at tooth base: class type DimLinear
        acm_template.Q            = 6      # number of stator slots (integer)
        acm_template.sleeve_length        = 2 # mm
        acm_template.fixed_air_gap_length = 0.75 # mm
        acm_template.mm_d_pm      = 6      # mm_d_pm          # manget depth
        acm_template.deg_alpha_rm = 60     # deg_alpha_rm     # angular span of the pole: class type DimAngular
        acm_template.deg_alpha_rs = 60     # deg_alpha_rs     # segment span: class type DimAngular
        acm_template.mm_d_ri      = 8      # mm_d_ri          # inner radius of rotor: class type DimLinear
        acm_template.mm_r_ri      = 40     # mm_r_ri          # rotor iron thickness: class type DimLinear
        acm_template.mm_d_rp      = 5      # mm_d_rp          # interpolar iron thickness: class type DimLinear
        acm_template.mm_d_rs      = 0*3      # mm_d_rs          # inter segment iron thickness: class type DimLinear
        acm_template.p = 2     # p     # number of pole pairs
        acm_template.s = 1     # s     # number of segments  

        acm_template.build_design_parameters_list()

        acm_template.driveWinding_Freq       = 1000
        acm_template.driveWinding_Rs         = 0.1 # TODO
        acm_template.driveWinding_zQ         = 1
        acm_template.driveWinding_CurrentAmp = None # this means it depends on the slot area
        acm_template.driveWinding_poles = 2*acm_template.p

        acm_template.Js = 4e6 # Arms/m^2
        acm_template.fill_factor = 0.45

        acm_template.stack_length = 100 # mm

        # logger = logging.getLogger(__name__) 
        # logger.info('spmsm_variant ID %s is initialized.', self.ID)


        # 让儿子能访问爸爸
        self.acm_template = acm_template
        self.acm_template.spec = self

class bearingless_spmsm_design(bearingless_spmsm_template):

    def __init__(self, spmsm_template=None, free_variables=None, counter=None, counter_loop=None):
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

        # self.spec = spmsm_template.spec

        #02 Geometry Data
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
        #                                  11  spmsm_template.Q            
        #                                  12  spmsm_template.sleeve_length
        #                                  13  spmsm_template.fixed_air_gap_length
        #                                  14  spmsm_template.mm_d_pm      
        #                                  15  spmsm_template.deg_alpha_rm 
        #                                  16  spmsm_template.deg_alpha_rs 
        #                                  17  spmsm_template.mm_d_ri      
        #                                  18  spmsm_template.mm_r_ri      
        #                                  19  spmsm_template.mm_d_rp      
        #                                  20  spmsm_template.mm_d_rs      
        #                                  21  spmsm_template.p
        #                                  22  spmsm_template.s
        #                                 ]
        if free_variables is None:
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

        deg_alpha_st        = free_variables[0]
        mm_d_so             = free_variables[1]
        mm_d_st             = free_variables[2]
        stator_outer_radius = free_variables[3]
        mm_w_st             = free_variables[4]
        sleeve_length       = free_variables[5]
        mm_d_pm             = free_variables[6]
        deg_alpha_rm        = free_variables[7]
        deg_alpha_rs        = free_variables[8]
        mm_d_ri             = free_variables[9]
        rotor_outer_radius  = free_variables[10] 
        mm_d_rp             = free_variables[11]
        mm_d_rs             = free_variables[12]

        self.deg_alpha_st = free_variables[0]
        self.deg_alpha_so = spmsm_template.deg_alpha_so
        self.mm_r_si      = rotor_outer_radius + (mm_d_pm - mm_d_rp) + sleeve_length + spmsm_template.fixed_air_gap_length
        self.mm_d_so      = free_variables[1]
        self.mm_d_sp      = spmsm_template.mm_d_sp # 0.5*mm_d_so
        self.mm_d_st      = free_variables[2]; stator_outer_radius = free_variables[3]
        self.mm_d_sy      = stator_outer_radius - spmsm_template.mm_d_sp - mm_d_st
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
        self.mm_d_ri      = free_variables[9]; rotor_outer_radius = free_variables[10]
        self.mm_r_ri      = rotor_outer_radius - mm_d_rp - mm_d_ri
        self.mm_d_rp      = free_variables[11]
        self.mm_d_rs      = free_variables[12]
        self.p = spmsm_template.p
        self.s = spmsm_template.s

        design_parameters = self.build_design_parameters_list()

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

        # #03 Mechanical Parameters
        # self.update_mechanical_parameters(slip_freq=50.0) #, syn_freq=500.)

        # #04 Material Condutivity Properties
        # if self.fea_config_dict is not None:
        #     self.End_Ring_Resistance = fea_config_dict['End_Ring_Resistance']
        #     self.Bar_Conductivity = fea_config_dict['Bar_Conductivity']
        # self.Copper_Loss = self.DriveW_CurrentAmp**2 / 2 * self.DriveW_Rs * 3
        # # self.Resistance_per_Turn = 0.01 # TODO


        # #05 Windings & Excitation
        # if self.fea_config_dict is not None:
        #     self.wily = winding_layout(self.fea_config_dict['DPNV'], self.Qs, self.DriveW_poles/2)


        # if self.DriveW_poles == 2:
        #     self.BeariW_poles = 4
        #     if self.DriveW_turns % 2 != 0:
        #         print('zQ=', self.DriveW_turns)
        #         raise Exception('This zQ does not suit for two layer winding.')
        # elif self.DriveW_poles == 4:
        #     self.BeariW_poles = 2;
        # else:
        #     raise Exception('Not implemented error.')
        # self.BeariW_turns      = self.DriveW_turns
        # self.BeariW_Rs         = self.DriveW_Rs * self.BeariW_turns / self.DriveW_turns
        # self.BeariW_CurrentAmp = 0.025 * self.DriveW_CurrentAmp/0.975 # extra 2.5% as bearing current
        # self.BeariW_Freq       = self.DriveW_Freq

        # if self.fea_config_dict is not None:
        #     self.dict_coil_connection = {41:self.wily.l41, 42:self.wily.l42, 21:self.wily.l21, 22:self.wily.l22} # 这里的2和4等价于leftlayer和rightlayer。

        # #06 Meshing & Solver Properties
        # self.max_nonlinear_iteration = 50 # 30 for transient solve
        # self.meshSize_Rotor = 1.8 #1.2 0.6 # mm


        # self.RSH = []
        # for pm in [-1, +1]:
        #     for v in [-1, +1]:
        #             self.RSH.append( (pm * self.Qr*(1-self.the_slip)/(0.5*self.DriveW_poles) + v)*self.DriveW_Freq )
        # print self.Qr, ', '.join("%g" % (rsh/self.DriveW_Freq) for rsh in self.RSH), '\n'

    def update_mechanical_parameters(self, syn_freq=None):
        if syn_freq is None:
            self.the_speed = self.DriveW_Freq*60. / (0.5*self.DriveW_poles) # rpm
            self.Omega = + self.the_speed / 60. * 2*pi
            self.omega = None # This variable name is devil! you can't tell its electrical or mechanical! #+ self.DriveW_Freq * (1-self.the_slip) * 2*pi
        else:
            raise Exception('Not implemented.')

    def draw_spmsm(self, toolJd):

        # Rotor Core
        list_segments = self.rotorCore.draw(toolJd)
        toolJd.bMirror = False
        toolJd.iRotateCopy = self.rotorCore.p*2
        region1 = toolJd.prepareSection(list_segments)

        # Rotor Magnet    
        list_regions = self.rotorMagnet.draw(toolJd)
        toolJd.bMirror = False
        toolJd.iRotateCopy = self.rotorMagnet.notched_rotor.p*2
        region2 = toolJd.prepareSection(list_regions)


        # Rotor Magnet
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

        part_ID_list = model.GetPartIDs()
        # view = app.View()
        # view.ClearSelect()
        # sel = view.GetCurrentSelection()
        # sel.SelectPart(123)
        # sel.SetBlockUpdateView(False)

        print(part_ID_list)
        raise KeyboardInterrupt

        if len(part_ID_list) != int(1 + 1 + 1 + self.Qr + self.Qs*2):
            msg = 'Number of Parts is unexpected. Should be %d but only %d.\n'%(1 + 1 + 1 + self.Qr + self.Qs*2, len(part_ID_list)) + self.show(toString=True)
            # utility.send_notification(text=msg)
            # return msg
            raise ExceptionBadNumberOfParts(msg)

        id_shaft = part_ID_list[0]
        id_rotorCore = part_ID_list[1]
        partIDRange_Cage = part_ID_list[2 : 2+int(self.Qr)]
        id_statorCore = part_ID_list[3+int(self.Qr)]
        partIDRange_Coil = part_ID_list[3+int(self.Qr) : 3+int(self.Qr) + int(self.Qs*2)]
        # partIDRange_AirWithinRotorSlots = part_ID_list[3+int(self.Qr) + int(self.Qs*2) : 3+int(self.Qr) + int(self.Qs*2) + int(self.Qr)]

        # print part_ID_list
        # print partIDRange_Cage
        # print partIDRange_Coil
        # print partIDRange_AirWithinRotorSlots
        group("Cage", partIDRange_Cage) # 59-44 = 15 = self.Qr - 1
        group("Coil", partIDRange_Coil) # 107-60 = 47 = 48-1 = self.Qs*2 - 1
        # group(u"AirWithinRotorSlots", partIDRange_AirWithinRotorSlots) # 123-108 = 15 = self.Qr - 1


        ''' Add Part to Set for later references '''
        def part_set(name, x, y):
            model.GetSetList().CreatePartSet(name)
            model.GetSetList().GetSet(name).SetMatcherType("Selection")
            model.GetSetList().GetSet(name).ClearParts()
            sel = model.GetSetList().GetSet(name).GetSelection()
            # print x,y
            sel.SelectPartByPosition(x,y,0) # z=0 for 2D
            model.GetSetList().GetSet(name).AddSelected(sel)

        # def edge_set(name,x,y):
        #     model.GetSetList().CreateEdgeSet(name)
        #     model.GetSetList().GetSet(name).SetMatcherType(u"Selection")
        #     model.GetSetList().GetSet(name).ClearParts()
        #     sel = model.GetSetList().GetSet(name).GetSelection()
        #     sel.SelectEdgeByPosition(x,y,0) # sel.SelectEdge(741)
        #     model.GetSetList().GetSet(name).AddSelected(sel)
        # edge_set(u"AirGapCoast", 0, self.Radius_OuterRotor+0.5*self.Length_AirGap)

        # Create Set for Shaft
        part_set("ShaftSet", 0.0, 0.0)

        # Create Set for 4 poles Winding
        R = 0.5*(self.Radius_InnerStatorYoke  +  (self.Radius_OuterRotor+self.Width_StatorTeethHeadThickness+self.Width_StatorTeethNeck)) 
            # THETA = (0.5*(self.Angle_StatorSlotSpan) -  0.05*(self.Angle_StatorSlotSpan-self.Angle_StatorSlotOpen))/180.*pi
        THETA = (0.5*(self.Angle_StatorSlotSpan) -  0.05*(self.Angle_StatorSlotSpan-self.Width_StatorTeethBody))/180.*pi
        X = R*cos(THETA)
        Y = R*sin(THETA)
        # l41=[ 'C', 'C', 'A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B', ]
        # l42=[ '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', '+', '+', '-', '-', ]
        count = 0
        for UVW, UpDown in zip(self.wily.l41,self.wily.l42):
            count += 1 
            part_set("Coil4%s%s %d"%(UVW,UpDown,count), X, Y)

            THETA += self.Angle_StatorSlotSpan/180.*pi
            X = R*cos(THETA)
            Y = R*sin(THETA)

        # Create Set for 2 poles Winding
            # THETA = (0.5*(self.Angle_StatorSlotSpan) +  0.05*(self.Angle_StatorSlotSpan-self.Angle_StatorSlotOpen))/180.*pi
        THETA = (0.5*(self.Angle_StatorSlotSpan) +  0.05*(self.Angle_StatorSlotSpan-self.Width_StatorTeethBody))/180.*pi
        X = R*cos(THETA)
        Y = R*sin(THETA)
        # l21=[ 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'A', 'A', ]
        # l22=[ '-', '-', '+', '+', '+', '+', '-', '-', '-', '-', '+', '+', '+', '+', '-', '-', '-', '-', '+', '+', '+', '+', '-', '-', ]
        count = 0
        for UVW, UpDown in zip(self.wily.l21,self.wily.l22):
            count += 1 
            part_set("Coil2%s%s %d"%(UVW,UpDown,count), X, Y)

            THETA += self.Angle_StatorSlotSpan/180.*pi
            X = R*cos(THETA)
            Y = R*sin(THETA)



        # Create Set for Bars and Air within rotor slots
        R = self.Location_RotorBarCenter
                                                                                # Another BUG:对于这种槽型，part_set在将AirWithin添加为set的时候，会错误选择到转子导条上！实际上，AirWithinRotorSlots对于JMAG来说完全是没有必要的！
        # R_airR = self.Radius_OuterRotor - 0.1*self.Length_HeadNeckRotorSlot # if the airWithin is too big, minus EPS is not enough anymore
        THETA = pi # it is very important (when Qr is odd) to begin the part set assignment from the first bar you plot.
        X = R*cos(THETA)
        Y = R*sin(THETA)
        list_xy_bars = []
        # list_xy_airWithinRotorSlot = []
        for ind in range(int(self.Qr)):
            natural_ind = ind + 1
            # print THETA / pi *180
            part_set("Bar %d"%(natural_ind), X, Y)
            list_xy_bars.append([X,Y])
            # # # part_set(u"AirWithin %d"%(natural_ind), R_airR*cos(THETA),R_airR*sin(THETA))
            # list_xy_airWithinRotorSlot.append([R_airR*cos(THETA),R_airR*sin(THETA)])

            THETA += self.Angle_RotorSlotSpan/180.*pi
            X = R*cos(THETA)
            Y = R*sin(THETA)

        # Create Set for Motion Region
        def part_list_set(name, list_xy, prefix=None):
            model.GetSetList().CreatePartSet(name)
            model.GetSetList().GetSet(name).SetMatcherType("Selection")
            model.GetSetList().GetSet(name).ClearParts()
            sel = model.GetSetList().GetSet(name).GetSelection() 
            for xy in list_xy:
                sel.SelectPartByPosition(xy[0],xy[1],0) # z=0 for 2D
                model.GetSetList().GetSet(name).AddSelected(sel)
        # part_list_set(u'Motion_Region', [[0,0],[0,self.Radius_Shaft+EPS]] + list_xy_bars + list_xy_airWithinRotorSlot) 
        part_list_set('Motion_Region', [[0,0],[0,self.Radius_Shaft+EPS]] + list_xy_bars) 

        # Create Set for Cage
        model.GetSetList().CreatePartSet("CageSet")
        model.GetSetList().GetSet("CageSet").SetMatcherType("MatchNames")
        model.GetSetList().GetSet("CageSet").SetParameter("style", "prefix")
        model.GetSetList().GetSet("CageSet").SetParameter("text", "Cage")
        model.GetSetList().GetSet("CageSet").Rebuild()

        return True        

    def add_magnetic_transient_study(self, app, model, dir_csv_output_folder, study_name):
        logger = logging.getLogger(__name__)
        spmsm_variant = self

        model.CreateStudy("Transient2D", study_name)
        app.SetCurrentStudy(study_name)
        study = model.GetStudy(study_name)

        # SS-ATA
        study.GetStudyProperties().SetValue("ApproximateTransientAnalysis", 1) # psuedo steady state freq is for PWM drive to use
        study.GetStudyProperties().SetValue("SpecifySlip", 0)
        study.GetStudyProperties().SetValue("OutputSteadyResultAs1stStep", 0)
        # study.GetStudyProperties().SetValue(u"TimePeriodicType", 2) # This is for TP-EEC but is not effective

        # misc
        study.GetStudyProperties().SetValue("ConversionType", 0)
        study.GetStudyProperties().SetValue("NonlinearMaxIteration", self.max_nonlinear_iteration)
        study.GetStudyProperties().SetValue("ModelThickness", 200) # [mm] Stack Length

        # Material
        self.add_material(study)

        # Conditions - Motion
        study.CreateCondition("RotationMotion", "RotCon") # study.GetCondition(u"RotCon").SetXYZPoint(u"", 0, 0, 1) # megbox warning
        print('the_speed:', self.the_speed)
        study.GetCondition("RotCon").SetValue("AngularVelocity", int(self.the_speed))
        study.GetCondition("RotCon").ClearParts()
        study.GetCondition("RotCon").AddSet(model.GetSetList().GetSet("Motion_Region"), 0)

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

        # # this is for the CAD parameters to rotate the rotor. the order matters for param_no to begin at 0.
        # if self.MODEL_ROTATE:
        #     self.add_cad_parameters(study)


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
            refarray[1][0] = 0.5/slip_freq_breakdown_torque #0.5 for 17.1.03l # 1 for 17.1.02y
            refarray[1][1] =    number_of_steps_2ndTTS                          # 16 for 17.1.03l #32 for 17.1.02y
            refarray[1][2] =        50
            refarray[2][0] = refarray[1][0] + 0.5/spmsm_variant.DriveW_Freq #0.5 for 17.1.03l 
            refarray[2][1] =    number_of_steps_2ndTTS  # also modify range_ss! # don't forget to modify below!
            refarray[2][2] =        50
            DM.GetDataSet("SectionStepTable").SetTable(refarray)
            number_of_total_steps = 1 + 2 * number_of_steps_2ndTTS # [Double Check] don't forget to modify here!
            study.GetStep().SetValue("Step", number_of_total_steps)
            study.GetStep().SetValue("StepType", 3)
            study.GetStep().SetTableProperty("Division", DM.GetDataSet("SectionStepTable"))

        else: # IEMDC19
            number_cycles_prolonged = 1 # 50
            DM = app.GetDataManager()
            DM.CreatePointArray("point_array/timevsdivision", "SectionStepTable")
            refarray = [[0 for i in range(3)] for j in range(4)]
            refarray[0][0] = 0
            refarray[0][1] =    1
            refarray[0][2] =        50
            refarray[1][0] = 1.0/slip_freq_breakdown_torque
            refarray[1][1] =    32 
            refarray[1][2] =        50
            refarray[2][0] = refarray[1][0] + 1.0/spmsm_variant.DriveW_Freq
            refarray[2][1] =    48 # don't forget to modify below!
            refarray[2][2] =        50
            refarray[3][0] = refarray[2][0] + number_cycles_prolonged/spmsm_variant.DriveW_Freq # =50*0.002 sec = 0.1 sec is needed to converge to TranRef
            refarray[3][1] =    number_cycles_prolonged*self.fea_config_dict['TranRef-StepPerCycle'] # =50*40, every 0.002 sec takes 40 steps 
            refarray[3][2] =        50
            DM.GetDataSet("SectionStepTable").SetTable(refarray)
            study.GetStep().SetValue("Step", 1 + 32 + 48 + number_cycles_prolonged*self.fea_config_dict['TranRef-StepPerCycle']) # [Double Check] don't forget to modify here!
            study.GetStep().SetValue("StepType", 3)
            study.GetStep().SetTableProperty("Division", DM.GetDataSet("SectionStepTable"))

        # add equations
        study.GetDesignTable().AddEquation("freq")
        study.GetDesignTable().AddEquation("slip")
        study.GetDesignTable().AddEquation("speed")
        study.GetDesignTable().GetEquation("freq").SetType(0)
        study.GetDesignTable().GetEquation("freq").SetExpression("%g"%((spmsm_variant.DriveW_Freq)))
        study.GetDesignTable().GetEquation("freq").SetDescription("Excitation Frequency")
        study.GetDesignTable().GetEquation("slip").SetType(0)
        study.GetDesignTable().GetEquation("slip").SetExpression("%g"%(spmsm_variant.the_slip))
        study.GetDesignTable().GetEquation("slip").SetDescription("Slip [1]")
        study.GetDesignTable().GetEquation("speed").SetType(1)
        study.GetDesignTable().GetEquation("speed").SetExpression("freq * (1 - slip) * %d"%(60/(spmsm_variant.DriveW_poles/2)))
        study.GetDesignTable().GetEquation("speed").SetDescription("mechanical speed of four pole")

        # speed, freq, slip
        study.GetCondition("RotCon").SetValue("AngularVelocity", 'speed')
        if self.fea_config_dict['DPNV']==False:
            app.ShowCircuitGrid(True)
            study.GetCircuit().GetComponent("CS4").SetValue("Frequency", "freq")
            study.GetCircuit().GetComponent("CS2").SetValue("Frequency", "freq")

        # max_nonlinear_iteration = 50
        # study.GetStudyProperties().SetValue(u"NonlinearMaxIteration", max_nonlinear_iteration)
        study.GetStudyProperties().SetValue("ApproximateTransientAnalysis", 1) # psuedo steady state freq is for PWM drive to use
        study.GetStudyProperties().SetValue("SpecifySlip", 1)
        study.GetStudyProperties().SetValue("OutputSteadyResultAs1stStep", 0)
        study.GetStudyProperties().SetValue("Slip", "slip") # overwrite with variables

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
            sel.SelectPartByPosition(-spmsm_variant.Radius_OuterStatorYoke+EPS, 0 ,0)
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
        study.GetMaterial("Shaft").SetValue("OutputResult", 0)
        study.GetMaterial("Cage").SetValue("OutputResult", 0)
        study.GetMaterial("Coil").SetValue("OutputResult", 0)
        # Rotor
        if True:
            cond = study.CreateCondition("Ironloss", "IronLossConRotor")
            cond.SetValue("BasicFrequencyType", 2)
            cond.SetValue("BasicFrequency", "freq")
                # cond.SetValue(u"BasicFrequency", u"slip*freq") # this require the signal length to be at least 1/4 of slip period, that's too long!
            cond.ClearParts()
            sel = cond.GetSelection()
            sel.SelectPartByPosition(-spmsm_variant.Radius_Shaft-EPS, 0 ,0)
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
        pass

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

    spmsm_template.driveWinding_Freq       = 1000
    spmsm_template.driveWinding_Rs         = 0.1 # TODO
    spmsm_template.driveWinding_zQ         = 1
    spmsm_template.driveWinding_CurrentAmp = None # this means it depends on the slot area
    spmsm_template.driveWinding_poles = 2*spmsm_template.p

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

    spmsm_template.driveWinding_Freq       = 1000
    spmsm_template.driveWinding_Rs         = 0.1 # TODO
    spmsm_template.driveWinding_zQ         = 1
    spmsm_template.driveWinding_CurrentAmp = None # this means it depends on the slot area
    spmsm_template.driveWinding_poles = 2*spmsm_template.p

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


def add_bpmsm_material():

    # -*- coding: utf-8 -*-
    app = designer.GetApplication()
    app.GetMaterialLibrary().CreateCustomMaterial(u"CarbonFiber", u"Custom Materials")
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"Density", 1.6)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"CoerciveForce", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"DemagnetizationCoerciveForce", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"MagnetizationSaturated", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"MagnetizationSaturated2", 0)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"YoungModulus", 70000)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"ShearModulus", 5000)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"PoissonRatio", 0.1)
    app.GetMaterialLibrary().GetUserMaterial(u"CarbonFiber").SetValue(u"Thermal Expansion", 2.1)
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



