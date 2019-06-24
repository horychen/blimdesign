
class bearingless_spmsm_template(object):
    def __init__(self, model_name_prefix='SPMSM'):
        self.model_name_prefix = model_name_prefix

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
            free_variables[3]  = spmsm_template.mm_r_si + spmsm_template.mm_d_sp + spmsm_template.mm_d_st
            free_variables[4]  = spmsm_template.mm_w_st         
            free_variables[5]  = spmsm_template.sleeve_length   
            free_variables[6]  = spmsm_template.mm_d_pm         
            free_variables[7]  = spmsm_template.deg_alpha_rm    
            free_variables[8]  = spmsm_template.deg_alpha_rs    
            free_variables[9]  = spmsm_template.mm_d_ri         
            free_variables[10] = spmsm_template.mm_r_ri + spmsm_template.mm_d_ri + spmsm_template.mm_d_rp
            free_variables[11] = spmsm_template.mm_d_rp         
            free_variables[12] = spmsm_template.mm_d_rs         

        deg_alpha_st    = free_variables[0]
        mm_d_so         = free_variables[1]
        mm_d_st         = free_variables[2]
        stator_outer_radius = free_variables[3]
        mm_w_st         = free_variables[4]
        sleeve_length   = free_variables[5]
        mm_d_pm         = free_variables[6]
        deg_alpha_rm    = free_variables[7]
        deg_alpha_rs    = free_variables[8]
        mm_d_ri         = free_variables[9]
        rotor_outer_radius = free_variables[10] 
        mm_d_rp         = free_variables[11]
        mm_d_rs         = free_variables[12]

        self.deg_alpha_st = free_variables[0]
        self.deg_alpha_so = spmsm_template.deg_alpha_so
        self.mm_r_si      = rotor_outer_radius + sleeve_length + spmsm_template.fixed_air_gap_length
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

    def update_mechanical_parameters(self, slip_freq=None, syn_freq=None):
        # This function is first introduced to derive the new slip for different fundamental frequencies.
        if syn_freq is None:
            syn_freq = self.DriveW_Freq
        else:
            if syn_freq != self.DriveW_Freq:
                raise Exception('I do not recommend to modify synchronous speed at instance level. Go update the initial design.')

        if syn_freq == 0.0: # lock rotor
            self.the_slip = 0. # this does not actually make sense
            if slip_freq == None:
                self.DriveW_Freq = self.slip_freq_breakdown_torque
                self.BeariW_Freq = self.slip_freq_breakdown_torque
            else:
                self.DriveW_Freq = slip_freq
                self.BeariW_Freq = slip_freq
        else:
            if slip_freq != None:
                # change slip
                self.the_slip = slip_freq / syn_freq
                self.slip_freq_breakdown_torque = slip_freq
            else:
                # change syn_freq so update the slip
                self.the_slip = self.slip_freq_breakdown_torque / syn_freq

            self.DriveW_Freq = syn_freq
            self.BeariW_Freq = syn_freq

        self.the_speed = self.DriveW_Freq*60. / (0.5*self.DriveW_poles) * (1 - self.the_slip) # rpm

        self.Omega = + self.the_speed / 60. * 2*pi
        self.omega = None # This variable name is devil! you can't tell its electrical or mechanical! #+ self.DriveW_Freq * (1-self.the_slip) * 2*pi
        # self.the_speed = + self.the_speed

        if self.fea_config_dict is not None:
            if self.fea_config_dict['flag_optimization'] == False: # or else it becomes annoying
                print('[Update ID:%s]'%(self.ID), self.slip_freq_breakdown_torque, self.the_slip, self.the_speed, self.Omega, self.DriveW_Freq, self.BeariW_Freq)

    def draw_spmsm(self, toolJd):

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

        # Stator Core
        list_regions = spmsm.stator_core.draw(toolJd)
        toolJd.bMirror = True
        toolJd.iRotateCopy = spmsm.stator_core.Q
        region3 = toolJd.prepareSection(list_regions)

        # Stator Winding
        list_regions = spmsm.coils.draw(toolJd)
        toolJd.bMirror = False
        toolJd.iRotateCopy = spmsm.coils.stator_core.Q
        region4 = toolJd.prepareSection(list_regions)

        # Import Model into Designer
        toolJd.doc.SaveModel(False) # True: Project is also saved. 
        model = toolJd.app.GetCurrentModel()
        model.SetName(self.name)
        model.SetDescription(self.show(toString=True))

    def show(self, toString=False):
        attrs = list(vars(self).items())
        key_list = [el[0] for el in attrs]
        val_list = [el[1] for el in attrs]
        the_dict = dict(list(zip(key_list, val_list)))
        sorted_key = sorted(key_list, key=lambda item: (int(item.partition(' ')[0]) if item[0].isdigit() else float('inf'), item)) # this is also useful for string beginning with digiterations '15 Steel'.
        tuple_list = [(key, the_dict[key]) for key in sorted_key]
        if toString==False:
            print('- Bearingless PMSM Individual #%s\n\t' % (self.ID), end=' ')
            print(', \n\t'.join("%s = %s" % item for item in tuple_list))
            return ''
        else:
            return '\n- Bearingless PMSM Individual #%s\n\t' % (self.ID) + ', \n\t'.join("%s = %s" % item for item in tuple_list)

# circumferential segmented rotor 
if __name__ == '!__main__':
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
    # logger.info('spmsm_variant ID %s is initialized.', self.ID)

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
    # logger.info('spmsm_variant ID %s is initialized.', self.ID)

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



