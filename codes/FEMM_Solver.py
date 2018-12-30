#coding:utf-8
from __future__ import division
import femm
from math import tan, pi, atan, cos, sin, sqrt, copysign
import numpy as np
from csv import reader as csv_reader

import logging
import os
from collections import OrderedDict

import sys
import subprocess

SELECT_ALL = 4
EPS = 1e-2

class FEMM_Solver(object):
    def __init__(self, im, flag_read_from_jmag=True, freq=0):

        self.deg_per_step = im.fea_config_dict['femm_deg_per_step'] # deg, we need this for show_results
        self.flag_read_from_jmag = flag_read_from_jmag

        self.freq = freq
        if freq == 0:
            self.flag_eddycurrent_solver = False
            self.flag_static_solver = not self.flag_eddycurrent_solver
            self.fraction = 1
        else:
            self.flag_eddycurrent_solver = True
            self.flag_static_solver = not self.flag_eddycurrent_solver
            self.fraction = 2

        self.stack_length = im.stack_length

        self.im = im
        self.dir_codes = im.fea_config_dict['dir_codes']

        if im.bool_initial_design == True:
            self.dir_run = im.fea_config_dict['dir_femm_files'] + im.fea_config_dict['model_name_prefix'] + '/'

            if not os.path.exists(self.dir_run):
                logger = logging.getLogger(__name__)
                logger.debug('FEMM: There is no run yet. Generate the run folder as %s.', self.dir_run)
                os.makedirs(self.dir_run)

            if not os.path.exists(self.dir_run + 'static-jmag/'):
                os.makedirs(self.dir_run + 'static-jmag/')
            if not os.path.exists(self.dir_run + 'static-femm/'):
                os.makedirs(self.dir_run + 'static-femm/')

            if flag_read_from_jmag == True:
                self.dir_run = self.dir_run + 'static-jmag/'
            else:
                self.dir_run = self.dir_run + 'static-femm/'
        else:
            self.dir_run = im.fea_config_dict['dir_femm_files'] + im.fea_config_dict['run_folder']

            if not os.path.exists(self.dir_run):
                logger = logging.getLogger(__name__)
                logger.debug('FEMM: There is no run yet. Generate the run folder as %s.', self.dir_run)
                os.makedirs(self.dir_run)

        self.output_file_name = self.get_output_file_name()
        self.rotor_slot_per_pole = int(im.Qr/im.DriveW_poles)
        self.rotor_phase_name_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def add_block_labels(self, fraction=1):
        im = self.im

        SERIES_CONNECTED = 1
        PARALLEL_CONNECTED = 0

        def block_label(group_no, material_name, p, meshsize_if_no_automesh, incircuit='<None>', turns=0, automesh=True, magdir=0):
            femm.mi_addblocklabel(p[0],p[1])
            femm.mi_selectlabel(p[0],p[1])
            femm.mi_setblockprop(material_name, automesh, meshsize_if_no_automesh, incircuit, magdir, group_no, turns)
            femm.mi_clearselected()

        # air region
        X = Y = -(im.Radius_OuterRotor+0.5*im.Length_AirGap) / 1.4142135623730951
        block_label(9, 'Air', (X, Y), 0.5, automesh=self.bool_automesh)

        # # Air Gap Boundary for Rotor Motion #2
        # block_label(9, '<No Mesh>',   (0, im.Radius_OuterRotor+0.5*im.Length_AirGap), 5, automesh=self.bool_automesh)
        # block_label(9, 'Air',         (0, im.Radius_OuterRotor+0.7*im.Length_AirGap), 0.5, automesh=self.bool_automesh)
        # block_label(9, 'Air',         (0, im.Radius_OuterRotor+0.3*im.Length_AirGap), 0.5, automesh=self.bool_automesh)


        # shaft
        if fraction == 1:
            block_label(100, '<No Mesh>',         (0, 0),  20)
            # block_label(100, 'Air',         (0, 0),  10, automesh=True) # for deeply-saturated rotor yoke

        # Iron Core
        X = Y = -(im.Radius_Shaft+EPS*10) / 1.4142135623730951
        block_label(100, 'My M-15 Steel',  (X, Y), 4, automesh=self.bool_automesh)
        X = Y = -(0.5*(im.Radius_InnerStatorYoke+im.Radius_OuterStatorYoke)) / 1.4142135623730951
        block_label(10, 'My M-15 Steel',  (X, Y), 4, automesh=self.bool_automesh)

        # Circuit Configuration
        if fraction == 1:
            # Pole-Specific Rotor Winding
            # Rotor Winding
            R = 0.5*(im.Location_RotorBarCenter + im.Location_RotorBarCenter2)
            angle_per_slot = 2*pi/im.Qr
            THETA_BAR = pi - angle_per_slot

            for i in range(self.rotor_slot_per_pole):
                circuit_name = 'r%s'%(self.rotor_phase_name_list[i])

                if self.flag_static_solver == True: #self.freq == 0: # Static FEA
                    femm.mi_addcircprop(circuit_name, self.dict_rotor_current_function[i](self.time), SERIES_CONNECTED)
                    # print self.dict_rotor_current_function[i](self.time)
                else:  # Eddy Current FEA (with multi-phase 4-bar cage... haha this is practically nothing)
                    femm.mi_addcircprop(circuit_name, 0, PARALLEL_CONNECTED) # PARALLEL for PS circuit

                THETA_BAR += angle_per_slot

                THETA = THETA_BAR
                X = R*cos(THETA); Y = R*sin(THETA)
                block_label(100, 'Aluminum', (X, Y), 3, automesh=self.bool_automesh, incircuit=circuit_name, turns=1)

                THETA = THETA_BAR + angle_per_slot*self.rotor_slot_per_pole
                X = R*cos(THETA); Y = R*sin(THETA)
                block_label(100, 'Aluminum', (X, Y), 3, automesh=self.bool_automesh, incircuit=circuit_name, turns=-1)

                THETA = THETA_BAR + angle_per_slot*2*self.rotor_slot_per_pole
                X = R*cos(THETA); Y = R*sin(THETA)
                block_label(100, 'Aluminum', (X, Y), 3, automesh=self.bool_automesh, incircuit=circuit_name, turns=1)

                THETA = THETA_BAR + angle_per_slot*3*self.rotor_slot_per_pole
                X = R*cos(THETA); Y = R*sin(THETA)
                block_label(100, 'Aluminum', (X, Y), 3, automesh=self.bool_automesh, incircuit=circuit_name, turns=-1)
        elif fraction == 4 or fraction == 2:
            # Cage 
            R = 0.5*(im.Location_RotorBarCenter + im.Location_RotorBarCenter2)
            angle_per_slot = 2*pi/im.Qr
            THETA_BAR = pi - angle_per_slot + EPS # add EPS for the half bar

            for i in range(self.rotor_slot_per_pole):
                circuit_name = 'r%s'%(self.rotor_phase_name_list[i])
                # Eddy Current FEA (with multi-phase 4-bar cage behave the same with PS rotor winding when no bearing current is excited!
                femm.mi_addcircprop(circuit_name, 0, PARALLEL_CONNECTED) # PARALLEL for PS circuit (valid only if there is no 2-pole field)

                THETA_BAR += angle_per_slot

                THETA = THETA_BAR
                X = R*cos(THETA); Y = R*sin(THETA)
                block_label(100, 'Aluminum', (X, Y), 3, automesh=self.bool_automesh, incircuit=circuit_name, turns=1) # rA+ ~ rH+

                if fraction == 2:
                    THETA = THETA_BAR + angle_per_slot*self.rotor_slot_per_pole
                    X = R*cos(THETA); Y = R*sin(THETA)
                    block_label(100, 'Aluminum', (X, Y), 3, automesh=self.bool_automesh, incircuit=circuit_name, turns=-1) # rA- However, this turns=-1 is not effective for PARALLEL_CONNECTED circuit

            # the other half bar 
            # THETA_BAR += angle_per_slot
            THETA = THETA + angle_per_slot - 2*EPS
            X = R*cos(THETA); Y = R*sin(THETA)
            block_label(100, 'Aluminum', (X, Y), 3, automesh=self.bool_automesh, incircuit='r%s'%(self.rotor_phase_name_list[0]), turns=-1) # However, this turns=-1 is not effective for PARALLEL_CONNECTED circuit

        # Stator Winding
        if self.flag_static_solver == True: #self.freq == 0: # static 
            femm.mi_addcircprop('dA', self.dict_stator_current_function[3](self.time), SERIES_CONNECTED)
            femm.mi_addcircprop('dB', self.dict_stator_current_function[4](self.time), SERIES_CONNECTED)
            femm.mi_addcircprop('dC', self.dict_stator_current_function[5](self.time), SERIES_CONNECTED)
            femm.mi_addcircprop('bA', self.dict_stator_current_function[0](self.time), SERIES_CONNECTED)
            femm.mi_addcircprop('bB', self.dict_stator_current_function[1](self.time), SERIES_CONNECTED)
            femm.mi_addcircprop('bC', self.dict_stator_current_function[2](self.time), SERIES_CONNECTED)
        else: # eddy current
            femm.mi_addcircprop('dA', '%g'                            %(im.DriveW_CurrentAmp), SERIES_CONNECTED)
            femm.mi_addcircprop('dB', '%g*(-0.5+I*0.8660254037844386)'%(im.DriveW_CurrentAmp), SERIES_CONNECTED)
            femm.mi_addcircprop('dC', '%g*(-0.5-I*0.8660254037844386)'%(im.DriveW_CurrentAmp), SERIES_CONNECTED)
            if fraction == 1: # i thought PS can be realized in FEMM
                femm.mi_addcircprop('bA', '%g'                            %(im.BeariW_CurrentAmp), SERIES_CONNECTED)
                femm.mi_addcircprop('bB', '%g*(-0.5+I*0.8660254037844386)'%(im.BeariW_CurrentAmp), SERIES_CONNECTED)
                femm.mi_addcircprop('bC', '%g*(-0.5-I*0.8660254037844386)'%(im.BeariW_CurrentAmp), SERIES_CONNECTED)
            elif fraction == 4 or fraction == 2:
                femm.mi_addcircprop('bA', 0, SERIES_CONNECTED)
                femm.mi_addcircprop('bB', 0, SERIES_CONNECTED)
                femm.mi_addcircprop('bC', 0, SERIES_CONNECTED)

        # dict_dir = {'+':1, '-':-1} # wrong 
        dict_dir = {'+':-1, '-':1}
        R = 0.5*(im.Radius_OuterRotor + im.Radius_InnerStatorYoke)
        angle_per_slot = 2*pi/im.Qs

        # torque winding's blocks
        THETA = - angle_per_slot + 0.5*angle_per_slot - 3.0/360 # This 3 deg must be less than 360/Qs/2
        count = 0
        for phase, up_or_down in zip(im.l41,im.l42):
            circuit_name = 'd' + phase
            THETA += angle_per_slot
            X = R*cos(THETA); Y = R*sin(THETA)
            count += 1
            if fraction == 4:
                if not (count > im.Qs*0.5+EPS and count <= im.Qs*0.75+EPS): 
                    continue
            if fraction == 2:
                if not (count > im.Qs*0.5+EPS): 
                    continue
            block_label(10, 'Copper', (X, Y), 8, automesh=self.bool_automesh, incircuit=circuit_name, turns=im.DriveW_turns*dict_dir[up_or_down])

        # bearing winding's blocks
        if fraction == 1:
            THETA = - angle_per_slot + 0.5*angle_per_slot + 3.0/360
            for phase, up_or_down in zip(im.l21,im.l22):
                circuit_name = 'b' + phase
                THETA += angle_per_slot
                X = R*cos(THETA); Y = R*sin(THETA)
                block_label(10, 'Copper', (X, Y), 8, automesh=self.bool_automesh, incircuit=circuit_name, turns=im.BeariW_turns*dict_dir[up_or_down])
        elif fraction == 4 or fraction == 2:
            # 危险！FEMM默认把没有设置incircuit的导体都在无限远短接在一起——也就是说，你可能把定子悬浮绕组也短接到鼠笼上去了！
            # 所以，一定要设置好悬浮绕组，而且要用serial-connected，电流给定为 0 A。
            THETA = - angle_per_slot + 0.5*angle_per_slot + 3.0/360
            count = 0
            for phase, up_or_down in zip(im.l21,im.l22):
                circuit_name = 'b' + phase
                THETA += angle_per_slot
                X = R*cos(THETA); Y = R*sin(THETA)
                count += 1
                if fraction == 4:
                    if not (count > im.Qs*0.5+EPS and count <= im.Qs*0.75+EPS): 
                        continue
                elif fraction == 2:
                    if not (count > im.Qs*0.5+EPS): 
                        continue
                block_label(10, 'Copper', (X, Y), 8, automesh=self.bool_automesh, incircuit=circuit_name, turns=im.BeariW_turns*dict_dir[up_or_down])

        # Boundary Conditions 
        # femm.mi_makeABC() # open boundary
        if fraction == 1:
            femm.mi_addboundprop('BC:A=0', 0,0,0, 0,0,0,0,0,0,0,0)

            femm.mi_selectarcsegment(0,-im.Radius_OuterStatorYoke)
            femm.mi_setarcsegmentprop(20, "BC:A=0", False, 10) # maxseg = 20 deg (only this is found effective)
            femm.mi_clearselected()
            femm.mi_selectarcsegment(0,im.Radius_OuterStatorYoke)
            femm.mi_setarcsegmentprop(20, "BC:A=0", False, 10)
            femm.mi_clearselected()

            femm.mi_selectarcsegment(0,-im.Radius_Shaft)
            femm.mi_setarcsegmentprop(20, "BC:A=0", False, 100)
            femm.mi_clearselected()
            femm.mi_selectarcsegment(0,im.Radius_Shaft)
            femm.mi_setarcsegmentprop(20, "BC:A=0", False, 100)
            femm.mi_clearselected()
        elif fraction == 4:
            femm.mi_addboundprop('BC:A=0', 0,0,0, 0,0,0,0,0,0,0,0)

            X = Y = -(im.Radius_OuterStatorYoke) / 1.4142135623730951
            femm.mi_selectarcsegment(X, Y)
            femm.mi_setarcsegmentprop(10, "BC:A=0", False, 10) # maxseg = 20 deg (only this is found effective)
            femm.mi_clearselected()

            X = Y = -(im.Radius_Shaft) / 1.4142135623730951
            femm.mi_selectarcsegment(0,-im.Radius_Shaft)
            femm.mi_setarcsegmentprop(10, "BC:A=0", False, 100)
            femm.mi_clearselected()

            femm.mi_addboundprop('apbc1', 0,0,0, 0,0,0,0,0, 5, 0,0)
            femm.mi_addboundprop('apbc2', 0,0,0, 0,0,0,0,0, 5, 0,0)
            femm.mi_addboundprop('apbc3', 0,0,0, 0,0,0,0,0, 5, 0,0)
            femm.mi_addboundprop('apbc5', 0,0,0, 0,0,0,0,0, 5, 0,0)
            femm.mi_addboundprop('apbc6', 0,0,0, 0,0,0,0,0, 5, 0,0) # http://www.femm.info/wiki/periodicboundaries

            R = im.Radius_Shaft+EPS
            femm.mi_selectsegment(0,-R)
            femm.mi_selectsegment(-R,0)
            femm.mi_setsegmentprop("apbc1", 4, False, False, 100)
            femm.mi_clearselected()

            R = im.Location_RotorBarCenter
            femm.mi_selectsegment(0,-R)
            femm.mi_selectsegment(-R,0)
            femm.mi_setsegmentprop("apbc2", 3, False, False, 100)
            femm.mi_clearselected()

            R = im.Radius_OuterRotor + 0.25*im.Length_AirGap
            femm.mi_selectsegment(0,-R)
            femm.mi_selectsegment(-R,0)
            femm.mi_setsegmentprop("apbc3", 0.5, False, False, 9)
            femm.mi_clearselected()

            R = im.Radius_OuterRotor + 0.75*im.Length_AirGap
            femm.mi_selectsegment(0,-R)
            femm.mi_selectsegment(-R,0)
            femm.mi_setsegmentprop("apbc5", 0.5, False, False, 9)
            femm.mi_clearselected()

            R = im.Radius_OuterStatorYoke-EPS
            femm.mi_selectsegment(0,-R)
            femm.mi_selectsegment(-R,0)
            femm.mi_setsegmentprop("apbc6", 4, False, False, 10)
            femm.mi_clearselected()
        elif fraction == 2:
            femm.mi_addboundprop('BC:A=0', 0,0,0, 0,0,0,0,0,0,0,0)

            X = Y = -(im.Radius_OuterStatorYoke) / 1.4142135623730951
            femm.mi_selectarcsegment(X, Y)
            femm.mi_setarcsegmentprop(10, "BC:A=0", False, 10) # maxseg = 20 deg (only this is found effective)
            femm.mi_clearselected()

            X = Y = -(im.Radius_Shaft) / 1.4142135623730951
            femm.mi_selectarcsegment(0,-im.Radius_Shaft)
            femm.mi_setarcsegmentprop(10, "BC:A=0", False, 100)
            femm.mi_clearselected()

            femm.mi_addboundprop('pbc1', 0,0,0, 0,0,0,0,0, 4, 0,0)
            femm.mi_addboundprop('pbc2', 0,0,0, 0,0,0,0,0, 4, 0,0)
            femm.mi_addboundprop('pbc3', 0,0,0, 0,0,0,0,0, 4, 0,0)
            femm.mi_addboundprop('pbc5', 0,0,0, 0,0,0,0,0, 4, 0,0)
            femm.mi_addboundprop('pbc6', 0,0,0, 0,0,0,0,0, 4, 0,0)

            R = im.Radius_Shaft+EPS
            femm.mi_selectsegment(-R,0)
            femm.mi_selectsegment(+R,0)
            femm.mi_setsegmentprop("pbc1", 4, False, False, 100)
            femm.mi_clearselected()

            R = im.Location_RotorBarCenter
            femm.mi_selectsegment(+R,0)
            femm.mi_selectsegment(-R,0)
            femm.mi_setsegmentprop("pbc2", 3, False, False, 100)
            femm.mi_clearselected()

            R = im.Radius_OuterRotor + 0.25*im.Length_AirGap
            femm.mi_selectsegment(+R,0)
            femm.mi_selectsegment(-R,0)
            femm.mi_setsegmentprop("pbc3", 0.5, False, False, 9)
            femm.mi_clearselected()

            R = im.Radius_OuterRotor + 0.75*im.Length_AirGap
            femm.mi_selectsegment(+R,0)
            femm.mi_selectsegment(-R,0)
            femm.mi_setsegmentprop("pbc5", 0.5, False, False, 9)
            femm.mi_clearselected()

            R = im.Radius_OuterStatorYoke-EPS
            femm.mi_selectsegment(+R,0)
            femm.mi_selectsegment(-R,0)
            femm.mi_setsegmentprop("pbc6", 4, False, False, 10)
            femm.mi_clearselected()

        # Air Gap Boundary for Rotor Motion #3
        # inner_angle = 0; outer_angle = 0
        # femm.mi_addboundprop('AGB4RM', 0,0,0, 0,0,0,0,0, 6, inner_angle, outer_angle)
        # R = im.Radius_OuterRotor+0.6*im.Length_AirGap
        # femm.mi_selectarcsegment(0,-R)
        # femm.mi_setarcsegmentprop(5, "AGB4RM", False, 9)
        # femm.mi_clearselected()
        # femm.mi_selectarcsegment(0,R)
        # femm.mi_setarcsegmentprop(5, "AGB4RM", False, 9)
        # femm.mi_clearselected()
        # R = im.Radius_OuterRotor+0.4*im.Length_AirGap
        # femm.mi_selectarcsegment(0,-R)
        # femm.mi_setarcsegmentprop(5, "AGB4RM", False, 9)
        # femm.mi_clearselected()
        # femm.mi_selectarcsegment(0,R)
        # femm.mi_setarcsegmentprop(5, "AGB4RM", False, 9)
        # femm.mi_clearselected()

        # Other arc-segment-specific mesh constraints are already done in draw_model()

    def draw_model(self, fraction=1):
        im = self.im

        from shapely.geometry import LineString
        from shapely.geometry import Point
        origin = Point(0,0)
        Stator_Sector_Angle = 2*pi/im.Qs*0.5
        Rotor_Sector_Angle = 2*pi/im.Qr*0.5

        def mirror_and_copyrotate(Q, Radius):
            # Mirror
            femm.mi_selectcircle(0,0,Radius+EPS,SELECT_ALL) # this EPS is sometime necessary to selece the arc at Radius.
            femm.mi_mirror2(0,0,-Radius,0, SELECT_ALL)

            # Rotate
            femm.mi_selectcircle(0,0,Radius+EPS,SELECT_ALL)
            femm.mi_copyrotate2(0, 0, 360./Q, int(Q)/fraction, SELECT_ALL)

        def create_circle(p, radius):
            return p.buffer(radius).boundary

        def get_node_at_intersection(c,l): # this works for c and l having one intersection only
            i = c.intersection(l)
            # femm.mi_addnode(i.coords[0][0], i.coords[0][1])
            return i.coords[0][0], i.coords[0][1]

        def draw_arc(p1, p2, angle, maxseg=1):
            femm.mi_drawarc(p1[0],p1[1],p2[0],p2[1],angle/pi*180,maxseg) # [deg]
        def add_arc(p1, p2, angle, maxseg=1):
            femm.mi_addarc(p1[0],p1[1],p2[0],p2[1],angle/pi*180,maxseg) # [deg]
        def draw_line(p1, p2):
            femm.mi_drawline(p1[0],p1[1],p2[0],p2[1])
        def add_line(p1, p2):
            femm.mi_addsegment(p1[0],p1[1],p2[0],p2[1])
        def get_postive_angle(p, origin=(0,0)):
            # using atan loses info about the quadrant
            return atan(abs((p[1]-origin[1]) / (p[0]-origin[0])))


        ''' Part: Stator '''
        # Draw Points as direction of CCW
        # P1
        P1 = (-im.Radius_OuterRotor-im.Length_AirGap, 0)

        # P2
        # Parallel to Line? No they are actually not parallel
        P2_angle = Stator_Sector_Angle -im.Angle_StatorSlotOpen*0.5/180*pi
        k = -tan(P2_angle)
        l_sector_parallel = LineString([(0,0), (-im.Radius_OuterStatorYoke, -im.Radius_OuterStatorYoke*k)])
        c = create_circle(origin, im.Radius_OuterRotor+im.Length_AirGap)
        P2 = get_node_at_intersection(c,l_sector_parallel)
        draw_arc(P2, P1, get_postive_angle(P2))

        # P3
        c = create_circle(origin, im.Radius_OuterRotor+im.Length_AirGap+im.Width_StatorTeethHeadThickness)
        P3 = get_node_at_intersection(c,l_sector_parallel)
        draw_line(P2, P3)

        # P4
        c = create_circle(origin, im.Radius_OuterRotor+im.Length_AirGap+im.Width_StatorTeethHeadThickness+im.Width_StatorTeethNeck)
        l = LineString([(0, 0.5*im.Width_StatorTeethBody), (-im.Radius_OuterStatorYoke, 0.5*im.Width_StatorTeethBody)])
        P4 = get_node_at_intersection(c,l)
        draw_line(P3, P4)

        # P5
        c = create_circle(origin, im.Radius_InnerStatorYoke)
        P5 = get_node_at_intersection(c,l)
        draw_line(P4, P5)

        # P6
        k = -tan(Stator_Sector_Angle)
        l_sector = LineString([(0,0), (-im.Radius_OuterStatorYoke, -im.Radius_OuterStatorYoke*k)])
        P6 = get_node_at_intersection(c,l_sector)
        draw_arc(P6, P5, Stator_Sector_Angle - get_postive_angle(P5))

        # P7
        c = create_circle(origin, im.Radius_OuterStatorYoke)
        P7 = get_node_at_intersection(c,l_sector)
        # draw_line(P6, P7)

        # P8
        P8 = (-im.Radius_OuterStatorYoke, 0)
        # draw_arc(P7, P8, Stator_Sector_Angle)
        # draw_line(P8, P1)

        # P_Coil
        l = LineString([(P3[0], P3[1]), (P3[0], im.Radius_OuterStatorYoke)])
        P_Coil = get_node_at_intersection(l_sector, l)
        draw_line(P4, P_Coil)
        draw_line(P6, P_Coil)

        mirror_and_copyrotate(im.Qs, im.Radius_OuterStatorYoke)



        ''' Part: Rotor '''
        # Draw Points as direction of CCW
        # P1
        # femm.mi_addnode(-im.Radius_Shaft, 0)
        P1 = (-im.Radius_Shaft, 0)

        # P2
        c = create_circle(origin, im.Radius_Shaft)
        # Line: y = k*x, with k = -tan(2*pi/im.Qr*0.5)
        P2_angle = P3_anlge = Rotor_Sector_Angle
        k = -tan(P2_angle)
        l_sector = LineString([(0,0), (-im.Radius_OuterStatorYoke, -im.Radius_OuterStatorYoke*k)])
        P2 = get_node_at_intersection(c,l_sector)
        # draw_arc(P2, P1, P2_angle)

        # P3
        c = create_circle(origin, im.Radius_OuterRotor)
        P3 = get_node_at_intersection(c,l_sector)
        # draw_line(P2, P3)


        # P4
        l = LineString([(-im.Location_RotorBarCenter, 0.5*im.Width_RotorSlotOpen), (-im.Radius_OuterRotor, 0.5*im.Width_RotorSlotOpen)])
        P4 = get_node_at_intersection(c,l)
        draw_arc(P3, P4, P3_anlge - get_postive_angle(P4))

        # P5
        p = Point(-im.Location_RotorBarCenter, 0)
        c = create_circle(p, im.Radius_of_RotorSlot)
        P5 = get_node_at_intersection(c,l)
        draw_line(P4, P5)

        # P6
        # femm.mi_addnode(-im.Location_RotorBarCenter, im.Radius_of_RotorSlot)
        P6 = (-im.Location_RotorBarCenter, im.Radius_of_RotorSlot)
        draw_arc(P6, P5, 0.5*pi - get_postive_angle(P5, c.centroid.coords[0]))
        # constraint to reduce element number
        femm.mi_selectarcsegment(P6[0], P6[1])
        femm.mi_setarcsegmentprop(8, "<None>", False, 100)
        femm.mi_clearselected()


        # P7
        # femm.mi_addnode(-im.Location_RotorBarCenter2, im.Radius_of_RotorSlot2)
        P7 = (-im.Location_RotorBarCenter2, im.Radius_of_RotorSlot2)
        draw_line(P6, P7)

        # P8
        # femm.mi_addnode(-im.Location_RotorBarCenter2+im.Radius_of_RotorSlot2, 0)
        P8 = (-im.Location_RotorBarCenter2+im.Radius_of_RotorSlot2, 0)
        draw_arc(P8, P7, 0.5*pi)
        # draw_line(P8, P1)
        # constraint to reduce element number
        femm.mi_selectarcsegment(P8[0], P8[1])
        femm.mi_setarcsegmentprop(8, "<None>", False, 100)
        femm.mi_clearselected()

        # P_Bar
        P_Bar = (-im.Location_RotorBarCenter-im.Radius_of_RotorSlot, 0)
        draw_arc(P5, P_Bar, get_postive_angle(P5))
        # add_line(P_Bar, P8)

        mirror_and_copyrotate(im.Qr, im.Radius_OuterRotor)

        # Boundary
        if fraction == 1:
            femm.mi_drawarc(im.Radius_Shaft,0, -im.Radius_Shaft,0, 180, 20) # 边界不要用太小的segment咯！避免剖分过细（这里设置无效）
            femm.mi_drawarc(-im.Radius_Shaft,0, im.Radius_Shaft,0, 180, 20)
            femm.mi_drawarc(im.Radius_OuterStatorYoke,0, -im.Radius_OuterStatorYoke,0, 180, 20)
            femm.mi_drawarc(-im.Radius_OuterStatorYoke,0, im.Radius_OuterStatorYoke,0, 180, 20)
        elif fraction == 4:
            femm.mi_drawarc(-im.Radius_Shaft,0, 0, -im.Radius_Shaft, 90, 10)
            femm.mi_drawarc(-im.Radius_OuterStatorYoke,0, 0, -im.Radius_OuterStatorYoke, 90, 10)
            femm.mi_selectrectangle(-EPS-im.Radius_Shaft,EPS,EPS-im.Radius_OuterStatorYoke,im.Radius_OuterStatorYoke,SELECT_ALL)
            femm.mi_selectrectangle(EPS,-EPS-im.Radius_Shaft,im.Radius_OuterStatorYoke,EPS-im.Radius_OuterStatorYoke,SELECT_ALL)
            femm.mi_deleteselected()

            # between 2rd and 3th quarters
            p1 = (-im.Location_RotorBarCenter2+im.Radius_of_RotorSlot2, 0)
            p2 = (-im.Radius_Shaft, 0)
            add_line(p1, p2)
            p2 = (-im.Location_RotorBarCenter-im.Radius_of_RotorSlot, 0)
            add_line(p1, p2)
            p1 = (-im.Radius_OuterRotor-0.5*im.Length_AirGap, 0) # for later extending for moverotate with anti-periodic boundary condition
            draw_line(p1, p2)
            p2 = (-im.Radius_OuterRotor-im.Length_AirGap, 0)
            draw_line(p1, p2)
            p1 = (-im.Radius_OuterStatorYoke, 0)
            add_line(p1, p2)

            # between 3rd and 4th quarters
            p1 = (0, -im.Location_RotorBarCenter2+im.Radius_of_RotorSlot2)
            p2 = (0, -im.Radius_Shaft)
            add_line(p1, p2)
            p2 = (0, -im.Location_RotorBarCenter-im.Radius_of_RotorSlot)
            add_line(p1, p2)
            p1 = (0, -im.Radius_OuterRotor-0.5*im.Length_AirGap)
            draw_line(p1, p2)
            p2 = (0, -im.Radius_OuterRotor-im.Length_AirGap)
            draw_line(p1, p2)
            p1 = (0, -im.Radius_OuterStatorYoke)
            add_line(p1, p2)
        elif fraction == 2:
            femm.mi_drawarc(-im.Radius_Shaft,0, im.Radius_Shaft,0, 180, 15)
            femm.mi_drawarc(-im.Radius_OuterStatorYoke,0, im.Radius_OuterStatorYoke,0, 180, 15)
            femm.mi_selectrectangle(EPS-im.Radius_OuterStatorYoke,EPS, -EPS+im.Radius_OuterStatorYoke,EPS+im.Radius_OuterStatorYoke, SELECT_ALL)
            femm.mi_deleteselected()

            # between 2rd and 3th quarters
            p1 = (-im.Location_RotorBarCenter2+im.Radius_of_RotorSlot2, 0)
            p2 = (-im.Radius_Shaft, 0)
            add_line(p1, p2)
            p2 = (-im.Location_RotorBarCenter-im.Radius_of_RotorSlot, 0)
            add_line(p1, p2)
            p1 = (-im.Radius_OuterRotor-0.5*im.Length_AirGap, 0) # for later extending for moverotate with anti-periodic boundary condition
            draw_line(p1, p2)
            p2 = (-im.Radius_OuterRotor-im.Length_AirGap, 0)
            draw_line(p1, p2)
            p1 = (-im.Radius_OuterStatorYoke, 0)
            add_line(p1, p2)

            # between 1rd and 4th quarters
            p1 = (+im.Location_RotorBarCenter2-im.Radius_of_RotorSlot2, 0)
            p2 = (+im.Radius_Shaft, 0)
            add_line(p1, p2)
            p2 = (+im.Location_RotorBarCenter+im.Radius_of_RotorSlot, 0)
            add_line(p1, p2)
            p1 = (+im.Radius_OuterRotor+0.5*im.Length_AirGap, 0) # for later extending for moverotate with anti-periodic boundary condition
            draw_line(p1, p2)
            p2 = (+im.Radius_OuterRotor+im.Length_AirGap, 0)
            draw_line(p1, p2)
            p1 = (+im.Radius_OuterStatorYoke, 0)
            add_line(p1, p2)
        else:
            raise Exception('not supported fraction = %d' % (fraction))
        # Air Gap Boundary for Rotor Motion #1
        # R = im.Radius_OuterRotor+0.6*im.Length_AirGap
        # femm.mi_drawarc(R,0, -R,0, 180, 5)
        # femm.mi_drawarc(-R,0, R,0, 180, 5)
        # R = im.Radius_OuterRotor+0.4*im.Length_AirGap
        # femm.mi_drawarc(R,0, -R,0, 180, 5)
        # femm.mi_drawarc(-R,0, R,0, 180, 5)

    def model_rotor_rotate(self):
        if self.deg_per_step != 0.0:

            # if False:
                # Air Gap Boundary for Rotor Motion #4
                # femm.mi_modifyboundprop("AGB4RM", 10, self.rotor_position_in_deg)
                # femm.mi_modifyboundprop("AGB4RM", 11, 0)
            # else:

            # 打开上一个FEM文件，然后将其模型旋转deg_per_step，用不到rotor_position_in_deg的！
            # 这是笨办法，现在的FEMM可以用Rotating Air Gap了，将两者比较验证有效性！
            # print 'zero freq. rotate the model by %g deg.' % (self.rotor_position_in_deg)
            femm.mi_selectgroup(100) # this only select the block labels
            femm.mi_selectcircle(0,0,self.im.Radius_OuterRotor+EPS,SELECT_ALL) # this selects the nodes, segments, arcs
            femm.mi_moverotate(0,0, self.deg_per_step)
            femm.mi_zoomnatural()
                # this error occurs for Qr=32 with EPS=0
                # Traceback (most recent call last):
                #   File "<string>", line 1, in <module>
                #   File "D:/Users/horyc/OneDrive - UW-Madison/femm/pyfemm_script.py", line 121, in <module>
                #     solver.run_rotating_static_FEA(deg_per_step)
                #   File "D:/Users/horyc/OneDrive - UW-Madison/codes/FEMM_Solver.py", line 403, in run_rotating_static_FEA
                #     self.model_rotor_rotate()
                #   File "D:/Users/horyc/OneDrive - UW-Madison/codes/FEMM_Solver.py", line 323, in model_rotor_rotate
                #     femm.mi_moverotate(0,0, self.deg_per_step)
                #   File "D:\Program Files\JMAG-Designer17.1\python2.7\lib\site-packages\femm\__init__.py", line 1590, in mi_moverotate
                #     callfemm('mi_moverotate(' + numc(bx) + numc(by) + num(shiftangle) + ')' );
                #   File "D:\Program Files\JMAG-Designer17.1\python2.7\lib\site-packages\femm\__init__.py", line 25, in callfemm
                #     x = HandleToFEMM.mlab2femm(myString).replace("[ ","[").replace(" ]","]").replace(" ",",").replace("I","1j");
                #   File "<COMObject femm.ActiveFEMM>", line 3, in mlab2femm
                # pywintypes.com_error: (-2147417851, 'The server threw an exception.', None, None)

        for i in range(self.rotor_slot_per_pole):
            circuit_name = 'r%s'%(self.rotor_phase_name_list[i])
            femm.mi_modifycircprop(circuit_name, 1, self.dict_rotor_current_function[i](self.time))

        theta = self.rotor_position_in_deg/180.*pi
        femm.mi_modifycircprop('dA', 1, self.dict_stator_current_function[3](self.time))
        femm.mi_modifycircprop('dB', 1, self.dict_stator_current_function[4](self.time))
        femm.mi_modifycircprop('dC', 1, self.dict_stator_current_function[5](self.time))
        femm.mi_modifycircprop('bA', 1, self.dict_stator_current_function[0](self.time))
        femm.mi_modifycircprop('bB', 1, self.dict_stator_current_function[1](self.time))
        femm.mi_modifycircprop('bC', 1, self.dict_stator_current_function[2](self.time))

        # ia, ib, ic = self.transformed_dict_stator_current_function(3, self.time, theta)
        # print'\t\t', ia, ib, ic,
        # femm.mi_modifycircprop('dA', 1, ia) # self.dict_stator_current_function[3](self.time)
        # femm.mi_modifycircprop('dB', 1, ib) # self.dict_stator_current_function[4](self.time)
        # femm.mi_modifycircprop('dC', 1, ic) # self.dict_stator_current_function[5](self.time)

        # print '\t', ia, ib, ic
        # ia, ib, ic = self.transformed_dict_stator_current_function(0, self.time, theta)
        # femm.mi_modifycircprop('bA', 1, ia) # self.dict_stator_current_function[0](self.time))
        # femm.mi_modifycircprop('bB', 1, ib) # self.dict_stator_current_function[1](self.time))
        # femm.mi_modifycircprop('bC', 1, ic) # self.dict_stator_current_function[2](self.time))

    def run_rotating_static_FEA(self): # deg_per_step is key parameter for this function
        self.flag_static_solver = True
        self.flag_eddycurrent_solver = False

        femm.openfemm(True) # bHide # False for debug
        femm.newdocument(0) # magnetic
        self.freq = 0 # static 
        self.probdef()

        if self.deg_per_step == 0.0:
            print 'Locked Rotor! Run 40 stEPS for one slip period.'
            self.im.update_mechanical_parameters(syn_freq=0.0)
        # read currents from previous ec solve
        self.read_current_from_EC_FEA() # DriveW_Freq and slip_freq_breakdown_torque are used here

        self.time = 0.0
        self.rotor_position_in_deg = 0.0
        self.add_material()
        self.draw_model()
        self.add_block_labels()

        # # debug here
        # femm.mi_maximize()
        # femm.mi_zoomnatural()
        # return

        output_file_name = None
        last_out_file_name = None

        if self.deg_per_step == 0.0:
            for i in range(40): # don't forget there
                self.time += 1.0/self.im.DriveW_Freq / 40. # don't forget here
                # self.rotor_position_in_deg = i # used in file naming
                print i, self.time, 's'

                last_out_file_name = output_file_name
                output_file_name = self.output_file_name + '%04d'%(i)
                if os.path.exists(output_file_name + '.ans'):
                    print '.ans file exists. skip this fem file: %s' % (output_file_name)
                    continue
                if last_out_file_name != None:
                    femm.opendocument(last_out_file_name + '.fem')
                    self.model_rotor_rotate()

                femm.mi_saveas(output_file_name + '.fem')
        else:
            for self.rotor_position_in_deg in np.arange(0, 180, self.deg_per_step):
                self.time = np.abs(self.rotor_position_in_deg/180*pi / self.im.Omega) # DEBUG: 查了这么久的BUG，原来就是转速用错了！应该用机械转速啊！

                print self.time, 's', self.rotor_position_in_deg, 'deg'

                last_out_file_name = output_file_name
                output_file_name = self.output_file_name + '%04d'%(10*self.rotor_position_in_deg)
                if os.path.exists(output_file_name + '.ans'):
                    print '.ans file exists. skip this fem file: %s' % (output_file_name)
                    continue
                if last_out_file_name != None:
                    femm.opendocument(last_out_file_name + '.fem')
                    self.model_rotor_rotate()

                femm.mi_saveas(output_file_name + '.fem')

                # if self.rotor_position_in_deg> 10:
                #     # debug here
                #     femm.mi_maximize()
                #     femm.mi_zoomnatural()
                #     return
            self.number_ans = len(np.arange(0, 180, self.deg_per_step))
        femm.closefemm()

        # call parasolve.py instead
        if False:
            '''直接创造360个fem文件，然后调用18个Python Instantce同步求解。https://stackoverflow.com/questions/19156467/run-multiple-instances-of-python-script-simultaneously'''

            if not os.path.exists(output_file_name + '.ans'):
                logging.getLogger().info('Rotating: %g rpm, %g deg.'%(self.im.the_speed, self.rotor_position_in_deg))
                femm.mi_analyze(1) # None for inherited. 1 for a minimized window,
                femm.mi_loadsolution()
            else:
                femm.opendocument(output_file_name + '.fem')
                femm.mi_loadsolution()

            # get the torque on the rotor
            femm.mo_groupselectblock(100)
            Fx = femm.mo_blockintegral(18) #-- 18 x (or r) part of steady-state weighted stress tensor force
            Fy = femm.mo_blockintegral(19) #--19 y (or z) part of steady-state weighted stress tensor force
            torque = femm.mo_blockintegral(22) #-- 22 = Steady-state weighted stress tensor torque
            femm.mo_clearblock()

            # write results to a data file
            with open(self.dir_run + "static_results.txt", "a") as f:

                results_rotor = "%g rpm. [Rotor] %g deg. %g Nm. %g N. %g N. " \
                    % ( self.im.the_speed, self.rotor_position_in_deg, torque, Fx, Fy)

                f.write(results_rotor + '\n')

    def parallel_solve(self, number_of_instantces=5, bool_watchdog_postproc=True):
        if True: # for running script in JMAG
            # os.system('python "%smethod_parallel_solve_4jmag.py" %s' % (self.dir_codes, self.dir_run))
            with open('temp.bat', 'w') as f:
                f.write('python "%smethod_parallel_solve_4jmag.py" %s %d' % (self.dir_codes, self.dir_run, number_of_instantces))
            os.startfile('temp.bat')
            # os.remove('temp.bat')

        else: # run script in other platforms such as command prompt
            procs = []
            for i in range(number_of_instantces):
                # proc = subprocess.Popen([sys.executable, 'parasolve.py', '{}in.csv'.format(i), '{}out.csv'.format(i)], bufsize=-1)
                proc = subprocess.Popen([sys.executable, 'parasolve.py', str(i), str(number_of_instantces), self.dir_run], bufsize=-1)
                procs.append(proc)

            for proc in procs:
                proc.wait()

        # TODO: loop for post_process results
        # search for python: constently check for new ans file
        if self.im.fea_config_dict['pc_name'] == 'Y730':
            print 'Initialize watchdog...'
        else:
            print 'watchdog is not installed on servers, quit.'
            return

        if bool_watchdog_postproc:
            import time
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            class MyHandler(FileSystemEventHandler):
                def __init__(self, solver):
                    self.count_ans = 0
                    self.bool_stop = False
                    self.solver = solver
                    super(FileSystemEventHandler, self).__init__()
                def on_created(self, event):
                    if '.ans' in event.src_path:
                        self.count_ans += 1
                        if self.solver.flag_eddycurrent_solver == True:
                            if self.count_ans == len(self.solver.freq_range):
                                self.solver.show_results(bool_plot=True)
                                self.bool_stop = True

                        if self.solver.flag_static_solver == True:
                            # write data to file per 10 .ans files

                            if self.count_ans == self.solver.number_ans or self.count_ans == int(0.5*self.solver.number_ans):
                                if self.solver.has_results():
                                    self.solver.show_results(bool_plot=True)
                                    self.bool_stop = True
                                else:
                                    self.solver.show_results(bool_plot=False)      
            event_handler = MyHandler(self)
            observer = Observer()
            observer.schedule(event_handler, path=self.dir_run, recursive=False)
            observer.start()
            try:
                while not event_handler.bool_stop:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            else:
                print 'after viewing the plot, watchdog is killed.'
                observer.stop()
            observer.join()

    def read_current_from_EC_FEA(self):
        if self.flag_read_from_jmag == True:
            self.read_current_conditions_from_JMAG()
        else:
            self.read_current_conditions_from_FEMM()

    def read_current_conditions_from_FEMM(self):
        pass

    def read_current_conditions_from_JMAG(self):
        try:
            print 'The breakdown torque slip frequency is', self.im.slip_freq_breakdown_torque
        except:
            raise Exception('no breakdown torque slip freqeuncy available.')

        self.dict_rotor_current_from_EC_FEA = []
        self.dict_stator_current_from_EC_FEA = []

        rotor_phase_name_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        with open(self.im.csv_previous_solve, 'r') as f:
            read_iterator = csv_reader(f, skipinitialspace=True)
            for row in self.whole_row_reader(read_iterator):
                try:
                    float(row[0])
                except:
                    continue
                else:
                    if np.abs(self.im.slip_freq_breakdown_torque - float(row[0])) < 1e-3:
                        # print row
                        ''' Rotor Current '''
                        beginning_column = 1 + 2*3*2 # title + drive/bearing * 3 phase * real/imag
                        for i in range(0, int(self.im.Qr/self.im.DriveW_poles)):
                            natural_i = i+1
                            current_phase_column = beginning_column + i * int(self.im.DriveW_poles) * 2
                            for j in range(int(self.im.DriveW_poles)):
                                natural_j = j+1
                                re = float(row[current_phase_column+2*j])
                                im = float(row[current_phase_column+2*j+1])
                                self.dict_rotor_current_from_EC_FEA.append( ("r%s%d"%(rotor_phase_name_list[i], natural_j), (re, im)) )

                        ''' Stator Current '''
                        beginning_column = 1 # title column is not needed
                        for i, phase in zip(range(0,12,2), ['2A','2B','2C','4A','4B','4C']): # 3 phase
                            natural_i = i+1
                            current_phase_column = beginning_column + i
                            re = float(row[current_phase_column])
                            im = float(row[current_phase_column+1])
                            self.dict_stator_current_from_EC_FEA.append( (phase, (re, im)) )

        print 'Rotor Current'
        self.dict_rotor_current_from_EC_FEA = OrderedDict(self.dict_rotor_current_from_EC_FEA)
        self.dict_rotor_current_function = []
        for key, item in self.dict_rotor_current_from_EC_FEA.iteritems():
            amp = np.sqrt(item[1]**2 + item[0]**2)
            phase = np.arctan2(item[0], -item[1]) # atan2(y, x), y=a, x=-b

            if '1' in key:
                self.dict_rotor_current_function.append(lambda t, amp=amp, phase=phase: amp * sin(2*pi*self.im.slip_freq_breakdown_torque*t + phase))
                print '\t', key, item, amp, phase/pi*180

        print 'Stator Current'
        self.dict_stator_current_from_EC_FEA = OrderedDict(self.dict_stator_current_from_EC_FEA)
        self.dict_stator_current_function_wrong = []
        self.dict_stator_current_function = []
        for key, item in self.dict_stator_current_from_EC_FEA.iteritems():
            amp = np.sqrt(item[1]**2 + item[0]**2)
            phase = np.arctan2(item[0], -item[1]) # atan2(y, x), y=a, x=-b
            self.dict_stator_current_function_wrong.append(lambda t, amp=amp, phase=phase: amp * sin(2*pi*self.im.DriveW_Freq*t + phase))
            # self.dict_stator_current_function.append(lambda t, amp=amp, phase=phase: amp * sin(2*pi*self.im.slip_freq_breakdown_torque*t + phase))
            self.dict_stator_current_function.append(lambda t, amp=amp, phase=phase: amp * sin(2*pi*self.im.DriveW_Freq*t + phase))
            print '\t', key, item, amp, phase/pi*180
        # self.dict_stator_current_function =[self.dict_stator_current_function_wrong[0],
        #                                     self.dict_stator_current_function_wrong[2],
        #                                     self.dict_stator_current_function_wrong[1],
        #                                     self.dict_stator_current_function_wrong[3],
        #                                     self.dict_stator_current_function_wrong[5],
        #                                     self.dict_stator_current_function_wrong[4],]
        # print self.dict_stator_current_function
        if False:

            from pylab import show, figure
            t = np.arange(0, 0.5, 1e-4)
            # t = np.arange(0, 0.5, 1e-3) # down sampling effect of TranFEAwi2TSS. 5e-2 is too less
            ax1 = figure(1).gca()
            ax2 = figure(2).gca()

            ax = ax1
            rotor_current_one_pole = np.zeros(t.__len__())
            for ind, func in enumerate(self.dict_rotor_current_function):
                ax.plot(t, [func(el) for el in t], label=ind)
                rotor_current_one_pole += np.array([func(el) for el in t])
            ax.plot(t, rotor_current_one_pole, label='one pole')

            ax = ax2
            iabc_wrong = []
            for ind, func in enumerate(self.dict_stator_current_function_wrong):
                if ind == 3 or ind == 0:
                    ax.plot(t, [-func(el) for el in t], label=str(ind)+'wrong-reversed')
                iabc_wrong.append(np.array([func(el) for el in t]))

            ax = ax1
            iabc = []
            for ind, func in enumerate(self.dict_stator_current_function):
                ax.plot(t, [func(el) for el in t], label=ind)
                iabc.append(np.array([func(el) for el in t]))
            # amplitude-invariant transform - the alpha-beta frame is actually rotating, because iabs is in rotor ref frame (slip frequency)
            ialbe = 2/3.* np.dot(np.array([ [1,      -0.5,       -0.5], 
                                            [0, sqrt(3)/2, -sqrt(3)/2] ]), np.array([iabc[3],iabc[4],iabc[5]]))
            print np.shape(np.array([iabc[3],iabc[4],iabc[5]]))
            print np.shape(ialbe)

            print self.im.omega/2/pi
            ids = []
            iqs = []
            ''' Speed negation is done by ? '''
            iabc_stationary_2_tor = []
            iabc_stationary_2_sus = []
            for i in range(len(t)):
                # theta = -self.im.omega * t[i]
                # theta = -self.im.DriveW_Freq*2*pi * t[i]
                # theta = self.im.omega * t[i]
                theta = self.im.DriveW_Freq*2*pi * t[i]

                # turn into stationary dq frame
                temp = np.dot( np.array([ [np.cos(theta), -np.sin(theta)], 
                                          [np.sin(theta),  np.cos(theta)] ]), np.array([ [ialbe[0][i]], 
                                                                                         [ialbe[1][i]] ]) )
                ids.append(temp[0])
                iqs.append(temp[1])

                iabc_stationary_2_tor.append(self.transformed_dict_stator_current_function(3, t[i], theta))
                iabc_stationary_2_sus.append(self.transformed_dict_stator_current_function(0, t[i], theta))

            ids = np.array(ids).T[0]
            iqs = np.array(iqs).T[0]
            # print ids
            # print iqs
            print 'idq', np.shape(ids), np.shape(iqs)
            # ax_r.plot(t, idq[0], label='i_ds')
            # ax_r.plot(t, idq[1], label='i_qs')
            # ax_r.plot(t, ialbe[0], label='alpha')
            # ax_r.plot(t, ialbe[1], label='beta')




            # tansform to phase coordinates
            ax = ax2
            iabc_stationary = 1.5 * np.dot(np.array([ [ 2/3.,          0], 
                                                      [-1/3.,  sqrt(3)/3],
                                                      [-1/3., -sqrt(3)/3] ]), np.array([ids, iqs]))
            ax.plot(t, iabc_stationary[0], label='i_a')
            # ax.plot(t, iabc_stationary[1], label='i_b')
            # ax.plot(t, iabc_stationary[2], label='i_c')

            ax.plot(t, [el[0] for el in iabc_stationary_2_sus], label='i_a_2_sus')

            ax.plot(t, [el[0] for el in iabc_stationary_2_tor], label='i_a_2_tor')
            # ax.plot(t, [el[1]+5 for el in iabc_stationary_2_tor], label='i_b_2')
            # ax.plot(t, [el[2]+5 for el in iabc_stationary_2_tor], label='i_c_2')

            ax1.legend()
            ax2.legend()
            show()

            quit()

    def transformed_dict_stator_current_function(self, index_phase_A, time, theta):
        ia_slip_freq = self.dict_stator_current_function[index_phase_A](time)
        ib_slip_freq = self.dict_stator_current_function[index_phase_A+1](time)
        ic_slip_freq = self.dict_stator_current_function[index_phase_A+2](time)

        iabc_vector_slip_freq = np.array([[ia_slip_freq, ib_slip_freq, ic_slip_freq]]).T
        ialbe = 2/3.* np.dot(np.array([ [1,      -0.5,       -0.5], 
                                        [0, sqrt(3)/2, -sqrt(3)/2] ]), iabc_vector_slip_freq)
        # print 'ialbe', ialbe

        # turn into stationary dq frame
        idq = np.dot( np.array([ [np.cos(theta), -np.sin(theta)], 
                                 [np.sin(theta),  np.cos(theta)] ]), ialbe )
        # print 'idq', idq

        iabc_stationary = 1.5 * np.dot(np.array([ [ 2/3.,          0], 
                                                  [-1/3.,  sqrt(3)/3],
                                                  [-1/3., -sqrt(3)/3] ]), idq)

        return iabc_stationary[0][0], iabc_stationary[1][0], iabc_stationary[2][0]

    def get_air_gap_B(self, number_of_points=360):
        im = self.im
        femm.opendocument(self.output_file_name + '.fem')
        femm.mi_loadsolution()

        list_B_magitude = []
        R = im.Radius_OuterRotor + 0.25*im.Length_AirGap
        for i in range(number_of_points):
            THETA = i / 180.0 * pi
            X = R*cos(THETA)
            Y = R*sin(THETA)
            B_vector_complex = femm.mo_getb(X, Y)
            B_X_complex = B_vector_complex[0]
            B_Y_complex = B_vector_complex[1]
            B_X_real = np.real(B_X_complex)
            B_Y_real = np.real(B_Y_complex)
            # Assume the magnitude is all due to radial component
            B_magitude = sqrt(B_X_real**2 + B_Y_real**2)
            inner_product = B_X_real * X + B_Y_real *Y
            list_B_magitude.append( B_magitude * copysign(1, inner_product) )
        return list_B_magitude

    def probdef(self):
        femm.mi_probdef(self.freq, 'millimeters', 'planar', 1e-8, # must < 1e-8
                        self.stack_length, 18, 1) # The acsolver parameter (default: 0) specifies which solver is to be used for AC problems: 0 for successive approximation, 1 for Newton.
                                             # 1 for 'I intend to try the acsolver of Newton, as this is the default for JMAG@[Nonlinear Calculation] Setting Panel in the [Study Properties] Dialog Box'
        femm.smartmesh(False) # let mi_smartmesh deside. You must turn it off in parasolver.py
        # femm.mi_smartmesh(False)
        self.bool_automesh = False # setting to false gives no effect?

    def add_material(self):
        # mi_addmaterial('matname', mu x, mu y, H c, J, Cduct, Lam d, Phi hmax, lam fill, LamType, Phi hx, Phi hy, nstr, dwire)
        femm.mi_getmaterial('Air')
        femm.mi_getmaterial('Copper') # for coil
        # femm.mi_getmaterial('18 AWG') # for coil
        # femm.mi_getmaterial('Aluminum, 1100') # for bar?
        # femm.mi_getmaterial('304 Stainless Steel') # for shaft?
        # femm.mi_getmaterial('M-15 Steel') # for Stator & Rotor Iron Cores (Nonlinear with B-H curve)

        # femm.mi_addmaterial('Air', 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0);
        # femm.mi_addmaterial('Aluminum', 1, 1, 0, 0, 35, 0, 0, 1, 0, 0, 0)
        femm.mi_addmaterial('Aluminum', 1, 1, 0, 0, self.im.fea_config_dict['Bar_Conductivity']*1e-6, 0, 0, 1, 0, 0, 0) # [MS/m]
        # femm.mi_addmaterial('Aluminum', 1, 1, 0, 0, 1/1.673e-2, 0, 0, 1, 0, 0, 0)

        # femm.mi_addmaterial('LinearIron', 2000, 2000, 0, 0, 0, 0, 0, 1, 0, 0, 0);

        if True:
            femm.mi_addmaterial('My M-15 Steel',0,0, 0,0,0,0,0, 1)
            BH = np.loadtxt(self.dir_codes + 'M-15-Steel-BH-Curve.txt', unpack=True, usecols=(0,1))
            bdata = BH[1]
            hdata = BH[0]
            for n in range(0,len(bdata)):
                femm.mi_addbhpoint('My M-15 Steel', bdata[n], hdata[n])


        if False:
            # A more interesting material to add is the iron with a nonlinear
            # BH curve.  First, we create a material in the same way as if we
            # were creating a linear material, except the values used for
            # permeability are merely placeholders.
            femm.mi_addmaterial('Arnon5',0,0,0,0,1.9,0.127,0,0.96)
            # A set of points defining the BH curve is then specified.
            BH = np.loadtxt(self.dir_codes + 'Arnon5_Kang_after_JMAG_Smoothed.txt', unpack=True, usecols=(0,1))
            bdata = BH[1]
            hdata = BH[0]
            for n in range(0,len(bdata)):
                femm.mi_addbhpoint('Arnon5', bdata[n], hdata[n])

    def get_output_file_name(self, booL_dir=True):
        if booL_dir == True:
            self.output_file_name = self.dir_run + '%d-%gHz'%(int(self.im.Qr), self.freq)
            return self.output_file_name
        else:
            return '%d-%gHz'%(int(self.im.Qr), self.freq)

    def whole_row_reader(self, reader):
        for row in reader:
            yield row[:]

    def has_results(self):
        a = [f for f in os.listdir(self.dir_run) if '.ans' in f].__len__()
        b = [f for f in os.listdir(self.dir_run) if '.fem' in f].__len__()
        if a == 0:
            return False
        print 'ans count: %d. fem count: %d.' % (a, b)
        return a == b

    def show_results(self, bool_plot=True):

        if self.flag_static_solver == True:
            print 'show results for static solver'
            return self.show_results_static(bool_plot=bool_plot)

        if self.flag_eddycurrent_solver == True:
            print 'show results for eddy current solver'
            return self.show_results_eddycurrent(bool_plot=bool_plot)

        return None

    def show_results_eddycurrent(self, bool_plot):
        im = self.im

        ans_file_list = os.listdir(self.dir_run)
        ans_file_list = [f for f in ans_file_list if '.ans' in f]

        femm.openfemm(True)
        for ind, f in enumerate( ans_file_list ):
            femm.opendocument(self.dir_run + f[:-4] + '.fem')
            femm.mi_loadsolution()

            # physical amount on rotor
            femm.mo_groupselectblock(100)
            Fx = femm.mo_blockintegral(18) #-- 18 x (or r) part of steady-state weighted stress tensor force
            Fy = femm.mo_blockintegral(19) #--19 y (or z) part of steady-state weighted stress tensor force
            torque = femm.mo_blockintegral(22) #-- 22 = Steady-state weighted stress tensor torque
            femm.mo_clearblock()

            # physical amount of Cage
            vals_results_rotor_current = []
            R = 0.5*(im.Location_RotorBarCenter + im.Location_RotorBarCenter2)
            angle_per_slot = 2*pi/im.Qr
            THETA_BAR = pi - angle_per_slot + EPS # add EPS for the half bar
            for i in range(self.rotor_slot_per_pole * int(4/self.fraction)):
                THETA_BAR += angle_per_slot
                THETA = THETA_BAR
                X = R*cos(THETA); Y = R*sin(THETA)
                femm.mo_selectblock(X, Y)
                vals_results_rotor_current.append(femm.mo_blockintegral(7)) # integrate for current
                femm.mo_clearblock()
            # the other half bar 
            THETA_BAR += angle_per_slot
            THETA = THETA_BAR - 2*EPS
            X = R*cos(THETA); Y = R*sin(THETA)
            femm.mo_selectblock(X, Y)
            vals_results_rotor_current.append(femm.mo_blockintegral(7)) # integrate for current
            femm.mo_clearblock()

            if float(f[3:-6]) == 3:
                print '\n', f[3:-6], 'Hz'
                for el in vals_results_rotor_current:
                    print abs(el)

            # write results to a data file
            with open(self.dir_run + "eddycurrent_results.txt", "a") as stream:
                stream.write("%s %g %g %g\n" % ( f[3:-6], torque, Fx, Fy ))
            femm.mo_close() 

        data = np.loadtxt(self.dir_run + "eddycurrent_results.txt", unpack=True, usecols=(0,1,2,3))

        # use dict to sort
        results_dict = {}
        for i in range(len(data[0])):
            results_dict[data[0][i]] = (data[1][i], data[2][i], data[3][i]) 
        keys_without_duplicates = list(OrderedDict.fromkeys(data[0]))
        keys_without_duplicates.sort()

        # write data in order
        with open(self.dir_run + "eddycurrent_results_sorted.txt", 'w') as f:
            for key in keys_without_duplicates:
                f.writelines('%g %g %g %g\n' % (key, results_dict[key][0], results_dict[key][1], results_dict[key][2]))
            print '[Results] the last key is', max(keys_without_duplicates), '[begin from 0]. the length of keys is', len(keys_without_duplicates)

        data = np.loadtxt(self.dir_run + "eddycurrent_results_sorted.txt", unpack=True, usecols=(0,1,2,3))
        self.plot_results(data)
        femm.closefemm()
        print 'done. append to eddycurrent_results.txt.'

    def show_results_static(self, bool_plot):
        # if exists .txt file, then load it
        missed_ans_file_list = []
        if os.path.exists(self.dir_run + "static_results.txt"):
            data = np.loadtxt(self.dir_run + "static_results.txt", unpack=True, usecols=(0,1,2,3))

            # use dict to eliminate duplicates
            results_dict = {}
            for i in range(len(data[0])):
                results_dict[data[0][i]] = (data[1][i], data[2][i], data[3][i]) 
            keys_without_duplicates = list(OrderedDict.fromkeys(data[0]))
            keys_without_duplicates.sort()

            # check for missed .ans files
            for i in range(0, int(max(keys_without_duplicates)), int(10*self.deg_per_step)):
                if i not in keys_without_duplicates:
                    missed_ans_file_list.append( self.get_output_file_name(False) + '%04d'%(i) + '.ans' )
            print 'missed:', missed_ans_file_list

            # typical print gives: 5 1813 1795 1799.0
            # print len(missed_ans_file_list), len(data[0]), len(keys_without_duplicates), keys_without_duplicates[-1]
            # quit()

            # write data without duplicates to file
            with open(self.dir_run + "static_results_no_duplicates.txt", 'w') as f:
                for key in keys_without_duplicates:
                    f.writelines('%g %g %g %g\n' % (key, results_dict[key][0], results_dict[key][1], results_dict[key][2]))
                print '[Results] the last key is', max(keys_without_duplicates), '[begin from 0]. the length of keys is', len(keys_without_duplicates)

            data = np.loadtxt(self.dir_run + "static_results_no_duplicates.txt", unpack=True, usecols=(0,1,2,3))

            last_index = len(data[0])
        else:
            last_index = 0

        ans_file_list = os.listdir(self.dir_run)
        ans_file_list = [f for f in ans_file_list if '.ans' in f]

        # last_index == 0 则说明是第一次运行后处理
        if last_index > 0:
            if len(ans_file_list) <= last_index:
                if bool_plot == True:
                    self.plot_results(data)
                return data
            else:
                print 'There are new .ans files. Now append them'

        # iter ans_file_list and missed_ans_file_list, and write to .txt 
        femm.openfemm(True) # bHide
        print 'there are total %d .ans files'%(len(ans_file_list))
        print 'I am going to append the rest %d ones.'%(len(ans_file_list) - last_index)
        for ind, f in enumerate( ans_file_list[last_index:] + missed_ans_file_list ):
            if ind >= len(ans_file_list[last_index:]):
                print '...open missed .ans files'
                if os.path.exists(self.dir_run + f) == False:
                    print 'run mi_analyze for %s' % (f)
                    femm.opendocument(self.dir_run + f[:-4] + '.fem')
                    femm.mi_analyze(1)
                else:
                    femm.opendocument(self.dir_run + f[:-4] + '.fem')                    
            else:
                print last_index + ind,
                femm.opendocument(self.dir_run + f[:-4] + '.fem')

            # load solution (if corrupted, re-run)
            try:
                femm.mi_loadsolution()
            except Exception, e:
                logger = logging.getLogger(__name__)
                logger.error(u'The .ans file to this .fem file is corrupted. re-run the .fem file %s'%(f), exc_info=True)
                femm.opendocument(self.dir_run + f[:-4] + '.fem')
                femm.mi_analyze(1)
                femm.mi_loadsolution()

            # get the physical amounts on the rotor
            try:
                femm.mo_groupselectblock(100)
                Fx = femm.mo_blockintegral(18) #-- 18 x (or r) part of steady-state weighted stress tensor force
                Fy = femm.mo_blockintegral(19) #--19 y (or z) part of steady-state weighted stress tensor force
                torque = femm.mo_blockintegral(22) #-- 22 = Steady-state weighted stress tensor torque
                femm.mo_clearblock()

                # Air Gap Boundary for Rotor Motion #5
                # gap_torque = femm.mo_gapintegral("AGB4RM", 0)
                # gap_force = femm.mo_gapintegral("AGB4RM", 1)
                # print gap_force, gap_torque, torque, Fx, Fy

                # write results to a data file
                with open(self.dir_run + "static_results.txt", "a") as stream:
                    stream.write("%s %g %g %g\n" % ( f[-8:-4], torque, Fx, Fy ))
            except Exception, e:
                logger = logging.getLogger(__name__)
                logger.error(u'Encounter error while integrating...', exc_info=True)
                raise e

            # avoid run out of RAM when there are a thousand of ans files loaded into femm...
            # if ind % 10 == 0:
            #     femm.closefemm() 
            #     femm.openfemm(True)
            femm.mo_close()  # use mo_ to close .ans file
            femm.mi_close()  # use mi_ to close .fem file

        print 'done. append to static_results.txt.'
        femm.closefemm()

        try:
            data
        except:
            print 'call this method again to plot...'
            return None
        if bool_plot == True:
            self.plot_results(data)
        return data

    def write_physical_data(self, results_list):
        with open(self.dir_run + "static_results.txt", "a") as f:
            results_rotor = ''
            for ind, row in enumerate(results_list):
                results_rotor += "%s %g %g %g\n" \
                    % ( row[0][-8:-4], row[1], row[2], row[3] )
            f.write(results_rotor)        

    def plot_results(self, data):
        from pylab import subplots, legend, show

        if self.flag_eddycurrent_solver:
            fig, axes = subplots(3, 1, sharex=True)
            ax = axes[0]; ax.plot(data[0], data[1],label='torque'); ax.legend(); ax.grid()
            ax = axes[1]; ax.plot(data[0], data[2],label='Fx'); ax.legend(); ax.grid()
            ax = axes[2]; ax.plot(data[0], data[3],label='Fy'); ax.legend(); ax.grid()
            show()

        if self.flag_static_solver:
            fig, axes = subplots(3, 1, sharex=True)
            ax = axes[0]; ax.plot(data[0]*0.1, data[1],label='torque'); ax.legend(); ax.grid()
            ax = axes[1]; ax.plot(data[0]*0.1, data[2],label='Fx'); ax.legend(); ax.grid()
            ax = axes[2]; ax.plot(data[0]*0.1, data[3],label='Fy'); ax.legend(); ax.grid()
            show()

    def run_frequency_sweeping(self, freq_range, fraction=2):
        self.flag_static_solver = False
        self.flag_eddycurrent_solver = True
        self.fraction = fraction

        for f in os.listdir(self.dir_run):
            os.remove(self.dir_run + f)

        femm.openfemm(True) # bHide # False for debug
        femm.newdocument(0) # magnetic
        self.freq_range = freq_range
        self.freq = freq_range[0]
        # Alternatively, the length of the machine could be scaled by the number of segments to make this correction automatically.
        self.stack_length *= fraction
        self.probdef()

        self.add_material()
        self.draw_model(fraction=fraction)
        self.add_block_labels(fraction=fraction)

        # # debug here
        # femm.mi_maximize()
        # femm.mi_zoomnatural()
        # return

        for freq in freq_range:
            self.freq = freq
            self.output_file_name = self.get_output_file_name()
            print self.output_file_name + '.ans',
            print os.path.exists(self.output_file_name + '.ans')
            if os.path.exists(self.output_file_name + '.ans'):
                continue
            self.probdef()        
            femm.mi_saveas(self.output_file_name + '.fem')

        # flux and current of circuit can be used for parameter identification
        if False:
            dict_circuits = {}
            # i = femm.mo_getprobleminfo()
            logging.getLogger().info('Sweeping: %g Hz.'%(self.freq))
            femm.mi_analyze(1) # None for inherited. 1 for a minimized window,
            femm.mi_loadsolution()

            # circuit
                # i1_re,i1_im, v1_re,v1_im, flux1_re,flux1_im = femm.mo_getcircuitproperties("dA")
                # i2_re,i2_im, v2_re,v2_im, flux2_re,flux2_im = femm.mo_getcircuitproperties("bA")
                # i3_re,i3_im, v3_re,v3_im, flux3_re,flux3_im = femm.mo_getcircuitproperties("rA")
            dict_circuits['dA'] = femm.mo_getcircuitproperties("dA")
            dict_circuits['dB'] = femm.mo_getcircuitproperties("dB")
            dict_circuits['dC'] = femm.mo_getcircuitproperties("dC")
            dict_circuits['bA'] = femm.mo_getcircuitproperties("bA")
            dict_circuits['bB'] = femm.mo_getcircuitproperties("bB")
            dict_circuits['bC'] = femm.mo_getcircuitproperties("bC")
            for i in range(self.rotor_slot_per_pole):
                circuit_name = 'r%s'%(self.rotor_phase_name_list[i])
                dict_circuits[circuit_name] = femm.mo_getcircuitproperties(circuit_name)

            # write results to a data file, multiplying by ? to get
            # the results for all ? poles of the machine. # 如果只分析了对称场，记得乘回少掉的那部分。
            with open(self.dir_run + "results.txt", "a") as f:

                results_circuits = "[DW] %g + j%g A. %g + j%g V. %g + j%g Wb. [BW] %g + j%g A. %g + j%g V. %g + j%g Wb. [BAR] %g + j%g A. %g + j%g V. %g + j%g Wb. " \
                    % ( np.real(dict_circuits['dA'][0]), np.imag(dict_circuits['dA'][0]),
                        np.real(dict_circuits['dA'][1]), np.imag(dict_circuits['dA'][1]),
                        np.real(dict_circuits['dA'][2]), np.imag(dict_circuits['dA'][2]),
                        np.real(dict_circuits['bA'][0]), np.imag(dict_circuits['bA'][0]),
                        np.real(dict_circuits['bA'][1]), np.imag(dict_circuits['bA'][1]),
                        np.real(dict_circuits['bA'][2]), np.imag(dict_circuits['bA'][2]),
                        np.real(dict_circuits['rA'][0]), np.imag(dict_circuits['rA'][0]),
                        np.real(dict_circuits['rA'][1]), np.imag(dict_circuits['rA'][1]),
                        np.real(dict_circuits['rA'][2]), np.imag(dict_circuits['rA'][2]))

                results_rotor = "%g Hz. [Rotor] %g Nm. %g N. %g N. " \
                    % ( freq, torque, Fx, Fy)

                f.write(results_rotor + '\n')

    def get_rotor_current_function(self, i=0):
        try:
            dict_rotor_current_function
        except:
            print '\n\n\nrotor current function does not exist. build it now...'
            self.read_current_conditions_from_JMAG()
        return self.dict_rotor_current_function[i]
