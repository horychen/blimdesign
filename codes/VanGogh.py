# coding:u8
from shapely.geometry import LineString
from shapely.geometry import Point
from math import tan, pi, atan, sqrt, sin, cos, copysign, atan2

CUSTOM = 2
JMAG = 1
FEMM = 0

class VanGogh(object):
    """One VanGogh for both FEMM and JMAG"""
    def __init__(self, im, child_index):
        self.im = im
        self.child_index = child_index
        self.plot_object_list = []

    def draw_model(self, fraction=1):
        # utility functions than wrap shapely functions

        if self.child_index == JMAG: # for being consistent with obselete codes
            self.plot_sketch_shaft() # Shaft if any
            self.draw_rotor(fraction)
            self.draw_stator(fraction)

        elif self.child_index == FEMM: # for easy selecting of objects
            self.draw_stator(fraction)
            try:
                self.draw_rotor(fraction)
            except Exception as e:
                raise e

        elif self.child_index == CUSTOM: 
            self.draw_rotor_without_non_accurate_shapely()
            self.draw_stator_without_non_accurate_shapely()

    def draw_stator(self, fraction):
        im = self.im

        origin = Point(0,0)
        Stator_Sector_Angle = 2*pi/im.Qs*0.5
        Rotor_Sector_Angle = 2*pi/im.Qr*0.5

        ''' Part: Stator '''
        if self.child_index == JMAG:
            self.init_sketch_statorCore()
        # Draw Points as direction of CCW
        # P1
        P1 = (-im.Radius_OuterRotor-im.Length_AirGap, 0)

        # P2
        # Parallel to Line? No they are actually not parallel
        P2_angle = Stator_Sector_Angle -im.Angle_StatorSlotOpen*0.5/180*pi
        k = -tan(P2_angle) # slope
        l_sector_parallel = LineString([(0,0), (-im.Radius_OuterStatorYoke, -im.Radius_OuterStatorYoke*k)])
        c = self.create_circle(origin, im.Radius_OuterRotor+im.Length_AirGap)
        P2 = self.get_node_at_intersection(c,l_sector_parallel)
        self.draw_arc(P2, P1, self.get_postive_angle(P2))


        # P3
        c = self.create_circle(origin, im.Radius_OuterRotor+im.Length_AirGap+im.Width_StatorTeethHeadThickness)
        P3 = self.get_node_at_intersection(c,l_sector_parallel)
        self.draw_line(P2, P3)

        # P4
        c = self.create_circle(origin, im.Radius_OuterRotor+im.Length_AirGap+im.Width_StatorTeethHeadThickness+im.Width_StatorTeethNeck)
        l = LineString([(0, 0.5*im.Width_StatorTeethBody), (-im.Radius_OuterStatorYoke, 0.5*im.Width_StatorTeethBody)])
        P4 = self.get_node_at_intersection(c,l)
        self.draw_line(P3, P4)

        # P5
        c = self.create_circle(origin, im.Radius_InnerStatorYoke)
        P5 = self.get_node_at_intersection(c,l)
        self.draw_line(P4, P5)

        # P6
        k = -tan(Stator_Sector_Angle)
        l_sector = LineString([(0,0), (-im.Radius_OuterStatorYoke, -im.Radius_OuterStatorYoke*k)])
        # P6 = self.get_node_at_intersection(c,l_sector)
        P6 = [ -im.Radius_InnerStatorYoke*cos(Stator_Sector_Angle), im.Radius_InnerStatorYoke*sin(Stator_Sector_Angle) ]
        self.draw_arc(P6, P5, Stator_Sector_Angle - self.get_postive_angle(P5))

        # P7
        # c = self.create_circle(origin, im.Radius_OuterStatorYoke)
        # P7 = self.get_node_at_intersection(c,l_sector)
        P7 = [ -im.Radius_OuterStatorYoke*cos(Stator_Sector_Angle), im.Radius_OuterStatorYoke*sin(Stator_Sector_Angle) ]

        # P8
        P8 = (-im.Radius_OuterStatorYoke, 0)

        if self.child_index == JMAG:
            self.draw_line(P6, P7)
            self.draw_arc(P7, P8, Stator_Sector_Angle)
            self.draw_line(P8, P1)
        # if self.child_index == JMAG:
            self.mirror_and_copyrotate(im.Qs, None, fraction,
                                        symmetry_type=2,  # 2: x-axis
                                        )
        # if self.child_index == JMAG:
            self.init_sketch_coil()

        # run#118
        # print 'Stator: P1-P8:'
        # print P1
        # print P2
        # print P3
        # print P4
        # print P5
        # print P6
        # print P7
        # print P8
        # quit() 

        # P_Coil
        l = LineString([(P3[0], P3[1]), (P3[0], im.Radius_OuterStatorYoke)])
        P_Coil = self.get_node_at_intersection(l_sector, l)

        if self.child_index == JMAG: # F*ck you, Shapely for putting me through this!
            #     temp = 0.7071067811865476*(P_Coil[1] - P4[1])
            #     P4 = [                      P4[0], P4[1] + temp]
            #     P5 = [P5[0] + 0.7071067811865476*(P4[0] - P5[0]), P5[1] + temp]
            #     P6 = []
            # we use sin cos to find P6 and P7, now this suffices.
            P6[1] -= 0.01
            # Conclusion: do not use the intersection between l_sector and a circle!
            # Conclusion: do not use the intersection between l_sector and a circle!
            # Conclusion: do not use the intersection between l_sector and a circle!
            # 总之，如果overlap了，merge一下是没事的，麻烦就在Coil它不能merge，所以上层绕组和下层绕组之间就产生了很多Edge Parts。

        self.draw_line(P4, P_Coil)
        self.draw_line(P6, P_Coil)

        if self.child_index == JMAG:
            # draw the outline of stator core for coil to form a region in JMAG
            self.draw_line(P4, P5)
            self.draw_arc(P6, P5, Stator_Sector_Angle - self.get_postive_angle(P5))
            self.mirror_and_copyrotate(im.Qs, None, fraction,
                                        edge4ref=self.artist_list[1], #'Line.2' 
                                        # symmetry_type=2,
                                        merge=False, # two layers of windings
                                        do_you_have_region_in_the_mirror=True # In short, this should be true if merge is false...
                                        )
            # it is super wierd that use edge Line.2 as symmetry axis will lead to redundant parts imported into JMAG Designers (Extra Coil and Stator Core Parts)
            # symmetry_type=2 will not solve this problem either
            # This is actually caused by the overlap of different regions, because the precision of shapely is shit!

        # FEMM does not model coil and stator core separately        
        if self.child_index == FEMM:
            self.mirror_and_copyrotate(im.Qs, im.Radius_OuterStatorYoke, fraction)

    def draw_rotor(self, fraction):
        im = self.im

        origin = Point(0,0)
        Stator_Sector_Angle = 2*pi/im.Qs*0.5
        Rotor_Sector_Angle = 2*pi/im.Qr*0.5

        ''' Part: Rotor '''
        if self.child_index == JMAG:
            self.init_sketch_rotorCore()
        # Draw Points as direction of CCW
        # P1
        # femm.mi_addnode(-im.Radius_Shaft, 0)
        P1 = (-im.Radius_Shaft, 0)

        # P2
        c = self.create_circle(origin, im.Radius_Shaft)
        # Line: y = k*x, with k = -tan(2*pi/im.Qr*0.5)
        P2_angle = P3_angle = Rotor_Sector_Angle
        k = -tan(P2_angle)
        l_sector = LineString([(0,0), (-im.Radius_OuterStatorYoke, -im.Radius_OuterStatorYoke*k)])
        try:
                # two ways to get P2
                # P2 = self.get_node_at_intersection(c,l_sector)
                # print P2
            P2 = ( -im.Radius_Shaft*cos(Rotor_Sector_Angle), im.Radius_Shaft*sin(Rotor_Sector_Angle) )
                # print P2
        except Exception as e:
            raise e # IOError: [Errno 9] Bad file descriptor??? it is possible that the .fem file is scaned by anti-virus software?

        # P3
        try:
            c = self.create_circle(origin, im.Radius_OuterRotor)
                # two ways to get P3
                # P3 = self.get_node_at_intersection(c,l_sector)
                # print P3
            P3 = ( -im.Radius_OuterRotor*cos(Rotor_Sector_Angle), im.Radius_OuterRotor*sin(Rotor_Sector_Angle) )
                # print P3
        except Exception as e:
            raise e # IOError: [Errno 9] Bad file descriptor??? it is possible that the .fem file is scaned by anti-virus software?

        # P4
        l = LineString([(-im.Location_RotorBarCenter, 0.5*im.Width_RotorSlotOpen), (-im.Radius_OuterRotor, 0.5*im.Width_RotorSlotOpen)])
        P4 = self.get_node_at_intersection(c,l)
        self.draw_arc(P3, P4, P3_angle - self.get_postive_angle(P4))

        # P5
        p = Point(-im.Location_RotorBarCenter, 0)
        c = self.create_circle(p, im.Radius_of_RotorSlot)
        P5 = self.get_node_at_intersection(c,l)
        self.draw_line(P4, P5)

        # P6
        # femm.mi_addnode(-im.Location_RotorBarCenter, im.Radius_of_RotorSlot)
        P6 = (-im.Location_RotorBarCenter, im.Radius_of_RotorSlot)
        self.draw_arc(P6, P5, 0.5*pi - self.get_postive_angle(P5, c.centroid.coords[0]))

        # P7
        # femm.mi_addnode(-im.Location_RotorBarCenter2, im.Radius_of_RotorSlot2)
        P7 = (-im.Location_RotorBarCenter2, im.Radius_of_RotorSlot2)
        self.draw_line(P6, P7)

        # P8
        # femm.mi_addnode(-im.Location_RotorBarCenter2+im.Radius_of_RotorSlot2, 0)
        P8 = (-im.Location_RotorBarCenter2+im.Radius_of_RotorSlot2, 0)
        self.draw_arc(P8, P7, 0.5*pi)
        if self.child_index == FEMM:
            self.some_solver_related_operations_rotor_before_mirror_rotation(im, P6, P8) # call this before mirror_and_copyrotate

        if self.child_index == JMAG:
            self.draw_line(P8, P1)
            self.draw_arc(P2, P1, P2_angle)
            self.draw_line(P2, P3)

        # if self.child_index == JAMG:
            self.mirror_and_copyrotate(im.Qr, im.Radius_OuterRotor, fraction,
                                        symmetry_type=2) 

        # if self.child_index == JAMG:
            self.init_sketch_cage()

        # P_Bar
        P_Bar = (-im.Location_RotorBarCenter-im.Radius_of_RotorSlot, 0)
        self.draw_arc(P5, P_Bar, self.get_postive_angle(P5))

        if self.child_index == JMAG:
            self.add_line(P_Bar, P8)
            # draw the outline of stator core for coil to form a region in JMAG
            self.draw_arc(P6, P5, 0.5*pi - self.get_postive_angle(P5, c.centroid.coords[0]))
            self.draw_line(P6, P7)
            self.draw_arc(P8, P7, 0.5*pi)

            self.mirror_and_copyrotate(im.Qr, None, fraction,
                                        symmetry_type=2
                                        # merge=False, # bars are not connected to each other, so you don't have to specify merge=False, they will not merge anyway...
                                        # do_you_have_region_in_the_mirror=True # In short, this should be true if merge is false...
                                        )

        if self.child_index == FEMM:
            self.mirror_and_copyrotate(im.Qr, im.Radius_OuterRotor, fraction)

        # if self.child_index == FEMM:
            self.some_solver_related_operations_fraction(im, fraction)

    def draw_rotor_without_non_accurate_shapely(self, fraction=1):
        # Shapely is very poor in accuracy, use your high school geometry knowledge to derive the coordinates!

        im = self.im

        origin = Point(0,0)
        Stator_Sector_Angle = 2*pi/im.Qs*0.5
        Rotor_Sector_Angle = 2*pi/im.Qr*0.5

        ''' Part: Rotor '''
        if self.child_index == JMAG:
            self.init_sketch_rotorCore()
        # Draw Points as direction of CCW
        # P1
        P1 = (-im.Radius_Shaft, 0)

        # P2
        P2 = ( -im.Radius_Shaft*cos(Rotor_Sector_Angle), im.Radius_Shaft*sin(Rotor_Sector_Angle) )

        # P3
        P2_angle = P3_angle = Rotor_Sector_Angle
        P3 = ( -im.Radius_OuterRotor*cos(Rotor_Sector_Angle), im.Radius_OuterRotor*sin(Rotor_Sector_Angle) )

        # P4
        P4 = (-sqrt(im.Radius_OuterRotor**2 - (0.5*im.Width_RotorSlotOpen)**2), 0.5*im.Width_RotorSlotOpen)
        self.draw_arc(P3, P4, P3_angle - self.get_postive_angle(P4), center=(0,0))

        # P5
        P5 = (-im.Location_RotorBarCenter-sqrt(im.Radius_of_RotorSlot**2 - (0.5*im.Width_RotorSlotOpen)**2), 0.5*im.Width_RotorSlotOpen)
        self.draw_line(P4, P5)

        # P6
        P6 = (-im.Location_RotorBarCenter, im.Radius_of_RotorSlot)
        self.draw_arc(P6, P5, 0.5*pi - self.get_postive_angle(P5, (-im.Location_RotorBarCenter, 0)), center=(-im.Location_RotorBarCenter, 0))

        # P7
        P7 = (-im.Location_RotorBarCenter2, im.Radius_of_RotorSlot2)
        self.draw_line(P6, P7)

        # P8
        P8 = (-im.Location_RotorBarCenter2+im.Radius_of_RotorSlot2, 0)
        self.draw_arc(P8, P7, 0.5*pi, center=(-im.Location_RotorBarCenter2, 0))

        if self.child_index == FEMM:
            self.some_solver_related_operations_rotor_before_mirror_rotation(im, P6, P8) # call this before mirror_and_copyrotate

        if self.child_index == JMAG:
            self.draw_line(P8, P1)
            self.draw_arc(P2, P1, P2_angle)
            self.draw_line(P2, P3)

            self.mirror_and_copyrotate(im.Qr, im.Radius_OuterRotor, fraction,
                                        symmetry_type=2) 
            self.init_sketch_cage()

        if self.child_index == CUSTOM:
            # self.draw_line(P8, P1, ls='-.')
            self.draw_arc(P2, P1, P2_angle)
            # self.draw_line(P2, P3, ls='-.')

        # P_Bar
        P_Bar = (-im.Location_RotorBarCenter-im.Radius_of_RotorSlot, 0)
        self.draw_arc(P5, P_Bar, self.get_postive_angle(P5, (-im.Location_RotorBarCenter, 0)), center=(-im.Location_RotorBarCenter, 0), ls=':')

        if self.child_index == JMAG:
            self.add_line(P_Bar, P8)
            # draw the outline of stator core for coil to form a region in JMAG
            self.draw_arc(P6, P5, 0.5*pi - self.get_postive_angle(P5, (-im.Location_RotorBarCenter, 0)), center=(-im.Location_RotorBarCenter, 0))
            self.draw_arc(P8, P7, 0.5*pi, center=(-im.Location_RotorBarCenter2, 0))

            self.mirror_and_copyrotate(im.Qr, None, fraction,
                                        symmetry_type=2
                                        # merge=False, # bars are not connected to each other, so you don't have to specify merge=False, they will not merge anyway...
                                        # do_you_have_region_in_the_mirror=True # In short, this should be true if merge is false...
                                        )

        if self.child_index == FEMM:
            self.mirror_and_copyrotate(im.Qr, im.Radius_OuterRotor, fraction)
            self.some_solver_related_operations_fraction(im, fraction)

        if self.child_index == CUSTOM:
            # self.draw_line(P8, P_Bar, ls='-.')
            self.mirror_and_copyrotate(im.Qr, im.Radius_OuterRotor, fraction)

            # self.Pr_list = [P1,P2,P3,P4,P5,P6,P7,P8,P_Bar]
            self.Pr_list = [np.array(P) for P in [P1,P2,P3,P4,P5,P6,P7,P8]]
            for P in self.Pr_list:
                self.ax.scatter(*P, c='k')

            # rotor objects
            self.rotor_object_list = self.plot_object_list
            self.plot_object_list = []

    def draw_stator_without_non_accurate_shapely(self, fraction=1):
        im = self.im

        origin = Point(0,0)
        Stator_Sector_Angle = 2*pi/im.Qs*0.5
        Rotor_Sector_Angle = 2*pi/im.Qr*0.5

        ''' Part: Stator '''
        if self.child_index == JMAG:
            self.init_sketch_statorCore()

        # Draw Points as direction of CCW
        # P1
        P1 = (-im.Radius_OuterRotor-im.Length_AirGap, 0)

        # P2
        Radius_InnerStator = im.Radius_OuterRotor+im.Length_AirGap
        P2_angle = im.Angle_StatorSlotOpen*0.5/180*pi
        P2_rot = (-Radius_InnerStator*cos(P2_angle), -Radius_InnerStator*sin(P2_angle))
        P2 = self.park_transform(P2_rot, Stator_Sector_Angle)
        self.draw_arc(P2, P1, self.get_postive_angle(P2))

        # P3
        P3_rot = (P2_rot[0]-im.Width_StatorTeethHeadThickness, P2_rot[1])
        P3 = self.park_transform(P3_rot, Stator_Sector_Angle)
        self.draw_line(P2, P3)

        # P4 (P4 is better to compute using intersection, error from shapely will not really cause a problem while modeling)
        c = self.create_circle(origin, im.Radius_OuterRotor+im.Length_AirGap+im.Width_StatorTeethHeadThickness+im.Width_StatorTeethNeck)
        l = LineString([(0, 0.5*im.Width_StatorTeethBody), (-im.Radius_OuterStatorYoke, 0.5*im.Width_StatorTeethBody)])
        P4 = self.get_node_at_intersection(c,l)
        self.draw_line(P3, P4)

        # P5
        # c = self.create_circle(origin, im.Radius_InnerStatorYoke)
        # P5 = self.get_node_at_intersection(c,l)
        P5 = (-sqrt(im.Radius_InnerStatorYoke**2 - (0.5*im.Width_StatorTeethBody)**2), 0.5*im.Width_StatorTeethBody)
        self.draw_line(P4, P5)

        # P6
        # k = -tan(Stator_Sector_Angle)
        # l_sector = LineString([(0,0), (-im.Radius_OuterStatorYoke, -im.Radius_OuterStatorYoke*k)])
        # P6 = self.get_node_at_intersection(c,l_sector)
        P6 = ( -im.Radius_InnerStatorYoke*cos(Stator_Sector_Angle), im.Radius_InnerStatorYoke*sin(Stator_Sector_Angle) )
        self.draw_arc(P6, P5, Stator_Sector_Angle - self.get_postive_angle(P5))


        # P7
        # c = self.create_circle(origin, im.Radius_OuterStatorYoke)
        # P7 = self.get_node_at_intersection(c,l_sector)
        P7 = [ -im.Radius_OuterStatorYoke*cos(Stator_Sector_Angle), im.Radius_OuterStatorYoke*sin(Stator_Sector_Angle) ]

        # P8
        P8 = (-im.Radius_OuterStatorYoke, 0)

        if self.child_index == JMAG:
            self.draw_line(P6, P7)
            self.draw_arc(P7, P8, Stator_Sector_Angle)
            self.draw_line(P8, P1)
            self.mirror_and_copyrotate(im.Qs, None, fraction,
                                        symmetry_type=2,  # 2: x-axis
                                        )
            self.init_sketch_coil()

        if self.child_index == CUSTOM:
            # self.draw_line(P6, P7, ls='-.')
            self.draw_arc(P7, P8, Stator_Sector_Angle)
            # self.draw_line(P8, P1, ls='-.')

        # P_Coil
        # l = LineString([(P3[0], P3[1]), (P3[0], im.Radius_OuterStatorYoke)])
        # P_Coil = self.get_node_at_intersection(l_sector, l)
        P_Coil = ( P3[0], abs(P3[0])*tan(Stator_Sector_Angle) )




        if self.child_index == JMAG: # F*ck you, Shapely for putting me through this!
            #     temp = 0.7071067811865476*(P_Coil[1] - P4[1])
            #     P4 = [                      P4[0], P4[1] + temp]
            #     P5 = [P5[0] + 0.7071067811865476*(P4[0] - P5[0]), P5[1] + temp]
            #     P6 = []
            # we use sin cos to find P6 and P7, now this suffices.
            P6[1] -= 0.01
            # Conclusion: do not use the intersection between l_sector and a circle!
            # Conclusion: do not use the intersection between l_sector and a circle!
            # Conclusion: do not use the intersection between l_sector and a circle!
            # 总之，如果overlap了，merge一下是没事的，麻烦就在Coil它不能merge，所以上层绕组和下层绕组之间就产生了很多Edge Parts。

        if self.child_index == CUSTOM:
            # self.Ps_list = [P1,P2,P3,P4,P5,P6,P7,P8,P_Coil]
            self.Ps_list = [np.array(P) for P in [P1,P2,P3,P4,P5,P6,P7,P8]]
            for P in self.Ps_list:
                self.ax.scatter(*P, c='k')

            # stator objects
            self.stator_object_list = self.plot_object_list
            self.plot_object_list = []

        else:
            self.draw_line(P4, P_Coil)
            self.draw_line(P6, P_Coil)

        if self.child_index == JMAG:
            # draw the outline of stator core for coil to form a region in JMAG
            self.draw_line(P4, P5)
            self.draw_arc(P6, P5, Stator_Sector_Angle - self.get_postive_angle(P5))
            self.mirror_and_copyrotate(im.Qs, None, fraction,
                                        edge4ref=self.artist_list[1], #'Line.2' 
                                        # symmetry_type=2,
                                        merge=False, # two layers of windings
                                        do_you_have_region_in_the_mirror=True # In short, this should be true if merge is false...
                                        )
            # it is super wierd that use edge Line.2 as symmetry axis will lead to redundant parts imported into JMAG Designers (Extra Coil and Stator Core Parts)
            # symmetry_type=2 will not solve this problem either
            # This is actually caused by the overlap of different regions, because the precision of shapely is shit!

        # FEMM does not model coil and stator core separately        
        if self.child_index == FEMM:
            self.mirror_and_copyrotate(im.Qs, im.Radius_OuterStatorYoke, fraction)

    @staticmethod
    def park_transform(P_rot, angle):
        return (  cos(angle)*P_rot[0] + sin(angle)*P_rot[1],
                 -sin(angle)*P_rot[0] + cos(angle)*P_rot[1] )


    # this method is found to be very low in accuracy, don't use it, just specify the center if know!
    @staticmethod
    def find_center_of_a_circle_using_2_points_and_arc_angle(p1, p2, angle):
        return find_center_and_radius_of_a_circle_using_2_points_and_arc_angle(p1, p2, angle)[0]

    @staticmethod
    def find_center_and_radius_of_a_circle_using_2_points_and_arc_angle(p1, p2, angle):
        distance = sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        radius   = distance*0.5 / sin(angle*0.5)
        c1 = Point(p1[0],p1[1]).buffer(radius).boundary
        c2 = Point(p2[0],p2[1]).buffer(radius).boundary
        i = c1.intersection(c2)
        # print len(i.geoms)
        # print i.geoms[0].coords[0]
        # print i.geoms[1].coords[0]

        # if len(i.geoms) > 2:
        #     print 'There are more than 2 intersections!'
        #     print p1, p2, angle
        #     print c1.coords[0], c1.coords[1], c1.coords[2]
        #     print c2.coords[0], c2.coords[1], c2.coords[2]
        #     for el in i.geoms:
        #         print '\t', el.coords[0]
        #     print 'Crazy Shapely.'
        #     # (0.013059471222689467, 0.46233200000038793)
        #     # (0.013059471222689467, 0.462332)
        #     # (-93.94908807906135, 4.615896979306289)
        #     # raise Exception('Too many intersections.')

        if i.geoms is None:
            raise Exception('There is no intersection.')

        for center in [ el.coords[0] for el in i.geoms ]: # shapely does not return intersections in a known order
            print center
            determinant = (center[0]-p1[0]) * (p2[1]-p1[1]) - (center[1]-p1[1]) * (p2[0]-p1[0])
            if copysign(1, determinant) < 0: # CCW from p1 to p2
                return center, radius

    @staticmethod
    def create_circle(p, radius):
        return p.buffer(radius).boundary

    @staticmethod
    def get_postive_angle(p, origin=(0,0)):
        # using atan loses info about the quadrant, so it is "positive"
        return atan(abs((p[1]-origin[1]) / (p[0]-origin[0])))

    @staticmethod
    def get_node_at_intersection(c,l): # this works for c and l having one intersection only
        i = c.intersection(l)
        # femm.mi_addnode(i.coords[0][0], i.coords[0][1])
        return i.coords[0][0], i.coords[0][1]




def csv_row_reader(handle):
    from csv import reader
    read_iterator = reader(handle, skipinitialspace=True)
    return whole_row_reader(read_iterator)        

def whole_row_reader(reader):
    for row in reader:
        yield row[:]



class VanGogh_Plotter(VanGogh):
    """VanGogh_Plotter is child class of VanGogh for plotting IM geometry in matplotlib."""
    def __init__(self, im, child_index=2):
        super(VanGogh_Plotter, self).__init__(im, child_index)

        self.add_line = self.draw_line
        self.add_arc  = self.draw_arc

        self.fig = figure(figsize=(8, 8), facecolor='w', edgecolor='k')
        self.ax = self.fig.add_subplot(111, aspect='equal')
        # self.ax = self.fig.gca()
        # self.ax.set_xlim([-100,0])
        self.ax.set_ylim([-10,20])

        # plt.gcf().gca().invert_yaxis()

    @staticmethod
    def mirror_and_copyrotate(Q, Radius, fraction):
        # Mirror
        # femm.mi_selectcircle(0,0,Radius+EPS,SELECT_ALL) # this EPS is sometime necessary to selece the arc at Radius.
        # femm.mi_mirror2(0,0,-Radius,0, SELECT_ALL)

        # Rotate
        # femm.mi_selectcircle(0,0,Radius+EPS,SELECT_ALL)
        # femm.mi_copyrotate2(0, 0, 360./Q, int(Q)/fraction, SELECT_ALL)

        return

    def draw_arc(self, p1, p2, angle, center=(0,0), **kwarg):
        # center, radius = self.find_center_and_radius_of_a_circle_using_2_points_and_arc_angle(p1, p2, angle) # ordered p1 and p2 are
        # print sqrt((p1[0]-center[0])**2 + (p1[1]-center[0])**2)
        # print sqrt((p2[0]-center[0])**2 + (p2[1]-center[0])**2)
        distance = sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        radius   = distance*0.5 / sin(angle*0.5)

        obj = self.pyplot_arc(radius, angle_span=angle, rotation=atan2(p1[1]-center[1],p1[0]-center[0]), center=center, **kwarg)
        self.plot_object_list.append(obj[0])
        return obj
        
    def pyplot_arc(self, radius, angle_span=3.14, center=(0,0), rotation=0, maxseg=0.1, ax=None, **kwarg):
        # make sure ax is not empty
        if ax is None:
            ax = self.ax
            # ax = plt.gcf().gca()

        # turn Polar into Cartesian
        def xy(radius, phi, center):
            return radius*np.cos(phi) + center[0], radius*np.sin(phi) + center[1]

        # get list of points
        # phis = np.arange(rotation, rotation+angle_span, 2*radius*np.pi / (360./maxseg))
        phis = np.linspace(rotation, rotation+angle_span, 360./maxseg)
        return ax.plot( *xy(radius, phis, center), c='k', **kwarg)
    
    def draw_line(self, p1, p2, ax=None, **kwarg):
        # make sure ax is not empty
        if ax is None:
            ax = self.ax

        obj = ax.plot( [p1[0], p2[0]], [p1[1], p2[1]], c='k', **kwarg)
        self.plot_object_list.append(obj[0])
        return obj


if __name__ == '__main__':
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    myfontsize = 13.5
    plt.rcParams.update({'font.size': myfontsize})

    from population import *
    im_list = []
    with open(r'D:\OneDrive - UW-Madison\c\pop\initial_design.txt', 'r') as f: 
        for row in csv_row_reader(f):
            im = bearingless_induction_motor_design([row[0]]+[float(el) for el in row[1:]], None)
            im_list.append(im)
    # print im.show(toString=True)

    # 示意图而已，改改尺寸吧
    im.Radius_OuterStatorYoke -= 25
    im.Radius_InnerStatorYoke -= 20
    im.Radius_Shaft += 10

    vg = VanGogh_Plotter(im, CUSTOM)
    vg.draw_model()

    # 
    BIAS = 1
    Stator_Sector_Angle = 2*pi/im.Qs*0.5
    Rotor_Sector_Angle = 2*pi/im.Qr*0.5

    # draw the mirrored slot
    new_objs = []
    for obj in vg.rotor_object_list:
        xy_data = obj.get_xydata().T
        new_obj = vg.ax.plot(xy_data[0], -array(xy_data[1]), 'k--', zorder=0)
        new_objs.append(new_obj[0])
    vg.rotor_object_list += new_objs

    new_objs = []
    for obj in vg.stator_object_list:
        xy_data = obj.get_xydata().T
        new_obj = vg.ax.plot(xy_data[0], -array(xy_data[1]), 'k--', zorder=0)
        new_objs.append(new_obj[0])
    vg.stator_object_list += new_objs

    # draw the rotated slot
    # rotor 
    angle = Rotor_Sector_Angle*2
    TR = array( [ [cos(angle), sin(angle)],
                 [-sin(angle), cos(angle)] ] )
    for obj in vg.rotor_object_list:
        xy_data = obj.get_xydata().T
        xy_data_rot = np.dot( TR, xy_data )
        # xy_data_rot = xy_data_rot.T
        vg.ax.plot(xy_data_rot[0], xy_data_rot[1], 'k--', zorder=0)
    # stator
    angle = Stator_Sector_Angle*2
    TS = array( [ [cos(angle), sin(angle)],
                 [-sin(angle), cos(angle)] ] )
    for obj in vg.stator_object_list:
        xy_data = obj.get_xydata().T
        xy_data_rot = np.dot( TS, xy_data )
        # xy_data_rot = xy_data_rot.T
        vg.ax.plot(xy_data_rot[0], xy_data_rot[1], 'k--', zorder=0)

    ################################################################
    # add labels to geometry
    ################################################################
    # xy = (-0.5*(im.Location_RotorBarCenter+im.Location_RotorBarCenter2), 2.5)
    # vg.ax.annotate('Parallel tooth', xytext=(xy[0],-(xy[1]+5)), xy=(xy[0],-xy[1]), xycoords='data', arrowprops=dict(arrowstyle="->"))

    xy = (-im.Radius_OuterStatorYoke, 0)
    vg.ax.annotate(r'$r_{so}=86.9$ mm', xytext=(xy[0]+2.5,xy[1]-0.25), xy=xy, xycoords='data', arrowprops=dict(arrowstyle="->"))
    xy = (-im.Radius_OuterRotor, -4)
    vg.ax.annotate(r'$r_{ro}=47.1$ mm', xytext=(xy[0]+2.5,xy[1]-0.25), xy=xy, xycoords='data', arrowprops=dict(arrowstyle="->"))

    def add_label_inside(vg, label, PA, PB):
        # vector_AB = (PB[0]-PA[0], PB[1]-PA[1])
        xy, xytext = PA, PB 
        vg.ax.annotate('', xytext=xytext, xy=xy, xycoords='data', arrowprops=dict(arrowstyle="->"))
        xy, xytext = xytext, xy
        vg.ax.annotate('', xytext=xytext, xy=xy, xycoords='data', arrowprops=dict(arrowstyle="->"))
        vg.ax.text(0.5+0.5*(xy[0]+xytext[0]), 0.+0.5*(xy[1]+xytext[1]), label,
                    bbox=dict(facecolor='w', edgecolor='r',boxstyle='round,pad=0',alpha=0.9))


    def add_label_outside(vg, label, PA, PB, ext_factor=1):
        vector_AB = np.array((PB[0]-PA[0], PB[1]-PA[1]))
        xy, xytext = PB, PB + vector_AB * ext_factor
        vg.ax.annotate('', xytext=xytext, xy=xy, xycoords='data', arrowprops=dict(arrowstyle="->"))
        xy, xytext = PA, PA - vector_AB * ext_factor
        vg.ax.annotate('', xytext=xytext, xy=xy, xycoords='data', arrowprops=dict(arrowstyle="->"))
        vg.ax.text(0.5+PB[0], -1.75+PB[1], label,
                    bbox=dict(facecolor='w', edgecolor='r',boxstyle='round,pad=0',alpha=0.9))


    def add_extension_line(vg, P, vector_ext, distance_factor=1):

        distance = sqrt(vector_ext[0]**2 + vector_ext[1]**2)
        vector_ext /= distance/distance_factor # a unit vector * distance_factor

        x = [ P[0], P[0]+vector_ext[0] ]
        y = [ P[1], P[1]+vector_ext[1] ]
        vg.ax.plot(x, y, 'k-', lw=0.6)

        return vector_ext

    #1
    Ps2 = vg.Ps_list[2-BIAS] * array((1,-1)) # mirror
    Ptemp1 = Ps2[0], Ps2[1]-5
    Ptemp1 = vg.park_transform(Ptemp1, -Stator_Sector_Angle) # this is not exact but close enough, you can find the exact point by solve a triangle with Radius_OuterRotor + Length_AirGap from Pr3
    vector_ext = np.array((Ptemp1[0] - Ps2[0], Ptemp1[1] - Ps2[1]))
    vector_ext = add_extension_line(vg, Ps2, vector_ext, distance_factor=2.5)
    Ptemp1 = (Ps2 + 0.5*vector_ext)

    Pr3 = vg.Pr_list[3-BIAS] * array((1,-1)) # mirror
    Ptemp2 = array((Pr3[0], Pr3[1]-5))
    Ptemp2 = vg.park_transform(Ptemp2, -Rotor_Sector_Angle)
    vector_ext = np.array((Ptemp2[0] - Pr3[0], Ptemp2[1] - Pr3[1]))
    vector_ext = add_extension_line(vg, Pr3, vector_ext, distance_factor=2.5)
    Ptemp2 = (Pr3 + 0.5*vector_ext)

    add_label_outside(vg, r'$\delta$', Ptemp1, Ptemp2, ext_factor=2)

    #2
    Ps4 = vg.Ps_list[4-BIAS]
    Ps5 = vg.Ps_list[5-BIAS]
    Ptemp = 0.5*(Ps4 + Ps5)
    add_label_inside(vg, r'$b_{\rm tooth,s}$', (Ptemp[0],-Ptemp[1]), Ptemp)

    #3
    Pr6 = array(vg.Pr_list[6-BIAS])
    Pr7 = array(vg.Pr_list[7-BIAS])
    Ptemp2 = 0.5*(Pr6 + Pr7) # middle point
    Ptemp1 = Ptemp2[0], -Ptemp2[1] # mirror
    Ptemp1 = np.dot( TR, Ptemp1 ) # rotate
    # vector_r67 = Pr7 - Pr6
    # vector_r67_perpendi = (-vector_r67[1], vector_r67[0])
    # some_angle = atan2(vector_r67[1], vector_r67[0])
    add_label_inside(vg, r'$b_{\rm tooth,r}$', Ptemp1, Ptemp2)

    #4 (This one is angle)
    Ps3 = vg.Ps_list[3-BIAS]
    Ptemp1 = Ps3 * np.array((1,-1)) # mirror
    Ptemp2 = np.dot( TS, Ptemp1 ) # rotate
    if False: # treat it as width
        add_label_inside(vg, r'$w_{\rm open,s}$', Ps3, Ptemp2)
    else: # it is an angle, ok.
        Ps2 = vg.Ps_list[2-BIAS]
        vector_ext = Ps3 - Ps2
        vector_ext = add_extension_line(vg, Ps3, vector_ext, distance_factor=3)
        Ptemp1 = (Ps3 + 0.5*vector_ext)

        vector_ext *= array([1,-1]) # mirror
        vector_ext = np.dot( TS, vector_ext ) # rotate
        vector_ext = add_extension_line(vg, Ptemp2, vector_ext, distance_factor=3)
        Ptemp2 = (Ptemp2 + 0.5*vector_ext)

        some_angle = atan2(vector_ext[1], vector_ext[0])
        print some_angle/pi*180, 180 - Stator_Sector_Angle/pi*180
        vg.pyplot_arc(im.Radius_OuterRotor+im.Length_AirGap+im.Width_StatorTeethHeadThickness+2, 
            angle_span=0.95*im.Angle_StatorSlotOpen/180*pi, rotation=some_angle-0.475*im.Angle_StatorSlotOpen/180*pi, center=(0,0), lw=0.6)
        # vg.ax.text(Ptemp1[0], Ptemp1[1], r'$w_{\rm open,s}$',
        #             bbox=dict(facecolor='w', edgecolor='r',boxstyle='round,pad=0',alpha=0.9))
        vg.ax.text(0.5*(Ptemp1[0]+Ptemp2[0])-5, 0.5*(Ptemp1[1]+Ptemp2[1]), r'$w_{\rm open,s}$',
                    bbox=dict(facecolor='w', edgecolor='r',boxstyle='round,pad=0',alpha=0.9))



    #5
    Pr4 = vg.Pr_list[4-BIAS]
    Pr5 = vg.Pr_list[5-BIAS]
    Ptemp1 = 0.5*(Pr4+Pr5)
    Ptemp2 = Ptemp1 * np.array((1,-1)) # mirror
    add_label_outside(vg, r'$w_{\rm open,r}$', Ptemp1, Ptemp2, ext_factor=2)

    #6
    Ps2 = vg.Ps_list[2-BIAS]
    Ps3 = vg.Ps_list[3-BIAS] 
    Ptemp1 = np.dot( TS, Ps2 )
    Ptemp11 = np.dot( TS, Ps3 )
    vector_ext = -np.array((Ptemp11[0] - Ptemp1[0], Ptemp11[1] - Ptemp1[1]))
    vector_ext = np.array((-vector_ext[1], vector_ext[0]))
    vector_ext = add_extension_line(vg, Ptemp1, vector_ext, distance_factor=1)
    Ptemp1 = (Ptemp1 + 0.5*vector_ext)

    Ptemp2 = np.dot( TS, Ps3 )
    Ptemp22 = np.dot( TS, Ps2 )
    vector_ext = np.array((Ptemp22[0] - Ptemp2[0], Ptemp22[1] - Ptemp2[1]))
    vector_ext = np.array((-vector_ext[1], vector_ext[0]))
    vector_ext = add_extension_line(vg, Ptemp2, vector_ext, distance_factor=1)
    Ptemp2 = (Ptemp2 + 0.5*vector_ext)

    add_label_outside(vg, r'$h_{\rm head,s}$', Ptemp1, Ptemp2, ext_factor=3)


    #7
    Pr4 = vg.Pr_list[4-BIAS]
    Ptemp1 = np.dot( TR, Pr4 )
    Ptemp11 = np.array((Pr4[0], Pr4[1]+5))
    Ptemp11 = np.dot( TR, Ptemp11 )
    vector_ext = np.array((Ptemp11[0] - Ptemp1[0], Ptemp11[1] - Ptemp1[1]))
    vector_ext = add_extension_line(vg, Ptemp1, vector_ext, distance_factor=6)
    Ptemp1 = (Ptemp1 + 0.5*vector_ext)

    Pr5 = vg.Pr_list[5-BIAS] 
    Ptemp2 = np.dot( TR, Pr5 )
    Ptemp22 = np.array((Pr5[0], Pr5[1]+5))
    Ptemp22 = np.dot( TR, Ptemp22 )
    vector_ext = np.array((Ptemp22[0] - Ptemp2[0], Ptemp22[1] - Ptemp2[1]))
    vector_ext = add_extension_line(vg, Ptemp2, vector_ext, distance_factor=6)
    Ptemp2 = (Ptemp2 + 0.5*vector_ext)

    add_label_outside(vg, r'$h_{\rm head,r}$', Ptemp1, Ptemp2, ext_factor=3)



    vg.ax.get_xaxis().set_visible(False)
    vg.ax.get_yaxis().set_visible(False)
    vg.fig.tight_layout()
    vg.fig.savefig(r'D:\OneDrive\[00]GetWorking\32 blimopti\p2019_ecce_bearingless_induction_full_paper\images\CAD_CrossSection.png')
    show()
    quit()

    # obsolete
    def perpendi_label(vg, label, PA, PB, distance=3, distance_factor=1, mirror=False):
        if mirror == True:
            PA = PA[0], -PA[1]
            PB = PB[0], -PB[1]
            distance *= -1

        new_PA = PA[0], PA[1]+distance + PB[1]-PA[1]
        new_PB = PB[0], PB[1]+distance 

        x = [ PA[0], new_PA[0] ]
        y = [ PA[1], new_PA[1] ]
        # x = [ PA[0] ]*2
        # y = [ PA[1], PA[1]+distance + PB[1]-PA[1] ]
        vg.ax.plot(x, y, 'k', lw=0.5)

        x = [ PB[0], new_PB[0] ]
        y = [ PB[1], new_PB[1] ]
        # x = [ PB[0] ]*2
        # y = [ PB[1], PB[1]+distance ]
        vg.ax.plot(x, y, 'k', lw=0.5)

        distance = sqrt((PA[0]-new_PA[0])**2 + (PA[1]-new_PA[1])**2)*distance_factor
        xy = array([0.5*(PA[0]+new_PA[0]), 0.5*(PA[1]+new_PA[1])])
        vector = array([new_PA[0]-PA[0], new_PA[1]-PA[1]])
        xytext = array([-vector[1], vector[0]]) / distance # perpendicular vector
        xytext = xy + xytext
        vg.ax.annotate(label, xytext=xytext, xy=xy, xycoords='data', arrowprops=dict(arrowstyle="->"))
    def parallel_label(vg, label, PA, PB, vector_extendion=None, translation=0, rotation=0, distance_factor=2, mirror=False):

        if vector_extendion is None:
            pass
        else:
            if mirror == True:
                PA = PA[0], -PA[1]
                PB = PB[0], -PB[1]
            if translation == 0:
                new_PA = vg.park_transform((PA[0],PA[1]), rotation)
                new_PB = vg.park_transform((PB[0],PB[1]), rotation)
            elif translation == 1:
                new_PA = PA[0], PA[1]+translation
                new_PB = PB[0], PB[1]+translation
            elif translation == -1:
                new_PA = PA[0]+translation, PA[1]
                new_PB = PB[0]+translation, PB[1]

            distance = sqrt((PA[0]-new_PA[0])**2 + (PA[1]-new_PA[1])**2)
            if distance == 0: # rotation == 0
                distance = 1
            unit_vector = array([new_PA[0]-PA[0], new_PA[1]-PA[1]])/distance
            new_PA = array([PA[0] + unit_vector[0] * distance_factor, PA[1] + unit_vector[1] * distance_factor])
            new_PB = array([PB[0] + unit_vector[0] * distance_factor, PB[1] + unit_vector[1] * distance_factor]) 

            # A
            x = [ PA[0], new_PA[0] ]
            y = [ PA[1], new_PA[1] ]
            vg.ax.plot(x, y, 'k-', lw=0.6)

            # B
            x = [ PB[0], new_PB[0] ]
            y = [ PB[1], new_PB[1] ]
            vg.ax.plot(x, y, 'k-', lw=0.6)

            xy = array([0.5*(PA[0]+new_PA[0]), 0.5*(PA[1]+new_PA[1])])
            xytext = distance_factor * array([-unit_vector[1], unit_vector[0]]) # perpendicular vector
            xytext = xy + xytext
            vg.ax.annotate('', xytext=xytext, xy=xy, xycoords='data', arrowprops=dict(arrowstyle="->"))
            vg.ax.text(xytext[0]+0.5, xytext[1]-0.5, label,
                        bbox=dict(facecolor='w', edgecolor='w',boxstyle='round,pad=0',alpha=1))



    # print '---Unit Test of VanGogh.'
    # # >>> c2 = Point(0.5,0).buffer(1).boundary
    # # >>> i = c.intersection(c2)
    # # >>> i.geoms[0].coords[0]
    # # (0.24999999999999994, -0.967031122079967)
    # # >>> i.geoms[1].coords[0]
    # # (0.24999999999999997, 0.967031122079967)

    # vg = VanGogh(None, 1)
    # from numpy import arccos
    # theta = arccos(0.25) * 2
    # print theta / pi * 180, 'deg'
    # print vg.find_center_of_a_circle_using_2_points_and_arc_angle((0.24999999999999994, -0.967031122079967),
    #                                                               (0.24999999999999994, 0.967031122079967),
    #                                                               theta)
    # print vg.find_center_of_a_circle_using_2_points_and_arc_angle((0.24999999999999994, 0.967031122079967),
    #                                                               (0.24999999999999994, -0.967031122079967),
    #                                                               theta)


    # quit()


    # import itertools
    # import matplotlib.patches as patches
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import sys

    # fig, ax = plt.subplots()

    # npoints = 5

    # # Calculate the xy coords for each point on the circle
    # s = 2 * np.pi / npoints
    # verts = np.zeros((npoints, 2))
    # for i in np.arange(npoints):
    #     angle = s * i
    #     x = npoints * np.cos(angle)
    #     y = npoints * np.sin(angle)
    #     verts[i] = [x, y]

    # # Plot the Bezier curves
    # numbers = [i for i in xrange(npoints)]
    # bezier_path = np.arange(0, 1.01, 0.01)
    # for a, b in itertools.product(numbers, repeat=2):
    #     if a == b:
    #         continue

    #     x1y1 = x1, y1 = verts[a]
    #     x2y2 = x2, y2 = verts[b]

    #     xbyb = xb, yb = [0, 0]

    #     # Compute and store the Bezier curve points
    #     x = (1 - bezier_path)** 2 * x1 + 2 * (1 - bezier_path) * bezier_path * xb + bezier_path** 2 * x2
    #     y = (1 - bezier_path)** 2 * y1 + 2 * (1 - bezier_path) * bezier_path * yb + bezier_path** 2 * y2

    #     ax.plot(x, y, 'k-')

    # x, y = verts.T
    # ax.scatter(x, y, marker='o', s=50, c='r')

    # ax.set_xlim(-npoints - 5, npoints + 6)
    # ax.set_ylim(-npoints - 5, npoints + 6)
    # ax.set(aspect=1)
    # plt.show()
    # quit()
