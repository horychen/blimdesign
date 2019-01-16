# coding:u8
from shapely.geometry import LineString
from shapely.geometry import Point
from math import tan, pi, atan, sqrt, sin, cos, copysign

CUSTOM = 2
JMAG = 1
FEMM = 0

class VanGogh(object):
    """One VanGogh for both FEMM and JMAG"""
    def __init__(self, im, child_index):
        self.im = im
        self.child_index = child_index

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




    @staticmethod
    def find_center_of_a_circle_using_2_points_and_arc_angle(p1, p2, angle):
        distance = sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        radius   = distance*0.5 / sin(angle*0.5)
        c1 = Point(p1[0],p1[1]).buffer(radius).boundary
        c2 = Point(p2[0],p2[1]).buffer(radius).boundary
        i = c1.intersection(c2)
        # print len(i.geoms)
        # print i.geoms[0].coords[0]
        # print i.geoms[1].coords[0]
        if len(i.geoms) > 2:
            for el in i.geoms:
                print el.coords[0]
            print 'Crazy Shapely.'
            # (0.013059471222689467, 0.46233200000038793)
            # (0.013059471222689467, 0.462332)
            # (-93.94908807906135, 4.615896979306289)
            # raise Exception('Too many intersections.')
        elif i.geoms is None:
            raise Exception('There is no intersection.')
        for center in [ el.coords[0] for el in i.geoms ]: # shapely does not return intersections in a known order
            determinant = (center[0]-p1[0]) * (p2[1]-p1[1]) - (center[1]-p1[1]) * (p2[0]-p1[0])
            if copysign(1, determinant) < 0: # CCW from p1 to p2
                return center

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

if __name__ == '__main__':
    print '---Unit Test of VanGogh.'

    # >>> c2 = Point(0.5,0).buffer(1).boundary
    # >>> i = c.intersection(c2)
    # >>> i.geoms[0].coords[0]
    # (0.24999999999999994, -0.967031122079967)
    # >>> i.geoms[1].coords[0]
    # (0.24999999999999997, 0.967031122079967)

    vg = VanGogh(None, 1)
    from numpy import arccos
    theta = arccos(0.25) * 2
    print theta / pi * 180, 'deg'
    print vg.find_center_of_a_circle_using_2_points_and_arc_angle((0.24999999999999994, -0.967031122079967),
                                                                  (0.24999999999999994, 0.967031122079967),
                                                                  theta)
    print vg.find_center_of_a_circle_using_2_points_and_arc_angle((0.24999999999999994, 0.967031122079967),
                                                                  (0.24999999999999994, -0.967031122079967),
                                                                  theta)
