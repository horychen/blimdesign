from population import VanGogh_JMAG

class VanGogh_JMAG_Designer(VanGogh):
    def __init__(self, motor_template, child_index=1, doNotRotateCopy=False):
        super(VanGogh, self).__init__(motor_template, child_index)
        self.doNotRotateCopy = doNotRotateCopy

        self.SketchName = None
        self.dict_count_arc = {}
        self.dict_count_region = {}

        self.count = 0 # counter of region

    def mirror_and_copyrotate(self, Q, Radius, fraction, 
                                edge4ref=None,
                                symmetry_type=None,
                                merge=True,
                                do_you_have_region_in_the_mirror=False
                                ):

        region = self.create_region([art.GetName() for art in self.artist_list]) 

        self.region_mirror_copy(region, edge4ref=edge4ref, symmetry_type=symmetry_type, merge=merge)
        self.count+=1
        # if self.count == 4: # debug
            # raise Exception
            # merge = True # When overlap occurs between regions because of copying, a boolean operation (sum) is executed and they are merged into 1 region.
        if not self.doNotRotateCopy:
            self.region_circular_pattern_360_origin(region, float(Q), merge=merge,
                                                    do_you_have_region_in_the_mirror=do_you_have_region_in_the_mirror)
        # print self.artist_list
        self.sketch.CloseSketch()

    def draw_arc(self, center, p1, p2, **kwarg):
        art = self.sketch.CreateArc(center[0], center[1], p1[0], p1[1], p2[0], p2[1])
        self.artist_list.append(art)

    def draw_line(self, p1, p2):
        # return self.line(p1[0],p1[1],p2[0],p2[1])
        art = self.sketch.CreateLine(p1[0],p1[1],p2[0],p2[1])
        self.artist_list.append(art)

    def add_line(self, p1, p2):
        draw_line(p1, p2)

    def plot_sketch_shaft(self):
        self.SketchName = "Shaft"
        SketchName = self.SketchName
        sketch = self.create_sketch(SketchName, "#D1B894")

        if self.doNotRotateCopy:
            Rotor_Sector_Angle = 2*pi/self.im.Qr*0.5
            PA = [ self.im.Radius_Shaft*-cos(Rotor_Sector_Angle), self.im.Radius_Shaft*sin(Rotor_Sector_Angle) ]
            PB = [ PA[0], -PA[1] ]
            self.draw_arc([0,0], PA, PB)
            self.draw_line(PA, [0,0])
            self.draw_line(PB, [0,0])
            self.doc.GetSelection().Clear()
            self.doc.GetSelection().Add(sketch.GetItem("Arc"))
            self.doc.GetSelection().Add(sketch.GetItem("Line"))
            self.doc.GetSelection().Add(sketch.GetItem("Line.2"))
        else:
            self.circle(0, 0, self.im.Radius_Shaft)

            self.doc.GetSelection().Clear()
            self.doc.GetSelection().Add(sketch.GetItem("Circle"))
        sketch.CreateRegions()

        sketch.CloseSketch()
        # sketch.SetProperty(u"Visible", 0)

    def init_sketch_statorCore(self):
        self.SketchName="Stator Core"
        sketch = self.create_sketch(self.SketchName, "#E8B5CE")
        return 

    def init_sketch_coil(self):
        self.SketchName="Coil"
        sketch = self.create_sketch(self.SketchName, "#EC9787")
        return

    def init_sketch_rotorCore(self):
        self.SketchName="Rotor Core"
        sketch = self.create_sketch(self.SketchName, "#FE840E")
        return

    def init_sketch_cage(self):
        self.SketchName="Cage"
        sketch = self.create_sketch(self.SketchName, "#8D9440")
        return


    # obsolete
    def draw_arc_using_shapely(self, p1, p2, angle, maxseg=1): # angle in rad
        center = self.find_center_of_a_circle_using_2_points_and_arc_angle(p1, p2, angle) # ordered p1 and p2 are
        art = self.sketch.CreateArc(center[0], center[1], p1[0], p1[1], p2[0], p2[1])
        self.artist_list.append(art)

    def add_arc_using_shapely(self, p1, p2, angle, maxseg=1): # angle in rad
        self.draw_arc_using_shapely(p1, p2, angle, maxseg)


    # Utility wrap function for JMAG
    def create_sketch(self, SketchName, color):
        self.artist_list = []

        try:self.dict_count_arc[SketchName]
        except: self.dict_count_arc[SketchName] = 0
        try:self.dict_count_region[SketchName]
        except: self.dict_count_region[SketchName] = 0
        ref1 = self.ass.GetItem("XY Plane")
        ref2 = self.doc.CreateReferenceFromItem(ref1)
        self.sketch = self.ass.CreateSketch(ref2)
        self.sketch.OpenSketch()
        self.sketch.SetProperty("Name", SketchName)
        self.sketch.SetProperty("Color", color)
        return self.sketch
    def circle(self, x,y,r):
        # SketchName = self.SketchName
        self.sketch.CreateVertex(x, y)
        # return self.circle(x, y, r)
        return self.sketch.CreateCircle(x, y, r)
    def line(self, x1,y1,x2,y2):
        # SketchName = self.SketchName
        self.sketch.CreateVertex(x1,y1)
        self.sketch.CreateVertex(x2,y2)
        # return self.line(x1,y1,x2,y2)
        return self.sketch.CreateLine(x1,y1,x2,y2)
    def create_region(self, l):
        SketchName = self.SketchName
        self.doc.GetSelection().Clear()
        for art_name in l:
            self.doc.GetSelection().Add(self.sketch.GetItem(art_name))
            # self.doc.GetSelection().Add(el)
        self.sketch.CreateRegions() # this returns None
        # self.sketch.CreateRegionsWithCleanup(0.05, True) # mm. difference at stator outter radius is up to 0.09mm! This turns out to be neccessary for shapely to work with JMAG. Shapely has poor 

        self.dict_count_region[SketchName] += 1
        if self.dict_count_region[SketchName]==1:
            return self.sketch.GetItem("Region")
        else:
            return self.sketch.GetItem("Region.%d"%(self.dict_count_region[SketchName]))
    def region_mirror_copy(self, region, edge4ref=None, symmetry_type=None, merge=True):
        mirror = self.sketch.CreateRegionMirrorCopy()
        mirror.SetProperty("Merge", merge)
        ref2 = self.doc.CreateReferenceFromItem(region)
        mirror.SetPropertyByReference("Region", ref2)

        # å¯¹ç§°è½´
        if edge4ref == None:
            if symmetry_type == None:
                print("At least give one of edge4ref and symmetry_type")
                raise Exception
            else:
                mirror.SetProperty("SymmetryType", symmetry_type)
        else:
            ref1 = self.sketch.GetItem(edge4ref.GetName()) # e.g., u"Line"
            ref2 = self.doc.CreateReferenceFromItem(ref1)
            mirror.SetPropertyByReference("Symmetry", ref2)

        # print region
        # print ass.GetItem(u"Region.1")
        if merge == False and region.GetName()=="Region":
            return self.ass.GetItem("Region.1")
    def region_circular_pattern_360_origin(self, region, Q_float, merge=True, do_you_have_region_in_the_mirror=False):
        circular_pattern = self.sketch.CreateRegionCircularPattern()
        circular_pattern.SetProperty("Merge", merge)

        ref2 = self.doc.CreateReferenceFromItem(region)
        circular_pattern.SetPropertyByReference("Region", ref2)
        face_region_string = circular_pattern.GetProperty("Region")
        face_region_string = face_region_string[0]
        # print circular_pattern.GetProperty("Region") # this will produce faceRegion references!

        if do_you_have_region_in_the_mirror == True:
            # è¿™é‡Œå‡è®¾face_region_stringæœ€åŽä¸¤ä½æ˜¯æ•°å­—
            if face_region_string[-7:-3] == 'Item':
                number_plus_1 = str(int(face_region_string[-3:-1]) + 1)
                refarray = [0 for i in range(2)]
                refarray[0] = "faceregion(TRegionMirrorPattern%s+%s_2)" % (number_plus_1, face_region_string)
                refarray[1] = face_region_string
                circular_pattern.SetProperty("Region", refarray)
                # print refarray[0]
                # print refarray[1]
            elif face_region_string[-6:-2] == 'Item':
                # è¿™é‡Œå‡è®¾face_region_stringæœ€åŽä¸€ä½æ˜¯æ•°å­—
                number_plus_1 = str(int(face_region_string[-2:-1]) + 1)
                refarray = [0 for i in range(2)]
                refarray[0] = "faceregion(TRegionMirrorPattern%s+%s_2)" % (number_plus_1, face_region_string)
                refarray[1] = face_region_string
                circular_pattern.SetProperty("Region", refarray)
            elif face_region_string[-8:-4] == 'Item':
                # è¿™é‡Œå‡è®¾face_region_stringæœ€åŽä¸‰ä½æ˜¯æ•°å­—
                number_plus_1 = str(int(face_region_string[-4:-1]) + 1)
                refarray = [0 for i in range(2)]
                refarray[0] = "faceregion(TRegionMirrorPattern%s+%s_2)" % (number_plus_1, face_region_string)
                refarray[1] = face_region_string
                circular_pattern.SetProperty("Region", refarray)



        circular_pattern.SetProperty("CenterType", 2)
        circular_pattern.SetProperty("Angle", "360/%d"%(int(Q_float)))
        circular_pattern.SetProperty("Instance", int(Q_float))

class VanGogh_BPMSM1(VanGogh_JMAG):
    def __init__(self, im, child_index=1, doNotRotateCopy=False):
        super(VanGogh_JMAG, self).__init__(im, child_index)
        self.doNotRotateCopy = doNotRotateCopy

        self.SketchName = None
        self.dict_count_arc = {}
        self.dict_count_region = {}

        self.count = 0 # counter of region

    def mirror_and_copyrotate(self, Q, Radius, fraction, 
                                edge4ref=None,
                                symmetry_type=None,
                                merge=True,
                                do_you_have_region_in_the_mirror=False
                                ):

        region = self.create_region([art.GetName() for art in self.artist_list]) 

        self.region_mirror_copy(region, edge4ref=edge4ref, symmetry_type=symmetry_type, merge=merge)
        self.count+=1
        # if self.count == 4: # debug
            # raise Exception
            # merge = True # When overlap occurs between regions because of copying, a boolean operation (sum) is executed and they are merged into 1 region.
        if not self.doNotRotateCopy:
            self.region_circular_pattern_360_origin(region, float(Q), merge=merge,
                                                    do_you_have_region_in_the_mirror=do_you_have_region_in_the_mirror)
        # print self.artist_list
        self.sketch.CloseSketch()

    def draw_arc(self, center, p1, p2, **kwarg):
        art = self.sketch.CreateArc(center[0], center[1], p1[0], p1[1], p2[0], p2[1])
        self.artist_list.append(art)

    def draw_line(self, p1, p2):
        # return self.line(p1[0],p1[1],p2[0],p2[1])
        art = self.sketch.CreateLine(p1[0],p1[1],p2[0],p2[1])
        self.artist_list.append(art)

    def add_line(self, p1, p2):
        draw_line(p1, p2)

    def plot_sketch_shaft(self):
        self.SketchName = "Shaft"
        SketchName = self.SketchName
        sketch = self.create_sketch(SketchName, "#D1B894")

        if self.doNotRotateCopy:
            Rotor_Sector_Angle = 2*pi/self.im.Qr*0.5
            PA = [ self.im.Radius_Shaft*-cos(Rotor_Sector_Angle), self.im.Radius_Shaft*sin(Rotor_Sector_Angle) ]
            PB = [ PA[0], -PA[1] ]
            self.draw_arc([0,0], PA, PB)
            self.draw_line(PA, [0,0])
            self.draw_line(PB, [0,0])
            self.doc.GetSelection().Clear()
            self.doc.GetSelection().Add(sketch.GetItem("Arc"))
            self.doc.GetSelection().Add(sketch.GetItem("Line"))
            self.doc.GetSelection().Add(sketch.GetItem("Line.2"))
        else:
            self.circle(0, 0, self.im.Radius_Shaft)

            self.doc.GetSelection().Clear()
            self.doc.GetSelection().Add(sketch.GetItem("Circle"))
        sketch.CreateRegions()

        sketch.CloseSketch()
        # sketch.SetProperty(u"Visible", 0)

    def init_sketch_statorCore(self):
        self.SketchName="Stator Core"
        sketch = self.create_sketch(self.SketchName, "#E8B5CE")
        return 

    def init_sketch_coil(self):
        self.SketchName="Coil"
        sketch = self.create_sketch(self.SketchName, "#EC9787")
        return

    def init_sketch_rotorCore(self):
        self.SketchName="Rotor Core"
        sketch = self.create_sketch(self.SketchName, "#FE840E")
        return

    def init_sketch_magnet(self):
        self.SketchName="Magnet"
        sketch = self.create_sketch(self.SketchName, "#8D9440")
        return






# if __name__ == '__main__':
#     import matplotlib.patches as mpatches
#     import matplotlib.pyplot as plt
#     plt.rcParams["font.family"] = "Times New Roman"

#     myfontsize = 13.5
#     plt.rcParams.update({'font.size': myfontsize})

#     im_list = []
#     with open(r'D:\OneDrive - UW-Madison\c\pop\initial_design.txt', 'r') as f: 
#         for row in csv_row_reader(f):
#             # fea_config_dict = {}
#             # fea_config_dict['DPNV'] = True
#             # fea_config_dict['flag_optimization'] = False
#             # fea_config_dict['End_Ring_Resistance'] = 0.0
#             im = bearingless_induction_motor_design([row[0]]+[float(el) for el in row[1:]], None)
#             im_list.append(im)
#     # print im.show(toString=True)

#     # 示意图而已，改改尺寸吧
#     im.Radius_OuterStatorYoke -= 37
#     im.Radius_InnerStatorYoke -= 20
#     im.Radius_Shaft += 20
#     # im.Location_RotorBarCenter2 += 5 # this will change the shape of rotor slot

#     vg = VanGogh_pyPlotter(im, CUSTOM)
#     vg.draw_model()

#     # PyX
#     vg.tikz.c = pyx.canvas.canvas() # clear the canvas because we want to redraw 90 deg with the data vg.tikz.track_path
#     from copy import deepcopy
#     def pyx_draw_path(vg, path, sign=1):
#         if len(path) == 4:
#             vg.tikz.draw_line(path[:2], path[2:4], untrack=True)
#         else:
#             vg.tikz.draw_arc(path[:2], path[2:4], path[4:6], relangle=sign*path[6], untrack=True)
#     def rotate(_, x, y):
#         return cos(_)*x + sin(_)*y, -sin(_)*x + cos(_)*y
#     def is_at_stator(im, path):
#         return sqrt(path[0]**2 + path[1]**2) > im.Radius_OuterRotor + 0.5*im.Length_AirGap
#     # 整体转动90度。
#     for path in vg.tikz.track_path:
#         path[0], path[1] = rotate(0.5*pi, path[0], path[1])
#         path[2], path[3] = rotate(0.5*pi, path[2], path[3])
#         pyx_draw_path(vg, path)
#     track_path_backup = deepcopy(vg.tikz.track_path)

#     # Rotate
#     for path in deepcopy(vg.tikz.track_path):
#         if is_at_stator(im, path):
#             Q = im.Qs
#         else:
#             Q = im.Qr
#         _ = 2*pi/Q
#         path[0], path[1] = rotate(_, path[0], path[1])
#         path[2], path[3] = rotate(_, path[2], path[3])
#         pyx_draw_path(vg, path)

#     # Mirror
#     for path in (vg.tikz.track_path): # track_path is passed by reference and is changed by mirror
#         path[0] *= -1
#         path[2] *= -1
#         pyx_draw_path(vg, path, sign=-1)
#     for path in (vg.tikz.track_path):
#         if sqrt(path[0]**2 + path[1]**2) > im.Radius_OuterRotor + 0.5*im.Length_AirGap:
#             Q = im.Qs
#         else:
#             Q = im.Qr
#         _ = 2*pi/Q
#         path[0], path[1] = rotate(_, path[0], path[1])
#         path[2], path[3] = rotate(_, path[2], path[3])
#         pyx

