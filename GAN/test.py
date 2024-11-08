#import pymesh2
#mesh = pymesh2.load_mesh("C:\\Users\\Lenovo\\Desktop\\Diplomova_prace\\blender\\object\\small_buildingA\\output\\rotation\\variant_1.obj")
import processing_obj as prc

#Problem.
object = prc.obj_to_voxel("C:\\Users\\Lenovo\\Desktop\\Diplomova_prace\\blender\\object\\small_buildingA\\output\\window_move\\variant_1.obj")
prc.voxel_to_obj(object,"test.obj")