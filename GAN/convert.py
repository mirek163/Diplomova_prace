import bpy
import random

building = bpy.data.objects.get("Building")
window = bpy.data.objects.get("Window")

if building and window:
    building_dims = building.dimensions
    building_location = building.location
    buffer = 0.1
    min_x = building_location.x - (building_dims.x/2) + (window.dimensions.x / 2) + buffer 
    max_x = building_location.x + (building_dims.x/2) - (window.dimensions.x / 2) - buffer 
    min_z = building_location.z - (building_dims.z/2) + (window.dimensions.z / 2) + buffer 
    max_z = building_location.z + (building_dims.z/2) - (window.dimensions.z / 2) - buffer 
    window.location.x = random.uniform(min_x, max_x)
    window.location.z = random.uniform(min_z, max_z)
    window.location.y = building_location.y  

else:
    print("Object not found.")