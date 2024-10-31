import trimesh


def convert_obj_to_ply(obj_file, ply_file):
    # Load the OBJ file
    mesh = trimesh.load(obj_file)

    # Export as PLY
    mesh.export(ply_file)


# Usage
convert_obj_to_ply('output.obj', 'output.ply')
