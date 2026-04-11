from .extract_gaussian_from_evaluated_mesh import *

def sna_b2_load_from_blender_object_F0CCB(OBJECT_BASE_NAME):
    OBJECT_BASE_NAME = OBJECT_BASE_NAME
    # ========== VARIABLES (EDIT THESE) ==========
    SOURCE_MESH_OBJECT = None  # Set this to target mesh object, or leave None to use active object
    #OBJECT_BASE_NAME = "GaussianSplat"  # Will auto-number: _001, _002, etc.
    # ============================================
    import numpy as np
    from math import pi
    import bpy
    import time

    def get_unique_object_name(base_name):
        """Generate unique object name with auto-numbering"""
        if base_name not in bpy.data.objects:
            return base_name
        counter = 1
        while f"{base_name}_{counter:03d}" in bpy.data.objects:
            counter += 1
        return f"{base_name}_{counter:03d}"

    try:
        # determine source mesh object
        if SOURCE_MESH_OBJECT is not None:
            source_obj = SOURCE_MESH_OBJECT
        else:
            source_obj = bpy.context.active_object
        if not source_obj:
            raise ValueError("no source mesh object specified and no active object")
        if source_obj.type != 'MESH':
            raise ValueError(f"object '{source_obj.name}' is not a mesh object")
        # check if mesh has gaussian attributes (check original mesh, not evaluated)
        if not check_mesh_has_gaussian_attributes(source_obj):
            raise ValueError(f"mesh object '{source_obj.name}' does not have required gaussian attributes (f_dc_0, f_dc_1, f_dc_2)")
        print(f"extracting gaussian data from evaluated mesh: {source_obj.name}")
        # generate or get uuid for source mesh
        import uuid
        if "gaussian_source_uuid" not in source_obj:
            source_obj["gaussian_source_uuid"] = str(uuid.uuid4())
        source_uuid = source_obj["gaussian_source_uuid"]
        # extract gaussian data from evaluated mesh
        gaussian_data_info = extract_gaussian_data_from_evaluated_mesh(source_obj)
        # create gaussian data array (59 floats per gaussian)
        num_gaussians = gaussian_data_info['num_points']
        sh_dim = 48
        total_dim = 3 + 4 + 3 + 1 + sh_dim
        gaussian_data = np.zeros((num_gaussians, total_dim), dtype=np.float32)
        # pack data in original order
        gaussian_data[:, 0:3] = gaussian_data_info['positions']
        gaussian_data[:, 3:7] = gaussian_data_info['rotations']
        gaussian_data[:, 7:10] = gaussian_data_info['scales']
        gaussian_data[:, 10] = gaussian_data_info['opacities'].flatten()
        # handle sh coefficients
        source_sh_coeffs = gaussian_data_info['sh_coeffs']
        if source_sh_coeffs.shape[1] >= sh_dim:
            gaussian_data[:, 11:11+sh_dim] = source_sh_coeffs[:, :sh_dim]
        else:
            gaussian_data[:, 11:11+source_sh_coeffs.shape[1]] = source_sh_coeffs

        # generate unique object name
        object_name = get_unique_object_name(OBJECT_BASE_NAME)

        # create blender empty object
        empty_object = bpy.data.objects.new(object_name, None)
        empty_object.empty_display_type = 'PLAIN_AXES'
        empty_object.empty_display_size = 0.1
        empty_object.matrix_world = source_obj.matrix_world.copy()  # match source object transform
        # Store data in object properties
        empty_object["gaussian_data"] = gaussian_data.tobytes()
        empty_object["gaussian_count"] = num_gaussians
        empty_object["sh_degree"] = gaussian_data_info['sh_dim']
        empty_object["is_gaussian_splat"] = True
        empty_object["is_mesh_source"] = True
        empty_object["is_evaluated_mesh"] = True  # Mark as using evaluated mesh
        empty_object["source_mesh_uuid"] = source_uuid  # Store UUID instead of name
        empty_object["source_mesh_name"] = source_obj.name  # Store name for reference/debugging
        empty_object["is_loaded"] = True
        empty_object["last_load_time"] = time.time()
        # Link to scene
        bpy.context.collection.objects.link(empty_object)
        # Initialize global cache if needed
        if not hasattr(bpy, 'gaussian_object_cache'):
            bpy.gaussian_object_cache = {}
        # Add to global cache
        bpy.gaussian_object_cache[object_name] = {
            'gaussian_data': gaussian_data,
            'gaussian_count': num_gaussians,
            'sh_degree': gaussian_data_info['sh_dim'],
            'object': empty_object,
            'source_mesh_uuid': source_uuid,
            'source_mesh_name': source_obj.name  # Keep name for reference
        }
        # Mark that global textures need rebuilding
        bpy.gaussian_global_needs_update = True
        total_objects = len(bpy.gaussian_object_cache)
        total_gaussians = sum(obj['gaussian_count'] for obj in bpy.gaussian_object_cache.values())
        print(f"Loaded {object_name}: {num_gaussians:,} gaussians from EVALUATED mesh '{source_obj.name}' (SH degree {gaussian_data_info['sh_dim']})")
        print(f"Total: {total_objects} objects, {total_gaussians:,} gaussians")
    except Exception as e:
        print(f"Error extracting from evaluated mesh: {e}")
        import traceback
        traceback.print_exc()
