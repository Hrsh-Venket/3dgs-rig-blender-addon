import bpy
import numpy as np
# Adjust this function to bind to the proxy mesh and move accordingly


def check_mesh_has_gaussian_attributes(mesh_obj):
    """Check if mesh object has basic gaussian attributes"""
    if not mesh_obj or not mesh_obj.data:
        return False
    # Check for basic gaussian attributes
    required_attrs = ['f_dc_0', 'f_dc_1', 'f_dc_2']
    available_attrs = [attr.name for attr in mesh_obj.data.attributes]
    return all(attr in available_attrs for attr in required_attrs)

def extract_attribute_data(mesh_data, attr_name):
    """Extract data from mesh attribute by name - optimized version"""
    if attr_name not in [attr.name for attr in mesh_data.attributes]:
        return None
    attr = mesh_data.attributes[attr_name]
    # Use foreach_get for much faster extraction
    data_array = np.zeros(len(attr.data), dtype=np.float32)
    attr.data.foreach_get("value", data_array)
    return data_array

def extract_gaussian_data_from_evaluated_mesh(mesh_obj):
    """Extract and process gaussian data from EVALUATED mesh object attributes"""
    # Get evaluated mesh data
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_object = mesh_obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated_object.data
    # Extract positions from evaluated vertices - optimized version
    num_points = len(evaluated_mesh.vertices)
    if num_points == 0:
        raise ValueError("Evaluated mesh has no vertices")
    # Use foreach_get for fast vertex coordinate extraction
    positions = np.zeros(num_points * 3, dtype=np.float32)
    evaluated_mesh.vertices.foreach_get("co", positions)
    positions = positions.reshape(-1, 3)
    # Get available attributes from evaluated mesh
    available_attrs = [attr.name for attr in evaluated_mesh.attributes]
    # Extract spherical harmonics from evaluated mesh
    if all(attr in available_attrs for attr in ['f_dc_0', 'f_dc_1', 'f_dc_2']):
        dc_0 = extract_attribute_data(evaluated_mesh, 'f_dc_0')
        dc_1 = extract_attribute_data(evaluated_mesh, 'f_dc_1')
        dc_2 = extract_attribute_data(evaluated_mesh, 'f_dc_2')
        features_dc = np.column_stack([dc_0, dc_1, dc_2])
        # Find f_rest fields
        f_rest_fields = [attr for attr in available_attrs if attr.startswith('f_rest_')]
        f_rest_fields = sorted(f_rest_fields, key=lambda x: int(x.split('_')[-1]))
        if f_rest_fields:
            features_extra_list = []
            for field in f_rest_fields:
                data = extract_attribute_data(evaluated_mesh, field)
                if data is not None:
                    features_extra_list.append(data)
            if features_extra_list:
                features_extra = np.column_stack(features_extra_list)
                num_f_rest = len(f_rest_fields)
                # Determine degree and coefficients to use
                if num_f_rest >= 45:
                    actual_degree = 3
                    coeffs_to_use = 45
                elif num_f_rest >= 24:
                    actual_degree = 2  
                    coeffs_to_use = 24
                elif num_f_rest >= 9:
                    actual_degree = 1
                    coeffs_to_use = 9
                else:
                    actual_degree = 0
                    coeffs_to_use = 0
                if coeffs_to_use > 0:
                    features_extra_used = features_extra[:, :coeffs_to_use]
                    coeffs_per_degree = (actual_degree + 1) ** 2 - 1
                    features_extra_reshaped = features_extra_used.reshape((num_points, 3, coeffs_per_degree))
                    features_extra_reshaped = np.transpose(features_extra_reshaped, [0, 2, 1])
                    features_dc_reshaped = features_dc.reshape(-1, 1, 3)
                    all_features = np.concatenate([features_dc_reshaped, features_extra_reshaped], axis=1)
                    sh_coeffs = all_features.reshape(num_points, -1)
                else:
                    sh_coeffs = features_dc
            else:
                sh_coeffs = features_dc
        else:
            sh_coeffs = features_dc
    else:
        # Default SH coeffs if not found
        print(f"Warning: f_dc attributes not found on evaluated mesh, using defaults")
        sh_coeffs = np.ones((num_points, 3)) * 0.28209479177387814
    # Extract scales from evaluated mesh
    if all(attr in available_attrs for attr in ['scale_0', 'scale_1', 'scale_2']):
        scale_0 = extract_attribute_data(evaluated_mesh, 'scale_0')
        scale_1 = extract_attribute_data(evaluated_mesh, 'scale_1')
        scale_2 = extract_attribute_data(evaluated_mesh, 'scale_2')
        scales = np.column_stack([scale_0, scale_1, scale_2])
        scales = np.exp(scales)  # Apply exponential
    else:
        print(f"Warning: scale attributes not found on evaluated mesh, using defaults")
        scales = np.ones((num_points, 3)) * 0.01
    # Extract rotations from evaluated mesh
    if all(attr in available_attrs for attr in ['rot_0', 'rot_1', 'rot_2', 'rot_3']):
        rot_0 = extract_attribute_data(evaluated_mesh, 'rot_0')
        rot_1 = extract_attribute_data(evaluated_mesh, 'rot_1')
        rot_2 = extract_attribute_data(evaluated_mesh, 'rot_2')
        rot_3 = extract_attribute_data(evaluated_mesh, 'rot_3')
        rotations = np.column_stack([rot_0, rot_1, rot_2, rot_3])
        # Normalize quaternions
        norms = np.linalg.norm(rotations, axis=1, keepdims=True)
        rotations = rotations / norms

        # --- Armature based roation of gaussians
        armature_obj = None
        for mod in mesh_obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object:
                armature_obj = mod.object
                break
        if armature_obj and len(mesh_obj.vertex_groups) > 0:
            n_groups = len(mesh_obj.vertex_groups)

            # Build weight matrix 
            # [N_vertices x N_bones] from vertex groups
            weight_matrix = np.zeros((num_points, n_groups), dtype=np.float32)
            for vert in mesh_obj.data.vertices:
                for g in vert.groups:
                    weight_matrix[vert.index, g.group] = g.weight
            # Get each bone's deformation quaternion (rest pose -> current pose)
            bone_quats = np.zeros((n_groups, 4), dtype=np.float32)
            bone_quats[:, 0] = 1.0 # Identity quaternion
            for vg in mesh_obj.vertex_groups:
                if vg.name in armature_obj.pose.bones:
                    pb = armature_obj.pose.bones[vg.name]
                    deform = pb.matrix @ pb.bone.matrix_local.inverted()
                    q = deform.to_quaternion()
                    bone_quats[vg.index] = [q.w, q.x, q.y, q.z]
            # Linear blend (using nlerp)
            # [N x n_groups] @ [n_groups x 4] -> [N x 4]
            blended = weight_matrix @ bone_quats
            norms = np.linalg.norm(blended, axis=1, keepdims=True)
            rigged = norms.flatten() > 1e-6
            blended[~rigged] = [1, 0, 0, 0]
            norms[~rigged] = 1.0
            blended /= norms

            # Quaternion Multiply:
            # bone rotation * gaussian rotation (vectorised)
            q1, q2 = blended, rotations
            rotations = np.column_stack([
                q1[:,0]*q2[:,0] - q1[:,1]*q2[:,1] - q1[:,2]*q2[:,2] - q1[:,3]*q2[:,3],
                q1[:,0]*q2[:,1] + q1[:,1]*q2[:,0] + q1[:,2]*q2[:,3] - q1[:,3]*q2[:,2],
                q1[:,0]*q2[:,2] - q1[:,1]*q2[:,3] + q1[:,2]*q2[:,0] + q1[:,3]*q2[:,1],
                q1[:,0]*q2[:,3] + q1[:,1]*q2[:,2] - q1[:,2]*q2[:,1] + q1[:,3]*q2[:,0],
            ])
            rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    else:
        print(f"Warning: rotation attributes not found on evaluated mesh, using defaults")
        rotations = np.zeros((num_points, 4))
        rotations[:, 0] = 1.0  # Identity quaternion
    # Extract opacity from evaluated mesh
    if 'opacity' in available_attrs:
        opacity_raw = extract_attribute_data(evaluated_mesh, 'opacity')
        opacity = 1.0 / (1.0 + np.exp(-opacity_raw))  # Apply sigmoid
    else:
        print(f"Warning: opacity attribute not found on evaluated mesh, using defaults")
        opacity = np.ones(num_points)
    return {
        'num_points': num_points,
        'positions': positions,
        'scales': scales,
        'rotations': rotations,
        'opacities': opacity,
        'sh_coeffs': sh_coeffs,
        'sh_dim': sh_coeffs.shape[1]
    }
