import bpy
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from .extract_gaussian_from_evaluated_mesh import check_mesh_has_gaussian_attributes
# from .sna_c2_refresh_all import sna_c2_refresh_all_4D367
from .sna_texture_creation import sna_texture_creation_FD1B2

def barycentric_coords(p, a, b, c):
    """Barycentric coordinates of point p in triangle a,b,c."""
    v0=b-a
    v1=c-a
    v2=p-a
    d00=v0.dot(v0)
    d01=v0.dot(v1)
    d11=v1.dot(v1)
    d20=v2.dot(v0)
    d21=v2.dot(v1)
    denom=d00*d11-d01*d01
    if abs(denom)<1e-10:
        return 1/3, 1/3, 1/3
    v = (d11*d20-d01*d21)/denom
    w = (d00*d21-d01*d20)/denom
    u = 1.0 - v - w
    return u, v, w

def tbn_to_quaternion(tangent, bitangent, normal):
    mat = Matrix((tangent, bitangent, normal)).transposed()
    return mat.to_quaternion()

class Bind_Gaussian_Splat_To_Proxy_Mesh(bpy.types.Operator):
    bl_idname="sna.bind_gaussian_splat_to_proxy_mesh"
    bl_label="3DGS Render: Bind Gaussians to Proxy Mesh"
    bl_description="Bind each gaussian to the nearest face on the selected proxy mesh"
    bl_options={"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if len(context.selected_objects) != 2:
            cls.poll_message_set(f"Select exactly 2 objects (have {len(context.selected_objects)})")
            return False
        if not context.active_object:
            cls.poll_message_set("No active object")
            return False
        if context.active_object.type != 'MESH':
            cls.poll_message_set(f"Active object is {context.active_object.type}, need MESH")
            return False
        if not check_mesh_has_gaussian_attributes(context.active_object):
            cls.poll_message_set(f"Active object '{context.active_object.name}' has no gaussian attributes (f_dc_0, f_dc_1, f_dc_2)")
            return False
        return True

    def execute(self, context):

        # Select objects
        gaussian_obj = bpy.context.active_object # shift click gaussian splat obj
        proxy_mesh_obj = [o for o in bpy.context.selected_objects if o != gaussian_obj][0] # click proxy mesh obj

        if proxy_mesh_obj.type != 'MESH':
            self.report({'ERROR'}, "Proxy object must be a mesh")
            return {'CANCELLED'}

        # Build BVH from proxy mesh (in proxy's local space)
        proxy_mesh = proxy_mesh_obj.data
        proxy_mesh.calc_loop_triangles()
        vertices = [v.co.copy() for v in proxy_mesh.vertices]
        triangles = [tuple(tri.vertices) for tri in proxy_mesh.loop_triangles]
        bvh = BVHTree.FromPolygons(vertices, triangles)

        # Coordinate transform: gaussian local -> proxy local
        g2w = gaussian_obj.matrix_world
        w2p = proxy_mesh_obj.matrix_world.inverted()
        g2p = w2p @ g2w

        # Read all gaussian vertex positions
        num_verts = len(gaussian_obj.data.vertices)
        positions = np.zeros(num_verts*3, dtype=np.float32)
        gaussian_obj.data.vertices.foreach_get("co", positions)
        positions = positions.reshape(-1, 3)

        # Allocate storage arrays
        face_indices = np.zeros(num_verts, dtype=np.float32)
        bary_u        = np.zeros(num_verts, dtype=np.float32)
        bary_v        = np.zeros(num_verts, dtype=np.float32)
        bary_w        = np.zeros(num_verts, dtype=np.float32)
        offset_t      = np.zeros(num_verts, dtype=np.float32)
        offset_b      = np.zeros(num_verts, dtype=np.float32)
        offset_n      = np.zeros(num_verts, dtype=np.float32)
        rest_quat_w   = np.zeros(num_verts, dtype=np.float32)
        rest_quat_x   = np.zeros(num_verts, dtype=np.float32)
        rest_quat_y   = np.zeros(num_verts, dtype=np.float32)
        rest_quat_z   = np.zeros(num_verts, dtype=np.float32)
        rest_len_t    = np.zeros(num_verts, dtype=np.float32)
        rest_len_b    = np.zeros(num_verts, dtype=np.float32)
        rest_len_n    = np.zeros(num_verts, dtype=np.float32)
        orig_rot_0    = np.zeros(num_verts, dtype=np.float32)
        orig_rot_1    = np.zeros(num_verts, dtype=np.float32)
        orig_rot_2    = np.zeros(num_verts, dtype=np.float32)
        orig_rot_3    = np.zeros(num_verts, dtype=np.float32)
        orig_scale_0  = np.zeros(num_verts, dtype=np.float32)
        orig_scale_1  = np.zeros(num_verts, dtype=np.float32)
        orig_scale_2  = np.zeros(num_verts, dtype=np.float32)
        
        mesh = gaussian_obj.data
        for attr_name, arr in [('rot_0', orig_rot_0), ('rot_1', orig_rot_1),
                               ('rot_2', orig_rot_2), ('rot_3', orig_rot_3),
                               ('scale_0', orig_scale_0), ('scale_1', orig_scale_1),
                               ('scale_2', orig_scale_2)]:
            if attr_name in [a.name for a in mesh.attributes]:
                mesh.attributes[attr_name].data.foreach_get("value", arr)




        # For each vertex in gaussian_obj
        for i in range(num_verts):
            g_pos_proxy = g2p @ Vector(positions[i])
            location, normal, face_index, dist = bvh.find_nearest(g_pos_proxy)

            # find nearest face on proxy
            if face_index is None:
                face_indices[i] = 0
                bary_u[i] = bary_v[i] = bary_w[i] = 1/3
                rest_quat_w[i] = 1.0
                rest_len_t[i] = 1.0
                rest_len_b[i] = 1.0
                rest_len_n[i] = 1.0
                
                continue

            face_indices[i] = float(face_index)

            tri = proxy_mesh.loop_triangles[face_index]
            v0 = proxy_mesh.vertices[tri.vertices[0]].co
            v1 = proxy_mesh.vertices[tri.vertices[1]].co
            v2 = proxy_mesh.vertices[tri.vertices[2]].co

            # Compute barycentric coordinates of hit point (loc on face that vertex is projected onto)
            u, v, w = barycentric_coords(location, v0, v1, v2)
            bary_u[i] = u
            bary_v[i] = v
            bary_w[i] = w

            # Build TBN (tangent/bitangent/normal) for face
            face_normal_raw = tri.normal # .normalized()
            tangent_raw = (v1 - v0) # .normalized()
            bitangent_raw = face_normal_raw.cross(tangent_raw) #.normalized()
            
            # Get for scaling information
            n_len = face_normal_raw.length
            t_len = tangent_raw.length
            b_len = bitangent_raw.length

            rest_len_n[i] = n_len
            rest_len_t[i] = t_len
            rest_len_b[i] = b_len
            
            face_normal = face_normal_raw.normalized()
            tangent = tangent_raw.normalized()
            bitangent = bitangent_raw.normalized()

            # Offset from hit point to vertex position
            offset_vec = g_pos_proxy - location
            offset_t[i] = offset_vec.dot(tangent)
            offset_b[i] = offset_vec.dot(bitangent)
            offset_n[i] = offset_vec.dot(face_normal)

            rest_q = tbn_to_quaternion(tangent, bitangent, face_normal)
            rest_quat_w[i] = rest_q.w
            rest_quat_x[i] = rest_q.x
            rest_quat_y[i] = rest_q.y
            rest_quat_z[i] = rest_q.z

        # Store all binding data as mesh attributes
        attr_data = {
            '_bind_face_idx':    face_indices,
            '_bind_bary_u':      bary_u,
            '_bind_bary_v':      bary_v,
            '_bind_bary_w':      bary_w,
            '_bind_offset_t':    offset_t,
            '_bind_offset_b':    offset_b,
            '_bind_offset_n':    offset_n,
            '_bind_rest_quat_w': rest_quat_w,
            '_bind_rest_quat_x': rest_quat_x,
            '_bind_rest_quat_y': rest_quat_y,
            '_bind_rest_quat_z': rest_quat_z,
            '_bind_rest_len_b':  rest_len_b,
            '_bind_rest_len_t':  rest_len_t,
            '_bind_rest_len_n':  rest_len_n,
            '_bind_orig_rot_0':  orig_rot_0,
            '_bind_orig_rot_1':  orig_rot_1,
            '_bind_orig_rot_2':  orig_rot_2,
            '_bind_orig_rot_3':  orig_rot_3,
            '_bind_orig_scale_0':orig_scale_0,
            '_bind_orig_scale_1':orig_scale_1,
            '_bind_orig_scale_2':orig_scale_2,

        }

        mesh = gaussian_obj.data
        for attr_name, data in attr_data.items():
            if attr_name in mesh.attributes:
                mesh.attributes.remove(mesh.attributes[attr_name])
            attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
            attr.data.foreach_set("value", data)

        # Store proxy reference
        gaussian_obj["_bind_proxy_mesh"] = proxy_mesh_obj.name

        # Print first 100 bound gaussians
        print(f"\n{'='*80}")
        print(f"Binding Summary: {gaussian_obj.name} -> {proxy_mesh_obj.name}")
        print(f"Total gaussians: {num_verts}, Proxy faces: {len(triangles)}")
        print(f"{'='*80}")
        n = min(100, num_verts)
        for i in range(n):
            print(f"  [{i:4d}] face={int(face_indices[i]):4d}  "
                  f"bary=({bary_u[i]:.3f}, {bary_v[i]:.3f}, {bary_w[i]:.3f})  "
                  f"offset=({offset_t[i]:.4f}, {offset_b[i]:.4f}, {offset_n[i]:.4f})  "
                  f"rest_q=({rest_quat_w[i]:.3f}, {rest_quat_x[i]:.3f}, {rest_quat_y[i]:.3f}, {rest_quat_z[i]:.3f})")
        if num_verts > 100:
            print(f"  ... ({num_verts - 100} more)")
        print(f"{'='*80}\n")

        self.report({'INFO'}, f"Bound {num_verts} gaussians to {len(triangles)} proxy faces")
        
        test_bind_data_integrity(gaussian_obj)
        return {'FINISHED'}

    def invoke(self, context, event):
        return self.execute(context)

# -- vectorised quaternion helpers --

def _quat_multiply(q1, q2):
    """(N,4) x (N,4) -> (N,4) quaternion product. Layout: [w,x,y,z]."""
    w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
    w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
    return np.column_stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def _quat_conjugate(q):
    """Conjugate (= inverse for unit quats). (N,4) -> (N,4)."""
    return q * np.array([1, -1, -1, -1], dtype=np.float32)

def _matrices_to_quaternions(matrices):
    """(N,3,3) rotation matrices -> (N,4) quaternions [w,x,y,z].
    Shepperd's method, fully vectorised."""
    N = matrices.shape[0]
    m00 = matrices[:,0,0]; m01 = matrices[:,0,1]; m02 = matrices[:,0,2]
    m10 = matrices[:,1,0]; m11 = matrices[:,1,1]; m12 = matrices[:,1,2]
    m20 = matrices[:,2,0]; m21 = matrices[:,2,1]; m22 = matrices[:,2,2]
    quats = np.zeros((N, 4), dtype=np.float32)
    trace = m00 + m11 + m22
    remaining = np.ones(N, dtype=bool)
    mask = remaining & (trace > 0)
    if mask.any():
        s = np.sqrt(trace[mask] + 1.0) * 2
        quats[mask, 0] = 0.25 * s
        quats[mask, 1] = (m21[mask] - m12[mask]) / s
        quats[mask, 2] = (m02[mask] - m20[mask]) / s
        quats[mask, 3] = (m10[mask] - m01[mask]) / s
        remaining[mask] = False
    mask = remaining & (m00 > m11) & (m00 > m22)
    if mask.any():
        s = np.sqrt(1.0 + m00[mask] - m11[mask] - m22[mask]) * 2
        quats[mask, 0] = (m21[mask] - m12[mask]) / s
        quats[mask, 1] = 0.25 * s
        quats[mask, 2] = (m01[mask] + m10[mask]) / s
        quats[mask, 3] = (m02[mask] + m20[mask]) / s
        remaining[mask] = False
    mask = remaining & (m11 > m22)
    if mask.any():
        s = np.sqrt(1.0 + m11[mask] - m00[mask] - m22[mask]) * 2
        quats[mask, 0] = (m02[mask] - m20[mask]) / s
        quats[mask, 1] = (m01[mask] + m10[mask]) / s
        quats[mask, 2] = 0.25 * s
        quats[mask, 3] = (m12[mask] + m21[mask]) / s
        remaining[mask] = False
    mask = remaining
    if mask.any():
        s = np.sqrt(1.0 + m22[mask] - m00[mask] - m11[mask]) * 2
        quats[mask, 0] = (m10[mask] - m01[mask]) / s
        quats[mask, 1] = (m02[mask] + m20[mask]) / s
        quats[mask, 2] = (m12[mask] + m21[mask]) / s
        quats[mask, 3] = 0.25 * s
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats /= np.maximum(norms, 1e-10)
    return quats


def _read_bind_attr(mesh_data, attr_name, num_verts):
    """Read a _bind_* float attribute from mesh data into a numpy array."""
    attr = mesh_data.attributes[attr_name]
    data = np.zeros(num_verts, dtype=np.float32)
    attr.data.foreach_get("value", data)
    return data


def Compute_New_World_Positions(gaussian_obj):
    """Read bind attributes from gaussian_obj mesh, evaluate the deformed proxy,
    and return (new_positions (N,3), rotation_delta (N,4)).

    new_positions: where each gaussian should be in the proxy's local space.
    rotation_delta: quaternion [w,x,y,z] to multiply with original gaussian rotation.
    Returns (None, None) if the object has no bind data.
    """
    proxy_name = gaussian_obj.get("_bind_proxy_mesh")
    if not proxy_name:
        return None, None
    proxy_obj = bpy.data.objects.get(proxy_name)
    if not proxy_obj:
        print(f"Proxy mesh '{proxy_name}' not found")
        return None, None

    mesh_data = gaussian_obj.data
    num = len(mesh_data.vertices)

    # Read bind attributes from the gaussian mesh
    face_idx = _read_bind_attr(mesh_data, '_bind_face_idx', num).astype(np.int32)
    bary = np.column_stack([
        _read_bind_attr(mesh_data, '_bind_bary_u', num),
        _read_bind_attr(mesh_data, '_bind_bary_v', num),
        _read_bind_attr(mesh_data, '_bind_bary_w', num),
    ])
    offsets = np.column_stack([
        _read_bind_attr(mesh_data, '_bind_offset_t', num),
        _read_bind_attr(mesh_data, '_bind_offset_b', num),
        _read_bind_attr(mesh_data, '_bind_offset_n', num),
    ])
    rest_q = np.column_stack([
        _read_bind_attr(mesh_data, '_bind_rest_quat_w', num),
        _read_bind_attr(mesh_data, '_bind_rest_quat_x', num),
        _read_bind_attr(mesh_data, '_bind_rest_quat_y', num),
        _read_bind_attr(mesh_data, '_bind_rest_quat_z', num),
    ])
    rest_len_t = _read_bind_attr(mesh_data, '_bind_rest_len_t', num)
    rest_len_b = _read_bind_attr(mesh_data, '_bind_rest_len_b', num)
    rest_len_n = _read_bind_attr(mesh_data, '_bind_rest_len_n', num)

    orig_rotations = np.column_stack([
        _read_bind_attr(mesh_data, '_bind_orig_rot_0', num),
        _read_bind_attr(mesh_data, '_bind_orig_rot_1', num),
        _read_bind_attr(mesh_data, '_bind_orig_rot_2', num),
        _read_bind_attr(mesh_data, '_bind_orig_rot_3', num),
    ])
    
    orig_scales = np.column_stack([
        _read_bind_attr(mesh_data, '_bind_orig_scale_0', num),
        _read_bind_attr(mesh_data, '_bind_orig_scale_1', num),
        _read_bind_attr(mesh_data, '_bind_orig_scale_2', num),
    ])

    # Evaluate the deformed proxy mesh
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_proxy = proxy_obj.evaluated_get(depsgraph)
    eval_mesh  = eval_proxy.data
    eval_mesh.calc_loop_triangles()

    n_verts = len(eval_mesh.vertices)
    n_tris  = len(eval_mesh.loop_triangles)

    proxy_verts = np.zeros(n_verts * 3, dtype=np.float32)
    eval_mesh.vertices.foreach_get("co", proxy_verts)
    proxy_verts = proxy_verts.reshape(-1, 3)

    tri_vert_ids = np.zeros(n_tris * 3, dtype=np.int32)
    eval_mesh.loop_triangles.foreach_get("vertices", tri_vert_ids)
    tri_vert_ids = tri_vert_ids.reshape(-1, 3)

    tri_normals = np.zeros(n_tris * 3, dtype=np.float32)
    eval_mesh.loop_triangles.foreach_get("normal", tri_normals)
    tri_normals = tri_normals.reshape(-1, 3)

    # Gather deformed triangle vertices for each gaussian's bound face
    v0 = proxy_verts[tri_vert_ids[face_idx, 0]]  # (N, 3)
    v1 = proxy_verts[tri_vert_ids[face_idx, 1]]
    v2 = proxy_verts[tri_vert_ids[face_idx, 2]]

    # Anchor = barycentric interpolation on deformed face
    anchor = bary[:, 0:1] * v0 + bary[:, 1:2] * v1 + bary[:, 2:3] * v2

    # Deformed TBN
    raw_normals = tri_normals[face_idx]
    raw_tangents = v1 - v0
    raw_bitangents = np.cross(raw_normals, raw_tangents)

    normals = raw_normals / np.maximum(np.linalg.norm(raw_normals, axis=1, keepdims=True), 1e-10)
    tangents = raw_tangents / np.maximum(np.linalg.norm(raw_tangents, axis=1, keepdims=True), 1e-10)
    bitangents = raw_bitangents / np.maximum(np.linalg.norm(raw_bitangents, axis=1, keepdims=True), 1e-10)

    t_len = np.linalg.norm(raw_tangents, axis=1)
    b_len = np.linalg.norm(raw_bitangents, axis=1)
    n_len = np.linalg.norm(raw_normals, axis=1)

    # New positions = anchor + offset in deformed TBN space (proxy-local)
    new_positions_proxy = (anchor
                         + offsets[:, 0:1] * tangents
                         + offsets[:, 1:2] * bitangents
                         + offsets[:, 2:3] * normals)

    # Transform from proxy-local to gaussian-local space
    # gaussian_data[:, 0:3] stores positions in the gaussian mesh's local space,
    # so we need: proxy_local -> world -> gaussian_local
    # p2g = np.array((gaussian_obj.matrix_world.inverted() @ proxy_obj.matrix_world).transposed())  # 4x4 col-major
    p2g_mat = gaussian_obj.matrix_world.inverted() @ proxy_obj.matrix_world # Blender Matrix Obj needed
    p2g = np.array(p2g_mat.transposed())
    ones = np.ones((new_positions_proxy.shape[0], 1), dtype=np.float32)
    pos_h = np.hstack([new_positions_proxy, ones])  # (N, 4)
    new_positions = (pos_h @ p2g)[:, :3]

    # Rotation delta: deformed_face_quat * inverse(rest_face_quat)
    tbn_matrices = np.stack([tangents, bitangents, normals], axis=-1)  # (N,3,3)
    deformed_q = _matrices_to_quaternions(tbn_matrices)
    rotation_delta = _quat_multiply(deformed_q, _quat_conjugate(rest_q))
    rotation_delta /= np.maximum(np.linalg.norm(rotation_delta, axis=1, keepdims=True), 1e-10)

    # ensure quat sign consistency
    negative_w = rotation_delta[:, 0] < 0
    rotation_delta[negative_w] *= -1

    # Transform rotation delta from proxu-local to gaussian-local space
    p2g_quat = np.array(p2g_mat.to_quaternion()).reshape(1, 4)
    rotation_delta = _quat_multiply(_quat_multiply(p2g_quat, rotation_delta), _quat_conjugate(p2g_quat))
    print(f"    DEBUG rotation_delta[0]: {rotation_delta[0]}")                                                                                          
    print(f"    DEBUG orig_rotations[0]: {orig_rotations[0]}")                                                                                          
    print(f"    DEBUG p2g_quat: {p2g_quat}")
    
    scale_ratios = np.column_stack([
        t_len / np.maximum(rest_len_t, 1e-10),
        b_len / np.maximum(rest_len_b, 1e-10),
        n_len / np.maximum(rest_len_n, 1e-10),
    ])

    return new_positions, rotation_delta, scale_ratios, orig_rotations, orig_scales


def test_bind_data_integrity(gaussian_obj):
    """Verify that bind attributes were stored correctly on the gaussian mesh.
    Call after binding. Prints pass/fail for each check."""
    mesh_data = gaussian_obj.data
    num = len(mesh_data.vertices)
    proxy_name = gaussian_obj.get("_bind_proxy_mesh")
    print(f"\n--- Bind data integrity test: {gaussian_obj.name} ---")

    # Check proxy reference exists
    if not proxy_name:
        print("FAIL: _bind_proxy_mesh not set on object")
        return False
    proxy_obj = bpy.data.objects.get(proxy_name)
    if not proxy_obj:
        print(f"FAIL: proxy object '{proxy_name}' not found in scene")
        return False
    print(f"PASS: proxy reference -> '{proxy_name}'")

    # Check all 11 attributes exist and have correct length
    attr_names = [
        '_bind_face_idx', '_bind_bary_u', '_bind_bary_v', '_bind_bary_w',
        '_bind_offset_t', '_bind_offset_b', '_bind_offset_n',
        '_bind_rest_quat_w', '_bind_rest_quat_x', '_bind_rest_quat_y', '_bind_rest_quat_z',
        '_bind_rest_len_t', '_bind_rest_len_b', '_bind_rest_len_n'
    ]
    all_present = True
    for name in attr_names:
        if name not in [a.name for a in mesh_data.attributes]:
            print(f"FAIL: attribute '{name}' missing")
            all_present = False
        elif len(mesh_data.attributes[name].data) != num:
            print(f"FAIL: attribute '{name}' has {len(mesh_data.attributes[name].data)} entries, expected {num}")
            all_present = False
    if all_present:
        print(f"PASS: all 11 bind attributes present with {num} entries each")

    # Check barycentric coords sum to ~1
    bary_u = _read_bind_attr(mesh_data, '_bind_bary_u', num)
    bary_v = _read_bind_attr(mesh_data, '_bind_bary_v', num)
    bary_w = _read_bind_attr(mesh_data, '_bind_bary_w', num)
    bary_sum = bary_u + bary_v + bary_w
    max_err = np.max(np.abs(bary_sum - 1.0))
    if max_err < 0.01:
        print(f"PASS: barycentric sums to 1.0 (max error {max_err:.6f})")
    else:
        print(f"FAIL: barycentric sum max error {max_err:.6f}")

    # Check rest quaternions are unit length
    rest_q = np.column_stack([
        _read_bind_attr(mesh_data, '_bind_rest_quat_w', num),
        _read_bind_attr(mesh_data, '_bind_rest_quat_x', num),
        _read_bind_attr(mesh_data, '_bind_rest_quat_y', num),
        _read_bind_attr(mesh_data, '_bind_rest_quat_z', num),
    ])
    quat_norms = np.linalg.norm(rest_q, axis=1)
    max_norm_err = np.max(np.abs(quat_norms - 1.0))
    if max_norm_err < 0.01:
        print(f"PASS: rest quaternions are unit length (max error {max_norm_err:.6f})")
    else:
        print(f"FAIL: rest quaternion norm max error {max_norm_err:.6f}")

    # Check face indices are in valid range
    face_idx = _read_bind_attr(mesh_data, '_bind_face_idx', num).astype(np.int32)
    proxy_mesh = proxy_obj.data
    proxy_mesh.calc_loop_triangles()
    n_tris = len(proxy_mesh.loop_triangles)
    out_of_range = np.sum((face_idx < 0) | (face_idx >= n_tris))
    if out_of_range == 0:
        print(f"PASS: all face indices in range [0, {n_tris})")
    else:
        print(f"FAIL: {out_of_range} face indices out of range [0, {n_tris})")

    # Round-trip test: compute positions from bind data on the undeformed proxy
    # and compare to original gaussian positions
    new_pos, _, _, _, _= Compute_New_World_Positions(gaussian_obj)
    if new_pos is not None:
        orig_pos = np.zeros(num * 3, dtype=np.float32)
        gaussian_obj.data.vertices.foreach_get("co", orig_pos)
        orig_pos = orig_pos.reshape(-1, 3)
        # Positions are in proxy local space, transform originals to proxy space
        g2w = gaussian_obj.matrix_world
        w2p = proxy_obj.matrix_world.inverted()
        g2p = np.array((w2p @ g2w).transposed())  # 4x4 column-major
        ones = np.ones((num, 1), dtype=np.float32)
        orig_h = np.hstack([orig_pos, ones])  # (N, 4)
        orig_in_proxy = (orig_h @ g2p)[:, :3]
        pos_err = np.linalg.norm(new_pos - orig_in_proxy, axis=1)
        max_pos_err = np.max(pos_err)
        mean_pos_err = np.mean(pos_err)
        if max_pos_err < 0.01:
            print(f"PASS: round-trip position error max={max_pos_err:.6f} mean={mean_pos_err:.6f}")
        else:
            print(f"WARN: round-trip position error max={max_pos_err:.6f} mean={mean_pos_err:.6f}")

    print("--- End integrity test ---\n")
    return True

