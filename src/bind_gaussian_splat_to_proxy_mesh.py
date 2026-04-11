import bpy
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from .extract_gaussian_from_evaluated_mesh import check_mesh_has_gaussian_attributes
from .sna_c2_refresh_all import sna_c2_refresh_all_4D367
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

        # For each vertex in gaussian_obj
        for i in range(num_verts):
            g_pos_proxy = g2p @ Vector(positions[i])
            location, normal, face_index, dist = bvh.find_nearest(g_pos_proxy)

            # find nearest face on proxy
            if face_index is None:
                face_indices[i] = 0
                bary_u[i] = bary_v[i] = bary_w[i] = 1/3
                rest_quat_w[i] = 1.0
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
            face_normal = tri.normal.normalized()
            tangent = (v1 - v0).normalized()
            bitangent = face_normal.cross(tangent).normalized()

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
        }

        mesh = gaussian_obj.data
        for attr_name, data in attr_data.items():
            if attr_name in mesh.attributes:
                mesh.attributes.remove(mesh.attributes[attr_name])
            attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
            attr.data.foreach_set("value", data)

        # Store proxy reference
        gaussian_obj["_bind_proxy_mesh"] = proxy_mesh_obj.name

        self.report({'INFO'}, f"Bound {num_verts} gaussians to {len(triangles)} proxy faces")
        return {'FINISHED'}

    def invoke(self, context, event):
        return self.execute(context)


class Refresh_Proxy_Gaussians(bpy.types.Operator):
    bl_idname = "sna.refresh_proxy_gaussians"
    bl_label = "3DGS Render: Refresh Proxy Gaussians"
    bl_description = "Re-extract gaussian data from evaluated proxy mesh and rebuild GPU textures"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        for obj in bpy.data.objects:
            if obj.get("_bind_proxy_mesh"):
                return True
        cls.poll_message_set("No proxy-bound gaussian objects found")
        return False

    def execute(self, context):
        sna_c2_refresh_all_4D367(True, False, True)
        sna_texture_creation_FD1B2()
        self.report({'INFO'}, "Proxy gaussians refreshed")
        return {'FINISHED'}

    def invoke(self, context, event):
        return self.execute(context)
