# import bpy
# from .important import *
#
# MODIFIER_NAME = 'KIRI_Weight_Vis'
# NODE_TREE_NAME = 'KIRI_WeightVis_GN'
# MATERIAL_NAME = 'KIRI_WeightVis_Mat'
#
#
# def _ensure_vis_material():
#     """Material: reads 'vis_weight' attribute → ColorRamp (blue-green-red) → base color."""
#     if MATERIAL_NAME in bpy.data.materials:
#         return bpy.data.materials[MATERIAL_NAME]
#
#     mat = bpy.data.materials.new(MATERIAL_NAME)
#     mat.use_nodes = True
#     nt = mat.node_tree
#     nt.nodes.clear()
#
#     attr = nt.nodes.new('ShaderNodeAttribute')
#     attr.attribute_name = 'vis_weight'
#     attr.attribute_type = 'GEOMETRY'
#
#     ramp = nt.nodes.new('ShaderNodeValToRGB')
#     els = ramp.color_ramp.elements
#     els[0].position, els[0].color = 0.0, (0, 0, 1, 1)
#     els[1].position, els[1].color = 1.0, (1, 0, 0, 1)
#     mid = els.new(0.5)
#     mid.color = (0, 1, 0, 1)
#
#     bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
#     out = nt.nodes.new('ShaderNodeOutputMaterial')
#
#     nt.links.new(attr.outputs['Fac'], ramp.inputs['Fac'])
#     nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
#     nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
#
#     return mat
#
#
# def _build_node_tree(vg_name):
#     """GN tree: read vertex group → store as 'vis_weight' → mesh to points → set material."""
#     if NODE_TREE_NAME in bpy.data.node_groups:
#         bpy.data.node_groups.remove(bpy.data.node_groups[NODE_TREE_NAME])
#
#     tree = bpy.data.node_groups.new(NODE_TREE_NAME, 'GeometryNodeTree')
#     tree.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
#     tree.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
#
#     n = tree.nodes
#     l = tree.links
#
#     gi = n.new('NodeGroupInput')
#
#     na = n.new('GeometryNodeInputNamedAttribute')
#     na.data_type = 'FLOAT'
#     na.inputs['Name'].default_value = vg_name
#
#     store = n.new('GeometryNodeStoreNamedAttribute')
#     store.data_type = 'FLOAT'
#     store.domain = 'POINT'
#     store.inputs['Name'].default_value = 'vis_weight'
#
#     m2p = n.new('GeometryNodeMeshToPoints')
#     m2p.mode = 'VERTICES'
#     m2p.inputs['Radius'].default_value = 0.005
#
#     sm = n.new('GeometryNodeSetMaterial')
#     sm.inputs['Material'].default_value = _ensure_vis_material()
#
#     go = n.new('NodeGroupOutput')
#
#     l.new(gi.outputs['Geometry'], store.inputs['Geometry'])
#     l.new(na.outputs['Attribute'], store.inputs['Value'])
#     l.new(store.outputs['Geometry'], m2p.inputs['Mesh'])
#     l.new(m2p.outputs['Points'], sm.inputs['Geometry'])
#     l.new(sm.outputs['Geometry'], go.inputs['Geometry'])
#
#     return tree
#
#
# class SNA_OT_Dgs_Render_Visualize_Weights(bpy.types.Operator):
#     bl_idname = "sna.dgs_render_visualize_weights"
#     bl_label = "3DGS Render: Visualize Weights"
#     bl_description = (
#         "Toggle weight visualization. Adds a geometry nodes modifier that "
#         "colours each vertex by the active vertex group weight "
#         "(blue=0, green=0.5, red=1). Click again to remove"
#     )
#     bl_options = {"REGISTER", "UNDO"}
#
#     @classmethod
#     def poll(cls, context):
#         obj = context.active_object
#         return obj and obj.type == 'MESH' and len(obj.vertex_groups) > 0
#
#     def execute(self, context):
#         obj = context.active_object
#
#         # Toggle off
#         if MODIFIER_NAME in obj.modifiers:
#             obj.modifiers.remove(obj.modifiers[MODIFIER_NAME])
#             bpy._weight_vis_running = False
#             self.report({'INFO'}, "Weight visualization stopped")
#             return {'FINISHED'}
#
#         # Toggle on
#         vg_index = obj.vertex_groups.active_index
#         if vg_index < 0:
#             self.report({'WARNING'}, "No active vertex group")
#             return {'CANCELLED'}
#
#         vg_name = obj.vertex_groups[vg_index].name
#         tree = _build_node_tree(vg_name)
#         mod = obj.modifiers.new(MODIFIER_NAME, 'NODES')
#         mod.node_group = tree
#
#         # Switch viewport to Material Preview so colours are visible
#         for area in context.screen.areas:
#             if area.type == 'VIEW_3D':
#                 for space in area.spaces:
#                     if space.type == 'VIEW_3D':
#                         space.shading.type = 'MATERIAL'
#
#         bpy._weight_vis_running = True
#         self.report({'INFO'}, f"Showing weights: {vg_name}")
#         return {'FINISHED'}
#
#     def invoke(self, context, event):
#         return self.execute(context)
