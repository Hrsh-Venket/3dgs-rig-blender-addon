from .important import *
from .extract_gaussian_from_evaluated_mesh import *

def sna_texture_creation_FD1B2():
    # ========== VARIABLES (EDIT THESE) ==========
    # No variables needed - builds from all cached objects
    # ============================================
    import bpy
    import gpu
    import os
    # ========== FALLBACK FUNCTIONS FOR CORRUPTED DATA ==========

    def find_source_object_by_uuid(source_uuid):
        """Find Blender object by gaussian_source_uuid"""
        for obj in bpy.data.objects:
            if obj.get("gaussian_source_uuid") == source_uuid:
                return obj
        return None

    def refresh_object_from_blender_source(obj):
        """Refresh gaussian data from Blender mesh source - fallback function"""
        try:
            source_uuid = obj.get("source_mesh_uuid")
            if not source_uuid:
                return False, "No source UUID found"
            # Find source object by UUID
            source_obj = find_source_object_by_uuid(source_uuid)
            if not source_obj:
                return False, f"Source object with UUID {source_uuid} not found"
            # Validate that source object has gaussian attributes
            if not check_mesh_has_gaussian_attributes(source_obj):
                return False, f"Source object '{source_obj.name}' missing gaussian attributes"
            print(f"  🔄 Fallback: Refreshing {obj.name} from source mesh {source_obj.name}")
            # Extract fresh data from evaluated mesh
            gaussian_data_info = extract_gaussian_data_from_evaluated_mesh(source_obj)
            # Create gaussian data array (59 floats per gaussian)
            num_gaussians = gaussian_data_info['num_points']
            sh_dim = 48
            total_dim = 3 + 4 + 3 + 1 + sh_dim
            gaussian_data = np.zeros((num_gaussians, total_dim), dtype=np.float32)
            # Pack data in original order
            gaussian_data[:, 0:3] = gaussian_data_info['positions']
            gaussian_data[:, 3:7] = gaussian_data_info['rotations']
            gaussian_data[:, 7:10] = gaussian_data_info['scales']
            gaussian_data[:, 10] = gaussian_data_info['opacities'].flatten()
            # Handle SH coefficients
            source_sh_coeffs = gaussian_data_info['sh_coeffs']
            if source_sh_coeffs.shape[1] >= sh_dim:
                gaussian_data[:, 11:11+sh_dim] = source_sh_coeffs[:, :sh_dim]
            else:
                gaussian_data[:, 11:11+source_sh_coeffs.shape[1]] = source_sh_coeffs
            # Update object properties with fresh data
            obj["gaussian_data"] = gaussian_data.tobytes()
            obj["gaussian_count"] = num_gaussians
            obj["sh_degree"] = gaussian_data_info['sh_dim']
            obj["last_load_time"] = time.time()
            return True, (gaussian_data, num_gaussians, gaussian_data_info['sh_dim'])
        except Exception as e:
            return False, f"Fallback refresh failed: {e}"

    def refresh_object_from_ply_source(obj):
        """Refresh gaussian data from PLY file - fallback function"""
        try:
            ply_filepath = obj.get("ply_filepath")
            if not ply_filepath or not os.path.exists(ply_filepath):
                return False, "PLY file not found or missing path"
            print(f"  🔄 Fallback: Refreshing {obj.name} from PLY {os.path.basename(ply_filepath)}")
            # Simple PLY loading (minimal implementation for fallback)
            from plyfile import PlyData
            plydata = PlyData.read(ply_filepath)
            vertex_element = plydata.elements[0]
            vertex_data = vertex_element.data
            available_fields = list(vertex_data.dtype.names)
            # Extract positions
            if 'x' in available_fields and 'y' in available_fields and 'z' in available_fields:
                positions = np.column_stack([vertex_data['x'], vertex_data['y'], vertex_data['z']])
                positions = np.ascontiguousarray(positions).astype(np.float32)
            else:
                return False, "PLY missing position coordinates"
            num_points = len(positions)
            # Extract SH coefficients (simplified)
            if all(attr in available_fields for attr in ['f_dc_0', 'f_dc_1', 'f_dc_2']):
                dc_0 = vertex_data['f_dc_0']
                dc_1 = vertex_data['f_dc_1'] 
                dc_2 = vertex_data['f_dc_2']
                sh_coeffs = np.column_stack([dc_0, dc_1, dc_2]).astype(np.float32)
            else:
                sh_coeffs = np.ones((num_points, 3), dtype=np.float32) * 0.28209479177387814
            # Extract scales
            if all(attr in available_fields for attr in ['scale_0', 'scale_1', 'scale_2']):
                scale_0 = vertex_data['scale_0']
                scale_1 = vertex_data['scale_1']
                scale_2 = vertex_data['scale_2']
                scales = np.column_stack([scale_0, scale_1, scale_2])
                scales = np.exp(scales).astype(np.float32)
            else:
                scales = np.ones((num_points, 3), dtype=np.float32) * 0.01
            # Extract rotations
            if all(attr in available_fields for attr in ['rot_0', 'rot_1', 'rot_2', 'rot_3']):
                rot_0 = vertex_data['rot_0']
                rot_1 = vertex_data['rot_1']
                rot_2 = vertex_data['rot_2']
                rot_3 = vertex_data['rot_3']
                rotations = np.column_stack([rot_0, rot_1, rot_2, rot_3])
                norms = np.linalg.norm(rotations, axis=1, keepdims=True)
                rotations = (rotations / norms).astype(np.float32)
            else:
                rotations = np.zeros((num_points, 4), dtype=np.float32)
                rotations[:, 0] = 1.0
            # Extract opacity
            if 'opacity' in available_fields:
                opacity = vertex_data['opacity']
                opacity = (1.0 / (1.0 + np.exp(-opacity))).astype(np.float32)
            else:
                opacity = np.ones(num_points, dtype=np.float32)
            # Create gaussian data array
            sh_dim = 48
            total_dim = 3 + 4 + 3 + 1 + sh_dim
            gaussian_data = np.zeros((num_points, total_dim), dtype=np.float32)
            # Pack data
            gaussian_data[:, 0:3] = positions
            gaussian_data[:, 3:7] = rotations
            gaussian_data[:, 7:10] = scales
            gaussian_data[:, 10] = opacity.flatten()
            if sh_coeffs.shape[1] >= sh_dim:
                gaussian_data[:, 11:11+sh_dim] = sh_coeffs[:, :sh_dim]
            else:
                gaussian_data[:, 11:11+sh_coeffs.shape[1]] = sh_coeffs
            # Update object properties
            obj["gaussian_data"] = gaussian_data.tobytes()
            obj["gaussian_count"] = num_points
            obj["sh_degree"] = sh_coeffs.shape[1]
            obj["last_load_time"] = time.time()
            return True, (gaussian_data, num_points, sh_coeffs.shape[1])
        except Exception as e:
            return False, f"PLY fallback failed: {e}"

    def auto_reconstruct_cache_for_script3():
        """Auto-reconstruct cache from scene objects with fallback for corrupted data"""
        try:
            # Find all gaussian objects in the scene
            gaussian_objects = []
            for obj in bpy.data.objects:
                if obj.get("is_gaussian_splat", False):
                    gaussian_objects.append(obj)
            if not gaussian_objects:
                return False
            print(f"Auto-reconstructing cache from {len(gaussian_objects)} scene objects...")
            # Initialize fresh cache
            bpy.gaussian_object_cache = {}
            total_gaussians = 0
            fallback_count = 0
            for obj in gaussian_objects:
                try:
                    # Extract data from object properties
                    data_bytes = obj.get("gaussian_data")
                    gaussian_count = obj.get("gaussian_count", 0)
                    sh_degree = obj.get("sh_degree", 48)
                    ply_filepath = obj.get("ply_filepath", "")
                    if not data_bytes or gaussian_count == 0:
                        print(f"{obj.name}: Missing data or zero count, skipping")
                        continue
                    # Try to reconstruct numpy array from bytes
                    try:
                        # Ensure we have raw bytes (IDPropertyArray may not
                        # expose the buffer protocol correctly for large arrays)
                        if not isinstance(data_bytes, (bytes, bytearray)):
                            # TODO: Check edge cases as bytes() creates an immuatble array
                            data_bytes = bytes(data_bytes)
                        gaussian_data = np.frombuffer(data_bytes, dtype=np.float32).reshape(gaussian_count, 59)
                        # Validate data integrity
                        if gaussian_data.shape != (gaussian_count, 59):
                            raise ValueError("Data shape validation failed")
                        # Check for reasonable values (basic sanity check)
                        if np.any(np.isnan(gaussian_data)) or np.any(np.isinf(gaussian_data)):
                            raise ValueError("Data contains NaN or infinity values")
                        print(f" {obj.name}: Successfully reconstructed from cache")
                    except (ValueError, TypeError) as e:
                        print(f"  {obj.name}: Cache data corrupted ({e})")
                        print(f"     Attempting fallback refresh...")
                        # Determine source type and attempt fallback
                        is_blender_source = obj.get("source_mesh_uuid") is not None
                        is_ply_source = ply_filepath and ply_filepath.strip()
                        fallback_success = False
                        if is_blender_source:
                            success, result = refresh_object_from_blender_source(obj)
                            if success:
                                gaussian_data, gaussian_count, sh_degree = result
                                fallback_success = True
                                fallback_count += 1
                            else:
                                print(f"     Blender source fallback failed: {result}")
                        elif is_ply_source:
                            success, result = refresh_object_from_ply_source(obj)
                            if success:
                                gaussian_data, gaussian_count, sh_degree = result
                                fallback_success = True
                                fallback_count += 1
                            else:
                                print(f"     PLY source fallback failed: {result}")
                        if not fallback_success:
                            print(f"     All fallback methods failed for {obj.name}, skipping")
                            continue
                    # Add to cache
                    source_info = ""
                    if obj.get("source_mesh_uuid"):
                        source_info = f"Mesh:{obj.get('source_mesh_name', 'Unknown')}"
                    elif ply_filepath:
                        source_info = f"PLY:{os.path.basename(ply_filepath)}"
                    bpy.gaussian_object_cache[obj.name] = {
                        'gaussian_data': gaussian_data,
                        'gaussian_count': gaussian_count,
                        'sh_degree': sh_degree,
                        'object': obj,
                        'ply_filepath': ply_filepath,
                        'source_info': source_info
                    }
                    total_gaussians += gaussian_count
                except Exception as e:
                    print(f" {obj.name}: Reconstruction failed completely: {e}")
                    continue
            if bpy.gaussian_object_cache:
                cache_status = f"Cache reconstructed: {len(bpy.gaussian_object_cache)} objects, {total_gaussians:,} gaussians"
                if fallback_count > 0:
                    cache_status += f" ({fallback_count} restored from source)"
                print(cache_status)
                return True
            else:
                return False
        except Exception as e:
            print(f"Auto-reconstruction failed: {e}")
            return False
    # ========== MAIN SCRIPT ==========
    try:
        # ========== AUTO-RECONSTRUCTION CHECK ==========
        # Check if we have cached objects, if not try to reconstruct
        if not hasattr(bpy, 'gaussian_object_cache') or not bpy.gaussian_object_cache:
            reconstruction_success = auto_reconstruct_cache_for_script3()
            if not reconstruction_success:
                raise ValueError("No gaussian objects found in scene - run script_1 first")
        print(f"Building global textures from {len(bpy.gaussian_object_cache)} objects:")
        # ========== MERGE DATA FROM ALL OBJECTS ==========
        all_gaussian_data = []
        all_object_metadata = []
        current_start_idx = 0
        for obj_name, obj_data in bpy.gaussian_object_cache.items():
            gaussian_data = obj_data['gaussian_data']
            gaussian_count = obj_data['gaussian_count']
            obj = obj_data['object']
            source_info = obj_data.get('source_info', 'Unknown')
            print(f"  - {obj_name}: {gaussian_count:,} gaussians ({source_info})")
            # Add to merged data
            all_gaussian_data.append(gaussian_data)
            # Store metadata for this object
            all_object_metadata.append({
                'name': obj_name,
                'start_idx': current_start_idx,
                'gaussian_count': gaussian_count,
                'object': obj
            })
            current_start_idx += gaussian_count
        # Merge all gaussian data into single array
        merged_gaussian_data = np.concatenate(all_gaussian_data, axis=0)
        total_gaussians = len(merged_gaussian_data)
        print(f"Total merged gaussians: {total_gaussians:,}")
        # ========== CREATE GLOBAL 3D GAUSSIAN TEXTURE ==========
        total_floats = merged_gaussian_data.size
        max_texture_dim = 16384
        # Calculate 3D texture dimensions using original method
        cube_root = int(np.ceil(np.power(total_floats, 1/3)))
        texture_depth = min(max_texture_dim, cube_root)
        texture_area = (total_floats + texture_depth - 1) // texture_depth
        texture_width = min(max_texture_dim, int(np.ceil(np.sqrt(texture_area))))
        texture_height = (texture_area + texture_width - 1) // texture_width
        # Pad data if needed
        flat_data = merged_gaussian_data.flatten()
        expected_size = texture_width * texture_height * texture_depth
        if len(flat_data) < expected_size:
            padded_data = np.zeros(expected_size, dtype=np.float32)
            padded_data[:len(flat_data)] = flat_data
            flat_data = padded_data
        # Create 3D texture
        buffer = gpu.types.Buffer('FLOAT', len(flat_data), flat_data.tolist())
        gaussian_texture = gpu.types.GPUTexture(
            (texture_width, texture_height, texture_depth), 
            format='R32F',
            data=buffer
        )
        # ========== CREATE GLOBAL INDICES TEXTURE ==========
        sorted_indices = np.arange(total_gaussians, dtype=np.float32)
        indices_width = min(max_texture_dim, len(sorted_indices))
        indices_height = (len(sorted_indices) + indices_width - 1) // indices_width
        expected_indices_size = indices_width * indices_height
        if len(sorted_indices) < expected_indices_size:
            padded_indices = np.zeros(expected_indices_size, dtype=np.float32)
            padded_indices[:len(sorted_indices)] = sorted_indices
            indices_data = padded_indices
        else:
            indices_data = sorted_indices
        indices_buffer = gpu.types.Buffer('FLOAT', len(indices_data), indices_data.tolist())
        indices_texture = gpu.types.GPUTexture(
            (indices_width, indices_height),
            format='R32F',
            data=indices_buffer
        )
        # ========== CREATE MULTI-OBJECT METADATA TEXTURE ==========
        num_objects = len(all_object_metadata)
        floats_per_object = 15
        total_metadata_floats = num_objects * floats_per_object
        metadata_width = min(max_texture_dim, total_metadata_floats)
        metadata_height = (total_metadata_floats + metadata_width - 1) // metadata_width
        expected_size = metadata_width * metadata_height
        metadata_data = np.zeros(expected_size, dtype=np.float32)
        # Fill metadata for each object
        for obj_idx, obj_meta in enumerate(all_object_metadata):
            base_idx = obj_idx * floats_per_object
            # Start index (uint32 bitcast to float32)
            uint32_start_idx = np.uint32(obj_meta['start_idx'])
            metadata_data[base_idx + 0] = uint32_start_idx.view(np.float32)
            metadata_data[base_idx + 1] = float(obj_meta['gaussian_count'])
            metadata_data[base_idx + 2] = 1.0  # Visible
            # Object transform matrix (3x4 = 12 floats)
            transform = obj_meta['object'].matrix_world
            matrix_idx = 0
            for col in range(4):
                for row in range(3):
                    metadata_data[base_idx + 3 + matrix_idx] = transform[row][col]
                    matrix_idx += 1
        metadata_buffer = gpu.types.Buffer('FLOAT', len(metadata_data), metadata_data.tolist())
        metadata_texture = gpu.types.GPUTexture(
            (metadata_width, metadata_height), 
            format='R32F', 
            data=metadata_buffer
        )
        # ========== STORE GLOBALLY ==========
        bpy.gaussian_texture = gaussian_texture
        bpy.gaussian_texture_width = texture_width
        bpy.gaussian_texture_height = texture_height
        bpy.gaussian_texture_depth = texture_depth
        bpy.gaussian_indices_texture = indices_texture
        bpy.gaussian_indices_width = indices_width
        bpy.gaussian_indices_height = indices_height
        bpy.gaussian_metadata_texture = metadata_texture
        bpy.gaussian_count = total_gaussians
        bpy.gaussian_object_metadata = all_object_metadata  # For transform tracking
        bpy.gaussian_global_needs_update = False  # Mark as updated
        bpy.gaussian_needs_depth_sort = True  # NEW: Signal viewport renderer to force depth sort
        print(f"Global textures created:")
        print(f"  Gaussian: {texture_width}x{texture_height}x{texture_depth}")
        print(f"  Indices: {indices_width}x{indices_height}")
        print(f"  Metadata: {metadata_width}x{metadata_height} for {num_objects} objects")
        print(f"  Depth sort flagged for next viewport render")
    except Exception as e:
        print(f"Error creating global textures: {e}")
        import traceback
        traceback.print_exc()
