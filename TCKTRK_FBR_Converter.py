# TCKTRK_FBR_Converter.py
# BrainVoyager Python Plugin for TCK/TRK <-> FBR Conversion
# Version 1.0 - 2025-05-09
# Author: Quentin Dessain

import sys
import os
import struct
import numpy as np
import nibabel as nib
from nibabel.streamlines import Tractogram, Field
from nibabel.orientations import aff2axcodes

try:
    import bva
    IS_BRAINVOYAGER_ENV = True
except ImportError:
    IS_BRAINVOYAGER_ENV = False
    print("Warning: bva not found. Non-BrainVoyager mode activated.")

try:
    from PySide6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
        QFileDialog, QLabel, QLineEdit, QComboBox, QCheckBox, QGroupBox,
        QRadioButton, QButtonGroup, QTextEdit, QColorDialog, QDoubleSpinBox,
        QSizePolicy
    )
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtCore import Qt
    IS_GUI_AVAILABLE = True
    if not IS_BRAINVOYAGER_ENV:
        from __feature__ import snake_case
except ImportError:
    IS_GUI_AVAILABLE = False
    print("Warning: PySide6 not found. GUI will not be available. Running examples in non-GUI mode.")

# --- FBRFile Class ---
class FBRFile:
    def __init__(self, filename):
        """
        Parameters:
          filename          - Path to the FBR file.
        """
        self.filename = filename

        self.magic = None
        self.file_version = None
        self.coords_type = None
        self.fibers_origin = None
        self.groups = []

    def read(self):

        with open(self.filename, 'rb') as f:
            # Read and verify magic bytes
            self.magic = f.read(4)
            if self.magic != b'\xa4\xd3\xc2\xb1':
                raise ValueError("Invalid FBR file: Incorrect magic bytes")

            # Read header fields
            self.file_version = struct.unpack('<I', f.read(4))[0]
            self.coords_type = struct.unpack('<I', f.read(4))[0]
            self.fibers_origin = struct.unpack('<3f', f.read(12))
            num_groups = struct.unpack('<I', f.read(4))[0]

            # Read groups
            for _ in range(num_groups):
                group = {}

                # Read group name (null-terminated string)
                group_name = bytearray()
                while True:
                    char = f.read(1)
                    if char == b'\x00': break
                    group_name += char
                group['name'] = group_name.decode('latin-1')

                # Read group properties
                group['visible'] = struct.unpack('<I', f.read(4))[0]
                group['animate'] = struct.unpack('<i', f.read(4))[0]
                group['thickness'] = struct.unpack('<f', f.read(4))[0]
                group['color'] = struct.unpack('<3B', f.read(3))
                num_fibers = struct.unpack('<I', f.read(4))[0]

                fibers = []
                for _ in range(num_fibers):
                    fiber = {}
                    num_points = struct.unpack('<I', f.read(4))[0]

                    # Read fiber points (raw)
                    if num_points == 0:  # Handle empty fibers
                        fiber['points'] = []
                        fiber['colors'] = []
                        fibers.append(fiber)
                        continue
                    points_data = struct.unpack(f'<{3 * num_points}f', f.read(12 * num_points))
                    fbr_written_y_block = points_data[:num_points]
                    fbr_written_z_block = points_data[num_points:2 * num_points]
                    fbr_written_x_block = points_data[2 * num_points:]

                    x_coords = [-val for val in fbr_written_x_block]
                    y_coords = [-val for val in fbr_written_y_block]
                    z_coords = [-val for val in fbr_written_z_block]
                    fiber['points'] = list(zip(x_coords, y_coords, z_coords))

                    # Read point colors (RGB)
                    colors_data = struct.unpack(f'<{3 * num_points}B', f.read(3 * num_points))
                    r_values = colors_data[:num_points]
                    g_values = colors_data[num_points:2 * num_points]
                    b_values = colors_data[2 * num_points:]
                    fiber['colors'] = list(zip(r_values, g_values, b_values))
                    fibers.append(fiber)
                group['fibers'] = fibers
                self.groups.append(group)
        return self

    def __repr__(self):
        return (f"FBRFile(filename={self.filename}, "
                f"file_version={self.file_version}, "
                f"coords_type={self.coords_type}, "
                f"groups={len(self.groups)})")


def import_tcktrk_to_fbr_logic(tcktrk_filename, fbr_filename, params, log_callback):
    log_callback(f"Starting import: {tcktrk_filename} -> {fbr_filename}")

    # --- Step 1. Read the TCK/TRK file ---
    try:
        tractogram_obj = nib.streamlines.load(tcktrk_filename)
        input_streamlines = list(tractogram_obj.streamlines)
    except Exception as e:
        log_callback(f"Error loading TCK/TRK file {tcktrk_filename}: {e}")
        return False

    log_callback(f"Read {len(input_streamlines)} streamlines from {tcktrk_filename}.")

    # --- Step 2. Prepare streamlines for FBR format ---
    processed_streamlines_for_fbr = []
    for sl_points_rasmm in input_streamlines:
        # FBR fibers usually need at least 1 point, meaningful fibers at least 2.
        if sl_points_rasmm.shape[0] < 2:
            continue

        # Apply coordinate transformation if specified
        if params['coord_offset'] is not None:
            sl_points_fbr_space = sl_points_rasmm + params['coord_offset']
        else:
            sl_points_fbr_space = sl_points_rasmm

        processed_streamlines_for_fbr.append(sl_points_fbr_space.astype(np.float32))

    if not processed_streamlines_for_fbr:
        log_callback(f"Warning: No valid streamlines (>= 2 points) to write to FBR file {fbr_filename}.")

    # Precompute default color array for efficiency
    _default_point_color_arr_uint8 = np.array(params['default_point_color'], dtype=np.uint8)

    # --- Step 3. Write the FBR file ---
    with open(fbr_filename, 'wb') as f:
        # Write FBR Header
        f.write(params['magic'])
        f.write(struct.pack('<I', params['file_version']))  # unsigned int
        f.write(struct.pack('<I', params['coords_type']))  # unsigned int
        f.write(struct.pack('<3f', *params['fibers_origin']))  # 3 floats

        num_groups = 1  # For simplicity, all streamlines go into a single group
        f.write(struct.pack('<I', num_groups))  # unsigned int

        # Write Group Data
        # Group Name (null-terminated string)
        group_name_bytes = params['group_name'].encode('latin-1')
        f.write(group_name_bytes)
        f.write(b'\x00')  # Null terminator

        f.write(struct.pack('<I', params['group_visible']))  # unsigned int
        f.write(struct.pack('<i', params['group_animate']))  # signed int
        f.write(struct.pack('<f', params['group_thickness']))  # float
        f.write(struct.pack('<3B', *params['group_color']))  # 3 unsigned bytes (RGB)

        num_fibers_in_group = len(processed_streamlines_for_fbr)
        f.write(struct.pack('<I', num_fibers_in_group))  # unsigned int

        # Write Fiber Data
        for fiber_idx, fiber_points_fbr_space in enumerate(processed_streamlines_for_fbr):
            num_points = fiber_points_fbr_space.shape[0]
            f.write(struct.pack('<I', num_points))  # unsigned int

            # Fiber Points (all X, then all Y, then all Z)
            x_coords = -fiber_points_fbr_space[:, 0]
            y_coords = -fiber_points_fbr_space[:, 1]
            z_coords = -fiber_points_fbr_space[:, 2]

            # Exchange x and z coordinates to match FBR format
            # Pack and write Y coordinates
            f.write(struct.pack(f'<{num_points}f', *y_coords))
            # Pack and write Z coordinates
            f.write(struct.pack(f'<{num_points}f', *z_coords))
            # Pack and write X coordinates
            f.write(struct.pack(f'<{num_points}f', *x_coords))

            # Fiber Point Colors (all R, then all G, then all B)
            r_channel_values = []
            g_channel_values = []
            b_channel_values = []

            # Fiber Point Colors (all R, then all G, then all B)
            # This block is optimized
            if params['color_scheme'] == 'orientation' and num_points >= 2:
                # Initialize all point colors to default. This handles points in
                # zero-length segments or if all segments are zero-length.
                point_colors_rgb = np.tile(_default_point_color_arr_uint8, (num_points, 1))

                # Calculate orientations for segments
                # segments vector: P_i+1 - P_i
                segments = np.diff(fiber_points_fbr_space, axis=0)  # Shape: (num_points-1, 3)
                segment_norms = np.linalg.norm(segments, axis=1)  # Shape: (num_points-1,)

                # Identify valid segments (non-zero norm)
                valid_segment_mask = segment_norms > 1e-9

                if np.any(valid_segment_mask):
                    valid_segments = segments[valid_segment_mask]
                    valid_segment_norms = segment_norms[valid_segment_mask]

                    # Calculate colors for these valid segments
                    unit_orientations = np.abs(valid_segments / valid_segment_norms[:, np.newaxis])
                    calculated_colors_for_valid_segments = np.clip(unit_orientations * 255.0, 0, 255).astype(np.uint8)

                    # Assign calculated colors:
                    # Point P_i gets color of segment (P_i, P_i+1)
                    # np.where(valid_segment_mask)[0] gives indices of valid segments.
                    # These correspond to the starting points of these segments.
                    point_indices_to_update = np.where(valid_segment_mask)[0]
                    point_colors_rgb[point_indices_to_update] = calculated_colors_for_valid_segments

                # The last point (P_N-1) takes the color of the segment leading to it (P_N-2, P_N-1).
                # This segment's color is already assigned to P_N-2.
                point_colors_rgb[num_points - 1] = point_colors_rgb[num_points - 2]

                r_channel_values = point_colors_rgb[:, 0].tolist()
                g_channel_values = point_colors_rgb[:, 1].tolist()
                b_channel_values = point_colors_rgb[:, 2].tolist()

            else:  # Default coloring (scheme is not 'orientation', OR num_points is 1)
                # All points get the default color.
                # num_points is guaranteed to be >= 1 here.
                point_colors_rgb = np.tile(_default_point_color_arr_uint8, (num_points, 1))
                r_channel_values = point_colors_rgb[:, 0].tolist()
                g_channel_values = point_colors_rgb[:, 1].tolist()
                b_channel_values = point_colors_rgb[:, 2].tolist()

            # Pack and write R colors
            f.write(struct.pack(f'<{num_points}B', *r_channel_values))
            # Pack and write G colors
            f.write(struct.pack(f'<{num_points}B', *g_channel_values))
            # Pack and write B colors
            f.write(struct.pack(f'<{num_points}B', *b_channel_values))

            if IS_GUI_AVAILABLE and fiber_idx % 100 == 0:  # Keep GUI responsive
                QApplication.process_events()

    log_callback(f"Successfully wrote {num_fibers_in_group} fibers to {fbr_filename}.")
    return True


# --- Conversion Logic: Export FBR to TCK/TRK ---
def export_fbr_to_tcktrk_logic(fbr_filename, tcktrk_filename, output_format, params, log_callback):
    log_callback(f"Starting export: {fbr_filename} -> {tcktrk_filename} (Format: {output_format})")
    try:
        fbr_obj = FBRFile(fbr_filename).read()
    except Exception as e:
        log_callback(f"Error reading FBR file: {e}")
        return False

    output_streamlines = []
    skipped_fibers_count = 0
    total_fibers_processed = 0

    for group in fbr_obj.groups:
        log_callback(f"Processing group: {group['name']} with {len(group['fibers'])} fibers.")
        for fiber_data in group['fibers']:
            total_fibers_processed += 1
            if len(fiber_data['points']) < 2:  # TCK/TRK generally needs >= 2 points
                skipped_fibers_count += 1
                continue

            tck_all = np.array(fiber_data['points'], dtype=np.float32)
            tck_x, tck_y, tck_z = tck_all[:, 0], tck_all[:, 1], tck_all[:, 2]

            streamline_tck_space = np.column_stack((tck_x, tck_y, tck_z)) - params['coord_offset']
            output_streamlines.append(streamline_tck_space.astype(np.float32))

            if IS_GUI_AVAILABLE and total_fibers_processed % 100 == 0:  # Keep GUI responsive
                QApplication.process_events()

    if skipped_fibers_count > 0:
        log_callback(f"Skipped {skipped_fibers_count} fibers with < 2 points.")
    log_callback(f"Extracted {len(output_streamlines)} streamlines for TCK/TRK output.")

    if not output_streamlines:
        log_callback("No valid streamlines to save.")
        return False

    nifti_affine_path = params.get('nifti_affine_path')  # Fetched from GUI
    if nifti_affine_path and os.path.exists(nifti_affine_path):
        try:
            log_callback(f"Attempting to load affine and header from NIFTI: {nifti_affine_path}")
            nifti_img = nib.load(nifti_affine_path)
            affine = np.eye(4)
            log_callback(f"Successfully loaded affine and header from {nifti_affine_path}.")

            header = {}
            if output_format.lower() == "trk":
                log_callback(f"Affine to be stored in TRK header:\n{nifti_img.affine}")
                header[Field.VOXEL_TO_RASMM] = nifti_img.affine.copy()
                header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nifti_img.affine))
            elif output_format.lower() == "tck":
                header[Field.VOXEL_TO_RASMM] = np.eye(4)
                header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nifti_img.affine)) # Maybe a mistake, to investigate
            voxel_sizes_np = nifti_img.header.get_zooms()[:3]
            header[Field.VOXEL_SIZES] = tuple(float(x) for x in voxel_sizes_np)
            dimensions_np = nifti_img.shape[:3]
            header[Field.DIMENSIONS] = tuple(int(x) for x in dimensions_np)

        except Exception as e:
            log_callback(f"Error loading NIFTI file '{nifti_affine_path}': {e}")
            return False
    elif nifti_affine_path:
        log_callback(f"NIFTI file for affine and header not found: '{nifti_affine_path}'.")
        return False
    else:
        log_callback("No NIFTI file specified for affine and header.")
        return False

    tractogram = Tractogram(output_streamlines, affine_to_rasmm=affine)

    try:
        nib.streamlines.save(tractogram, tcktrk_filename, header=header)
        log_callback(f"Successfully wrote {output_format.upper()} file: {tcktrk_filename}")
        return True
    except Exception as e:
        log_callback(f"Error saving {output_format.upper()} file: {e}")
        return False


# --- Main GUI Dialog Class ---
if IS_GUI_AVAILABLE:
    class TCKTRKConverterDialog(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.set_window_title("TCK/TRK <-> FBR Converter")
            if IS_BRAINVOYAGER_ENV:
                self.minimum_width = 600
            else:
                self.set_minimum_width(600)
            self.init_ui()

        def _create_color_button(self, initial_color_tuple=(200, 200, 200)):
            button = QPushButton()
            if IS_BRAINVOYAGER_ENV:
                button.minimum_height = 25
            else:
                button.set_minimum_height(25)
            q_color = QColor(*initial_color_tuple)
            button.set_style_sheet(f"background-color: {q_color.name()}")
            button.clicked.connect(lambda: self._pick_color(button))
            button.set_property("color_tuple", initial_color_tuple)
            return button

        def _pick_color(self, button):
            current_color_tuple = button.property("color_tuple")
            color = QColorDialog.get_color(QColor(*current_color_tuple), self, "Select Color")
            if color.is_valid():
                new_color_tuple = (color.red(), color.green(), color.blue())
                button.set_style_sheet(f"background-color: {color.name()}")
                button.set_property("color_tuple", new_color_tuple)

        def init_ui(self):
            main_layout = QVBoxLayout(self)

            # Mode Selection
            mode_group = QGroupBox("Conversion Mode")
            mode_layout = QHBoxLayout()
            self.import_radio = QRadioButton("Import (TCK/TRK -> FBR)")
            self.export_radio = QRadioButton("Export (FBR -> TCK/TRK)")
            self.import_radio.set_checked(True)
            mode_layout.add_widget(self.import_radio)
            mode_layout.add_widget(self.export_radio)
            mode_group.set_layout(mode_layout)
            main_layout.add_widget(mode_group)

            self.mode_button_group = QButtonGroup(self)
            self.mode_button_group.add_button(self.import_radio, 0)
            self.mode_button_group.add_button(self.export_radio, 1)
            self.mode_button_group.idClicked.connect(self._update_ui_for_mode)

            # Export Format (Only for export mode)
            self.export_format_group = QGroupBox("Export Format")
            export_format_layout = QHBoxLayout()
            self.tck_radio = QRadioButton("TCK")
            self.trk_radio = QRadioButton("TRK")
            self.tck_radio.set_checked(True)
            export_format_layout.add_widget(self.tck_radio)
            export_format_layout.add_widget(self.trk_radio)
            self.export_format_group.set_layout(export_format_layout)
            main_layout.add_widget(self.export_format_group)
            self.tck_radio.toggled.connect(self._update_trk_affine_ui_visibility)
            self.trk_radio.toggled.connect(self._update_trk_affine_ui_visibility)

            # NIFTI affine selection UI
            self.nifti_affine_group = QGroupBox("Header from NIFTI")
            nifti_affine_layout = QHBoxLayout()
            self.nifti_affine_edit = QLineEdit()
            if IS_BRAINVOYAGER_ENV:
                self.nifti_affine_edit.placeholder_text = "Path to reference .nii or .nii.gz for TRK affine and TCK/TRK header"
            else:
                self.nifti_affine_edit.set_placeholder_text("Path to reference .nii or .nii.gz for TRK affine and TCK/TRK header")
            self.nifti_affine_button = QPushButton("Browse...")
            self.nifti_affine_button.clicked.connect(self._select_nifti_affine_file)
            nifti_affine_layout.add_widget(QLabel("NIFTI File:"))
            nifti_affine_layout.add_widget(self.nifti_affine_edit)
            nifti_affine_layout.add_widget(self.nifti_affine_button)
            self.nifti_affine_group.set_layout(nifti_affine_layout)
            if IS_BRAINVOYAGER_ENV:
                self.nifti_affine_group.visible = False
            else:
                self.nifti_affine_group.set_visible(False)
            main_layout.add_widget(self.nifti_affine_group)



            # Conversion FBR->TCK/TRK Parameters
            self.fbr_export_params_group = QGroupBox("FBR Parameters (for Export to TCK/TRK)")
            fbr_export_params_layout = QVBoxLayout()

            row_offset_export = QHBoxLayout()  # offset for FBR -> TCK/TRK
            row_offset_export.add_widget(QLabel("Coordinate Offset (X,Y,Z):"))
            self.offset_x_export_edit = QLineEdit("0.0")
            self.offset_y_export_edit = QLineEdit("0.0")
            self.offset_z_export_edit = QLineEdit("0.0")
            row_offset_export.add_widget(self.offset_x_export_edit)
            row_offset_export.add_widget(self.offset_y_export_edit)
            row_offset_export.add_widget(self.offset_z_export_edit)
            fbr_export_params_layout.add_layout(row_offset_export)

            self.fbr_export_params_group.set_layout(fbr_export_params_layout)
            main_layout.add_widget(self.fbr_export_params_group)


            # FBR Parameters
            self.fbr_params_group = QGroupBox("FBR Parameters (for Import to FBR)")
            fbr_params_layout = QVBoxLayout()

            row1 = QHBoxLayout()
            row1.add_widget(QLabel("File Version:"))
            self.file_version_edit = QLineEdit("5")
            row1.add_widget(self.file_version_edit)
            row1.add_widget(QLabel("Coords Type:"))
            self.coords_type_combo = QComboBox()
            self.coords_type_combo.add_items(["0 (Talairach)", "1 (MNI)", "2 (Native/Voxel)"])
            if IS_BRAINVOYAGER_ENV:
                self.coords_type_combo.current_index = 2  # Default to Native/Voxel
            else:
                self.coords_type_combo.set_current_index(2)  # Default to Native/Voxel
            row1.add_widget(self.coords_type_combo)
            fbr_params_layout.add_layout(row1)

            row_origin = QHBoxLayout()
            row_origin.add_widget(QLabel("Fibers Origin (X,Y,Z):"))
            self.origin_x_edit = QLineEdit("0.0")
            self.origin_y_edit = QLineEdit("0.0")
            self.origin_z_edit = QLineEdit("0.0")
            row_origin.add_widget(self.origin_x_edit)
            row_origin.add_widget(self.origin_y_edit)
            row_origin.add_widget(self.origin_z_edit)
            fbr_params_layout.add_layout(row_origin)

            row_offset = QHBoxLayout()  # offset for TCK/TRK -> FBR (and reversed for FBR -> TCK/TRK)
            row_offset.add_widget(QLabel("Coordinate Offset (X,Y,Z):"))
            self.offset_x_edit = QLineEdit("0.0")
            self.offset_y_edit = QLineEdit("0.0")
            self.offset_z_edit = QLineEdit("0.0")
            row_offset.add_widget(self.offset_x_edit)
            row_offset.add_widget(self.offset_y_edit)
            row_offset.add_widget(self.offset_z_edit)
            fbr_params_layout.add_layout(row_offset)

            row2 = QHBoxLayout()
            row2.add_widget(QLabel("Group Name:"))
            self.group_name_edit = QLineEdit("ImportedTracks")
            row2.add_widget(self.group_name_edit)
            self.group_visible_check = QCheckBox("Visible")
            self.group_visible_check.set_checked(True)
            row2.add_widget(self.group_visible_check)
            self.group_animate_check = QCheckBox("Animate")
            row2.add_widget(self.group_animate_check)
            fbr_params_layout.add_layout(row2)

            row3 = QHBoxLayout()
            row3.add_widget(QLabel("Group Thickness:"))
            self.group_thickness_spin = QDoubleSpinBox()
            if IS_BRAINVOYAGER_ENV:
                self.group_thickness_spin.decimals = 2
                self.group_thickness_spin.single_step = 0.1
                self.group_thickness_spin.value = 0.3
            else:
                self.group_thickness_spin.set_decimals(2)
                self.group_thickness_spin.set_single_step(0.1)
                self.group_thickness_spin.set_value(0.3)
            row3.add_widget(self.group_thickness_spin)
            row3.add_widget(QLabel("Group Color:"))
            self.group_color_button = self._create_color_button((255, 255, 255))
            row3.add_widget(self.group_color_button)
            fbr_params_layout.add_layout(row3)

            row4 = QHBoxLayout()
            row4.add_widget(QLabel("Point Color Scheme:"))
            self.color_scheme_combo = QComboBox()
            self.color_scheme_combo.add_items(["orientation", "default"])
            self.color_scheme_combo.currentTextChanged.connect(self._update_point_color_visibility)
            row4.add_widget(self.color_scheme_combo)
            self.default_point_color_label = QLabel("Default Point Color:")
            row4.add_widget(self.default_point_color_label)
            self.default_point_color_button = self._create_color_button((200, 200, 200))
            row4.add_widget(self.default_point_color_button)
            fbr_params_layout.add_layout(row4)

            self.fbr_params_group.set_layout(fbr_params_layout)
            main_layout.add_widget(self.fbr_params_group)

            # File Selection Import
            self.import_files_group = QGroupBox("Input/Output Files")
            import_file_layout = QVBoxLayout()
            self.import_input_file_edit = QLineEdit()
            self.import_input_file_button = QPushButton("Browse...")
            self.import_input_file_button.clicked.connect(self._select_input_file)
            import_input_row = QHBoxLayout()
            import_input_row.add_widget(QLabel("Input File:"))
            import_input_row.add_widget(self.import_input_file_edit)
            import_input_row.add_widget(self.import_input_file_button)
            import_file_layout.add_layout(import_input_row)

            self.import_output_file_edit = QLineEdit()
            self.import_output_file_button = QPushButton("Browse...")
            self.import_output_file_button.clicked.connect(self._select_output_file)
            import_output_row = QHBoxLayout()
            import_output_row.add_widget(QLabel("Output File:"))
            import_output_row.add_widget(self.import_output_file_edit)
            import_output_row.add_widget(self.import_output_file_button)
            import_file_layout.add_layout(import_output_row)

            if IS_BRAINVOYAGER_ENV:
                self.import_input_file_edit.placeholder_text = "Select TCK or TRK file"
                self.import_output_file_edit.placeholder_text = "Enter FBR output name"
            else:
                self.import_input_file_edit.set_placeholder_text("Select TCK or TRK file")
                self.import_output_file_edit.set_placeholder_text("Enter FBR output name")

            self.import_files_group.set_layout(import_file_layout)
            main_layout.add_widget(self.import_files_group)

            # File Selection Export
            self.export_files_group = QGroupBox("Input/Output Files")
            export_file_layout = QVBoxLayout()
            self.export_input_file_edit = QLineEdit()
            self.export_input_file_button = QPushButton("Browse...")
            self.export_input_file_button.clicked.connect(self._select_input_file)
            export_input_row = QHBoxLayout()
            export_input_row.add_widget(QLabel("Input File:"))
            export_input_row.add_widget(self.export_input_file_edit)
            export_input_row.add_widget(self.export_input_file_button)
            export_file_layout.add_layout(export_input_row)

            self.export_output_file_edit = QLineEdit()
            self.export_output_file_button = QPushButton("Browse...")
            self.export_output_file_button.clicked.connect(self._select_output_file)
            export_output_row = QHBoxLayout()
            export_output_row.add_widget(QLabel("Output File:"))
            export_output_row.add_widget(self.export_output_file_edit)
            export_output_row.add_widget(self.export_output_file_button)
            export_file_layout.add_layout(export_output_row)

            if IS_BRAINVOYAGER_ENV:
                self.export_input_file_edit.placeholder_text = "Select FBR file"
                self.export_output_file_edit.placeholder_text = "Enter TCK/TRK output name"
            else:
                self.export_input_file_edit.set_placeholder_text("Select FBR file")
                self.export_output_file_edit.set_placeholder_text("Enter TCK/TRK output name")

            self.export_files_group.set_layout(export_file_layout)
            main_layout.add_widget(self.export_files_group)

            # Convert Button
            self.convert_button = QPushButton("Convert")
            self.convert_button.clicked.connect(self._convert_files)
            main_layout.add_widget(self.convert_button)

            # Status Log
            self.status_log = QTextEdit()
            if IS_BRAINVOYAGER_ENV:
                self.status_log.read_only = True
            else:
                self.status_log.set_read_only(True)
            main_layout.add_widget(self.status_log)

            self._update_ui_for_mode()  # Initial UI setup based on mode
            self._update_point_color_visibility()

        def _update_point_color_visibility(self, scheme=None):
            if IS_BRAINVOYAGER_ENV:
                if scheme is None:
                    scheme = self.color_scheme_combo.current_text
                is_default_scheme = (scheme == "default")
                self.default_point_color_label.visible = is_default_scheme
                self.default_point_color_button.visible = is_default_scheme
            else:
                if scheme is None:
                    scheme = self.color_scheme_combo.current_text()
                is_default_scheme = (scheme == "default")
                self.default_point_color_label.set_visible(is_default_scheme)
                self.default_point_color_button.set_visible(is_default_scheme)

        def _select_nifti_affine_file(self):
            filename, _ = QFileDialog.get_open_file_name(
                self, "Select NIFTI File for TRK Affine", "",
                "NIFTI Files (*.nii *.nii.gz);;All Files (*)"
            )
            if filename:
                if IS_BRAINVOYAGER_ENV:
                    self.nifti_affine_edit.text = filename
                else:
                    self.nifti_affine_edit.set_text(filename)

        def _update_trk_affine_ui_visibility(self):
            if IS_BRAINVOYAGER_ENV:
                is_export_mode = self.export_radio.checked
                is_trk_format = self.trk_radio.checked
                self.nifti_affine_group.set_enabled(is_export_mode)# and is_trk_format)
                self.nifti_affine_group.visible = is_export_mode# and is_trk_format
            else:
                is_export_mode = self.export_radio.is_checked()
                is_trk_format = self.trk_radio.is_checked()
                self.nifti_affine_group.set_enabled(is_export_mode)# and is_trk_format)
                self.nifti_affine_group.set_visible(is_export_mode)# and is_trk_format)

        def _log(self, message):
            self.status_log.append(message)
            QApplication.process_events()  # Keep GUI responsive

        def _update_ui_for_mode(self):
            if IS_BRAINVOYAGER_ENV:
                is_import_mode = self.import_radio.checked
                self.fbr_params_group.set_enabled(is_import_mode)
                self.fbr_params_group.visible = is_import_mode
                self.export_format_group.set_enabled(not is_import_mode)
                self.export_format_group.visible = not is_import_mode
                self.fbr_export_params_group.set_enabled(not is_import_mode)
                self.fbr_export_params_group.visible = not is_import_mode
                self.import_files_group.visible = is_import_mode
                self.export_files_group.visible = not is_import_mode
            else:
                is_import_mode = self.import_radio.is_checked()
                self.fbr_params_group.set_enabled(is_import_mode)
                self.fbr_params_group.set_visible(is_import_mode)
                self.export_format_group.set_enabled(not is_import_mode)
                self.export_format_group.set_visible(not is_import_mode)
                self.fbr_export_params_group.set_enabled(not is_import_mode)
                self.fbr_export_params_group.set_visible(not is_import_mode)
                self.import_files_group.set_visible(is_import_mode)
                self.export_files_group.set_visible(not is_import_mode)
            self.set_window_title("TCK/TRK -> FBR Converter" if is_import_mode else "FBR -> TCK/TRK Converter")
            self._update_trk_affine_ui_visibility()

        def _select_input_file(self):
            if IS_BRAINVOYAGER_ENV:
                is_import_mode = self.import_radio.checked
            else:
                is_import_mode = self.import_radio.is_checked()
            caption = "Select Input TCK/TRK File" if is_import_mode else "Select Input FBR File"
            filter_str = "Track Files (*.tck *.trk);;All Files (*)" if is_import_mode else "Fiber Files (*.fbr);;All Files (*)"
            filename, _ = QFileDialog.get_open_file_name(self, caption, "", filter_str)
            if filename:

                if is_import_mode:
                    if IS_BRAINVOYAGER_ENV:
                        self.import_input_file_edit.text = filename
                    else:
                        self.import_input_file_edit.set_text(filename)
                    # Auto-suggest output filename
                    base, ext = os.path.splitext(filename)
                    if IS_BRAINVOYAGER_ENV:
                        self.import_output_file_edit.text = base + "_imported.fbr"
                    else:
                        self.import_output_file_edit.set_text(base + "_imported.fbr")
                else:
                    if IS_BRAINVOYAGER_ENV:
                        self.export_input_file_edit.text = filename
                    else:
                        self.export_input_file_edit.set_text(filename)
                    # Auto-suggest output filename
                    base, ext = os.path.splitext(filename)
                    if IS_BRAINVOYAGER_ENV:
                        default_out_ext = ".tck" if self.tck_radio.checked else ".trk"
                        self.export_output_file_edit.text = base + "_exported" + default_out_ext
                    else:
                        default_out_ext = ".tck" if self.tck_radio.is_checked() else ".trk"
                        self.export_output_file_edit.set_text(base + "_exported" + default_out_ext)

        def _select_output_file(self):
            if IS_BRAINVOYAGER_ENV:
                is_import_mode = self.import_radio.checked
                caption = "Save Output FBR File" if is_import_mode else "Save Output TCK/TRK File"
                default_ext = ".fbr" if is_import_mode else (".tck" if self.tck_radio.checked else ".trk")
                filter_str = f"Fiber Files (*{default_ext});;All Files (*)"
            else:
                is_import_mode = self.import_radio.is_checked()
                caption = "Save Output FBR File" if is_import_mode else "Save Output TCK/TRK File"
                default_ext = ".fbr" if is_import_mode else (".tck" if self.tck_radio.is_checked() else ".trk")
                filter_str = f"Fiber Files (*{default_ext});;All Files (*)"


            # Suggest a name based on input if output is empty
            if IS_BRAINVOYAGER_ENV:
                if is_import_mode:
                    suggested_path = self.import_output_file_edit.text
                    if not suggested_path and self.import_input_file_edit.text:
                        base, _ = os.path.splitext(self.import_input_file_edit.text)
                        suggested_path = (base + default_ext) if is_import_mode else (base + "_exported" + default_ext)
                else:
                    suggested_path = self.export_output_file_edit.text
                    if not suggested_path and self.export_input_file_edit.text:
                        base, _ = os.path.splitext(self.export_input_file_edit.text)
                        suggested_path = (base + default_ext) if is_import_mode else (base + "_exported" + default_ext)
            else:
                if is_import_mode:
                    suggested_path = self.import_output_file_edit.text()
                    if not suggested_path and self.import_input_file_edit.text():
                        base, _ = os.path.splitext(self.import_input_file_edit.text())
                        suggested_path = (base + default_ext) if is_import_mode else (base + "_exported" + default_ext)
                else:
                    suggested_path = self.export_output_file_edit.text()
                    if not suggested_path and self.export_input_file_edit.text():
                        base, _ = os.path.splitext(self.export_input_file_edit.text())
                        suggested_path = (base + default_ext) if is_import_mode else (base + "_exported" + default_ext)
            filename, _ = QFileDialog.get_save_file_name(self, caption, suggested_path, filter_str)
            if filename:
                # Ensure correct extension
                if not filename.lower().endswith(default_ext.lower()):
                    filename += default_ext
                if is_import_mode:
                    if IS_BRAINVOYAGER_ENV:
                        self.import_output_file_edit.text = filename
                    else:
                        self.import_output_file_edit.set_text(filename)
                else:
                    if IS_BRAINVOYAGER_ENV:
                        self.export_output_file_edit.text = filename
                    else:
                        self.export_output_file_edit.set_text(filename)

        def _gather_params(self):
            params = {}
            # FBR specific params (used for import, some might be relevant for export defaults if needed)
            params['magic'] = b'\xa4\xd3\xc2\xb1'
            if IS_BRAINVOYAGER_ENV:
                params['file_version'] = int(self.file_version_edit.text)
                params['coords_type'] = self.coords_type_combo.current_index  # 0, 1, or 2
                params['fibers_origin'] = (
                    float(self.origin_x_edit.text),
                    float(self.origin_y_edit.text),
                    float(self.origin_z_edit.text)
                )
                if self.import_radio.checked:
                    # Universal coordinate offset
                    params['coord_offset'] = np.array([
                        float(self.offset_x_edit.text),
                        float(self.offset_y_edit.text),
                        float(self.offset_z_edit.text)
                    ], dtype=np.float32)
                else:
                    # Universal coordinate offset
                    params['coord_offset'] = np.array([
                        float(self.offset_x_export_edit.text),
                        float(self.offset_y_export_edit.text),
                        float(self.offset_z_export_edit.text)
                    ], dtype=np.float32)

                params['nifti_affine_path'] = None
                if self.nifti_affine_group.visible:  # Check actual visibility
                    nifti_path_text = self.nifti_affine_edit.text.strip()
                    if nifti_path_text:
                        params['nifti_affine_path'] = nifti_path_text

                params['group_name'] = self.group_name_edit.text
                params['group_visible'] = 1 if self.group_visible_check.checked else 0
                params['group_animate'] = 1 if self.group_animate_check.checked else 0  # Should be signed int
                params['group_thickness'] = self.group_thickness_spin.value
                params['group_color'] = self.group_color_button.property("color_tuple")
                params['color_scheme'] = self.color_scheme_combo.current_text
                params['default_point_color'] = self.default_point_color_button.property("color_tuple")


            else:
                params['file_version'] = int(self.file_version_edit.text())
                params['coords_type'] = self.coords_type_combo.current_index()  # 0, 1, or 2
                params['fibers_origin'] = (
                    float(self.origin_x_edit.text()),
                    float(self.origin_y_edit.text()),
                    float(self.origin_z_edit.text())
                )
                if self.import_radio.is_checked():
                    # Universal coordinate offset
                    params['coord_offset'] = np.array([
                        float(self.offset_x_edit.text()),
                        float(self.offset_y_edit.text()),
                        float(self.offset_z_edit.text())
                    ], dtype=np.float32)
                else:
                    # Universal coordinate offset
                    params['coord_offset'] = np.array([
                        float(self.offset_x_export_edit.text()),
                        float(self.offset_y_export_edit.text()),
                        float(self.offset_z_export_edit.text())
                    ], dtype=np.float32)

                params['nifti_affine_path'] = None
                if self.nifti_affine_group.is_visible():  # Check actual visibility
                    nifti_path_text = self.nifti_affine_edit.text().strip()
                    if nifti_path_text:
                        params['nifti_affine_path'] = nifti_path_text

                params['group_name'] = self.group_name_edit.text()
                params['group_visible'] = 1 if self.group_visible_check.is_checked() else 0
                params['group_animate'] = 1 if self.group_animate_check.is_checked() else 0  # Should be signed int
                params['group_thickness'] = self.group_thickness_spin.value()
                params['group_color'] = self.group_color_button.property("color_tuple")
                params['color_scheme'] = self.color_scheme_combo.current_text()
                params['default_point_color'] = self.default_point_color_button.property("color_tuple")

            return params

        def _convert_files(self):
            self.status_log.clear()
            params = self._gather_params()
            if (IS_BRAINVOYAGER_ENV and self.import_radio.checked) or (not IS_BRAINVOYAGER_ENV and self.import_radio.is_checked()):
                if IS_BRAINVOYAGER_ENV:
                    input_file = self.import_input_file_edit.text
                    output_file = self.import_output_file_edit.text
                else:
                    input_file = self.import_input_file_edit.text()
                    output_file = self.import_output_file_edit.text()
                if not input_file or not output_file:
                    self._log("Error: Input and Output files must be specified.")
                    return

                if not (input_file.lower().endswith(".tck") or input_file.lower().endswith(".trk")):
                    self._log("Error: Input for import mode must be a .tck or .trk file.")
                    return
                if not output_file.lower().endswith(".fbr"):
                    self._log("Warning: Output for import mode should ideally be .fbr. Appending if necessary.")
                    if '.' not in os.path.basename(output_file): output_file += ".fbr"  # simple append
                import_tcktrk_to_fbr_logic(input_file, output_file, params, self._log)
            else:  # Export mode
                if IS_BRAINVOYAGER_ENV:
                    input_file = self.export_input_file_edit.text
                    output_file = self.export_output_file_edit.text
                else:
                    input_file = self.export_input_file_edit.text()
                    output_file = self.export_output_file_edit.text()
                if not input_file or not output_file:
                    self._log("Error: Input and Output files must be specified.")
                    return

                if not input_file.lower().endswith(".fbr"):
                    self._log("Error: Input for export mode must be a .fbr file.")
                    return

                if IS_BRAINVOYAGER_ENV:
                    if self.nifti_affine_group.visible and not self.nifti_affine_edit.text:
                        self._log("Error: A NIFTI reference file must be specified when exporting in trk format.")
                        return

                    output_format = "tck" if self.tck_radio.checked else "trk"
                else:
                    if self.nifti_affine_group.is_visible() and not self.nifti_affine_edit.text():
                        self._log("Error: A NIFTI reference file must be specified when exporting in trk format.")
                        return

                    output_format = "tck" if self.tck_radio.is_checked() else "trk"


                expected_ext = "." + output_format
                if not output_file.lower().endswith(expected_ext):
                    self._log(
                        f"Warning: Output for export mode as {output_format.upper()} should ideally be {expected_ext}. Appending ...")
                    output_file += expected_ext
                export_fbr_to_tcktrk_logic(input_file, output_file, output_format, params, self._log)


if __name__ == '__main__':
    if not IS_GUI_AVAILABLE:
        print("Running in standalone mode.")
        # Basic test of logic if no GUI

        # Create dummy files and test functions directly
        # Dummy TCK for testing import
        dummy_tck_file = "dummy_test.tck"
        s1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype='f4')
        s2 = np.array([[3, 0, 0], [4, 1, 1], [5, 2, 2]], dtype='f4')
        tg = Tractogram([s1, s2], affine_to_rasmm=np.eye(4))
        nib.streamlines.save(tg, dummy_tck_file)
        print(f"Created {dummy_tck_file}")

        test_params_import = {
            'magic': b'\xa4\xd3\xc2\xb1', 'file_version': 5, 'coords_type': 2,
            'fibers_origin': (0.0, 0.0, 0.0), 'group_name': 'TestGroup',
            'group_visible': 1, 'group_animate': 0, 'group_thickness': 0.3,
            'group_color': (255, 0, 0), 'color_scheme': 'orientation',
            'default_point_color': (100, 100, 100),
            'coord_offset': np.array([0.0, 0.0, 0.0], dtype=np.float32)
        }
        dummy_fbr_file = "dummy_test.fbr"
        print(f"\nTesting Import Logic: {dummy_tck_file} -> {dummy_fbr_file}")
        success_import = import_tcktrk_to_fbr_logic(dummy_tck_file, dummy_fbr_file, test_params_import, print)

        if success_import:
            print(f"\nTesting Export Logic: {dummy_fbr_file} -> dummy_test_exported.tck")
            test_params_export = {
                'coord_offset': np.array([0.0, 0.0, 0.0], dtype=np.float32)
            }
            export_fbr_to_tcktrk_logic(dummy_fbr_file, "dummy_test_exported.tck", "tck", test_params_export, print)


    else:  # Running GUI
        app = QApplication.instance()  # Get the existing application instance
        if not app:  # Create if does not exist (should not happen in BV plugin context)
            app = QApplication(sys.argv)

        dialog = TCKTRKConverterDialog()
        if IS_BRAINVOYAGER_ENV:
            dialog.show()
        else:
            dialog.exec()
            sys.exit(app.exec())