import os
import sys
import yaml
import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from tqdm import tqdm
from scipy import integrate
import stat

# Toggle to generate PNGs alongside NPYs
GENERATE_PLOTS = True

def load_config():
    path = os.environ.get('SYSTEM_CONFIG_FILE')
    if not path:
        print("ERROR: SYSTEM_CONFIG_FILE not set.")
        sys.exit(1)
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def _slug_float(val, digits=4):
    try:
        return f"{float(val):.{digits}g}"
    except Exception:
        return str(val)

def _build_results_folder(config):
    """
    Creates the verbose folder name based on system, physics, grid, time, and field mode.
    """
    base = config['output_base_folder']
    parts = ["final_results"]
    systod = config.get('system_module', 'unknown').split('.')[-1]
    parts.append(systod)
    
    # Physics Params
    phys = config.get('physics_params', {})
    if phys:
        priority = ["A", "epsilon", "omega", "sigma", "rho", "beta", "alpha", "gamma"]
        p_parts = [f"{k}{_slug_float(phys[k])}" for k in priority if k in phys]
        p_parts += [f"{k}{_slug_float(phys[k])}" for k in sorted(phys.keys()) 
                    if k not in priority and isinstance(phys[k], (int, float))]
        if p_parts: parts.append("_".join(p_parts))

    # Grid Params
    grid = config.get('grid_params', {})
    if grid:
        parts.append(f"nx{grid.get('grid_resolution_x')}ny{grid.get('grid_resolution_y')}")
        
    # Time Params
    tp = config.get('time_params', {})
    if tp:
        parts.append(f"T{_slug_float(tp.get('total_integration_time_end', 0))}")
    
    # Field Mode - check for environment override first
    field_mode_override = os.environ.get("FIELD_MODE_OVERRIDE")
    if field_mode_override:
        field_mode = field_mode_override
    else:
        field_mode = config.get('field_mode', 'standard')
    
    # Add field mode suffix if not standard (to distinguish orthogonal results)
    if field_mode != 'standard':
        parts.append(f"field_{field_mode}")

    return os.path.join(base, "_".join(parts))

def _set_file_permissions(filepath):
    """Set file permissions to 644 (rw-r--r--)"""
    try:
        os.chmod(filepath, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    except Exception:
        pass  # Ignore permission errors

def _set_dir_permissions(dirpath):
    """Set directory permissions to 755 (rwxr-xr-x)"""
    try:
        os.chmod(dirpath, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    except Exception:
        pass  # Ignore permission errors

def _save_npy_with_permissions(filepath, data):
    """Save numpy array and set permissions to 644"""
    np.save(filepath, data)
    _set_file_permissions(filepath)

# ----------------------------------------------------------------------
# PLOTTING LOGIC 1: FINSTOD SEGMENTED (From Pipeline 1)
# ----------------------------------------------------------------------
def normalize_segmented(score_grid, type_grid):
    """
    Normalizes scores based on population segments:
    - T  -> [0, P]
    - UC -> [P, 0.99]
    - UU -> 1.0
    """
    mask_nodata = (type_grid == -1)
    out_grid = np.full(score_grid.shape, np.nan)
    out_grid[mask_nodata] = np.nan

    valid_mask = ~mask_nodata
    if not np.any(valid_mask):
        return np.ma.masked_invalid(out_grid), 0.5

    flat_scores = score_grid[valid_mask]
    flat_types = type_grid[valid_mask]

    t_mask  = (flat_types == 0)
    uc_mask = (flat_types == 1)

    count_t  = np.sum(t_mask)
    count_uc = np.sum(uc_mask)
    total_non_uu = count_t + count_uc

    # Calculate Pivot P
    if total_non_uu == 0:
        P = 0.5
    else:
        P = count_t / total_non_uu
        P = max(0.05, min(0.95, P))

    # --- T region ---
    if count_t > 0:
        vals = flat_scores[t_mask]
        vmin, vmax = np.min(vals), np.max(vals)
        if vmax > vmin:
            norm_vals = (vals - vmin) / (vmax - vmin) * P
        else:
            norm_vals = np.full_like(vals, P / 2)
        out_grid[(type_grid == 0) & valid_mask] = norm_vals

    # --- UC region ---
    if count_uc > 0:
        vals = flat_scores[uc_mask]
        vmin, vmax = np.min(vals), np.max(vals)
        width = 0.99 - P
        if vmax > vmin:
            norm_vals = P + (vals - vmin) / (vmax - vmin) * width
        else:
            norm_vals = np.full_like(vals, P + width / 2)
        out_grid[(type_grid == 1) & valid_mask] = norm_vals

    # --- UU region ---
    out_grid[(type_grid == 2) & valid_mask] = 1.0

    return np.ma.masked_invalid(out_grid), P

def _compute_figure_size(grid_params, base_width=8):
    """
    Compute figure dimensions that respect the physical aspect ratio of the data.
    Returns (width, height) in inches.
    
    For systems with extreme aspect ratios (like LiCN with ~25:1), we use a
    golden ratio based approach for aesthetically pleasing figures rather than
    stretching to match the raw data aspect.
    """
    x_range = grid_params['x_max'] - grid_params['x_min']
    y_range = grid_params['y_max'] - grid_params['y_min']
    data_aspect = y_range / x_range  # height/width ratio of the data domain
    
    # Golden ratio for aesthetically pleasing proportions
    phi = (1 + 5**0.5) / 2  # ~1.618
    
    # For extreme aspect ratios (> 3 or < 0.33), use golden ratio
    # For moderate aspect ratios, use the actual data aspect
    if data_aspect > 3.0:
        # Very tall data (like LiCN): use golden ratio height
        visual_aspect = 1.0 / phi  # ~0.618, making figure landscape
    elif data_aspect < 0.33:
        # Very wide data: use golden ratio width
        visual_aspect = phi  # ~1.618
    else:
        # Normal aspect ratios: use actual data aspect, clamped to reasonable bounds
        max_aspect = 2.0
        min_aspect = 0.5
        visual_aspect = max(min_aspect, min(max_aspect, data_aspect))
    
    width = base_width
    height = base_width * visual_aspect
    return width, height


def save_segmented_plot(score_data, type_data, out_path, title, grid_params, is_log=False):
    """
    Generates the specific FINSTOD plot (Raw/Log) with T/UC/UU labels.
    """
    # Log transform if requested
    if is_log:
        proc_scores = np.log1p(score_data)
        title += " (Log Scale)"
    else:
        proc_scores = score_data
        title += " (Raw Scale)"

    norm_data, P = normalize_segmented(proc_scores, type_data)

    if np.all(np.isnan(norm_data)):
        return

    # Compute figure size based on data aspect ratio
    fig_width, fig_height = _compute_figure_size(grid_params)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    ax.set_facecolor('white')
    
    extent = [grid_params['x_min'], grid_params['x_max'], grid_params['y_min'], grid_params['y_max']]

    cmap = plt.colormaps.get_cmap("viridis").copy()
    cmap.set_bad(color="white")

    # Use aspect='auto' to let figure dimensions control the visual aspect ratio
    im = ax.imshow(norm_data, extent=extent, origin="lower", cmap=cmap, vmin=0, vmax=1, aspect='auto')

    # Create colorbar with same height as plot
    # Use shrink=1.0 to ensure colorbar matches axes height exactly
    cbar = plt.colorbar(im, ax=ax, shrink=1.0)
    cbar.set_label("Stability Index")
    
    # Adjust layout to remove extra blank space on the left
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(left=0.05, right=0.92, top=0.95, bottom=0.1)

    # Separator Line
    if 0 < P < 1:
        cbar.ax.hlines(P, 0, 1, colors='white', linewidth=1.5)

    # Labels on Colorbar
    text_effects = [pe.withStroke(linewidth=2.5, foreground='black')]
    
    # UU
    cbar.ax.text(2.5, 1.0, "UU", va='center', ha='left', fontsize=10, color='black')

    # UC
    if np.any(type_data == 1):
        y_uc = P + (1.0 - P) / 2.0
        txt = cbar.ax.text(0.5, y_uc, "UC", va='center', ha='center', fontsize=9, fontweight='bold', color='white')
        txt.set_path_effects(text_effects)

    # T
    if np.any(type_data == 0):
        y_t = P / 2.0
        txt = cbar.ax.text(0.5, y_t, "T", va='center', ha='center', fontsize=9, fontweight='bold', color='white')
        txt.set_path_effects(text_effects)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path)
    _set_file_permissions(out_path)
    plt.close(fig)

# ----------------------------------------------------------------------
# PLOTTING LOGIC 2: GENERIC HEATMAP (FTLE / LB)
# ----------------------------------------------------------------------
def plot_generic_heatmap(data, title, filename, folder, grid_params, plot_mode="standard", is_log=False):
    x_min, x_max = grid_params['x_min'], grid_params['x_max']
    y_min, y_max = grid_params['y_min'], grid_params['y_max']
    
    # Compute figure size based on data aspect ratio
    fig_width, fig_height = _compute_figure_size(grid_params, base_width=6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

    if plot_mode == "lb":
        if is_log:
            # Log scale version
            log_data = np.log(data + 1e-10)
            mn, mx = np.min(log_data), np.max(log_data)
            norm_data = (log_data - mn) / (mx - mn) if mx > mn else np.zeros_like(log_data)
            im = ax.imshow(norm_data, extent=[x_min, x_max, y_min, y_max], 
                          cmap="jet", aspect='auto', origin="lower")
            label = "Normalized Logarithmic LB (Log Scale)"
            title += " (Log Scale)"
        else:
            # Exact match to "Replica LB Paper.py" style
            # Logarithmic normalization (as per Figure 4 caption)
            log_data = np.log(data + 1e-10)
            mn, mx = np.min(log_data), np.max(log_data)
            norm_data = (log_data - mn) / (mx - mn) if mx > mn else np.zeros_like(log_data)
            im = ax.imshow(norm_data, extent=[x_min, x_max, y_min, y_max], 
                          cmap="jet", aspect='auto', origin="lower", 
                          vmin=0.0, vmax=0.45)
            label = "Normalized Logarithmic Sum of Lagrangian Betweenness"
            title += " (Raw Scale)"
        
    elif plot_mode == "difference":
        if is_log:
            # Log scale version of difference
            log_data = np.log(np.abs(data) + 1e-10)
            # Preserve sign
            sign_data = np.sign(data)
            log_signed = sign_data * log_data
            abs_max = np.max(np.abs(log_signed)) if np.max(np.abs(log_signed)) > 0 else 1.0
            # Normalize to [-1, 1] range
            norm_data = log_signed / abs_max if abs_max > 0 else log_signed
            im = ax.imshow(norm_data, extent=[x_min, x_max, y_min, y_max], 
                          cmap="RdBu_r", aspect='auto', origin="lower", 
                          vmin=-1.0, vmax=1.0)
            label = "Normalized Difference (Fwd - Bwd) (Log Scale)"
            title += " (Log Scale)"
        else:
            # Normalize difference data to [-1, 1] range
            abs_max = np.max(np.abs(data)) if np.max(np.abs(data)) > 0 else 1.0
            norm_data = data / abs_max if abs_max > 0 else data
            im = ax.imshow(norm_data, extent=[x_min, x_max, y_min, y_max], 
                          cmap="RdBu_r", aspect='auto', origin="lower", 
                          vmin=-1.0, vmax=1.0)
            label = "Normalized Difference (Fwd - Bwd)"
            title += " (Raw Scale)"
        
    else:
        # Standard FTLE (Viridis)
        if is_log:
            log_data = np.log(data - np.nanmin(data) + 1e-10)
            mn, mx = np.nanmin(log_data), np.nanmax(log_data)
            norm_data = (log_data - mn) / (mx - mn) if mx > mn else np.zeros_like(log_data)
            im = ax.imshow(norm_data, extent=[x_min, x_max, y_min, y_max], 
                          cmap="viridis", aspect='auto', origin="lower")
            label = "Normalized Value (Log Scale)"
            title += " (Log Scale)"
        else:
            mn, mx = np.nanmin(data), np.nanmax(data)
            norm_data = (data - mn) / (mx - mn) if mx > mn else np.zeros_like(data)
            im = ax.imshow(norm_data, extent=[x_min, x_max, y_min, y_max], 
                          cmap="viridis", aspect='auto', origin="lower")
            label = "Normalized Value"
            title += " (Raw Scale)"

    # Create colorbar with same height as plot
    # Use shrink=1.0 to ensure colorbar matches axes height exactly
    cbar = fig.colorbar(im, ax=ax, label=label, shrink=1.0)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Adjust layout to remove extra blank space on the left
    # Note: tight_layout should be called before subplots_adjust for best results
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(left=0.05, right=0.92, top=0.95, bottom=0.1)
    
    out_path = os.path.join(folder, filename)
    plt.savefig(out_path, bbox_inches="tight")
    _set_file_permissions(out_path)
    plt.close(fig)

# ----------------------------------------------------------------------
# Note: OLB calculation has been moved to C++ backend (calc_metrics.cpp)
# OLB is now computed on-the-fly at each time slice, matching LB's approach
# The old Python-based snapshot interpolation method has been removed
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# MAIN AGGREGATION
# ----------------------------------------------------------------------
def main():
    print("=" * 80)
    print("--- Aggregating Results (Holistic Pipeline) ---")
    print("=" * 80)
    cfg = load_config()
    
    # Check for field mode override
    field_mode_override = os.environ.get("FIELD_MODE_OVERRIDE")
    if field_mode_override:
        print(f"[INFO] Field Mode Override: {field_mode_override}")
    else:
        field_mode = cfg.get('field_mode', 'standard')
        print(f"[INFO] Field Mode (from config): {field_mode}")
    
    # 1. Setup Folders
    partial_dir = os.path.join(cfg['output_base_folder'], "partial_results")
    final_base_dir = _build_results_folder(cfg)
    os.makedirs(final_base_dir, exist_ok=True)
    _set_dir_permissions(final_base_dir)
    print(f"Output Base Directory: {final_base_dir}")
    
    gp = cfg['grid_params']
    ny = int(gp['grid_resolution_y'])
    nx = int(gp['grid_resolution_x'])
    snaps = cfg['time_params']['snapshot_times']
    tau = cfg['time_params']['total_integration_time_end']
    
    # Get analysis controls
    ac = cfg.get('analysis_controls', {})
    
    # 2. Aggregate snapshot-based metrics (Local and Global separately)
    print("\n--- Loading Snapshot Data ---")
    all_local_snap_data = {}
    all_global_snap_data = {}
    
    for t_snap in tqdm(snaps, desc="Loading Snapshots", unit="snap"):
        t_val = float(t_snap)
        
        # Load LOCAL metrics (from files without "global" in name)
        # IMPORTANT: Must exclude files with "global" in the name to avoid confusion
        local_pattern = f"results_chunk_*_snap_{t_val:.2f}.h5"
        all_local_candidates = glob.glob(os.path.join(partial_dir, local_pattern))
        # Filter out any files that contain "global" in the name (shouldn't happen, but safety check)
        local_files = [f for f in all_local_candidates if "_global_" not in os.path.basename(f)]
        local_files.sort(key=lambda x: int(os.path.basename(x).split('chunk_')[1].split('_')[0]))
        
        if local_files:
            local_metrics = ['fwd_ftle', 'bwd_ftle', 
                           'fwd_stod_score', 'fwd_stod_type', 'fwd_finstod_score', 'fwd_finstod_type',
                           'bwd_stod_score', 'bwd_stod_type', 'bwd_finstod_score', 'bwd_finstod_type',
                           'fwd_fli', 'bwd_fli', 'fwd_ld', 'bwd_ld']
            
            local_snap_data = {}
            for met in local_metrics:
                dtype = np.int32 if "type" in met else np.float64
                full_grid = np.zeros((ny, nx), dtype=dtype)
                current_row = 0
                has_data = False
                
                for fpath in local_files:
                    with h5py.File(fpath, 'r') as f:
                        if met in f:
                            chunk = f[met][:]
                            rows_in_chunk = chunk.shape[0]
                            cols_in_chunk = chunk.shape[1] if len(chunk.shape) > 1 else 1
                            
                            # Validate chunk dimensions
                            if cols_in_chunk != nx:
                                print(f"  [WARN] Chunk {os.path.basename(fpath)} has {cols_in_chunk} cols, expected {nx}")
                                continue
                            
                            # Only process if we haven't filled the grid yet
                            if current_row < ny:
                                end_row = min(current_row + rows_in_chunk, ny)
                                eff_rows = end_row - current_row
                                # Only take the rows we need
                                full_grid[current_row : end_row, :] = chunk[:eff_rows, :]
                                current_row += eff_rows
                                has_data = True
                            else:
                                print(f"  [WARN] Skipping chunk {os.path.basename(fpath)} - grid already full (current_row={current_row}, ny={ny})")
                
                if has_data:
                    local_snap_data[met] = full_grid
            
            if local_snap_data:
                all_local_snap_data[t_val] = local_snap_data
        
        # Load GLOBAL metrics (from files with "global" in name)
        global_pattern = f"results_chunk_*_global_snap_{t_val:.2f}.h5"
        global_files = glob.glob(os.path.join(partial_dir, global_pattern))
        global_files.sort(key=lambda x: int(os.path.basename(x).split('chunk_')[1].split('_')[0]))
        
        if global_files:
            global_metrics = ['fwd_stod_score', 'fwd_stod_type', 'fwd_finstod_score', 'fwd_finstod_type',
                            'bwd_stod_score', 'bwd_stod_type', 'bwd_finstod_score', 'bwd_finstod_type']
            
            global_snap_data = {}
            for met in global_metrics:
                dtype = np.int32 if "type" in met else np.float64
                full_grid = np.zeros((ny, nx), dtype=dtype)
                current_row = 0
                has_data = False
                
                for fpath in global_files:
                    with h5py.File(fpath, 'r') as f:
                        if met in f:
                            chunk = f[met][:]
                            rows_in_chunk = chunk.shape[0]
                            cols_in_chunk = chunk.shape[1] if len(chunk.shape) > 1 else 1
                            
                            # Validate chunk dimensions
                            if cols_in_chunk != nx:
                                print(f"  [WARN] Global chunk {os.path.basename(fpath)} has {cols_in_chunk} cols, expected {nx}")
                                continue
                            
                            # Only process if we haven't filled the grid yet
                            if current_row < ny:
                                end_row = min(current_row + rows_in_chunk, ny)
                                eff_rows = end_row - current_row
                                # Only take the rows we need
                                full_grid[current_row : end_row, :] = chunk[:eff_rows, :]
                                current_row += eff_rows
                                has_data = True
                            else:
                                print(f"  [WARN] Skipping global chunk {os.path.basename(fpath)} - grid already full (current_row={current_row}, ny={ny})")
                
                if has_data:
                    global_snap_data[met] = full_grid
            
            if global_snap_data:
                all_global_snap_data[t_val] = global_snap_data

    # 3. Organize outputs by metric type into subfolders
    print("\n--- Creating Metric Subfolders ---")
    metric_folders = {}
    
    # FTLE metrics
    if ac.get('compute_ftle_forward', False) or ac.get('compute_ftle_backward', False):
        if ac.get('compute_ftle_forward', False):
            metric_folders['ftle_forward'] = os.path.join(final_base_dir, "ftle_forward")
            os.makedirs(metric_folders['ftle_forward'], exist_ok=True)
            _set_dir_permissions(metric_folders['ftle_forward'])
        if ac.get('compute_ftle_backward', False):
            metric_folders['ftle_backward'] = os.path.join(final_base_dir, "ftle_backward")
            os.makedirs(metric_folders['ftle_backward'], exist_ok=True)
            _set_dir_permissions(metric_folders['ftle_backward'])
        if ac.get('compute_ftle_superimposition', False):
            metric_folders['ftle_superimposition'] = os.path.join(final_base_dir, "ftle_superimposition")
            os.makedirs(metric_folders['ftle_superimposition'], exist_ok=True)
            _set_dir_permissions(metric_folders['ftle_superimposition'])
    
    # Local STOD metrics
    if ac.get('compute_local_stod_forward', False) or ac.get('compute_local_stod_backward', False):
        if ac.get('compute_local_stod_forward', False):
            metric_folders['local_stod_forward'] = os.path.join(final_base_dir, "local_stod_forward")
            os.makedirs(metric_folders['local_stod_forward'], exist_ok=True)
        if ac.get('compute_local_stod_backward', False):
            metric_folders['local_stod_backward'] = os.path.join(final_base_dir, "local_stod_backward")
            os.makedirs(metric_folders['local_stod_backward'], exist_ok=True)
        if ac.get('compute_local_stod_superimposition', False):
            metric_folders['local_stod_superimposition'] = os.path.join(final_base_dir, "local_stod_superimposition")
            os.makedirs(metric_folders['local_stod_superimposition'], exist_ok=True)
    
    # Local FINSTOD metrics
    if ac.get('compute_local_finstod_forward', False) or ac.get('compute_local_finstod_backward', False):
        if ac.get('compute_local_finstod_forward', False):
            metric_folders['local_finstod_forward'] = os.path.join(final_base_dir, "local_finstod_forward")
            os.makedirs(metric_folders['local_finstod_forward'], exist_ok=True)
        if ac.get('compute_local_finstod_backward', False):
            metric_folders['local_finstod_backward'] = os.path.join(final_base_dir, "local_finstod_backward")
            os.makedirs(metric_folders['local_finstod_backward'], exist_ok=True)
        if ac.get('compute_local_finstod_superimposition', False):
            metric_folders['local_finstod_superimposition'] = os.path.join(final_base_dir, "local_finstod_superimposition")
            os.makedirs(metric_folders['local_finstod_superimposition'], exist_ok=True)
    
    # Global STOD metrics
    if ac.get('compute_global_stod_forward', False) or ac.get('compute_global_stod_backward', False):
        if ac.get('compute_global_stod_forward', False):
            metric_folders['global_stod_forward'] = os.path.join(final_base_dir, "global_stod_forward")
            os.makedirs(metric_folders['global_stod_forward'], exist_ok=True)
        if ac.get('compute_global_stod_backward', False):
            metric_folders['global_stod_backward'] = os.path.join(final_base_dir, "global_stod_backward")
            os.makedirs(metric_folders['global_stod_backward'], exist_ok=True)
        if ac.get('compute_global_stod_superimposition', False):
            metric_folders['global_stod_superimposition'] = os.path.join(final_base_dir, "global_stod_superimposition")
            os.makedirs(metric_folders['global_stod_superimposition'], exist_ok=True)
    
    # Global FINSTOD metrics
    if ac.get('compute_global_finstod_forward', False) or ac.get('compute_global_finstod_backward', False):
        if ac.get('compute_global_finstod_forward', False):
            metric_folders['global_finstod_forward'] = os.path.join(final_base_dir, "global_finstod_forward")
            os.makedirs(metric_folders['global_finstod_forward'], exist_ok=True)
        if ac.get('compute_global_finstod_backward', False):
            metric_folders['global_finstod_backward'] = os.path.join(final_base_dir, "global_finstod_backward")
            os.makedirs(metric_folders['global_finstod_backward'], exist_ok=True)
        if ac.get('compute_global_finstod_superimposition', False):
            metric_folders['global_finstod_superimposition'] = os.path.join(final_base_dir, "global_finstod_superimposition")
            os.makedirs(metric_folders['global_finstod_superimposition'], exist_ok=True)
    
    # LB folder
    if ac.get('compute_lb', False):
        metric_folders['lb'] = os.path.join(final_base_dir, "lb")
        os.makedirs(metric_folders['lb'], exist_ok=True)
    
    # OLB folders
    if ac.get('compute_olb_local_stod', False):
        metric_folders['olb_local_stod'] = os.path.join(final_base_dir, "olb_local_stod")
        os.makedirs(metric_folders['olb_local_stod'], exist_ok=True)
    if ac.get('compute_olb_local_finstod', False):
        metric_folders['olb_local_finstod'] = os.path.join(final_base_dir, "olb_local_finstod")
        os.makedirs(metric_folders['olb_local_finstod'], exist_ok=True)
    if ac.get('compute_olb_global_stod', False):
        metric_folders['olb_global_stod'] = os.path.join(final_base_dir, "olb_global_stod")
        os.makedirs(metric_folders['olb_global_stod'], exist_ok=True)
    if ac.get('compute_olb_global_finstod', False):
        metric_folders['olb_global_finstod'] = os.path.join(final_base_dir, "olb_global_finstod")
        os.makedirs(metric_folders['olb_global_finstod'], exist_ok=True)
    
    # FLI (Fast Lyapunov Indicator) folders
    if ac.get('compute_fli_forward', False):
        metric_folders['fli_forward'] = os.path.join(final_base_dir, "fli_forward")
        os.makedirs(metric_folders['fli_forward'], exist_ok=True)
        _set_dir_permissions(metric_folders['fli_forward'])
    if ac.get('compute_fli_backward', False):
        metric_folders['fli_backward'] = os.path.join(final_base_dir, "fli_backward")
        os.makedirs(metric_folders['fli_backward'], exist_ok=True)
        _set_dir_permissions(metric_folders['fli_backward'])
    if ac.get('compute_fli_superimposition', False):
        metric_folders['fli_superimposition'] = os.path.join(final_base_dir, "fli_superimposition")
        os.makedirs(metric_folders['fli_superimposition'], exist_ok=True)
        _set_dir_permissions(metric_folders['fli_superimposition'])
    
    # LD (Lagrangian Descriptor) folders
    if ac.get('compute_ld_forward', False):
        metric_folders['ld_forward'] = os.path.join(final_base_dir, "ld_forward")
        os.makedirs(metric_folders['ld_forward'], exist_ok=True)
        _set_dir_permissions(metric_folders['ld_forward'])
    if ac.get('compute_ld_backward', False):
        metric_folders['ld_backward'] = os.path.join(final_base_dir, "ld_backward")
        os.makedirs(metric_folders['ld_backward'], exist_ok=True)
        _set_dir_permissions(metric_folders['ld_backward'])
    if ac.get('compute_ld_superimposition', False):
        metric_folders['ld_superimposition'] = os.path.join(final_base_dir, "ld_superimposition")
        os.makedirs(metric_folders['ld_superimposition'], exist_ok=True)
        _set_dir_permissions(metric_folders['ld_superimposition'])

    # 4. Process and save LOCAL metrics by type
    print("\n--- Processing Local Metrics ---")
    for t_val, snap_data in tqdm(all_local_snap_data.items(), desc="Local Metrics", unit="snap"):
        # FTLE Forward
        if 'fwd_ftle' in snap_data and 'ftle_forward' in metric_folders:
            folder = metric_folders['ftle_forward']
            npy_path = os.path.join(folder, f"ftle_forward_raw_snap_{t_val:.2f}.npy")
            _save_npy_with_permissions(npy_path, snap_data['fwd_ftle'])
            if GENERATE_PLOTS:
                plot_generic_heatmap(snap_data['fwd_ftle'], f"FTLE Forward (t={t_val})", 
                                    f"ftle_forward_raw_snap_{t_val:.2f}.png", folder, gp, "standard", is_log=False)
                plot_generic_heatmap(snap_data['fwd_ftle'], f"FTLE Forward (t={t_val})", 
                                    f"ftle_forward_log_snap_{t_val:.2f}.png", folder, gp, "standard", is_log=True)
        
        # FTLE Backward
        if 'bwd_ftle' in snap_data and 'ftle_backward' in metric_folders:
            folder = metric_folders['ftle_backward']
            np.save(os.path.join(folder, f"ftle_backward_raw_snap_{t_val:.2f}.npy"), snap_data['bwd_ftle'])
            if GENERATE_PLOTS:
                plot_generic_heatmap(snap_data['bwd_ftle'], f"FTLE Backward (t={t_val})", 
                                    f"ftle_backward_raw_snap_{t_val:.2f}.png", folder, gp, "standard", is_log=False)
                plot_generic_heatmap(snap_data['bwd_ftle'], f"FTLE Backward (t={t_val})", 
                                    f"ftle_backward_log_snap_{t_val:.2f}.png", folder, gp, "standard", is_log=True)
        
        # FTLE Superimposition (validate both exist)
        if ('fwd_ftle' in snap_data and 'bwd_ftle' in snap_data and 
            'ftle_superimposition' in metric_folders):
            folder = metric_folders['ftle_superimposition']
            diff = snap_data['fwd_ftle'] - snap_data['bwd_ftle']
            np.save(os.path.join(folder, f"ftle_superimposition_raw_snap_{t_val:.2f}.npy"), diff)
            if GENERATE_PLOTS:
                plot_generic_heatmap(diff, f"FTLE Superimposition (t={t_val})", 
                                    f"ftle_superimposition_raw_snap_{t_val:.2f}.png", folder, gp, "difference", is_log=False)
                plot_generic_heatmap(diff, f"FTLE Superimposition (t={t_val})", 
                                    f"ftle_superimposition_log_snap_{t_val:.2f}.png", folder, gp, "difference", is_log=True)
        
        # Local STOD Forward
        if 'fwd_stod_score' in snap_data and 'fwd_stod_type' in snap_data and 'local_stod_forward' in metric_folders:
            folder = metric_folders['local_stod_forward']
            np.save(os.path.join(folder, f"local_stod_forward_raw_snap_{t_val:.2f}.npy"), snap_data['fwd_stod_score'])
            np.save(os.path.join(folder, f"local_stod_forward_type_snap_{t_val:.2f}.npy"), snap_data['fwd_stod_type'])
            if GENERATE_PLOTS:
                save_segmented_plot(snap_data['fwd_stod_score'], snap_data['fwd_stod_type'],
                                    os.path.join(folder, f"local_stod_forward_raw_snap_{t_val:.2f}.png"),
                                    f"Local STOD Forward (t={t_val})", gp, is_log=False)
                save_segmented_plot(snap_data['fwd_stod_score'], snap_data['fwd_stod_type'],
                                    os.path.join(folder, f"local_stod_forward_log_snap_{t_val:.2f}.png"),
                                    f"Local STOD Forward (t={t_val})", gp, is_log=True)
        
        # Local STOD Backward
        if 'bwd_stod_score' in snap_data and 'bwd_stod_type' in snap_data and 'local_stod_backward' in metric_folders:
            folder = metric_folders['local_stod_backward']
            np.save(os.path.join(folder, f"local_stod_backward_raw_snap_{t_val:.2f}.npy"), snap_data['bwd_stod_score'])
            np.save(os.path.join(folder, f"local_stod_backward_type_snap_{t_val:.2f}.npy"), snap_data['bwd_stod_type'])
            if GENERATE_PLOTS:
                save_segmented_plot(snap_data['bwd_stod_score'], snap_data['bwd_stod_type'],
                                    os.path.join(folder, f"local_stod_backward_raw_snap_{t_val:.2f}.png"),
                                    f"Local STOD Backward (t={t_val})", gp, is_log=False)
                save_segmented_plot(snap_data['bwd_stod_score'], snap_data['bwd_stod_type'],
                                    os.path.join(folder, f"local_stod_backward_log_snap_{t_val:.2f}.png"),
                                    f"Local STOD Backward (t={t_val})", gp, is_log=True)
        
        # Local STOD Superimposition (validate both exist)
        if ('fwd_stod_score' in snap_data and 'bwd_stod_score' in snap_data and 
            'local_stod_superimposition' in metric_folders):
            folder = metric_folders['local_stod_superimposition']
            diff = snap_data['fwd_stod_score'] - snap_data['bwd_stod_score']
            np.save(os.path.join(folder, f"local_stod_superimposition_raw_snap_{t_val:.2f}.npy"), diff)
            if GENERATE_PLOTS:
                plot_generic_heatmap(diff, f"Local STOD Superimposition (t={t_val})",
                                    f"local_stod_superimposition_raw_snap_{t_val:.2f}.png", folder, gp, "difference", is_log=False)
                plot_generic_heatmap(diff, f"Local STOD Superimposition (t={t_val})",
                                    f"local_stod_superimposition_log_snap_{t_val:.2f}.png", folder, gp, "difference", is_log=True)
        
        # Local FINSTOD Forward
        if 'fwd_finstod_score' in snap_data and 'fwd_finstod_type' in snap_data and 'local_finstod_forward' in metric_folders:
            folder = metric_folders['local_finstod_forward']
            np.save(os.path.join(folder, f"local_finstod_forward_raw_snap_{t_val:.2f}.npy"), snap_data['fwd_finstod_score'])
            np.save(os.path.join(folder, f"local_finstod_forward_type_snap_{t_val:.2f}.npy"), snap_data['fwd_finstod_type'])
            if GENERATE_PLOTS:
                save_segmented_plot(snap_data['fwd_finstod_score'], snap_data['fwd_finstod_type'],
                                    os.path.join(folder, f"local_finstod_forward_raw_snap_{t_val:.2f}.png"),
                                    f"Local FINSTOD Forward (t={t_val})", gp, is_log=False)
                save_segmented_plot(snap_data['fwd_finstod_score'], snap_data['fwd_finstod_type'],
                                    os.path.join(folder, f"local_finstod_forward_log_snap_{t_val:.2f}.png"),
                                    f"Local FINSTOD Forward (t={t_val})", gp, is_log=True)
        
        # Local FINSTOD Backward
        if 'bwd_finstod_score' in snap_data and 'bwd_finstod_type' in snap_data and 'local_finstod_backward' in metric_folders:
            folder = metric_folders['local_finstod_backward']
            np.save(os.path.join(folder, f"local_finstod_backward_raw_snap_{t_val:.2f}.npy"), snap_data['bwd_finstod_score'])
            np.save(os.path.join(folder, f"local_finstod_backward_type_snap_{t_val:.2f}.npy"), snap_data['bwd_finstod_type'])
            if GENERATE_PLOTS:
                save_segmented_plot(snap_data['bwd_finstod_score'], snap_data['bwd_finstod_type'],
                                    os.path.join(folder, f"local_finstod_backward_raw_snap_{t_val:.2f}.png"),
                                    f"Local FINSTOD Backward (t={t_val})", gp, is_log=False)
                save_segmented_plot(snap_data['bwd_finstod_score'], snap_data['bwd_finstod_type'],
                                    os.path.join(folder, f"local_finstod_backward_log_snap_{t_val:.2f}.png"),
                                    f"Local FINSTOD Backward (t={t_val})", gp, is_log=True)
        
        # Local FINSTOD Superimposition (validate both exist)
        if ('fwd_finstod_score' in snap_data and 'bwd_finstod_score' in snap_data and 
            'local_finstod_superimposition' in metric_folders):
            folder = metric_folders['local_finstod_superimposition']
            diff = snap_data['fwd_finstod_score'] - snap_data['bwd_finstod_score']
            np.save(os.path.join(folder, f"local_finstod_superimposition_raw_snap_{t_val:.2f}.npy"), diff)
            if GENERATE_PLOTS:
                plot_generic_heatmap(diff, f"Local FINSTOD Superimposition (t={t_val})",
                                    f"local_finstod_superimposition_raw_snap_{t_val:.2f}.png", folder, gp, "difference", is_log=False)
                plot_generic_heatmap(diff, f"Local FINSTOD Superimposition (t={t_val})",
                                    f"local_finstod_superimposition_log_snap_{t_val:.2f}.png", folder, gp, "difference", is_log=True)
        
        # FLI Forward
        if 'fwd_fli' in snap_data and 'fli_forward' in metric_folders:
            folder = metric_folders['fli_forward']
            npy_path = os.path.join(folder, f"fli_forward_raw_snap_{t_val:.2f}.npy")
            _save_npy_with_permissions(npy_path, snap_data['fwd_fli'])
            if GENERATE_PLOTS:
                plot_generic_heatmap(snap_data['fwd_fli'], f"FLI Forward (t={t_val})", 
                                    f"fli_forward_raw_snap_{t_val:.2f}.png", folder, gp, "standard", is_log=False)
                plot_generic_heatmap(snap_data['fwd_fli'], f"FLI Forward (t={t_val})", 
                                    f"fli_forward_log_snap_{t_val:.2f}.png", folder, gp, "standard", is_log=True)
        
        # FLI Backward
        if 'bwd_fli' in snap_data and 'fli_backward' in metric_folders:
            folder = metric_folders['fli_backward']
            npy_path = os.path.join(folder, f"fli_backward_raw_snap_{t_val:.2f}.npy")
            _save_npy_with_permissions(npy_path, snap_data['bwd_fli'])
            if GENERATE_PLOTS:
                plot_generic_heatmap(snap_data['bwd_fli'], f"FLI Backward (t={t_val})", 
                                    f"fli_backward_raw_snap_{t_val:.2f}.png", folder, gp, "standard", is_log=False)
                plot_generic_heatmap(snap_data['bwd_fli'], f"FLI Backward (t={t_val})", 
                                    f"fli_backward_log_snap_{t_val:.2f}.png", folder, gp, "standard", is_log=True)
        
        # FLI Superimposition (validate both exist)
        if ('fwd_fli' in snap_data and 'bwd_fli' in snap_data and 
            'fli_superimposition' in metric_folders):
            folder = metric_folders['fli_superimposition']
            diff = snap_data['fwd_fli'] - snap_data['bwd_fli']
            npy_path = os.path.join(folder, f"fli_superimposition_raw_snap_{t_val:.2f}.npy")
            _save_npy_with_permissions(npy_path, diff)
            if GENERATE_PLOTS:
                plot_generic_heatmap(diff, f"FLI Superimposition (t={t_val})", 
                                    f"fli_superimposition_raw_snap_{t_val:.2f}.png", folder, gp, "difference", is_log=False)
                plot_generic_heatmap(diff, f"FLI Superimposition (t={t_val})", 
                                    f"fli_superimposition_log_snap_{t_val:.2f}.png", folder, gp, "difference", is_log=True)
        
        # LD Forward
        if 'fwd_ld' in snap_data and 'ld_forward' in metric_folders:
            folder = metric_folders['ld_forward']
            npy_path = os.path.join(folder, f"ld_forward_raw_snap_{t_val:.2f}.npy")
            _save_npy_with_permissions(npy_path, snap_data['fwd_ld'])
            if GENERATE_PLOTS:
                plot_generic_heatmap(snap_data['fwd_ld'], f"LD Forward (t={t_val})", 
                                    f"ld_forward_raw_snap_{t_val:.2f}.png", folder, gp, "standard", is_log=False)
                plot_generic_heatmap(snap_data['fwd_ld'], f"LD Forward (t={t_val})", 
                                    f"ld_forward_log_snap_{t_val:.2f}.png", folder, gp, "standard", is_log=True)
        
        # LD Backward
        if 'bwd_ld' in snap_data and 'ld_backward' in metric_folders:
            folder = metric_folders['ld_backward']
            npy_path = os.path.join(folder, f"ld_backward_raw_snap_{t_val:.2f}.npy")
            _save_npy_with_permissions(npy_path, snap_data['bwd_ld'])
            if GENERATE_PLOTS:
                plot_generic_heatmap(snap_data['bwd_ld'], f"LD Backward (t={t_val})", 
                                    f"ld_backward_raw_snap_{t_val:.2f}.png", folder, gp, "standard", is_log=False)
                plot_generic_heatmap(snap_data['bwd_ld'], f"LD Backward (t={t_val})", 
                                    f"ld_backward_log_snap_{t_val:.2f}.png", folder, gp, "standard", is_log=True)
        
        # LD Superimposition (validate both exist)
        if ('fwd_ld' in snap_data and 'bwd_ld' in snap_data and 
            'ld_superimposition' in metric_folders):
            folder = metric_folders['ld_superimposition']
            diff = snap_data['fwd_ld'] - snap_data['bwd_ld']
            npy_path = os.path.join(folder, f"ld_superimposition_raw_snap_{t_val:.2f}.npy")
            _save_npy_with_permissions(npy_path, diff)
            if GENERATE_PLOTS:
                plot_generic_heatmap(diff, f"LD Superimposition (t={t_val})", 
                                    f"ld_superimposition_raw_snap_{t_val:.2f}.png", folder, gp, "difference", is_log=False)
                plot_generic_heatmap(diff, f"LD Superimposition (t={t_val})", 
                                    f"ld_superimposition_log_snap_{t_val:.2f}.png", folder, gp, "difference", is_log=True)

    # 5. Process and save GLOBAL metrics by type
    print("\n--- Processing Global Metrics ---")
    for t_val, snap_data in tqdm(all_global_snap_data.items(), desc="Global Metrics", unit="snap"):
        # Global STOD Forward
        if 'fwd_stod_score' in snap_data and 'fwd_stod_type' in snap_data and 'global_stod_forward' in metric_folders:
            folder = metric_folders['global_stod_forward']
            np.save(os.path.join(folder, f"global_stod_forward_raw_snap_{t_val:.2f}.npy"), snap_data['fwd_stod_score'])
            np.save(os.path.join(folder, f"global_stod_forward_type_snap_{t_val:.2f}.npy"), snap_data['fwd_stod_type'])
            if GENERATE_PLOTS:
                save_segmented_plot(snap_data['fwd_stod_score'], snap_data['fwd_stod_type'],
                                    os.path.join(folder, f"global_stod_forward_raw_snap_{t_val:.2f}.png"),
                                    f"Global STOD Forward (t={t_val})", gp, is_log=False)
                save_segmented_plot(snap_data['fwd_stod_score'], snap_data['fwd_stod_type'],
                                    os.path.join(folder, f"global_stod_forward_log_snap_{t_val:.2f}.png"),
                                    f"Global STOD Forward (t={t_val})", gp, is_log=True)
        
        # Global STOD Backward
        if 'bwd_stod_score' in snap_data and 'bwd_stod_type' in snap_data and 'global_stod_backward' in metric_folders:
            folder = metric_folders['global_stod_backward']
            np.save(os.path.join(folder, f"global_stod_backward_raw_snap_{t_val:.2f}.npy"), snap_data['bwd_stod_score'])
            np.save(os.path.join(folder, f"global_stod_backward_type_snap_{t_val:.2f}.npy"), snap_data['bwd_stod_type'])
            if GENERATE_PLOTS:
                save_segmented_plot(snap_data['bwd_stod_score'], snap_data['bwd_stod_type'],
                                    os.path.join(folder, f"global_stod_backward_raw_snap_{t_val:.2f}.png"),
                                    f"Global STOD Backward (t={t_val})", gp, is_log=False)
                save_segmented_plot(snap_data['bwd_stod_score'], snap_data['bwd_stod_type'],
                                    os.path.join(folder, f"global_stod_backward_log_snap_{t_val:.2f}.png"),
                                    f"Global STOD Backward (t={t_val})", gp, is_log=True)
        
        # Global STOD Superimposition (validate both exist)
        if ('fwd_stod_score' in snap_data and 'bwd_stod_score' in snap_data and 
            'global_stod_superimposition' in metric_folders):
            folder = metric_folders['global_stod_superimposition']
            diff = snap_data['fwd_stod_score'] - snap_data['bwd_stod_score']
            np.save(os.path.join(folder, f"global_stod_superimposition_raw_snap_{t_val:.2f}.npy"), diff)
            if GENERATE_PLOTS:
                plot_generic_heatmap(diff, f"Global STOD Superimposition (t={t_val})",
                                    f"global_stod_superimposition_raw_snap_{t_val:.2f}.png", folder, gp, "difference", is_log=False)
                plot_generic_heatmap(diff, f"Global STOD Superimposition (t={t_val})",
                                    f"global_stod_superimposition_log_snap_{t_val:.2f}.png", folder, gp, "difference", is_log=True)
        
        # Global FINSTOD Forward
        if 'fwd_finstod_score' in snap_data and 'fwd_finstod_type' in snap_data and 'global_finstod_forward' in metric_folders:
            folder = metric_folders['global_finstod_forward']
            np.save(os.path.join(folder, f"global_finstod_forward_raw_snap_{t_val:.2f}.npy"), snap_data['fwd_finstod_score'])
            np.save(os.path.join(folder, f"global_finstod_forward_type_snap_{t_val:.2f}.npy"), snap_data['fwd_finstod_type'])
            if GENERATE_PLOTS:
                save_segmented_plot(snap_data['fwd_finstod_score'], snap_data['fwd_finstod_type'],
                                    os.path.join(folder, f"global_finstod_forward_raw_snap_{t_val:.2f}.png"),
                                    f"Global FINSTOD Forward (t={t_val})", gp, is_log=False)
                save_segmented_plot(snap_data['fwd_finstod_score'], snap_data['fwd_finstod_type'],
                                    os.path.join(folder, f"global_finstod_forward_log_snap_{t_val:.2f}.png"),
                                    f"Global FINSTOD Forward (t={t_val})", gp, is_log=True)
        
        # Global FINSTOD Backward
        if 'bwd_finstod_score' in snap_data and 'bwd_finstod_type' in snap_data and 'global_finstod_backward' in metric_folders:
            folder = metric_folders['global_finstod_backward']
            np.save(os.path.join(folder, f"global_finstod_backward_raw_snap_{t_val:.2f}.npy"), snap_data['bwd_finstod_score'])
            np.save(os.path.join(folder, f"global_finstod_backward_type_snap_{t_val:.2f}.npy"), snap_data['bwd_finstod_type'])
            if GENERATE_PLOTS:
                save_segmented_plot(snap_data['bwd_finstod_score'], snap_data['bwd_finstod_type'],
                                    os.path.join(folder, f"global_finstod_backward_raw_snap_{t_val:.2f}.png"),
                                    f"Global FINSTOD Backward (t={t_val})", gp, is_log=False)
                save_segmented_plot(snap_data['bwd_finstod_score'], snap_data['bwd_finstod_type'],
                                    os.path.join(folder, f"global_finstod_backward_log_snap_{t_val:.2f}.png"),
                                    f"Global FINSTOD Backward (t={t_val})", gp, is_log=True)
        
        # Global FINSTOD Superimposition (validate both exist)
        if ('fwd_finstod_score' in snap_data and 'bwd_finstod_score' in snap_data and 
            'global_finstod_superimposition' in metric_folders):
            folder = metric_folders['global_finstod_superimposition']
            diff = snap_data['fwd_finstod_score'] - snap_data['bwd_finstod_score']
            np.save(os.path.join(folder, f"global_finstod_superimposition_raw_snap_{t_val:.2f}.npy"), diff)
            if GENERATE_PLOTS:
                plot_generic_heatmap(diff, f"Global FINSTOD Superimposition (t={t_val})",
                                    f"global_finstod_superimposition_raw_snap_{t_val:.2f}.png", folder, gp, "difference", is_log=False)
                plot_generic_heatmap(diff, f"Global FINSTOD Superimposition (t={t_val})",
                                    f"global_finstod_superimposition_log_snap_{t_val:.2f}.png", folder, gp, "difference", is_log=True)

    # 6. Aggregate LB (time-independent)
    if ac.get('compute_lb', False) and 'lb' in metric_folders:
        print("\n--- Aggregating LB Results ---")
        lb_pattern = os.path.join(partial_dir, "results_chunk_*_exact_lb.h5")
        lb_files = glob.glob(lb_pattern)
        
        if lb_files:
            lb_files.sort(key=lambda x: int(x.split('chunk_')[1].split('_')[0]))
            
            # Initialize arrays for all variants
            full_lb_fxb = np.zeros((ny, nx), dtype=np.float64)
            full_lb_fpb = np.zeros((ny, nx), dtype=np.float64)
            full_lb_fmb = np.zeros((ny, nx), dtype=np.float64)
            current_row = 0
            
            for fpath in tqdm(lb_files, desc="Stitching LB", unit="file"):
                with h5py.File(fpath, 'r') as f:
                    # Load all variants
                    chunk_fxb = None
                    chunk_fpb = None
                    chunk_fmb = None
                    
                    if 'exact_lb_fxb' in f:
                        chunk_fxb = f['exact_lb_fxb'][:]
                    elif 'exact_lb' in f:  # Backward compatibility
                        chunk_fxb = f['exact_lb'][:]
                    elif 'lb' in f:  # Old format
                        chunk_fxb = f['lb'][:]
                    
                    if 'exact_lb_fpb' in f:
                        chunk_fpb = f['exact_lb_fpb'][:]
                    
                    if 'exact_lb_fmb' in f:
                        chunk_fmb = f['exact_lb_fmb'][:]
                    
                    if chunk_fxb is None:
                        continue
                    
                    rows_in_chunk = chunk_fxb.shape[0]
                    end_row = min(current_row + rows_in_chunk, ny)
                    eff_rows = end_row - current_row
                    
                    full_lb_fxb[current_row : end_row, :] = chunk_fxb[:eff_rows, :]
                    if chunk_fpb is not None:
                        full_lb_fpb[current_row : end_row, :] = chunk_fpb[:eff_rows, :]
                    if chunk_fmb is not None:
                        full_lb_fmb[current_row : end_row, :] = chunk_fmb[:eff_rows, :]
                    
                    current_row += eff_rows

            folder = metric_folders['lb']
            
            # Save F×B (classical LB)
            np.save(os.path.join(folder, "lb_fxb_raw.npy"), full_lb_fxb)
            if GENERATE_PLOTS:
                plot_generic_heatmap(full_lb_fxb, f"LB: Forward × Backward (Tau={tau})", 
                                    "lb_fxb_raw.png", folder, gp, plot_mode="lb", is_log=False)
                plot_generic_heatmap(full_lb_fxb, f"LB: Forward × Backward (Tau={tau})", 
                                    "lb_fxb_log.png", folder, gp, plot_mode="lb", is_log=True)
            
            # Save F+B
            if np.any(full_lb_fpb != 0):
                np.save(os.path.join(folder, "lb_fpb_raw.npy"), full_lb_fpb)
                if GENERATE_PLOTS:
                    plot_generic_heatmap(full_lb_fpb, f"LB: Forward + Backward (Tau={tau})", 
                                        "lb_fpb_raw.png", folder, gp, plot_mode="lb", is_log=False)
                    plot_generic_heatmap(full_lb_fpb, f"LB: Forward + Backward (Tau={tau})", 
                                        "lb_fpb_log.png", folder, gp, plot_mode="lb", is_log=True)
            
            # Save F-B
            if np.any(full_lb_fmb != 0):
                np.save(os.path.join(folder, "lb_fmb_raw.npy"), full_lb_fmb)
                if GENERATE_PLOTS:
                    plot_generic_heatmap(full_lb_fmb, f"LB: Forward - Backward (Tau={tau})", 
                                        "lb_fmb_raw.png", folder, gp, plot_mode="lb", is_log=False)
                    plot_generic_heatmap(full_lb_fmb, f"LB: Forward - Backward (Tau={tau})", 
                                        "lb_fmb_log.png", folder, gp, plot_mode="lb", is_log=True)
            
            # Backward compatibility: save as lb_raw.npy (F×B variant)
            np.save(os.path.join(folder, "lb_raw.npy"), full_lb_fxb)
            if GENERATE_PLOTS:
                plot_generic_heatmap(full_lb_fxb, f"Lagrangian Betweenness (Tau={tau})", 
                                    "lb_raw.png", folder, gp, plot_mode="lb", is_log=False)
                plot_generic_heatmap(full_lb_fxb, f"Lagrangian Betweenness (Tau={tau})", 
                                    "lb_log.png", folder, gp, plot_mode="lb", is_log=True)
        else:
            print("  [WARN] No LB files found.")
    
    # 7. Aggregate OLB (time-independent, computed on-the-fly in C++)
    print("\n--- Aggregating OLB Results ---")
    
    # OLB Local STOD
    if ac.get('compute_olb_local_stod', False) and 'olb_local_stod' in metric_folders:
        print("  > Aggregating OLB Local STOD...")
        olb_pattern = os.path.join(partial_dir, "results_chunk_*_exact_olb_local_stod.h5")
        olb_files = glob.glob(olb_pattern)
        
        if olb_files:
            olb_files.sort(key=lambda x: int(x.split('chunk_')[1].split('_')[0]))
            
            # Initialize arrays for all variants
            full_olb_fxb = np.zeros((ny, nx), dtype=np.float64)
            full_olb_fpb = np.zeros((ny, nx), dtype=np.float64)
            full_olb_fmb = np.zeros((ny, nx), dtype=np.float64)
            current_row = 0
            
            for fpath in tqdm(olb_files, desc="Stitching OLB Local STOD", unit="file"):
                with h5py.File(fpath, 'r') as f:
                    # Load all variants
                    chunk_fxb = None
                    chunk_fpb = None
                    chunk_fmb = None
                    
                    if 'exact_olb_stod_fxb' in f:
                        chunk_fxb = f['exact_olb_stod_fxb'][:]
                    elif 'exact_olb_stod' in f:  # Backward compatibility
                        chunk_fxb = f['exact_olb_stod'][:]
                    
                    if 'exact_olb_stod_fpb' in f:
                        chunk_fpb = f['exact_olb_stod_fpb'][:]
                    
                    if 'exact_olb_stod_fmb' in f:
                        chunk_fmb = f['exact_olb_stod_fmb'][:]
                    
                    if chunk_fxb is None:
                        continue
                    
                    rows_in_chunk = chunk_fxb.shape[0]
                    end_row = min(current_row + rows_in_chunk, ny)
                    eff_rows = end_row - current_row
                    
                    full_olb_fxb[current_row : end_row, :] = chunk_fxb[:eff_rows, :]
                    if chunk_fpb is not None:
                        full_olb_fpb[current_row : end_row, :] = chunk_fpb[:eff_rows, :]
                    if chunk_fmb is not None:
                        full_olb_fmb[current_row : end_row, :] = chunk_fmb[:eff_rows, :]
                    
                    current_row += eff_rows

            folder = metric_folders['olb_local_stod']
            
            # Save F×B (classical OLB)
            np.save(os.path.join(folder, "olb_local_stod_fxb_raw.npy"), full_olb_fxb)
            if GENERATE_PLOTS:
                plot_generic_heatmap(full_olb_fxb, f"OLB Local STOD: Forward × Backward (Tau={tau})", 
                                    "olb_local_stod_fxb_raw.png", folder, gp, plot_mode="lb", is_log=False)
                plot_generic_heatmap(full_olb_fxb, f"OLB Local STOD: Forward × Backward (Tau={tau})", 
                                    "olb_local_stod_fxb_log.png", folder, gp, plot_mode="lb", is_log=True)
            
            # Save F+B
            if np.any(full_olb_fpb != 0):
                np.save(os.path.join(folder, "olb_local_stod_fpb_raw.npy"), full_olb_fpb)
                if GENERATE_PLOTS:
                    plot_generic_heatmap(full_olb_fpb, f"OLB Local STOD: Forward + Backward (Tau={tau})", 
                                        "olb_local_stod_fpb_raw.png", folder, gp, plot_mode="lb", is_log=False)
                    plot_generic_heatmap(full_olb_fpb, f"OLB Local STOD: Forward + Backward (Tau={tau})", 
                                        "olb_local_stod_fpb_log.png", folder, gp, plot_mode="lb", is_log=True)
            
            # Save F-B
            if np.any(full_olb_fmb != 0):
                np.save(os.path.join(folder, "olb_local_stod_fmb_raw.npy"), full_olb_fmb)
                if GENERATE_PLOTS:
                    plot_generic_heatmap(full_olb_fmb, f"OLB Local STOD: Forward - Backward (Tau={tau})", 
                                        "olb_local_stod_fmb_raw.png", folder, gp, plot_mode="lb", is_log=False)
                    plot_generic_heatmap(full_olb_fmb, f"OLB Local STOD: Forward - Backward (Tau={tau})", 
                                        "olb_local_stod_fmb_log.png", folder, gp, plot_mode="lb", is_log=True)
            
            # Backward compatibility: save as olb_local_stod_raw.npy (F×B variant)
            np.save(os.path.join(folder, "olb_local_stod_raw.npy"), full_olb_fxb)
            if GENERATE_PLOTS:
                plot_generic_heatmap(full_olb_fxb, f"OLB Local STOD (Tau={tau})", 
                                    "olb_local_stod_raw.png", folder, gp, plot_mode="lb", is_log=False)
                plot_generic_heatmap(full_olb_fxb, f"OLB Local STOD (Tau={tau})", 
                                    "olb_local_stod_log.png", folder, gp, plot_mode="lb", is_log=True)
        else:
            print("    [WARN] No OLB Local STOD files found.")
    
    # OLB Local FINSTOD
    if ac.get('compute_olb_local_finstod', False) and 'olb_local_finstod' in metric_folders:
        print("  > Aggregating OLB Local FINSTOD...")
        olb_pattern = os.path.join(partial_dir, "results_chunk_*_exact_olb_local_finstod.h5")
        olb_files = glob.glob(olb_pattern)
        
        if olb_files:
            olb_files.sort(key=lambda x: int(x.split('chunk_')[1].split('_')[0]))
            
            # Initialize arrays for all variants
            full_olb_fxb = np.zeros((ny, nx), dtype=np.float64)
            full_olb_fpb = np.zeros((ny, nx), dtype=np.float64)
            full_olb_fmb = np.zeros((ny, nx), dtype=np.float64)
            current_row = 0
            
            for fpath in tqdm(olb_files, desc="Stitching OLB Local FINSTOD", unit="file"):
                with h5py.File(fpath, 'r') as f:
                    # Load all variants
                    chunk_fxb = None
                    chunk_fpb = None
                    chunk_fmb = None
                    
                    if 'exact_olb_finstod_fxb' in f:
                        chunk_fxb = f['exact_olb_finstod_fxb'][:]
                    elif 'exact_olb_finstod' in f:  # Backward compatibility
                        chunk_fxb = f['exact_olb_finstod'][:]
                    
                    if 'exact_olb_finstod_fpb' in f:
                        chunk_fpb = f['exact_olb_finstod_fpb'][:]
                    
                    if 'exact_olb_finstod_fmb' in f:
                        chunk_fmb = f['exact_olb_finstod_fmb'][:]
                    
                    if chunk_fxb is None:
                        continue
                    
                    rows_in_chunk = chunk_fxb.shape[0]
                    end_row = min(current_row + rows_in_chunk, ny)
                    eff_rows = end_row - current_row
                    
                    full_olb_fxb[current_row : end_row, :] = chunk_fxb[:eff_rows, :]
                    if chunk_fpb is not None:
                        full_olb_fpb[current_row : end_row, :] = chunk_fpb[:eff_rows, :]
                    if chunk_fmb is not None:
                        full_olb_fmb[current_row : end_row, :] = chunk_fmb[:eff_rows, :]
                    
                    current_row += eff_rows

            folder = metric_folders['olb_local_finstod']
            
            # Save F×B (classical OLB)
            np.save(os.path.join(folder, "olb_local_finstod_fxb_raw.npy"), full_olb_fxb)
            if GENERATE_PLOTS:
                plot_generic_heatmap(full_olb_fxb, f"OLB Local FINSTOD: Forward × Backward (Tau={tau})", 
                                    "olb_local_finstod_fxb_raw.png", folder, gp, plot_mode="lb", is_log=False)
                plot_generic_heatmap(full_olb_fxb, f"OLB Local FINSTOD: Forward × Backward (Tau={tau})", 
                                    "olb_local_finstod_fxb_log.png", folder, gp, plot_mode="lb", is_log=True)
            
            # Save F+B
            if np.any(full_olb_fpb != 0):
                np.save(os.path.join(folder, "olb_local_finstod_fpb_raw.npy"), full_olb_fpb)
                if GENERATE_PLOTS:
                    plot_generic_heatmap(full_olb_fpb, f"OLB Local FINSTOD: Forward + Backward (Tau={tau})", 
                                        "olb_local_finstod_fpb_raw.png", folder, gp, plot_mode="lb", is_log=False)
                    plot_generic_heatmap(full_olb_fpb, f"OLB Local FINSTOD: Forward + Backward (Tau={tau})", 
                                        "olb_local_finstod_fpb_log.png", folder, gp, plot_mode="lb", is_log=True)
            
            # Save F-B
            if np.any(full_olb_fmb != 0):
                np.save(os.path.join(folder, "olb_local_finstod_fmb_raw.npy"), full_olb_fmb)
                if GENERATE_PLOTS:
                    plot_generic_heatmap(full_olb_fmb, f"OLB Local FINSTOD: Forward - Backward (Tau={tau})", 
                                        "olb_local_finstod_fmb_raw.png", folder, gp, plot_mode="lb", is_log=False)
                    plot_generic_heatmap(full_olb_fmb, f"OLB Local FINSTOD: Forward - Backward (Tau={tau})", 
                                        "olb_local_finstod_fmb_log.png", folder, gp, plot_mode="lb", is_log=True)
            
            # Backward compatibility: save as olb_local_finstod_raw.npy (F×B variant)
            np.save(os.path.join(folder, "olb_local_finstod_raw.npy"), full_olb_fxb)
            if GENERATE_PLOTS:
                plot_generic_heatmap(full_olb_fxb, f"OLB Local FINSTOD (Tau={tau})", 
                                    "olb_local_finstod_raw.png", folder, gp, plot_mode="lb", is_log=False)
                plot_generic_heatmap(full_olb_fxb, f"OLB Local FINSTOD (Tau={tau})", 
                                    "olb_local_finstod_log.png", folder, gp, plot_mode="lb", is_log=True)
        else:
            print("    [WARN] No OLB Local FINSTOD files found.")
    
    # OLB Global STOD
    if ac.get('compute_olb_global_stod', False) and 'olb_global_stod' in metric_folders:
        print("  > Aggregating OLB Global STOD...")
        olb_pattern = os.path.join(partial_dir, "results_chunk_*_exact_olb_global_stod.h5")
        olb_files = glob.glob(olb_pattern)
        
        if olb_files:
            olb_files.sort(key=lambda x: int(x.split('chunk_')[1].split('_')[0]))
            
            # Initialize arrays for all variants
            full_olb_fxb = np.zeros((ny, nx), dtype=np.float64)
            full_olb_fpb = np.zeros((ny, nx), dtype=np.float64)
            full_olb_fmb = np.zeros((ny, nx), dtype=np.float64)
            current_row = 0
            
            for fpath in tqdm(olb_files, desc="Stitching OLB Global STOD", unit="file"):
                with h5py.File(fpath, 'r') as f:
                    # Load all variants
                    chunk_fxb = None
                    chunk_fpb = None
                    chunk_fmb = None
                    
                    if 'exact_olb_stod_fxb' in f:
                        chunk_fxb = f['exact_olb_stod_fxb'][:]
                    elif 'exact_olb_stod' in f:  # Backward compatibility
                        chunk_fxb = f['exact_olb_stod'][:]
                    
                    if 'exact_olb_stod_fpb' in f:
                        chunk_fpb = f['exact_olb_stod_fpb'][:]
                    
                    if 'exact_olb_stod_fmb' in f:
                        chunk_fmb = f['exact_olb_stod_fmb'][:]
                    
                    if chunk_fxb is None:
                        continue
                    
                    rows_in_chunk = chunk_fxb.shape[0]
                    end_row = min(current_row + rows_in_chunk, ny)
                    eff_rows = end_row - current_row
                    
                    full_olb_fxb[current_row : end_row, :] = chunk_fxb[:eff_rows, :]
                    if chunk_fpb is not None:
                        full_olb_fpb[current_row : end_row, :] = chunk_fpb[:eff_rows, :]
                    if chunk_fmb is not None:
                        full_olb_fmb[current_row : end_row, :] = chunk_fmb[:eff_rows, :]
                    
                    current_row += eff_rows

            folder = metric_folders['olb_global_stod']
            
            # Save F×B (classical OLB)
            np.save(os.path.join(folder, "olb_global_stod_fxb_raw.npy"), full_olb_fxb)
            if GENERATE_PLOTS:
                plot_generic_heatmap(full_olb_fxb, f"OLB Global STOD: Forward × Backward (Tau={tau})", 
                                    "olb_global_stod_fxb_raw.png", folder, gp, plot_mode="lb", is_log=False)
                plot_generic_heatmap(full_olb_fxb, f"OLB Global STOD: Forward × Backward (Tau={tau})", 
                                    "olb_global_stod_fxb_log.png", folder, gp, plot_mode="lb", is_log=True)
            
            # Save F+B
            if np.any(full_olb_fpb != 0):
                np.save(os.path.join(folder, "olb_global_stod_fpb_raw.npy"), full_olb_fpb)
                if GENERATE_PLOTS:
                    plot_generic_heatmap(full_olb_fpb, f"OLB Global STOD: Forward + Backward (Tau={tau})", 
                                        "olb_global_stod_fpb_raw.png", folder, gp, plot_mode="lb", is_log=False)
                    plot_generic_heatmap(full_olb_fpb, f"OLB Global STOD: Forward + Backward (Tau={tau})", 
                                        "olb_global_stod_fpb_log.png", folder, gp, plot_mode="lb", is_log=True)
            
            # Save F-B
            if np.any(full_olb_fmb != 0):
                np.save(os.path.join(folder, "olb_global_stod_fmb_raw.npy"), full_olb_fmb)
                if GENERATE_PLOTS:
                    plot_generic_heatmap(full_olb_fmb, f"OLB Global STOD: Forward - Backward (Tau={tau})", 
                                        "olb_global_stod_fmb_raw.png", folder, gp, plot_mode="lb", is_log=False)
                    plot_generic_heatmap(full_olb_fmb, f"OLB Global STOD: Forward - Backward (Tau={tau})", 
                                        "olb_global_stod_fmb_log.png", folder, gp, plot_mode="lb", is_log=True)
            
            # Backward compatibility: save as olb_global_stod_raw.npy (F×B variant)
            np.save(os.path.join(folder, "olb_global_stod_raw.npy"), full_olb_fxb)
            if GENERATE_PLOTS:
                plot_generic_heatmap(full_olb_fxb, f"OLB Global STOD (Tau={tau})", 
                                    "olb_global_stod_raw.png", folder, gp, plot_mode="lb", is_log=False)
                plot_generic_heatmap(full_olb_fxb, f"OLB Global STOD (Tau={tau})", 
                                    "olb_global_stod_log.png", folder, gp, plot_mode="lb", is_log=True)
        else:
            print("    [WARN] No OLB Global STOD files found.")
    
    # OLB Global FINSTOD
    if ac.get('compute_olb_global_finstod', False) and 'olb_global_finstod' in metric_folders:
        print("  > Aggregating OLB Global FINSTOD...")
        olb_pattern = os.path.join(partial_dir, "results_chunk_*_exact_olb_global_finstod.h5")
        olb_files = glob.glob(olb_pattern)
        
        if olb_files:
            olb_files.sort(key=lambda x: int(x.split('chunk_')[1].split('_')[0]))
            
            # Initialize arrays for all variants
            full_olb_fxb = np.zeros((ny, nx), dtype=np.float64)
            full_olb_fpb = np.zeros((ny, nx), dtype=np.float64)
            full_olb_fmb = np.zeros((ny, nx), dtype=np.float64)
            current_row = 0
            
            for fpath in tqdm(olb_files, desc="Stitching OLB Global FINSTOD", unit="file"):
                with h5py.File(fpath, 'r') as f:
                    # Load all variants
                    chunk_fxb = None
                    chunk_fpb = None
                    chunk_fmb = None
                    
                    if 'exact_olb_finstod_fxb' in f:
                        chunk_fxb = f['exact_olb_finstod_fxb'][:]
                    elif 'exact_olb_finstod' in f:  # Backward compatibility
                        chunk_fxb = f['exact_olb_finstod'][:]
                    
                    if 'exact_olb_finstod_fpb' in f:
                        chunk_fpb = f['exact_olb_finstod_fpb'][:]
                    
                    if 'exact_olb_finstod_fmb' in f:
                        chunk_fmb = f['exact_olb_finstod_fmb'][:]
                    
                    if chunk_fxb is None:
                        continue
                    
                    rows_in_chunk = chunk_fxb.shape[0]
                    end_row = min(current_row + rows_in_chunk, ny)
                    eff_rows = end_row - current_row
                    
                    full_olb_fxb[current_row : end_row, :] = chunk_fxb[:eff_rows, :]
                    if chunk_fpb is not None:
                        full_olb_fpb[current_row : end_row, :] = chunk_fpb[:eff_rows, :]
                    if chunk_fmb is not None:
                        full_olb_fmb[current_row : end_row, :] = chunk_fmb[:eff_rows, :]
                    
                    current_row += eff_rows

            folder = metric_folders['olb_global_finstod']
            
            # Save F×B (classical OLB)
            np.save(os.path.join(folder, "olb_global_finstod_fxb_raw.npy"), full_olb_fxb)
            if GENERATE_PLOTS:
                plot_generic_heatmap(full_olb_fxb, f"OLB Global FINSTOD: Forward × Backward (Tau={tau})", 
                                    "olb_global_finstod_fxb_raw.png", folder, gp, plot_mode="lb", is_log=False)
                plot_generic_heatmap(full_olb_fxb, f"OLB Global FINSTOD: Forward × Backward (Tau={tau})", 
                                    "olb_global_finstod_fxb_log.png", folder, gp, plot_mode="lb", is_log=True)
            
            # Save F+B
            if np.any(full_olb_fpb != 0):
                np.save(os.path.join(folder, "olb_global_finstod_fpb_raw.npy"), full_olb_fpb)
                if GENERATE_PLOTS:
                    plot_generic_heatmap(full_olb_fpb, f"OLB Global FINSTOD: Forward + Backward (Tau={tau})", 
                                        "olb_global_finstod_fpb_raw.png", folder, gp, plot_mode="lb", is_log=False)
                    plot_generic_heatmap(full_olb_fpb, f"OLB Global FINSTOD: Forward + Backward (Tau={tau})", 
                                        "olb_global_finstod_fpb_log.png", folder, gp, plot_mode="lb", is_log=True)
            
            # Save F-B
            if np.any(full_olb_fmb != 0):
                np.save(os.path.join(folder, "olb_global_finstod_fmb_raw.npy"), full_olb_fmb)
                if GENERATE_PLOTS:
                    plot_generic_heatmap(full_olb_fmb, f"OLB Global FINSTOD: Forward - Backward (Tau={tau})", 
                                        "olb_global_finstod_fmb_raw.png", folder, gp, plot_mode="lb", is_log=False)
                    plot_generic_heatmap(full_olb_fmb, f"OLB Global FINSTOD: Forward - Backward (Tau={tau})", 
                                        "olb_global_finstod_fmb_log.png", folder, gp, plot_mode="lb", is_log=True)
            
            # Backward compatibility: save as olb_global_finstod_raw.npy (F×B variant)
            np.save(os.path.join(folder, "olb_global_finstod_raw.npy"), full_olb_fxb)
            if GENERATE_PLOTS:
                plot_generic_heatmap(full_olb_fxb, f"OLB Global FINSTOD (Tau={tau})", 
                                    "olb_global_finstod_raw.png", folder, gp, plot_mode="lb", is_log=False)
                plot_generic_heatmap(full_olb_fxb, f"OLB Global FINSTOD (Tau={tau})", 
                                    "olb_global_finstod_log.png", folder, gp, plot_mode="lb", is_log=True)
        else:
            print("    [WARN] No OLB Global FINSTOD files found.")
    
    # Note: OLB is now computed on-the-fly in C++ (matching LB approach)
    # The old Python-based OLB computation using snapshots has been removed
    # OLB aggregation is handled in section 7 above

    print("\n" + "=" * 80)
    print(f"Aggregation Complete. Results saved in: {final_base_dir}")
    print(f"Metric subfolders created: {len(metric_folders)}")
    print("=" * 80)

if __name__ == "__main__":
    main()



