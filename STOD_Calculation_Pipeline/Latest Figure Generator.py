import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation

# =============================================================================
# CONFIGURATION
# =============================================================================

plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'

BASE_DATA_PATH = "/Volumes/PortableSSD/STOD"
OUTPUT_DIR = "/Users/pablo/Desktop/Programar/STOD Paper Creation/Final_Paper_Figures_V6"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_MAP = {
    "linear": {
        "dir": "final_results_hyperbolic_linear_alpha0.5_nx1000ny1000_T15",
        "T_max": 15, "extent": [-1, 1, -1, 1], "xlabel": "x", "ylabel": "y", "aspect": 1.0
    },
    "nonlinear": {
        "dir": "final_results_nonlinear_saddle_beta0.005_alpha0.5_gamma0.005_nx1000ny1000_T20",
        "T_max": 20, "extent": [-1, 1, -1, 1], "xlabel": "x", "ylabel": "y", "aspect": 1.0
    },
    "pendulum": {
        "dir": "final_results_pendulum_lambda1_nx1000ny1000_T20",
        "T_max": 20, "extent": [-3.14159, 3.14159, -4, 4], "xlabel": r"$\theta$", "ylabel": r"$\omega$", "aspect": 0.6
    },
    "lorenz": {
        "dir": "final_results_lorenz_sigma10_rho28_beta2.667_nx1600ny1200_T10",
        "T_max": 10, "extent": [-20, 20, -30, 30], "xlabel": "x", "ylabel": "y", "aspect": 1.5
    },
    "duffing": {
        "dir": "final_results_duffing_omega1.2_beta1_alpha-1_gamma0.5_delta0.3_nx1000ny1000_T20",
        "T_max": 20, "extent": [-2, 2, -2, 2], "xlabel": "x", "ylabel": "v", "aspect": 1.0
    },
    "doublegyre": {
        "dir": "final_results_doublegyre_A0.1_epsilon0.25_omega0.6283_nx2000ny1000_T15",
        "T_max": 15, "extent": [0, 2, 0, 1], "xlabel": "x", "ylabel": "y", "aspect": 0.5
    }
}

# =============================================================================
# UTILITIES & FinSTOD LOGIC
# =============================================================================

def normalize_minmax(a):
    if a is None: return None
    a = np.array(a, dtype=float)
    a[~np.isfinite(a)] = 0.0
    amin, amax = np.nanmin(a), np.nanmax(a)
    return (a - amin) / (amax - amin + 1e-12) if amax > amin else np.zeros_like(a)

def normalize_segmented(score_grid, type_grid):
    if score_grid is None or type_grid is None: return None, 0.5
    mask_nodata = (type_grid == -1)
    out_grid = np.full(score_grid.shape, np.nan)
    valid_mask = ~mask_nodata
    if not np.any(valid_mask): return np.ma.masked_invalid(out_grid), 0.5
    
    flat_scores, flat_types = score_grid[valid_mask], type_grid[valid_mask]
    t_mask, uc_mask = (flat_types == 0), (flat_types == 1)
    total_non_uu = np.sum(t_mask) + np.sum(uc_mask)
    P = max(0.05, min(0.95, np.sum(t_mask) / total_non_uu if total_non_uu > 0 else 0.5))
    
    if np.sum(t_mask) > 0:
        vals = flat_scores[t_mask]
        vmin, vmax = np.min(vals), np.max(vals)
        out_grid[(type_grid == 0) & valid_mask] = (vals - vmin) / (vmax - vmin + 1e-12) * P if vmax > vmin else P/2
    if np.sum(uc_mask) > 0:
        vals = flat_scores[uc_mask]
        vmin, vmax = np.min(vals), np.max(vals)
        out_grid[(type_grid == 1) & valid_mask] = P + (vals - vmin) / (vmax - vmin + 1e-12) * (0.99 - P) if vmax > vmin else P + 0.45
    out_grid[(type_grid == 2) & valid_mask] = 1.0
    return np.ma.masked_invalid(out_grid), P

def load_data(sys_key, metric, direction, t_snap, apply_log=False, get_type=False):
    conf = SYSTEM_MAP[sys_key]
    # Map metric names to folder names
    mapping = {
        "ftle": f"ftle_{direction}", 
        "fli": f"fli_{direction}", 
        "ld": f"ld_{direction}", 
        "stod": f"local_stod_{direction}", 
        "finstod": f"local_finstod_{direction}"
    }
    subdir = mapping.get(metric.lower())
    
    prefix = "local_" if "tod" in metric.lower() else ""
    suffix = "type" if get_type else "raw"
    
    for fmt in [f"{float(t_snap):.2f}", f"{float(t_snap):.1f}"]:
        fpath = os.path.join(BASE_DATA_PATH, conf["dir"], subdir, f"{prefix}{metric.lower()}_{direction}_{suffix}_snap_{fmt}.npy")
        if os.path.exists(fpath):
            data = np.load(fpath)
            if apply_log and not get_type:
                if "tod" in metric.lower():
                    return np.log1p(data)
                else:
                    return np.log(data - np.nanmin(data) + 1e-10)
            return data
    return None

def setup_ax(ax, title, conf, is_bl):
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis='both', labelsize=12, direction='out', labelbottom=is_bl, labelleft=is_bl)
    if is_bl:
        ax.set_xlabel(conf["xlabel"], fontsize=14)
        ax.set_ylabel(conf["ylabel"], fontsize=14)

def add_segmented_colorbar(fig, ax, im, type_data, P):
    """
    Add a colorbar with T/UC/UU typology labels for FinSTOD plots.
    
    Parameters:
    -----------
    fig : matplotlib figure
    ax : matplotlib axes containing the image
    im : the imshow object
    type_data : array with type codes (0=T, 1=UC, 2=UU)
    P : the boundary between T and UC regions (from normalize_segmented)
    """
    cbar = plt.colorbar(im, ax=ax, shrink=1.0)
    cbar.ax.tick_params(labelsize=12)
    
    # Separator line between T and UC regions
    if 0 < P < 1:
        cbar.ax.hlines(P, 0, 1, colors='white', linewidth=1.5)
    
    # Text with stroke effect for visibility
    text_effects = [pe.withStroke(linewidth=2.5, foreground='black')]
    
    # UU label (always at top)
    cbar.ax.text(3.5, 1.0, "UU", va='center', ha='left', fontsize=10, color='black')
    
    # UC label (if present)
    if np.any(type_data == 1):
        y_uc = P + (1.0 - P) / 2.0
        txt = cbar.ax.text(0.5, y_uc, "UC", va='center', ha='center', fontsize=8, fontweight='bold', color='white')
        txt.set_path_effects(text_effects)
    
    # T label (if present)
    if np.any(type_data == 0):
        y_t = P / 2.0
        txt = cbar.ax.text(0.5, y_t, "T", va='center', ha='center', fontsize=8, fontweight='bold', color='white')
        txt.set_path_effects(text_effects)

def fix_fig_for_video(rows, cols, sys_key, base_size=5, dpi=150):
    asp = SYSTEM_MAP[sys_key]["aspect"]
    w_inches, h_inches = cols * base_size, rows * base_size * asp
    if int(w_inches * dpi) % 2 != 0: w_inches += 1/dpi
    if int(h_inches * dpi) % 2 != 0: h_inches += 1/dpi
    return plt.subplots(rows, cols, figsize=(w_inches, h_inches), dpi=dpi)

# =============================================================================
# STATICS (Matching Inventory V5o)
# =============================================================================

def process_statics():
    print(">>> Generating All Static Figures...")
    
    # 1. Linear Saddles (Backward & Forward)
    for d in ["backward", "forward"]:
        conf = SYSTEM_MAP["linear"]
        asp = conf["aspect"]
        fig, axs = plt.subplots(1, 2, figsize=(10, 5 * asp))
        t = 15 if d == "backward" else 0
        # Use log version for LD in linear
        ld = load_data("linear", "ld", d, t, apply_log=True)
        axs[0].imshow(normalize_minmax(ld), origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto')
        setup_ax(axs[0], f"LD (Log) $\\tau=15$", conf, True)
        score = load_data("linear", "finstod", d, t)
        dtype = load_data("linear", "finstod", d, t, get_type=True)
        norm, P = normalize_segmented(score, dtype)
        im = axs[1].imshow(norm, origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto', vmin=0, vmax=1)
        setup_ax(axs[1], f"FinSTOD $\\tau=15$", conf, False)
        
        # Add colorbar with T/UC/UU typology only for forward (first figure in paper)
        if d == "forward":
            add_segmented_colorbar(fig, axs[1], im, dtype, P)
        
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"linear_{d}_1x2_tau15.pdf"), dpi=600)
        fig.savefig(os.path.join(OUTPUT_DIR, f"linear_{d}_1x2_tau15.png"), dpi=300)
        plt.close()

    # 2. Nonlinear Saddles (2x2s) - MUTED
    # conf = SYSTEM_MAP["nonlinear"]
    # asp = conf["aspect"]
    # nonlinear_tasks = [
    #     ("backward", 11, "nonlinear_back_tau11_2x2"),
    #     ("backward", 20, "nonlinear_back_tau20_2x2"),
    #     ("forward", 0, "nonlinear_forw_tau20_2x2"),
    #     ("forward", 11, "nonlinear_forw_tau9_2x2")
    # ]
    # for direction, t, name in nonlinear_tasks:
    #     ...

    # 2.5 Pendulum
    conf = SYSTEM_MAP["pendulum"]
    asp = conf["aspect"]
    T_max = conf["T_max"]
    
    # Figure 1: Forward t=0 (tau=20) - FTLE, FLI, LD (Raw) + FinSTOD (Log)
    fig, axs = plt.subplots(2, 2, figsize=(8, 8 * asp))
    t_snap = 0
    tau = T_max - t_snap  # tau = 20
    for idx, (m, lbl) in enumerate([("ftle","FTLE"), ("fli","FLI"), ("ld","LD"), ("finstod","FinSTOD")]):
        ax = axs.flatten()[idx]
        is_log = (m == "finstod")  # Only FinSTOD in log
        data = load_data("pendulum", m, "forward", t_snap, apply_log=is_log)
        if m == "finstod":
            dtype = load_data("pendulum", m, "forward", t_snap, get_type=True); norm, _ = normalize_segmented(data, dtype)
            ax.imshow(norm, origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto', vmin=0, vmax=1)
            lbl_final = f"{lbl} (Log)"
        else:
            ax.imshow(normalize_minmax(data), origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto')
            lbl_final = lbl
        setup_ax(ax, f"{lbl_final} $\\tau={tau:.1f}$", conf, idx==2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"pendulum_forw_tau20_raw_2x2.pdf"), dpi=600)
    fig.savefig(os.path.join(OUTPUT_DIR, f"pendulum_forw_tau20_raw_2x2.png"), dpi=300)
    plt.close()
    
    # Figure 2: Forward t=0 (tau=20) - ALL Log
    fig, axs = plt.subplots(2, 2, figsize=(8, 8 * asp))
    for idx, (m, lbl) in enumerate([("ftle","FTLE"), ("fli","FLI"), ("ld","LD"), ("finstod","FinSTOD")]):
        ax = axs.flatten()[idx]
        # All metrics use log transformation
        data = load_data("pendulum", m, "forward", t_snap, apply_log=True)
        if m == "finstod":
            dtype = load_data("pendulum", m, "forward", t_snap, get_type=True)
            norm, _ = normalize_segmented(data, dtype)
            ax.imshow(norm, origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto', vmin=0, vmax=1)
        else:
            ax.imshow(normalize_minmax(data), origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto')
        setup_ax(ax, f"{lbl} (Log) $\\tau={tau:.1f}$", conf, idx==2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"pendulum_forw_tau20_log_2x2.pdf"), dpi=600)
    fig.savefig(os.path.join(OUTPUT_DIR, f"pendulum_forw_tau20_log_2x2.png"), dpi=300)
    plt.close()
    
    # Figure 3: Forward t=0 (tau=20) - Solo FinSTOD Raw
    fig, ax = plt.subplots(1, 1, figsize=(6, 6 * asp))
    data = load_data("pendulum", "finstod", "forward", t_snap, apply_log=False)
    dtype = load_data("pendulum", "finstod", "forward", t_snap, get_type=True)
    norm, _ = normalize_segmented(data, dtype)
    ax.imshow(norm, origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto', vmin=0, vmax=1)
    setup_ax(ax, f"FinSTOD $\\tau={tau:.1f}$", conf, True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"pendulum_forw_tau20_finstod_raw.pdf"), dpi=600)
    fig.savefig(os.path.join(OUTPUT_DIR, f"pendulum_forw_tau20_finstod_raw.png"), dpi=300)
    plt.close()
    
    # Figure 4: Backward t=20 (tau=20) - FTLE, FLI, LD (Raw) + FinSTOD (Log)
    fig, axs = plt.subplots(2, 2, figsize=(8, 8 * asp))
    t_snap_back = 20
    tau_back = t_snap_back  # For backward, tau = t
    for idx, (m, lbl) in enumerate([("ftle","FTLE"), ("fli","FLI"), ("ld","LD"), ("finstod","FinSTOD")]):
        ax = axs.flatten()[idx]
        is_log = (m == "finstod")  # Only FinSTOD in log
        data = load_data("pendulum", m, "backward", t_snap_back, apply_log=is_log)
        if m == "finstod":
            dtype = load_data("pendulum", m, "backward", t_snap_back, get_type=True); norm, _ = normalize_segmented(data, dtype)
            ax.imshow(norm, origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto', vmin=0, vmax=1)
            lbl_final = f"{lbl} (Log)"
        else:
            ax.imshow(normalize_minmax(data), origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto')
            lbl_final = lbl
        setup_ax(ax, f"{lbl_final} $\\tau={tau_back:.1f}$", conf, idx==2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"pendulum_back_tau20_raw_2x2.pdf"), dpi=600)
    fig.savefig(os.path.join(OUTPUT_DIR, f"pendulum_back_tau20_raw_2x2.png"), dpi=300)
    plt.close()

    # 3. Global Grids & Comparisons
    tasks = [
        ("lorenz", "forward", [9, 7, 5, 0], "lorenz_forward_4x4"),
        ("duffing", "backward", [1, 2, 3, 4, 5], "duffing_back_5x4"),
        ("duffing", "backward", [10, 11, 12, 13, 14, 15], "duffing_back_6x4"),
        ("duffing", "forward", [19, 18, 17, 16, 15], "duffing_forw_5x4"),
        ("duffing", "forward", [4, 3, 2, 1], "duffing_forw_4x4"),
        ("doublegyre", "forward", [14, 13, 12, 11, 10], "dg_forw_5x4_A"),
        ("doublegyre", "forward", [5, 4, 3, 2, 1], "dg_forw_5x4_B")
    ]
    for skey, direction, times, name in tasks:
        conf = SYSTEM_MAP[skey]; rows = len(times); asp = conf["aspect"]
        fig, axs = plt.subplots(rows, 4, figsize=(12, 3 * asp * rows))
        for i, t in enumerate(times):
            tau = t if direction == "backward" else (conf["T_max"] - t)
            for j, (m, lbl) in enumerate([("ftle","FTLE"), ("fli","FLI"), ("ld","LD"), ("finstod","FinSTOD")]):
                ax = axs[i, j]; data = load_data(skey, m, direction, t, apply_log=(m=="finstod"))
                if m == "finstod":
                    dtype = load_data(skey, m, direction, t, get_type=True); norm, _ = normalize_segmented(data, dtype)
                    ax.imshow(norm, origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto', vmin=0, vmax=1)
                else:
                    ax.imshow(normalize_minmax(data), origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto')
                setup_ax(ax, f"{lbl} $\\tau={tau:.1f}$", conf, (i==rows-1 and j==0))
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"), dpi=600)
        fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=300)
        plt.close()

    # 4. Specific 2x2 Comparisons
    for skey, direction, t, name in [
        ("duffing", "backward", 20, "duffing_back_tau20_2x2"),
        ("duffing", "backward", 10, "duffing_back_tau10_2x2"),
        ("duffing", "forward", 0, "duffing_forw_tau20_2x2"),
        ("doublegyre", "forward", 5, "dg_forw_tau10_2x2")
    ]:
        conf = SYSTEM_MAP[skey]; asp = conf["aspect"]
        fig, axs = plt.subplots(2, 2, figsize=(8, 8 * asp))
        tau = t if direction == "backward" else (conf["T_max"] - t)
        for idx, (m, lbl) in enumerate([("ftle","FTLE"), ("fli","FLI"), ("ld","LD"), ("finstod","FinSTOD")]):
            ax = axs.flatten()[idx]; data = load_data(skey, m, direction, t, apply_log=(m=="finstod"))
            if m == "finstod":
                dtype = load_data(skey, m, direction, t, get_type=True); norm, _ = normalize_segmented(data, dtype)
                ax.imshow(norm, origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto', vmin=0, vmax=1)
            else:
                ax.imshow(normalize_minmax(data), origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto')
            setup_ax(ax, f"{lbl} $\\tau={tau:.1f}$", conf, idx==2)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"), dpi=600)
        fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=300)
        plt.close()

# =============================================================================
# VIDEOS (Matching Inventory V5o)
# =============================================================================

def process_videos():
    print(">>> Generating All Videos...")
    for skey in SYSTEM_MAP.keys():
        if skey == "nonlinear":
            continue  # Muted for now
        conf = SYSTEM_MAP[skey]
        for direction in ["forward", "backward"]:
            dir_path = os.path.join(BASE_DATA_PATH, conf["dir"], f"ftle_{direction}")
            if not os.path.exists(dir_path): continue
            raw_snaps = []
            for f in os.listdir(dir_path):
                if "snap_" in f and f.endswith(".npy"):
                    t_val = float(f.split("snap_")[1].replace(".npy",""))
                    tau = t_val if direction == "backward" else (conf["T_max"] - t_val)
                    if tau > 0: raw_snaps.append((t_val, tau))
            snaps = sorted(raw_snaps, key=lambda x: x[1])
            if not snaps: continue

            # Logic: Saddles = Log, Global = Raw (Per V5o inventory list)
            use_log_solo = (skey in ["linear", "nonlinear"])
            solo_suffix = "SoloLog" if use_log_solo else "SoloRaw"
            
            # --- Solo Video ---
            fig_s, ax_s = fix_fig_for_video(1, 1, skey, base_size=6)
            writer = animation.FFMpegWriter(fps=2, bitrate=5000)
            
            # Prefix for ordering in repository
            order_map = {"linear": "A", "pendulum": "B", "lorenz": "C", "duffing": "D", "doublegyre": "E"}
            prefix_order = order_map.get(skey, "Z")
            
            # Solo video logic
            if skey == "pendulum":
                # Pendulum solo is FinSTOD Raw
                use_log_solo = False
            elif skey in ["linear", "nonlinear"]:
                # Saddles solo is FinSTOD Log
                use_log_solo = True
            else:
                use_log_solo = False
                
            solo_suffix = "SoloLog" if use_log_solo else "SoloRaw"
            
            with writer.saving(fig_s, os.path.join(OUTPUT_DIR, f"{prefix_order}_{skey}_{direction}_{solo_suffix}.mp4"), dpi=150):
                for t, tau in snaps:
                    ax_s.clear()
                    score = load_data(skey, "finstod", direction, t, apply_log=use_log_solo)
                    dtype = load_data(skey, "finstod", direction, t, get_type=True)
                    norm, _ = normalize_segmented(score, dtype)
                    ax_s.imshow(norm, origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto', vmin=0, vmax=1)
                    title_solo = f"FinSTOD (Log) $\\tau={tau:.2f}$" if use_log_solo else f"FinSTOD $\\tau={tau:.2f}$"
                    setup_ax(ax_s, title_solo, conf, True)
                    fig_s.tight_layout(); writer.grab_frame()
            plt.close(fig_s)

            # --- Comparison Video ---
            if skey == "linear":
                # Linear uses 1x2 layout (LD vs FinSTOD Raw) matching the static figures
                fig_c, axs_c = fix_fig_for_video(1, 2, skey, base_size=5)
                with writer.saving(fig_c, os.path.join(OUTPUT_DIR, f"{prefix_order}_{skey}_{direction}_Comparison.mp4"), dpi=150):
                    for t, tau in snaps:
                        for idx, (m, lbl) in enumerate([("ld","LD"), ("finstod","FinSTOD")]):
                            ax = axs_c.flatten()[idx]; ax.clear()
                            # Linear uses Log for LD, Raw for FinSTOD
                            is_log = (m == "ld")
                            data = load_data(skey, m, direction, t, apply_log=is_log)
                            if m == "finstod":
                                dtype = load_data(skey, m, direction, t, get_type=True); norm, _ = normalize_segmented(data, dtype)
                                ax.imshow(norm, origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto', vmin=0, vmax=1)
                                lbl_final = "FinSTOD"
                            else:
                                ax.imshow(normalize_minmax(data), origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto')
                                lbl_final = f"{lbl} (Log)"
                            setup_ax(ax, f"{lbl_final} $\\tau={tau:.2f}$", conf, idx==0)
                        fig_c.tight_layout(); writer.grab_frame()
                plt.close(fig_c)
            elif skey == "pendulum":
                # Pendulum comparison is 2x2 (FTLE, FLI, LD, FinSTOD)
                # All raw, but FinSTOD in log
                fig_c, axs_c = fix_fig_for_video(2, 2, skey, base_size=5)
                with writer.saving(fig_c, os.path.join(OUTPUT_DIR, f"{skey}_{direction}_Comparison.mp4"), dpi=150):
                    for t, tau in snaps:
                        for idx, (m, lbl) in enumerate([("ftle","FTLE"), ("fli","FLI"), ("ld","LD"), ("finstod","FinSTOD")]):
                            ax = axs_c.flatten()[idx]; ax.clear()
                            is_log = (m == "finstod")
                            data = load_data(skey, m, direction, t, apply_log=is_log)
                            if m == "finstod":
                                dtype = load_data(skey, m, direction, t, get_type=True); norm, _ = normalize_segmented(data, dtype)
                                ax.imshow(norm, origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto', vmin=0, vmax=1)
                                lbl_final = f"FinSTOD (Log)"
                            else:
                                ax.imshow(normalize_minmax(data), origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto')
                                lbl_final = lbl
                            setup_ax(ax, f"{lbl_final} $\\tau={tau:.2f}$", conf, idx==2)
                        fig_c.tight_layout(); writer.grab_frame()
                plt.close(fig_c)
            elif skey == "nonlinear":
                # Nonlinear comparison is 2x2 (FTLE, FLI, LD, FinSTOD)
                # Standard metrics in log, FinSTOD in raw
                fig_c, axs_c = fix_fig_for_video(2, 2, skey, base_size=5)
                with writer.saving(fig_c, os.path.join(OUTPUT_DIR, f"{skey}_{direction}_Comparison.mp4"), dpi=150):
                    for t, tau in snaps:
                        for idx, (m, lbl) in enumerate([("ftle","FTLE"), ("fli","FLI"), ("ld","LD"), ("finstod","FinSTOD")]):
                            ax = axs_c.flatten()[idx]; ax.clear()
                            is_log = (m != "finstod")
                            data = load_data(skey, m, direction, t, apply_log=is_log)
                            if m == "finstod":
                                dtype = load_data(skey, m, direction, t, get_type=True); norm, _ = normalize_segmented(data, dtype)
                                ax.imshow(norm, origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto', vmin=0, vmax=1)
                                lbl_final = "FinSTOD"
                            else:
                                ax.imshow(normalize_minmax(data), origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto')
                                lbl_final = f"{lbl} (Log)"
                            setup_ax(ax, f"{lbl_final} $\\tau={tau:.2f}$", conf, idx==2)
                        fig_c.tight_layout(); writer.grab_frame()
                plt.close(fig_c)
            else:
                # Other systems use 2x2 layout (FTLE, FLI, LD, FinSTOD)
                # Standard metrics raw, FinSTOD log
                fig_c, axs_c = fix_fig_for_video(2, 2, skey, base_size=5)
                with writer.saving(fig_c, os.path.join(OUTPUT_DIR, f"{skey}_{direction}_Comparison.mp4"), dpi=150):
                    for t, tau in snaps:
                        for idx, (m, lbl) in enumerate([("ftle","FTLE"), ("fli","FLI"), ("ld","LD"), ("finstod","FinSTOD")]):
                            ax = axs_c.flatten()[idx]; ax.clear()
                            # Comparisons in Global are typically Log for FinSTOD
                            is_log = (m == "finstod") 
                            data = load_data(skey, m, direction, t, apply_log=is_log)
                            if m == "finstod":
                                dtype = load_data(skey, m, direction, t, get_type=True); norm, _ = normalize_segmented(data, dtype)
                                ax.imshow(norm, origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto', vmin=0, vmax=1)
                                lbl_final = f"FinSTOD (Log)"
                            else:
                                ax.imshow(normalize_minmax(data), origin='lower', extent=conf["extent"], cmap='viridis', aspect='auto')
                                lbl_final = lbl
                            setup_ax(ax, f"{lbl_final} $\\tau={tau:.2f}$", conf, idx==2)
                        fig_c.tight_layout(); writer.grab_frame()
                plt.close(fig_c)

if __name__ == "__main__":
    process_statics()
    process_videos()
    print(f"\nAll inventory files from V5o successfully recreated in: {OUTPUT_DIR}")
