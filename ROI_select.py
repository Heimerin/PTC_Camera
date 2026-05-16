import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from astropy.io import fits
import tkinter as tk
from tkinter import filedialog, simpledialog
import csv
import numpy as np

# Global variables
current_roi_info = None
roi_stats = None
roi_confirmed = False
roi_name = "ROI_01"

def get_file_path_gui():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select FITS file (RAW or RGB)",
        filetypes=[("FITS files", "*.fits;*.fit;*.fts"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

def ask_roi_name():
    root = tk.Tk()
    root.withdraw()
    name = simpledialog.askstring("ROI Name", "Enter a name for this ROI:", parent=root)
    root.destroy()
    return name

def normalize_for_display(img_data):
    """
    Normalizes data for display.
    - If 3D (RGB), normalize to 0-1 range for matplotlib.
    - If 2D (Raw), keep as is (matplotlib handles scaling via vmin/vmax).
    """
    if img_data.ndim == 3:
        # Check if data is integer or float
        dmin, dmax = img_data.min(), img_data.max()
        if dmax > dmin:
            return (img_data - dmin) / (dmax - dmin)
        else:
            return np.zeros_like(img_data)
    return img_data

def calculate_stats(data_slice):
    """
    Calculates statistics depending on dimensions.
    - 3D (RGB): Stats per Color Channel.
    - 2D (Raw): Stats using Bayer-like split (2x2 grid) or global stats.
    """
    stats_out = {}
    channels = []
    
    # CASE 1: 3D RGB Image (H, W, 3)
    if data_slice.ndim == 3:
        # data_slice is (H, W, 3)
        labels = ['Red', 'Green', 'Blue']
        for i in range(3):
            ch_data = data_slice[:, :, i]
            channels.append((labels[i], ch_data))
            
    # CASE 2: 2D Raw Image (H, W)
    else:
        # Assume Bayer pattern for scientific raw, split into 4 sub-grids
        # (Using simple 2x2 stride to separate colors without knowing exact pattern)
        channels = [
            ('Ch1', data_slice[0::2, 0::2]),
            ('Ch2', data_slice[0::2, 1::2]),
            ('Ch3', data_slice[1::2, 0::2]),
            ('Ch4', data_slice[1::2, 1::2])
        ]

    # Calculate stats for identified channels
    chan_stats = []
    for label, ch_data in channels:
        if ch_data.size == 0: continue
        
        c_min = np.min(ch_data)
        c_max = np.max(ch_data)
        c_mean = np.mean(ch_data)
        c_std = np.std(ch_data)
        
        # Uniformity: (Max - Min) / Max
        if c_max > 0:
            unif = ((c_max - c_min) / c_max) * 100.0
        else:
            unif = 0.0
            
        chan_stats.append({
            'label': label,
            'min': c_min,
            'max': c_max,
            'mean': c_mean,
            'std': c_std,
            'uniformity': unif
        })
    
    if not chan_stats: return None

    # Find Worst Case (Most variation)
    worst_case = max(chan_stats, key=lambda x: x['uniformity'])
    
    # Global average brightness
    global_mean = np.mean([x['mean'] for x in chan_stats])

    return {
        'worst_uniformity': worst_case['uniformity'],
        'worst_channel': worst_case['label'],
        'global_mean': global_mean,
        'channels': chan_stats
    }

def on_select(eclick, erelease):
    global current_roi_info, roi_stats
    
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    
    # Coordinates
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    
    width = x_max - x_min
    height = y_max - y_min
    
    if width < 1 or height < 1: return

    # Access data from axes
    ax = plt.gca()
    full_data = ax.source_data
    
    # Handle slicing
    # Note: full_data is (H, W, 3) or (H, W)
    if full_data.ndim == 3:
        roi_slice = full_data[y_min:y_max, x_min:x_max, :]
    else:
        roi_slice = full_data[y_min:y_max, x_min:x_max]
        
    if roi_slice.size == 0: return

    # Calculate Stats
    stats = calculate_stats(roi_slice)
    roi_stats = stats
    
    current_roi_info = {
        'x_start': x_min, 'y_start': y_min,
        'width': width, 'height': height
    }
    
    # Update Title
    title_txt = (
        f"ROI: {width}x{height} | Mean: {stats['global_mean']:.1f}\n"
        f"Worst Uniformity: {stats['worst_uniformity']:.2f}% ({stats['worst_channel']})\n"
        f"Press ENTER to save."
    )
    ax.set_title(title_txt, fontsize=10, backgroundcolor='white')
    plt.draw()

def on_key(event):
    global roi_confirmed, roi_name
    if event.key == 'enter':
        if current_roi_info:
            user_name = ask_roi_name()
            if user_name:
                roi_name = user_name
                roi_confirmed = True
                plt.close()

def select_roi_gui():
    fits_filename = get_file_path_gui()
    if not fits_filename: return

    try:
        with fits.open(fits_filename) as hdul:
            data = hdul[0].data
            if data is None:
                # Sometimes data is in extension 1
                if len(hdul) > 1: data = hdul[1].data
                else: raise ValueError("No data found in FITS")

            # DATA PREPROCESSING FOR 3D
            # FITS standard for RGB is usually (3, H, W). Matplotlib wants (H, W, 3).
            is_rgb = False
            if data.ndim == 3:
                is_rgb = True
                if data.shape[0] == 3: # Case (3, H, W) -> Transpose to (H, W, 3)
                    data = np.transpose(data, (1, 2, 0))
                # Else: Assume (H, W, 3), keeping as is.

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.source_data = data  # Store reference for callback
            
            if is_rgb:
                # Display RGB
                display_data = normalize_for_display(data.astype(float))
                ax.imshow(display_data, origin='lower')
                mode_str = "RGB Mode (Stats per color channel)"
            else:
                # Display RAW/Mono with False Color
                vmin, vmax = np.percentile(data, [1, 99])
                ax.imshow(data, cmap='turbo', origin='lower', vmin=vmin, vmax=vmax)
                mode_str = "RAW/Mono Mode (Stats per Bayer sub-grid)"

            ax.set_title(f"File: {fits_filename.split('/')[-1]}\n{mode_str}", fontsize=11)

            toggle_selector = RectangleSelector(
                ax, on_select, useblit=True, button=[1],
                minspanx=5, minspany=5, spancoords='pixels', interactive=True,
                props=dict(facecolor='none', edgecolor='red', linewidth=2, linestyle='--')
            )
            
            fig.canvas.mpl_connect('key_press_event', on_key)
            plt.show()

            if roi_confirmed and current_roi_info:
                save_roi_to_csv(current_roi_info, roi_stats, roi_name)

    except Exception as e:
        print(f"Error: {e}")

def save_roi_to_csv(roi, stats, name, output_csv='roi_config.csv'):
    output = {
        'name': name,
        **roi,
        'mean_adu': round(stats['global_mean'], 2),
        'uniformity_worst': round(stats['worst_uniformity'], 3),
        'worst_channel': stats['worst_channel']
    }
    
    file_exists = False
    try:
        with open(output_csv, 'r') as f: file_exists = True
    except FileNotFoundError: pass

    with open(output_csv, 'a', newline='') as f:
        fieldnames = ['name', 'x_start', 'y_start', 'width', 'height', 'mean_adu', 'uniformity_worst', 'worst_channel']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()
        writer.writerow(output)
        
    print(f"\n[SUCCESS] ROI '{name}' saved.")
    print(f"Stats: {stats['worst_uniformity']:.2f}% variation in {stats['worst_channel']} channel.")

if __name__ == "__main__":
    select_roi_gui()
