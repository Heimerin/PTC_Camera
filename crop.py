import os
import csv
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from astropy.io import fits
from pathlib import Path

# --- Configuration ---
ROI_CONFIG_FILE = 'roi_config.csv'

def get_folder_gui():
    """Select folder containing FITS images."""
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select Folder with Source FITS Images")
    root.destroy()
    return folder_selected

def load_rois_from_csv(csv_file):
    """Reads ROI config and returns a dict of available ROIs."""
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return None
    
    rois = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rois[row['name']] = {
                'x': int(row['x_start']),
                'y': int(row['y_start']),
                'w': int(row['width']),
                'h': int(row['height'])
            }
    return rois

def select_roi_dialog(rois):
    """GUI to let user choose one ROI from the list."""
    if not rois: return None
    
    roi_names = list(rois.keys())
    
    # Simple workaround for a selection list using index input
    msg = "Available ROIs:\n"
    for i, name in enumerate(roi_names):
        msg += f"[{i}] {name}\n"
    
    root = tk.Tk()
    root.withdraw()
    
    choice = simpledialog.askinteger("Select ROI", msg + "\nEnter ROI number:")
    root.destroy()
    
    if choice is not None and 0 <= choice < len(roi_names):
        return roi_names[choice], rois[roi_names[choice]]
    return None, None

def crop_and_save(src_folder, roi_name, roi_data):
    """Batch processes all FITS files in the folder."""
    
    # 1. Create Output Directory
    # Sanitize name to be safe for folders
    safe_roi_name = "".join([c for c in roi_name if c.isalnum() or c in (' ', '_', '-')]).strip()
    out_dir = os.path.join(src_folder, f"Cropped_{safe_roi_name}")
    os.makedirs(out_dir, exist_ok=True)
    
    # ROI Coords
    x, y, w, h = roi_data['x'], roi_data['y'], roi_data['w'], roi_data['h']
    
    # Verify Bayer Phase Safety (Scientific Requirement)
    # If x or y is odd, it flips the Bayer pattern (RGGB -> GRBG). Warn or fix.
    if x % 2 != 0 or y % 2 != 0:
        print(f"WARNING: ROI '{roi_name}' starts at odd coordinates ({x},{y}).")
        print("This may break Bayer demosaicing. Shifting to even numbers...")
        x = x - (x % 2)
        y = y - (y % 2)

    files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.fits', '.fit', '.fts'))]
    
    print(f"Processing {len(files)} files into '{out_dir}'...")
    
    for filename in files:
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(out_dir, filename)
        
        try:
            with fits.open(src_path) as hdul:
                # Handle Data
                data = hdul[0].data
                header = hdul[0].header
                
                # Check for extensions if primary is empty
                if data is None and len(hdul) > 1:
                    data = hdul[1].data
                    header = hdul[1].header

                # Handle 3D (RGB) vs 2D (Mono/Raw)
                if data.ndim == 3:
                    # Determine axis order: (3, H, W) vs (H, W, 3)
                    if data.shape[0] == 3: # (3, H, W)
                        cropped_data = data[:, y:y+h, x:x+w]
                    else: # (H, W, 3)
                        cropped_data = data[y:y+h, x:x+w, :]
                else:
                    # 2D Case
                    cropped_data = data[y:y+h, x:x+w]
                
                # Update Header (Critical for Astrometry)
                # CRPIX maps pixel coords to sky coords. 
                # Since we cropped x pixels from left, the new reference pixel shifts by -x.
                if 'CRPIX1' in header: header['CRPIX1'] -= x
                if 'CRPIX2' in header: header['CRPIX2'] -= y
                
                # Add History
                header.add_history(f"Cropped to ROI '{roi_name}' at [{x}:{x+w}, {y}:{y+h}]")
                
                # Save to new file
                fits.writeto(dst_path, cropped_data, header, overwrite=True)
                print(f"Saved: {filename}")
                
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    # Success Message
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Done", f"Batch processing complete.\nSaved {len(files)} files to:\n{out_dir}")
    root.destroy()

if __name__ == "__main__":
    # 1. Load ROIs
    available_rois = load_rois_from_csv(ROI_CONFIG_FILE)
    
    if not available_rois:
        print("No ROIs found in csv.")
    else:
        # 2. Select ROI
        selected_name, selected_data = select_roi_dialog(available_rois)
        
        if selected_name:
            # 3. Select Folder
            src_folder = get_folder_gui()
            
            if src_folder:
                # 4. Execute
                crop_and_save(src_folder, selected_name, selected_data)
            else:
                print("No folder selected.")
        else:
            print("No ROI selected.")
