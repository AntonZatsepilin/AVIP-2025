from PIL import Image
import numpy as np
from os import path

def split_rgb_components(img_array):
    r = np.zeros_like(img_array)
    r[:, :, 0] = img_array[:, :, 0] 
    g = np.zeros_like(img_array)
    g[:, :, 1] = img_array[:, :, 1]  
    b = np.zeros_like(img_array)
    b[:, :, 2] = img_array[:, :, 2]  
    
    return r, g, b

def rgb_to_hsi(img_array):
    img_normalized = img_array / 255.0
    hsi = np.zeros_like(img_normalized)
    
    for i in range(img_normalized.shape[0]):
        for j in range(img_normalized.shape[1]):
            r, g, b = img_normalized[i, j]
        
            i_val = (r + g + b) / 3.0
            
            min_val = min(r, g, b)
            if (r + g + b) == 0:
                s_val = 0.0
            else:
                s_val = 1.0 - (3 * min_val) / (r + g + b)
            
            numerator = 0.5 * ((r - g) + (r - b))
            denominator = np.sqrt((r - g)**2 + (r - b)*(g - b)) + 1e-6
            theta = np.arccos(numerator / denominator)
            h_val = np.degrees(theta)
            if b > g:
                h_val = 360.0 - h_val
            
            hsi[i, j] = [h_val, s_val, i_val]
    
    return hsi

def hsi_to_rgb(hsi_array):
    rgb = np.zeros_like(hsi_array)
    
    for i in range(hsi_array.shape[0]):
        for j in range(hsi_array.shape[1]):
            h, s, i_val = hsi_array[i, j]
            
            if s == 0:
                r = g = b = i_val
            else:
                h = h % 360.0
                sector = h / 60.0
                sector_int = int(sector)
                fractional = sector - sector_int
                
                p = i_val * (1 - s)
                q = i_val * (1 - s * fractional)
                t = i_val * (1 - s * (1 - fractional))
                
                if sector_int == 0:
                    r, g, b = i_val, t, p
                elif sector_int == 1:
                    r, g, b = q, i_val, p
                elif sector_int == 2:
                    r, g, b = p, i_val, t
                elif sector_int == 3:
                    r, g, b = p, q, i_val
                elif sector_int == 4:
                    r, g, b = t, p, i_val
                else:
                    r, g, b = i_val, p, q
            
            rgb[i, j] = np.clip([r, g, b], 0, 1) * 255
    
    return rgb.astype(np.uint8)

def invert_intensity(img_array):
    hsi = rgb_to_hsi(img_array)
    hsi[:, :, 2] = 1.0 - hsi[:, :, 2]
    return hsi_to_rgb(hsi)