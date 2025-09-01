import numpy as np
from skimage.color import lab2rgb #tlumacz dla modelu

def lab_to_rgb(L, ab): #czlowiek widziw rgb
    
    #L: [N,1,H,W], ab: [N,2,H,W]  → RGB [N,H,W,3]
 
    L = L * 100. #przywracamy skale
    ab = ab * 128.
    Lab = np.concatenate([L, ab], axis=1).transpose(0,2,3,1)
    rgb = []
    for img in Lab:
        rgb.append(lab2rgb(img))
    return np.stack(rgb, axis=0)
