# -*- coding: utf-8 -*-
import os
from torch.utils.data import Dataset
from PIL import Image #wczytywanie obrazow
import numpy as np
import torch
import torchvision.transforms as T #resize
from skimage.color import rgb2lab #rgb -> Lab

#definiuje niestandardowy dataset do kolorowania obrazów
#L - wejscie -  jasnosc
#ab - target - kolor

class ColorizationDataset(Dataset): #konstruktor
    def __init__(self, root_dir, img_size=128, subset="train_color"):
 
        self.root_dir = os.path.join(root_dir, subset) #wybor folderu
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Folder {self.root_dir} does not exist. Check dataset structure.")

        # collecting image files
        self.files = []
        for root, _, files in os.walk(self.root_dir): #przechodzi przez podfoldery
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.files.append(os.path.join(root, f))

        if len(self.files) == 0:
            raise ValueError(f"No images found in {self.root_dir}!")

        self.transform = T.Compose([
            T.Resize((img_size, img_size)), #standarzyacja rozmiaru
            T.ToTensor() #zamiana obraz z [h,w,3] w tensor [c,h,w], skalowanie do [0,1]
        ])

    def __len__(self): #dlugosc datasetu
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transform(img).permute(1,2,0).numpy()  # rgb2lab potrzebujemy uk³adu HWC, wiêc: .permute(1,2,0) i konwersja do numpy.

        #lab i normalizacja
        lab = rgb2lab(img).astype("float32")
        lab = lab / [100., 128., 128.]  #L [0,1], ab [-1,1]
        L = lab[:,:,0:1]   # luminance [hw1]
        ab = lab[:,:,1:3]  # chrominance [hw2]

        #convert to C,H,W and PyTorch tensors
        L = torch.from_numpy(np.transpose(L, (2,0,1))).float() #(b 1 h w)
        ab = torch.from_numpy(np.transpose(ab, (2,0,1))).float() #(b 2 h w)

        return {"L": L, "ab": ab}

    #lab = L - jasnosc, a- zielonyczerwony, b - niebieskizolty
    #b - batch size, c -channels - liczba kanalow, h - height, w - width