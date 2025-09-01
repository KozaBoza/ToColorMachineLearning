import torch
import torch.nn as nn #gotowe konwolujcje, aktywacje, pooling

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU() #warstwa aktywacji ReLU
        )
    def forward(self, x):
        return self.conv(x) #przepuszcza dane wejsciowe przeez blok

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = UNetBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.mid = UNetBlock(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upblock2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upblock1 = UNetBlock(128, 64)
        self.final = nn.Conv2d(64, 2, 1)  # output ab

    def forward(self, x): #modyfikuje dane wejsciowe
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        m = self.mid(p2)
        u2 = self.up2(m)
        u2 = self.upblock2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(u2)
        u1 = self.upblock1(torch.cat([u1, d1], dim=1))
        return self.final(u1)

    #conv2d - mala macierz przesuwa sie po obrazie i wykrywa okreslone wzorce
    #pooling - zmniejsza rozmiar obrazu
    #aktywacje  - wprowadza nielinowsoc
    #relu = max(0,x)