import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm #pasek postepu w konsoli
import argparse #terminal cd.
from src.data import ColorizationDataset #klasa datasetu
from src.model import UNet #siec neuronowa, przewiduje kolorki
from src.utils import lab_to_rgb #konweruje wyniki z lab spowrotem do rgb
import os #foldery, zapis
import matplotlib.pyplot as plt #wynikowe obrazki

def train(args):
    dataset = ColorizationDataset(
        root_dir=args.data_dir,
        img_size=args.img_size,
        subset="train_color"
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    #przerzuca model na gpu cpu
    model = UNet().to(device)
    criterion = nn.L1Loss() #funckja strary - porownuje przewidziane preds i prawdziwe ab
    optimizer = optim.Adam(model.parameters(), lr=1e-3) #ADAM - opytmalizue, aktualizuje wagi

    os.makedirs("checkpoints", exist_ok=True) #przygotowanie folderow
    os.makedirs("samples", exist_ok=True)

    for epoch in range(args.epochs): #glowna petla treningowa
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in loop:
            L = batch["L"].to(device)   # (batch_size, 1, H, W)
            ab = batch["ab"].to(device) # (batch_size, 2, H, W)

            optimizer.zero_grad() #resetuje gradient
            preds = model(L) #przewiduje kolory
            loss = criterion(preds, ab) #jak bardzo przeiwydanie rozni sie od prawy
            loss.backward() #aktualizacja wagi sieci
            optimizer.step() 

            loop.set_postfix(loss=loss.item()) #wyswietla

        torch.save(model.state_dict(), f"checkpoints/epoch{epoch+1}.pt")

        with torch.no_grad(): #zapisywanie przykladowych wynikow
            out_ab = model(L[:2])
            out_rgb = lab_to_rgb(L[:2].cpu().numpy(), out_ab.cpu().numpy())
            for i in range(2):
                plt.imsave(f"samples/epoch{epoch+1}_sample{i}.png", out_rgb[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/images")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=128)
    args = parser.parse_args()
    train(args)

#funckja straty - liczy roznice, mniejsza strata lepsze przeiwdywania
#adam, aktualizacja wag - 