import torch
from PIL import Image #obrazy
import torchvision.transforms as T #gotowe transformacje
import argparse, os
from src.model import UNet #siec neuronowa
from src.utils import lab_to_rgb
from skimage.color import rgb2lab
import numpy as np

def load_and_prep(img_path, size=256):
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([T.Resize((size,size)), T.ToTensor()])
    img = transform(img).permute(1,2,0).numpy()
    lab = rgb2lab(img).astype("float32")
    lab = lab / [100.,128.,128.]
    L = lab[:,:,0:1]
    L = np.transpose(L, (2,0,1))
    return torch.tensor(L).unsqueeze(0).float() #przygotowane wejscie do sieci

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    if os.path.isfile(args.input_path):
        paths = [args.input_path]
    else:
        paths = [os.path.join(args.input_path,f) for f in os.listdir(args.input_path)
                 if f.lower().endswith((".jpg",".png",".jpeg"))]

    for p in paths:
        L = load_and_prep(p, args.img_size).to(device)
        with torch.no_grad():
            ab = model(L)
            rgb = lab_to_rgb(L.cpu().numpy(), ab.cpu().numpy())[0]
        out_name = os.path.join(args.out_dir, os.path.basename(p))
        Image.fromarray((rgb*255).astype("uint8")).save(out_name)
        print("Saved:", out_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="samples")
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()
    main(args)
