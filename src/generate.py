import torch
from generator import Generator
from config import config
import matplotlib.pyplot as plt
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS_IMG = config["CHANNELS_IMG"]
NOISE_DIM = config["NOISE_DIM"]
FEATURES_GEN = config["FEATURES_GEN"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    gen.load_state_dict(torch.load("gen.pt"))
    noise = torch.randn(1, NOISE_DIM, 1, 1).to(device)
    image = gen(noise)
    print(image.shape)
    plt.imshow(image.squeeze().to("cpu").permute(1, 2, 0))
    plt.show()
    save_image(image, "results.jpg", normalize=True)
