import torch as T
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from typing_extensions import Self
from typing import Callable
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


LATENT_DIM: int = 128
BATCH_SIZE: int = 128
IMG_SIZE: int = 28
EPOCHS: int = 25
LR: float = 5e-4


class MNISTDataset(Dataset):
    def __init__(
        self: Self,
        filename: str = "data.csv",
        transforms: Callable = None,
        target_transforms: Callable = None,
    ) -> None:
        self.data: np.ndarray = pd.read_csv(filename).to_numpy()
        self.transforms: Callable = transforms
        self.target_transforms: Callable = target_transforms

    def __len__(self: Self) -> int:
        return len(self.data)

    def __getitem__(self: Self, idx: int) -> tuple[np.ndarray, int]:
        img: np.ndarray = self.data[idx, 1:]
        label: int = self.data[idx, 0]

        if self.transforms is not None:
            img = self.transforms(img)
        if self.target_transforms is not None:
            label = self.target_transforms(label)

        return img, label


class Generator(nn.Module):
    def __init__(self: Self, latent_dim: int) -> None:
        super().__init__()

        self.fc1: nn.Linear = nn.Linear(latent_dim, 7 * 7 * 64)
        self.ct1: nn.ConvTranspose2d = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.ct2: nn.ConvTranspose2d = nn.ConvTranspose2d(32, 16, 4, stride=2)
        self.c1: nn.Conv2d = nn.Conv2d(16, 1, kernel_size=7)

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        x = F.relu(self.fc1(x))

        # ((((7 - 1) * 2 + 4) - 1) * 2 + 4) - (7 - 1)
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.ct1(x))
        x = F.relu(self.ct2(x))
        x = self.c1(x)

        return x


class Discriminator(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()

        self.c1: nn.Conv2d = nn.Conv2d(1, 10, kernel_size=5)
        self.p1: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2)
        self.c2: nn.Conv2d = nn.Conv2d(10, 20, kernel_size=5)
        self.p2: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2)
        self.fc1: nn.Linear = nn.Linear(320, 50)
        self.fc2: nn.Linear = nn.Linear(50, 1)

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        x = F.relu(self.p1(F.dropout2d(self.c1(x))))
        x = F.relu(self.p2(F.dropout2d(self.c2(x))))
        # (((28 - (5 - 1)) / 2  - (5 - 1)) / 2) ^ 2 * 20
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))

        return x


def main() -> None:
    device: T.device = T.device("cuda" if T.cuda.is_available() else "cpu")

    t: transforms.Compose = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.reshape(IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda x: x.astype(np.float32)),
            transforms.Lambda(lambda x: x / 255),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(device)),
        ]
    )
    dataset: MNISTDataset = MNISTDataset(transforms=t)
    train_dataset, test_dataset = random_split(dataset, [0.75, 0.25])
    train_dataloader: DataLoader = DataLoader(train_dataset, BATCH_SIZE)
    test_dataloader: DataLoader = DataLoader(test_dataset, 50)

    generator: Generator = Generator(LATENT_DIM).to(device)
    discriminator: Discriminator = Discriminator().to(device)

    criterion: nn.BCELoss = nn.BCELoss()
    gen_optim: optim.Adam = optim.Adam(generator.parameters(), LR)
    disc_optim: optim.Adam = optim.Adam(discriminator.parameters(), LR)

    for epoch in range(1, EPOCHS + 1):
        for imgs, labels in train_dataloader:
            z: T.Tensor = T.randn((imgs.size(0), LATENT_DIM), device=device)
            ones: T.Tensor = T.ones((imgs.size(0), 1), device=device)
            zeros: T.Tensor = T.zeros((imgs.size(0), 1), device=device)

            # train generator
            gen_optim.zero_grad()
            fake_imgs: T.Tensor = generator(z)
            y_fake_hat: T.Tensor = discriminator(fake_imgs)
            gen_loss: T.Tensor = criterion(y_fake_hat, ones)
            gen_loss.backward()
            gen_optim.step()

            # train discriminator
            disc_optim.zero_grad()
            real_loss: T.Tensor = criterion(discriminator(imgs), ones)
            fake_loss: T.Tensor = criterion(discriminator(fake_imgs.detach()), zeros)
            disc_loss: T.Tensor = (real_loss + fake_loss) / 2
            disc_loss.backward()
            disc_optim.step()

        print(f"Epoch {epoch}")
    
    T.save(generator.state_dict(), "trained_generator.pt")
    T.save(discriminator.state_dict(), "trained_discriminator.pt")

    z: T.Tensor = T.randn((25, LATENT_DIM), device=device)
    with T.no_grad():
        imgs: T.Tensor = generator(z)
    imgs = 255 * imgs.cpu().reshape((-1, IMG_SIZE, IMG_SIZE))
    _, subplt = plt.subplots(5, 5)
    for i in range(5):
        for j in range(5):
            subplt[i][j].axis("off")
            subplt[i][j].imshow(imgs[5 * i + j], cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
