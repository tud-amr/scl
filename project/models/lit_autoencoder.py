from argparse import ArgumentParser
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn.functional as F


class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 512)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(8, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(-1, 512)
        z = self.fc1(z)
        return z

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        z = z.view(-1, 512)
        z = self.fc1(z)
        z = self.fc2(z)
        z = z.view(-1, 8, 8, 8)
        z = self.decoder(z)
        loss = F.mse_loss(z, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        z = z.view(-1, 512)
        z = self.fc1(z)
        z = self.fc2(z)
        z = z.view(-1, 8, 8, 8)
        z = self.decoder(z)
        loss = F.mse_loss(z, x)
        self.log('val_loss', loss)
        return loss.detach().clone()

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        z = z.view(-1, 512)
        z = self.fc1(z)
        z = self.fc2(z)
        z = z.view(-1, 8, 8, 8)
        z = self.decoder(z)
        loss = F.mse_loss(z, x)
        self.log('test_loss', loss)
        return loss.detach().clone()

    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o for o in outputs]) / len(outputs)
        self.log('val_loss', val_loss_mean, prog_bar=True)

    def on_save_checkpoint(self, checkpoint):
        torch.save(self.state_dict(), 'eth_autoencoder.h5')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--data_dir', type=str, default='../../data/mnist')
    parser.add_argument('--save_dir', type=str, default='../../saves/autoencoder')
    parser.add_argument('--save_name', type=str, default='default')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    transform = transforms.Compose([
        transforms.Resize((60, 60)),
        transforms.ToTensor()
    ])
    dataset = MNIST(args.data_dir, train=True, download=True, transform=transform)
    mnist_test = MNIST(args.data_dir, train=False, download=True, transform=transform)
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, num_workers=args.num_workers)

    # ------------
    # model
    # ------------
    model = LitAutoEncoder()

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=f"{args.save_dir}/{args.save_name}",
    )
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    # print(result)


if __name__ == '__main__':
    main()
