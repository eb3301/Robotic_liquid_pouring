import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import LiquidVolumeDataset    
from transforms import Compose, RandomFlip, RandomRotate, AddGaussianNoise, ElasticDeformation, RandomScaling
from model import VolumeNet, VolumeNN, VolumeNetPret
import os
import matplotlib.pyplot as plt

# Config
CSV_PATH = "/home/edo/thesis/LiquidGenesis/vol_est/src/samples_new.csv"
ROOT_DIR = "/home/edo/thesis/LiquidGenesis/vol_est/processed"
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3 #/4*BATCH_SIZE
TARGET_SIZE = (256, 192) #(640, 480)
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "/home/edo/thesis/LiquidGenesis/vol_est/checkpoints/best_model_ResNet_1.pth"


def plot_predictions(model, val_loader, device, epoch, out_dir="/home/edo/thesis/LiquidGenesis/vol_est/plots"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for masks, volumes in val_loader:
            masks, volumes = masks.to(device), volumes.to(device).unsqueeze(1)
            outputs = model(masks)
            y_true.extend(volumes.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  # linea ideale
    plt.xlabel("True Volume (ml)")
    plt.ylabel("Predicted Volume (ml)")
    plt.title(f"Pred vs True (Epoch {epoch})")
    plt.savefig(f"{out_dir}/pred_vs_true_epoch{epoch}.png")
    plt.close()

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def main():
    # Augmentazione per training
    aug = Compose([
        RandomFlip(),
        RandomRotate(degrees=10),
        #ElasticDeformation(alpha=15, sigma=5),
        #RandomScaling(scale_range=(0.9, 1.1)),
        AddGaussianNoise(mean=0.0, std=0.005)
    ])

    dataset = LiquidVolumeDataset(
        csv_path=CSV_PATH,
        root_dir=ROOT_DIR,
        target_size=TARGET_SIZE,
        transform=aug
    )

    # Split train/val
    val_len = int(len(dataset) * VAL_SPLIT)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Modello
    #model = VolumeNet(input_channels=2,size=TARGET_SIZE).to(DEVICE)
    #model = VolumeNN(input_channels=4,size=TARGET_SIZE).to(DEVICE)
    model = VolumeNetPret(backbone_name="resnet18", input_channels=4, pretrained=True).to(DEVICE)

    # Loss & optimizer
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float("inf")

    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['loss']
        print(f"Riprendendo dal checkpoint, epoch {start_epoch}")

    # Start training
    print("Training started")

    early_stopping = EarlyStopping(patience=10, min_delta=0.01)

    for epoch in range(start_epoch+1, EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        for masks, volumes in train_loader:
            masks, volumes = masks.to(DEVICE), volumes.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(masks)
            loss = mse_loss(outputs, volumes)
            batch_mae = mae_loss(outputs, volumes).item()

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * masks.size(0)
            train_mae += batch_mae * masks.size(0)

        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for masks, volumes in val_loader:
                masks, volumes = masks.to(DEVICE), volumes.to(DEVICE).unsqueeze(1)
                outputs = model(masks)
                loss = mse_loss(outputs, volumes)
                batch_mae = mae_loss(outputs, volumes).item()
                val_loss += loss.item() * masks.size(0)
                val_mae += batch_mae * masks.size(0)


        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

        # Salva miglior modello
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, CHECKPOINT_PATH)
            print(f"   -> Nuovo best model salvato (Val Loss {best_val_loss:.4f})")

        if epoch % 5 == 0:
            plot_predictions(model, val_loader, DEVICE, epoch)


if __name__ == "__main__":
    main()
