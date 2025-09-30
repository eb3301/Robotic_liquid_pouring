import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import LiquidVolumeDataset    
from transforms import Compose, RandomFlip, RandomRotate, AddGaussianNoise, ElasticDeformation, RandomScaling
from model import VolumeNN, VolumeNetPret
import os
import matplotlib.pyplot as plt

# Config
DIR = "/home/edo/thesis/LiquidGenesis/vol_est"
CSV_PATH = DIR + "/src/samples_new+real+video.csv"
ROOT_DIR = DIR + "/dataset"
BATCH_SIZE = 16 # 16, 32
EPOCHS = 200
LR = 1e-3 # 1e-3, 1e-4
TARGET_SIZE = (256, 192)
VAL_SPLIT = 0.25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = DIR + "/checkpoints/best_model_ResNet_2.pth"



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

class WeightedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        """
        alpha: peso per MSE
        beta: peso per MAE
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        return self.alpha * self.mse(y_pred, y_true) + self.beta * self.mae(y_pred, y_true)


def main():
    # Augmentazione per training
    aug = Compose([
        RandomFlip(),
        RandomRotate(degrees=10),
        # ElasticDeformation(alpha=15, sigma=5),
        # RandomScaling(scale_range=(0.9, 1.1)),
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
    # model = VolumeNN(input_channels=4, size= TARGET_SIZE)
    model = VolumeNetPret(backbone_name="ResNet18", input_channels=4, pretrained=True).to(DEVICE)

    # Loss & optimizer
    criterion = WeightedLoss(alpha=0.7, beta=0.3)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4) #1e-3, 1e-4
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

    print("Training started")
    early_stopping = EarlyStopping(patience=10, min_delta=0.01)

    for epoch in range(start_epoch+1, EPOCHS + 1):
        # Training
        model.train()
        train_loss, train_mae = 0.0, 0.0
        for masks, volumes in train_loader:
            masks, volumes = masks.to(DEVICE), volumes.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(masks)
            loss = criterion(outputs, volumes)
            batch_mae = nn.L1Loss()(outputs, volumes).item()

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * masks.size(0)
            train_mae += batch_mae * masks.size(0)

        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_mae = 0.0, 0.0
        with torch.no_grad():
            for masks, volumes in val_loader:
                masks, volumes = masks.to(DEVICE), volumes.to(DEVICE).unsqueeze(1)
                outputs = model(masks)
                loss = criterion(outputs, volumes)
                batch_mae = nn.L1Loss()(outputs, volumes).item()
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
            plot_predictions(model, val_loader, DEVICE, epoch)
            print(f"   -> Nuovo best model salvato (Val Loss {best_val_loss:.4f})")


if __name__ == "__main__":
    main()

#   TUNING IPERPARAMETRI:
#     • BATCH_SIZE (default 16)
#           ◦ Batch più grande = stime del gradiente più stabili ma maggiore consumo di RAM/VRAM.
#           ◦ Batch più piccolo = più rumore ma training più regolare, utile se la memoria è limitata.
#     • VAL_SPLIT (default 0.2 = 20%) - Percentuale del dataset usata per validazione.
#           ◦ Più alto = validazione più affidabile ma meno dati per il training.
#     • TARGET_SIZE (default (256,192)) - Risoluzione delle maschere in input.
#           ◦ Più grande = più dettagli, ma più lento e pesante.
#           ◦ Più piccolo = training veloce, ma rischio di perdere informazioni.
#     • Augmentazioni (transforms)
#           ◦ RandomFlip() → aumenta robustezza a simmetrie.
#           ◦ RandomRotate(degrees=10) → aumenta robustezza a rotazioni.
#           ◦ AddGaussianNoise(std=0.005) → migliora generalizzazione, ma troppo rumore può degradare.
#           ◦ ElasticDeformation, RandomScaling → aumentano variabilità, utili se i dati sono pochi.
#     • VolumeNetPret con backbone "ResNet18" - Architettura scelta
#           ◦ modelli più grandi (es. ResNet34, ResNet50) → maggiore capacità, ma rischio overfitting e uso di più GPU RAM.
#     • LR (Learning Rate, default 1e-3) (crit.)
#           ◦ Più alto = training veloce ma instabile/divergente.
#           ◦ Più basso = training stabile ma lento.
#     • optimizer = Adam(..., weight_decay=1e-4)
#           ◦ weight_decay = regolarizzazione L2 → riduce overfitting. Puoi provare valori diversi (1e-5, 0, ecc.).
#           ◦ Puoi anche cambiare ottimizzatore (SGD + momentum) per esperimenti.
#     • scheduler = ReduceLROnPlateau(factor=0.5, patience=3)→ Riduce il LR quando la loss di validazione non migliora.
#           ◦ factor = quanto ridurre (0.5 → dimezza il LR).
#           ◦ patience = quante epoche aspettare prima di ridurre.
#     • WeightedLoss(alpha=0.7, beta=0.3)→ Combinazione di MSE e MAE.
#           ◦ Più alpha = maggiore importanza all’errore quadratico (penalizza outlier fortemente).
#           ◦ Più beta = maggiore importanza all’errore assoluto (più robusto agli outlier).
#     • EarlyStopping(patience=10, min_delta=0.01)
#           ◦ patience = quante epoche tollerare senza miglioramento.
#           ◦ min_delta = minimo miglioramento richiesto per azzerare il contatore.
