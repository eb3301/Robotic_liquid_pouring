import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sde_model import SDEModel
from sde_dataset import SDUDataset
import torchvision.transforms as T
import os

# Parametri
EPOCHS = 75
BATCH_SIZE = 6
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 4e-5
SAVE_PATH = "sde_model.pth"

# Trasformazioni
transform = T.Compose([
    T.Resize((300, 300)),
    T.ToTensor(),
    T.ColorJitter(brightness=0.2, contrast=0.2),
])

# Dataset
train_dataset = SDUDataset(
    img_dir="data/images",
    mask_dir="data/masks",
    depth_dir="data/depths",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SDEModel().to(device)

# Loss
seg_criterion = nn.CrossEntropyLoss()
def scale_invariant_depth_loss(pred, target, mask):
    diff = torch.log(pred + 1e-8) - torch.log(target + 1e-8)
    diff = diff * mask  # considera solo l’oggetto
    n = mask.sum()
    if n == 0:
        return torch.tensor(0.0, device=pred.device)
    return (diff ** 2).sum() / n - ((diff.sum()) ** 2) / (n ** 2)

# Ottimizzatore
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        imgs, masks, depths = [x.to(device) for x in batch]

        optimizer.zero_grad()
        seg_logits, pred_depths = model(imgs)

        # Compute losses
        loss_seg = seg_criterion(seg_logits, masks)
        loss_depth = scale_invariant_depth_loss(pred_depths, depths, masks > 0)
        loss = loss_seg + loss_depth

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # Riduci LR se non migliora
    if epoch % 10 == 0 and epoch > 0:
        for g in optimizer.param_groups:
            g['lr'] *= 0.5

# Salva modello
torch.save(model.state_dict(), SAVE_PATH)
print(f"[✔] Modello salvato in: {SAVE_PATH}")
