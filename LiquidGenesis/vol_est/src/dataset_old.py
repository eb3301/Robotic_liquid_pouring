import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transforms import Compose, RandomFlip, RandomRotate, AddGaussianNoise

class LiquidVolumeDataset2(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, target_size=None):
        """
        csv_path: percorso al CSV (es. 'processed/samples.csv')
        root_dir: cartella radice che contiene i file (es. 'processed')
        transform: eventuali trasformazioni aggiuntive sui tensori (callable)
        target_size: (H, W) per ridimensionare le maschere, None = dimensione originale
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            self.samples = list(reader)

        if not self.samples:
            raise RuntimeError(f"Nessun campione trovato in {csv_path}")

    def __len__(self):
        return len(self.samples)

    # def load_npy(self, rel_path):
    #     path = os.path.join(self.root_dir, rel_path)
    #     arr = np.load(path).astype(np.float32)

    #     # Ridimensiona se target_size è impostato
    #     if self.target_size is not None and arr.shape != self.target_size:
    #         import cv2
    #         arr = cv2.resize(arr, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)

    #     return arr

    def load_npy(self, rel_path):
        path = os.path.join(self.root_dir, rel_path)
        arr = np.load(path).astype(np.float32)

        # Ridimensiona se target_size è impostato, mantenendo l'aspect ratio
        if self.target_size is not None and arr.shape != self.target_size:
            import cv2
            
            h_orig, w_orig = arr.shape
            h_target, w_target = self.target_size

            # Calcola il fattore di scala per mantenere l'aspect ratio
            scale = min(h_target / h_orig, w_target / w_orig)
            
            new_h = int(h_orig * scale)
            new_w = int(w_orig * scale)

            # Ridimensiona l'immagine
            resized_arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # Crea una nuova matrice per l'immagine finale con padding
            padded_arr = np.zeros((h_target, w_target), dtype=np.float32)

            # Calcola il punto di inserimento per centrare l'immagine
            start_h = (h_target - new_h) // 2
            start_w = (w_target - new_w) // 2
            
            padded_arr[start_h:start_h+new_h, start_w:start_w+new_w] = resized_arr
            
            arr = padded_arr

        return arr
    
    def __getitem__(self, idx):
        row = self.samples[idx]

        liquid = self.load_npy(row["liquid_path"])
        container = self.load_npy(row["container_path"])

        # Normalizzazione: se profondità, possiamo normalizzare in 0-1
        max_val = max(np.max(liquid), np.max(container), 1e-6)
        liquid = liquid / max_val
        container = container / max_val

        # Stack canali: shape -> (2, H, W)
        masks = np.stack([liquid, container], axis=0)
        masks_tensor = torch.from_numpy(masks).float()

        # Volume target
        volume = torch.tensor(float(row["volume_ml"]), dtype=torch.float32)

        # Applica eventuale transform
        if self.transform:
            masks_tensor = self.transform(masks_tensor)

        return masks_tensor, volume

class LiquidVolumeDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, target_size=(160, 214)):
        """
        csv_path: percorso al CSV (es. 'processed/samples.csv')
        root_dir: cartella radice che contiene i file (es. 'processed')
        transform: eventuali trasformazioni aggiuntive sui tensori (callable)
        target_size: (H, W) di output per le depth map
        log_depth: se True, applica exp() per convertire da log a lineare
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            self.samples = list(reader)

        if not self.samples:
            raise RuntimeError(f"Nessun campione trovato in {csv_path}")
        
    def _log_to_linear(self, depth):
            """
            Prova a capire se la depth è in scala logaritmica e,
            in caso positivo, applica np.exp().
            """
            mask = depth > 0
            if not mask.any():
                return depth  # nessun pixel valido

            max_val = np.max(depth[mask])
            mean_val = np.mean(depth[mask])

            # euristica: se i valori sono piccoli (tipicamente log) e
            # exp() produce valori molto più grandi, assumiamo che sia log-scale
            if max_val < 20 and mean_val < 10:
                depth_exp = np.exp(depth)
                if np.mean(depth_exp[mask]) > mean_val * 2:
                    return depth_exp

            return depth
    
    def _load_and_process_depth(self, rel_path):
        path = os.path.join(self.root_dir, rel_path)
        depth = np.load(path).astype(np.float32)

        # 1) Conversione log->lineare se necessario
        depth = self._log_to_linear(depth)

        # 2) Normalizzazione solo sui pixel validi (non zero)
        mask = depth != 0
        if mask.any():
            mean = np.mean(depth[mask])
            std = np.std(depth[mask])
            depth[mask] = (depth[mask] - mean + 10) / (std + 1e-6)

            # 3) Scaling con ground truth depth (mediana dei pixel validi)
            gt_depth = np.median(depth[mask])
            depth = depth * gt_depth

        # 4) Converto in tensor e resize
        depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        depth_t = F.interpolate(depth_t, size=self.target_size, mode="bilinear", align_corners=False)
        depth_t = depth_t.squeeze(0)  # [1,H,W]

        return depth_t
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        liquid_depth = self._load_and_process_depth(row["liquid_path"])
        container_depth = self._load_and_process_depth(row["container_path"])

        # Stack canali: [2,H,W]
        masks_tensor = torch.cat([liquid_depth, container_depth], dim=0)

        # Volume target
        volume = torch.tensor(float(row["volume_ml"]), dtype=torch.float32)

        # Applica eventuali trasformazioni (augmentation)
        if self.transform:
            masks_tensor = self.transform(masks_tensor)

        return masks_tensor, volume

# Esempio di utilizzo
if __name__ == "__main__":
    from transforms import Compose, RandomFlip, RandomRotate, AddGaussianNoise

    aug = Compose([
        RandomFlip(),
        RandomRotate(degrees=10),
        AddGaussianNoise(mean=0.0, std=0.005)
    ])

    dataset = LiquidVolumeDataset(
        csv_path="/home/edo/thesis/LiquidGenesis/vol_est/src/samples.csv",
        root_dir="/home/edo/thesis/LiquidGenesis/vol_est/processed",
        target_size=(160, 214),  # come nel repo GitHub
        transform=aug,
        log_depth=False  # metti True se i dati sono in scala logaritmica
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch_idx, (masks, volumes) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(" masks shape:", masks.shape)   # [B, 2, H, W]
        print(" volumes:", volumes)
        break