import torch
import random
import numpy as np
import cv2

class RandomFlip:
    def __call__(self, tensor):
        # tensor: (C, H, W)
        if random.random() < 0.5:  # flip orizzontale
            tensor = torch.flip(tensor, dims=[2])
        if random.random() < 0.5:  # flip verticale
            tensor = torch.flip(tensor, dims=[1])
        return tensor

class RandomRotate:
    def __init__(self, degrees=10):
        self.degrees = degrees

    def __call__(self, tensor):
        angle = random.uniform(-self.degrees, self.degrees)
        c, h, w = tensor.shape
        rotated = []
        for ch in range(c):
            arr = tensor[ch].numpy()
            m = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            arr_rot = cv2.warpAffine(arr, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            rotated.append(arr_rot)
        rotated = np.stack(rotated, axis=0)
        return torch.from_numpy(rotated).float()

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

class ElasticDeformation:
    def __init__(self, alpha=20, sigma=4):
        """
        alpha: intensità dello spostamento
        sigma: smoothing della deformazione
        """
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, tensor):
        c, h, w = tensor.shape

        # Genera campi di spostamento casuali
        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (17, 17), self.sigma) * self.alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (17, 17), self.sigma) * self.alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        deformed = []
        for ch in range(c):
            arr = tensor[ch].numpy()
            warped = cv2.remap(arr, map_x, map_y,
                               interpolation=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
            deformed.append(warped)
        return torch.from_numpy(np.stack(deformed, axis=0)).float()

class RandomScaling:
    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, tensor):
        c, h, w = tensor.shape
        scale = random.uniform(*self.scale_range)

        scaled = []
        for ch in range(c):
            arr = tensor[ch].numpy()
            resized = cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            # Se più grande → crop centrale
            if resized.shape[0] > h or resized.shape[1] > w:
                start_y = (resized.shape[0] - h) // 2
                start_x = (resized.shape[1] - w) // 2
                arr_resized = resized[start_y:start_y+h, start_x:start_x+w]
            else:
                # Se più piccolo → pad centrale
                arr_resized = np.zeros((h, w), dtype=np.float32)
                y_off = (h - resized.shape[0]) // 2
                x_off = (w - resized.shape[1]) // 2
                arr_resized[y_off:y_off+resized.shape[0], x_off:x_off+resized.shape[1]] = resized

            scaled.append(arr_resized)
        return torch.from_numpy(np.stack(scaled, axis=0)).float()

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, tensor):
        for t in self.transforms:
            tensor = t(tensor)
        return tensor
