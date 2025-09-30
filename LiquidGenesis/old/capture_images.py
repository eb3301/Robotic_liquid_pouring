#!/usr/bin/env python3

import cv2
import depthai as dai
import os
import time

SAVE_DIR = "dataset_custom"
os.makedirs(SAVE_DIR, exist_ok=True)

pipeline = dai.Pipeline()
# Crea il nodo per la fotocamera
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")

# Configurazioni per alta qualità
camRgb.setPreviewSize(900, 900)  # Risoluzione 4K per la preview
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)  # Risoluzione massima del sensore
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Impostazioni di qualità aggiuntive
camRgb.setIspScale(1, 1)  # Nessun downscaling dell'ISP
camRgb.setFps(30)  # FPS ottimale per qualità

# Impostazioni di autofocus e esposizione
camRgb.initialControl.setAutoExposureEnable()
camRgb.initialControl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
camRgb.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
#camRgb.initialControl.setManualFocus(0)  # Imposta un valore di focus manuale (modifica in base alla distanza)

camRgb.preview.link(xoutRgb.input)

# Inizializza il dispositivo con il pipeline
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    print("Premi 's' per salvare un'immagine, 'q' per uscire.")
    
    while True:
        # Acquisisce un frame dalla fotocamera
        inRgb = qRgb.get()
        frame = inRgb.getCvFrame()

        # Visualizza l'immagine in tempo reale
        cv2.imshow("Live RGB", frame)

        # Stampa le dimensioni dell'immagine per debug
        #print(f"Frame dimensioni: {frame.shape}")

        # Controllo dei tasti premuti
        key = cv2.waitKey(1)
        
        if key == ord('s'):
            timestamp = int(time.time())
            filename = f"{SAVE_DIR}/img_{timestamp}.png"
            # Salva l'immagine in formato PNG con compressione minima (0)
            cv2.imwrite(filename, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"[✔] Immagine salvata: {filename}")
        
        if key == ord('q'):
            break

# Chiude tutte le finestre di OpenCV
cv2.destroyAllWindows()
