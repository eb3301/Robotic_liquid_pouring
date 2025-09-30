#!/usr/bin/env python3

import cv2
import depthai as dai

# Crea la pipeline
pipeline = dai.Pipeline()
output_streams = []
# Simula i socket delle camere (CAM_A, CAM_B, ecc.)
sockets = [dai.CameraBoardSocket.CAM_A]
for socket in sockets:
    cam = pipeline.createColorCamera()
    cam.setBoardSocket(socket)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    xout = pipeline.createXLinkOut()
    stream_name = f"cam_{socket.name}"
    xout.setStreamName(stream_name)
    cam.preview.link(xout.input)
    output_streams.append(stream_name)

with dai.Device(pipeline) as device:
    outputQueues = {name: device.getOutputQueue(name=name, maxSize=4, blocking=False) for name in output_streams}
    print("Premi 'q' per uscire.")
    while True:
        for name, queue in outputQueues.items():
            if queue.has():
                videoIn = queue.get()
                frame = videoIn.getCvFrame()
                cv2.imshow(name, frame)
        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()