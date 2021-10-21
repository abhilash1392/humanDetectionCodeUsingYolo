import cv2
import numpy as np
import matplotlib.pyplot as plt 

image = plt.imread('images/girl.png')

classes = None
with open('darknet/data/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]


Width = image.shape[1]
Height = image.shape[0]

# read pre-trained model and config file
net = cv2.dnn.readNet('darknet/yolov3.weights', 'darknet/cfg/yolov3.cfg')

# create input blob 
# set input blob for the network
net.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False))

# run inference through the network
# and gather predictions from output layers

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
outs = net.forward(output_layers)


class_ids = []
confidences = []
boxes = []

#create bounding box 
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.1:
            center_x = int(detection[0] * Width)
            print("confidence")



#check if is people detection

