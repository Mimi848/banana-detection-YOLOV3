
import cv2 as cv
import sys
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw() # GUI INPUT
filename = askopenfilename(
    title="Choose an image",
    filetypes=[('image files', ('.png', '.jpg', '.jpeg'))])

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image


classesFile = "coco.names"
classes = None

with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def drawPred(classId, conf, left, top, right, bottom):
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

def bananaFound(frame, outs):
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold and classId == 46:
                return True
    return False

if(filename):
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    # Open the image file
    frame = cv.imread(filename)
    outputFile = filename[:-5]+'_yolo_out_py.jpg'
    while cv.waitKey(1) < 0:
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))




        #DO YOUR CODE HERE
        if(bananaFound(frame, outs)):
            print("Banana Mil gia ha")
        else:
            print(" Nahiiii milaaaaaa")

        cv.waitKey(3000)
        sys.exit(1)

