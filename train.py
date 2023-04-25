import cv2 as cv
from imageai.Detection import ObjectDetection as od
from imageai.Detection import ObjectDetection

import numpy as np
import requests as req
import os as os
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="/home/link232323/model/")
trainer.setTrainConfig(object_names_array=["checker", "concrete", "triangle"], batch_size=4, num_experiments=200, train_from_pretrained_model="/home/link232323/yolov3.pt")
trainer.trainModel()