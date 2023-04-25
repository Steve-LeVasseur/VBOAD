from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("/home/link232323/model/yolov3_model_mAP-0.89317_epoch-63.pt")
detector.setJsonPath("/home/link232323/model/json/model_yolov3_detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="sidewalk.png", output_image_path="sidewalk-detected.png")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])