from imageai.Detection.Custom import CustomVideoObjectDetection
import os

execution_path = os.getcwd()

video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("/home/link232323/model/models/yolov3_model_mAP-0.90437_epoch-97.pt")
video_detector.setJsonPath("/home/link232323/model/json/model_yolov3_detection_config.json")
video_detector.loadModel()

video_detector.detectObjectsFromVideo(input_file_path="20230326_125808.mp4",
                                        output_file_path=os.path.join(execution_path, "campus_detected"),
                                        frames_per_second=30,
                                        minimum_percentage_probability=40,
                                        log_progress=True)