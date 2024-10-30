from video_segmentation_multi_model_function import generate_segmented_video

INPUT_PATH = './input_videos/video3.mp4'
OUTPUT_PATH = './output_videos/segmentation3.mp4'

VEHICLES_MODEL_PATH = './models/vehicles/vehicles2_model.pth.tar'
PEDESTRIANS_MODEL_PATH = './models/pedestrians/pedestrians3_model.pth.tar'
ROAD_MODEL_PATH = './models/road/road_model.pth.tar'
ROAD_MARKS_MODEL_PATH ='./models/road_marks/road_marks3.pth.tar'

generate_segmented_video(INPUT_PATH,OUTPUT_PATH,
                          vehicles_model_path=VEHICLES_MODEL_PATH,
                          pedestrian_model_path=PEDESTRIANS_MODEL_PATH,
                          road_model_path=ROAD_MODEL_PATH,
                          road_marks_model_path=  ROAD_MARKS_MODEL_PATH   
                          )