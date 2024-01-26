import cv2, time, json, argparse
from ultralytics import YOLO
from roi import ROI
from vilt import ViLTPAR
from PIL import Image
from MTNN import MultiTaskPAR

import os

def load_yolo(yolo_model_path):
    """
    Load a YOLO model.

    Parameters:
    - yolo_model_path (str): Path to the YOLO model.

    Returns:
    - YOLO: YOLO model instance.
    """
    return YOLO(yolo_model_path)

def load_vilt(vilt_model_path):
    """
    Load a ViLT model.

    Parameters:
    - vilt_model_path (str): Path to the ViLT model.

    Returns:
    - ViLTPAR: ViLT model instance.
    """
    return ViLTPAR(vilt_model_path)

def load_mtnn(mtnn_model_path):
    """
    Load a MultiTaskPAR model.

    Parameters:
    - mtnn_model_path (str): Path to the MultiTaskPAR model.

    Returns:
    - MultiTaskPAR: MultiTaskPAR model instance.
    """
    return MultiTaskPAR(mtnn_model_path)



def open_video(video_path):
    """
    Open a video file using OpenCV.

    Parameters:
    - video_path (str): Path to the video file.

    Returns:
    - cv2.VideoCapture: VideoCapture object.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        exit()
    return cap


def process_frames(yolo_model, par_model, cap, rois, tracking_data, fps):
    """
    Process frames from the video, update tracking data, and display the annotated frames.

    Parameters:
    - yolo_model (YOLO): YOLO model for object detection and tracking.
    - par_model (ViltPAR or MultiTaskPAR): model for pedestrian attribute extraction.
    - cap (cv2.VideoCapture): VideoCapture object for reading video frames.
    - rois (ROI): Instance of the ROI class containing region of interest information.
    - tracking_data (dict): Dictionary to store tracking data.
    - fps (int): Frames per second of the video.

    Returns:
    - None
    """
    # Number of frames to wait before updating tracking information
    frames_to_wait = fps * 1.3
    frame_counter = frames_to_wait

    while True:
        # Read the next frame from the video
        success, frame = cap.read()

        # Break the loop if no more frames
        if not success:
            break

        # Use YOLO model for object detection and tracking
        results = yolo_model.track(frame, persist=True, classes=0, verbose=False)

        # Compute bounding box informations
        bbinfo = calculate_bbox_info(results)

        # Decide whether to perform attribute extraction in the current frame
        if frame_counter >= frames_to_wait:
            flag_par = True
            frame_counter = 0
        else:
            flag_par = False
            frame_counter += 1

        # Update tracking data based on the current frame
        update_data(frame, bbinfo, tracking_data, rois, par_model, flag_par)

        # Display the annotated frame with bounding boxes and ROIs
        annotated_frame = plot_bboxes(bbinfo, tracking_data, frame)
        rois.add_roi_to_image(annotated_frame)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

def update_data(frame, bbinfo, tracking_data, rois, par_model, flag_par):
    """
    Update tracking data based on object positions in ROIs and PAR.

    Parameters:
    - frame (numpy.ndarray): The current frame.
    - bbinfo (list): List containing information about detected objects, including object ID, center coordinates, and angles.
    - tracking_data (dict): Dictionary to store tracking data.
    - rois (ROI): Instance of the ROI class containing region of interest information.
    - par_model: An instance of the ViltPAR or MultiTaskPAR for attribute extraction.
    - flag_par (bool): Flag to determine whether attribute extraction should be performed.
    """

    for info in bbinfo:
        # Extraction of box attributes
        obj_id = info[0]
        centers = info[1]
        # coordinates of the four corners of each box
        angles = info[2]

        # Check if the center of the box is in one of the two ROIs
        is_in_roi1, is_in_roi2 = rois.point_in_rois((centers[0], centers[1]))


        if obj_id not in tracking_data:
            # Initialize tracking data for the object ID
            tracking_data[obj_id] = {
                "gender": None,
                "hat": None,
                "bag": None,
                "upper_color": None,
                "lower_color": None,
                "roi1_passages": 0,
                "roi1_persistence_time": 0,
                "roi1_flag": False,
                "roi2_passages": 0,
                "roi2_persistence_time": 0,
                "roi2_flag": False
            }

        if flag_par:

            #Extraction of the crop for each person
            cropped_frame = crop_objects(frame, angles)
            attributes = par_model.extract_attributes(cropped_frame)

            # PAR attributes update
            tracking_data[obj_id]['gender'] = attributes[0]
            tracking_data[obj_id]['hat'] = attributes[1]
            tracking_data[obj_id]['bag'] = attributes[2]
            tracking_data[obj_id]['upper_color'] = attributes[3]
            tracking_data[obj_id]['lower_color'] = attributes[4]
        # Roi attributes update
        if is_in_roi1:
            update_roi_statistic(tracking_data, obj_id, "roi1")
        elif is_in_roi2:
            update_roi_statistic(tracking_data, obj_id, "roi2")


def update_roi_statistic(tracking_data, obj_id, roi):
    """
    Update tracking data dictionary with the number of passages and persistence time.

    Parameters:
    - tracking_data (dict): Dictionary to store tracking data.
    - obj_id (str): Object ID.
    - roi (str): Region of interest identifier ("roi1" or "roi2").
    """
    # Define strings for dictionary keys
    str1 = "_passages"
    str2 = "_persistence_time"
    flag = "_flag"

    # Check if the object has entered the ROI
    if not tracking_data[obj_id][roi + flag]:
        # Increment the number of passages
        tracking_data[obj_id][roi + str1] += 1
        # Set the flag to indicate the object has entered
        tracking_data[obj_id][roi + flag] = True

    # Increment the persistence time
    tracking_data[obj_id][roi + str2] += 1



def calculate_bbox_info(results):
    """
    Calculate the information of bounding boxes given a results object.

    Parameters:
    - results (ultralytics.YOLO): YOLO results object containing information about detected objects.

    Returns:
    - list: List of tuples, each containing the track ID and coordinates of the center and the corners of a bounding box ("id_x", (cx, cy), (x1, y1, x2, y2)).
    """
    bbinfo = []

    # Iterate through each detection in the YOLO results
    for result in results:
        # Extract bounding box information from the results
        boxes = result.boxes.cpu().numpy()

        # Check if both xyxys and ids are non-empty
        if boxes.xyxy is None or boxes.id is None:
            continue

        xyxys = boxes.xyxy
        ids = boxes.id  # Use result.id to get track IDs

        # Process each bounding box in the current detection
        for xyxy, track_id in zip(xyxys, ids):
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            # Calculate the center of the bounding box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Convert track_id to a string in the format "id_x"
            track_id_str = f"id_{int(track_id)}"

            # Append a tuple with track ID, bb center, and coordinates to the bbinfo list
            bbinfo.append((track_id_str, (cx, cy), (x1, y1, x2, y2)))

    return bbinfo



def plot_bboxes(bbinfo, tracking_data, frame):
    """
    Plot bounding boxes and associated information on the input frame.

    Parameters:
    - bbinfo (list): List containing information about bounding boxes.
    - tracking_data (dict): Dictionary containing tracking information.
    - frame (numpy.ndarray): Input frame on which bounding boxes will be drawn.

    Returns:
    - numpy.ndarray: Frame with drawn bounding boxes and labels.
    """
    for info in bbinfo:
        obj_id = info[0]
        angles = info[2]

        x1 = angles[0]
        y1 = angles[1]
        x2 = angles[2]
        y2 = angles[3]

        tracking_info = tracking_data.get(obj_id, {})

        # Draw the bounding box in red
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Add label with ID above the box (ID as an integer)
        id_label_position = (x1, y1 - 10)
        cv2.putText(frame, f'{obj_id}', id_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Add label with PAR attributes on the right of the box
        gender_label_position = (x2 + 2, y1 + 15)
        cv2.putText(frame, f"Gender: {tracking_info.get('gender', 0)}", gender_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        hat_label_position = (x2 + 2, y1 + 30)
        cv2.putText(frame, f"Hat: {tracking_info.get('hat', 0)}", hat_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        bag_label_position = (x2 + 2, y1 + 45)
        cv2.putText(frame, f"Bag: {tracking_info.get('bag', 0)}", bag_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        upcol_label_position = (x2 + 2, y1 + 60)
        cv2.putText(frame, f"Upper Color: {tracking_info.get('upper_color', 0)}", upcol_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        lowcol_label_position = (x2 + 2, y1 + 75)
        cv2.putText(frame, f"Lower Color: {tracking_info.get('lower_color', 0)}", lowcol_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return frame

def crop_objects(frame, angles):
    """
    Utility function that crops the frame to obtain images of people detected in the scene.

    Parameters:
    - frame (numpy.ndarray): Current scene frame.
    - angles (tuple): Bounding box coordinates (x1, y1, x2, y2).

    Returns:
    - Image.Image: Cropped image containing detected people.
    """
    # Extract bounding box coordinates
    x1, y1, x2, y2 = angles

    # Crop the section related to the bounded box
    cropped_image = frame[y1:y2, x1:x2].copy()

    # Convert the cropped array to an Image object
    cropped_image = Image.fromarray(cropped_image)

    return cropped_image


def save_tracking_statistics(tracking_data, output_file, fps):
    """
    Save tracking statistics for each object in the tracking data to a JSON file.

    Parameters:
    - tracking_data (dict): Dictionary containing tracking data.
    - output_file (str): The name of the output JSON file.
    - fps (float): Frames per second of the video.
    """
    output_list = []

    for obj_id, data in tracking_data.items():
        entry = {
            "id": int(obj_id.split('_')[1]),
            "gender": data.get("gender", "unknown"),
            "hat": data.get("hat", False),
            "bag": data.get("bag", False),
            "upper_color": data.get("upper_color", "unknown"),
            "lower_color": data.get("lower_color", "unknown"),
            "roi1_passages": data.get("roi1_passages", 0),
            "roi1_persistence_time": round(data.get("roi1_persistence_time", 0) / fps, 2),
            "roi2_passages": data.get("roi2_passages", 0),
            "roi2_persistence_time": round(data.get("roi2_persistence_time", 0) / fps, 2)
        }
        output_list.append(entry)

    output_data = {"people": output_list}

    with open(output_file, 'w') as json_file:
        json.dump(output_data, json_file, indent=2)


# def main():

#     # Create an argument parser with descriptions for command-line arguments
#     parser = argparse.ArgumentParser(description='Process video frames with YOLO and ViLT models.')
#     parser.add_argument('--video', type=str, required=True, help='Path to the input video file (mp4).')
#     parser.add_argument('--configuration', type=str, required=True, help='Path to the ROI configuration file (txt).')
#     parser.add_argument('--results', type=str, required=True, help='Path to the output JSON format file (txt).')

#     # Parse the command-line arguments
#     args = parser.parse_args()

#     # Load YOLO model
#     yolo_model_path = 'yolo_models/yolov8s.pt'
#     yolo_model = load_yolo(yolo_model_path)
#     yolo_model.to("cuda")

#     # variable to choose which method to use for the par
#     vilt = True
#     if vilt:
#         # Load ViLT model
#         vilt_model_path = 'dandelin/vilt-b32-finetuned-vqa'
#         vilt_model = load_vilt(vilt_model_path)
#         vilt_model.to("cuda")
#         par_model = vilt_model
#     else:
#         # Load MTNN model
#         mtnn_model_path = 'mtnn_best_model.pth'
#         mtnn_model = load_mtnn(mtnn_model_path)
#         par_model = mtnn_model


#     # Open the video file
#     video_path = args.video
#     cap = open_video(video_path)

#     # Create an ROI manager and read ROIs from the JSON file
#     roi_manager = ROI(args.configuration, video_path)

#     # Initialize tracking data dictionary
#     tracking_data = {}

#     # Get video fps
#     fps = cap.get(cv2.CAP_PROP_FPS)


#     # Process frames using YOLO, and PAR model
#     process_frames(yolo_model, par_model, cap, roi_manager, tracking_data, fps)

#     # Save tracking statistics to an output JSON file
#     save_tracking_statistics(tracking_data, args.results, fps)

# if __name__ == "__main__":
#     main()


def process_videos_in_folder(source_folder, destination_folder):
    # List all video files in the given source folder
    video_files = [f for f in os.listdir(source_folder) if f.endswith('.mp4')]

    # Process each video in the source folder
    for video_file in video_files:
        video_path = os.path.join(source_folder, video_file)

        print(f"Processing video: {video_file}")

        # Load YOLO model
        yolo_model_path = 'yolo_models/yolov8s.pt'
        yolo_model = load_yolo(yolo_model_path)
        yolo_model.to("cuda")

        # Choose between ViLT and MTNN model for attribute extraction
        vilt = True
        if vilt:
            # Load ViLT model
            vilt_model_path = 'dandelin/vilt-b32-finetuned-vqa'
            vilt_model = load_vilt(vilt_model_path)
            vilt_model.to("cuda")
            par_model = vilt_model
        else:
            # Load MTNN model
            mtnn_model_path = 'mtnn_best_model.pth'
            mtnn_model = load_mtnn(mtnn_model_path)
            mtnn_model.to("cuda")
            par_model = mtnn_model

        # Open the video file
        cap = open_video(video_path)

        # Create an ROI manager and read ROIs from the JSON file
        roi_manager = ROI('config.txt', video_path)

        # Initialize tracking data dictionary
        tracking_data = {}

        # Get video fps
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Process frames using YOLO, and PAR model
        process_frames(yolo_model, par_model, cap, roi_manager, tracking_data, fps)

        # Save tracking statistics to a single output JSON file for each video
        output_file = os.path.join(destination_folder, f'{video_file.split(".")[0]}_results.txt')
        save_tracking_statistics(tracking_data, output_file, fps)

        # Release the video capture object
        cap.release()

if __name__ == "__main__":
    # Specify the source and destination folders
    source_folder_path = 'C:\\Users\\fsarn\\ArtificialVision\\Artificial_vision\\Project\\Test_Data'
    destination_folder_path = 'C:\\Users\\fsarn\\ArtificialVision\\Artificial_vision\\Project\\Test_Data\\GT'

    # Process videos in the source folder and save results in the destination folder
    process_videos_in_folder(source_folder_path, destination_folder_path)

