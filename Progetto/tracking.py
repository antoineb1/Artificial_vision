import cv2, time, json, argparse
from ultralytics import YOLO
from roi import ROI
from par import ViLTPAR
from PIL import Image


def load_yolo(yolo_model_path):
    return YOLO(yolo_model_path)

def load_vilt(vilt_model_path):
    return ViLTPAR(vilt_model_path)

def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        exit()
    return cap



def process_frames(yolo_model, vilt_model, cap, rois, tracking_data, fps):
    """
    Process frames from the video, update tracking data, and display the annotated frames.

    Parameters:
    - model (YOLO): YOLO model for object detection and tracking.
    - cap (cv2.VideoCapture): VideoCapture object for reading video frames.
    - rois (ROI): Instance of the ROI class containing region of interest information.
    - tracking_data (dict): Dictionary to store tracking data.

    Returns:
    - None
    """
    frames_to_wait = fps * 1.3
    frame_counter = frames_to_wait + 1

    while True:
        success, frame = cap.read()

        if not success:
            break  # Break the loop if no more frames

        results = yolo_model.track(frame, persist=True, classes=0, verbose=False)

        # Update dwell time in ROIs and number of passages
        bbinfo = calculate_bbox_info(results)

        if frame_counter >= frames_to_wait:
            flag_par = True
            frame_counter = 0
        else:
            flag_par = False
            frame_counter += 1

        update_data(frame, bbinfo, tracking_data, rois, vilt_model, flag_par)

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

def update_data(frame, bbinfo, tracking_data, rois, vilt, flag_par):
    """
    Update tracking data based on object positions in ROIs.

    Parameters:
    - centers (list): List of tuples containing object IDs and their center coordinates.
    - tracking_data (dict): Dictionary to store tracking data.
    - rois (ROI): Instance of the ROI class containing region of interest information.
    """

    for info in bbinfo:

        obj_id = info[0]
        centers = info[1]
        angles = info[2]

        is_in_roi1, is_in_roi2 = rois.point_in_rois((centers[0], centers[1]))

        # ROIs            
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
            cropped_frame = crop_objects(frame, angles)
            attributes = vilt.extract_attributes(Image.fromarray(cropped_frame))

            tracking_data[obj_id]['gender'] = attributes[0]
            tracking_data[obj_id]['hat'] = attributes[1]
            tracking_data[obj_id]['bag'] = attributes[2]
            tracking_data[obj_id]['upper_color'] = attributes[3]
            tracking_data[obj_id]['lower_color'] = attributes[4]
        if is_in_roi1:
            update_dict(tracking_data, obj_id, "roi1")
        elif is_in_roi2:
            update_dict(tracking_data, obj_id, "roi2")

def update_dict(tracking_data, obj_id, roi):
    """
    Update tracking data dictionary with the number of passages and persistence time.

    Parameters:
    - tracking_data (dict): Dictionary to store tracking data.
    - obj_id (str): Object ID.
    - roi (str): Region of interest identifier ("roi1" or "roi2").
    """
    str1 = "_passages"
    str2 = "_persistence_time"
    flag = "_flag"
    
    if not tracking_data[obj_id][roi + flag]:
        tracking_data[obj_id][roi + str1] += 1
        tracking_data[obj_id][roi + flag] = True
    tracking_data[obj_id][roi + str2] += 1



# def print_tracking_statistics(tracking_data, fps):
#     """
#     Print tracking statistics for each object in the tracking data.

#     Parameters:
#     - tracking_data: Dictionary containing tracking data.
#     - fps: Frames per second of the video.
#     """
#     for obj_id, data in tracking_data.items():
#         gender = data.get("gender", 0)
#         hat = data.get("hat", 0)
#         bag = data.get("bag", 0)
#         upper_color = data.get("upper_color", 0)
#         lower_color = data.get("lower_color", 0)
#         roi1_passages = data.get("roi1_passages", 0)
#         roi2_passages = data.get("roi2_passages", 0)
#         roi1_persistence_time = data.get("roi1_persistence_time", 0) / fps
#         roi2_persistence_time = data.get("roi2_persistence_time", 0) / fps

#         print(f"Object ID {obj_id}:")
#         print(f"\tgender: {gender}")
#         print(f"\that: {hat}")
#         print(f"\tbag: {bag}")
#         print(f"\tupper_color: {upper_color}")
#         print(f"\tlower_color: {lower_color}")
#         print(f"\tROI1 Passages: {roi1_passages}")
#         print(f"\tROI2 Passages: {roi2_passages}")
#         print(f"\tROI1 Persistence Time: {roi1_persistence_time:.2f} seconds")
#         print(f"\tROI2 Persistence Time: {roi2_persistence_time:.2f} seconds")
#         print()



def calculate_bbox_info(results):
    """
    Calculate the information of bounding boxes given a results object.

    Parameters:
    - results (ultralytics.YOLO): YOLO results object containing information about detected objects.

    Returns:
    - list: List of tuples, each containing the track ID and coordinates of the center of a bounding box ("id_x", (cx, cy), (x1, y1, x2, y2)).
    """
    bbinfo = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        xyxys = boxes.xyxy
        ids = boxes.id  # Use result.id to get track IDs

        for xyxy, track_id in zip(xyxys, ids):
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            # Calculate the center of the bounding box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Convert track_id to a string in the format "id_x"
            track_id_str = f"id_{int(track_id)}"

            # Return id, bb center and coordinates
            bbinfo.append((track_id_str, (cx, cy), (x1, y1, x2, y2)))

    return bbinfo


def plot_bboxes(bbinfo, tracking_data, frame):
    for info in bbinfo:

        obj_id = info[0]
        angles = info[2]

        x1 = angles[0]
        y1 = angles[1]
        x2 = angles[2]
        y2 = angles[3]

        tracking_info = tracking_data.get(obj_id, {})

        # Disegna il bounding box rosso
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Aggiungi etichetta con l'ID sopra al box (ID come intero)
        id_label_position = (x1, y1 - 10)
        cv2.putText(frame, f'{obj_id}', id_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

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
    Utility function which crop the frame in order to obtain just the images of people detected in the scene

    Parameters:
    - results (ultralytics.YOLO): YOLO results object containing information about detected objects.
    - frame: current scene frame

    Returns:
    - list: list of all the cropped images containing detected people
    """
    x1 = angles[0]
    y1 = angles[1]
    x2 = angles[2]
    y2 = angles[3]

    # Crop the section related to the bounded box
    cropped_image = frame[y1:y2, x1:x2].copy()

    return cropped_image
    
def save_tracking_statistics(tracking_data, output_file, fps):
    """
    Save tracking statistics for each object in the tracking data to a JSON file.

    Parameters:
    - tracking_data: Dictionary containing tracking data.
    - output_file: The name of the output JSON file.
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

def main():

    parser = argparse.ArgumentParser(description='Process video frames with YOLO and ViLT models.')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file (mp4).')
    parser.add_argument('--configuration', type=str, required=True, help='Path to the ROI configuration file (txt).')
    parser.add_argument('--results', type=str, required=True, help='Path to the output JSON format file (txt).')

    args = parser.parse_args()

    # Load a model
    yolo_model_path = 'yolov8n.pt'
    yolo_model = load_yolo(yolo_model_path)

    vilt_model_path = 'dandelin/vilt-b32-finetuned-vqa'
    vilt_model = load_vilt(vilt_model_path)

    # Open the video file
    video_path = args.video
    cap = open_video(video_path)

    # Create an ROI manager and read ROIs from the JSON file
    roi_manager = ROI(args.configuration, video_path)
    
    tracking_data = {}

    # Get video fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Process frames
    process_frames(yolo_model, vilt_model, cap, roi_manager, tracking_data, fps)

    save_tracking_statistics(tracking_data, args.results, fps)

if __name__ == "__main__":
    main()
