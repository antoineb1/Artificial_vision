import cv2
from ultralytics import YOLO
from roi import ROI  


def load_model(model_path):
    return YOLO(model_path)

def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        exit()
    return cap

# def process_frame(model, cap, rois):
#     success, frame = cap.read()
#     if success:
#         results = model.track(frame, persist=True, classes=0)
#         # annotated_frame = results[0].plot()

#         centers = calculate_bbox_centers(results)
#         for center in centers:
#             print("ENTRA: ",rois.point_in_roi(center))

#         # Print of boxes and roi
#         annotated_frame = plot_bboxes(results,frame) 
#         rois.add_roi_to_image(annotated_frame)  # Use the add_roi_to_image method from the imported ROI class
#         cv2.imshow("YOLOv8 Tracking", annotated_frame)
#         return True
#     else:
#         return False

# def process_frame(model, cap, rois,tracking_data):
#     success, frame = cap.read()
#     if success:
#         results = model.track(frame, persist=True, classes=0)
#         # annotated_frame = results[0].plot()

#         # Aggiornamento tempo di permanenza nelle roi e numero di passaggi
#         centers = calculate_bbox_centers(results)

#         update_data(centers,tracking_data,rois)


#         # Print of boxes and roi
#         annotated_frame = plot_bboxes(results,frame) 
#         rois.add_roi_to_image(annotated_frame)  # Use the add_roi_to_image method from the imported ROI class
#         cv2.imshow("YOLOv8 Tracking", annotated_frame)
#         return True
#     else:
#         return False        



# def update_data(centers,tracking_data,rois):
#     # Per ogni punto vedo se è presente nelle due roi
#     for center in centers:
#         is_in_roi1,is_in_roi2 = rois.point_in_rois(center[1],center[2])

#         if is_in_roi1:
#             update_data(tracking_data,id,"roi1")
#         elif is_in_roi2:
#             update_data(tracking_data,id,"roi2")

#     return

# def update_dict(tracking_data,id,roi):
#     str1 = "_passages"
#     str2 = "_persistence_time"
#     if id in tracking_data:
#         if not tracking_data[id]["flag"]:
#             tracking_data[id][roi+str1] += 1
#             tracking_data[id]["flag"] = True
#         tracking_data[id][roi+str2] += 1
#     else:
#         tracking_data[id] = {}
#         tracking_data[id][roi+str1] = 0
#         tracking_data[id][roi+str2] = 0
#         tracking_data[id]["flag"] = False
#     return



def process_frame(model, cap, rois, tracking_data):
    """
    Process a frame from the video, update tracking data, and display the annotated frame.

    Parameters:
    - model (YOLO): YOLO model for object detection and tracking.
    - cap (cv2.VideoCapture): VideoCapture object for reading video frames.
    - rois (ROI): Instance of the ROI class containing region of interest information.
    - tracking_data (dict): Dictionary to store tracking data.

    Returns:
    - bool: True if the frame is successfully processed, False otherwise.
    """
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True, classes=0)

        # Update dwell time in ROIs and number of passages
        centers = calculate_bbox_centers(results)
        update_data(centers, tracking_data, rois)

        # Display the annotated frame with bounding boxes and ROIs
        annotated_frame = plot_bboxes(results, frame) 
        rois.add_roi_to_image(annotated_frame)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        return True
    else:
        return False

def update_data(centers, tracking_data, rois):
    """
    Update tracking data based on object positions in ROIs.

    Parameters:
    - centers (list): List of tuples containing object IDs and their center coordinates.
    - tracking_data (dict): Dictionary to store tracking data.
    - rois (ROI): Instance of the ROI class containing region of interest information.
    """
    for center in centers:
        is_in_roi1, is_in_roi2 = rois.point_in_rois((center[1], center[2]))
        obj_id = center[0]
        if obj_id in tracking_data:
            if is_in_roi1:
                update_dict(tracking_data, obj_id, "roi1")
            elif is_in_roi2:
                update_dict(tracking_data, obj_id, "roi2")
        else: 
            # Initialize tracking data for the object ID
            tracking_data[obj_id] = {
                "roi1_passages": 0,
                "roi1_persistence_time": 0,
                "roi1_flag": False,
                "roi2_passages": 0,
                "roi2_persistence_time": 0,
                "roi2_flag": False
            }

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



def print_tracking_statistics(tracking_data, fps):
    """
    Print tracking statistics for each object in the tracking data.

    Parameters:
    - tracking_data: Dictionary containing tracking data.
    - fps: Frames per second of the video.
    """
    for obj_id, data in tracking_data.items():
        roi1_passages = data.get("roi1_passages", 0)
        roi2_passages = data.get("roi2_passages", 0)
        roi1_persistence_time = data.get("roi1_persistence_time", 0) / fps
        roi2_persistence_time = data.get("roi2_persistence_time", 0) / fps

        print(f"Object ID {obj_id}:")
        print(f"  ROI1 Passages: {roi1_passages}")
        print(f"  ROI2 Passages: {roi2_passages}")
        print(f"  ROI1 Persistence Time: {roi1_persistence_time:.2f} seconds")
        print(f"  ROI2 Persistence Time: {roi2_persistence_time:.2f} seconds")
        print()



def calculate_bbox_centers(results):
    """
    Calculate the centers of bounding boxes given a results object.

    Parameters:
    - results (ultralytics.YOLO): YOLO results object containing information about detected objects.

    Returns:
    - list: List of tuples, each containing the track ID and coordinates of the center of a bounding box ("id_x", cx, cy).
    """
    centers = []
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

            centers.append((track_id_str, cx, cy))

    return centers






def plot_bboxes(results, frame):
    for result in results:
        boxes = result.boxes.cpu().numpy()
        xyxys = boxes.xyxy
        ids = boxes.id

        for xyxy, track_id in zip(xyxys, ids):
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            # Disegna il bounding box rosso
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Aggiungi etichetta con l'ID sopra al box (ID come intero)
            label_position = (x1, y1 - 10)
            cv2.putText(frame, f'ID: {int(track_id)}', label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return frame




def main():
    # Load a model
    model_path = 'yolov8n.pt'
    model = load_model(model_path)

    # Open the video file
    video_path = "\\Users\\marco\\OneDrive\\Desktop\\Artificial Vision\\Progetto\\trimmed_prova.mp4"
    cap = open_video(video_path)

    # Create an ROI manager and read ROIs from the JSON file
    roi_manager = ROI('\\Users\\marco\\OneDrive\\Desktop\\Artificial Vision\\Progetto\\config.txt',video_path)

    tracking_data = {}

    # Loop through the video frames
    while process_frame(model, cap, roi_manager,tracking_data):
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # Get video fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    print_tracking_statistics(tracking_data, fps)

    # Rilascia l'oggetto di acquisizione video e chiudi la finestra di visualizzazione
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
