import cv2
import json

class ROI:
    def __init__(self, file_path, video_path):
        """
        Initializes the ROI object with ROI information from the JSON file.

        Parameters:
        - file_path (str): Path to the JSON file containing ROI information.
        - video_path (str): Path to the video file to read dimensions from.
        """
        self.rois = self._read_info_from_file(file_path, video_path)

    def _read_info_from_file(self, file_path, video_path):
        """
        Reads ROI information from the JSON file and calculates absolute coordinates based on video dimensions.

        Parameters:
        - file_path (str): Path to the JSON file.
        - video_path (str): Path to the video file.

        Returns:
        - dict: Dictionary containing ROI information.
        """
        # Read video dimensions
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            exit()

        image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Read ROI information from the JSON file
        with open(file_path, 'r') as file:
            rois = json.load(file)

        # Calculate absolute coordinates
        for roi_name, roi_info in rois.items():
            x, y, width, height = roi_info["x"], roi_info["y"], roi_info["width"], roi_info["height"]
            x_abs, y_abs, width_abs, height_abs = int(x * image_width), int(y * image_height), int(width * image_width), int(height * image_height)
            roi_info["x_abs"], roi_info["y_abs"], roi_info["width_abs"], roi_info["height_abs"] = x_abs, y_abs, width_abs, height_abs

        return rois

    def add_roi_to_image(self, image):
        """
        Adds ROIs to the image.

        Parameters:
        - image (numpy.ndarray): Image to which ROIs will be added.
        """
        for i, (roi_name, roi_info) in enumerate(self.rois.items()):
            x_abs, y_abs, width_abs, height_abs = roi_info["x_abs"], roi_info["y_abs"], roi_info["width_abs"], roi_info["height_abs"]

            # Green color for the first ROI, blue for the second
            color = (0, 180, 0) if i == 0 else (180, 0, 0)

            # Draw rectangle
            cv2.rectangle(image, (x_abs, y_abs), (x_abs + width_abs, y_abs + height_abs), color, 2)

            label_position = (x_abs + 5, y_abs + 20)
            cv2.putText(image, roi_name, label_position, cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 1)

    def point_in_rois(self, point):
        """
        Checks if a point is inside each of the ROIs.

        Parameters:
        - point (tuple): Coordinates of the point to be checked (x, y).

        Returns:
        - tuple: Two boolean values, the first indicating if the point is in ROI1, and the second indicating if the point is in ROI2.
        """
        x, y = point
        in_roi1 = False
        in_roi2 = False

        for roi_name, roi_info in self.rois.items():
            x_abs, y_abs, width_abs, height_abs = roi_info["x_abs"], roi_info["y_abs"], roi_info["width_abs"], roi_info["height_abs"]
            
            # Check if the point is inside the current ROI
            if x_abs <= x <= x_abs + width_abs and y_abs <= y <= y_abs + height_abs:
                if roi_name == "roi1":
                    in_roi1 = True
                elif roi_name == "roi2":
                    in_roi2 = True

        return in_roi1, in_roi2


