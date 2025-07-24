# Vehicle Counting with YOLO and OpenCV

This project utilizes YOLO and OpenCV to count vehicles passing through a defined region in a video.

## Requirements

- OpenCV (`cv2`)
- ultralytics (YOLO)
- YOLO model file (`yolo11n.pt`)
- Input video (`vehicle_counting.mp4`)

## Installation

1. Install the required libraries:
    ```sh
    pip install opencv-python ultralytics
    ```

## Usage

1. Place the input video (`vehicle_counting.mp4`) and YOLO model file (`yolo11n.pt`) in the project directory.
2. Run the main script:
    ```sh
    python main.py
    ```
3. The output video (`object_counting_output.avi`) will be generated, displaying the IN/OUT vehicle counts on each frame.

## Customization

- Change the counting region by editing the `region_points` variable in [`main.py`](d:\Projects_COMPUTER_VISION\venv\Project_Vehicle_Counting\main.py).
- You can display labels, confidence scores, etc. by adjusting the parameters of [`solutions.ObjectCounter`](d:\Projects_COMPUTER_VISION\venv\Project_Vehicle_Counting\main.py).

## File Structure

- [`main.py`](d:\Projects_COMPUTER_VISION\venv\Project_Vehicle_Counting\main.py): Main script for video processing and vehicle counting.
- `vehicle_counting.mp4`: Input video.
- `yolo11n.pt`: YOLO model file.
- `object_counting_output.avi`: Output video.

## Contact

For any questions, please contact: nguyenphuongv07@gmail.com
