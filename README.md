YOLOv8 + DeepFace Real-Time Face Recognition
This project combines YOLOv8 for real-time face detection with DeepFace for face recognition, enabling accurate identification of people through a webcam.

Features
Real-time face detection: Uses YOLOv8 to detect faces quickly in the camera feed.

Face recognition: Utilizes DeepFace with VGG-Face (or other supported models) to match detected faces against a local database of images.

Customizable threshold: Easily adjust recognition sensitivity using a cosine distance threshold.

Visual feedback: Draws colored bounding boxes on the video feed — green for recognized faces with names, red for unknown faces labeled as "Anonymous".

Supports multiple faces: Can detect and recognize multiple faces in the same frame.

Easy database structure: Each person has a folder in the database containing multiple images, automatically recognized by their folder name.

Requirement:

-Python 3.8+
-ultralytics (YOLOv8)
-opencv-python
-deepface

How to Use

just Clone the repository and put your dataset and use the (.pt) file for yolo 


