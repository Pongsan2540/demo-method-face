import matplotlib.pyplot as plt
import math
import cv2
from PIL import Image
import os
import shutil
import torch
import glob
import asyncio
import numpy as np

async def main_search(text_search, file_name):

    # โหลดโมเดล Dlib สำหรับตรวจจับใบหน้า
    #detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor('/home/kudsonadmin/Documents/test_method_face/shape_predictor_68_face_landmarks.dat')  # Dlib's pre-trained model for facial landmarks

    print(file_name)
    '''

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # วาดกรอบใบหน้า
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cropped_image = image[y:y + h, x:x + w]

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # คำนวณ landmarks ของใบหน้า
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # วาดจุด landmarks บนใบหน้า
        for (x1, y1) in shape:
            cv2.circle(image, (x1, y1), 1, (0, 0, 255), 2)

        # คำนวณมุม pitch, yaw, roll ของใบหน้า
        image_points = np.array([
            shape[30],  # Nose tip
            shape[8],   # Chin
            shape[36],  # Left eye left corner
            shape[45],  # Right eye right corner
            shape[48],  # Left mouth corner
            shape[54]   # Right mouth corner
        ], dtype='double')

        model_points = np.array([
            (0.0, 0.0, 0.0),    # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ])

        # Camera matrix
        focal_length = image.shape[1]
        center = (image.shape[1] // 2, image.shape[0] // 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype='double')

        dist_coeffs = np.zeros((4, 1))  # No lens distortion

        # คำนวณมุมของใบหน้า
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        if success:
            # แสดงมุมต่างๆ
            (rot_mat, _) = cv2.Rodrigues(rotation_vector)
            proj_mat = np.hstack((rot_mat, translation_vector))
            euler_angles = cv2.decomposeProjectionMatrix(proj_mat)[6]
            pitch, yaw, roll = euler_angles

            print(f"Pitch: {pitch}, Yaw: {yaw}, Roll: {roll}")

            # ตรวจสอบหน้าตรง
            if abs(yaw) < 15 and roll < 0.6:  # ทิศทางที่ใบหน้าอยู่ในมุมที่หน้าตรง
                cv2.putText(image, "Face is Straight", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(image, "Face is Not Straight", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)







    #result = translate_thai_to_english(text_search)
    text_eng = await process_translation(text_search)  # รับผลลัพธ์จากฟังก์ชัน

    #file_name = "video.mp4"
    video_path = "./uploads/"+str(file_name)
    output_dir = "video_frames"
    N = 30 
    #text_search = "Men wear blue shirts and wear mass"

    keep_fream(video_path, output_dir, N)

    torch.set_num_threads(4)
    print(clip.available_models())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Model is running on: {device}")
    model = SentenceTransformer('clip-ViT-B-32', device=device)

    img_names = list(glob.glob(output_dir + '/*.jpg'))
    print("Images:", len(img_names))
    batch_size = 16  # ลองปรับให้เหมาะสมกับแรมที่มี
    img_emb = []

    for i in range(0, len(img_names), batch_size):
        print(i)
        batch_files = img_names[i:i+batch_size]
        batch_imgs = [Image.open(f) for f in batch_files]
        batch_emb = model.encode(batch_imgs, batch_size=batch_size, convert_to_tensor=True)
        
        img_emb.extend(batch_emb)
        
        # ปิดไฟล์ภาพที่เปิดอยู่
        for img in batch_imgs:
            img.close()

    print(text_eng)
    search(model, img_emb, img_names, str(text_eng))
    '''