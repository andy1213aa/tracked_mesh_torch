import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

class FaceMesh():
    
    def __init__(self, batch_size, kpt_num):
        
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5)
        self.kpt_num = kpt_num
        self.batch_size = batch_size
        self.error_idx = 0
    def detect(self, images):
        
        kpt = np.zeros((self.batch_size, self.kpt_num, 2))
        
        
        for i, img in enumerate(images):
            # Convert the BGR image to RGB before processing.
            
            results = self.face_mesh.process(img)

            
            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                # print(f'{i}_fuck')
                cv2.imwrite(f'pred_{self.error_idx}.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                self.error_idx+= 1
                continue
            
            annotated_image = img.copy()
            for face_landmarks in results.multi_face_landmarks:
                
                for idx, data_point in enumerate(face_landmarks.landmark):
                    kpt[i, idx, 0] = data_point.x * annotated_image.shape[1]
                    kpt[i, idx, 1] = data_point.y * annotated_image.shape[0]

        return kpt

