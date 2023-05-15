import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class FaceMesh():

    def __init__(self, batch_size, kpt_num):

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self.kpt_num = kpt_num
        self.batch_size = batch_size
        self.error_idx = 0

        self.silhouette = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
            365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
            132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        self.lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        self.lipsUpperInner = [
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
        ]
        self.lipsLowerInner = [
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
        ]

        self.rightEyeUpper0 = [246, 161, 160, 159, 158, 157, 173]
        self.rightEyeLower0 = [33, 7, 163, 144, 145, 153, 154, 155, 133]
        self.rightEyeUpper1 = [247, 30, 29, 27, 28, 56, 190]
        self.rightEyeLower1 = [130, 25, 110, 24, 23, 22, 26, 112, 243]
        self.rightEyeUpper2 = [113, 225, 224, 223, 222, 221, 189]
        self.rightEyeLower2 = [226, 31, 228, 229, 230, 231, 232, 233, 244]
        self.rightEyeLower3 = [143, 111, 117, 118, 119, 120, 121, 128, 245]

        self.rightEyebrowUpper = [156, 70, 63, 105, 66, 107, 55, 193]
        self.rightEyebrowLower = [35, 124, 46, 53, 52, 65]

        self.rightEyeIris = [473, 474, 475, 476, 477]

        self.leftEyeUpper0 = [466, 388, 387, 386, 385, 384, 398]
        self.leftEyeLower0 = [263, 249, 390, 373, 374, 380, 381, 382, 362]
        self.leftEyeUpper1 = [467, 260, 259, 257, 258, 286, 414]
        self.leftEyeLower1 = [359, 255, 339, 254, 253, 252, 256, 341, 463]
        self.leftEyeUpper2 = [342, 445, 444, 443, 442, 441, 413]
        self.leftEyeLower2 = [446, 261, 448, 449, 450, 451, 452, 453, 464]
        self.leftEyeLower3 = [372, 340, 346, 347, 348, 349, 350, 357, 465]

        self.leftEyebrowUpper = [383, 300, 293, 334, 296, 336, 285, 417]
        self.leftEyebrowLower = [265, 353, 276, 283, 282, 295]

        self.leftEyeIris = [468, 469, 470, 471, 472]

        self.midwayBetweenEyes = [168]

        self.noseTip = [1]
        self.noseBottom = [2]
        self.noseRightCorner = [98]
        self.noseLeftCorner = [327]

        self.rightCheek = [205]
        self.leftCheek = [425]

        self.feature_idx = (self.silhouette + self.lipsUpperOuter +
                            self.lipsLowerOuter + self.lipsUpperInner +
                            self.lipsLowerInner + self.rightEyeUpper0 +
                            self.rightEyeLower0 + self.rightEyebrowUpper +
                            self.rightEyebrowLower + self.leftEyeUpper0 +
                            self.leftEyeLower0 + self.leftEyebrowUpper +
                            self.leftEyebrowLower + self.noseTip +
                            self.noseBottom + self.noseRightCorner +
                            self.noseLeftCorner)

    def detect(self, images, type):

        kpt = np.zeros((self.batch_size, self.kpt_num, 2))
        feature_kpt = np.zeros((self.batch_size, len(self.feature_idx), 2))

        for i, img in enumerate(images):
            # Convert the BGR image to RGB before processing.

            results = self.face_mesh.process(img)

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                # print(f'{i}_fuck')
                print(f'Image type: {type}')
                print(f'max: {np.max(img)}')
                print(f'min: {np.min(img)}')
                print(f'shape: {img.shape}')

                cv2.imwrite(
                    f'err_{type}_{self.error_idx}.png',
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )
                self.error_idx += 1
                continue

            annotated_image = img.copy()
            for face_landmarks in results.multi_face_landmarks:

                for idx, data_point in enumerate(face_landmarks.landmark):
                    kpt[i, idx, 0] = data_point.x * annotated_image.shape[1]
                    kpt[i, idx, 1] = data_point.y * annotated_image.shape[0]
            feature_kpt[i] = kpt[i][self.feature_idx]
        return feature_kpt
