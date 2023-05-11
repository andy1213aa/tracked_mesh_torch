
import numpy as np
import cv2

def decode_image(features):
    # get BGR image from bytes

    features["img"] = np.frombuffer(features["img"], dtype=np.float32).reshape(
        (512, 334, 3)).copy()
    # features["img"] = cv2.cvtColor(features["img"] , cv2.COLOR_BGR2RGB)
    features["vtx"] = np.frombuffer(features["vtx"], dtype=np.float32).reshape(
        (7306, 3)).copy()

    features["vtx_mean"] = np.frombuffer(features["vtx_mean"],
                                         dtype=np.float32).reshape(
                                             (7306, 3)).copy()

    features["tex"] = np.frombuffer(features["tex"], dtype=np.float32).reshape(
        (1024, 1024, 3)).copy() / 255.
    features["verts_uvs"] = np.frombuffer(features["verts_uvs"],
                                          dtype=np.float32).reshape(
                                              (-1, 2)).copy()
    features["faces_uvs"] = np.frombuffer(features["faces_uvs"],
                                          dtype=np.float32).reshape(
                                              (-1, 3)).copy()
    features["verts_idx"] = np.frombuffer(features["verts_idx"],
                                          dtype=np.float32).reshape(
                                              (-1, 3)).copy(),
    features["head_pose"] = np.frombuffer(features["head_pose"],
                                          dtype=np.float32).reshape(
                                              (3, 4)).copy()

    features["intricsic_camera"] = np.frombuffer(features["intricsic_camera"],
                                                 dtype=np.float32).reshape(
                                                     (3, 3)).copy()

    features['focal'] = np.stack([
        features["intricsic_camera"][0, 0], features["intricsic_camera"][1, 1]
    ], )

    features['princpt'] = np.stack([
        features["intricsic_camera"][0, 2], features["intricsic_camera"][1, 2]
    ], )
    features["extrinsic_camera"] = np.frombuffer(features["extrinsic_camera"],
                                                 dtype=np.float32).reshape(
                                                     (3, 4)).copy()
    return features


