import numpy as np



def decode_image(features):
    # get BGR image from bytes

    features["camID"] =  np.squeeze(np.frombuffer(features["camID"], dtype=np.int64))

    features["img"] = np.frombuffer(features["img"],
                                    dtype=np.uint8).astype(np.float32).reshape(
                                        (512, 334, 3))

    features["vtx"] = np.frombuffer(features["vtx"], dtype=np.float32).reshape(
        (7306, 3))

    features["texture"] = np.frombuffer(
        features["texture"], dtype=np.uint8).astype(np.float32).reshape(
            (1024, 1024, 3)) / 255.

    features["verts_uvs"] = np.frombuffer(features["verts_uvs"],
                                          dtype=np.float32).reshape((-1, 2))

    features["faces_uvs"] = np.frombuffer(features["faces_uvs"],
                                          dtype=np.float32).reshape((-1, 3))

    features["verts_idx"] = np.frombuffer(features["verts_idx"],
                                          dtype=np.float32).reshape((-1, 3))

    features["head_pose"] = np.frombuffer(features["head_pose"],
                                          dtype=np.float32).reshape((3, 4))

    return features
