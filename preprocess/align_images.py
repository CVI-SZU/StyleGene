import bz2
from .face_alignment import image_align
from .landmarks_detector import LandmarksDetector
from configs import path_ckpt_landmark68

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def init_landmark():
    landmarks_model_path = unpack_bz2(path_ckpt_landmark68)
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    return landmarks_detector


landmarks_detector = init_landmark()


def align_face(raw_img, output_size=256):
    try:
        face_landmarks = landmarks_detector.get_landmarks(raw_img)[0]
        aligned_face = image_align(raw_img, face_landmarks, output_size=output_size, transform_size=1024)
        return aligned_face
    except:
        return raw_img
