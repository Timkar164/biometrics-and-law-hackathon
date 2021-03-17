
import dlib
from skimage import io
from scipy.spatial import distance



sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

def description(img):
  dets = detector(img, 1)
  
  for k, d in enumerate(dets):
      #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #  k, d.left(), d.top(), d.right(), d.bottom()))
      shape = sp(img, d)
      win1.clear_overlay()
      win1.add_overlay(d)
      win1.add_overlay(shape)
      
  face_descriptor = facerec.compute_face_descriptor(img, shape)
  
  return np.array(face_descriptor)

def new_users(img):
  return description(img)

def users_verification(img, descriptions):
  
  first_descriptions = description(img)
  main_descriptions = descriptions
  
  if distance.euclidean(first_descriptions, main_descriptions) <0.6:
    return 1
  else:
    return 0

