from django.shortcuts import render
from django.http import HttpResponse,StreamingHttpResponse, HttpResponseServerError,HttpResponseRedirect

from django.views.decorators import gzip
from imutils.video import VideoStream
from imutils.video import FPS
import cv2
import time
import imutils
import math
import numpy as np
id_s = 0
line = 300
auth = False
import pymongo
import dlib
from imutils import paths
import numpy as np
import shutil
import imutils
import pickle
import cv2
import os
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS

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
def reg():
    print('введите свои данные')
    name = input('введите ваше имя на Английском: ')
    fam = input('введите вашу фамилию на Английском: ')
    true = input('ваше имя: ' + str(name) + ' ваша фамилия: ' + str(fam) + '  y/n/exit')
    if 'y' in true:
        return [name,fam]
    elif 'exit' in true:
        return False
    else:
        reg()
def delet():
     print('введите дааные удаляемого человека')
     name = input('введите  имя на Английском: ')
     fam = input('введите  фамилию на Английском: ')
     true = input('имя: ' + str(name) + ' фамилия: ' + str(fam) + '  y/n')
     if 'y' in true:
         try:
             sn=fam+'_'+name
             
             shutil.rmtree('dataset/'+sn)
         except:
              return 'not fail'
     elif 'exit' in true:
        return False
     else:
         delet()
def fase_save(name):
            
            print('программа снимает ваше лицо')
            sn=name[1]+'_'+name[0]
            os.mkdir('dataset/'+sn)
            os.getcwd()
            faceCascade = cv2.CascadeClassifier('face_detection_model/haarcascade_frontalface_default.xml')
            video_capture = cv2.VideoCapture(0)
            i =0
            
               
            foto = 0
            fram = 0 
            while foto < 9:
    # Capture frame-by-frame
                    
                    ret, frame = video_capture.read()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            minSize=(100, 100),
                            flags=cv2.CASCADE_SCALE_IMAGE
                            )
    # Draw a rectangle around the faces
                    print(fram)
                    if fram > 0 and fram%80 ==0:
                            for (x, y, w, h) in faces:
                             cv2.imwrite("dataset/"+sn+"/0000"+ str(i) +".jpg", frame)
                             foto +=1
                             i += 1
                      
    # Display the resulting frame
                    fram+=1
                    print('готово ' + str(foto)+ '/9 фото')
                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
# When everything is done, release the capture
            video_capture.release()
            cv2.destroyAllWindows()

def build():
 dataset = 'dataset'
 embeddings = 'output/embeddings.pickle'
 detetor = 'face_detection_model'

 em_model = 'openface_nn4.small2.v1.t7'
 # load our serialized face detector from disk
 print("[INFO] loading face detector...")
 protoPath = os.path.sep.join([detetor, "deploy.prototxt"])
 modelPath = os.path.sep.join([detetor,
    "res10_300x300_ssd_iter_140000.caffemodel"])
 detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

 # load our serialized face embedding model from disk
 print("[INFO] loading face recognizer...")
 embedder = cv2.dnn.readNetFromTorch(em_model)

 # grab the paths to the input images in our dataset
 print("[INFO] quantifying faces...")
 imagePaths = list(paths.list_images(dataset))

 # initialize our lists of extracted facial embeddings and
 # corresponding people names
 knownEmbeddings = []
 knownNames = []

 # initialize the total number of faces processed
 total = 0

 # loop over the image paths
 for (i, imagePath) in enumerate(imagePaths):
   try:
     # extract the person name from the image path
     print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
     name = imagePath.split(os.path.sep)[-2]

     # load the image, resize it to have a width of 600 pixels (while
     # maintaining the aspect ratio), and then grab the image
     # dimensions
     image = cv2.imread(imagePath)
     image = imutils.resize(image, width=600)
     (h, w) = image.shape[:2]

     # construct a blob from the image
     imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

     # apply OpenCV's deep learning-based face detector to localize
     # faces in the input image
     detector.setInput(imageBlob)
     detections = detector.forward()

     # ensure at least one face was found
     if len(detections) > 0:
         # we're making the assumption that each image has only ONE
         # face, so find the bounding box with the largest probability
         i = np.argmax(detections[0, 0, :, 2])
         confidence = detections[0, 0, i, 2]

         # ensure that the detection with the largest probability also
         # means our minimum probability test (thus helping filter out
         # weak detections)
         if confidence > 0:  
             # compute the (x, y)-coordinates of the bounding box for
             # the face
             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
             (startX, startY, endX, endY) = box.astype("int")

             # extract the face ROI and grab the ROI dimensions
             face = image[startY:endY, startX:endX]
             (fH, fW) = face.shape[:2]

             # ensure the face width and height are sufficiently large
             if fW < 20 or fH < 20:
                  continue

             # construct a blob for the face ROI, then pass the blob
             # through our face embedding model to obtain the 128-d
             # quantification of the face
             faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
             embedder.setInput(faceBlob)
             vec = embedder.forward()

             # add the name of the person + corresponding face
             # embedding to their respective lists
             knownNames.append(name)
             knownEmbeddings.append(vec.flatten())
             total += 1
   except:
      False
      print('image incorrect')
 # dump the facial embeddings + names to disk
 print("[INFO] serializing {} encodings...".format(total))
 data = {"embeddings": knownEmbeddings, "names": knownNames}
 f = open(embeddings, "wb")
 f.write(pickle.dumps(data))
 f.close()
def train():
 embeddings = 'output/embeddings.pickle'
 # load the face embeddings
 print("[INFO] loading face embeddings...")
 data = pickle.loads(open(embeddings, "rb").read())
 
 # encode the labels
 print("[INFO] encoding labels...")
 le = LabelEncoder()
 labels = le.fit_transform(data["names"])

 # train the model used to accept the 128-d embeddings of the face and
 # then produce the actual face recognition
 print("[INFO] training model...")
 recognizer = SVC(C=1.0, kernel="linear", probability=True)
 recognizer.fit(data["embeddings"], labels)
 rec = 'output/recognizer.pickle'
 # write the actual face recognition model to disk
 f = open(rec, "wb")
 f.write(pickle.dumps(recognizer))
 f.close()
 lee = 'output/le.pickle'
 # write the label encoder to disk
 f = open(lee, "wb")
 f.write(pickle.dumps(le))
 f.close()
def detect():
 detetor ='face_detection_model'
 em_model = 'openface_nn4.small2.v1.t7'
 rec = 'output/recognizer.pickle'
 lee ='output/le.pickle'
 # load our serialized face detector from disk
 print("[INFO] loading face detector...")
 protoPath = os.path.sep.join([detetor, "deploy.prototxt"])
 modelPath = os.path.sep.join([detetor,
    "res10_300x300_ssd_iter_140000.caffemodel"])
 detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

 # load our serialized face embedding model from disk
 print("[INFO] loading face recognizer...")
 embedder = cv2.dnn.readNetFromTorch(em_model)

 # load the actual face recognition model along with the label encoder
 recognizer = pickle.loads(open(rec, "rb").read())
 le = pickle.loads(open(lee, "rb").read())

 # initialize the video stream, then allow the camera sensor to warm up
 print("[INFO] starting video stream...")
 vs = VideoStream(src=0).start()
 time.sleep(2.0)

 # start the FPS throughput estimator
 #fps = FPS().start()

 # loop over frames from the video file stream
 while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    
    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    inframe = []
    obj ={}
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.2:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            
            # draw the bounding box of the face along with the
            # associated probability
            if proba > 0.2:
              text = "{}: {:.2f}%".format(name, proba * 100)
              y = startY - 10 if startY - 10 > 10 else startY + 10
              cv2.rectangle(frame, (startX, startY), (endX, endY),
                  (0, 0, 255), 2)
              cv2.putText(frame, text, (startX, y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
              obj['name']= name
              obj['ver'] = proba
              obj['kord']=[startX,startY,endX,endY]
              inframe.append(obj)
            else:
              name = 'unknown'
              text = "{}: {:.2f}%".format(name, proba * 100)
              y = startY - 10 if startY - 10 > 10 else startY + 10
              cv2.rectangle(frame, (startX, startY), (endX, endY),
                  (0, 0, 255), 2)
              cv2.putText(frame, text, (startX, y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
              obj['name']= name
              obj['ver'] = proba
              inframe.append(obj)
            
        
    #print('next_frane')
    # update the FPS counter
    #fps.update()
    print(inframe)
    del(inframe)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

 # stop the timer and display FPS information
 '''fps.stop()
 print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
 print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))'''

 # do a bit of cleanup
 cv2.destroyAllWindows()
 vs.stop()
def get_line():
    global line
    
    return line
def line_edit(r):
    global line
    line = r
    
def get_info(mas,box):
    global id_s
    d = 1000
    imas = 0
    info = []
   
    for i in range(len(mas)):
        #print((box[0]+box[2])/2  - (mas[i]['box'][0]+mas[i]['box'][2])/2)
        x = int(math.fabs((box[0]+box[2])/2 - (mas[i]['box'][0]+mas[i]['box'][2])/2))
        
        y = int(math.fabs((box[1]+box[3])/2 - (mas[i]['box'][1]+mas[i]['box'][3])/2))
        
        dn = int(math.sqrt(x*x + y*y))
        
        if dn < d:
            d = dn
            imas = i
   
    if d < 100:
        info.append(mas[imas]['id'])
        #print('заход if')
        speed = [0,0]
        speed[0] = (box[0]+box[2])/2 - (mas[i]['box'][0]+mas[i]['box'][2])/2
        speed[1] = (box[1]+box[3])/2 - (mas[i]['box'][1]+mas[i]['box'][3])/2
        info.append(speed)
        return info
    else:
        id_s+=1
        #print('заход елсе')
        info.append(id_s)
        info.append([0,0])
        return info
class VideoCamera(object):
    
    def __init__(self,path):
        #self.video = cv2.VideoCapture(path)
        self.net = cv2.dnn_DetectionModel('person-detection-retail-0002.xml',
                            'person-detection-retail-0002.bin')
        self.video = VideoStream('00348.mts').start()
        self.last_obj = []
        self.sh = 0
       
        self.pipl_cross = []
        client = pymongo.MongoClient("localhost", 27017)
        self.db = client.person
    def __del__(self):
        self.video.read()

    def get_frame(self):
        frame = self.video.read()
        myrez = []
        LINE = get_line()
        frame = imutils.resize(frame, width=1000)
        _, confidences, boxes = self.net.detect(frame, confThreshold=0.5)
        for confidence, box in zip(list(confidences), boxes):
    
          cv2.rectangle(frame, box, color=(0, 255, 0))
       
          if True:
              obj ={}
              
              obj['label'] = 'person '
              obj['box'] = box
              obj_inf = get_info(self.last_obj,obj['box'])
              obj['id']=  obj_inf[0]
              obj['speed'] = obj_inf[1]
            
              y = box[1] - 15 if box[1] - 15 > 15 else box[1] + 15
              cv2.putText(frame,'person' + str(obj['id']), (box[0], y),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
     
              myrez.append(obj)
      
              if obj['box'][1] < LINE and  obj['box'][3] > LINE and not(obj['id'] in self.pipl_cross):
                             self.sh+=1
                             self.pipl_cross.append(obj['id'])
                             self.db.person.insert_one({'id':str(obj['id']),'time':str(int(time.time())),'all':str(self.sh)})
                             f= open('othet.txt','a')
                             f.write(' id: ' + str(obj['id']) + ' time: ' + str(int(time.time())) + ' всего: ' + str(self.sh) + '\n')
                             f.close()
              del(obj)

        del(self.last_obj)
 
        cv2.putText(frame, str(self.sh) , (20,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0) , 3) 
        cv2.line(frame,(0,LINE),(2000,LINE ),(0,255,0),thickness=2)
 
        self.last_obj = []
        for j in range(len(myrez)):
                      self.last_obj.append(myrez[j])
        del(myrez)
       
        jpeg = cv2.imencode('.jpg',frame)[1].tostring()
        '''print(jpeg)
        j = jpeg.tobytes()
        print(j)'''
        return jpeg
    

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def indexscreen(request): 
    try:
    
        template = "screens.html"
        return render(request,template)
    except HttpResponseServerError:
        print("aborted")
def changeline(request):
    global line
    print('-------------------------------------------')
    print(request)
    line = str(request).split('/')
    line = int(line[2])
    print(str(line))
    print('-------------------------------------------')
    return HttpResponseRedirect('/stream/screen')
@gzip.gzip_page
def dynamic_stream(request,num=0,stream_path="0"):
    
    stream_path = 'add your camera stream here that can rtsp or http'
    return StreamingHttpResponse(gen(VideoCamera(stream_path)),content_type="multipart/x-mixed-replace;boundary=frame")

