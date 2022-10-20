from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2 
import datetime as dt
import pyfirmata
import requests
import base64
from datetime import datetime
import os.path
from PIL import Image
import threading
from concurrent.futures import ThreadPoolExecutor, wait

def worker():
    """thread worker function"""
    print('Worker')


    
#VARIABLES
puerto = "COM6" #Puerto COM de emulación en USB
pinA = (13) #PIN donde va conectado el aspersor
pinB = (12) #PIN donde va conectada la ventanilla
pinC = (11) #PIN donde va la alarma por bostezo
##tarjeta = pyfirmata.Arduino(puerto) #Conexión con placa Arduino
##tarjeta.digital[pinA].write(1)

#FUNCIONES
def encender_bomba():
  #tarjeta.digital[pinA].write(0)
  time.sleep(0.3)
  #tarjeta.digital[pinA].write(1)
  

def abrir_ventanilla():	
  #tarjeta.digital[pinB].write(1)
  time.sleep(0.3)
  #tarjeta.digital[pinB].write(0)

executor = ThreadPoolExecutor(max_workers=200)

def send_post_flota(evento,frame):
	frame_resize = imutils.resize(frame, width=480)
	retval, imagen_jpg = cv2.imencode('.jpg', frame_resize)
	imagen_codificada =base64.b64encode(imagen_jpg)
	image_codificada_str= str(imagen_codificada)
	imagen_final = image_codificada_str.replace("b'", "data:image/jpeg;base64,")
	imagen_final = imagen_final.rstrip(imagen_final[-1])
	
	payload =   {"fecha": dt.datetime.now().isoformat(),
  				"idTipoEvento": evento,
  				"idVehiculo": 1,
  				"idConductor": 2,
  				"image": imagen_final
				 }
	
	executor.submit(requests.post,'http://api.dissleep.com.ar/api/Evento',json=payload, timeout=20)

	
def send_post(evento):
	executor.submit(requests.post,'http://juanagustinmasi-001-site1.ctempurl.com/api/Display/SendEvent?even=' + evento, timeout=20)


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

def mouth_aspect_ratio(inner_mouth):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(inner_mouth[1], inner_mouth[7])
	B = dist.euclidean(inner_mouth[2], inner_mouth[6])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(inner_mouth[3], inner_mouth[5])
	D = dist.euclidean(inner_mouth[0], inner_mouth[4])

	# compute the inner_mouth aspect ratio
	imar = (A + B + C) / (2.0 * D)

	# return the eye aspect ratio
	return imar
 

# c	onstruct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 20
INNER_MOUTH_AR_THRESH_APERTURA = 0.8
INNER_MOUTH_AR_THRESH_CIERRE = 0.3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
YAWNS = 0
OPEN = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(fStart, fEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
(imStart, imEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)
start =  dt.datetime.now()
yawns_start = dt.datetime.now()
yawns_detection_time = dt.datetime.now()
# loop over frames from the video stream

presencia=0

img_default = 'RE'
img_bombadeagua = 'MS'
img_alarma_microsuenio = 'AM'
img_sindeteccion = 'SD'
img_ventanilla = 'AV'

while True:
	#send_post(img_default)
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=480)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	
	top = 100
	right = 100
	bottom = 300
	left = 300

	cv2.rectangle(frame,(top,right),(top+bottom,right+left),(0,0,255),2)
		
	face = frame[right:right+left, top:top+bottom]
	frame = cv2.GaussianBlur(frame,(23, 23), 30)
	frame[right:right+face.shape[0], top:top+face.shape[1]] = face
	

	face_detection = gray[right:right+left, top:top+bottom]
	gray = cv2.GaussianBlur(gray,(23, 23), 30)
	gray[right:right+face.shape[0], top:top+face.shape[1]] = face_detection


#	frame2=img_default

	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	if rects:
		presencia = 1
	else:
		presencia = 0

	# loop over the face detections
	for rect in rects:
		presencia=1
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		jaw = shape[fStart:fEnd]
		innerMouth = shape[imStart:imEnd]
		imar = mouth_aspect_ratio(innerMouth)


		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		jawHull = cv2.convexHull(jaw)
		innerMouthHull = cv2.convexHull(innerMouth)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		#cv2.drawContours(frame, [jawHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [innerMouthHull], -1, (0, 255, 0), 1)


		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			now =  dt.datetime.now()
			print("reconozco")
			if (now - start).total_seconds() > 1.5 and (now - start).total_seconds() < 3:
				TOTAL += 1
				#frame2=img_ventanilla
				abrir_ventanilla()	
				send_post(img_ventanilla) # Envío post de indicio de microsueño
				send_post_flota(1,frame)

			if (now - start).total_seconds() > 3:
				TOTAL += 1

				encender_bomba()
				send_post(img_bombadeagua) # Envío post de microsueño
				send_post_flota(3,frame)
				start =  dt.datetime.now()


		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:

			# reset the eye frame counter
			start =  dt.datetime.now()
			
			send_post(img_default)
			
		#Abre	
		if imar >= INNER_MOUTH_AR_THRESH_APERTURA:
			OPEN = 1
		
		#Si esta abierto y cierra sumo 1, si es el primero comienzo a contar
		if OPEN == 1:
			if imar < INNER_MOUTH_AR_THRESH_CIERRE:
				YAWNS +=1
				OPEN = 0
				if YAWNS == 1:
					yawns_start = dt.datetime.now()	
				

		if YAWNS >= 1:
			yawns_now = dt.datetime.now()
			if (yawns_now - yawns_start).total_seconds() >= 30:
				yawns_start = dt.datetime.now()
				YAWNS = 0
				send_post(img_default)
				time.sleep(3)
			else:
				if YAWNS >= 3:
					
				
				#	frame2=img_ventanilla
					send_post(img_alarma_microsuenio) #Envío post de principio de microsueño
					send_post_flota(2,frame)
					yawns_start = dt.datetime.now()
					YAWNS = 0



			

	
	#presencia=0
	# frame de falta de presencia
	if presencia==0 :
		send_post(img_sindeteccion)
	
	
	# show the frame
	#cv2.imshow("Frame", frame)
	#cv2.imshow("Frame2",frame2)
	#key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	#if key == ord("q"):
	#	break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
