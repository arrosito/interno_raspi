# USAGE
 #python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from asyncio import sleep
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
#import winsound 
import datetime as dt

# configuracion puerto para arduino
import pyfirmata
puerto = "/dev/ttyACM0" #Puerto COM de emulación en USB
pin = (13) #PIN donde va conectado el LED
pin_bomba = (12) #PIN donde va conectado el LED

#Conexión con placa Arduino
print("Conectando con Arduino por USB...")
tarjeta = pyfirmata.Arduino(puerto)


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
 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.28
EYE_AR_CONSEC_FRAMES = 20
INNER_MOUTH_AR_THRESH = 0.5

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
YAWNS = 0

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
# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=768)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
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
		cv2.drawContours(frame, [jawHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [innerMouthHull], -1, (0, 255, 0), 1)
		
		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			now =  dt.datetime.now()
			
			if (now - start).total_seconds() > 1.3:
				TOTAL += 1
				#Enciendo el led
				tarjeta.digital[pin_bomba].write(0)
				tarjeta.digital[pin].write(1)

			if (now - start).total_seconds() > 2.5:
				TOTAL += 1
				#Apago led
				#Enciendo bomba
				tarjeta.digital[pin].write(0)
				tarjeta.digital[pin_bomba].write(1)
	
				

		else:
			# reset the eye frame counter
			start =  dt.datetime.now()
			tarjeta.digital[pin_bomba].write(0)
			tarjeta.digital[pin].write(0)

		if imar >= INNER_MOUTH_AR_THRESH:
			YAWNS +=1
			

		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		cv2.putText(frame, "Detecciones: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "IMAR: {:.2f}".format(imar), (300, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "Bostezos: {}".format(YAWNS), (10, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	YAWNS = 0
	# show the frame
	#cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()