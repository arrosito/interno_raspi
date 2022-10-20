#Imports
import numpy as np
import cv2
import datetime

#Inicializo los archivos de reconocimiento
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

#Estado inicial
first_read = True

#Comienzo a capturar video
cap = cv2.VideoCapture(0)
ret,img = cap.read()

start =  datetime.datetime.now()
end =  datetime.datetime.now()


while(ret):
    ret,img = cap.read()
    #Convierto la imagen a escala de grises
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Filtro para remover impurezas de la imagen
    gray = cv2.bilateralFilter(gray,5,1,1)

    #Detecto la region de la imagen donde se encuentra la cara
    faces = face_cascade.detectMultiScale(gray, 1.3, 5,minSize=(200,200))

    #Si encuentro una region
    if(len(faces)>0):
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            #Tomo region de la cara donde deberian estar los ojos
            roi_face = gray[y:y+h,x:x+w]
            roi_face_clr = img[y:y+h,x:x+w]

            #Detecto patron de ojos
            eyes = eye_cascade.detectMultiScale(roi_face,1.3,5,minSize=(50,50))

            #Examino la longitud de los ojos
            if(len(eyes)>=2):
                    cv2.putText(img, "Ojos abiertos", (70,70), cv2.FONT_HERSHEY_PLAIN, 2,(0,255,0),2)
                    start =  datetime.datetime.now()
                    end =  datetime.datetime.now()
            else:
                    cv2.putText(img, "Ojos cerrados", (70,70), cv2.FONT_HERSHEY_PLAIN, 2,(255,0,0),2)
                    end = datetime.datetime.now()

            total = end - start

            if((end - start).total_seconds() > 2):
                cv2.putText(img, "2 segundos!!!!!!!!!!", (100,120), cv2.FONT_HERSHEY_PLAIN, 2,(255,0,0),2)
           

    else:
        cv2.putText(img,"No se detecta rostro",(100,100),cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)

    #Controlling the algorithm with keys
    cv2.imshow('img',img)
    a = cv2.waitKey(1)
    if(a==ord('q')):
        break

cap.release()
cv2.destroyAllWindows()