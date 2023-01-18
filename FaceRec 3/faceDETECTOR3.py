import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('D:/VIT CLG/mod5edi/FaceRec 3/classifiers/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("D:/VIT CLG/mod5edi/FaceRec 3/FaceRecTrain/trainingdata.yml")
id=0
#font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
font = cv2.FONT_HERSHEY_SIMPLEX
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(id==2):
            id="KK"
        if id==1:
            id="Karan"
        if id==3:
            id="KDK"
        if id==4:
            id="DON"
        if id==5:
            id='Arjun'
        if id==6:
            id="Aryan"
        #cv2.putText(img,"Karan",(x,y+h),font,255,(255, 0, 0))
        cv2.putText(img, str(id), (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
