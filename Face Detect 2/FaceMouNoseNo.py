import cv2
#-----------------------------------------
#    Face Detection and Recognition 
#-----------------------------------------

#img- To find the feature i.e face
#color - color of rectangle to be drawn over face  
#text - Display name of body part (Mouth, eyes, etc..)
#scaleFactor - Our classifier will be scale some extend to detect smaller face as well as larger face
#minNeighbors -  How many features we need to prove it is a face..!!   

def draw_boundary(img, classifier, scaleFactor,minNeighbors, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert the img into gray scale img to extract features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = [] #List to hold co-ordinate for face
    for(x,y,w,h) in features:   # x_co-ordinate, y_co-ordinate, width and height
        cv2.rectangle(img, (x,y), (x+w,y+h), color,2) #Method to draw rectangle 
        id,_ = clf.predicts(gray_img[y:y+h, x:x+w])
        if id == 1:
            cv2.putText(img, "Karan", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA) 
            #Diplay text little over rect i.e 4px over
        
        coords = [x,y,w,h] #Update co-ordinates
    return coords

def recognize(img, clf, faceCascade):
    color = {"blue": (255,255,0,0 ), "red": (0,0,255), "green": (0,255,0), "white": (255,255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color["white"], "Face", clf)
    return img

def detect(img, faceCascade, eyeCascade,noseCascade, mouthCascade):
    color = {"blue": (255,255,0,0 ), "red": (0,0,255), "green": (0,255,0), "white": (255,255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color["blue"], "Face")

    if len(coords) == 4:    #[x,y,w,h] No face detects means coords value is 0
        #Crop face because to detect mouth and nose area of interest is now only face
        ro_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        coords = draw_boundary(ro_img, eyeCascade, 1.1, 14, color["red"], "Eyes") 
        coords = draw_boundary(ro_img, noseCascade, 1.1, 5, color["green"], "Nose") 
        coords = draw_boundary(ro_img, mouthCascade, 1.1, 20, color["white"], "Mouth") 
        

        
    return img

#Load classifier
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
noseCascade = cv2.CascadeClassifier("Nariz.xml")
mouthCascade = cv2.CascadeClassifier("mouth.xml")

# Loading custom classifier to recognize
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifierFaces.yml")

#Read video stream from web cam 
video_capture = cv2.VideoCapture(0) # Write 0 if you are using inbuilt web cam
                                    # Write -1 for external web cam

while True:
    # Reading image from video stream
    _, img = video_capture.read() # Read video as image
    # video_capture.read() returns two parameters but I only use img and remaining is _
    #img = detect(img, faceCascade, eyeCascade, noseCascade, mouthCascade)
    # Call method we defined above
    img = recognize(img, clf, faceCascade)
    # Writing processed image in a new window
    cv2.imshow("Face Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # If user enter 'q' break loop
        break

video_capture.release()
cv2.destroyAllWindows()
     