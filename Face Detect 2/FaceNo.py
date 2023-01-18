import cv2
#----------------------------------------------------------
#    Face Detection and Recognition By Com Engg TY-24
#----------------------------------------------------------

#img- To find the feature i.e face
#color - color of rectangle to be drawn over face  
#text - Display name of body part (Mouth, eyes, etc..)
#scaleFactor - Our classifier will be scale some extend to detect smaller face as well as larger face
#minNeighbors -  How many features we need to prove it is a face..!!   

def draw_boundary(img, classifier, scaleFactor,minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert the img into gray scale img to extract features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = [] #List to hold co-ordinate for face
    for(x,y,w,h) in features:   # x_co-ordinate, y_co-ordinate, width and height
        cv2.rectangle(img, (x,y), (x+w,y+h), color,2) #Method to draw rectangle 
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA) 
        #Diplay text little over rect i.e 4px over
        coords = [x,y,w,h] #Update co-ordinates
    return coords, img

def detect(img, faceCascade):
    color = {"blue": (255,255,0,0 ), "red": (0,0,255), "green": (0,255,0)}
    coords, img = draw_boundary(img, faceCascade, 1.1, 10, color["blue"], "Face")
    return img

#Load classifier
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#Read video stream from web cam 
video_capture = cv2.VideoCapture(0) # Write 0 if you are using inbuilt web cam
                                    # Write -1 for external web cam

while True:
    _, img = video_capture.read() # Read video as image
    # video_capture.read() returns two parameters but I only use img and remaining is _
    img = detect(img, faceCascade)
    cv2.imshow("Face Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # If user enter 'q' break loop
        break

video_capture.release()
cv2.destroyAllWindows()
     