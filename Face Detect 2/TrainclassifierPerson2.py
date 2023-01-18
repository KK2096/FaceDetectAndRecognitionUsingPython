import numpy as np
from PIL import Image
import os, cv2

#data_dir -  Directory in which images are present
def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)] #Append all the images in list
    #Two lists 1. For face 2. ids of users face
    #On the same index the face corresponds ids of the users face
    faces = []
    ids = []
    for image in path:
        img = Image.open(image).convert('L') #Opening an image and converting them into gray scale
        imageNp = np.array(img, 'uint8') #Convert the image to numpy to train our classifier
        id = int(os.path.split(image)[1].split(".")[1])
        faces.append(imageNp) #Append image in numpy format to the faces
        ids.append(id)  #Append id to ids list
    ids = np.array(ids)  #Convert id list into numpy format
    #Feed classifier 
    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifierFaces.yml")

train_classifier("Data",)