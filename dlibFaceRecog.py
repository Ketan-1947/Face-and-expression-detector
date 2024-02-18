import dlib
import cv2 as cv
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier

## loading face data
data = np.load('dlib_face_data.npy')

FaceX = data[:, 0]  
Facey = data[:, 1:].astype(int)

## training face model
FaceModel = KNeighborsClassifier()
FaceModel.fit(Facey, FaceX)


## loading expression data
data = np.load('dlib_expression_data.npy')

ExpressionX = data[:, 0]
Expressiony = data[:, 1:].astype(int)

## training expression model
ExpressionModel = KNeighborsClassifier()
ExpressionModel.fit(Expressiony, ExpressionX)


cap =  cv.VideoCapture(0)   
face_detector = dlib.get_frontal_face_detector()
face_points = dlib.shape_predictor('face_shape68.dat')

while True:
    ret, frame = cap.read()
    frame  = cv.flip(frame, 1)

    faces = face_detector(frame)
    show_face = frame
    if faces:
        face = faces[0]
        ## getting face from screen
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        show_face = frame[y1:y2, x1:x2]
        show_face = cv.resize(show_face , (300, 300))
        show_gray_face = cv.cvtColor(show_face, cv.COLOR_BGR2GRAY)
        name = FaceModel.predict([show_gray_face.flatten()])[0]
        cv.putText(frame, name, (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ## getting face from cropped image
        new_faces = face_detector(show_face)
        if new_faces:
            new_face = new_faces[0]
            landmarks = face_points(show_face , new_face)
            face_data = []
            for n in range(48, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv.circle(show_face, (x, y), 1, (0, 0, 255), -1)
                face_data.append([x-landmarks.part(27).x, y-landmarks.part(27).y])
            face_data = (np.array(face_data)).flatten()
            exp = ExpressionModel.predict([face_data])[0]

            cv.putText(frame, exp, (x1, y2+20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            


    if ret:
        cv.imshow('frame', frame)

    Key = cv.waitKey(1)
    if Key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

        