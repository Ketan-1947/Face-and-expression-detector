import cv2 as cv
import dlib
import numpy as np
import os

cap = cv.VideoCapture(0)

face_detector = dlib.get_frontal_face_detector()
face_points = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

faces_data = []
names = []

expressions_data=[]
expressions=[]

name=input('enter name: ')
expression = input('enter expression: ')

while True:
    ret, frame = cap.read()
    frame  = cv.flip(frame, 1)

    faces = face_detector(frame)
    show_face = frame

    if faces:
        face = faces[0]
        face_data = []
        expression_data = []

        ## getting face from screen
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        show_face = frame[y1:y2, x1:x2]
        show_face = cv.resize(show_face , (300, 300))

        gray = cv.cvtColor(show_face, cv.COLOR_BGR2GRAY)
        gray = gray.flatten()
        face_data.append(gray)

        ## getting face from cropped image
        new_faces = face_detector(show_face)

        if new_faces:
            new_face = new_faces[0]
            landmarks = face_points(show_face , new_face)
            
            for n in range(48, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv.circle(show_face, (x, y), 1, (0, 0, 255), -1)
                expression_data.append([x-landmarks.part(27).x, y-landmarks.part(27).y])
    
    if ret:
        cv.imshow('frame', frame)
        cv.imshow('face', show_face)
    
    key = cv.waitKey(1) 
    if key == ord('q'):
        break

    if key == ord('c'):
        face_data = (np.array(face_data)).flatten()
        faces_data.append(face_data)
        names.append([name])

    if key == ord('e'):
        expression_data = (np.array(expression_data)).flatten()
        expressions_data.append(expression_data)
        expressions.append([expression])

    if key == ord('g'):
        expression = input('enter expression: ')

    if key == ord('n'):
        name = input('enter name: ')


## adding face data to file

X1=np.array(names)
y1=np.array(faces_data)

FaceData = np.hstack([X1,y1])

if os.path.exists('dlib_face_data.npy'):
    old_data = np.load('dlib_face_data.npy')
    FaceData = np.vstack([old_data, FaceData])
np.save('dlib_face_data.npy', FaceData)


##adding expression data to file

X2 = np.array(expressions)
y2 = np.array(expressions_data)

ExpressionData = np.hstack([X2,y2])

if os.path.exists('dlib_expression_data.npy'):
    old_data = np.load('dlib_expression_data.npy')
    ExpressionData = np.vstack([old_data, ExpressionData])
np.save('dlib_expression_data.npy', ExpressionData)

cap.release()
cv.destroyAllWindows()
    