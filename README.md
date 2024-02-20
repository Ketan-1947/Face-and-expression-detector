This is a Face and Expressions predictor which utilizes frontal_face_detection by dlib to get
frontal face features and face point predictor to get face features.

This considers mouth as primary source to predict expression.


<h2>How to run on your local device</h2>
  <ol>
    <li>ownload "shape_predictor_68_face_landmarks.dat" from <a href="https://github.com/tzutalin/dlib-android/blob/master/data/shape_predictor_68_face_landmarks.dat">here</a></li>
    <li>Run FaceData.py to gather data</li>
      <ul>
        <li>press "c" to capture "FACIAL FEATURES"</li>
        <li>press "e" to capture "EXPRESSION'S FEATURES"</li>
        <li>press "g" to enter a "NEW EXPRESSION"</li>
        <li>press "n" to enter a "NEW NAME"</li>
      </ul>
    <li>capture atleast five faces and expressions for first run and atleast one each time you gather a new face or expression</li>
    </ol>


<h2>FACE RECOGNITION</h2>
<h4>This program uses KNN model to predict both face and expression</h4>
