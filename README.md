This is a Face and Expressions predictor which utilizes frontal_face_detection by dlib to get
frontal face features and face point predictor to get face features.

This considers mouth as primary source to predict expression.

<p>
<h2>How to run on your local device</h2>
  <li>
    <ol>download "shape_predictor_68_face_landmarks.dat" from <a href="https://github.com/tzutalin/dlib-android/blob/master/data/shape_predictor_68_face_landmarks.dat">here</a></ol>
    <ol>Run FaceData.py to gather data</ol>
      <li>
        <ul>press "c" to capture "FACIAL FEATURES"</ul>
        <ul>press "e" to capture "EXPRESSION'S FEATURES"</ul>
        <ul>press "g" to enter a "NEW EXPRESSION"</ul>
        <ul>press "n" to enter a "NEW NAME"</ul>
      </li>
    <ol>capture atleast five faces and expressions for first run and atleast one each time you gather a new face or expression</ol>
  </li>
</p>

<h2>FACE RECOGNITION</h2>
<h3>This program uses KNN model to predict both face and expression</h3>
