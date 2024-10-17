# miface
midiapipe facedection fun arrange

> [!TIP]
> Overlap 2d human image on the real time cam.
> face is not too real, looks like doll.

# Arrangement
**you can change image after using other image on your face **

```
# human landmark picture loading
target_face_path = "./soseki.png"  <--- changable
landmark_path = "./landmark.json"

landmark.json for adjusting real time face matching area

```



# reuqirement

```
pip install scipy
pip install mediapipe
pip install opencv-python

```
# About mediapipe

https://ai.google.dev/edge/mediapipe/solutions/guide

- This script is utilized face detection function
- On the mesh, face 2d image set as triangles

# sample 

![Test Image 3]([sample.webm](https://github.com/mi-kaneyon/miface/blob/main/sample.webm))
