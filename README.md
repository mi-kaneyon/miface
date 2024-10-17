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
# explain 

```
python many_face.py

```

## original version
```

2d_face.py

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


- animal example

![Test Image 3](/fuku.png)

- human face

[sample.webm](https://github.com/user-attachments/assets/e0f47281-67f2-44e4-9001-17a84bcce910)

