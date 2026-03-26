import mediapipe as mp
print("Mediapipe version:", mp.__version__)
mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(max_num_faces=1) as face_mesh:
    print("FaceMesh initialized successfully")
