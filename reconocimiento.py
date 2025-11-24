import cv2
import face_recognition
import os

# ------------------------------
# 1. Cargar imágenes conocidas
# ------------------------------

ruta_conocidos = "fotos_conocidos"
nombres_conocidos = []
codigos_conocidos = []

for archivo in os.listdir(ruta_conocidos):
    if archivo.endswith((".jpg", ".png", ".jpeg")):
        ruta_imagen = os.path.join(ruta_conocidos, archivo)

        imagen = face_recognition.load_image_file(ruta_imagen)
        codigo = face_recognition.face_encodings(imagen)[0]

        codigos_conocidos.append(codigo)
        nombre = os.path.splitext(archivo)[0]  # nombre sin extensión
        nombres_conocidos.append(nombre)

print("🟢 Se cargaron las imágenes conocidas correctamente.")

# ------------------------------
# 2. Abrir la cámara
# ------------------------------
camara = cv2.VideoCapture(0)

print("📷 Iniciando reconocimiento...")

while True:
    ret, frame = camara.read()
    if not ret:
        break

    pequeño = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_pequeño = pequeño[:, :, ::-1]

    ubicaciones = face_recognition.face_locations(rgb_pequeño)
    codigos = face_recognition.face_encodings(rgb_pequeño, ubicaciones)

    for codigo_rostro, ubicacion in zip(codigos, ubicaciones):

        coincidencias = face_recognition.compare_faces(codigos_conocidos, codigo_rostro)
        nombre = "DESCONOCIDO"

        if True in coincidencias:
            indice = coincidencias.index(True)
            nombre = nombres_conocidos[indice]

        top, right, bottom, left = ubicacion
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, nombre, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)

    cv2.imshow("Reconocimiento Facial", frame)

    if cv2.waitKey(1) == 27:  # Tecla ESC
        break

camara.release()
cv2.destroyAllWindows()

