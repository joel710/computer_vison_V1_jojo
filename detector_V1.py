import cv2
import numpy as np

# Charger le modèle YOLO pré-entraîné
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

# Utiliser la webcam (0 pour la webcam par défaut)
cap = cv2.VideoCapture(0)

while True:
    # Lire chaque image de la webcam
    ret, frame = cap.read()

    # Récupérer les informations sur les dimensions de l'image
    height, width, channels = frame.shape

    # Prétraitement de l'image pour la détection d'objets
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Obtenez les prédictions
    outs = net.forward(layer_names)

    # Initialiser des listes pour les boîtes, les confidences et les classes détectées
    boxes = []
    confidences = []
    class_ids = []

    # Analyser les sorties de la couche YOLO
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Seuil de confiance
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculer les coordonnées de la boîte englobante
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Appliquer la suppression non maximale pour éliminer les boîtes redondantes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Dessiner les boîtes détectées sur l'image
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Couleur verte pour la boîte

            # Dessiner la boîte et afficher la classe et la confiance
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Afficher l'image avec les boîtes détectées
    cv2.imshow("Object Detection", frame)

    # Quitter la boucle si la touche 'q' est enfoncée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et fermer la fenêtre
cap.release()
cv2.destroyAllWindows()
# by @jojo
