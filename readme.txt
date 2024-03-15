Le code fourni utilise OpenCV (cv2) pour effectuer la détection d'objets en temps réel à l'aide du modèle YOLOv3 pré-entraîné. Il capture des images à partir de la webcam, les prétraite, effectue la détection d'objets, applique la suppression non maximale pour éliminer les boîtes redondantes, et dessine des boîtes autour des objets détectés avec leurs labels et confiances correspondantes. La boucle continue jusqu'à ce que la touche 'q' soit enfoncée pour quitter.

Le code utilise les fichiers `yolov3.weights` (poids du modèle), `yolov3.cfg` (configuration du modèle YOLOv3), et `coco.names` (noms des classes) pour la détection des objets.

Pour exécuter ce code, assurez-vous d'avoir OpenCV installé et les fichiers de modèle nécessaires disponibles dans le répertoire de travail.
