# Utilisation de la plateform Colab de Google

Aller l'adresse suivante : https://colab.research.google.com/notebooks/welcome.ipynb#recent=true
Base : Jupyter transforme votre navigateur web en interpréteur interactif. Il combine plusieurs langages
de programmation notamment Python.

Lorsque vou êtes sur cette page web, connectez-vous avec votre compte Goole (en haut à droite) puis connectez-vous à
un "environnement d'exécution hébergé"

Si vous souhaitez bénéficier d'une accélération matérielle : Menu Modifirer -> Paramètres du Notebook

A partir de là vous bénéficier d'un environnement virtuel hébergé sur les serveurs de Google. Vous pouvez exécuter le code
que vous souhaitez.
Le projet que vous produisez peut être sauvegardé sur le Google Drive.
Ce même projet peut utiliser des données (images MNIST par exemple) que vous aurez stockées sur votre Google Drive.

Pour cela il faudra monter le google drive de la manière suivante :

```
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
```
Il pourrait vous être demandé une authorisation sous la forme d'une suite de caractère.

Attention certains modules ne sont pas dispo dans cet environnement car ils font planter l'application. Par conséquent, Google
propose une version adaptée de ces modules : cv2 est dans ce as. Il faut importer
```from google.colab.patches import cv2_imshow``` pour pouvoir afficher une image avec la fonction ```cv2_imshow(image)```
