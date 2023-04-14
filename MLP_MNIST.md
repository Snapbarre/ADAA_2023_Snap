# Multi-layers Perceptron for handwritten digits recognition

Charger les dépendances i.e. les modules requis pour la suite du script

```
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plot
import cv2
```
## Chargement du dataset

Il existe plusieurs possibilités de télécharger le datasets mnist, mais vous pouvez les récupérer de façon assez simple à partir des datasets exemples de google:

```
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
mnist_train_images=train_images/255
mnist_train_labels=train_labels
mnist_test_images=test_images/255
mnist_test_labels=test_labels

with tf.Session() as s:
  encoded_train_labels = tf.one_hot(mnist_train_labels,10).eval()
  encoded_test_labels = tf.one_hot(mnist_test_labels,10).eval()
```

Vous pouvez aussi les afficher en éxécutant le code suivant

```
taille_batch = 10

images = mnist_train_images[:taille_batch,...]
labels = mnist_train_labels[:taille_batch]

# import matplotlib for visualization
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

for index, image in enumerate(images):
    print("Label:",labels[index])
    print("Digit in the image", np.argmax(labels[index]))
    plt.imshow(image.reshape(28,28),cmap='gray')
    plt.show()
```
En adaptant ces lignes de codes, vous pouvez afficher les images chargée à partir des datasets de votre google drive.

## Architecture du réseau neuronal

A ce point il est important d'analyser le shape de cette dataset i.e. ce qui a été stocké dans les différentes variables.

A partir de maintenant nous commençons à appeler les méthodes de la classe tensorflow.
Ces deux placeholders représentent les endroits où seront placés l'image d'entrée (l'image du chiffre manuscrit) et le résultat du réseau (i.e. le chiffre prédit par le réseau).
```
ph_images=tf.placeholder(shape=(None, 784), dtype=tf.float32)
ph_labels=tf.placeholder(shape=(None, 10), dtype=tf.float32)
```
Les paramètres de l'apprentissage sont le nombre de neurones par couche (nbr_ni), le learning_rate, la taille du batch d'apprentissage (taille_batch) et le nombre d'itération d'apprentissage (nbre_entrainement).
```
nbr_ni=100
learning_rate=0.0001
taille_batch=100
nbr_entrainement=200
```
Ensuite nous définissons les différents poids du réseau wci, les différents biais bci avec la mtéthode ```tf.Variable()```.
Les valeurs initiales des wci sont initialisées de manière aléatoire selon une loi normale tronquée ```tf.truncated_normal()```. Attention, lorsque la méthode ```tf.truncated_normal()``` n'est pas disponible, vous utiliserez la méthode ```tf.normal()```.

Les valeurs initiales des bci sont initialisées à 0 avec la méthode ```np.zeros()```.

```
wci=tf.Variable(tf.truncated_normal(shape=(784, nbr_ni)), dtype=tf.float32)
bci=tf.Variable(np.zeros(shape=(nbr_ni)), dtype=tf.float32)
```
sci est d'abord le résultat de la somme pondrée des entrées. Ce résultat passe ensuite dans la fonction d'activation (ici sigmoid).
```
sci=tf.matmul(ph_images, wci)+bci
sci=tf.nn.sigmoid(sci)
```

Idem que précédemment mais pour les poids et les biais de sortie.

```
wcs=tf.Variable(tf.truncated_normal(shape=(nbr_ni, 10)), dtype=tf.float32)
bcs=tf.Variable(np.zeros(shape=(10)), dtype=tf.float32)
scs=tf.matmul(sci, wcs)+bcs
scso=tf.nn.softmax(scs)
```

Pour la phase d'apprentissage, nous devons définir une fonction de perte à optimiser (loss) et la méthode d'optimisation à utiliser (GradientDescentOptimizer). La fonction de loss est définie comme la crosse entropie. D'autres méthodes d'optimisation sont disponibles et vous pourrez les tester.

```
loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_labels, logits=scs)
train=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(scso, 1), tf.argmax(ph_labels, 1)), dtype=tf.float32))
```

```ft.Session()``` démarre la session qui est appelé s.

Cette session regroupe les différentes itérations d'apprentissage et les prédictions sur la base de test.

Il est important de noter que, puisque python se sert de l'indentation pour "regrouper" certaines actions sous une même boucle par exemple, il faudra veiller à ce que ```s.run()``` soit à un niveau supérieur au niveau d'indentation de ```tf.Session()``` afin que la session s ne soit pas close.

```
with tf.Session() as s:
    
    # Initialisation des variables
    s.run(tf.global_variables_initializer())

    tab_acc_train=[]
    tab_acc_test=[]
    
    for id_entrainement in range(nbr_entrainement):
        print("ID entrainement", id_entrainement)
        for batch in range(0, len(mnist_train_images), taille_batch):
            # lancement de l'apprentissage en passant la commande "train". feed_dict est l'option désignant ce qui est
            # placé dans les placeholders
            s.run(train, feed_dict={
                ph_images: mnist_train_images[batch:batch+taille_batch].reshape(taille_batch,-1),
                ph_labels: mnist_train_labels[batch:batch+taille_batch]
            })

        # Prédiction du modèle sur les batchs du dataset de training
        tab_acc=[]
        for batch in range(0, len(mnist_train_images), taille_batch):
            # lancement de la prédiction en passant la commande "accuracy". feed_dict est l'option désignant ce qui est
            # placé dans les placeholders
            acc=s.run(accuracy, feed_dict={
                ph_images: mnist_train_images[batch:batch+taille_batch].reshape(taille_batch,-1),
                ph_labels: mnist_train_labels[batch:batch+taille_batch]
            })
            # création le tableau des accuracies
            tab_acc.append(acc)
        
        # calcul de la moyenne des accuracies 
        print("accuracy train:", np.mean(tab_acc))
        tab_acc_train.append(1-np.mean(tab_acc))
        
        # Prédiction du modèle sur les batchs du dataset de test
        tab_acc=[]
        for batch in range(0, len(mnist_test_images), taille_batch):
            acc=s.run(accuracy, feed_dict={
                ph_images: mnist_test_images[batch:batch+taille_batch].reshape(taille_batch,-1),
                ph_labels: mnist_test_labels[batch:batch+taille_batch]
            })
            tab_acc.append(acc)
        print("accuracy test :", np.mean(tab_acc))
        tab_acc_test.append(1-np.mean(tab_acc))   
        resulat=s.run(scso, feed_dict={ph_images: mnist_test_images[0:taille_batch].reshape(taille_batch,-1)})
   ```
La dernière ligne de cette session applique de modèle ainsi obtenu sur un batch d'images de test. le résultat sera ensuite affiché (courbe et décision).
   
   Ici on affiche la courbe qui permet de voir comment l'apprentissage a évolué au cours des itérations.
   ```
    plot.ylim(0, 1)
    plot.grid()
    plot.plot(tab_acc_train, label="Train error")
    plot.plot(tab_acc_test, label="Test error")
    plot.legend(loc="upper right")
    plot.show()
   ```
   Affiche le résultat obtenu pour chaque image test du batch.
   ```
   np.set_printoptions(formatter={'float': '{:0.3f}'.format})
    for image in range(taille_batch):
        print("image", image)
        print("sortie du réseau:", resulat[image], np.argmax(resulat[image]))
        print("sortie attendue :", mnist_test_labels[image], np.argmax(mnist_test_labels[image]))
        cv2.imshow('image', mnist_test_images[image].reshape(28, 28))
        if cv2.waitKey()&0xFF==ord('q'):
            break
```
