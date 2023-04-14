# Votre premier réseau convolutionnel



# Convolutional Neural Network for handwritten digits recognition


L'objectif de ce TP est de développer en python un réseau de neurones convolutif que nous allons apprendre pour reconnaitres les chiffres manuscript de 0 à 9. Comme dans le TP sur le MLP vous utiliserez la base d'images MNIST.

> Question 3 : Bien analyser la structuration du script qui est assez standard et que vous retrouverez dans la suite des expérimentations. Attacher une importance particulière à la structuration des tableaux de données qui servent à l'apprentissage.

Le réseau est composé de 4 couches de convolution. La dimension du noyau des 4 couches est 5 x 5. Les deux premières couches de convolution comportent 16 noyaux de convolutions. Les deux dernières couches de convolution comportent 32 noyaux de convolutions. La sortie de chaque couche de convolution sera activée au travers d'une fonction Relu.
Une couche de pooling sera ajoutée après l'activation de la deuxième et de la quatrième couche convolutive. Le pooling sera calculé sur un voisinage ```2 x 2``` et un strides de 2 dans les deux directions ```strides=[1, 2, 2, 1]```. L'option de padding sera ```padding='SAME'```

Le résultat de la dernière couche de pooling devra être "applati" par la méthode ```tf.compat.v1.layers.flatten```.
La couche applatie sera totalement connectée à une couche de 512 neurones qui seront activés par une fonction sigmoid. Cette couche FC_1 est connectée à une autre couche FC_2 composée de 10 neurones (un neurone par chiffre) avec une activation de type sigmoid.

Voici les différentes étapes pour écrire le nouveau script python.


## Etape 1

Dans la suite de ce TP nous allon repartir de ce que nous avons développé lors du TP MLP.
Nous allons notamment récupérer les lignes suivantes. Ces lignes définissent les parties lecture des images et labels du dataset MNIST, la définition de certaines variables (learning_rate, taille_batch etc.), les placeholder et le l'activation de la session.

L'apprentissage du réseau sera réalisé à partir d'un mini batch d'images de taille 28 x 28. Ce batch comporte  donc ```taille_batch``` images. Le nombre d'itérations de l'apprentissage est fixé par la variable ```epoch_nbr```.
Créer ces deux variables et les initialiser respectivement à 100 et 5. La paramètre ```learning_rate``` restera le même i.e. ```learning_rate=0.001```

```
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plot
import cv2
from tensorflow.examples.tutorials.mnist import input_data

taille_batch=100
epoch_nbr=3
learning_rate=0.001

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
mnist_train_images=train_images/255
mnist_train_labels=train_labels
mnist_test_images=test_images/255
mnist_test_labels=test_labels

mnist_train_images = mnist_train_images.reshape((*mnist_train_images.shape,1))
mnist_test_images = mnist_test_images.reshape((*mnist_test_images.shape,1))

with tf.Session() as s:
  encoded_train_labels = tf.one_hot(mnist_train_labels,10).eval()
  encoded_test_labels = tf.one_hot(mnist_test_labels,10).eval()

"""
# Si la dataset est sur votre drive

mnist_train_images=np.fromfile("dataset/mnist/train-images.idx3-ubyte", dtype=np.uint8)[16:].reshape(-1, 28, 28, 1)/255
mnist_train_labels=np.eye(10)[np.fromfile("dataset/mnist/train-labels.idx1-ubyte", dtype=np.uint8)[8:]]
mnist_test_images=np.fromfile("dataset/mnist/t10k-images.idx3-ubyte", dtype=np.uint8)[16:].reshape(-1, 28, 28, 1)/255
mnist_test_labels=np.eye(10)[np.fromfile("dataset/mnist/t10k-labels.idx1-ubyte", dtype=np.uint8)[8:]]
"""

ph_images=tf.placeholder(shape=(None, 28, 28, 1), dtype=tf.float32)
ph_labels=tf.placeholder(shape=(None, 10), dtype=tf.float32)

#(..... A COMPLETER AVEC L'ARCHIECTURE DU RESEAU CONVOLUTIF ......)

with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    
    tab_train=[]
    tab_test=[]
    
    for id_entrainement in np.arange(nbr_entrainement):
        tab_accuracy_train=[]
        tab_accuracy_test=[]
        print("> Entrainement", id_entrainement)
        for batch in np.arange(0, len(mnist_train_images), taille_batch):
            s.run(train, feed_dict={
                ph_images: mnist_train_images[batch:batch+taille_batch],
                ph_labels: encoded_train_labels[batch:batch+taille_batch]
            })
        for batch in np.arange(0, len(mnist_train_images), taille_batch):
            precision=s.run(accuracy, feed_dict={
                ph_images: mnist_train_images[batch:batch+taille_batch],
                ph_labels: encoded_train_labels[batch:batch+taille_batch]
            })
            tab_accuracy_train.append(precision)
        for batch in np.arange(0, len(mnist_test_images), taille_batch):
            precision=s.run(accuracy, feed_dict={
                ph_images: mnist_test_images[batch:batch+taille_batch],
                ph_labels: encoded_test_labels[batch:batch+taille_batch]
            })
            tab_accuracy_test.append(precision)
        print("  train:", np.mean(tab_accuracy_train))
        tab_train.append(1-np.mean(tab_accuracy_train))
        print("  test :", np.mean(tab_accuracy_test))
        tab_test.append(1-np.mean(tab_accuracy_test))

    plot.ylim(0, 1)
    plot.grid()
    plot.plot(tab_train, label="Train error")
    plot.plot(tab_test, label="Test error")
    plot.legend(loc="upper right")
    plot.show()
    
    resulat=s.run(scso, feed_dict={ph_images: mnist_test_images[0:taille_batch]})
    np.set_printoptions(formatter={'float': '{:0.3f}'.format})
    for image in range(taille_batch):
        print("image", image)
        print("sortie du réseau:", resulat[image], np.argmax(resulat[image]))
        print("sortie attendue :", mnist_test_labels[image], np.argmax(mnist_test_labels[image]))
        cv2.imshow('image', mnist_test_images[image])
        if cv2.waitKey()&0xFF==ord('q'):
            break
```

### Etape 2

Ecrire le code de création de l'architecture du réseau convolutif selon le schéma suivant :

```
couche_prec -- x w --> result_conv --> + b --> result
```

Pour cela vous utiliserez les trois lignes suivantes :

```
# Création des pondérations en les initialisant selon une loi normale
w=tf.Variable(tf.random.truncated_normal(shape=(taille_noyau, taille_noyau, int(couche_prec.get_shape()[-1]), nbr_noyau)))
# Création des biais en les initialisant à 0
b=np.zeros(nbr_noyau)
# Définition de l'opération de convolution sur la couche_prec
result_conv=tf.nn.conv2d(couche_prec, w, strides=[1, 1, 1, 1], padding='SAME')
# Ajout du biais
result=result_conv+b
```

La méthode ```tf.nn.conv2d()``` définit un ensemble de ```nbr_noyau``` convolution 2D. Chaque convolution est réalisée à partir d'un noyau  ```taille_noyau x taille_noyau x profondeur_couche_précédente```. Les noyaux parcourent l'image avec un  ```strides=[1,1,1,1]``` et un ```padding='SAME'```. Toutes les convolutions sont finalement définies par les pondérations ```w``` initialisées à partir d'une loi normale.

(Attention selon les versions de la lib Tensorflow, il faudra utiliser ```tf.random.truncated_normal``` ou ```tf.random_normal```)

Génralement, un biais est ajouté aux résultats de chaque convolution. Ici, les biais b sont de valeur nulles.

Pour l'activation Relu, vous utiliserez la méthode ```tf.nn.relu()```. Pour la couche de pooling, vous utiliserez la méthode ```tf.nn.max_pool()```

Vous designerez un block de convolution composé de:

```
input --> convolution --> relu --> convolution --> relu --> pooling --> result
```

en utilisant ```tf.nn.max_pool``` pour la couche de pooling
Vous designerez ensuite un encodeur de deux blocks de convolutions succesifs

### Etape 3

Mettre en place l'applatissement de la dernière couche de convolution (avec la méthode tf.contrib.layers.flatten)
Créer les deux couches FC en vous réfèrerant au TP MLP.

### Etape 4

Dans cette étape, définissez la fonction de loss et la métrique de précision comme comme dans le TP MLP i.e. ```loss=tf.nn.softmax_cross_entropy_with_logits_v2()``` et ```accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(scso, 1), tf.argmax(ph_labels, 1)), tf.float32))```

L'apprentissage sera défini par la méthode ```tf.train.AdamOptimizer(learning_rate).minimize(loss)```
Vous pourrez également tester la mathode ```tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)``` utilisée dans le TP MLP. 

> Question 4 : Vous comparerez les performances du modèle obtenu avec l'une et l'autre méthode d'optimisation. Par ailleurs, vous observerez comment varie les courbes de performance. Pour cette cette dernère observation, vous pourrez augmenter le nombre d'époch.

### Etape 5
Créer les deux fonctions suivantes afin d'alléger le code.

```
def convolution(couche_prec, taille_noyau, nbr_noyau):
    
    return result
        
def fc(couche_prec, nbr_neurone):
 
    return result
```
### Etape 6

Il est possible d'obtenir de bien meilleurs résultats en ajoutant des couches de Normalisation.
Ces couches son généralement ajoutées juste avant la fonction d'activation.
Dans l'article de S. Ioffe et C. Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, les auteurs proposent de réaliser une normalisation par lot. Vous trouverez les équations qui vous permettront de comprendre le code qui suit et les démonstrations qui ont permis aux auteurs de montrer que cette solution était viable.

Cette normalisation se résume en ces quelques lignes de codes :

```
def normalisation(couche_prec):
    mean, var=tf.nn.moments(couche_prec, [0])
    scale=tf.Variable(tf.ones(shape=(np.shape(couche_prec)[-1])))
    beta=tf.Variable(tf.zeros(shape=(np.shape(couche_prec)[-1])))
    result=tf.nn.batch_normalization(couche_prec, mean, var, beta, scale, 0.001)
    return result
```
Vous remarquerez que les variables scale et beta sont deux variables tf donc elles seront modifiées au cours de l'apprentissage.
Ajouter ce code à votre script et créer une couche de normalisation après chaque couche de convolution et la première couche fully connected. Ces nouvelles couches se situent avant la couche d'activation.

Lancer l'apprentissage et le test du réseau ainsi modifié.

> Question 5 : Que remarquez vous ?




### Etape 7

Voici quelques lignes pour sauvegarder le modèle que vous avez produits.
Lorsque l'apprentissage a nécessité plusieurs heures d'attente (et cela malgrés l'usage d'un ou plusieurs GPU), ces lignes sont primordiales.

Dans un premier temps il faut instancier la classe ```Save=tf.train.Saver()``` pour finalement appeler la méthode ```Save.save(s,./CNN_MNIST_model')```
La méthode save() crée 4 fichiers :
```
CNN_MNIST_model.meta
CNN_MNIST_model.index
CNN_MNIST_model.data-00000-of-00001
checkpoint
```
C'est grâce à ces 4 fichiers qu'il est possible de reconstruire le graph et le réseau afin de l'utiliser dans un autre cadre d'application.

Pour cela il faut utiliser les lignes de codes suivantes :
```
with tf.Session() as s:
    saver=tf.train.import_meta_graph('./model/modele.meta')
    saver.restore(s, tf.train.latest_checkpoint('./model/'))
    graph=tf.get_default_graph()
    images=graph.get_tensor_by_name("images:0")
    sortie=graph.get_tensor_by_name("sortie:0")
    is_training=graph.get_tensor_by_name("is_training:0")
```
Attention, comme vous pouvez le constater, nous récupérons le placeholder is_training.
Or jusqu'à maintenant, nous ne définissions pas ce placeholder.
```is_training``` permet de définir si nous sommes en phase de training ou en phase de test.
Afin de ne pas modifier le code que nous avons écrit et notamment la fonction normalisation, 
nous allons utiliser la fonction ```tf.layers.batch_normalization(result, training=ph_is_training, momentum=momentum)```
qui le fait déjà et qui gère déjà le placeholder is_trainig.
La variable ```momentum=0.99``` est l'initialisation par défaut.

Donc ramplacer tous les appels ```normalisation``` par cette fonction.
Ne changer rien aux paramètres de cette fonction sauf la couche d'entrée (ici result).

Il faudra également ajouter la ligne suivante pour créer ce placeholder dans votre archicture :

```
ph_is_training=tf.placeholder_with_default(False, (), name='is_training')
```

voici le code correction :
```
import cv2
import numpy as np
import tensorflow as tf
from google.colab.patches import cv2_imshow

#cap=cv2.VideoCapture(0)
np.set_printoptions(formatter={'float': '{:0.3f}'.format})
with tf.Session() as s:
    saver=tf.train.import_meta_graph('modele.meta')
    saver.restore(s, tf.train.latest_checkpoint('/content'))
    graph=tf.get_default_graph()
    images=graph.get_tensor_by_name("images:0")
    sortie=graph.get_tensor_by_name("sortie:0")
    is_training=graph.get_tensor_by_name("is_training:0")
    #while True:
    frame=cv2.imread('/content/drive/My Drive/0.png')
    test=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    test=cv2.resize(test, (28, 28))
    for x in range(28):
        for y in range(28):
            if test[y][x]<110:
                test[y][x]=1
            else:
                test[y][x]=0
    cv2_imshow(cv2.resize(test, (120, 120))*255)
    prediction=s.run(sortie, feed_dict={images: [test.reshape(28, 28, 1)], is_training: False})
    #prediction=s.run(sortie, feed_dict={images: [test.reshape(28, 28, 1)]})
    print(prediction, np.argmax(prediction))
    #if cv2.waitKey(20)&0xFF==ord('q'):
    #    break
#cap.release()
#cv2.destroyAllWindows()
```

