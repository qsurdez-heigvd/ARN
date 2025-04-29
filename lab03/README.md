# Practical Work 3 - Mice's sleep stages classification with MLP

Authors: Quentin Surdez, REDACTED

## 1. Introduction

Nous avons reçu des données EEG de plusieurs souris. Ces données possèdent plusieurs état et à ces états sont associés des amplitudes à certaines fréquences. Nous devons mettre en place des MLPs afin de classifier les données sans savoir quel est leur état évidemment.

Ce rapport permet d'expliciter notre évolution au cours de l'implémentation de MLP d'abord binaire pour classifier les données selon si les souris sont dans un état éveillé ou endormi. Ensuite, nous avons créé un MLP qui permet de classifier entre 3 classes différentes, soit éveillé, sommeil REM, sommeil N-REM.

Les objectifs de ce travail ont été les suivants:

- Explorer et sélectionner les features qui donneront le plus d'informations concernant l'état dans lequel la souris se trouve
- Adresser le problème qui a été relevé durant le premier travail de non-équilibrage du dataset de base
- Développer et optimiser des architectures neuronales pour une bonne classification
- Évaluer différentes stratégies d'optimistation et leur impact sur la performance du modèle

## 2. Preprocessing

La première étape est celle du preprocessing où nous chargeons notre dataset et nous sélectionnons des features d'intérêt. Nous avions pu observer dans le laboratoire précédent que les faetures ayant le plus d'intérêt pour déterminer l'état dans lequel la souris se trouve sont les plus basses. Nous avons donc choisi de garder les amplitudes de 1 à 25 Hertz.

Après ce choix de features, nous devons transformer les données et les normaliser. Cela va permettre de remettre sur une échelle commune les différentes plages de valeurs. Pour ce faire nous utilisons un `StandardScaler` de la librairie `scikit-learn`.

Ensuite, nous entraînons et validons les données à grâce à la validation croisée. Nous choisissons une validation croisée à 3-fold. Nous mélangeons les données afin d'éviter tout biais, surtout dans notre cas où la majeure partie des données se trouvent dans les amplitudes de basses valeurs. 

## 3. Classification binaire

Nous encodons les états éveillé, sommeil REM et sommeil N-REM par 1 pour éveillé et 0 pour sommeil REM et sommeil N-REM. Nous agissons directement sur le dataset afin d'obtenir. Nous ne voulons pas encoder entre -1 et 1 afin de pouvoir utiliser la fonction sigmoid pour l'output.

### 3.1 Résumé du modèle

Le modèle est en soi très simple:

- Comme le vecteur d'input est de 25, pour les 25 features, il y a donc 25 neurones
- Ces 25 neurones sont connectés de manière dense à une première couche cachée composée de 32 neurones, la fonction d'activation choisie est relu. Nous l'avons choisi pour sa simplicité et son efficacité. 32 neurones donnent un bon résultat.
- L'output est composé d'un seul neurone qui est soit actif, soit inactif et la fonction d'activation choisie est sigmoid comme l'output est entre 0 et 1.
- L'optimizer choisi est la descente de gradient stochastique avec un learning rate de 0.01 et un momentum de 0.9. D'autres valeurs ont été essayées sans grand changement dans les résultats.
- La loss function choisie est la binary cross entropy au vu du problème de catégorisation binaire.
- Le nombre d'epochs est de 100. On observe que moins d'epochs serait intéressant après plusieurs essais pour éviter l'overfitting.

### 3.2 Résultat

Voici la matrice ayant donné le meilleur score ainsi que le F1 score associé. Le F1 score global est 0.8928.

![cm binary](./Screenshot%202025-04-08%20at%2018.18.02.png)

Graphique de l'entraînement et de la validation

![graph binary](Screenshot%202025-04-08%20at%2018.15.18.png)

On observe que les courbes sont éloignées l'une de l'autre. La courbe de validation a une pente positive à partir de 20 epochs alors que la courbe de training continue à avoir une courbe négative même proche des 100 epochs. Cela peut nous induire sur un chemin comme quoi notre dataset de training n'est pas forcément représentatif de celui de validation. Ou bien on observe de l'overfitting sur les données de training !

Une amélioration à ce modèle serait soit d'amener de la régulation avec de l'early stopping par exemple, ou bien de réduire le nombre d'epochs.

L'autre caractéristique principale de ces courbes sont les piques qui les composent. En essayant avec la loss function MSE, ces piques disparaissent. Cela fait penser que la binary cross validation nécessite plus d'architecture autour d'elle que de simples couches cachées. La solution pour réduire ces piques peut être de changer d'optimizer ou de loss function comme on le verra par la suite.

On observe dans notre matrice de confusion que notre modèle a plus de peine à catégoriser la classe asleep que la classe awake. Cela nous amène au problème qu'on avait pu déterminer lors de l'exploration de nos données qui sont clairement déséquilibrées.

Une amélioration du preprocessing serait de faire de l'oversampling pour les classes qui n'ont que peu de représentation.

Le F1-score global est de 0.8928 ce qui est un bon score vu que très proche de 1. Ce dernier est une bonne métrique pour les 

## 4. Classification 3 classes 

Ici, nous encodons les états selon les trois classes qui sont présentes dans le dataset soit éveillé, sommeil REM, sommeil N-REM. Nous utilisons un `LabelEncoder` afin d'encoder les états. Ce dernier est mis à disposition par le module `preprocessing` de la librairie `scikit-learn`. Ensuite, nous appliquons `to_categorical` afin d'avoir des vecteurs de 3 dimensions représentant chacune de nos classes. 

Ce preprocessing supplémentaire permet une meilleure prise en main par la fonction de loss categorical crossentropy.

### 4.1 Résumé du modèle

Le modèle est assez basique comme pour celui binaire:

- Comme le vecteur d'input est de 25, pour les 25 features, il y a donc 25 neurones
- Ces 25 neurones sont connectés de manière dense à une première couche cachée composée de 16 neurones, la fonction d'activation choisie est relu. Nous l'avons choisi pour sa simplicité et son efficacité. 16 neurones donnent un bon résultat comparé à 32 dans ce cas-ci. On observe une baisse de l'efficacité corrélée à une hausse du nombre de neurones.
- Une deuxième couche cachée de 8 neurones. La fonction d'activation est toujours relu pour les mêmes raisons. 8 a été choisi car c'est la moitié de 16 et que les résultats avec ce nombre étaient concluants.
- La dernière couche possède 3 neurones, comme le nombre de classes. Ici, la fonction d'activation est softmax. Cette dernière est particulièrement adaptée lors des choix entre plusieurs classes comme elle ramène les résultats entre 0-1
- Nous continuons d'utiliser l'optimizer descente de gradient stochastique
- Nous utilisons la loss function categorical cross entropy comme elle est taillée pour le problème de catégorisation
- Le nombre d'epochs est de 100. On observe que moins d'epochs serait intéressant après plusieurs essais pour éviter l'overfitting.

### 4.2 Résultat

Voici la matrice ayant donné le meilleur score ainsi que le F1 score associé. Le F1 score global est 0.7742.

![cm three-class](./Screenshot%202025-04-08%20at%2018.34.42.png)

Graphique de l'entraînement et de la validation
![graph three-class](./Screenshot%202025-04-08%20at%2018.34.12.png)

Les observations sont particulièrement semblables au graphe de la catégorisation binaire. En effet, le même optimizer ainsi qu'une variante de la fonction de loss cross entropy ont été utilisé dans les deux cas.

Les conclusions restent les mêmes, insérer du Dropout pour potentiellement amoindrir l'overfitting, oversampler les classes sous-représentées.

Le F1 score global étant de 0.7742, c'est une assez bonne valeur. En effet, on ajoute des catégories qui sont très déséquilibrées par rapport aux autres (~2000 N-REM vs. ~12000 awake), et pourtant le F1 score ne drop pas énormément. 

Cependant, il est temps de passer aux améliorations discutées dans les analyses des deux graphiques précédents ! Yay ^^

## 5. Amélioration

Pour la compétition, nous avons décidé de tester plusieurs améliorations que nous avons pu soit lire dans la littérature.

La première amélioration apportée est au niveau du préprocessing des données. En effet, nous souhaitons oversampler les données qui sont en moins grandes quantités que les autres. Pour ce faire, nous avons trouvé une librairie s'appelent `imbalanced-learn` qui permet l'implémentation de plusieurs méthodes d'oversampling. Nous avons choisi une méthode assez basique, mais pas forcément naïve. C'est la méthode SMOTE pour Synthetic Minority Over-sampling Technique qui va chercher les voisins des valeurs des classes minoritaires, "tirer une ligne entre la valeur et ses voisins" et créer une nouvelle valeur sur cette ligne. 

La deuxième amélioration que nous avons mis en place est le `Dropout` entre les différentes layers. Cela permet de forcer le training set de ne pas overfit sur ses données et d'amener une meilleure généralisation

La troisième amélioration est d'insérer des couches de `BatchNormalization` entre les différentes hidden layers. Cela permet d'aller plus vite de 1, mais aussi de garder les gradients dans une échelle acceptable. C'est plutôt utile dans les réseaux complexes et très denses, mais je souhaitais l'utiliser dans le cadre du cours pour savoir comment le mettre en place. 

La quatrième amélioration apportée est l'early stopping. Cette technique est une technique de régularisation. Elle permet de prévenir l'overfitting. Cependant elle fait apparaître un nouveau hyper paramètre qui est la patience ou le nombre d'epochs d'attente acceptable pour une amélioration avant de stopper.

La cinquième amélioration est l'utilisation de l'optimizer Adam. Cet optimizer va permettre d'adapter les learning rates en fonction de la magnitude des gradients récents ainsi que d'appliquer un certain momentum. Il est souvent utiliser dans les réseaux de neurones. 

Enfin, nous avons aussi choisi la métrique pour compiler le modèle. Nous avons choisi `CategoricalAccuracy`. En effet, nous voulions mettre en avant lorsque le modèle choisit correctement la bonne catégorie. Cela nous a permis d'en apprendre un peu plus sur les options de la fonction `compile` et son fonctionnement de manière générale.

Le tout dernier changement effectué est l'ajout d'une couche cachée avec 32 neurones. On a pu essayé sans et les résultats étaient un moins bons, de très peu, mais moins bons quand mêmes


### 5.2 Résultat

Voici la matrice ayant donné le meilleur score ainsi que le F1 associé. Le F1 score global est de 0.8442.

![cm amelio](./Screenshot%202025-04-08%20at%2019.06.49.png)

Graphique de l'entraînement et de la validation

![graph amelio](./Screenshot%202025-04-08%20at%2019.06.07.png)

Il y a passablement de changement par rapport aux précédents graphes. Premièrement on observe que la courbe de validation est en-dessous de la courbe de training. Cela s'explique par les techniques de régulation mises en place. Cela montre que le modèle peut mieux généraliser et n'overfit pas sur ses data de training. Le pattern est malgré tout passablement étrange et atypique. J'ai pu tester sur l'expérience précédente et c'est dû à `Dropout`.

On observe des courbes très lisses comparées aux deux graphes précédents. Cela est dû, principalement, à l'utilisation de régulation ainsi que l'optimizer Adam.

On peut aussi voir qu'il y a très peu de variances entre les modèles. Les différentes `BatchNormalization` peuvent en être la cause, tout comme le nouvel optimizer.

Ensuite, on voit que le nombre d'epochs est proche de 50. En effet, l'early stopping mis en place a pour conséquence que les 3 folds ont un nombre différent d'epochs. Voici le graphe pour chaque fold:

![graph each fold](./Screenshot%202025-04-08%20at%2019.12.17.png)


## 6. Conclusion

Nous avons pu découvrir de nouvelles techniques pour optimiser un MLP à des fins de classification. Nous avons pu comprendre les différents challenges qui se cachent derrière à la fois le preprocessing de données ainsi que le choix de l'architecture.

Nous avons pu itérer sur plusieurs modèles et hyper paramètres d'un modèle. Ces différentes itérations nous ont permis de bien comprendre les impacts de différents choix comme le nombre de neurones présents dans les hidden layers, le choix de l'optimizer. 

Et sincèrement c'était fun de découvrir des nouvelles techniques comme les `BatchNormalization` ou la métrique `CategoricalAccuracy` !


