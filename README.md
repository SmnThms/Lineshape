# Lineshape

Parmi les principales modifications :


- les matrices de passages sont maintenant calculées (via Racah) ;


- le calcul est plus rapide (un peu moins d'une minute par forme de raie, au lieu d'environ trois auparavant), en optimisant un peu l'utilisation des tableaux numpy ;


- pour plus de lisibilité, il y a une petite couche d'orienté objet (classes Niveau, Passage, Hamiltonien et Raie) ; les résultats sont produits par des fonctions, et donc appelables et traitables depuis l'extérieur du programme sans avoir à le modifier à chaque fois ; et les différentes étapes de calcul sont réparties dans quatre fichiers :

     I. Bases et changements de base
     
     II. Hamiltoniens
     
     III. Calcul de la forme de raie
     
     IV. Traitement du résultat
