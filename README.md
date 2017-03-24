# Lineshape

Dernières versions du programme de calcul de forme de raie : fluo3S_H_mars2017.


Parmi les principales modifications, par rapport à fluo3S_H_fev2017 :


- les matrices de passages sont maintenant calculées (via Racah) ;


- le calcul est plus rapide (un peu moins d'une minute par forme de raie, au lieu d'environ trois auparavant), en optimisant un peu l'utilisation des tableaux numpy ;


- pour plus de lisibilité, il y a une petite couche d'orienté objet (classes Niveau, Passage, Hamiltonien et Raie) ; les résultats sont produits par des fonctions, et donc appelables et traitables depuis l'extérieur du programme sans avoir à le modifier à chaque fois ; et les différentes étapes de calcul sont réparties dans quatre fichiers :

          I. Bases et changements de base
     
          II. Hamiltoniens
     
          III. Calcul de la forme de raie
     
          IV. Traitement du résultat


Le programme '_Recherche de la distribution de vitesse_' réalise les courbes qui permettent d'étudier l'influence de v0 et sigma.

Les programmes '_…\_avec\_D…_' calculent les matrices de changement de bases pour l'hydrogène et le deutérium en incluant les niveaux D.
