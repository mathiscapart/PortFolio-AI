---
titre: Projets
source: projets.md
---

# Projets

## Moteur de recherche sémantique interne

Recherche sur quatre-vingt mille documents techniques hétérogènes, du PDF scanné
au ticket de support. La difficulté n'était pas le modèle mais le découpage :
les tableaux de spécifications perdaient leur sens une fois coupés au milieu.
J'ai construit un découpage sensible à la structure qui préserve les tableaux
entiers. Le taux de réponses jugées utiles est passé de quarante à
soixante-dix-huit pour cent.

## Détection d'anomalies sur capteurs industriels

Douze mille capteurs, une remontée toutes les trente secondes. Le piège était le
déséquilibre des classes : moins d'une mesure sur dix mille correspond à une
panne réelle. Un modèle naïf atteignait quatre-vingt-dix-neuf virgule neuf pour
cent de précision en ne prédisant jamais rien.

## Assistant vocal embarqué

Reconnaissance vocale et synthèse en local sur carte graphique grand public,
sans appel réseau. Contrainte principale : faire tenir trois modèles en douze
gigaoctets de mémoire vidéo sans que le rechargement de l'un n'ajoute des
secondes de latence à chaque tour de parole.
