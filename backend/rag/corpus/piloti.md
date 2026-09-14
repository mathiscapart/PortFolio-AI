---
titre: Piloti
source: piloti.md
---

# Piloti, l'outil de gestion d'un groupe scout

## Piloti en bref

Piloti est un projet personnel, mené sur mon temps libre et non dans mon
travail : une application web que j'ai conçue et développée pour gérer un groupe
des Scouts et Guides de France : inventaire et prêts de matériel, incidents,
finances, planning, lieux de camp, communication et suivi pédagogique, avec des
rôles pour les chefs, les parents et les jeunes. L'interface est entièrement en
français. L'application est auto-hébergée, conforme au RGPD (mineurs,
consentement parental) et publiée en logiciel libre sur
github.com/mathiscapart/piloti. Je l'ai développé en Next.js, React et
TypeScript, avec Prisma, SQLite et better-auth, déployé avec Docker, Traefik et
un tunnel Cloudflare. C'est mon plus gros projet personnel : plus de
250 commits et une cinquantaine de modèles de données.

## Piloti : la stack technique

Les technologies de Piloti : l'application est écrite en TypeScript en mode
strict avec Next.js 16 et son App Router, React 19, Tailwind CSS v4 et
shadcn/ui pour l'interface. La base de données est SQLite, via Prisma 7, en
développement comme en production. better-auth pour l'authentification, Zod pour la validation,
Resend pour les emails, web-push pour les notifications, Vitest pour les tests,
pnpm comme gestionnaire de paquets. En production : Docker, Traefik et un
tunnel Cloudflare.

## Piloti : les rôles et le périmètre d'unité

Piloti distingue l'administrateur, le responsable de groupe, les chefs, le
trésorier, la secrétaire, le responsable matériel, les parents et les jeunes.
Toute la matrice de droits vit dans une seule fonction de permission. Un chef
n'est chef que de sa branche, pas du groupe entier : il ne peut écrire le suivi
pédagogique, pointer les présences ou gérer les événements et leur budget que
pour son unité. La lecture reste ouverte à tout l'encadrement, parce qu'un
groupe scout fonctionne en entraide. En cas de doute, l'accès est refusé.

## Piloti : inventaire, prêts et incidents

Le module matériel gère l'inventaire avec des catégories modifiables depuis
l'administration, les prêts avec leur suivi, les dons de matériel et les
signalements d'incidents avec photos. L'import CSV de matériel est borné, et
les exports CSV sont protégés contre l'injection de formules.

## Piloti : finances et cotisations

Le module finances gère les campagnes de cotisation avec relances et
exemptions, le budget de chaque événement, les dépenses, les caisses et les
tickets de caisse. Le quotient familial existe dans le code mais est masqué de
l'interface : le groupe ne souhaite pas collecter cette donnée de revenu pour
l'instant.

## Piloti : planning et événements

Le planning gère les événements, les inscriptions, les présences, les tâches
et les rappels. On ne s'inscrit qu'aux événements de sa branche, et un parent
inscrit ses enfants depuis son propre compte. Le calendrier s'exporte en
abonnement iCal, avec un lien révocable.

## Piloti : lieux de camp

Chaque lieu de camp a sa fiche, avec des avis filtrables par branche et par
année de camp. L'adresse est géocodée via la Base Adresse Nationale, avec
OpenStreetMap en repli, et l'itinéraire s'ouvre dans Google Maps, Waze ou
Plans. Le propriétaire du terrain, qui n'a pas de compte, reçoit un lien
unique et expirant pour valider ou refuser la conservation de ses coordonnées ;
tant qu'il n'a pas répondu, elles restent invisibles dans l'application.

## Piloti : communication et modération

Piloti intègre des salons de discussion par branche, des messages privés, des
réactions, des sondages, des annonces et des notifications push. Les messages
peuvent être signalés ; la modération masque un contenu plutôt que de le
supprimer, pour garder la preuve. La messagerie privée dépend de l'âge, pour
protéger les mineurs.

## Piloti : suivi pédagogique et tableau de bord

Le suivi pédagogique couvre les étapes de progression, les badges, les
objectifs et les notes de suivi des jeunes. Le tableau de bord s'adapte au
rôle : centre d'action, prochain rendez-vous et bloc dédié aux parents. Une
page de transparence est consacrée à l'empreinte écologique de l'IA.

## Piloti : RGPD et protection des mineurs

La protection des données des mineurs guide la conception de Piloti. Personne
ne peut s'inscrire seul avant quinze ans, et tout mineur doit avoir
l'autorisation d'un responsable légal. Chaque consentement est conservé avec la
version des textes acceptés. La date de naissance n'est modifiable que par un
administrateur, avec une trace. Supprimer un compte revient à l'anonymiser :
l'historique reste cohérent, mais plus rien n'identifie la personne, et le
contenu de ses messages est effacé, sauf ceux visés par un signalement.

## Piloti : architecture de sécurité

Toute modification de données passe par une fonction qui écrit la modification
et son entrée de journal d'audit dans la même transaction : aucune mutation
sans trace. Les routes sont protégées par un proxy, les mots de passe font au
moins douze caractères, et le premier lancement se fait par une page de
création du compte administrateur plutôt que par des comptes prédéfinis. J'ai
aussi mené des audits de sécurité sur le projet et corrigé, par exemple, un
IDOR dans la modération et l'accès public aux fichiers envoyés. J'ai vérifié
certains refus côté serveur en forgeant directement les appels, interface
contournée.

## Piloti : déploiement et infrastructure

En production, aucun port n'est exposé : le trafic passe par Cloudflare, un
tunnel cloudflared, puis Traefik avec en-têtes de sécurité, CSP et limitation
de débit, jusqu'à l'application sur un réseau Docker isolé sans accès à
Internet. Un environnement de staging séparé tourne avec des données fictives,
derrière Cloudflare Access. Le dépôt étant public, j'ai écarté le runner GitHub
auto-hébergé : c'est la machine qui va chercher le code fusionné, jamais GitHub
qui pousse du code à exécuter.

## Piloti : CI et sauvegardes

La CI GitHub Actions enchaîne lint, vérification des types et build, puis
détection de secrets avec gitleaks, audit des dépendances et analyse statique
avec Semgrep. Les sauvegardes sont quotidiennes, cohérentes même pendant que
l'application écrit, chiffrées et authentifiées avec age à clé publique, puis
répliquées hors site. La machine sauvegardée ne peut pas relire ses propres
archives, ce qui la protège d'un rançongiciel, et une restauration réelle
vérifie qu'une archive est exploitable.

## Piloti : licence et décisions

Piloti est sous licence AGPL-3.0 : un prestataire qui hébergerait une version
modifiée doit en publier le code, ce que ni la licence MIT ni la GPL
n'imposeraient pour une application web. Chaque choix technique structurant est
documenté dans un journal de décisions, avec son contexte, le choix retenu et
ses conséquences ; le projet en compte plus de trente.
