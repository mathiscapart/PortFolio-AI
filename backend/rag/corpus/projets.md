---
titre: Projets
source: projets.md
---

# Projets

## PortFolio-AI, ce portfolio vocal

Le site sur lequel vous êtes. Un visiteur pose sa question à voix haute ; la
reconnaissance vocale Kyutai la transcrit en direct, une recherche dans une base
vectorielle Qdrant retrouve les passages utiles de mon parcours, le modèle
Qwen3 rédige la réponse et Pocket TTS la lit avec une copie de ma voix. Tout
tourne en local sur mon homelab, sur une carte AMD avec ROCm sous Windows,
exposé par Traefik et un tunnel Cloudflare, sans aucune API d'inférence
externe.

### Les défis techniques de PortFolio-AI

Tout tient sur une carte graphique de 12 Go : chaque modèle a été mesuré avant
d'être retenu. Qwen3 8B tourne à 61 tokens par seconde, la reconnaissance
vocale en streaming prend 2,5 Go de mémoire vidéo, et la synthèse vocale tourne
sur le processeur, dans un process séparé, parce qu'elle était quarante fois
plus lente sous la version ROCm de PyTorch. Le premier son de la réponse arrive
environ deux secondes et demie après la fin de la question. La CI vérifie les
tests, le build, les secrets et les dépendances vulnérables.

## Piloti

Piloti, mon application de gestion d'un groupe scout, est décrit en détail dans
sa propre fiche.

## LaRuche, plateforme de honeypots

Plateforme modulaire de pots de miel pour la recherche en sécurité et le
renseignement sur les menaces : honeypots SSH, FTP et HTTP. Un reverse proxy
Apache sert les vraies ressources d'un WordPress pour rendre le leurre difficile
à détecter. Les journaux sont enrichis avec GeoIP, AbuseIPDB et GreyNoise, le
comportement des attaquants est classé, puis Fluent Bit les envoie vers des
tableaux de bord OpenObserve. Un module d'attaque simule des attaques réalistes
pour valider le dispositif. Déployé avec Docker Compose.

### Mon rôle dans LaRuche

LaRuche est un projet d'équipe. Sur LaRuche, j'ai réalisé la gestion des
connexions et le modèle d'émulation du honeypot SSH, c'est-à-dire ce qui fait
croire à l'attaquant qu'il est sur un vrai serveur.

### Les leurres de LaRuche

Le honeypot SSH écoute sur les ports 22 et 2222 et accepte des identifiants
faibles ; le honeypot FTP écoute sur les ports 21 et 2121 et accepte la
connexion anonyme. Le honeypot HTTP imite un WordPress 6.5.2 : fausse page de
connexion qui capture les identifiants, API REST, phpinfo, phpMyAdmin et un
fichier .env piégé. Devant lui, un vrai Apache 2.4.57 sert les fichiers
statiques authentiques de WordPress avec leurs vrais en-têtes et pages
d'erreur.

### L'analyse et la validation dans LaRuche

L'analyseur classe chaque attaquant en bot, bruteforceur, humain ou scanner.
Le module d'attaque, non destructif, découvre les services avec nmap puis les
attaque avec les listes de mots de SecLists, pour vérifier que toute la chaîne
de détection enregistre bien l'activité. La CI passe Ruff, Bandit, pytest et
Trivy, et les versions sont publiées automatiquement avec semantic-release.

## SecureShop Blue Team

Mise en situation d'une équipe de défense face à une boutique en ligne
volontairement vulnérable. L'objectif : détecter, alerter et répondre aux
attaques en temps réel sans modifier l'application. La stack comprend le WAF
BunkerWeb, OpenObserve pour les journaux, Prometheus et Grafana pour les
métriques, Blackbox Exporter pour la disponibilité, Alertmanager pour les
alertes et Wazuh comme SIEM. Chaque outil a été choisi après comparaison
d'alternatives, par exemple BunkerWeb face à Coraza et ModSecurity. C'est un
projet d'équipe.

### Les choix d'architecture de SecureShop

La boutique, écrite en PHP, expose des failles du Top 10 OWASP : XSS, CSRF,
injection SQL, IDOR et authentification cassée. La défense est découpée en
quatre stacks Docker isolées par des réseaux dédiés : application, WAF,
supervision et SIEM. OpenObserve a été préféré à ELK, trop gourmand, et à Loki,
limité pour analyser le contenu des journaux. Wazuh a été préféré à Splunk,
coûteux, et à Graylog, sans réponse active. BunkerWeb bannit automatiquement
les adresses au comportement suspect.

### Les règles de détection de SecureShop

Des règles Wazuh sur mesure détectent l'upload et l'exécution d'un webshell
grâce au contrôle d'intégrité des fichiers, les motifs d'injection SQL, le
brute force (huit tentatives en deux minutes depuis une même adresse), les
scans de répertoires, les outils connus comme sqlmap, nikto ou hydra,
l'énumération IDOR des factures et l'accès aux fichiers sensibles comme .env ou
.git. Des alertes signalent aussi l'indisponibilité du site et les connexions
au panneau d'administration.

## Cluster Kubernetes k3s

Déploiement d'un cluster k3s sur trois nœuds hébergeant WordPress, PrestaShop
et phpMyAdmin, avec le stockage distribué Longhorn, le load balancer MetalLB et
l'ingress Nginx. J'ai aussi automatisé l'installation de k3s, Longhorn et
l'ingress avec des playbooks Ansible.

### Détails du cluster k3s

Le cluster comprend un nœud maître et deux nœuds de travail, installés par
scripts puis par Ansible. Il héberge aussi Redict, le fork libre de Redis. La
supervision est déployée avec Helm : Prometheus et Grafana, avec des tableaux
de bord pour les sondes Blackbox et pour le stockage Longhorn.

## Infrastructure Dunder Mifflin : automatisation et supervision

Playbooks Ansible pour déployer, configurer et maintenir l'infrastructure
Dunder Mifflin, avec des rôles modulaires pour les exporters. Supervision de
cette infrastructure avec Prometheus, Grafana et Alertmanager : exporters
système, web, base de données et Keepalived, plus des sondes de disponibilité.
J'ai également écrit des playbooks Ansible pour ProFTPD.

### Détails de l'infrastructure Dunder Mifflin

Un playbook déploie l'ensemble des services et du monitoring, un second
applique la maintenance et les mises à jour sans interrompre les services. La
supervision, lancée avec Docker Compose, suit les ressources système avec Node
Exporter, la bascule haute disponibilité VRRP avec l'exporter Keepalived, les
serveurs Nginx et Apache, la base MySQL, et la disponibilité en HTTP, ICMP et
TCP avec Blackbox Exporter.

## Travel Paradise, gestion de visites touristiques

Système de gestion de visites touristiques avec des espaces pour les
administrateurs et les guides, et des pages de statistiques pour analyser les
performances. Le projet comprend une API, une application web d'organisation et
une application mobile.

### Les applications de Travel Paradise

L'application web des organisations, en React 19, React Router, Tailwind CSS
et TypeScript, permet de créer une organisation et son administrateur, de gérer
les utilisateurs et les guides, et de suivre les visites, les réservations, les
notes et le taux de présence des visiteurs. L'application mobile, en React
Native avec Expo, est réservée aux guides : elle affiche leurs visites, leur
profil et des graphiques. L'authentification repose sur des jetons JWT.

## GoRadio, streaming radio en Go

Application web en Go pour écouter plusieurs stations de radio, avec un
horoscope mis à jour chaque jour à sept heures, une base SQLite et des métriques
applicatives exposées. Le front est en HTML, CSS et JavaScript.

## Autres projets

Une messagerie en temps réel par WebSocket en Go. GRAAL, une API de
réservation de voitures en Node.js avec PostgreSQL et Docker. SuperMarket, un
site de vente en ligne avec Symfony. iBlog, un blog pour l'informatique en
TypeScript. KardPrint, une API d'impression en Python. Un générateur de QR codes
en Python. En première année, un casse-briques avec PyGame et un démineur rétro.
Enfin, en Python : un jeu du pendu en console, une calculatrice et une
visionneuse d'images avec Tkinter.
