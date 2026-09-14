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

## Piloti, gestion du matériel d'un groupe scout

Application web de gestion du matériel pour un groupe des Scouts et Guides de
France : inventaire, prêts, incidents et tableau de bord, entièrement en
français. Stack Next.js, React, TypeScript, Tailwind, Prisma et SQLite, avec
authentification. Logiciel libre sous licence AGPL.

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

## SecureShop Blue Team

Mise en situation d'une équipe de défense face à une boutique en ligne
volontairement vulnérable. L'objectif : détecter, alerter et répondre aux
attaques en temps réel sans modifier l'application. La stack comprend le WAF
BunkerWeb, OpenObserve pour les journaux, Prometheus et Grafana pour les
métriques, Blackbox Exporter pour la disponibilité, Alertmanager pour les
alertes et Wazuh comme SIEM. Chaque outil a été choisi après comparaison
d'alternatives, par exemple BunkerWeb face à Coraza et ModSecurity. C'est un
projet d'équipe.

## Cluster Kubernetes k3s

Déploiement d'un cluster k3s sur trois nœuds hébergeant WordPress, PrestaShop
et phpMyAdmin, avec le stockage distribué Longhorn, le load balancer MetalLB et
l'ingress Nginx. J'ai aussi automatisé l'installation de k3s, Longhorn et
l'ingress avec des playbooks Ansible.

## Infrastructure Dunder Mifflin : automatisation et supervision

Playbooks Ansible pour déployer, configurer et maintenir l'infrastructure
Dunder Mifflin, avec des rôles modulaires pour les exporters. Supervision de
cette infrastructure avec Prometheus, Grafana et Alertmanager : exporters
système, web, base de données et Keepalived, plus des sondes de disponibilité.
J'ai également écrit des playbooks Ansible pour ProFTPD.

## Travel Paradise, gestion de visites touristiques

Système de gestion de visites touristiques avec des espaces pour les
administrateurs et les guides, et des pages de statistiques pour analyser les
performances. Le projet comprend une API, une application web d'organisation et
une application mobile.

## GoRadio, streaming radio en Go

Application web en Go pour écouter plusieurs stations de radio, avec un
horoscope mis à jour chaque jour à sept heures, une base SQLite et des métriques
applicatives exposées.

## Autres projets

Une messagerie en temps réel par WebSocket en Go. GRAAL, une API de
réservation de voitures en Node.js avec PostgreSQL. SuperMarket, un site de
vente en ligne avec Symfony. iBlog, un blog pour l'informatique en TypeScript.
KardPrint, une API d'impression en Python. Un générateur de QR codes en Python.
En première année, un casse-briques avec PyGame et un démineur rétro.
