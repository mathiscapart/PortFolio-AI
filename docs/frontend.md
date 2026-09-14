# Frontend — export statique Next.js

Code dans `frontend/`. Deux pages : l'accueil avec l'interface vocale
(`components/VoiceChat.tsx`) et la page parcours, générée au build à partir du
corpus Markdown du RAG (indexable, lisible même backend éteint). S'y ajoutent
`sitemap.xml`, `robots.txt`, l'image Open Graph et les favicons.

## Variables d'environnement (lues au build)

- `NEXT_PUBLIC_API_URL` : URL de l'API, intégrée au bundle statique. En
  production `https://mathiscapart.xyz/api`. **Obligatoire** pour un build de
  production : sans elle, le build échoue exprès plutôt que d'envoyer chaque
  visiteur vers sa propre machine.
- `CORPUS_DIR` : dossier Markdown de la page parcours, relatif à `frontend/`.
  Par défaut le vrai corpus `../backend/rag/corpus`. Avec
  `../backend/rag/corpus_demo`, le site affiche la persona fictive de
  démonstration et un bandeau l'annonce (voir [corpus-demo.md](corpus-demo.md)).

## Commandes

```powershell
cd frontend
npm ci
$env:NEXT_PUBLIC_API_URL = "https://mathiscapart.xyz/api"
npm run build   # export statique dans out/ ; en production, c'est la mise en ligne
npx next dev    # développement local
```

## Interface vocale

Contrat WebSocket `/voice` : le navigateur envoie des trames PCM16 mono 24 kHz
de 80 ms puis `{"type":"end"}` ; le serveur renvoie `transcript`, `token`, des
trames audio PCM16 24 kHz, puis `sources` (bloc terminal) ou `error`.

Points propres aux navigateurs mobiles, tous couverts par des tests :

- Les AudioContext sont créés et repris **dans le geste** de l'utilisateur,
  avant toute attente : sinon Safari iOS les laisse muets.
- Ils sont **réutilisés** d'une question à l'autre, et le module de capture
  n'est chargé qu'une fois : les recréer empêchait d'enchaîner sur iOS.
- Le micro tourne à la fréquence de l'appareil et est rééchantillonné en 24 kHz
  (`Reechantillonneur`, `lib/voix.ts`).
- Seuls les événements du socket actif sont écoutés : Safari émet parfois une
  erreur tardive sur le socket d'une question terminée.

## Tests

```powershell
cd frontend
npx vitest run
npx tsc --noEmit
```

Vitest et jsdom, avec `@testing-library/react`. Ils couvrent la conversion et
le découpage PCM, le rééchantillonnage, l'interprétation des messages `/voice`,
la garantie anti demi-réponse (« Réponse interrompue » si le socket se ferme
sans bloc terminal), l'enchaînement de questions et les événements tardifs d'un
ancien socket.
