# Frontend — export statique Next.js

Page parcours (contenu du corpus RAG, indexable) + composant chat branché
sur le SSE de `POST /chat`.

## Variables d'environnement

- `CORPUS_DIR` (temps de build) : dossier Markdown source de la page
  parcours, relatif à `frontend/`. Par défaut `../backend/rag/corpus_demo`
  (persona fictive "Camille Verne") tant que `backend/rag/corpus/*.md`
  contient encore des marqueurs "À REMPLIR". Une fois le vrai corpus rédigé :
  `CORPUS_DIR=../backend/rag/corpus npm run build`.
- `NEXT_PUBLIC_API_URL` (temps de build, intégrée au bundle statique) : URL
  de l'API FastAPI. Par défaut `http://localhost:8000`.

## Commandes

```
npm install
npm run build   # export statique dans out/
npm start       # sert out/ localement (via `serve`)
```

## Tests

```
npm test
```

Vitest et jsdom, avec `@testing-library/react`. Les tests couvrent le parseur SSE
du composant `Chat` : filtrage du commentaire `: ping`, reassemblage d'un bloc
coupe entre deux lectures, `event: error`, sources en bloc terminal, et surtout
le cas d'un flux coupe proprement sans bloc terminal -- qui doit afficher
"reponse interrompue" plutot qu'une demi-reponse presentee comme complete.
