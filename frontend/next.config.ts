import type { NextConfig } from "next";

// Export statique : portfolio indexable, pas de serveur Node à héberger
// (décision prise au plan v0, étape 6). Pas de next/image optimisé sans
// serveur, on ne l'utilise pas ici.
const nextConfig: NextConfig = {
  output: "export",
  // Génère parcours/index.html : sans ça, nginx redirige /parcours vers le
  // dossier /parcours/ (qui ne contient pas de page) et répond 403.
  trailingSlash: true,
};

export default nextConfig;
