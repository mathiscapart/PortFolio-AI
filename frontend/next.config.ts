import type { NextConfig } from "next";

// Export statique : portfolio indexable, pas de serveur Node à héberger
// (décision prise au plan v0, étape 6). Pas de next/image optimisé sans
// serveur, on ne l'utilise pas ici.
const nextConfig: NextConfig = {
  output: "export",
};

export default nextConfig;
