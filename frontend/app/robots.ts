import type { MetadataRoute } from "next";
import { SITE_URL } from "@/lib/site";

// Cloudflare ajoute ses propres directives (blocage des robots d'entraînement
// IA) à ce fichier ; celui-ci n'apporte que l'autorisation et le sitemap.
export const dynamic = "force-static";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: { userAgent: "*", allow: "/" },
    sitemap: `${SITE_URL}/sitemap.xml`,
  };
}
