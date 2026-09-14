import type { MetadataRoute } from "next";
import { SITE_URL } from "@/lib/site";

// Export statique : sans force-static, la route n'est pas générée au build.
export const dynamic = "force-static";

export default function sitemap(): MetadataRoute.Sitemap {
  const maintenant = new Date();
  return [
    { url: `${SITE_URL}/`, lastModified: maintenant, changeFrequency: "monthly", priority: 1 },
    { url: `${SITE_URL}/parcours/`, lastModified: maintenant, changeFrequency: "monthly", priority: 0.8 },
  ];
}
