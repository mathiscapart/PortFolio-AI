import type { Metadata } from "next";
import { Bricolage_Grotesque, Source_Serif_4 } from "next/font/google";
import "./globals.css";
import { LIENS, NOM, POSTE, RESUME, SITE_URL } from "@/lib/site";

// Deux familles, deux rôles : la grotesque porte l'interface, la serif porte
// ce qui est dit (réponses et parcours), lu comme de la prose.
const grotesque = Bricolage_Grotesque({ subsets: ["latin"], variable: "--police-interface" });
const serif = Source_Serif_4({ subsets: ["latin"], variable: "--police-texte" });

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: { default: `${NOM} — ${POSTE}`, template: `%s — ${NOM}` },
  description: RESUME,
  authors: [{ name: NOM, url: SITE_URL }],
  openGraph: {
    type: "profile",
    locale: "fr_FR",
    siteName: NOM,
    title: `${NOM} — ${POSTE}`,
    description: RESUME,
    images: [{ url: "/og.png", width: 1200, height: 630, alt: `${NOM}, ${POSTE}` }],
  },
  twitter: { card: "summary_large_image" },
};

// Données structurées schema.org : relient le site à la personne et à ses
// profils (sameAs), ce qui aide un moteur à associer le nom au bon site.
const PERSONNE = {
  "@context": "https://schema.org",
  "@type": "Person",
  name: NOM,
  jobTitle: POSTE,
  url: SITE_URL,
  worksFor: { "@type": "Organization", name: "Doublet" },
  affiliation: { "@type": "EducationalOrganization", name: "SUPINFO" },
  sameAs: [LIENS.linkedin, LIENS.github],
  knowsAbout: ["Intelligence artificielle", "DevSecOps", "Infrastructure", "Cybersécurité", "RAG"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr" className={`${grotesque.variable} ${serif.variable}`}>
      <body>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(PERSONNE).replace(/</g, "\\u003c") }}
        />
        {children}
      </body>
    </html>
  );
}
