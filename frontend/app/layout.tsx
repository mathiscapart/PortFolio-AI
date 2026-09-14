import type { Metadata } from "next";
import { Bricolage_Grotesque, Source_Serif_4 } from "next/font/google";
import "./globals.css";
import { corpusEstFictif } from "@/lib/corpus";

// Deux familles, deux rôles : la grotesque porte l'interface, la serif porte
// ce qui est dit (réponses et parcours), lu comme de la prose.
const grotesque = Bricolage_Grotesque({ subsets: ["latin"], variable: "--police-interface" });
const serif = Source_Serif_4({ subsets: ["latin"], variable: "--police-texte" });

export const metadata: Metadata = {
  title: "Portfolio IA",
  description: "Portfolio interactif : parcours et assistant vocal qui répond sur son contenu.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr" className={`${grotesque.variable} ${serif.variable}`}>
      <body>
        {corpusEstFictif() && (
          <div role="note" className="bandeau-demo">
            <strong>Démonstration technique.</strong> Le parcours et les réponses
            portent sur une persona fictive, générée pour illustrer la chaîne
            vocale. Ce ne sont pas les informations du propriétaire de ce domaine.
          </div>
        )}
        {children}
      </body>
    </html>
  );
}
