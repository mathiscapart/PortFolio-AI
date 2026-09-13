import type { Metadata } from "next";
import "./globals.css";
import { corpusEstFictif } from "@/lib/corpus";

export const metadata: Metadata = {
  title: "Portfolio IA",
  description: "Portfolio interactif : parcours et assistant qui répond sur son contenu.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr">
      <body>
        {corpusEstFictif() && (
          <div
            role="note"
            style={{
              background: "#7a2e00",
              color: "#fff",
              padding: "0.75rem 1rem",
              fontSize: "0.95rem",
              lineHeight: 1.4,
            }}
          >
            <strong>Demonstration technique.</strong> Le parcours affiche et les
            reponses de l&apos;assistant portent sur une persona entierement
            fictive, generee pour illustrer la chaine RAG. Ce ne sont pas les
            informations reelles du proprietaire de ce domaine.
          </div>
        )}
        {children}
      </body>
    </html>
  );
}
