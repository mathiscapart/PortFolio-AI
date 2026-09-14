import type { Metadata } from "next";
import Entete from "@/components/Entete";
import { lireCorpus } from "@/lib/corpus";
import { NOM } from "@/lib/site";

export const metadata: Metadata = {
  title: "Parcours, compétences et projets",
  description: `Expériences, formation, compétences et projets de ${NOM} : IA, DevSecOps, infrastructure et cybersécurité.`,
  alternates: { canonical: "/parcours/" },
};

// Rendu au build, à partir du Markdown du corpus RAG : la page reste lisible
// même backend éteint, et indexable puisque générée en statique.
export default function Parcours() {
  const sections = lireCorpus();

  return (
    <main className="page">
      <Entete lienInterne={{ href: "/", libelle: "Poser une question à voix haute" }} />
      <div className="parcours">
        {sections.map((section) => (
          <article
            key={section.source}
            className="corpus-section"
            dangerouslySetInnerHTML={{ __html: section.html }}
          />
        ))}
      </div>
    </main>
  );
}
