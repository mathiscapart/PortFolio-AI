import Link from "next/link";
import { lireCorpus } from "@/lib/corpus";

// Rendu au build, à partir du Markdown du corpus RAG (jamais d'identité
// écrite en dur ici) : la page reste lisible même backend éteint, et
// indexable puisque générée en statique.
export default function Parcours() {
  const sections = lireCorpus();

  return (
    <main className="page">
      <header className="entete">
        <Link href="/" className="marque">
          Portfolio
        </Link>
        <nav>
          <Link href="/">Poser une question à voix haute</Link>
        </nav>
      </header>
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
