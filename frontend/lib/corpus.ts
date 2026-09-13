import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { marked } from "marked";

// Le contenu de la page parcours vient du corpus Markdown du RAG, jamais
// d'une identité écrite en dur ici. Le vrai corpus (`backend/rag/corpus/`)
// contient encore des marqueurs "À REMPLIR" (cf. CLAUDE.md) : tant qu'il
// n'est pas rédigé, CORPUS_DIR pointe par défaut sur le corpus de démo
// (persona fictive "Camille Verne"). À rebasculer sur le vrai corpus via la
// variable d'environnement une fois `backend/rag/corpus/*.md` rédigé.
const CORPUS_DIR = process.env.CORPUS_DIR
  ? path.resolve(process.cwd(), process.env.CORPUS_DIR)
  : path.resolve(process.cwd(), "..", "backend", "rag", "corpus_demo");

// Derive du MEME chemin que celui reellement servi : la banniere ne peut
// donc pas mentir ni deriver. Elle disparait d'elle-meme des que CORPUS_DIR
// pointe sur le vrai corpus.
export function corpusEstFictif(): boolean {
  return CORPUS_DIR.endsWith("corpus_demo");
}

export interface SectionCorpus {
  titre: string;
  source: string;
  html: string;
}

export function lireCorpus(): SectionCorpus[] {
  const fichiers = fs
    .readdirSync(CORPUS_DIR)
    .filter((nom) => nom.endsWith(".md") && nom.toLowerCase() !== "readme.md")
    .sort();

  return fichiers.map((nom) => {
    const brut = fs.readFileSync(path.join(CORPUS_DIR, nom), "utf-8");
    const { data, content } = matter(brut);
    return {
      titre: typeof data.titre === "string" ? data.titre : nom,
      source: typeof data.source === "string" ? data.source : nom,
      html: marked.parse(content, { async: false }) as string,
    };
  });
}
