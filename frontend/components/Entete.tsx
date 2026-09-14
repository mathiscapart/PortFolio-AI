import Link from "next/link";
import { LIENS, NOM } from "@/lib/site";

// En-tête commun : le nom en marque (signal principal pour une recherche sur
// le nom), la navigation et les profils externes. rel="me" relie le site aux
// profils ; noopener car ils s'ouvrent dans un nouvel onglet.
export default function Entete({ lienInterne }: { lienInterne: { href: string; libelle: string } }) {
  return (
    <header className="entete">
      <Link href="/" className="marque">
        {NOM}
      </Link>
      <nav aria-label="Navigation principale">
        <Link href={lienInterne.href}>{lienInterne.libelle}</Link>
        <a href={LIENS.linkedin} rel="me noopener" target="_blank">
          LinkedIn
        </a>
        <a href={LIENS.github} rel="me noopener" target="_blank">
          GitHub
        </a>
      </nav>
    </header>
  );
}
