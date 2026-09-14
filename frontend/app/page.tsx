import type { Metadata } from "next";
import Link from "next/link";
import Entete from "@/components/Entete";
import VoiceChat from "@/components/VoiceChat";
import { POSTE } from "@/lib/site";

export const metadata: Metadata = {
  alternates: { canonical: "/" },
};

export default function Accueil() {
  return (
    <main className="page">
      <Entete lienInterne={{ href: "/parcours/", libelle: "Lire le parcours" }} />

      <div className="accroche">
        <p className="identite">{POSTE}</p>
        <h1>Posez-moi une question sur mon parcours, à voix haute.</h1>
        <p>
          En alternance chez Doublet et en cinquième année à SUPINFO, je
          travaille à la croisée de l&apos;intelligence
          artificielle, de la cybersécurité et de l&apos;infrastructure.
          L&apos;assistant vous répond à l&apos;oral, uniquement à partir du{" "}
          <Link href="/parcours/">parcours écrit</Link>, et indique d&apos;où vient
          chaque réponse.
        </p>
      </div>

      <VoiceChat />
    </main>
  );
}
