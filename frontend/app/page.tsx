import Link from "next/link";
import VoiceChat from "@/components/VoiceChat";

export default function Accueil() {
  return (
    <main>
      <nav>
        <Link href="/">Accueil</Link>
        <Link href="/parcours">Parcours</Link>
      </nav>
      <h1>Portfolio interactif</h1>
      <p>
        Parlez pour poser une question sur le parcours : la réponse est
        générée à partir du contenu de la page{" "}
        <Link href="/parcours">parcours</Link>, avec les sources citées.
      </p>
      <VoiceChat />
    </main>
  );
}
