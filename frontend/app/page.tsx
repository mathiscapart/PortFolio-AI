import Link from "next/link";
import Chat from "@/components/Chat";

export default function Accueil() {
  return (
    <main>
      <nav>
        <Link href="/">Accueil</Link>
        <Link href="/parcours">Parcours</Link>
      </nav>
      <h1>Portfolio interactif</h1>
      <p>
        Posez une question sur le parcours ci-dessous : les réponses sont
        générées à partir du contenu de la page{" "}
        <Link href="/parcours">parcours</Link>, avec les sources citées.
      </p>
      <Chat />
    </main>
  );
}
