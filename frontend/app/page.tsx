import Link from "next/link";
import VoiceChat from "@/components/VoiceChat";

export default function Accueil() {
  return (
    <main className="page">
      <header className="entete">
        <Link href="/" className="marque">
          Portfolio
        </Link>
        <nav>
          <Link href="/parcours/">Lire le parcours</Link>
        </nav>
      </header>

      <div className="accroche">
        <h1>Posez-moi une question sur mon parcours, à voix haute.</h1>
        <p>
          L&apos;assistant vous répond à l&apos;oral, uniquement à partir
          du <Link href="/parcours/">parcours écrit</Link>, et indique d&apos;où
          vient chaque réponse.
        </p>
      </div>

      <VoiceChat />
    </main>
  );
}
