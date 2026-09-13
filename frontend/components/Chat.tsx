"use client";

import { useState } from "react";

// Contrat SSE de POST /chat (vérifié en direct, cf. backend/api/main.py) :
//   : ping                          -> commentaire SSE, à ignorer
//   data: {"token": "..."}          -> un bloc par token, sans champ "event"
//   event: sources / data: {...}    -> bloc terminal
//   event: error / data: {...}      -> panne mi-flux, pas de "sources" après
//
// EventSource ne fait que du GET ; /chat est un POST, donc on parse le flux
// nous-mêmes via fetch + ReadableStream plutôt que d'utiliser EventSource.

// NEXT_PUBLIC_* est inlinee au build : sans valeur, un build de production
// enverrait chaque visiteur vers sa propre machine. On echoue au build.
const API_URL =
  process.env.NEXT_PUBLIC_API_URL ??
  (process.env.NODE_ENV === "production"
    ? (() => {
        throw new Error("NEXT_PUBLIC_API_URL est obligatoire pour un build de production");
      })()
    : "http://localhost:8000");

interface Source {
  source: string;
  titre?: string;
  score: number;
}

interface Message {
  role: "utilisateur" | "assistant";
  texte: string;
  sources?: Source[];
  erreur?: string;
}

export default function Chat() {
  const [question, setQuestion] = useState("");
  const [historique, setHistorique] = useState<Message[]>([]);
  const [enCours, setEnCours] = useState(false);

  async function envoyer(e: React.FormEvent) {
    e.preventDefault();
    const message = question.trim();
    if (!message || enCours) return;

    setQuestion("");
    setHistorique((h) => [...h, { role: "utilisateur", texte: message }]);
    setEnCours(true);

    const indexReponse = historique.length + 1;
    setHistorique((h) => [...h, { role: "assistant", texte: "" }]);

    let lecteur: ReadableStreamDefaultReader<Uint8Array> | undefined;
    try {
      const reponse = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, k: 5 }),
      });
      if (!reponse.ok || !reponse.body) {
        throw new Error(`requête refusée (${reponse.status})`);
      }

      lecteur = reponse.body.getReader();
      const decodeur = new TextDecoder();
      let tampon = "";
      let termine = false;

      for (;;) {
        const { done, value } = await lecteur.read();
        if (done) {
          // Un dernier bloc peut rester si le flux finit par un simple saut de ligne.
          if (tampon.trim()) termine = traiterBloc(tampon, indexReponse) || termine;
          break;
        }
        tampon += decodeur.decode(value, { stream: true });

        // Un bloc SSE est délimité par une ligne vide.
        let indexSeparateur;
        while ((indexSeparateur = tampon.indexOf("\n\n")) !== -1) {
          const bloc = tampon.slice(0, indexSeparateur);
          tampon = tampon.slice(indexSeparateur + 2);
          termine = traiterBloc(bloc, indexReponse) || termine;
        }
      }

      // Ni `sources` ni `error` : flux coupe proprement (Ollama tue, proxy qui
      // timeout). Sans ca, une demi-reponse s'affiche comme si elle etait
      // complete -- le pire mode de defaillance pour un portfolio.
      if (!termine) {
        setHistorique((h) =>
          h.map((m, i) => (i === indexReponse ? { ...m, erreur: "reponse interrompue" } : m))
        );
      }
    } catch (err) {
      console.error(err);
      const messageErreur =
        "Assistant indisponible pour le moment - la page parcours reste consultable.";
      setHistorique((h) =>
        h.map((m, i) => (i === indexReponse ? { ...m, erreur: messageErreur } : m))
      );
    } finally {
      // Sans annulation, la generation Ollama continue dans le vide : sur une
      // seule GPU, quelques flux orphelins saturent la file.
      await lecteur?.cancel().catch(() => {});
      setEnCours(false);
    }
  }

  function traiterBloc(bloc: string, index: number): boolean {
    let evenement: string | undefined;
    let donnee: string | undefined;

    for (const ligne of bloc.split("\n")) {
      if (ligne.startsWith(":")) continue; // commentaire SSE (ex: ": ping")
      if (ligne.startsWith("event:")) evenement = ligne.slice("event:".length).trim();
      if (ligne.startsWith("data:")) donnee = ligne.slice("data:".length).trim();
    }
    if (!donnee) return false;

    if (evenement === "sources") {
      const { sources } = JSON.parse(donnee) as { sources: Source[] };
      setHistorique((h) => h.map((m, i) => (i === index ? { ...m, sources } : m)));
      return true;
    }
    if (evenement === "error") {
      const { error } = JSON.parse(donnee) as { error: string };
      setHistorique((h) => h.map((m, i) => (i === index ? { ...m, erreur: error } : m)));
      return true;
    }
    const { token } = JSON.parse(donnee) as { token: string };
    setHistorique((h) => h.map((m, i) => (i === index ? { ...m, texte: m.texte + token } : m)));
    return false;
  }

  return (
    <div className="chat">
      <div className="chat-historique" role="log" aria-live="polite">
        {historique.map((m, i) => (
          <div key={i} className={`chat-message ${m.role}`}>
            {m.texte || (enCours && m.role === "assistant" && i === historique.length - 1 ? "..." : "")}
            {m.erreur && <div className="chat-erreur">{m.erreur}</div>}
            {m.sources && m.sources.length > 0 && (
              <div className="chat-sources">
                Sources : {m.sources.map((s) => s.titre ?? s.source).join(", ")}
              </div>
            )}
          </div>
        ))}
      </div>
      <form className="chat-formulaire" onSubmit={envoyer}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Poser une question sur le parcours"
          disabled={enCours}
        />
        <button type="submit" disabled={enCours}>
          Envoyer
        </button>
      </form>
    </div>
  );
}
