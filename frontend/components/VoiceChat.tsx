"use client";

import { useEffect, useRef, useState } from "react";
import {
  CODE_WORKLET_CAPTURE,
  DecoupeurPCM,
  FREQUENCE_ECHANTILLONNAGE,
  LecteurAudioProgressif,
  analyserMessageVoix,
  type SourceCitee,
} from "../lib/voix";

// Contrat WebSocket de /voice (imposé, ne pas modifier — cf. CLAUDE.md) :
//   client -> serveur : trames BINAIRES PCM16 mono 24000 Hz, tranches de 80 ms
//                        {"type":"end"} (texte) quand le visiteur cesse de parler
//   serveur -> client : {"type":"transcript","text":"..."}   transcription en direct
//                        {"type":"token","text":"..."}        tokens du LLM
//                        trames BINAIRES PCM16 mono 24000 Hz  réponse vocale
//                        {"type":"sources","sources":[...]}   bloc terminal
//                        {"type":"error","message":"..."}

// NEXT_PUBLIC_* est inlinee au build : sans valeur, un build de production
// enverrait chaque visiteur vers sa propre machine. On echoue au build.
const API_URL =
  process.env.NEXT_PUBLIC_API_URL ??
  (process.env.NODE_ENV === "production"
    ? (() => {
        throw new Error("NEXT_PUBLIC_API_URL est obligatoire pour un build de production");
      })()
    : "http://localhost:8000");

const WS_URL = API_URL.replace(/^http/, "ws") + "/voice";

type Etat = "repos" | "ecoute" | "reflexion" | "reponse" | "refus-micro" | "erreur";

export default function VoiceChat() {
  const [etat, setEtat] = useState<Etat>("repos");
  const [transcript, setTranscript] = useState("");
  const [reponse, setReponse] = useState("");
  const [sources, setSources] = useState<SourceCitee[]>([]);
  const [erreur, setErreur] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const ctxCaptureRef = useRef<AudioContext | null>(null);
  const ctxLectureRef = useRef<AudioContext | null>(null);
  const lecteurRef = useRef<LecteurAudioProgressif | null>(null);
  const pistesRef = useRef<MediaStreamTrack[]>([]);
  const decoupeurRef = useRef(new DecoupeurPCM());
  // Garantie anti demi-réponse (même principe que l'ancien Chat.tsx) : sans
  // bloc terminal ("sources" ou "error") avant la fermeture du socket, on
  // doit le signaler plutôt que de laisser la dernière réponse affichée
  // passer pour complète.
  const termineRef = useRef(false);

  function couperMicro() {
    pistesRef.current.forEach((p) => p.stop());
    pistesRef.current = [];
    ctxCaptureRef.current?.close().catch(() => {});
    ctxCaptureRef.current = null;
  }

  function nettoyer() {
    wsRef.current?.close();
    wsRef.current = null;
    couperMicro();
    ctxLectureRef.current?.close().catch(() => {});
    ctxLectureRef.current = null;
    lecteurRef.current = null;
  }

  // Micro relâché, AudioContext fermé, WebSocket fermé au démontage : un
  // micro laissé ouvert est un problème de confiance sur un portfolio.
  useEffect(() => nettoyer, []);

  async function demarrer() {
    setErreur(null);
    setTranscript("");
    setReponse("");
    setSources([]);
    termineRef.current = false;
    decoupeurRef.current = new DecoupeurPCM();

    let flux: MediaStream;
    try {
      flux = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setEtat("refus-micro");
      return;
    }
    pistesRef.current = flux.getTracks();

    const ctxCapture = new AudioContext({ sampleRate: FREQUENCE_ECHANTILLONNAGE });
    ctxCaptureRef.current = ctxCapture;
    const ctxLecture = new AudioContext({ sampleRate: FREQUENCE_ECHANTILLONNAGE });
    ctxLectureRef.current = ctxLecture;
    lecteurRef.current = new LecteurAudioProgressif(ctxLecture);

    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => setEtat("ecoute");
    ws.onmessage = (evenement) => traiterMessage(evenement.data);
    ws.onerror = () => {
      termineRef.current = true; // panne réseau signalée explicitement, pas une coupure silencieuse
      setErreur("Assistant vocal indisponible pour le moment - la page parcours reste consultable.");
      setEtat("erreur");
    };
    ws.onclose = () => {
      if (!termineRef.current) {
        setErreur("Réponse interrompue - la page parcours reste consultable.");
        setEtat("erreur");
      }
    };

    const source = ctxCapture.createMediaStreamSource(flux);
    const urlWorklet = URL.createObjectURL(
      new Blob([CODE_WORKLET_CAPTURE], { type: "application/javascript" })
    );
    try {
      await ctxCapture.audioWorklet.addModule(urlWorklet);
      const noeud = new AudioWorkletNode(ctxCapture, "capture-pcm");
      noeud.port.onmessage = (e) => envoyerTranches(e.data as Float32Array);
      // Un noeud non relié à la destination n'est pas garanti d'être rendu :
      // on le relie via un gain à 0 pour rester silencieux côté visiteur.
      const collecteur = ctxCapture.createGain();
      collecteur.gain.value = 0;
      source.connect(noeud);
      noeud.connect(collecteur);
      collecteur.connect(ctxCapture.destination);
    } catch {
      // Repli pour les navigateurs sans AudioWorklet : ScriptProcessorNode
      // est déprécié mais reste universellement supporté.
      const processeur = ctxCapture.createScriptProcessor(4096, 1, 1);
      processeur.onaudioprocess = (e) => envoyerTranches(e.inputBuffer.getChannelData(0));
      source.connect(processeur);
      processeur.connect(ctxCapture.destination);
    } finally {
      URL.revokeObjectURL(urlWorklet);
    }
  }

  function envoyerTranches(bloc: Float32Array) {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    for (const tranche of decoupeurRef.current.decouper(bloc)) {
      ws.send(tranche);
    }
  }

  function terminerTour() {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "end" }));
    couperMicro();
    setEtat("reflexion");
  }

  function traiterMessage(donnee: string | ArrayBuffer) {
    const message = analyserMessageVoix(donnee);
    switch (message.type) {
      case "audio":
        lecteurRef.current?.jouer(message.trame);
        setEtat("reponse");
        break;
      case "transcript":
        // Le serveur émet des fragments successifs, pas le texte cumulé.
        setTranscript((t) => t + message.text);
        break;
      case "token":
        // Le serveur peut clore l'écoute seul (plafond de 30 s) : le micro ne
        // doit pas rester ouvert pendant la réponse.
        couperMicro();
        setEtat("reponse");
        setReponse((r) => r + message.text);
        break;
      case "sources":
        setSources(message.sources);
        termineRef.current = true;
        setEtat("repos");
        wsRef.current?.close();
        break;
      case "error":
        termineRef.current = true; // panne signalée explicitement par le serveur
        setErreur(message.message);
        setEtat("erreur");
        wsRef.current?.close();
        break;
      case "inconnu":
        break; // évolution du contrat non reconnue : ignorée, ne casse pas le flux
    }
  }

  function reessayer() {
    nettoyer();
    setEtat("repos");
    setErreur(null);
  }

  return (
    <div className="voice-chat">
      {etat === "refus-micro" && (
        <div className="voice-erreur">
          Micro refusé : impossible de démarrer la conversation vocale. La page{" "}
          <a href="/parcours">parcours</a> reste consultable.
          <button onClick={reessayer}>Réessayer</button>
        </div>
      )}

      {etat === "erreur" && (
        <div className="voice-erreur">
          {erreur}
          <button onClick={reessayer}>Réessayer</button>
        </div>
      )}

      {etat === "repos" && (
        <button className="voice-bouton" onClick={demarrer}>
          Parler
        </button>
      )}

      {etat === "ecoute" && (
        <button className="voice-bouton voice-bouton-actif" onClick={terminerTour}>
          Terminé de parler
        </button>
      )}

      {etat === "reflexion" && <p className="voice-statut">Réflexion en cours…</p>}
      {etat === "reponse" && <p className="voice-statut">Réponse en cours…</p>}

      <div className="voice-transcript" role="log" aria-live="polite">
        {transcript && <p className="voice-transcript-texte">« {transcript} »</p>}
        {reponse && <p className="voice-reponse">{reponse}</p>}
      </div>

      {sources.length > 0 && (
        <div className="voice-sources">
          Sources : {sources.map((s) => s.titre ?? s.source).join(", ")}
        </div>
      )}
    </div>
  );
}
