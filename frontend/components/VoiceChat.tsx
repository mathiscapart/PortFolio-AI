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

type Etat = "repos" | "connexion" | "ecoute" | "reflexion" | "reponse" | "refus-micro" | "erreur";

interface Tour {
  id: number;
  question: string;
  reponse: string;
  sources: SourceCitee[];
}

const EXEMPLES = [
  "Quel est votre parcours ?",
  "Quelles technologies utilisez-vous ?",
  "Quel poste cherchez-vous ?",
];

const STATUTS: Record<Etat, string> = {
  repos: "Appuyez sur Parler ou sur Espace, posez votre question, puis terminez.",
  connexion: "Connexion à l'assistant…",
  ecoute: "Je vous écoute.",
  reflexion: "Je cherche dans le parcours…",
  reponse: "Je réponds.",
  "refus-micro": "",
  erreur: "",
};

export default function VoiceChat() {
  const [etat, setEtat] = useState<Etat>("repos");
  const [tours, setTours] = useState<Tour[]>([]);
  const [erreur, setErreur] = useState<string | null>(null);
  const [analyseur, setAnalyseur] = useState<AnalyserNode | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const ctxCaptureRef = useRef<AudioContext | null>(null);
  const ctxLectureRef = useRef<AudioContext | null>(null);
  const lecteurRef = useRef<LecteurAudioProgressif | null>(null);
  const analyseurLectureRef = useRef<AnalyserNode | null>(null);
  const pistesRef = useRef<MediaStreamTrack[]>([]);
  const decoupeurRef = useRef(new DecoupeurPCM());
  const finLectureRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Garantie anti demi-réponse : sans bloc terminal ("sources" ou "error")
  // avant la fermeture du socket, on le signale plutôt que de laisser la
  // dernière réponse affichée passer pour complète.
  const termineRef = useRef(false);

  function couperMicro() {
    pistesRef.current.forEach((p) => p.stop());
    pistesRef.current = [];
    ctxCaptureRef.current?.close().catch(() => {});
    ctxCaptureRef.current = null;
  }

  function nettoyer() {
    if (finLectureRef.current) clearTimeout(finLectureRef.current);
    wsRef.current?.close();
    wsRef.current = null;
    couperMicro();
    ctxLectureRef.current?.close().catch(() => {});
    ctxLectureRef.current = null;
    lecteurRef.current = null;
    setAnalyseur(null);
  }

  // Micro relâché, AudioContext fermé, WebSocket fermé au démontage : un
  // micro laissé ouvert est un problème de confiance sur un portfolio.
  useEffect(() => nettoyer, []);

  function majTourCourant(maj: (t: Tour) => Tour) {
    setTours((liste) => (liste.length ? [maj(liste[0]), ...liste.slice(1)] : liste));
  }

  async function demarrer() {
    nettoyer();
    setErreur(null);
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
    setEtat("connexion");
    setTours((liste) => [{ id: Date.now(), question: "", reponse: "", sources: [] }, ...liste]);

    const ctxCapture = new AudioContext({ sampleRate: FREQUENCE_ECHANTILLONNAGE });
    ctxCaptureRef.current = ctxCapture;
    const ctxLecture = new AudioContext({ sampleRate: FREQUENCE_ECHANTILLONNAGE });
    ctxLectureRef.current = ctxLecture;
    const analyseurLecture = ctxLecture.createAnalyser();
    analyseurLecture.connect(ctxLecture.destination);
    analyseurLectureRef.current = analyseurLecture;
    lecteurRef.current = new LecteurAudioProgressif(ctxLecture, analyseurLecture);

    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => setEtat("ecoute");
    ws.onmessage = (evenement) => traiterMessage(evenement.data);
    ws.onerror = () => {
      termineRef.current = true; // panne réseau signalée explicitement, pas une coupure silencieuse
      echouer("L'assistant vocal ne répond pas pour le moment. Réessayez dans un instant ou lisez le parcours.");
    };
    ws.onclose = () => {
      if (!termineRef.current) {
        echouer("Réponse interrompue : la connexion a été coupée. Réessayez ou lisez le parcours.");
      }
    };

    const source = ctxCapture.createMediaStreamSource(flux);
    const analyseurMicro = ctxCapture.createAnalyser();
    source.connect(analyseurMicro);
    setAnalyseur(analyseurMicro);

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

  function echouer(message: string) {
    setErreur(message);
    setEtat("erreur");
    couperMicro();
    setAnalyseur(null);
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
    setAnalyseur(null);
    setEtat("reflexion");
  }

  /** Coupe la réponse en cours (voix comprise) à la demande du visiteur. */
  function arreter() {
    termineRef.current = true;
    nettoyer();
    setEtat("repos");
  }

  function passerALaVoix() {
    // Le serveur peut clore l'écoute seul (plafond de 30 s) : le micro ne
    // doit pas rester ouvert pendant la réponse.
    if (pistesRef.current.length) couperMicro();
    setAnalyseur(analyseurLectureRef.current);
    setEtat("reponse");
  }

  function traiterMessage(donnee: string | ArrayBuffer) {
    const message = analyserMessageVoix(donnee);
    switch (message.type) {
      case "audio":
        lecteurRef.current?.jouer(message.trame);
        passerALaVoix();
        break;
      case "transcript":
        // Le serveur émet des fragments successifs, pas le texte cumulé.
        majTourCourant((t) => ({ ...t, question: t.question + message.text }));
        break;
      case "token":
        passerALaVoix();
        majTourCourant((t) => ({ ...t, reponse: t.reponse + message.text }));
        break;
      case "sources": {
        termineRef.current = true;
        majTourCourant((t) => ({ ...t, sources: message.sources }));
        wsRef.current?.close();
        // Le texte est complet, mais la voix peut encore parler : on ne rend
        // la main qu'une fois l'audio déjà reçu entièrement joué.
        const reste = lecteurRef.current?.resteAJouer() ?? 0;
        finLectureRef.current = setTimeout(() => {
          setAnalyseur(null);
          setEtat("repos");
        }, reste * 1000);
        break;
      }
      case "error":
        termineRef.current = true; // panne signalée explicitement par le serveur
        echouer(message.message);
        wsRef.current?.close();
        break;
      case "inconnu":
        break; // évolution du contrat non reconnue : ignorée, ne casse pas le flux
    }
  }

  // Espace pour parler / terminer / arrêter, sauf quand un bouton a le focus :
  // il réagit déjà nativement à Espace.
  const actionRef = useRef<() => void>(() => {});
  useEffect(() => {
    function surTouche(e: KeyboardEvent) {
      if (e.code !== "Space" || e.repeat) return;
      const cible = e.target as HTMLElement | null;
      if (cible?.closest("button, a, input, textarea")) return;
      e.preventDefault();
      actionRef.current();
    }
    window.addEventListener("keydown", surTouche);
    return () => window.removeEventListener("keydown", surTouche);
  }, []);

  let bouton: { libelle: string; action: () => void; variante: string; desactive?: boolean };
  if (etat === "ecoute") bouton = { libelle: "Terminé de parler", action: terminerTour, variante: "visiteur" };
  else if (etat === "connexion") bouton = { libelle: "Connexion…", action: () => {}, variante: "visiteur", desactive: true };
  else if (etat === "reflexion" || etat === "reponse") bouton = { libelle: "Arrêter", action: arreter, variante: "assistant" };
  else bouton = { libelle: etat === "repos" ? "Parler" : "Réessayer de parler", action: demarrer, variante: "repos" };
  actionRef.current = bouton.desactive ? () => {} : bouton.action;

  const voix = etat === "reponse" || etat === "reflexion" ? "assistant" : "visiteur";

  return (
    <section className="voix" data-etat={etat} aria-label="Assistant vocal">
      <div className="scene">
        <Onde analyseur={analyseur} voix={voix} reflexion={etat === "reflexion"} />
        <div className="commandes">
          <button
            type="button"
            className={`bouton-voix bouton-${bouton.variante}`}
            onClick={bouton.action}
            disabled={bouton.desactive}
          >
            <span className="pastille" aria-hidden="true" />
            {bouton.libelle}
          </button>
          <p className="statut" aria-live="polite">
            {STATUTS[etat]}
          </p>
        </div>

        {etat === "refus-micro" && (
          <p className="alerte" role="alert">
            Le micro est bloqué. Autorisez-le depuis l&apos;icône à gauche de la barre
            d&apos;adresse, puis réessayez. Le <a href="/parcours/">parcours écrit</a> reste
            disponible.
          </p>
        )}
        {etat === "erreur" && erreur && (
          <p className="alerte" role="alert">
            {erreur}
          </p>
        )}
      </div>

      <div className="echanges" role="log" aria-live="polite">
        {tours.length === 0 ? (
          <div className="exemples">
            <p>Par exemple :</p>
            <ul>
              {EXEMPLES.map((q) => (
                <li key={q}>« {q} »</li>
              ))}
            </ul>
          </div>
        ) : (
          tours.map((tour) => (
            <article key={tour.id} className="tour">
              <p className="tour-question">
                {tour.question ? `« ${tour.question.trim()} »` : "…"}
              </p>
              {tour.reponse && <p className="tour-reponse">{tour.reponse}</p>}
              {tour.sources.length > 0 && (
                <p className="tour-sources">
                  {/* Les 8 extraits vont au LLM, mais seuls les mieux classés
                      (triés par score côté serveur) informent le visiteur. */}
                  D&apos;après : {[...new Set(tour.sources.map((s) => s.titre ?? s.source))].slice(0, 3).join(", ")}
                </p>
              )}
            </article>
          ))
        )}
      </div>
    </section>
  );
}

/**
 * Onde du signal réel : le micro pendant l'écoute, la voix de l'assistant
 * pendant la réponse. Au repos, une ligne plate ; en réflexion, une
 * respiration lente (sauf mouvement réduit).
 */
function Onde({
  analyseur,
  voix,
  reflexion,
}: {
  analyseur: AnalyserNode | null;
  voix: "visiteur" | "assistant";
  reflexion: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext?.("2d");
    if (!canvas || !ctx) return;
    const mouvementReduit = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const donnees = analyseur ? new Float32Array(analyseur.fftSize) : null;
    let image = 0;
    // Gain automatique lissé : une voix réelle plafonne vers 0,1 d'amplitude,
    // l'onde resterait plate sans normalisation.
    let pic = 0.05;

    function dessiner(t: number) {
      const ratio = window.devicePixelRatio || 1;
      const { clientWidth: l, clientHeight: h } = canvas!;
      if (canvas!.width !== l * ratio) {
        canvas!.width = l * ratio;
        canvas!.height = h * ratio;
      }
      ctx!.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx!.clearRect(0, 0, l, h);
      const style = getComputedStyle(canvas!);
      ctx!.strokeStyle = style.getPropertyValue(`--couleur-${voix}`).trim() || "currentColor";
      ctx!.lineWidth = 2.5;
      ctx!.lineCap = "round";
      ctx!.lineJoin = "round";
      ctx!.beginPath();

      const points = 96;
      for (let i = 0; i <= points; i++) {
        const x = (i / points) * l;
        // Enveloppe : l'onde s'éteint vers les bords, comme un fil tendu.
        const enveloppe = Math.sin((i / points) * Math.PI);
        let y = 0;
        if (analyseur && donnees) {
          y = (donnees[Math.floor((i / points) * (donnees.length - 1))] / pic) * 0.85;
        } else if (reflexion && !mouvementReduit) {
          y = Math.sin(i / 6 + t / 400) * 0.22 * (0.6 + 0.4 * Math.sin(t / 700));
        }
        const py = h / 2 + Math.max(-1, Math.min(1, y)) * enveloppe * (h / 2 - 4);
        if (i === 0) ctx!.moveTo(x, py);
        else ctx!.lineTo(x, py);
      }
      ctx!.stroke();
    }

    function boucle(t: number) {
      if (analyseur && donnees) {
        analyseur.getFloatTimeDomainData(donnees);
        let max = 0;
        for (const v of donnees) max = Math.max(max, Math.abs(v));
        // Monte vite, redescend lentement ; plancher pour ne pas amplifier le souffle.
        pic = Math.max(0.04, max > pic ? max : pic * 0.97 + max * 0.03);
      }
      dessiner(t);
      if (analyseur || (reflexion && !mouvementReduit)) image = requestAnimationFrame(boucle);
    }
    image = requestAnimationFrame(boucle);
    return () => cancelAnimationFrame(image);
  }, [analyseur, voix, reflexion]);

  return <canvas ref={canvasRef} className="onde" aria-hidden="true" />;
}
