// Utilitaires du flux vocal /voice : conversion PCM, découpage en tranches,
// lecture progressive et interprétation des messages serveur. Séparés du
// composant pour rester testables sans microphone ni WebSocket réels.

export const FREQUENCE_ECHANTILLONNAGE = 24000;
export const ECHANTILLONS_PAR_TRANCHE = Math.round(FREQUENCE_ECHANTILLONNAGE * 0.08); // 80 ms

/** Convertit un buffer Float32 (Web Audio) en PCM 16 bits signé, little-endian. */
export function float32VersPcm16(entree: Float32Array): ArrayBuffer {
  const sortie = new Int16Array(entree.length);
  for (let i = 0; i < entree.length; i++) {
    const echantillon = Math.max(-1, Math.min(1, entree[i]));
    sortie[i] = echantillon < 0 ? echantillon * 0x8000 : echantillon * 0x7fff;
  }
  return sortie.buffer;
}

/** Convertit un buffer PCM 16 bits reçu du serveur en Float32 pour le Web Audio API. */
export function pcm16VersFloat32(entree: ArrayBuffer): Float32Array<ArrayBuffer> {
  const source = new Int16Array(entree);
  const sortie = new Float32Array(source.length);
  for (let i = 0; i < source.length; i++) {
    sortie[i] = source[i] / (source[i] < 0 ? 0x8000 : 0x7fff);
  }
  return sortie;
}

/**
 * Découpe un flux continu d'échantillons Float32 en tranches PCM16 de taille
 * fixe (80 ms = 1920 échantillons à 24 kHz). Le micro livre des blocs de
 * taille arbitraire (128 échantillons en AudioWorklet, 4096 en
 * ScriptProcessor) qui ne s'alignent jamais sur la tranche attendue par le
 * serveur : on accumule et on ne restitue que des tranches complètes, le
 * reliquat attend le prochain appel.
 */
export class DecoupeurPCM {
  private reliquat = new Float32Array(0);

  decouper(bloc: Float32Array): ArrayBuffer[] {
    const fusion = new Float32Array(this.reliquat.length + bloc.length);
    fusion.set(this.reliquat);
    fusion.set(bloc, this.reliquat.length);

    const tranches: ArrayBuffer[] = [];
    let offset = 0;
    while (offset + ECHANTILLONS_PAR_TRANCHE <= fusion.length) {
      tranches.push(float32VersPcm16(fusion.subarray(offset, offset + ECHANTILLONS_PAR_TRANCHE)));
      offset += ECHANTILLONS_PAR_TRANCHE;
    }
    this.reliquat = fusion.subarray(offset);
    return tranches;
  }
}

/**
 * Ramène un flux micro de la fréquence matérielle (44,1 ou 48 kHz) à 24 kHz.
 * Imposer 24 kHz à l'AudioContext ne marche pas partout : Safari iOS garde la
 * fréquence matérielle pour le micro, Firefox refuse de relier deux fréquences.
 * Interpolation linéaire, avec la position fractionnaire et le dernier
 * échantillon conservés entre deux blocs pour ne pas créer de rupture.
 */
export class Reechantillonneur {
  private readonly pas: number;
  private position = 0;
  private precedent = 0;

  constructor(frequenceSource: number) {
    this.pas = frequenceSource / FREQUENCE_ECHANTILLONNAGE;
  }

  convertir(bloc: Float32Array): Float32Array {
    if (this.pas === 1) return bloc;
    const sortie: number[] = [];
    // Tableau virtuel : l'indice -1 désigne le dernier échantillon du bloc précédent.
    while (this.position < bloc.length - 1) {
      const i = Math.floor(this.position);
      const fraction = this.position - i;
      const a = i < 0 ? this.precedent : bloc[i];
      sortie.push(a + (bloc[i + 1] - a) * fraction);
      this.position += this.pas;
    }
    this.position -= bloc.length;
    if (bloc.length) this.precedent = bloc[bloc.length - 1];
    return Float32Array.from(sortie);
  }
}

// Code de l'AudioWorkletProcessor de capture, injecté via Blob URL : évite de
// servir un fichier statique séparé dans un export Next.js ("output: export").
// Il ne fait que relayer les blocs bruts (128 échantillons) au thread
// principal, où DecoupeurPCM les regroupe en tranches de 80 ms.
export const CODE_WORKLET_CAPTURE = `
class ProcesseurCapture extends AudioWorkletProcessor {
  process(entrees) {
    const canal = entrees[0]?.[0];
    if (canal) this.port.postMessage(canal.slice(0));
    return true;
  }
}
registerProcessor("capture-pcm", ProcesseurCapture);
`;

/**
 * Lit une suite de trames PCM16 reçues au fil de l'eau, sans attendre la fin
 * de la réponse : chaque trame est planifiée juste après la précédente sur
 * un AudioContext dédié à la lecture (distinct de celui de capture).
 */
export class LecteurAudioProgressif {
  private prochainDebut = 0;

  // `sortie` : noeud de destination, un AnalyserNode pour visualiser la voix.
  constructor(
    private contexte: AudioContext,
    private sortie: AudioNode = contexte.destination
  ) {}

  /** Secondes d'audio déjà reçu qui restent à jouer. */
  resteAJouer(): number {
    return Math.max(0, this.prochainDebut - this.contexte.currentTime);
  }

  jouer(trame: ArrayBuffer) {
    const echantillons = pcm16VersFloat32(trame);
    const buffer = this.contexte.createBuffer(1, echantillons.length, FREQUENCE_ECHANTILLONNAGE);
    buffer.copyToChannel(echantillons, 0);

    const source = this.contexte.createBufferSource();
    source.buffer = buffer;
    source.connect(this.sortie);

    const debut = Math.max(this.contexte.currentTime, this.prochainDebut);
    source.start(debut);
    this.prochainDebut = debut + buffer.duration;
  }
}

export interface SourceCitee {
  source: string;
  titre?: string;
  score: number;
}

export type MessageVoix =
  | { type: "transcript"; text: string }
  | { type: "token"; text: string }
  | { type: "sources"; sources: SourceCitee[] }
  | { type: "error"; message: string }
  | { type: "audio"; trame: ArrayBuffer }
  | { type: "inconnu" };

/**
 * Interprète un message reçu de /voice. Isolé du composant pour rester
 * testable sans mock de WebSocket ni d'AudioContext. Tout ce qui ne
 * correspond pas exactement au contrat (type absent, champ du mauvais type,
 * JSON invalide, type non reconnu) retombe sur "inconnu" et doit être ignoré
 * par l'appelant plutôt que de casser le flux.
 */
export function analyserMessageVoix(donnee: string | ArrayBuffer): MessageVoix {
  if (donnee instanceof ArrayBuffer) return { type: "audio", trame: donnee };

  try {
    const message = JSON.parse(donnee);
    if (message?.type === "transcript" && typeof message.text === "string") {
      return { type: "transcript", text: message.text };
    }
    if (message?.type === "token" && typeof message.text === "string") {
      return { type: "token", text: message.text };
    }
    if (message?.type === "sources" && Array.isArray(message.sources)) {
      return { type: "sources", sources: message.sources };
    }
    if (message?.type === "error" && typeof message.message === "string") {
      return { type: "error", message: message.message };
    }
  } catch {
    // JSON invalide : traité comme un type inconnu ci-dessous
  }
  return { type: "inconnu" };
}
