import { describe, expect, it } from "vitest";

import {
  DecoupeurPCM,
  ECHANTILLONS_PAR_TRANCHE,
  analyserMessageVoix,
  float32VersPcm16,
  pcm16VersFloat32,
} from "../lib/voix";

// Remplace tests/chat-sse.test.tsx (Chat.tsx a été retiré au profit de
// VoiceChat.tsx, cf. décision produit vocal seul) : ces tests couvrent la
// logique pure du nouveau composant — conversion PCM, découpage en tranches
// de 80 ms et interprétation des messages /voice — plutôt qu'un parseur SSE
// qui n'existe plus.

describe("conversion PCM16 <-> Float32", () => {
  it("fait un aller-retour sans perte significative", () => {
    const original = new Float32Array([0, 0.5, -0.5, 1, -1]);
    const reconverti = pcm16VersFloat32(float32VersPcm16(original));
    for (let i = 0; i < original.length; i++) {
      expect(reconverti[i]).toBeCloseTo(original[i], 3);
    }
  });
});

describe("DecoupeurPCM", () => {
  it("ne restitue que des tranches complètes de 80 ms", () => {
    const decoupeur = new DecoupeurPCM();
    const bloc = new Float32Array(128).fill(0.1); // taille d'un quantum AudioWorklet
    const tranches = decoupeur.decouper(bloc);
    expect(tranches).toHaveLength(0); // 128 < 1920, tout part au reliquat
  });

  it("accumule plusieurs blocs jusqu'à former une tranche exploitable", () => {
    const decoupeur = new DecoupeurPCM();
    let total = 0;
    for (let i = 0; i < 20; i++) {
      const tranches = decoupeur.decouper(new Float32Array(128).fill(0.1));
      total += tranches.length;
    }
    // 20 * 128 = 2560 échantillons > 1920 : au moins une tranche complète produite
    expect(total).toBeGreaterThanOrEqual(1);
  });

  it("produit des tranches de la taille attendue par le serveur (80 ms à 24 kHz)", () => {
    const decoupeur = new DecoupeurPCM();
    const [tranche] = decoupeur.decouper(new Float32Array(ECHANTILLONS_PAR_TRANCHE).fill(0.1));
    expect(tranche.byteLength).toBe(ECHANTILLONS_PAR_TRANCHE * 2); // Int16 = 2 octets
  });
});

describe("analyserMessageVoix", () => {
  it("reconnaît une trame binaire comme de l'audio", () => {
    const trame = new ArrayBuffer(4);
    expect(analyserMessageVoix(trame)).toEqual({ type: "audio", trame });
  });

  it("reconnaît transcript, token, sources et error", () => {
    expect(analyserMessageVoix('{"type":"transcript","text":"bonjour"}')).toEqual({
      type: "transcript",
      text: "bonjour",
    });
    expect(analyserMessageVoix('{"type":"token","text":"Bon"}')).toEqual({
      type: "token",
      text: "Bon",
    });
    expect(
      analyserMessageVoix('{"type":"sources","sources":[{"source":"a.md","score":0.9}]}')
    ).toEqual({ type: "sources", sources: [{ source: "a.md", score: 0.9 }] });
    expect(analyserMessageVoix('{"type":"error","message":"panne"}')).toEqual({
      type: "error",
      message: "panne",
    });
  });

  it("ignore un type inconnu sans planter", () => {
    expect(analyserMessageVoix('{"type":"ping"}')).toEqual({ type: "inconnu" });
  });

  it("ignore un JSON invalide sans planter", () => {
    expect(analyserMessageVoix("pas du json")).toEqual({ type: "inconnu" });
  });

  it("ignore un message avec un champ du mauvais type", () => {
    expect(analyserMessageVoix('{"type":"token","text":42}')).toEqual({ type: "inconnu" });
  });
});
