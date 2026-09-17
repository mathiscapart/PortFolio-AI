import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import VoiceChat from "../components/VoiceChat";

// Le composant vocal dépend de getUserMedia/AudioContext/WebSocket : jsdom ne
// les fournit pas, donc ce test se limite au rendu initial (avant toute
// interaction), qui n'appelle aucune de ces API. Le contrat WebSocket est
// couvert séparément par tests/voix.test.ts (analyserMessageVoix).
describe("VoiceChat", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("affiche le bouton d'entrée et une zone de transcription accessible au repos", () => {
    render(<VoiceChat />);
    expect(screen.getByRole("button", { name: /Parler/i })).toBeTruthy();
    const zone = screen.getByRole("log");
    expect(zone.getAttribute("aria-live")).toBe("polite");
  });

  // Le GPU vit sur un PC allumé à la demande : le front, lui, reste en ligne.
  it("annonce l'IA hors ligne et renvoie vers LinkedIn quand /health échoue", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    render(<VoiceChat />);

    expect(await screen.findByText(/assistant vocal est hors ligne/i)).toBeTruthy();
    const lien = screen.getByRole("link", { name: /LinkedIn/i });
    expect(lien.getAttribute("href")).toBe("https://www.linkedin.com/in/mathis-capart/");
    expect((screen.getByRole("button", { name: /Parler/i }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("considère l'IA hors ligne quand /health répond en erreur (502 du proxy)", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 502 }));
    render(<VoiceChat />);

    expect(await screen.findByText(/assistant vocal est hors ligne/i)).toBeTruthy();
  });

  it("laisse Parler actif quand /health répond", async () => {
    render(<VoiceChat />);

    await waitFor(() => expect(fetch).toHaveBeenCalledWith(expect.stringMatching(/\/health$/), expect.anything()));
    expect(screen.queryByText(/hors ligne/i)).toBeNull();
    expect((screen.getByRole("button", { name: /Parler/i }) as HTMLButtonElement).disabled).toBe(false);
  });
});

// --- Doublures des API navigateur absentes de jsdom (micro, audio, socket) ---
// L'objectif n'est pas de tester la capture micro mais de pouvoir déclencher
// demarrer() jusqu'à l'ouverture du WebSocket, seul canal qui porte la
// garantie anti demi-réponse.

const noeud = () => ({ connect: () => {}, disconnect: () => {} });

class FakeAudioContext {
  static instances = 0;
  constructor() {
    FakeAudioContext.instances++;
  }
  destination = {};
  currentTime = 0;
  sampleRate = 48000;
  resume() {
    return Promise.resolve();
  }
  suspend() {
    return Promise.resolve();
  }
  audioWorklet = {
    addModule: () => Promise.reject(new Error("AudioWorklet indisponible sous jsdom")),
  };
  createMediaStreamSource() {
    return noeud();
  }
  createAnalyser() {
    return { ...noeud(), fftSize: 2048, getFloatTimeDomainData: () => {} };
  }
  createGain() {
    return { ...noeud(), gain: { value: 0 } };
  }
  createScriptProcessor() {
    return { ...noeud(), onaudioprocess: null };
  }
  createBufferSource() {
    return { connect: () => {}, start: () => {} };
  }
  createBuffer() {
    return {};
  }
  close() {
    return Promise.resolve();
  }
}

class FakeWebSocket {
  static OPEN = 1;
  static dernier: FakeWebSocket | undefined;
  readyState = FakeWebSocket.OPEN;
  binaryType = "";
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string | ArrayBuffer }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;

  constructor(public url: string) {
    FakeWebSocket.dernier = this;
  }
  send() {}
  close() {
    this.readyState = 3;
  }
}

/** Rend VoiceChat, clique sur "Parler" et attend l'ouverture du faux WebSocket. */
async function demarrerConversation() {
  render(<VoiceChat />);
  fireEvent.click(screen.getByRole("button", { name: /Parler/i }));
  await waitFor(() => expect(FakeWebSocket.dernier).toBeDefined());
  return FakeWebSocket.dernier!;
}

describe("VoiceChat — garantie anti demi-réponse", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));
    vi.stubGlobal("AudioContext", FakeAudioContext);
    vi.stubGlobal("WebSocket", FakeWebSocket);
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [] }) },
    });
    FakeWebSocket.dernier = undefined;
    FakeAudioContext.instances = 0;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("affiche « Réponse interrompue » si le socket se ferme sans sources ni erreur", async () => {
    const ws = await demarrerConversation();

    act(() => {
      ws.onmessage?.({ data: JSON.stringify({ type: "token", text: "Bonjour" }) });
    });
    act(() => {
      ws.onclose?.();
    });

    expect(screen.getByText(/Réponse interrompue/i)).toBeTruthy();
  });

  it("n'affiche aucune interruption quand le tour se termine par des sources", async () => {
    const ws = await demarrerConversation();

    act(() => {
      ws.onmessage?.({ data: JSON.stringify({ type: "token", text: "Bonjour" }) });
    });
    act(() => {
      ws.onmessage?.({
        data: JSON.stringify({ type: "sources", sources: [{ source: "a.md", score: 0.9 }] }),
      });
    });
    act(() => {
      ws.onclose?.();
    });

    expect(screen.queryByText(/Réponse interrompue/i)).toBeNull();
  });

  it("affiche l'erreur serveur et pas « Réponse interrompue » quand le tour se termine par une erreur", async () => {
    const ws = await demarrerConversation();

    act(() => {
      ws.onmessage?.({ data: JSON.stringify({ type: "error", message: "Assistant indisponible" }) });
    });
    act(() => {
      ws.onclose?.();
    });

    expect(screen.getByText("Assistant indisponible")).toBeTruthy();
    expect(screen.queryByText(/Réponse interrompue/i)).toBeNull();
  });

  it("enchaîne deux questions en réutilisant les mêmes contextes audio", async () => {
    // Safari iOS limite et ferme de façon asynchrone les AudioContext : les
    // recréer à chaque question empêchait d'enchaîner sur mobile.
    const premier = await demarrerConversation();
    act(() => {
      premier.onmessage?.({
        data: JSON.stringify({ type: "sources", sources: [{ source: "a.md", score: 0.9 }] }),
      });
    });
    await screen.findByRole("button", { name: "Parler" });

    FakeWebSocket.dernier = undefined;
    fireEvent.click(screen.getByRole("button", { name: "Parler" }));
    await waitFor(() => expect(FakeWebSocket.dernier).toBeDefined());

    expect(FakeWebSocket.dernier).not.toBe(premier);
    expect(FakeAudioContext.instances).toBe(2);
  });

  it("ignore un événement tardif de l'ancien socket pendant la question suivante", async () => {
    // Safari iOS émet parfois "error"/"close" sur le socket d'une question
    // terminée, après le démarrage de la suivante : cela coupait la nouvelle.
    const premier = await demarrerConversation();
    act(() => {
      premier.onmessage?.({
        data: JSON.stringify({ type: "sources", sources: [{ source: "a.md", score: 0.9 }] }),
      });
    });
    // Événement tardif pendant que la voix de la première réponse joue encore.
    act(() => premier.onerror?.());
    expect(screen.queryByRole("alert")).toBeNull();

    await screen.findByRole("button", { name: "Parler" });
    FakeWebSocket.dernier = undefined;
    fireEvent.click(screen.getByRole("button", { name: "Parler" }));
    await waitFor(() => expect(FakeWebSocket.dernier).toBeDefined());
    act(() => FakeWebSocket.dernier!.onopen?.());

    act(() => {
      premier.onerror?.();
      premier.onclose?.();
    });

    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByRole("button", { name: "Terminé de parler" })).toBeTruthy();
  });
});

