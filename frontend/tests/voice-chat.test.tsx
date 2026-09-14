import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import VoiceChat from "../components/VoiceChat";

// Le composant vocal dépend de getUserMedia/AudioContext/WebSocket : jsdom ne
// les fournit pas, donc ce test se limite au rendu initial (avant toute
// interaction), qui n'appelle aucune de ces API. Le contrat WebSocket est
// couvert séparément par tests/voix.test.ts (analyserMessageVoix).
describe("VoiceChat", () => {
  it("affiche le bouton d'entrée et une zone de transcription accessible au repos", () => {
    render(<VoiceChat />);
    expect(screen.getByRole("button", { name: /Parler/i })).toBeTruthy();
    const zone = screen.getByRole("log");
    expect(zone.getAttribute("aria-live")).toBe("polite");
  });
});

// --- Doublures des API navigateur absentes de jsdom (micro, audio, socket) ---
// L'objectif n'est pas de tester la capture micro mais de pouvoir déclencher
// demarrer() jusqu'à l'ouverture du WebSocket, seul canal qui porte la
// garantie anti demi-réponse.

class FakeAudioContext {
  destination = {};
  currentTime = 0;
  audioWorklet = {
    addModule: () => Promise.reject(new Error("AudioWorklet indisponible sous jsdom")),
  };
  createMediaStreamSource() {
    return { connect: () => {} };
  }
  createGain() {
    return { gain: { value: 0 }, connect: () => {} };
  }
  createScriptProcessor() {
    return { connect: () => {}, onaudioprocess: null };
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
    vi.stubGlobal("AudioContext", FakeAudioContext);
    vi.stubGlobal("WebSocket", FakeWebSocket);
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [] }) },
    });
    FakeWebSocket.dernier = undefined;
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
});
