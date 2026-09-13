import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import Chat from "../components/Chat";

// Saut de ligne reel : un bloc SSE est separe par une ligne vide.
const LF = `
`;
const bloc = (contenu: string) => contenu + LF + LF;

/** Fabrique une reponse fetch dont le corps rend les morceaux fournis, dans l'ordre. */
function reponseFlux(morceaux: string[]) {
  const encodeur = new TextEncoder();
  let i = 0;
  return {
    ok: true,
    status: 200,
    body: {
      getReader: () => ({
        read: async () =>
          i < morceaux.length
            ? { done: false, value: encodeur.encode(morceaux[i++]) }
            : { done: true, value: undefined },
        cancel: async () => {},
      }),
    },
  } as unknown as Response;
}

async function poser(morceaux: string[]) {
  vi.stubGlobal("fetch", vi.fn(async () => reponseFlux(morceaux)));
  render(<Chat />);
  fireEvent.change(screen.getByPlaceholderText(/Poser une question/i), {
    target: { value: "Une question" },
  });
  fireEvent.submit(screen.getByRole("button", { name: /Envoyer/i }));
}

const PING = bloc(": ping");
const TOKEN_A = bloc('data: {"token": "Bonjour"}');
const TOKEN_B = bloc('data: {"token": " Mathis"}');
const SOURCES = bloc('event: sources' + LF + 'data: {"sources": [{"source": "parcours.md", "titre": "Qui je suis", "score": 0.8}]}');

describe("parseur SSE du chat", () => {
  it("signale un flux coupe sans bloc terminal au lieu de le presenter comme complet", async () => {
    // Ollama tue en OOM, ou tunnel qui timeout : le socket se ferme proprement
    // sans event terminal. Sans garde, une demi-phrase s'affiche comme la
    // reponse complete de l'assistant.
    await poser([PING, TOKEN_A, TOKEN_B]);
    await waitFor(() => expect(screen.getByText(/reponse interrompue/i)).toBeTruthy());
  });

  it("n'affiche pas d'erreur quand le bloc sources termine le flux", async () => {
    await poser([PING, TOKEN_A, SOURCES]);
    await waitFor(() => expect(screen.getByText(/Qui je suis/)).toBeTruthy());
    expect(screen.queryByText(/reponse interrompue/i)).toBeNull();
  });

  it("n'affiche jamais le commentaire ping dans la reponse", async () => {
    await poser([PING, TOKEN_A, SOURCES]);
    await waitFor(() => expect(screen.getByText(/Bonjour/)).toBeTruthy());
    expect(screen.queryByText(/ping/i)).toBeNull();
  });

  it("reassemble un bloc coupe entre deux lectures", async () => {
    await poser([PING, 'data: {"tok', 'en": "Recolle"}' + LF + LF, SOURCES]);
    await waitFor(() => expect(screen.getByText(/Recolle/)).toBeTruthy());
  });

  it("affiche l'erreur portee par un event error", async () => {
    await poser([PING, TOKEN_A, bloc('event: error' + LF + 'data: {"error": "generation interrompue"}')]);
    await waitFor(() => expect(screen.getByText(/generation interrompue/i)).toBeTruthy());
  });
});
