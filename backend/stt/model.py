from dataclasses import dataclass
import math
import time
import sentencepiece
import sphn
import torch

from moshi.models import loaders, MimiModel, LMModel, LMGen

@dataclass
class InferenceState:
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen

    def __init__(
        self,
        mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        batch_size: int,
        device: str | torch.device,
    ):
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.lm_gen = LMGen(lm, temp=0, temp_text=0, use_sampling=False)
        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.batch_size = batch_size
        self.mimi.streaming_forever(batch_size)
        self.lm_gen.streaming_forever(batch_size)
        # Pour `step()` : le tout premier pas doit etre joue deux fois (meme
        # contrainte que `run()`, cf. plus bas), les suivants une seule fois.
        self._premiere_trame_traitee = False

    def reinitialiser(self):
        """Repart d'un flux vierge : l'etat est partage par toutes les sessions
        vocales, sans reset la suivante heriterait du contexte de la precedente."""
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        self._premiere_trame_traitee = False

    def step(self, chunk: torch.Tensor) -> str:
        """Traite une seule trame (streaming temps reel) et renvoie le texte
        produit, eventuellement vide. Contrepartie de `run()` pour un flux qui
        arrive au fil de l'eau plutot que d'un seul bloc."""
        codes = self.mimi.encode(chunk)
        if not self._premiere_trame_traitee:
            self.lm_gen.step(codes)
            self._premiere_trame_traitee = True
        tokens = self.lm_gen.step(codes)
        if tokens is None:
            return ""
        assert tokens.shape[1] == 1
        one_text = tokens[0, 0].cpu()
        if one_text.item() in (0, 3):
            return ""
        return self.text_tokenizer.id_to_piece(one_text.item()).replace("▁", " ")

    def run(self, in_pcms: torch.Tensor):
        ntokens = 0
        first_frame = True
        chunks = [
            c
            for c in in_pcms.split(self.frame_size, dim=2)
            if c.shape[-1] == self.frame_size
        ]
        start_time = time.time()
        all_text = []
        for chunk in chunks:
            codes = self.mimi.encode(chunk)
            if first_frame:
                tokens = self.lm_gen.step(codes)
                first_frame = False
            tokens = self.lm_gen.step(codes)
            if tokens is None:
                continue
            assert tokens.shape[1] == 1
            one_text = tokens[0, 0].cpu()
            if one_text.item() not in [0, 3]:
                text = self.text_tokenizer.id_to_piece(one_text.item())
                text = text.replace("▁", " ")
                all_text.append(text)
            ntokens += 1
        dt = time.time() - start_time
        print(
            f"processed {ntokens} steps in {dt:.0f}s, {1000 * dt / ntokens:.2f}ms/step"
        )
        return "".join(all_text)
    

_DEPOT = "kyutai/stt-1b-en_fr"


def choisir_device() -> str:
    """CUDA est aussi le nom du backend ROCm sous PyTorch : la detection vaut
    pour la RX 7700 XT comme pour une carte NVIDIA."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _charger_checkpoint(device: str):
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(_DEPOT)
    mimi = checkpoint_info.get_mimi(device=device)
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    lm = checkpoint_info.get_moshi(device=device)
    return checkpoint_info, mimi, text_tokenizer, lm


def charger_state(device: str | None = None, batch_size: int = 1) -> tuple["InferenceState", int]:
    """Charge le modele une seule fois (a appeler au demarrage de l'API, pas
    par requete) et renvoie un `InferenceState` pret pour `step()`, ainsi que
    le nombre de trames de silence a injecter en fin de session.

    Ce nombre purge le delai interne du modele (`audio_delay_seconds`) : sans
    cette purge, les derniers mots prononces par le visiteur restent bloques
    dans le pipeline et ne sont jamais transcrits.
    """
    device = device or choisir_device()
    checkpoint_info, mimi, text_tokenizer, lm = _charger_checkpoint(device)
    state = InferenceState(mimi, text_tokenizer, lm, batch_size=batch_size, device=device)
    pad_right_secondes = checkpoint_info.stt_config.get("audio_delay_seconds", 0.0) + 1.0
    trames_de_purge = math.ceil(pad_right_secondes * mimi.sample_rate / state.frame_size)
    return state, trames_de_purge


def transcribe(chemin_audio: str, device: str | None = None, batch_size: int = 1) -> str:
    """Transcrit un fichier audio. Charge le modele a l'appel, jamais a l'import.

    Avant, tout ce bloc s'executait au niveau module : importer `model` chargeait
    plusieurs Go de poids et lisait un fichier en dur. Aucun appelant ne pouvait
    donc importer ce module sans payer une inference complete.
    """
    device = device or choisir_device()
    checkpoint_info, mimi, text_tokenizer, lm = _charger_checkpoint(device)

    in_pcms, _ = sphn.read(chemin_audio, sample_rate=mimi.sample_rate)
    in_pcms = torch.from_numpy(in_pcms).to(device=device)

    stt_config = checkpoint_info.stt_config
    pad_left = int(stt_config.get("audio_silence_prefix_seconds", 0.0) * 24000)
    pad_right = int((stt_config.get("audio_delay_seconds", 0.0) + 1.0) * 24000)
    in_pcms = torch.nn.functional.pad(in_pcms, (pad_left, pad_right), mode="constant")
    in_pcms = in_pcms[None, 0:1].expand(1, -1, -1)

    state = InferenceState(mimi, text_tokenizer, lm, batch_size=batch_size, device=device)
    return state.run(in_pcms)
