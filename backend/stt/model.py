from dataclasses import dataclass
import time
import sentencepiece
import sphn
import textwrap
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
    

device = "cuda"
checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
mimi = checkpoint_info.get_mimi(device=device)
text_tokenizer = checkpoint_info.get_text_tokenizer()
lm = checkpoint_info.get_moshi(device=device)
in_pcms, _ = sphn.read("test.mp3", sample_rate=mimi.sample_rate)
in_pcms = torch.from_numpy(in_pcms).to(device=device)

stt_config = checkpoint_info.stt_config
pad_left = int(stt_config.get("audio_silence_prefix_seconds", 0.0) * 24000)
pad_right = int((stt_config.get("audio_delay_seconds", 0.0) + 1.0) * 24000)
in_pcms = torch.nn.functional.pad(in_pcms, (pad_left, pad_right), mode="constant")
in_pcms = in_pcms[None, 0:1].expand(1, -1, -1)

state = InferenceState(mimi, text_tokenizer, lm, batch_size=1, device=device)
text = state.run(in_pcms)
print(textwrap.fill(text, width=100))
