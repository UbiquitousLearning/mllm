import struct
from typing import Iterable, Tuple

from sentencepiece import SentencePieceProcessor  # type: ignore

MAGIC_NUM = 23333
out_fname = "./vocab.mllm"
fname_tokenizer = "./tokenizer.model"
sentencepiece_tokenizer = SentencePieceProcessor(str(fname_tokenizer))
vocab_file = open(out_fname, "wb+")
vocab_file.write(struct.pack("<i", MAGIC_NUM))


def sentencepiece_tokens(
        tokenizer: SentencePieceProcessor,
) -> Iterable[Tuple[bytes, float]]:
    for i in range(tokenizer.vocab_size()):
        text: bytes
        if tokenizer.is_unknown(i):
            text = " \u2047 ".encode("utf-8")
        elif tokenizer.is_control(i):
            text = b""
        elif tokenizer.is_byte(i):
            piece = tokenizer.id_to_piece(i)
            if len(piece) != 6:
                raise Exception(f"Invalid token: {piece}")
            byte_value = int(piece[3:-1], 16)
            text = struct.pack("B", byte_value)
        else:
            text = tokenizer.id_to_piece(i).replace("\u2581", " ").encode("utf-8")
        score: float = tokenizer.get_score(i)
        yield text, score


def write_vocab(vocab_file, tokenizer):
    vocab_file.write(struct.pack("<i", tokenizer.vocab_size()))
    idx = 0
    for token, score in sentencepiece_tokens(tokenizer):
        vocab_file.write(struct.pack("<i", idx))
        vocab_file.write(struct.pack("<i", len(token)))
        vocab_file.write(token)
        vocab_file.write(struct.pack("<f", score))
        idx += 1


if __name__ == "__main__":
    write_vocab(vocab_file, sentencepiece_tokenizer)
    vocab_file.close()
