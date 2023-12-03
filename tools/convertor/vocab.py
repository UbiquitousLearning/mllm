import json
import struct
from typing import Iterable, Tuple
import argparse

MAGIC_NUM = 23333

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_file", type=str, default="tokenizer.model"
)
parser.add_argument("--output_file", type=str, default="vocab.mllm")
parser.add_argument(
    "--type",
    choices=["BPE", "Unigram"],
    default="BPE",
)


def sentencepiece_tokens(
        tokenizer
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


def write_unigram(vocab_file, tokenizer_config):
    vocab_file.write(struct.pack("<i", len(tokenizer_config["vocab"])))
    idx = 0
    for token_score in tokenizer_config["vocab"]:
        token, score = token_score[0], token_score[1]
        vocab_file.write(struct.pack("<i", idx))
        vocab_file.write(struct.pack("<i", len(token)))
        vocab_file.write(token.encode("utf-8"))
        vocab_file.write(struct.pack("<f", score))
        idx += 1
        print(token, score)


if __name__ == "__main__":
    args = parser.parse_args()
    output_file = args.output_file
    input_file = args.input_file
    with open(output_file, "wb+") as vocab_file:
        if args.type == "BPE":
            from sentencepiece import SentencePieceProcessor  # type: ignore

            sentencepiece_tokenizer = SentencePieceProcessor(str(input_file))
            vocab_file = open(output_file, "wb+")
            write_vocab(vocab_file, sentencepiece_tokenizer)
        elif args.type == "Unigram":
            tokenizer_config = json.load(open(input_file, "r"))
            vocab_file = open(output_file, "wb+")
            config = tokenizer_config["model"]
            if config["type"] == "Unigram":
                write_unigram(vocab_file, config)
            else:
                raise Exception("Not implemented! Only Unigram Supported!")

        else:
            raise Exception("Not implemented")
