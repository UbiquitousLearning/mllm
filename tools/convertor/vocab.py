import json
import struct
from typing import Iterable, Tuple
import argparse
import os

MAGIC_NUM = 23333
parser = argparse.ArgumentParser()
parser.add_argument(
    # "--input_file", type=str, default="tokenizer.model"
    "--input_file", type=str, default="tokenizer.json"
)
# parser.add_argument("--output_file", type=str, default="vocab.mllm")
parser.add_argument("--output_file", type=str, default="vocab_uni.mllm")
parser.add_argument(
    "--type",
    choices=["BPE", "Unigram"],
    # default="BPE",
    default="Unigram",
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
    print(len(tokenizer_config["vocab"]))
    idx = 0
    vocab = tokenizer_config["vocab"]
    vocab_type = tokenizer_config["type"]
    if type(vocab) is dict:
        vocab = [ [k, v] for k, v in vocab.items()]
        for token in added_vocab:
            print(token)
            token_str = token["content"]
            token_score = token["id"]
            if vocab.count(token_str) == 0:
                vocab.append([token_str, token_score])
    vocab_file.write(struct.pack("<i", len(vocab)))
    for token_score in vocab:
        token, score = token_score[0], token_score[1]
        if vocab_type == "BPE":
            idx = int(token_score[1])
        vocab_file.write(struct.pack("<i", idx))
        # # print(idx)
        # if (71002 <= idx):
        #     print(token)

        token_ = token.encode("utf-8")
        vocab_file.write(struct.pack("<i", len(token_)))
        vocab_file.write(token_)
        vocab_file.write(struct.pack("<f", score))
        print(idx, token)
        if vocab_type != "BPE":
            idx += 1
        # print(token, score)


if __name__ == "__main__":
    global added_vocab
    args = parser.parse_args()
    output_file = args.output_file
    input_file = args.input_file
    added_vocab = []
    with open(output_file, "wb+") as vocab_file:
        vocab_file.write(struct.pack("<i", MAGIC_NUM))

        if args.type == "BPE" and input_file.endswith(".model"):
            from sentencepiece import SentencePieceProcessor  # type: ignore

            sentencepiece_tokenizer = SentencePieceProcessor(str(input_file))
            write_vocab(vocab_file, sentencepiece_tokenizer)
        elif args.type == "BPE" and os.path.basename(input_file) =="vocab.json":
            vocabs = json.load(open(input_file, "r"))
            config = {"vocab": vocabs, "type": "BPE"}
            write_unigram(vocab_file, config)

        else:
            tokenizer_config = json.load(open(input_file, "r"))

            # elif args.type == "Unigram":
            config = tokenizer_config["model"]
            added_vocab = tokenizer_config.get("added_tokens",[])
            merges = config.get("merges",[])
            if config["type"] == "BPE" and len(merges) != 0:
                with open(f"{output_file}.merges.txt", "w+") as f:
                    for merge in merges:
                        f.write(f"{merge}\n")

            if config["type"] == "Unigram" or config["type"] == "BPE":
                write_unigram(vocab_file, config)
            else:
                raise Exception("Not implemented! Only Unigram Supported!")