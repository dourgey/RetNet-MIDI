import argparse
import os

import mido

import midi_util
from midi_util import VocabConfig

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # p.add_argument(
    #     "filename",
    #     type=str,
    #     help="The MIDI file to convert to text",
    #     default=r"C:\Users\zhaoj\OneDrive\tmp\midi训练集\wuyuetianMayday-_paiduidongwu_Party_Animals.mid"
    # )
    p.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to output text file",
        default="test.jsonl"
    )
    p.add_argument(
        "--vocab_config",
        type=str,
        default="./vocab_config.json",
        help="Path to vocab config file",
    )

    args = p.parse_args()

    cfg = VocabConfig.from_json(args.vocab_config)

    filename = r"C:\Users\zhaoj\OneDrive\tmp\midi训练集\wuyuetianMayday-_paiduidongwu_Party_Animals.mid"

    # mid = mido.MidiFile(args.filename)
    mid = mido.MidiFile(filename)
    text = midi_util.convert_midi_to_str(cfg, mid)
    
    if args.output is not None:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
