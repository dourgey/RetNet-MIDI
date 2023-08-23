from tokenizers import Tokenizer

# from modules.retnet.modeling_retnet import RetNetConfig, RetNetModel
#
#
#
#
# config = RetNetConfig()


tokenizer = Tokenizer.from_file(r"..\modules\MIDI-LLM-tokenizer\tokenizer-midi.json")

with open("../data/test.jsonl", 'r', encoding='utf-8') as f:
    lines = f.readlines()
sample = lines[0]
print(tokenizer.encode(sample).ids)
