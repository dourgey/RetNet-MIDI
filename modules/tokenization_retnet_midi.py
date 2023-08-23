from typing import Any, Dict, List, Optional, Tuple
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from tokenizers import Tokenizer

class RetNetMIDITokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file,  unk_token="<pad>", bos_token="<start>", eos_token="<end>", pad_token="<pad>", sep_token="<sep>", add_bos_token=True, add_eos_token=False, clean_up_tokenization_spaces=False):
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, pad_token=pad_token, sep_token=sep_token, add_bos_token=add_bos_token, add_eos_token=add_eos_token, clean_up_tokenization_spaces=clean_up_tokenization_spaces)
        self.vocab_file = vocab_file
        self.model = Tokenizer.from_file(vocab_file)
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token


    @property
    def vocab_size(self) -> int:
        return self.model.get_vocab_size()

    def get_vocab(self) -> Dict[str, int]:
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i) : i for i in range(self.vocab_size)}
        return vocab

    def _tokenize(self, text: str) -> List[int]:
        return self.model.encode(text).tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.model.token_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.model.id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.model.decode(tokens, skip_special_tokens=False)





if __name__ == '__main__':
    vocab_file = r"C:\Users\zhaoj\OneDrive\Project\RetNet-MIDI\modules\MIDI-LLM-tokenizer\tokenizer-midi.json"
    tokenizer = RetNetMIDITokenizer(vocab_file)
    print(tokenizer(["<start> pi:2f:7 pi:36:7 pi:3b:7 t123 <end> <sep> <start> pi:43:7 t58 pi:43:0 t3 pi:45:7", "t1 pi:47:9 pi:4c:9 pi:4f:0 <end>"], padding=True, return_tensors='pt'))