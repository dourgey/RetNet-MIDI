import json
import os
from pathlib import Path
import tqdm
import sqlite3
from torch.utils.data import Dataset, DataLoader

class BreadMidiDataset(Dataset):
    def __init__(self, db_path, max_dataset_length):
        self.dir_path = Path(db_path)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        self.max_len = max_dataset_length


        if max_dataset_length in [1024, 2048, 4096, 8192, 16384, 32768, 8192]:
            kv = {1024: 106971, 2048: 172704, 4096: 278835, 8192: 433837, 16384: 672644, 32768: 809395}
            self.max_idx = kv[max_dataset_length]
        else:
            self.max_idx = self.cursor.execute("select count(id) from midi_pretrain_dataset where token_length <= ?",
                                               (self.max_len,)).fetchone()[0]



    def __getitem__(self, index):
        return self.cursor.execute("""select sentence from midi_pretrain_dataset where id = ?""", (index,)).fetchone()[0]


    def __len__(self):
        return self.max_idx

    @staticmethod
    def _read_specific_line(filename, index):
        """
        快速读取大文件中的某一行
        :param filename:
        :param index:
        :return:
        """
        with open(filename, 'r', encoding='utf-8') as file:
            i = 0
            while True:
                if i == index:
                    return file.readline()
                i += 1
                file.readline()
        raise IOError


# TODO:
# collator for midi data
class MidiCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        pass


if __name__ == '__main__':

    dataset = BreadMidiDataset(r"C:\Users\zhaoj\Desktop\sqlite\dataset.db", 4096)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=24, shuffle=False, num_workers=0, collate_fn=None)
    for data in tqdm.tqdm(dataloader):
        print(data)
        break


