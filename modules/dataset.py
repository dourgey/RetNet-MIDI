import json
import os
from pathlib import Path
import tqdm
from torch.utils.data import Dataset, DataLoader

class BreadMidiDataset(Dataset):
    def __init__(self, dir_path, max_dataset_length):
        self.dir_path = Path(dir_path)

        file_list = {'output_file_1.txt': 34719, 'output_file_2.txt': 41621, 'output_file_3.txt': 41621,
                     'output_file_4.txt': 41621, 'output_file_5.txt': 41621, 'output_file_6.txt': 41621,
                     'output_file_7.txt': 41621, 'output_file_8.txt': 41621, 'output_file_9.txt': 41621,
                     'output_file_10.txt': 41621, 'output_file_11.txt': 41621, 'output_file_12.txt': 41621,
                     'output_file_13.txt': 41621, 'output_file_14.txt': 41621, 'output_file_15.txt': 41621,
                     'output_file_16.txt': 41621, 'output_file_17.txt': 41621, 'output_file_18.txt': 41621,
                     'output_file_19.txt': 41621, 'output_file_20.txt': 41621}

        self.file_list = {x: file_list[x] for x in file_list}

        assert os.path.exists(self.dir_path)
        for x in self.file_list:
            assert os.path.exists(self.dir_path / x)

        self.length = min(max_dataset_length, sum(self.file_list.values()))



    def __getitem__(self, index):
        count = 0
        for i in range(1, 21):
            if count + self.file_list[f"output_file_{i}.txt"] < index:
                count += self.file_list[f"output_file_{i}.txt"]
            else:
                break
        line = self._read_specific_line(self.dir_path / f"output_file_{i}.txt", index - count)

        return json.loads(line)["sentence"]


    def __len__(self):
        return self.length

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
        pass

    def __call__(self, batch):
        pass


if __name__ == '__main__':
    dataset = BreadMidiDataset("../data/training_data", 100)
    dataloader = DataLoader(dataset, batch_size=24, shuffle=False, num_workers=0, collate_fn=None)
    for data in tqdm.tqdm(dataloader):
        print(data)

