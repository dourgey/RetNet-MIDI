import json
import os
import queue

import tqdm
import statistics
import matplotlib.pyplot as plt
import multiprocessing
from modules.tokenization_retnet_midi import RetNetMIDITokenizer


def find_files(root_dir: str, postfix: tuple = ('.midi', '.mid', '.kar', '.rmi', '.smf')) -> list:
    """
    找到路径及其子路径下指定后缀的所有文件
    :param root_dir:
    :param postfix:
    :return:
    """
    files = []

    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查文件扩展名是否为 MIDI 相关的后缀
            if filename.endswith(postfix):
                path = os.path.join(foldername, filename)
                absolute_path = os.path.abspath(path)  # 将路径转换为绝对路径
                files.append(absolute_path)

    return files


jsonl_files = find_files("../data/bread-midi-dataset", ('.midi', '.mid', '.kar', '.rmi', '.smf', "jsonl"))
vocab_file = r"..\modules\MIDI-LLM-tokenizer\tokenizer-midi.json"
tokenizer = RetNetMIDITokenizer(vocab_file)


def process_file(i, file):
    with open(file, 'r', encoding='utf-8') as f:
        d = f.read().strip()
    print(f"第 {i} 个文件处理完成".format(i))
    return {"file": file, "token_length": len(tokenizer(d)['input_ids'])}


if __name__ == "__main__":

    pool = multiprocessing.Pool(12)  # 创建进程池
    for i, file in tqdm.tqdm(enumerate(jsonl_files)):
        pool.apply_async(process_file, (i, file,))

    pool = multiprocessing.Pool(12)  # 创建进程池
    results = queue.Queue()

    print("要处理的文件数量", len(jsonl_files))

    for i, file in tqdm.tqdm(enumerate(jsonl_files)):
        result = pool.apply_async(process_file, (i, file,))
        results.put(result)


    def queue_to_list(queue_obj):
        result = []
        while not queue_obj.empty():
            item = queue_obj.get()
            result.append(item)
        return result


    results = queue_to_list(results)

    data = []
    count = 0

    for result in results:
        count += 1
        data.append({"idx": count, **result.get(), 'label': 0})

    sorted_data = sorted(data, key=lambda x: x["token_length"])

    with open("data_meta.json", 'w', encoding='utf-8') as f:
        for d in sorted_data:
            f.write(json.dumps(d) + '\n')

    # 假设 sorted_data 包含了已经按 "token_length" 排序的数据
    token_lengths = [item["token_length"] for item in sorted_data]

    # 平均值
    average = statistics.mean(token_lengths)

    # 中位数
    median = statistics.median(token_lengths)

    # 最小值
    min_value = min(token_lengths)

    # 最大值
    max_value = max(token_lengths)

    print(f"平均值: {average}")
    print(f"中位数: {median}")
    print(f"最小值: {min_value}")
    print(f"最大值: {max_value}")

    # 绘制直方图
    plt.hist(token_lengths, bins=20, edgecolor='k')  # 可以调整 bin 的数量
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.title("Token Length Histogram")

    # 保存直方图到本地文件
    plt.savefig("token_length_histogram.png")

    # 假设您有一个包含已排序数据的 sorted_data 列表
    num_files = 20
    total_items = len(sorted_data)
    items_per_file = total_items // num_files  # 计算每个文件包含的数据项数量

    # 将数据平均分成 num_files 个子列表
    data_split = [sorted_data[i:i + items_per_file] for i in range(0, total_items, items_per_file)]

    if os.path.exists("../data/training_data"):
        pass
    else:
        os.mkdir("../data/training_data")

    # 创建并写入分割后的数据到不同的文件
    for i, data_chunk in enumerate(data_split):
        filename = f"training_data/output_file_{i + 1}.txt"  # 根据您的文件命名规则进行更改
        with open(filename, 'w') as file:
            for item in data_chunk:
                with open(item['file'], 'r', encoding='utf-8') as f:
                    s = f.read().strip()
                item['sentence'] = s
                file.write(json.dumps(item, ensure_ascii=False) + '\n')
