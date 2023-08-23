import os

# 定义一个函数来遍历文件夹及其子文件夹并找到所有的MIDI文件
def find_midi_files(root_dir):
    midi_files = []

    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查文件扩展名是否为 MIDI 相关的后缀
            if filename.endswith(('.midi', '.mid', '.kar', '.rmi', '.smf')):
                midi_path = os.path.join(foldername, filename)
                midi_files.append(midi_path)

    return midi_files


files = find_midi_files('../data')
print(files)
print(len(files))