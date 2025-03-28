# 获取项目根目录的绝对路径
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

def load_t2v_history():
    """
    按创建时间顺序读取project_root\output\t2v文件夹下的MP4文件和对应的txt文件。
    提取txt文件第一行中'prompt:'后的内容，并与MP4文件路径组合成列表。
    :return: 包含prompt和MP4文件路径的二维列表
    """
    t2v_dir = os.path.join(project_root, 'output', 't2v')
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(t2v_dir):
        os.makedirs(t2v_dir)
    # 获取所有MP4文件
    mp4_files = [f for f in os.listdir(t2v_dir) if f.endswith('.mp4')]
    # 按创建时间降序排序
    mp4_files.sort(key=lambda x: os.path.getctime(os.path.join(t2v_dir, x)), reverse=True)
    t2v_history_list = []
    for mp4_file in mp4_files[:]:  # 使用切片复制列表，避免在迭代时修改原列表
        # 对应的txt文件名
        txt_file = os.path.splitext(mp4_file)[0] + '.txt'
        txt_path = os.path.join(t2v_dir, txt_file)
        mp4_path = os.path.join(t2v_dir, mp4_file)
        if not os.path.exists(txt_path):
            # 如果txt文件不存在，跳过当前MP4文件
            mp4_files.remove(mp4_file)
            continue
        with open(txt_path, 'r', encoding='utf-8') as f:
            # 读取整个文件内容
            content = f.read()
            start_index = content.find('prompt:')
            if start_index == -1:
                # 如果没有 'prompt:'，跳过当前 MP4 文件
                mp4_files.remove(mp4_file)
                continue
            start_index += len('prompt:')
            end_index = content.find('negative_prompt')
            if end_index == -1:
                # 如果没有 'negative_prompt'，就取到文件末尾
                prompt = content[start_index:].strip()
            else:
                prompt = content[start_index:end_index].strip()
    
            t2v_history_list.append([prompt, mp4_path])

    return t2v_history_list

t2v_history_list = load_t2v_history()

def load_i2v_history():
    """
    按创建时间顺序读取project_root\output\i2v文件夹下的MP4文件和对应的txt文件。
    提取txt文件第一行中'prompt:'后的内容，并与MP4文件路径组合成列表。
    :return: 包含prompt和MP4文件路径的二维列表
    """
    i2v_dir = os.path.join(project_root, 'output', 'i2v')
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(i2v_dir):
        os.makedirs(i2v_dir)
    # 获取所有MP4文件
    mp4_files = [f for f in os.listdir(i2v_dir) if f.endswith('.mp4')]
    # 按创建时间降序排序
    mp4_files.sort(key=lambda x: os.path.getctime(os.path.join(i2v_dir, x)), reverse=True)
    i2v_history_list = []
    for mp4_file in mp4_files[:]:  # 使用切片复制列表，避免在迭代时修改原列表
        # 对应的txt文件名
        txt_file = os.path.splitext(mp4_file)[0] + '.txt'
        txt_path = os.path.join(i2v_dir, txt_file)
        mp4_path = os.path.join(i2v_dir, mp4_file)
        if not os.path.exists(txt_path):
            # 如果txt文件不存在，跳过当前MP4文件
            mp4_files.remove(mp4_file)
            continue
        with open(txt_path, 'r', encoding='utf-8') as f:
            # 读取整个文件内容
            content = f.read()
            start_index = content.find('prompt:')
            if start_index == -1:
                # 如果没有 'prompt:'，跳过当前 MP4 文件
                mp4_files.remove(mp4_file)
                continue
            start_index += len('prompt:')
            end_index = content.find('negative_prompt')
            if end_index == -1:
                # 如果没有 'negative_prompt'，就取到文件末尾
                prompt = content[start_index:].strip()
            else:
                prompt = content[start_index:end_index].strip()

            i2v_history_list.append([prompt, mp4_path])

    return i2v_history_list

i2v_history_list = load_i2v_history()

