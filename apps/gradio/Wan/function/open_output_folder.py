import os

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


def t2v_open_output_folder():
    output_dir = os.path.join(project_root,"output", "t2v")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 根据操作系统打开文件夹
    os.startfile(output_dir)

def i2v_open_output_folder():
    output_dir = os.path.join(project_root,"output", "i2v")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 根据操作系统打开文件夹
    os.startfile(output_dir)
