import os

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


def open_output_folder():
    output_dir = os.path.join(project_root,"output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 根据操作系统打开文件夹
    os.startfile(output_dir)
