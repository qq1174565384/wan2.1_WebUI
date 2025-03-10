# -*- coding: utf-8 -*-


import os

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 新的 cmd 文件的完整路径
new_cmd_file = os.path.join(current_dir, "启动.cmd")

# 要写入新 cmd 文件的内容
content = """python qidong.py
pause
"""

# 写入内容到新的 cmd 文件
with open(new_cmd_file, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"已在 {current_dir} 目录下生成新的 cmd 文件：{new_cmd_file}")
