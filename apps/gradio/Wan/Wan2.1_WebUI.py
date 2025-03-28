import sys
# 添加提示信息
print("正在使用的python为：", sys.executable)

import subprocess

print("正在加载请稍等。。。")
def wan_generation_video():
    # 使用当前解释器启动 wan_1_3b_text_to_video.py
    subprocess.run([sys.executable, "wan_generation_video.py"])

if __name__ == "__main__":
    wan_generation_video()
