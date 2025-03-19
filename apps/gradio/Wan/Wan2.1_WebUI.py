import sys
print(sys.executable)

import subprocess

def wan_1_3b_generation_video():
    # 使用当前解释器启动 wan_1_3b_text_to_video.py
    subprocess.run([sys.executable, "wan_1_3b_generation_video.py"])

if __name__ == "__main__":
    wan_1_3b_generation_video()
