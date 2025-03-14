import gradio as gr
import webbrowser
# 导入 wan_1.3b_text_to_video.py 中的函数
from wan_1_3b_text_to_video import text_to_video


# 创建新的 Gradio 界面来调用 text_to_video 函数
video_iface = gr.Interface(
    fn=text_to_video,  # 要调用的函数
    inputs=gr.Textbox(label="输入文本"),  # 输入组件
    outputs=gr.Video(label="生成的视频"),  # 输出组件
    title="文本到视频转换"  # 界面标题
)

# 启动 Gradio 应用
if __name__ == "__main__":
    # 启动 Gradio 应用并自动打开浏览器
    video_iface.launch(inbrowser=True)
