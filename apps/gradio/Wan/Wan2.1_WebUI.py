import gradio as gr
import webbrowser

# 定义一个简单的函数，用于处理输入并返回输出
def greet(name):
    return f"Hello, {name}!"

# 创建 Gradio 界面
iface = gr.Interface(
    fn=greet,  # 要调用的函数
    inputs=gr.Textbox(label="Your Name"),  # 输入组件
    outputs=gr.Textbox(label="Greeting"),  # 输出组件
    title="Simple Gradio App"  # 界面标题
)

# 启动 Gradio 应用
if __name__ == "__main__":
    # 启动 Gradio 应用并自动打开浏览器
    iface.launch(inbrowser=True)