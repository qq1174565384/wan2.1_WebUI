import gradio as gr
from t2v_ui import create_t2v_ui
from t2v_button_events import setup_button_events as t2v_button_events
import requests
from ModelManager import ModelManager
# # 定义css样式
custom_css = """
/* 按钮样式 */
#button {
    background-color: #2263dd; /* 暗蓝色背景 */
    color: #f8f9fa; /* 浅灰色文字 */
    padding: 12px 24px;
    font-size: 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.3s ease 0s;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2); /* 添加轻微阴影 */
    height: 90px; /* 设置按钮高度 */
}

#button:hover {
    background-color: #2c3136; /* 更深的暗灰色背景 */
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4); /* 增强阴影效果 */
}

#button_jinyong {
    background-color: #2c3136; /* 暗蓝色背景 */
    color: #f8f9fa; /* 浅灰色文字 */
    padding: 12px 24px;
    font-size: 20px;
    border: none;
    border-radius: 4px;
    cursor: default;
    transition: all 0.3s ease 0s;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2); /* 添加轻微阴影 */
    height: 90px; /* 设置按钮高度 */
}

#button_jinyong:hover {
    background-color: #2c3136; /* 更深的暗灰色背景 */
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4); /* 增强阴影效果 */
}

#button2 {
    background-color: #2263dd; /* 暗蓝色背景 */
    height: 78px; /* 设置按钮高度 */
    border-radius: 4px;
}

#button2:hover {
    background-color: #2c3136; /* 更深的暗灰色背景 */
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4); /* 增强阴影效果 */
}

/* 复选框样式 */
#custom-checkbox input[type="checkbox"] {
    appearance: none; /* 移除默认样式 */
    width: 20px;
    height: 20px;
    background-color: #444;
    border-radius: 4px;
    position: relative;
    cursor: pointer;
    outline: none;
    border: 2px solid ##2263dd; /* 边框颜色 */
    margin-top: 00px; /* 上边距 */
}

#custom-checkbox input[type="checkbox"]:checked {
    background-color: #2c3136; /* 选中时的背景颜色 */
}

#custom-checkbox input[type="checkbox"]:after {
    content: '';
    position: absolute;
    display: none;
}

#custom-checkbox input[type="checkbox"]:checked:after {
    display: block;
    top: 3px;
    left: 6px;
    width: 6px;
    height: 12px;
    border: solid #2c3136; /* 对勾的颜色 */
    border-width: 0 2px 2px 0;
    transform: rotate(45deg);
}

#custom-checkbox label {
    color: #f8f9fa; /* 文本颜色 */
    font-size: 15px;
    height: 70px; 
    cursor: pointer;
    margin-left: 20px; /* 文本与复选框之间的间距 */
    margin-top: 0px; /* 上边距 */
}
"""
# # 创建 Gradio 界面
with gr.Blocks(css=custom_css,theme=gr.themes.Base()) as demo:
    # 显示一个 HTML 标题，居中、32px 字体大小、加粗，并在底部留出 20px 边距
    gr.HTML("""
               <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                   Wan2.1 (DiffSynth-Studio版）
               </div>
               """)
    # 定义一个状态变量，用于存储任务 ID，初始值为空字符串
    task_id = gr.State(value="")
    # 定义一个状态变量，用于存储任务状态，初始值为 False
    status = gr.State(value=False)
    # 定义一个状态变量，用于存储当前任务类型，初始值为 't2v'（文本到视频）
    task = gr.State(value="t2v")
    #主界面
    with gr.Row():
        with gr.Column():
            with gr.Row(): 
                with gr.Tabs():
                    # 文本到视频标签页
                    with gr.TabItem("文生视频"):
                        (   t2v_prompt, t2v_negative_prompt, t2v_input_image, t2v_input_video,
                            t2v_denoising_strength, t2v_seed, t2v_rand_device, t2v_resolution,
                            t2v_num_frames, t2v_cfg_scale, t2v_num_inference_steps, t2v_sigma_shift,
                            t2v_tiled, t2v_tile_size, t2v_tile_stride, output_fps, output_quality,
                            result_gallery, run_t2v_button, run_t2v_button_Disable,
                            open_folder_button, t2v_history
                        ) = create_t2v_ui()
                                
                           
                    # 图像到视频标签页
                    with gr.TabItem("图生视频"):
                        with gr.Row():
                            i2v_developing = gr.Textbox(show_label = False,label="开发中", value="施工中...", interactive=False)
                        
                    # 视频到视频标签页
                    with gr.TabItem("视频生视频"):
                        with gr.Row():
                            v2v_developing = gr.Textbox(show_label = False,label="开发中", value="施工中...", interactive=False)
                    # 视频到视频标签页

                    with gr.TabItem("模型管理"):
                        with gr.Row():
                            ModelManager()
                    with gr.TabItem("Wan-Video LoRA训练"):
                        with gr.Row():
                            Wan_LoRA_training = gr.Textbox(show_label = False,label="开发中", value="施工中...", interactive=False)    

   # 设置按钮事件
    t2v_button_events(
        run_t2v_button, run_t2v_button_Disable, t2v_prompt, t2v_negative_prompt, t2v_input_image, t2v_input_video,
        t2v_denoising_strength, t2v_seed, t2v_rand_device, t2v_resolution, t2v_num_frames, t2v_cfg_scale,
        t2v_num_inference_steps, t2v_sigma_shift, t2v_tiled, t2v_tile_size, t2v_tile_stride, output_fps, output_quality,
        result_gallery, t2v_history, open_folder_button
    )
    # 显示github上md文件的内容
    with gr.Row():
        with gr.Column():
            # 本地文件路径
            local_md_path = "Version.md"
            md_url = "https://raw.githubusercontent.com/qq1174565384/wan2.1_WebUI/refs/heads/main/apps/gradio/Wan/Version.md"
            try:
                # 尝试读取本地文件
                with open(local_md_path, 'r', encoding='utf-8') as file:
                    md_content = file.read()
            except FileNotFoundError:
                try:
                    # 本地文件不存在，尝试从网络下载
                    response = requests.get(md_url)
                    response.raise_for_status()
                    md_content = response.text
                except requests.RequestException as e:
                    md_content = f"无法下载 Markdown 文件: {e}"
            gr.Markdown(md_content) 

        with gr.Column():
            # 本地文件路径
            local_md_path = "developing.md"
            md_url = "https://raw.githubusercontent.com/qq1174565384/wan2.1_WebUI/refs/heads/main/apps/gradio/Wan/developing.md"
            try:
                # 尝试读取本地文件
                with open(local_md_path, 'r', encoding='utf-8') as file:
                    md_content = file.read()
            except FileNotFoundError:
                try:
                    # 本地文件不存在，尝试从网络下载
                    response = requests.get(md_url)
                    response.raise_for_status()
                    md_content = response.text
                except requests.RequestException as e:
                    md_content = f"无法下载 Markdown 文件: {e}"
            gr.Markdown(md_content) 

        with gr.Column():
            # 本地文件路径
            local_md_path = "contributor.md"
            md_url = "https://raw.githubusercontent.com/qq1174565384/wan2.1_WebUI/refs/heads/main/apps/gradio/Wan/contributor.md"
            try:
                # 尝试读取本地文件
                with open(local_md_path, 'r', encoding='utf-8') as file:
                    md_content = file.read()
            except FileNotFoundError:
                try:
                    # 本地文件不存在，尝试从网络下载
                    response = requests.get(md_url)
                    response.raise_for_status()
                    md_content = response.text
                except requests.RequestException as e:
                    md_content = f"无法下载 Markdown 文件: {e}"
            gr.Markdown(md_content) 
        with gr.Column():
                    # 本地文件路径
                    local_md_path = "help.md"
                    md_url = "https://raw.githubusercontent.com/qq1174565384/wan2.1_WebUI/refs/heads/main/apps/gradio/Wan/help.md"
                    try:
                        # 尝试读取本地文件
                        with open(local_md_path, 'r', encoding='utf-8') as file:
                            md_content = file.read()
                    except FileNotFoundError:
                        try:
                            # 本地文件不存在，尝试从网络下载
                            response = requests.get(md_url)
                            response.raise_for_status()
                            md_content = response.text
                        except requests.RequestException as e:
                            md_content = f"无法下载 Markdown 文件: {e}"
                    gr.Markdown(md_content) 
            
            

# 启动
demo.launch(inbrowser=True, allowed_paths=["../../../output/t2v"])
