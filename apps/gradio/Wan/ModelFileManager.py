import gradio as gr
from modelscope import snapshot_download, dataset_snapshot_download
import os
# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def ModelFileManager():
    with gr.Row():
        # 参数调节
        # Download models
        def check_model_exist(model_path):
            return os.path.isdir(model_path)  # 检查文件夹是否存在

        def download_model(model_id, local_dir):
            snapshot_download(model_id, local_dir=local_dir)
            return "已下载", gr.Button.update(interactive=True), gr.Button.update(interactive=True)

        def open_folder(local_dir):
            os.startfile(local_dir) if os.name == 'nt' else os.system(f'xdg-open {local_dir}')

        with gr.Column():   
            # 第一个模型 Wan-AI/Wan2.1-T2V-1.3B
            model_id_1 = "Wan-AI/Wan2.1-T2V-1.3B"
            # 移除多余逗号
            local_dir_1 = os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-1.3B")
            exist_1 = check_model_exist(local_dir_1)
            status_text_1 = gr.Textbox(value="已存在" if exist_1 else "未存在", label=f"{model_id_1} (文生视频用）状态", interactive=False)
            download_btn_1 = gr.Button(value="下载", interactive=not exist_1)
            open_btn_1 = gr.Button(value="打开所在文件夹", interactive=exist_1)
            local_dir_state_1 = gr.State(value=local_dir_1)
        
            download_btn_1.click(
                fn=download_model,
                inputs=[gr.State(model_id_1), local_dir_state_1],
                outputs=[status_text_1, download_btn_1, open_btn_1]
            )
        
        with gr.Column():
            # 第三个模型 Wan-AI/Wan2.1-T2V-14B
            model_id_3 = "Wan-AI/Wan2.1-T2V-14B"
            # 移除多余逗号和重复路径
            local_dir_3 = os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-14B")
            exist_3 = check_model_exist(local_dir_3)
            status_text_3 = gr.Textbox(value="已存在" if exist_3 else "未存在", label=f"{model_id_3} (暂时没用）状态", interactive=False)
            download_btn_3 = gr.Button(value="下载", interactive=not exist_3)
            open_btn_3 = gr.Button(value="打开所在文件夹", interactive=exist_3)
            local_dir_state_3 = gr.State(value=local_dir_3)
        
            download_btn_3.click(
                fn=download_model,
                inputs=[gr.State(model_id_3), local_dir_state_3],
                outputs=[status_text_3, download_btn_3, open_btn_3]
            )
        with gr.Column():
            # 第二个模型 Wan-AI/Wan2.1-I2V-14B-480P
            model_id_2 = "Wan-AI/Wan2.1-I2V-14B-480P"
            # 移除多余逗号和重复路径
            local_dir_2 = os.path.join(project_root, "models", "Wan-AI", "Wan2.1-I2V-14B-480P")
            exist_2 = check_model_exist(local_dir_2)
            status_text_2 = gr.Textbox(value="已存在" if exist_2 else "未存在", label=f"{model_id_2} 图生视频用）状态", interactive=False)
            download_btn_2 = gr.Button(value="下载", interactive=not exist_2)
            open_btn_2 = gr.Button(value="打开所在文件夹", interactive=exist_2)
            local_dir_state_2 = gr.State(value=local_dir_2)
        
            download_btn_2.click(
                fn=download_model,
                inputs=[gr.State(model_id_2), local_dir_state_2],
                outputs=[status_text_2, download_btn_2, open_btn_2]
            )

        with gr.Column():
            # 第二个模型 Wan-AI/Wan2.1-I2V-14B-480P
            model_id_4 = "Wan-AI/Wan2.1-I2V-14B-720P"
            # 移除多余逗号和重复路径
            local_dir_4 = os.path.join(project_root, "models", "Wan-AI", "Wan2.1-I2V-14B-720P")
            exist_4 = check_model_exist(local_dir_4)
            status_text_4 = gr.Textbox(value="已存在" if exist_4 else "未存在", label=f"{model_id_4}  (暂时没用）状态", interactive=False)
            download_btn_4 = gr.Button(value="下载", interactive=not exist_4)
            open_btn_4 = gr.Button(value="打开所在文件夹", interactive=exist_4)
            local_dir_state_4 = gr.State(value=local_dir_4)
        
            download_btn_4.click(
                fn=download_model,
                inputs=[gr.State(model_id_4), local_dir_state_4],
                outputs=[status_text_4, download_btn_4, open_btn_4]
            )





        # 别忘了为每个打开按钮添加点击事件
        open_btn_1.click(fn=open_folder, inputs=[local_dir_state_1])
        open_btn_2.click(fn=open_folder, inputs=[local_dir_state_2])
        open_btn_3.click(fn=open_folder, inputs=[local_dir_state_3])
        open_btn_4.click(fn=open_folder, inputs=[local_dir_state_4])