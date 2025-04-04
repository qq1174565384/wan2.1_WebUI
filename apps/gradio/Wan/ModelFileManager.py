import gradio as gr
from modelscope import snapshot_download, dataset_snapshot_download
from diffsynth import download_models
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def ModelFileManager():
    # 定义重新检测函数
    def refresh_check():
        results = []
        model_ids = [
            "Wan-AI/Wan2.1-T2V-1.3B",
            "Wan-AI/Wan2.1-T2V-14B",
            "Wan-AI/Wan2.1-I2V-14B-480P",
            "Wan-AI/Wan2.1-I2V-14B-720P",
            "QwenPrompt/qwen2-1.5b-instruct",
            "Salesforce/blip-image-captioning-large"
        ]
        local_dirs = [
            os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-1.3B"),
            os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-14B"),
            os.path.join(project_root, "models", "Wan-AI", "Wan2.1-I2V-14B-480P"),
            os.path.join(project_root, "models", "Wan-AI", "Wan2.1-I2V-14B-720P"),
            os.path.join(project_root, "models", "QwenPrompt", "qwen2-1.5b-instruct"),
            os.path.join(project_root, "models", "Salesforce", "blip-image-captioning-large")

        ]
        def check_model_exist(model_path):
            return os.path.isdir(model_path)  # 检查文件夹是否存在
        
        status_values = []
        download_btn_updates = []
        open_btn_updates = []


        for local_dir in local_dirs:
            exist = check_model_exist(local_dir)
            status_values.append("已存在" if exist else "未存在")
            download_btn_updates.append(gr.update(
                value="文件夹已存在" if exist else "下载",
                interactive=not exist
            ))
            open_btn_updates.append(gr.update(
                interactive=exist
            ))

        # 合并所有需要返回的值
        return status_values + download_btn_updates + open_btn_updates

    with gr.Column():
        with gr.Row(equal_height=True):
            # 参数调节
            
            # Download models
            def check_model_exist(model_path):
                return os.path.isdir(model_path)  # 检查文件夹是否存在

            def download_model(model_id, local_dir):
                snapshot_download(model_id, local_dir=local_dir)
                return "已下载", gr.update(interactive=True), gr.update(interactive=True)
            
            def download_Qwen_model(model_id, local_dir):  
                os.chdir(project_root)
                download_models(["QwenPrompt"])
                return "已下载", gr.update(interactive=True), gr.update(interactive=True)
            def download_blip_model(model_id, local_dir):  
                os.chdir(project_root)
                local_model_path = os.path.join(project_root, 'models','Salesforce','blip-image-captioning-large')
                # 设置从清华镜像站下载模型
                mirror_url = "https://hf-mirror.com"
                try:
                    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", mirror=mirror_url)
                    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", mirror=mirror_url)
                    # 保存处理器和模型到本地
                    processor.save_pretrained(local_model_path)
                    model.save_pretrained(local_model_path)
                    return "已下载", gr.update(interactive=True), gr.update(interactive=True)
                except Exception as e:
                    print(f"从清华镜像站下载 Salesforce/blip-image-captioning-large 模型失败: {e}")
                    return "下载失败，请检查网络连接", gr.update(interactive=False), gr.update(interactive=False)

            def open_folder(local_dir):
                os.startfile(local_dir) if os.name == 'nt' else os.system(f'xdg-open {local_dir}')

            #t2v模型
            with gr.Column(variant="panel"):
                with gr.Row():  
                    model_t2v =  gr.HTML("""
                                            <div style="text-align: center; font-size: 20px; font-weight: bold; margin-bottom: 00px;">
                                                t2v模型
                                            </div>
                                            """)
                with gr.Row():      
                    with gr.Column(min_width=10):   
                        # 第一个模型 Wan-AI/Wan2.1-T2V-1.3B
                        model_id_1 = "Wan-AI/Wan2.1-T2V-1.3B"
                        # 移除多余逗号
                        local_dir_1 = os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-1.3B")
                        exist_1 = check_model_exist(local_dir_1)
                        status_text_1 = gr.Textbox(value="文件夹已存在" if exist_1 else "未存在", label=f"{model_id_1}", interactive=False)
                        download_btn_1 = gr.Button(value="下载", interactive=not exist_1,elem_id="button2")
                        open_btn_1 = gr.Button(value="打开所在文件夹", interactive=exist_1,elem_id="button2")
                        local_dir_state_1 = gr.State(value=local_dir_1)
                    
                        download_btn_1.click(
                            fn=download_model,
                            inputs=[gr.State(model_id_1), local_dir_state_1],
                            outputs=[status_text_1, download_btn_1, open_btn_1]
                        )
                    
                    with gr.Column(min_width=10):
                            # 第三个模型 Wan-AI/Wan2.1-T2V-14B
                            model_id_3 = "Wan-AI/Wan2.1-T2V-14B"
                            # 移除多余逗号和重复路径
                            local_dir_3 = os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-14B")
                            exist_3 = check_model_exist(local_dir_3)
                            status_text_3 = gr.Textbox(value="文件夹已存在" if exist_3 else "未存在", label=f"{model_id_3}", interactive=False)
                            download_btn_3 = gr.Button(value="下载", interactive=not exist_3,elem_id="button2")
                            open_btn_3 = gr.Button(value="打开所在文件夹", interactive=exist_3,elem_id="button2")
                            local_dir_state_3 = gr.State(value=local_dir_3)
                        
                            download_btn_3.click(
                                fn=download_model,
                                inputs=[gr.State(model_id_3), local_dir_state_3],
                                outputs=[status_text_3, download_btn_3, open_btn_3]
                            )
            #i2v模型
            with gr.Column(variant="panel"):
                with gr.Row():  
                    model_v2v =  gr.HTML("""
                                            <div style="text-align: center; font-size: 20px; font-weight: bold; margin-bottom: 00px;">
                                                i2v模型
                                            </div>
                                            """)
                with gr.Row():      
                    with gr.Column(min_width=10):
                        # 第二个模型 Wan-AI/Wan2.1-I2V-14B-480P
                        model_id_2 = "Wan-AI/Wan2.1-I2V-14B-480P"
                        # 移除多余逗号和重复路径
                        local_dir_2 = os.path.join(project_root, "models", "Wan-AI", "Wan2.1-I2V-14B-480P")
                        exist_2 = check_model_exist(local_dir_2)
                        status_text_2 = gr.Textbox(value="文件夹已存在" if exist_2 else "未存在", label=f"{model_id_2}", interactive=False)
                        download_btn_2 = gr.Button(value="下载", interactive=not exist_2,elem_id="button2")
                        open_btn_2 = gr.Button(value="打开所在文件夹", interactive=exist_2,elem_id="button2")
                        local_dir_state_2 = gr.State(value=local_dir_2)
                    
                        download_btn_2.click(
                            fn=download_model,
                            inputs=[gr.State(model_id_2), local_dir_state_2],
                            outputs=[status_text_2, download_btn_2, open_btn_2]
                        )

                    with gr.Column(min_width=10):
                        model_id_4 = "Wan-AI/Wan2.1-I2V-14B-720P"
                        # 移除多余逗号和重复路径
                        local_dir_4 = os.path.join(project_root, "models", "Wan-AI", "Wan2.1-I2V-14B-720P")
                        exist_4 = check_model_exist(local_dir_4)
                        status_text_4 = gr.Textbox(value="文件夹已存在" if exist_4 else "未存在", label=f"{model_id_4}", interactive=False)
                        download_btn_4 = gr.Button(value="下载", interactive=not exist_4,elem_id="button2")
                        open_btn_4 = gr.Button(value="打开所在文件夹", interactive=exist_4,elem_id="button2")
                        local_dir_state_4 = gr.State(value=local_dir_4)
                    
                        download_btn_4.click(
                            fn=download_model,
                            inputs=[gr.State(model_id_4), local_dir_state_4],
                            outputs=[status_text_4, download_btn_4, open_btn_4]
                        )
            #工具           
            with gr.Column(variant="panel"):
                with gr.Row():  
                    model_tools =  gr.HTML("""
                                            <div style="text-align: center; font-size: 20px; font-weight: bold; margin-bottom: 00px;">
                                                工具
                                            </div>
                                            """)
                with gr.Row():      
                    with gr.Column(min_width=10):
                        #千问模型
                        model_id_5 = "QwenPrompt/qwen2-1.5b-instruct"
                        # 移除多余逗号和重复路径
                        local_dir_5 = os.path.join(project_root, "models", "QwenPrompt", "qwen2-1.5b-instruct")
                        exist_5 = check_model_exist(local_dir_5)

                        status_text_5 = gr.Textbox(value="文件夹已存在" if exist_5 else "未存在", label=f"{model_id_5}  ", interactive=False)
                        download_btn_5 = gr.Button(value="下载", interactive=not exist_5,elem_id="button2")
                        open_btn_5 = gr.Button(value="打开所在文件夹", interactive=exist_5,elem_id="button2")
                        local_dir_state_5 = gr.State(value=local_dir_5)
                    
                        download_btn_5.click(
                            fn=download_Qwen_model,
                            inputs=[gr.State(model_id_5), local_dir_state_5],
                            outputs=[status_text_5, download_btn_5, open_btn_5]

                        )

                    with gr.Column(min_width=10):
                #千问模型
                        model_id_6 = "Salesforce/blip-image-captioning-large"
                        # 移除多余逗号和重复路径
                        local_dir_6 = os.path.join(project_root, "models", "Salesforce", "blip-image-captioning-large")
                        exist_6 = check_model_exist(local_dir_6)
                        status_text_6 = gr.Textbox(value="文件夹已存在" if exist_6 else "未存在", label=f"{model_id_6}   ", interactive=False)
                        download_btn_6 = gr.Button(value="下载", interactive=not exist_6,elem_id="button2")
                        open_btn_6 = gr.Button(value="打开所在文件夹", interactive=exist_6, elem_id="button2")
                        local_dir_state_6 = gr.State(value=local_dir_6)
                    
                        download_btn_6.click(
                            fn=download_blip_model,
                            inputs=[gr.State(model_id_6), local_dir_state_6],
                            outputs=[status_text_6, download_btn_6, open_btn_6]
                    
                        )
        with gr.Row():
            refresh_btn = gr.Button("重新检测", elem_id="button2")
            refresh_btn.click(fn=refresh_check, outputs=[
                    status_text_1, status_text_3, status_text_2, status_text_4, status_text_5,status_text_6,
                    download_btn_1, download_btn_3, download_btn_2, download_btn_4, download_btn_5,download_btn_6,
                    open_btn_1, open_btn_3, open_btn_2, open_btn_4, open_btn_5, open_btn_6
                ]
            )






        # 别忘了为每个打开按钮添加点击事件
        open_btn_1.click(fn=open_folder, inputs=[local_dir_state_1])
        open_btn_2.click(fn=open_folder, inputs=[local_dir_state_2])
        open_btn_3.click(fn=open_folder, inputs=[local_dir_state_3])
        open_btn_4.click(fn=open_folder, inputs=[local_dir_state_4])
        open_btn_5.click(fn=open_folder, inputs=[local_dir_state_5])
        open_btn_6.click(fn=open_folder, inputs=[local_dir_state_6])