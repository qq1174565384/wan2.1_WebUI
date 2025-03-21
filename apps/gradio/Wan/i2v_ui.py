import gradio as gr

def create_i2v_ui():
    with gr.Row():
        # 参数调节
        with gr.Column():
            with gr.Row():
                i2v_input_image = gr.Image(label="输入图像", type="pil", value=None, visible=True)
            with gr.Row():
                # 定义一个文本框，用于输入文本到视频的提示词
                i2v_prompt = gr.Textbox(
                    label="正面提示词",
                    value="",
                    placeholder="请输入提示词",
                    lines=5,
                )
            with gr.Row():
                i2v_negative_prompt = gr.Textbox(
                    label="负面提示词",
                    value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    placeholder="请输入负面提示词",
                    lines=5,
                )
            with gr.Row():
                # 定义一个下拉选择框，用于选择视频分辨率
                i2v_resolution = gr.Dropdown(
                    label="分辨率",
                    choices=[
                        "480*832", "832*480", "720*1280", 
                        "1280*720", "960*960", "720*1280",
                        "1088*832", "832*1088"
                    ],
                    value="832*480",
                    allow_custom_value=True  # 允许用户输入自定义值
                )
                # 生成帧数
                i2v_num_frames = gr.Dropdown(label="生成帧数",choices=[
                        33, 81, 121, 
                        153, 201
                    ],
                     value=33,
                    allow_custom_value=True  # 允许用户输入自定义值
                    )
                i2v_denoising_strength = gr.Slider(visible=False, minimum=0, maximum=1, step=0.1, label="去噪强度", value=1, interactive=True, show_reset_button=False)     
            with gr.Row():  
                i2v_num_inference_steps = gr.Slider(minimum=1, maximum=75, step=1, label="迭代步数 (Steps)", value=25, interactive=True, show_reset_button=False)
            with gr.Row():   
                i2v_cfg_scale = gr.Slider(minimum=1, maximum=30, step=0.1, label="提示词引导系数 (CFG Scale)", value=5, interactive=True, show_reset_button=False)
                i2v_input_image = gr.Image(label="输入图像", type="pil", value=None, visible=False)
                i2v_input_video = gr.Video(label="输入视频", value=None, visible=False)
                i2v_rand_device = gr.Textbox(value="cuda", label="随机设备", placeholder="cuda or cpu", interactive=True, visible=False) 
                i2v_sigma_shift = gr.Slider(visible=False, label="Sigma Shift", value=5)
            
            with gr.Row(): 
                i2v_tile_size = gr.Textbox(visible=True, label="分块大小(tile_size)", value="(30, 52)", interactive=True)
                i2v_tile_stride = gr.Textbox(visible=True, label="分块步长(tile_stride)", value="(15, 26)", interactive=True)    
                i2v_tiled = gr.Checkbox(label="  分块生成（减少显存使用）", value=True, elem_id="custom-checkbox")    
            with gr.Row():  
                with gr.Column(scale=1, min_width=1): 
                    i2v_seed = gr.Number(label="随机数种子 (Seed)", value=-1)
                with gr.Column(scale=1, min_width=1):
                    run_i2v_button = gr.Button("生 成", min_width=20, elem_id="button", visible=True)  
                    run_i2v_button_Disable = gr.Button("生 成 ing... ", min_width=20, elem_id="button_jinyong", visible=False)  

        # 显示生成的视频
        with gr.Column():
            with gr.Row():
                # 定义一个视频显示组件，用于显示生成的视频
                i2v_result_gallery = gr.Video(label='生成视频',
                                          interactive=False,
                                          height=525)
             
            with gr.Row():
                # 定义输出视频的参数fps，quality
                with gr.Column(min_width=1):
                    i2v_output_fps = gr.Slider(minimum=1, maximum=60, step=1, label="FPS", value=15)
                with gr.Column(min_width=1):
                    i2v_output_quality =  gr.Slider(minimum=1, maximum=10, step=1, label="保存质量", value=9)
                with gr.Column(min_width=1):
                    # 定义一个按钮，用于打开输出文件夹
                    i2v_open_folder_button = gr.Button("打开输出文件夹", elem_id="button2")       
        
            with gr.Row():
                # 添加参数展示文本框
                with gr.Column(min_width=100):
                    i2v_params_display = gr.Markdown(
                        """
                        - **input_image**：输入图像，用于生成视频的基础。
                        - **prompt**：指定生成视频内容的文本描述。
                        - **negative_prompt**：用于排除不希望出现在生成视频中的特征。比如指定“色调艳丽，过曝”等负面特征，让生成的视频避免出现这些情况。
                        - **seed**：随机种子，用于控制生成的随机性。设置相同的种子可以保证每次生成的结果一致，方便复现特定的生成效果。
                        - **num_frames**：生成视频的帧数，帧数越多，视频生成时间越长。
                        - **cfg_scale**：控制生成结果与提示词的匹配程度。
                        - **num_inference_steps**：推理步数，指定生成过程中的迭代次数。步数越多，生成的质量可能越高，但计算时间也会相应增加。
                        - **tile_size**：分块的大小，以元组形式表示，例如 (30, 52) 表示分块的高度和宽度。
                        - **tile_stride**：分块的步长，同样以元组形式表示，用于控制分块之间的重叠程度。
                        - **分块生成**：可以减少显存的使用，特别是在处理大尺寸图像或视频时。
                        """,
                        label="生成参数帮助",
                    )

    # 历史
    with gr.Row():
        from function.history import i2v_history_list
        i2v_history = gr.Examples(i2v_history_list,
                                  inputs=[i2v_prompt, i2v_result_gallery],
                                  outputs=[i2v_result_gallery],
                                  label="历史记录（仅显示提示词和视频，详细参数请打开输出文件夹查看txt文件)",
                                  examples_per_page = 5
                                  )
    
    return i2v_prompt, i2v_negative_prompt, i2v_input_image, i2v_input_video, i2v_denoising_strength, i2v_seed, i2v_rand_device, i2v_resolution, i2v_num_frames, i2v_cfg_scale, i2v_num_inference_steps, i2v_sigma_shift, i2v_tiled, i2v_tile_size, i2v_tile_stride, i2v_output_fps, i2v_output_quality, i2v_result_gallery, run_i2v_button, run_i2v_button_Disable, i2v_open_folder_button, i2v_history
