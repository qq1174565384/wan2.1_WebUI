import gradio as gr

def create_t2v_ui():
    with gr.Row():
        # 参数调节
        with gr.Column():
            with gr.Row():
                t2v_ModelChoices = gr.Dropdown(scale=4,
                        label="模型选择",
                        choices=[
                            "Wan-AI/Wan2.1-T2V-1.3B", 
                            "Wan-AI/Wan2.1-T2V-14B"
                        ],
                        value = "Wan-AI/Wan2.1-T2V-1.3B",
                    )
                t2v_loadChoices = gr.Dropdown(scale=1,min_width=10,
                        label="模型管理设备",
                        choices=[
                            "CPU", 
                            "CUDA"
                        ],
                        value = "CPU",
                    )
            with gr.Row():
                # 定义一个文本框，用于输入文本到视频的提示词
                t2v_prompt = gr.Textbox(scale=4,
                    label="正面提示词",
                    value="特写镜头|视频中，镜头面对一位动漫女仆的脸庞，柔和的光线洒在她的皮肤上，勾勒出细腻的轮廓，镜头缓缓环绕拉远，在废墟中展示出了她带血的全身，勾线动画。",
                    placeholder="请输入提示词",
                    lines=5,
                )
                with gr.Column(scale=1,min_width=10):
                    # 定义按钮，用于提示词优化
                    t2v_prompt_reference_button = gr.Button("提示词参考", elem_id="button2")
                    t2v_prompt_refiner_button = gr.Button("提示词优化", elem_id="button2")
                    
            with gr.Row():
                t2v_negative_prompt = gr.Textbox(
                    label="负面提示词",
                    value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    placeholder="请输入负面提示词",
                    lines=5,
                )
            with gr.Row():
                # 定义一个下拉选择框，用于选择视频分辨率
                t2v_resolution = gr.Dropdown(
                    label="分辨率",
                    choices=[
                        "480*832", "832*480", "720*1280", 
                        "1280*720", "960*960", 
                        "1088*832", "832*1088"
                    ],
                    value="832*480",
                    allow_custom_value=True  # 允许用户输入自定义值
                )
                # 生成帧数
                t2v_num_frames = gr.Dropdown(label="生成帧数",choices=[
                        33, 81, 121, 
                        153, 201
                    ],
                     value=33,
                    allow_custom_value=True  # 允许用户输入自定义值
                    )
                t2v_denoising_strength = gr.Slider(visible=False, minimum=0, maximum=1, step=0.1, label="去噪强度", value=1, interactive=True, show_reset_button=False)     
            with gr.Row():  
                t2v_num_inference_steps = gr.Slider(minimum=1, maximum=75, step=1, label="迭代步数 (Steps)", value=25, interactive=True, show_reset_button=False)
            with gr.Row():   
                t2v_cfg_scale = gr.Slider(minimum=1, maximum=30, step=0.1, label="提示词引导系数 (CFG Scale)", value=5, interactive=True, show_reset_button=False)
                t2v_input_image = gr.Image(label="输入图像", type="pil", value=None, visible=False)
                t2v_input_video = gr.Video(label="输入视频", value=None, visible=False)
                t2v_rand_device = gr.Textbox(value="cuda", label="随机设备", placeholder="cuda or cpu", interactive=True, visible=False) 
                t2v_sigma_shift = gr.Slider(visible=False, label="Sigma Shift", value=5)
            
            with gr.Row(): 
                t2v_tile_size = gr.Textbox(visible=True, label="分块大小(tile_size)", value="(30, 52)", interactive=True)
                t2v_tile_stride = gr.Textbox(visible=True, label="分块步长(tile_stride)", value="(15, 26)", interactive=True)    
                t2v_tiled = gr.Checkbox(label="  分块生成（减少显存使用）", value=True, elem_id="custom-checkbox")   

            with gr.Row():     
                t2v_num_persistent_param_in_dit = gr.Slider(
                   minimum=0, maximum=64000000000, step=1000000000, label="num_persistent_param_in_dit(持久化参数量) ", value=0, interactive=True
                ) 

            with gr.Row():  
                with gr.Column(scale=1, min_width=1): 
                    t2v_seed = gr.Number(label="随机数种子 (Seed)", value=-1)
                with gr.Column(scale=1, min_width=1):
                    run_t2v_button = gr.Button("生 成", min_width=20, elem_id="button", visible=True)  
                    run_t2v_button_Disable = gr.Button("生 成 ing... ", min_width=20, elem_id="button_jinyong", visible=False)  

        # 显示生成的视频
        with gr.Column():
            with gr.Row():
                # 定义一个视频显示组件，用于显示生成的视频
                result_gallery = gr.Video(label='生成视频',
                                          interactive=False,
                                          height=525)
             
            with gr.Row():
                # 定义输出视频的参数fps，quality
                with gr.Column(min_width=1):
                    output_fps = gr.Slider(minimum=1, maximum=60, step=1, label="FPS", value=15)
                with gr.Column(min_width=1):
                    output_quality =  gr.Slider(minimum=1, maximum=10, step=1, label="保存质量", value=9)
                with gr.Column(min_width=1):
                    # 定义一个按钮，用于打开输出文件夹
                    t2v_open_folder_button = gr.Button("打开输出文件夹", elem_id="button2")       
        
            with gr.Row():
                # 添加参数展示文本框
                with gr.Column(min_width=100):
                    params_display = gr.Markdown(
                        """
                        - **ModelManagerdevice**：选择模型管理设备，支持CPU和CUDA。CPU更占内存，CUDA更占显存。
                        - **prompt**：指定生成视频内容的文本描述。
                        - **negative_prompt**：用于排除不希望出现在生成视频中的特征。比如指定“色调艳丽，过曝”等负面特征，让生成的视频避免出现这些情况。
                        - **seed**：随机种子，用于控制生成的随机性。设置相同的种子可以保证每次生成的结果一致，方便复现特定的生成效果。
                        - **num_frames**：生成视频的帧数，帧数越多，视频生成时间越长。
                        - **cfg_scale**：控制生成结果与提示词的匹配程度。
                        - **num_inference_steps**：推理步数，指定生成过程中的迭代次数。步数越多，生成的质量可能越高，但计算时间也会相应增加。
                        - **tile_size**：分块的大小，以元组形式表示，例如 (30, 52) 表示分块的高度和宽度。
                        - **tile_stride**：分块的步长，同样以元组形式表示，用于控制分块之间的重叠程度。
                        - **分块生成**：可以减少显存的使用，特别是在处理大尺寸图像或视频时。
                        - **num_persistent_param_in_dit**：设置持久化参数的大小，单位为字节。（越大越占显存，生成速度越快），如果显存爆了反而更慢。
                        """,
                        label="生成参数帮助",
                    )

    # 历史
    with gr.Row():
        from function.history import t2v_history_list
        t2v_history = gr.Examples(t2v_history_list,
                                  inputs=[t2v_prompt, result_gallery],
                                  outputs=[result_gallery],
                                  label="历史记录（仅显示提示词和视频，详细参数请打开输出文件夹查看txt文件)",
                                  examples_per_page = 5
                                  )
    
    return (
        t2v_prompt, 
        t2v_negative_prompt, 
        t2v_input_image, 
        t2v_input_video, 
        t2v_denoising_strength, 
        t2v_seed, 
        t2v_rand_device, 
        t2v_resolution, 
        t2v_num_frames, 
        t2v_cfg_scale, 
        t2v_num_inference_steps, 
        t2v_sigma_shift, 
        t2v_tiled, 
        t2v_tile_size, 
        t2v_tile_stride, 
        output_fps, 
        output_quality, 
        result_gallery, 
        run_t2v_button, 
        run_t2v_button_Disable, 
        t2v_open_folder_button, 
        t2v_history,
        t2v_prompt_reference_button,
        t2v_prompt_refiner_button,
        t2v_ModelChoices,
        t2v_num_persistent_param_in_dit,
        t2v_loadChoices
    )