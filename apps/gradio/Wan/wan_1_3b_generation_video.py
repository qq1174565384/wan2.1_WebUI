
import os
import subprocess
import gradio as gr
from examples import t2v_examples, i2v_examples
from function.video_generation import generate_video_from_text, generate_video_from_image, generate_video_from_video
from function.open_output_folder import open_output_folder

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# # 定义css样式
custom_css = """
/* 按钮样式 */
#button {
    background-color: #3938c7; /* 暗蓝色背景 */
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

#button2 {
    background-color: #3938c7; /* 暗蓝色背景 */
    height: 78px; /* 设置按钮高度 */
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
    border: 2px solid #3938c7; /* 边框颜色 */
    margin-top: 00px; /* 上边距 */
}

#custom-checkbox input[type="checkbox"]:checked {
    background-color: #3938c7; /* 选中时的背景颜色 */
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
    border: solid #f8f9fa; /* 对勾的颜色 */
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
with gr.Blocks(css=custom_css) as demo:
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
                    with gr.TabItem("文生视频") as t2v_tab:
                        with gr.Row():
                            #参数调节
                            with gr.Column(): 
                                with gr.Row():
                                    # 定义一个文本框，用于输入文本到视频的提示词
                                    t2v_prompt = gr.Textbox(
                                        label="正面提示词",
                                        value="",
                                        placeholder="请输入提示词",
                                        lines=5,
                                    )
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
                                            "1280*720", "960*960", "720*1280",
                                            "1088*832", "832*1088"
                                        ],
                                        value="832*480",
                                        allow_custom_value=True  # 允许用户输入自定义值
                                    )
                                    # 生成帧数
                                    t2v_num_frames = gr.Number(label="生成帧数", value=33)
                                    t2v_denoising_strength = gr.Slider(visible=False,minimum=0, maximum=1, step=0.1, label="去噪强度", value=1,interactive=True,show_reset_button=False)     
                                with gr.Row():  
                                    t2v_num_inference_steps = gr.Slider(minimum=1, maximum=75, step=1,label="迭代步数 (Steps)", value=25,interactive=True,show_reset_button=False)
                                with gr.Row():   
                                
                                    t2v_cfg_scale = gr.Slider(minimum=1, maximum=30, step=0.1,label="提示词引导系数 (CFG Scale)", value=5,interactive=True,show_reset_button=False)
                                    t2v_input_image = gr.Image(label="输入图像", type="pil", value=None,visible=False)
                                    t2v_input_video = gr.Video(label="输入视频", value=None,visible=False)
                                    t2v_rand_device = gr.Textbox(value="cuda",label="随机设备",placeholder="cuda or cpu",interactive=True,visible=False) 
                                    t2v_sigma_shift = gr.Slider(visible=False,label="Sigma Shift", value=5)
                                with gr.Row():  
                                    with gr.Column(scale=1,min_width=1): 
                                        t2v_seed = gr.Number(label="随机数种子 (Seed)", value=-1)
                                    with gr.Column(scale=1,min_width=1):
                                        run_t2v_button = gr.Button("生 成",min_width=20,elem_id="button")
                                with gr.Row(): 
                                    t2v_tile_size = gr.Textbox(visible=True,label="分块大小(tile_size)", value="(30, 52)",interactive=True)
                                    t2v_tile_stride = gr.Textbox(visible=True,label="分块步长(tile_stride)", value="(15, 26)",interactive=True)    
                                    t2v_tiled = gr.Checkbox(label="  分块生成（减少显存使用）", value=True,elem_id="custom-checkbox")    
                            #显示生成的视频
                            with gr.Column():

                                with gr.Row():
                                    # 定义一个视频显示组件，用于显示生成的视频
                                    result_gallery = gr.Video(label='Generated Video',
                                                                interactive=False,
                                                                height=532)
                                
                                with gr.Row():
                                    # 定义输出视频的参数fps，quality
                                    with gr.Column(min_width=1):
                                        output_fps = gr.Slider(minimum=1, maximum=60, step=1,label="FPS", value=15)
                                    with gr.Column(min_width=1):
                                        output_quality =  gr.Slider(minimum=1, maximum=10, step=1,label="保存质量", value=9)
                                    with gr.Column(min_width=1):
                                        # 定义一个按钮，用于打开输出文件夹
                                        open_folder_button = gr.Button("打开输出文件夹",elem_id="button2")       
                                    
                                with gr.Row():
                                    # 添加参数展示文本框
                                    # 添加参数展示文本框
                                    with gr.Column(min_width=100):
                                        params_display = gr.Markdown(
                                            """
                                            - **prompt**：指定生成视频内容的文本描述。
                                            - **negative_prompt**：用于排除不希望出现在生成视频中的特征。比如指定“色调艳丽，过曝”等负面特征，让生成的视频避免出现这些情况。
                                            - **denoising_strength**：去噪强度，值越大，对输入（图像或视频）。
                                            - **seed**：随机种子，用于控制生成的随机性。设置相同的种子可以保证每次生成的结果一致，方便复现特定的生成效果。
                                            - **num_frames**：生成视频的帧数，默认是 81 帧，帧数越多，视频时长可能越长。
                                            - **cfg_scale**：控制生成结果与提示词的匹配程度。
                                            - **num_inference_steps**：推理步数，指定生成过程中的迭代次数。步数越多，生成的质量可能越高，但计算时间也会相应增加。
                                            - **tile_size**：分块的大小，以元组形式表示，例如 (30, 52) 表示分块的高度和宽度。
                                            - **tile_stride**：分块的步长，同样以元组形式表示，用于控制分块之间的重叠程度。
                                            - **分块生成**：可以减少显存的使用，特别是在处理大尺寸图像或视频时。
                                            """,
                                            label="生成参数帮助",
                                )
                                        
                    # 图像到视频标签页
                    with gr.TabItem("图生视频") as i2v_tab:
                        with gr.Row():    
                           i2v_input_image = gr.Image(label="输入图像", type="pil", value=None)
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
                            i2v_num_frames = gr.Number(label="生成帧数", value=33)
                            i2v_denoising_strength = gr.Slider(minimum=0, maximum=1, step=0.1, label="去噪强度", value=1,interactive=True,show_reset_button=False)
                        
                            
                            
                        with gr.Row():  
                            i2v_num_inference_steps = gr.Slider(minimum=1, maximum=75, step=1,label="迭代步数 (Steps)", value=25,interactive=True,show_reset_button=False)
                        with gr.Row():   
                        
                            i2v_cfg_scale = gr.Slider(minimum=1, maximum=30, step=0.1,label="提示词引导系数 (CFG Scale)", value=5,interactive=True,show_reset_button=False)


                            i2v_input_video = gr.Video(label="输入视频", value=None,visible=False)
                            i2v_rand_device = gr.Textbox(value="cuda",label="随机设备",placeholder="cuda or cpu",interactive=True,visible=False)  
                            i2v_sigma_shift = gr.Slider(visible=False,label="Sigma Shift", value=5)
                        with gr.Row():  
                            with gr.Column(scale=1,min_width=1): 
                                i2v_seed = gr.Number(label="随机数种子 (Seed)", value=-1)
                            with gr.Column(scale=1,min_width=1):
                                run_i2v_button = gr.Button("生 成",min_width=20,elem_id="button")
                        with gr.Row(): 
                            i2v_tile_size = gr.Textbox(visible=True,label="分块大小(tile_size)", value="(30, 52)",interactive=True)
                            i2v_tile_stride = gr.Textbox(visible=True,label="分块步长(tile_stride)", value="(15, 26)",interactive=True)    
                            i2v_tiled = gr.Checkbox(label="  分块生成（减少显存使用）", value=True,elem_id="custom-checkbox")    
                   
                    # 视频到视频标签页
                    with gr.TabItem("视频生视频") as v2v_tab:
                        with gr.Row():    
                            v2v_input_video = gr.Video(label="输入视频", value=None)
                        with gr.Row():
                            # 定义一个文本框，用于输入文本到视频的提示词
                           v2v_prompt = gr.Textbox(
                                label="正面提示词",
                                value="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感四周草地的生机。中景侧面移动视角。",
                                placeholder="请输入提示词",
                                lines=5,
                            )
                        with gr.Row():
                            v2v_negative_prompt = gr.Textbox(
                                label="负面提示词",
                                value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                                placeholder="请输入负面提示词",
                                lines=5,
                            )
                        with gr.Row():
                            # 定义一个下拉选择框，用于选择视频分辨率
                            v2v_resolution = gr.Dropdown(
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
                            v2v_num_frames = gr.Number(label="生成帧数", value=33)
                            v2v_denoising_strength = gr.Slider(minimum=0, maximum=1, step=0.1, label="去噪强度", value=1,interactive=True,show_reset_button=False)
                        
                            
                        with gr.Row():  
                            v2v_num_inference_steps = gr.Slider(minimum=1, maximum=75, step=1,label="迭代步数 (Steps)", value=25,interactive=True,show_reset_button=False)
                        with gr.Row():   
                        
                            v2v_cfg_scale = gr.Slider(minimum=1, maximum=30, step=0.1,label="提示词引导系数 (CFG Scale)", value=5,interactive=True,show_reset_button=False)

                            v2v_input_image = gr.Image(label="输入图像", type="pil", value=None,visible=False)
                            
                            v2v_rand_device=gr.Textbox(value="cuda",label="随机设备",placeholder="cuda or cpu",interactive=True,visible=False)
                            v2v_sigma_shift = gr.Slider(visible=False,label="Sigma Shift", value=5)
                        with gr.Row():  
                            with gr.Column(scale=1,min_width=1): 
                                v2v_seed = gr.Number(label="随机数种子 (Seed)", value=-1)
                            with gr.Column(scale=1,min_width=1):
                                run_v2v_button = gr.Button("生 成",min_width=20,elem_id="button")
                        with gr.Row(): 
                            v2v_tile_size = gr.Textbox(visible=True,label="分块大小(tile_size)", value="(30, 52)",interactive=True)
                            v2v_tile_stride = gr.Textbox(visible=True,label="分块步长(tile_stride)", value="(15, 26)",interactive=True)    
                            v2v_tiled = gr.Checkbox(label="  分块生成（减少显存使用）", value=True,elem_id="custom-checkbox")           

        

    # # 定义一个隐藏的视频组件，用于示例展示
    # fake_video = gr.Video(label='Examples', visible=False, interactive=False)
    # 定义一个可见的行组件，用于显示文本到视频的示例
    with gr.Row(visible=True) as t2v_eg:
        gr.Examples(t2v_examples,
                    inputs=[t2v_prompt, result_gallery],
                    outputs=[result_gallery])

    # 定义一个隐藏的行组件，用于显示图像到视频的示例
    with gr.Row(visible=False) as i2v_eg:
        gr.Examples(i2v_examples,
                    inputs=[i2v_prompt, i2v_input_image, result_gallery],
                    outputs=[result_gallery])
    # 定义一个隐藏的行组件，用于显示图像到视频的示例
    with gr.Row(visible=False) as v2v_eg:
        gr.Examples(i2v_examples,visible=False,
                    inputs=[v2v_prompt, v2v_input_video, result_gallery],
                    outputs=[result_gallery])
        

    # 通用的标签页切换函数
    def switch_tab(t2v_visible, i2v_visible, v2v_visible, task_mode):
        """
        通用的标签页切换函数
        :param t2v_visible: 文本到视频示例行的可见性
        :param i2v_visible: 图像到视频示例行的可见性
        :param v2v_visible: 视频到视频示例行的可见性
        :param task_mode: 当前任务模式
        :return: 文本到视频示例行、图像到视频示例行、视频到视频示例行和任务状态
        """
        return gr.Row(visible=t2v_visible), gr.Row(visible=i2v_visible), gr.Row(visible=v2v_visible), task_mode
    
    # 切换到文本到视频（Text to Video）标签页时调用此函数
    def switch_t2v_tab():
        """
        切换到文本到视频（Text to Video）标签页时调用此函数
        显示文本到视频示例行，隐藏图像到视频和视频到视频示例行，并将任务状态设置为 't2v'
        """
        return switch_tab(True, False, False, "t2v")
    
    # 切换到图像到视频（Image to Video）标签页时调用此函数
    def switch_i2v_tab():
        """
        切换到图像到视频（Image to Video）标签页时调用此函数
        隐藏文本到视频和视频到视频示例行，显示图像到视频示例行，并将任务状态设置为 'i2v'
        """
        return switch_tab(False, True, False, "i2v")
    
    # 切换到视频到视频（Video to Video）标签页时调用此函数
    def switch_v2v_tab():
        """
        切换到视频到视频（Video to Video）标签页时调用此函数
        隐藏文本到视频和图像到视频示例行，显示视频到视频示例行，并将任务状态设置为 'v2v'
        """
        return switch_tab(False, False, True, "v2v")

    # 当用户选择图像到视频标签页时，调用 switch_i2v_tab 函数更新界面
    i2v_tab.select(switch_i2v_tab, outputs=[t2v_eg, i2v_eg, v2v_eg, task])
    # 当用户选择文本到视频标签页时，调用 switch_t2v_tab 函数更新界面
    t2v_tab.select(switch_t2v_tab, outputs=[t2v_eg, i2v_eg, v2v_eg, task])
    # 当用户选择视频到视频标签页时，调用 switch_v2v_tab 函数更新界面
    v2v_tab.select(switch_v2v_tab, outputs=[t2v_eg, i2v_eg, v2v_eg, task])



    
    # 修改生成按钮的点击事件
    run_t2v_button.click(
        fn = generate_video_from_text,
        inputs=[
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
            output_quality
        ],
        outputs=result_gallery  # 只输出视频
    )

    # 为其他按钮添加参数保存功能
    run_i2v_button.click(
        fn = generate_video_from_image,
        inputs=[
            i2v_prompt,
            i2v_negative_prompt,
            i2v_input_image,
            i2v_input_video,
            i2v_denoising_strength,
            i2v_seed,
            i2v_rand_device,
            i2v_resolution,
            i2v_num_frames,
            i2v_cfg_scale,
            i2v_num_inference_steps,
            i2v_sigma_shift,
            i2v_tiled,
            i2v_tile_size,
            i2v_tile_stride,
            output_fps,
            output_quality
        ],
        outputs=result_gallery  # 只输出视频
    )

    run_v2v_button.click(
        fn=generate_video_from_video,
        inputs=[
            v2v_prompt,
            v2v_negative_prompt,
            v2v_input_image,
            v2v_input_video,
            v2v_denoising_strength,
            v2v_seed,
            v2v_rand_device,
            v2v_resolution,
            v2v_num_frames,
            v2v_cfg_scale,
            v2v_num_inference_steps,
            v2v_sigma_shift,
            v2v_tiled,
            v2v_tile_size,
            v2v_tile_stride,
            output_fps,
            output_quality
        ],
        outputs=result_gallery  # 只输出视频
    )


    # 绑定按钮点击事件
    open_folder_button.click(
        fn=open_output_folder
    )

# 启动
demo.launch(inbrowser=True,quiet=True)

