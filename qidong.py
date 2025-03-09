import os
import json
import gradio as gr
import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video
from modelscope import snapshot_download
from datetime import datetime
import random
import webbrowser

# 定义本地模型目录
local_dir = "models/Wan-AI/Wan2.1-T2V-1.3B"

# 需要检查/下载的模型文件列表
model_files = [
    "diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth"
]

# 检查所有模型文件是否存在于本地目录中
def check_models_exist(local_dir, model_files):
    return all(os.path.exists(os.path.join(local_dir, file)) for file in model_files)

# 如果模型不存在，则进行下载
if not check_models_exist(local_dir, model_files):
    print("正在下载模型...")
    snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir=local_dir)
else:
    print("模型已存在本地。")

# 加载模型
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [os.path.join(local_dir, file) for file in model_files],
    torch_dtype=torch.bfloat16,  # 可以设置 `torch_dtype=torch.float8_e4m3fn` 启用FP8量化。
)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

# 配置文件路径
config_file = "config.json"

# 默认配置
default_config = {
    "positive_prompt": "特写镜头|视频中，镜头面对一位动漫女仆的脸庞，柔和的光线洒在她的皮肤上，勾勒出细腻的轮廓，镜头缓缓环绕拉远，在废墟中展示出了她带血的全身，勾线动画。",
    "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    "num_inference_steps": 20,
    "seed": 574124451,
    "fps": 15,
    "quality": 5
}

# 如果配置文件存在则加载配置，否则使用默认配置
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
else:
    config = default_config


def generate_video_from_text(prompt, negative_prompt, num_inference_steps, seed, fps, quality):
    # 根据文本提示生成视频
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        seed=seed, tiled=True
    )

    # 创建输出目录（如果不存在）
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 基于当前时间戳生成唯一的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"Wan2.1_{timestamp}.mp4"
    video_path = os.path.join(output_dir, video_filename)

    # 保存生成的视频
    save_video(video, video_path, fps=fps, quality=quality)

    # 将参数保存到文本文件
    params = {
        "positive_prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "fps": fps,
        "quality": quality
    }
    param_filename = f"Wan2.1_{timestamp}.txt"
    param_path = os.path.join(output_dir, param_filename)

    with open(param_path, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

    return video_path


def update_config_and_generate_video(prompt, negative_prompt, num_inference_steps, seed, fps, quality):
    # 更新配置文件并生成视频
    new_config = {
        "positive_prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "fps": fps,
        "quality": quality
    }
    with open(config_file, 'w') as f:
        json.dump(new_config, f)

    return generate_video_from_text(prompt, negative_prompt, num_inference_steps, seed, fps, quality)


def reset_to_defaults():
    # 恢复默认参数
    return (
        default_config["positive_prompt"],
        default_config["negative_prompt"],
        default_config["num_inference_steps"],
        default_config["seed"],
        default_config["fps"],
        default_config["quality"]
    )


def list_history_videos():
    # 列出历史生成的视频
    output_dir = "output"
    history_files = []
    if os.path.exists(output_dir):
        for filename in sorted(os.listdir(output_dir)):
            if filename.endswith(".mp4"):
                video_path = os.path.join(output_dir, filename)
                param_path = os.path.join(output_dir, filename.replace(".mp4", ".txt"))
                with open(param_path, 'r') as f:
                    params = {}
                    for line in f:
                        key, value = line.strip().split(": ", 1)
                        params[key] = value
                history_files.append((video_path, params))
    return history_files


def load_params(params):
    # 加载参数
    return (
        params.get("positive_prompt", default_config["positive_prompt"]),
        params.get("negative_prompt", default_config["negative_prompt"]),
        int(params.get("num_inference_steps", default_config["num_inference_steps"])),
        int(params.get("seed", default_config["seed"])),
        int(params.get("fps", default_config["fps"])),
        int(params.get("quality", default_config["quality"]))
    )


def generate_random_seed():
    # 生成随机种子
    return random.randint(0, 2 ** 32 - 1)


def open_output_directory():
    # 打开输出目录
    output_dir = "output"
    if os.name == 'nt':  # Windows
        os.startfile(output_dir)
    elif os.name == 'posix':  # Linux or macOS
        os.system(f'open {output_dir}')


# 创建Gradio界面
with gr.Blocks() as iface:
    gr.Markdown("# Wan2.1文生视频1.3b青春版")
    gr.Markdown("更改提示词输出属于你的视频.")

    with gr.Row():
        with gr.Column(scale=1):  # 左列用于输入字段
            
            positive_prompt = gr.Textbox(lines=2, placeholder=default_config["positive_prompt"],
                                             label="正向提示词", value=config["positive_prompt"])
            
            negative_prompt = gr.Textbox(lines=2, value=config["negative_prompt"], label="反向提示词")

            with gr.Row():
                with gr.Column(scale=1): 
                     fps = gr.Slider(minimum=10, maximum=60, step=1, value=config["fps"], label="帧率(FPS)")
                with gr.Column(scale=1): 
                     quality = gr.Slider(minimum=1, maximum=10, step=1, value=config["quality"], label="生成质量")

            with gr.Row():
                with gr.Column(scale=1): 
                     num_inference_steps = gr.Slider(minimum=10, maximum=100, step=1,
                                                value=config["num_inference_steps"], label="采样步数",scale=2)
                     seed = gr.Number(value=config["seed"], label="随机种子",scale=1)
                     random_seed_button = gr.Button("random seed",scale=1)
                     random_seed_button.click(generate_random_seed, outputs=seed)

            with gr.Row():
                generate_button = gr.Button("文生视频")
                reset_button = gr.Button("恢复默认参数")

        with gr.Column(scale=1):  # 右列用于视频输出
            output_video = gr.Video(label="文生视频",height=616)
    gr.Markdown("# 其他功能")
    with gr.Row():
        show_history_button = gr.Button("展示生成历史")
        open_output_button = gr.Button("打开输出目录")

    history_output = gr.List([], label="生成历史")


    def create_history_item(video_path, params):
        # 创建历史记录项
        with gr.Column():
            video_component = gr.Video(value=video_path, label="Video")
            with gr.Row():
                for key, value in params.items():
                    gr.Markdown(f"**{key}**: {value}")
                reload_button = gr.Button("Reload Parameters")
                reload_button.click(load_params, inputs=[params], outputs=[
                    positive_prompt, negative_prompt, num_inference_steps, seed, fps, quality
                ])
            return gr.Column([video_component, reload_button])


    def update_history(history_items):
        # 更新历史记录
        history_list = []
        for video_path, params in history_items:
            history_item = create_history_item(video_path, params)
            history_list.append(history_item)
        return history_list


    show_history_button.click(list_history_videos, outputs=history_output).then(
        fn=update_history,
        inputs=[history_output],
        outputs=[history_output]
    )

    open_output_button.click(open_output_directory)

    generate_button.click(update_config_and_generate_video,
                          inputs=[positive_prompt, negative_prompt, num_inference_steps, seed, fps, quality],
                          outputs=output_video)

    reset_button.click(reset_to_defaults,
                       outputs=[positive_prompt, negative_prompt, num_inference_steps, fps, quality])

# 启动界面，不共享
app = iface.launch()

# 获取启动应用的本地URL
url = app.local_url

# 使用webbrowser模块打开浏览器并跳转到本地链接
try:
    webbrowser.open(url)
except Exception as e:
    print(f"Failed to open browser: {e}")


