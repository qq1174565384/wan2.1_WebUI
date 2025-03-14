
import os
import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download
import gradio as gr

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [
        os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-1.3B", "diffusion_pytorch_model.safetensors"),
        os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-1.3B", "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth"),
    ],
    torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

# 文字生成视频
text2video = pipe(
    prompt="",
    negative_prompt="",
    input_image=None,
    input_video=None,
    denoising_strength=1,
    seed=None,
    rand_device="cpu",
    height=480,
    width=832,
    num_frames=81,
    cfg_scale=5,
    num_inference_steps=50,
    sigma_shift=5,
    tiled=True,
    tile_size=(30, 52),
    tile_stride=(15, 26),
)
save_video(text2video, "video1.mp4", fps=15, quality=5)

# 定义一个函数来处理视频生成
def generate_video_from_text(prompt, negative_prompt, input_image, input_video, denoising_strength, seed, rand_device, height, width, num_frames, cfg_scale, num_inference_steps, sigma_shift, tiled, tile_size, tile_stride):
    # 将字符串转换为元组
    tile_size = eval(tile_size)
    tile_stride = eval(tile_stride)
    
    text2video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=input_image,
        input_video=input_video,
        denoising_strength=denoising_strength,
        seed=int(seed),
        rand_device=rand_device,
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        cfg_scale=cfg_scale,
        num_inference_steps=int(num_inference_steps),
        sigma_shift=sigma_shift,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    video_path = "video1.mp4"
    save_video(text2video, video_path, fps=15, quality=5)
    return video_path

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("### 文本到视频生成")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="提示词", value="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感四周草地的生机。中景侧面移动视角。")
            negative_prompt = gr.Textbox(label="负面提示词", value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
            input_image = gr.Image(label="输入图像", type="pil", value=None)
            input_video = gr.Video(label="输入视频", value=None)
            denoising_strength = gr.Slider(minimum=0, maximum=1, step=0.1, label="去噪强度", value=1)
            seed = gr.Number(label="随机种子", value=0)
            rand_device = gr.Textbox(label="随机数设备", value="cpu")
            height = gr.Number(label="视频高度", value=480)
            width = gr.Number(label="视频宽度", value=832)
            num_frames = gr.Number(label="帧数", value=81)
            cfg_scale = gr.Number(label="CFG 缩放因子", value=5)
            num_inference_steps = gr.Number(label="推理步数", value=50)
            sigma_shift = gr.Number(label="噪声偏移量", value=5)
            tiled = gr.Checkbox(label="是否分块生成", value=True)
            tile_size = gr.Textbox(label="分块大小", value="(30, 52)")
            tile_stride = gr.Textbox(label="分块步长", value="(15, 26)")
        with gr.Column():
            output_video = gr.Video(label="生成的视频")
            btn = gr.Button("生成视频")
            btn.click(generate_video_from_text, inputs=[prompt, negative_prompt, input_image, input_video, denoising_strength, seed, rand_device, height, width, num_frames, cfg_scale, num_inference_steps, sigma_shift, tiled, tile_size, tile_stride], outputs=output_video)

demo.launch(inbrowser=True)     