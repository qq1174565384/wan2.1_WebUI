import os
import json
import gradio as gr
import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video
from modelscope import snapshot_download
from datetime import datetime
import subprocess

# Define the local directory for models
local_dir = "models/Wan-AI/Wan2.1-T2V-1.3B"

# List of model files to check/download
model_files = [
    "diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth"
]

# Function to check if all model files exist in the local directory
def check_models_exist(local_dir, model_files):
    return all(os.path.exists(os.path.join(local_dir, file)) for file in model_files)

# Download models only if they do not exist
if not check_models_exist(local_dir, model_files):
    print("Downloading models...")
    snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir=local_dir)
else:
    print("Models already exist locally.")

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [os.path.join(local_dir, file) for file in model_files],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

# Configuration file path
config_file = "config.json"

# Default configuration
default_config = {
    "positive_prompt": "特写镜头|视频中，镜头面对一位动漫女仆的脸庞，柔和的光线洒在她的皮肤上，勾勒出细腻的轮廓，镜头缓缓环绕拉远，在废墟中展示出了她带血的全身，勾线动画。",
    "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    "num_inference_steps": 20,
    "seed": -1,
    "fps": 15,
    "quality": 5
}

# Load configuration from file if it exists
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
else:
    config = default_config

def generate_video_from_text(prompt, negative_prompt, num_inference_steps, seed, fps, quality):
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        seed=seed, tiled=True
    )
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate unique filename based on current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"Wan2.1_{timestamp}.mp4"
    video_path = os.path.join(output_dir, video_filename)
    
    save_video(video, video_path, fps=fps, quality=quality)
    
    # Save parameters to a text file
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

# Create Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Text-to-Video Generator")
    gr.Markdown("Generate a video based on a text prompt with adjustable parameters.")
    
    with gr.Row():
        positive_prompt = gr.Textbox(lines=2, placeholder=config["positive_prompt"], label="Positive Prompt")
        negative_prompt = gr.Textbox(lines=2, value=config["negative_prompt"], label="Negative Prompt")
    
    with gr.Row():
        num_inference_steps = gr.Slider(minimum=10, maximum=100, step=1, value=config["num_inference_steps"], label="Number of Inference Steps")
        seed = gr.Number(value=config["seed"], label="Seed")
    
    with gr.Row():
        fps = gr.Slider(minimum=10, maximum=60, step=1, value=config["fps"], label="Frames Per Second (FPS)")
        quality = gr.Slider(minimum=1, maximum=10, step=1, value=config["quality"], label="Quality")
    
    generate_button = gr.Button("Generate Video")
    output_video = gr.Video(label="Generated Video")
    
    generate_button.click(update_config_and_generate_video, 
                          inputs=[positive_prompt, negative_prompt, num_inference_steps, seed, fps, quality], 
                          outputs=output_video)

# Launch the interface without sharing
app = iface.launch()

# Get the URL of the launched app
url = app.local_url

# Open the browser with the URL using subprocess (Windows specific)
try:
    subprocess.Popen(['start', url], shell=True)
except Exception as e:
    print(f"Failed to open browser: {e}")



