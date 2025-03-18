import os
import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download
import random
import ast
from datetime import datetime


# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# 加载模型
model_paths = [
    os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-1.3B", "diffusion_pytorch_model.safetensors"),
    os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-1.3B", "models_t5_umt5-xxl-enc-bf16.pth"),
    os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth"),
]

for path in model_paths:
    if not os.path.exists(path):
        print(f"模型文件 {path} 不存在，请检查路径。")
        raise FileNotFoundError(f"模型文件 {path} 不存在，请检查路径。")

model_manager = ModelManager(device="cuda")
try:
    model_manager.load_models(
        model_paths,
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
except Exception as e:
    print(f"模型加载失败: {e}")
    # 可以根据具体情况进行进一步处理，例如退出程序或重试
    raise

# 创建管道
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)


# 定义t2v函数
def generate_video_from_text(
    t2v_prompt="",  # 设置默认值为空字符串,
    t2v_negative_prompt="",  # 设置默认值为空字符串
    t2v_input_image=None,  # 设置默认值为 None
    t2v_input_video=None,  # 设置默认值为 None
    t2v_denoising_strength=1,  # 设置默认值为 1
    t2v_seed=-1,  # 设置默认值为 -1
    t2v_rand_device="cuda",  # 设置默认值为 "cuda"
    t2v_resolution="832*480",  # 设置默认值为 "1280*720"
    t2v_num_frames=81,  # 设置默认值为 81
    t2v_cfg_scale=5,  # 设置默认值为 5
    t2v_num_inference_steps=50,  # 设置默认值为 50
    t2v_sigma_shift=5,  # 设置默认值为 5
    t2v_tiled=True,  # 设置默认值为 True
    t2v_tile_size="(30, 52)",  # 设置默认值为 "(30, 52)"
    t2v_tile_stride="(15, 26)",  # 设置默认值为 "(15, 26)"
    output_fps=15,
    output_quality=9,
):
    t2v_seed = t2v_seed if t2v_seed >= 0 else random.randint(0, 2147483647)
    # 确保 t2v_resolution 是字符串类型
    t2v_resolution = str(t2v_resolution)
   
    
    t2v_width, t2v_height = map(int, t2v_resolution.split('*'))
    
    # 使用 ast.literal_eval 替代 eval
    t2v_tile_size = ast.literal_eval(t2v_tile_size)
    t2v_tile_stride = ast.literal_eval(t2v_tile_stride)
    print("正在生成视频...")
    try:
        t2v = pipe(
            prompt=t2v_prompt,
            negative_prompt=t2v_negative_prompt,
            input_image=t2v_input_image,
            input_video=t2v_input_video,
            denoising_strength=t2v_denoising_strength,
            seed=int(t2v_seed),
            rand_device=t2v_rand_device,
            height=int(t2v_height),
            width=int(t2v_width),
            num_frames=int(t2v_num_frames),
            cfg_scale=t2v_cfg_scale,
            num_inference_steps=int(t2v_num_inference_steps),
            sigma_shift=t2v_sigma_shift,
            tiled=t2v_tiled,
            tile_size=t2v_tile_size,
            tile_stride=t2v_tile_stride,
        )
    except Exception as e:
        print(f"视频生成失败: {e}")
        t2v = None
    # Create output directory if it doesn't exist
    output_dir = os.path.join(project_root,"output", "t2v")  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate unique filename based on current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"Wan2_1_i2v_{timestamp}.mp4"
    video_path = os.path.join(output_dir, video_filename)
    print("正在保存视频...")
    save_video(t2v, video_path, fps=output_fps, quality=output_quality)
    print("正在保存预览图...")
    # Save preview image
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preview_image = Image.fromarray(frame)
        preview_path = os.path.join(output_dir, f"{os.path.splitext(video_filename)[0]}.jpg")
        preview_image.save(preview_path)
    print("正在保存参数文件...")
    # Save parameters to a text file
    params = {
                "prompt": t2v_prompt,
                "negative_prompt": t2v_negative_prompt,
                "input_image": t2v_input_image,
                "input_video": t2v_input_video,
                "denoising_strength": t2v_denoising_strength,
                "seed": t2v_seed,
                "rand_device": t2v_rand_device,
                "height": t2v_height,
                "width": t2v_width,
                "num_frames": t2v_num_frames,
                "cfg_scale": t2v_cfg_scale,
                "num_inference_steps": t2v_num_inference_steps,
                "sigma_shift": t2v_sigma_shift,
                "tiled": t2v_tiled,
                "tile_size": t2v_tile_size,
                "tile_stride": t2v_tile_stride,
    }
    param_filename = f"Wan2_1_i2v_{timestamp}.txt"
    param_path = os.path.join(output_dir, param_filename)
    
    with open(param_path, 'w', encoding='utf-8') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print("任务成功")
    return video_path

# 定义i2v函数
def generate_video_from_image(
    i2v_prompt,
    i2v_negative_prompt,
    i2v_input_image=None,  # 设置默认值为 None
    i2v_input_video=None,  # 设置默认值为 None
    i2v_denoising_strength=1,  # 设置默认值为 1
    i2v_seed=-1,  # 设置默认值为 -1
    i2v_rand_device="cuda",  # 设置默认值为 "cuda"
    i2v_resolution="832*480",  # 设置默认值为 "1280*720"
    i2v_num_frames=81,  # 设置默认值为 81
    i2v_cfg_scale=5,  # 设置默认值为 5
    i2v_num_inference_steps=50,  # 设置默认值为 50
    i2v_sigma_shift=5,  # 设置默认值为 5
    i2v_tiled=True,  # 设置默认值为 True
    i2v_tile_size="(30, 52)",  # 设置默认值为 "(30, 52)"
    i2v_tile_stride="(15, 26)",  # 设置默认值为 "(15, 26)"
    output_fps=15,
    output_quality=9
    ):
    i2v_seed = i2v_seed if i2v_seed >= 0 else random.randint(0, 2147483647)
    # 确保 i2v_resolution 是字符串类型
    i2v_resolution = str(i2v_resolution)
    # 解析分辨率
    try:
        i2v_width, i2v_height = map(int, i2v_resolution.split('*'))
    except (ValueError, IndexError):
        i2v_width, i2v_height = 1280, 720  # 使用默认值
    # 使用 ast.literal_eval 替代 eval
    i2v_tile_size = ast.literal_eval(i2v_tile_size)
    i2v_tile_stride = ast.literal_eval(i2v_tile_stride)

    try:
        i2v = pipe(
            prompt=i2v_prompt,
            negative_prompt=i2v_negative_prompt,
            input_image=i2v_input_image,
            input_video=i2v_input_video,
            denoising_strength=i2v_denoising_strength,
            seed=int(i2v_seed),
            rand_device=i2v_rand_device,
            height=int(i2v_height),
            width=int(i2v_width),
            num_frames=int(i2v_num_frames),
            cfg_scale=i2v_cfg_scale,
            num_inference_steps=int(i2v_num_inference_steps),
            sigma_shift=i2v_sigma_shift,
            tiled=i2v_tiled,
            tile_size=i2v_tile_size,
            tile_stride=i2v_tile_stride,
        )
    except Exception as e:
        print(f"视频生成失败: {e}")
        i2v = None
    # Create output directory if it doesn't exist
    output_dir = os.path.join(project_root,"output", "i2v")  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate unique filename based on current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"Wan2_1_i2v_{timestamp}.mp4"
    video_path = os.path.join(output_dir, video_filename)
    
    save_video(i2v, video_path, fps=output_fps, quality=output_quality)

    # Save preview image
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preview_image = Image.fromarray(frame)
        preview_path = os.path.join(output_dir, f"{os.path.splitext(video_filename)[0]}.jpg")
        preview_image.save(preview_path)

    # 保存参数到文本文件
    params = { 
                "prompt": i2v_prompt,
                "negative_prompt": i2v_negative_prompt,
                "input_image": i2v_input_image,
                "input_video": i2v_input_video,
                "denoising_strength": i2v_denoising_strength,
                "seed": i2v_seed,
                "rand_device": i2v_rand_device,
                "height": i2v_height,
                "width": i2v_width,
                "num_frames": i2v_num_frames,
                "cfg_scale": i2v_cfg_scale,
                "num_inference_steps": i2v_num_inference_steps,
                "sigma_shift": i2v_sigma_shift,
                "tiled": i2v_tiled,
                "tile_size": i2v_tile_size,
                "tile_stride": i2v_tile_stride,
    }
    param_filename = f"Wan2_1_i2v_{timestamp}.txt"
    param_path = os.path.join(output_dir, param_filename)
    
    with open(param_path, 'w', encoding='utf-8') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

    
   
    return video_path



# 定义v2v函数
def generate_video_from_video(
    v2v_prompt,
    v2v_negative_prompt,
    v2v_input_image=None,  # 设置默认值为 None
    v2v_input_video=None,  # 设置默认值为 None
    v2v_denoising_strength=1,  # 设置默认值为 1
    v2v_seed=-1,  # 设置默认值为 -1
    v2v_rand_device="cuda",  # 设置默认值为 "cuda"
    v2v_resolution="832*480",  # 设置默认值为 "1280*720"
    v2v_num_frames=81,  # 设置默认值为 81
    v2v_cfg_scale=5,  # 设置默认值为 5
    v2v_num_inference_steps=50,  # 设置默认值为 50
    v2v_sigma_shift=5,  # 设置默认值为 5
    v2v_tiled=True,  # 设置默认值为 True
    v2v_tile_size="(30, 52)",  # 设置默认值为 "(30, 52)"
    v2v_tile_stride="(15, 26)", # 设置默认值为 "(15, 26)"
    output_fps=15,
    output_quality=9,
):
    v2v_seed = v2v_seed if v2v_seed >= 0 else random.randint(0, 2147483647)
    # 确保 v2v_resolution 是字符串类型
    v2v_resolution = str(v2v_resolution)
    # 解析分辨率
    try:
        v2v_width, v2v_height = map(int, v2v_resolution.split('*'))
    except (ValueError, IndexError):
        v2v_width, v2v_height = 1280, 720  # 使用默认值
    # 使用 ast.literal_eval 替代 eval
    v2v_tile_size = ast.literal_eval(v2v_tile_size)
    v2v_tile_stride = ast.literal_eval(v2v_tile_stride)
    try:
        v2v = pipe(
            prompt=v2v_prompt,
            negative_prompt=v2v_negative_prompt,
            input_image=v2v_input_image,
            input_video=v2v_input_video,
            denoising_strength=v2v_denoising_strength,
            seed=int(v2v_seed),
            rand_device=v2v_rand_device,
            height=int(v2v_height),
            width=int(v2v_width),
            num_frames=int(v2v_num_frames),
            cfg_scale=v2v_cfg_scale,
            num_inference_steps=int(v2v_num_inference_steps),
            sigma_shift=v2v_sigma_shift,
            tiled=v2v_tiled,
            tile_size=v2v_tile_size,
            tile_stride=v2v_tile_stride,
        )
    except Exception as e:
        print(f"视频生成失败: {e}")
        v2v = None
     # Create output directory if it doesn't exist
    output_dir = os.path.join(project_root,"output", "v2v")  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate unique filename based on current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"Wan2_1_v2v_{timestamp}.mp4"
    video_path = os.path.join(output_dir, video_filename)
    
    save_video(v2v, video_path, fps=output_fps, quality=output_quality)

    # Save preview image
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preview_image = Image.fromarray(frame)
        preview_path = os.path.join(output_dir, f"{os.path.splitext(video_filename)[0]}.jpg")
        preview_image.save(preview_path)
    
    # Save parameters to a text file
    params = {
                "prompt": v2v_prompt,
                "negative_prompt": v2v_negative_prompt,
                "input_image": v2v_input_image,
                "input_video": v2v_input_video,
                "denoising_strength": v2v_denoising_strength,
                "seed": v2v_seed,
                "rand_device": v2v_rand_device,
                "height": v2v_height,
                "width": v2v_width,
                "num_frames": v2v_num_frames,
                "cfg_scale": v2v_cfg_scale,
                "num_inference_steps": v2v_num_inference_steps,
                "sigma_shift": v2v_sigma_shift,
                "tiled": v2v_tiled,
                "tile_size": v2v_tile_size,
                "tile_stride": v2v_tile_stride,
    }
    param_filename = f"Wan2_1_v2v_{timestamp}.txt"
    param_path = os.path.join(output_dir, param_filename)
    
    with open(param_path, 'w', encoding='utf-8') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    
    return video_path