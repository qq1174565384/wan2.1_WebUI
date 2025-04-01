import os
import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download
import random
import ast
from datetime import datetime
import os.path
import threading


# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

global t2v_model_state,i2v_model_state, model_manager, pipe
t2v_model_state = False  # 初始化模型状态为 False
i2v_model_state = False  # 初始化模型状态为 False
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
    t2v_ModelChoices="Wan-AI/Wan2.1-T2V-1.3B",
    t2v_num_persistent_param_in_dit=0,
    t2v_loadChoices="RAM"
):
    global t2v_model_state,i2v_model_state, model_manager, pipe
    def download_and_load_model(model_choice, project_root):
        global t2v_model_state,i2v_model_state, model_manager, pipe
        if model_choice == "Wan-AI/Wan2.1-T2V-1.3B":
            model_name = "Wan2.1-T2V-1.3B"
            model_paths = [
                os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-1.3B", "diffusion_pytorch_model.safetensors"),
                os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-1.3B", "models_t5_umt5-xxl-enc-bf16.pth"),
                os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth"),
            ]
            torch_dtype = torch.bfloat16
        else:
            model_name = "Wan2.1-T2V-14B"
            model_paths = [
                [
                    os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00001-of-00006.safetensors"),
                    os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00002-of-00006.safetensors"),
                    os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00003-of-00006.safetensors"),
                    os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00004-of-00006.safetensors"),
                    os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00005-of-00006.safetensors"),
                    os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00006-of-00006.safetensors"),
                ],
                os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-14B", "models_t5_umt5-xxl-enc-bf16.pth"),
                os.path.join(project_root, "models", "Wan-AI", "Wan2.1-T2V-14B", "Wan2.1_VAE.pth"),
            ]
            torch_dtype = torch.float8_e4m3fn

        model_dir = os.path.join(project_root, "models", "Wan-AI", model_name)
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            try:
                snapshot_download(model_choice, local_dir=model_dir)
            except Exception as e:
                print(f"模型下载失败: {e}")
                raise

        # 根据 t2v_loadChoices 设置设备
        if t2v_loadChoices == "CPU":
            device = "cpu"
        elif t2v_loadChoices == "CUDA":
            device = "cuda"
        else:
            print(f"无效的 t2v_loadChoices 值: {t2v_loadChoices}，使用默认值 cpu")
            device = "cpu"

        model_manager = ModelManager(device=device) 
        
        if not t2v_model_state:  # 根据 t2v_model_state 值判断是否执行模型加载
            try:
                model_manager.load_models(
                    model_paths,
                    torch_dtype=torch_dtype,
                )
            except Exception as e:
                print(f"模型加载失败: {e}")
                raise

            # 创建管道
            pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
            pipe.enable_vram_management(num_persistent_param_in_dit=t2v_num_persistent_param_in_dit)

        return model_manager, pipe

    if t2v_ModelChoices == "Wan-AI/Wan2.1-T2V-1.3B":
        model_manager, pipe = download_and_load_model(
            "Wan-AI/Wan2.1-T2V-1.3B", project_root
        )
    else:
        model_manager, pipe = download_and_load_model(
            "Wan-AI/Wan2.1-T2V-14B", project_root
        )

    t2v_model_state = True
    i2v_model_state = False

    # 处理输入参数
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
            # TeaCache parameters
            # tea_cache_l1_thresh=0.05, # The larger this value is, the faster the speed, but the worse the visual quality.
            # tea_cache_model_id="Wan2.1-T2V-1.3B", # Choose one in (Wan2.1-T2V-1.3B, Wan2.1-T2V-14B, Wan2.1-I2V-14B-480P, Wan2.1-I2V-14B-720P).
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
    i2v_prompt="",  # 设置默认值为空字符串,
    i2v_negative_prompt="",  # 设置默认值为空字符串
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
    i2v_output_fps=15,
    i2v_output_quality=9,
    i2v_num_persistent_param_in_dit=0,
    i2v_ModelChoices = "Wan-AI/Wan2.1-I2V-14B-480P",
    i2v_loadChoices = "CPU"
):
    global t2v_model_state,i2v_model_state, model_manager, pipe

    def download_and_load_model(model_choice, project_root):
       
        global t2v_model_state,i2v_model_state, model_manager, pipe
        #按需加载模型    
        if model_choice == "Wan-AI/Wan2.1-I2V-14B-480P":
            model_name = "Wan2.1-I2V-14B-480P"
             #检测模型是否存在
            model_dir = os.path.join(project_root, "models", "Wan-AI", model_name)
            if not os.path.exists(model_dir) or not os.listdir(model_dir):
                try:
                    snapshot_download(model_choice, local_dir=model_dir)
                except Exception as e:
                    print(f"模型下载失败: {e}")
                    raise
            model_paths1 = [os.path.join(project_root, "models", "Wan-AI", model_name, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")],
            model_paths2 = [
                [
                    os.path.join(project_root, "models", "Wan-AI", model_name, "diffusion_pytorch_model-00001-of-00007.safetensors"),
                    os.path.join(project_root, "models", "Wan-AI", model_name, "diffusion_pytorch_model-00002-of-00007.safetensors"),
                    os.path.join(project_root, "models", "Wan-AI", model_name, "diffusion_pytorch_model-00003-of-00007.safetensors"),
                    os.path.join(project_root, "models", "Wan-AI", model_name, "diffusion_pytorch_model-00004-of-00007.safetensors"),
                    os.path.join(project_root, "models", "Wan-AI", model_name, "diffusion_pytorch_model-00005-of-00007.safetensors"),
                    os.path.join(project_root, "models", "Wan-AI", model_name, "diffusion_pytorch_model-00006-of-00007.safetensors"),
                    os.path.join(project_root, "models", "Wan-AI", model_name, "diffusion_pytorch_model-00007-of-00007.safetensors"),
                ],
                os.path.join(project_root, "models", "Wan-AI", model_name, "models_t5_umt5-xxl-enc-bf16.pth"),
                os.path.join(project_root, "models", "Wan-AI", model_name, "Wan2.1_VAE.pth"),
            ]
        else:
            model_name = "Wan2.1-I2V-14B-720P"
             #检测模型是否存在
            model_dir = os.path.join(project_root, "models", "Wan-AI", model_name)
            if not os.path.exists(model_dir) or not os.listdir(model_dir):
                try:
                    snapshot_download(model_choice, local_dir=model_dir)
                except Exception as e:
                    print(f"模型下载失败: {e}")
                    raise
            model_paths1 = [os.path.join(project_root, "models", "Wan-AI", model_name, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")],
            model_paths2 = [
                [
                    os.path.join(project_root, "models", "Wan-AI", model_name, "diffusion_pytorch_model-00001-of-00007.safetensors"),
                    os.path.join(project_root, "models", "Wan-AI", model_name, "diffusion_pytorch_model-00002-of-00007.safetensors"),
                    os.path.join(project_root, "models", "Wan-AI", model_name, "diffusion_pytorch_model-00003-of-00007.safetensors"),
                    os.path.join(project_root, "models", "Wan-AI", model_name, "diffusion_pytorch_model-00004-of-00007.safetensors"),
                    os.path.join(project_root, "models", "Wan-AI", model_name, "diffusion_pytorch_model-00005-of-00007.safetensors"),
                    os.path.join(project_root, "models", "Wan-AI", model_name, "diffusion_pytorch_model-00006-of-00007.safetensors"),
                    os.path.join(project_root, "models", "Wan-AI", model_name, "diffusion_pytorch_model-00007-of-00007.safetensors"),
                ],
                os.path.join(project_root, "models", "Wan-AI", model_name, "models_t5_umt5-xxl-enc-bf16.pth"),
                os.path.join(project_root, "models", "Wan-AI", model_name, "Wan2.1_VAE.pth"),
            ]

       

        # 根据 t2v_loadChoices 设置设备
        if i2v_loadChoices == "CPU":
            device = "cpu"
        elif i2v_loadChoices == "CUDA":
            device = "cuda"
        else:
            print(f"无效的 i2v_loadChoices 值: {i2v_loadChoices}，使用默认值 cpu")
            device = "cpu"


        # 加载模型
        model_manager = ModelManager(device=device) 

        if not i2v_model_state:
            try:
                model_manager.load_models(
                    model_paths1,
                    torch_dtype=torch.float32, # Image Encoder is loaded with float32
                )
                model_manager.load_models(
                    model_paths2,
                    torch_dtype=torch.float16,
                )
            except Exception as e:
                print(f"模型加载失败: {e}")
                raise

            # 创建管道
            pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
            pipe.enable_vram_management(num_persistent_param_in_dit=i2v_num_persistent_param_in_dit)

        return model_manager, pipe

    if i2v_ModelChoices == "Wan-AI/Wan2.1-I2V-14B-480P":
        model_manager, pipe = download_and_load_model(
            "Wan-AI/Wan2.1-I2V-14B-480P", project_root
        )
    else:
        model_manager, pipe = download_and_load_model(
            "Wan-AI/Wan2.1-I2V-14B-720P", project_root
        )

    t2v_model_state = False
    i2v_model_state = True





    # 处理输入参数
    i2v_seed = i2v_seed if i2v_seed >= 0 else random.randint(0, 2147483647)

    # 确保 t2v_resolution 是字符串类型
    i2v_resolution = str(i2v_resolution)
   
    
    i2v_width, i2v_height = map(int, i2v_resolution.split('*'))
    
    print("正在生成视频...")
    try:
        i2v = pipe(
            prompt=i2v_prompt,
            negative_prompt=i2v_negative_prompt,
            input_image=i2v_input_image,
            seed=int(i2v_seed),
            height=int(i2v_height),
            width=int(i2v_width),
            num_frames=int(i2v_num_frames),
            num_inference_steps=int(i2v_num_inference_steps),
            tiled=i2v_tiled,

        )


    except Exception as e:
        print(f"视频生成失败: {e}")
        i2v = None
    # Create output directory if it doesn't exist
    output_dir = os.path.join(project_root,"output", "i2v")  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate unique filename based on current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"Wan2_1_i2v_{timestamp}.mp4"
    video_path = os.path.join(output_dir, video_filename)
    print("正在保存视频...")
    save_video(i2v, video_path, fps=i2v_output_fps, quality=i2v_output_quality)
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
                "prompt": i2v_prompt,
                "negative_prompt": i2v_negative_prompt,
                "input_image": "file",
                "seed": i2v_seed,
                "num_frames": i2v_num_frames,
                "num_inference_steps": i2v_num_inference_steps,
 
    }
    param_filename = f"Wan2_1_i2v_{timestamp}.txt"
    param_path = os.path.join(output_dir, param_filename)
    
    with open(param_path, 'w', encoding='utf-8') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print("任务成功")
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