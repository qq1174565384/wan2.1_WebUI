from transformers import AutoTokenizer
import torch
from diffsynth import ModelManager, FluxImagePipeline, download_models, QwenPrompt
import os

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

def prompt_translate(raw_prompt):

    os.chdir(project_root)
    model_dir = os.path.join(project_root, "models", "QwenPrompt")
    try:
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            print(f"千问模型不存在，正在下载:")
            download_models(["QwenPrompt"])
    except Exception as download_error:  # 给异常变量一个更具描述性的名字
        print(f"模型下载失败: {download_error}")
        raise

    try:
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
        model_manager.load_models([
            "models/QwenPrompt/qwen2-1.5b-instruct",
        ])
        print("模型加载中...")
        # 获取模型和模型路径
        model, model_path = model_manager.fetch_model("qwen_prompt", require_model_path=True)
        # 定义系统提示词
        system_prompt = """你是翻译。你的任务是对给定的英文尽可能贴切的翻译成中文。中文描述不应超过 250 字。"""
        # 初始化分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 构建消息
        messages = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': raw_prompt
            }
        ]
        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # 对输入文本进行编码
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        print ("正在翻译描述词...")
        # 生成输出
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.get('attention_mask'),  # 添加 attention_mask
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # 解码生成的结果
        prompt = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"翻译的描述词为: {prompt}")
        return prompt
    except Exception as processing_error:  # 给异常变量一个更具描述性的名字
        print(f"处理过程中发生错误: {processing_error}")
        raise
