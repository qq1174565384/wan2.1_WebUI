from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
# 修改为 models 文件夹
local_model_path = os.path.join(project_root, 'models','Salesforce','blip-image-captioning-large')

# 配置请求重试机制
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

def prompt_inference(image):
    os.chdir(project_root)

    # 设置 Hugging Face 镜像源
    os.environ['HF_ENDPOINT'] = 'https://mirror.aliyun.com/hugging-face-models'

    try:
        print("正在尝试从本地加载blip-image-captioning-large模型...")
        # 尝试从本地加载处理器和模型
        processor = BlipProcessor.from_pretrained(local_model_path)
        model = BlipForConditionalGeneration.from_pretrained(local_model_path)
    except OSError:
        # 如果本地没有，从 Hugging Face 下载并保存到本地
        print("本地加载失败，正在尝试从 Hugging Face 下载...")
        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", timeout=30)
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", timeout=30)
            # 保存处理器和模型到本地
            processor.save_pretrained(local_model_path)
            model.save_pretrained(local_model_path)
        except Exception as e:
            print(f"从 Hugging Face 下载失败: {e}")
            return "模型下载失败，请检查网络连接"

    # 打开图片
    image = image

    # 预处理图片
    inputs = processor(image, return_tensors="pt")

    print("正在生成描述...")
    # 生成描述
    out = model.generate(**inputs, max_new_tokens=50)

    # 解码生成的描述
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption