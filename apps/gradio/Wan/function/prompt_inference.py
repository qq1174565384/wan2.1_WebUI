
import os

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

def prompt_inference(image):
    # os.chdir(project_root)
    # model_id = 'iic/mplug_image-captioning_coco_base_zh'
    # model_dir = os.path.join(project_root, "models", "icc","mplug_image-captioning_coco_base_zh")
    # if not os.path.exists(model_dir) or not os.listdir(model_dir):
    #     try:
    #         snapshot_download(model_id, local_dir=model_dir)
    #     except Exception as e:
    #         print(f"模型下载失败: {e}")
    #         raise
    # model = Model.from_pretrained(model_dir)
    # preprocessor = MPlugPreprocessor(model_dir)        
    # input_caption = image
    # pipeline_caption = pipeline(Tasks.image_captioning, model=model, preprocessor=preprocessor)

    # result = pipeline_caption(input_caption)

    # print(result)

    # return result

    result = "正在施工中"
    print("正在施工中")
    return result