


从DiffSynth Studio分支构建




## 安装

从源代码安装（推荐）：

``` sh
git clone https://github.com/qq1174565384/wan2.1_WebUI.git
```
<!--
```
cd wan2.1_WebUI
```
```
pip install -e .
```
-->

下载模型（二选一）
Download models using huggingface-cli:
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
```

Download models using modelscope-cli:
``` sh
pip install modelscope
modelscope download Wan-AI/Wan2.1-T2V-14B --local_dir ./Wan2.1-T2V-14B
```


