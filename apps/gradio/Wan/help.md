## 帮助
这个是基于阿里摩搭开源的[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)制作的WebUI。

[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)是一个 Diffusion 引擎,重构了包括 Text Encoder、UNet、VAE 等在内的架构，在提高计算性能的同时，保持与开源社区模型的兼容性。

还使用了

[TeaCache](https://github.com/ali-vilab/TeaCache) Timestep Embedding Aware Cache ，这是一种免训练缓存方法，可估计并利用不同时间步长模型输出之间的波动差异，从而加快推理速度。Wan2.1速度提高约 2 倍。

[Flash Attention](https://github.com/Dao-AILab/flash-attention)具有 IO 感知的快速且节省内存的精确注意力计算。大幅节省显存使用，越大的计算量越明显。Wan2.1显存使用最低降低至6GB。之前12GB起步。



### 一些有用的参考


[通义万相AI生视频—使用指南](https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz?spm=5176.29623064.0.0.41ed1ece1a40s1&utm_scene=person_space)

[Wan2.1开源地址](https://github.com/Wan-Video/Wan2.1/tree/main)


[通义万相Wan2.1视频生成模型地址](https://modelscope.cn/collections/tongyiwanxiang-Wan21-shipinshengcheng-67ec9b23fd8d4f)

### 注意

本软件免费提供，如您通过其他渠道付费获得本软件，请立即退款并投诉相应商家。

若想帮助我改进或加速本项目:

使用交流群：185205010

功能反馈群：296693761

或者直接联系我：
邮箱：qq1192057376.com

或者为我充电加速我的工作：https://space.bilibili.com/147981408

或者帮我的仓库贡献代码：https://github.com/qq1174565384/wan2.1_WebUI

为了确保您的贡献被正确记录，记得添加名字哦。

