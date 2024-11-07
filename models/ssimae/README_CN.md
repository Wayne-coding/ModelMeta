

<!-- TOC -->

- [目录](
- [SSIM-AE描述](
- [模型架构](
- [数据集](
- [特性](
    - [混合精度](
- [环境要求](
- [快速入门](
    - [训练过程](
        - [训练](
        - [ModelArt训练](
    - [推理过程](
        - [Ascend910处理器环境推理](
        - [导出](
        - [执行推理](
- [模型描述](
- [随机情况说明](
- [ModelZoo主页](

<!-- /TOC -->



自动编码器已经成为无监督的缺陷检测的常用方法。一般通过自动编码器重建图像，然后对重建图和原图在像素级别上进行比较，距离大于一定的阈值则认为是缺陷。但基于距离的损失函数在图像中某些边缘区域重建不准确时造成了较大的误差，且当缺陷在强度上大致一致而在结构上区别较大时，基于距离的损失函数无法检测出这些缺陷。通过采用更复杂的编码器也无法改善这些问题，基于此，该论文采用结构相似性感知损失函数来检查像素之间的相互依赖性，并同时考虑亮度，对比度和结构信息。

[论文](https://www.researchgate.net/publication/326222902)：Improving Unsupervised Defect Segmentation by Applying Structural Similarity To Autoencoders



SSIM-AE由一连串对称的卷积层和反卷积层组成，具体的网络结构如下表：

| Layer      | Output Size | Kernel | Stride | Padding |
|------------|-------------|:------:|--------|---------|
| Input      | 128*128\*1  |        |        |         |
| Conv1      | 64\*64*32   |  4*4   | 2      | 1       |
| Conv2      | 32\*32*32   |  4*4   | 2      | 1       |
| Conv3      | 32\*32*32   |  3*3   | 1      | 1       |
| Conv4      | 16\*16*64   |  4*4   | 2      | 1       |
| Conv5      | 16\*16*64   |  3*3   | 1      | 1       |
| Conv6      | 8\*8*128    |  4*4   | 2      | 1       |
| Conv7      | 8\*8*64     |  3*3   | 1      | 1       |
| Conv8      | 8\*8*32     |  3*3   | 1      | 1       |
| Conv9      | 1*1\*d      |  8*8   | 1      | 0       |
| ConvTrans1 | 8\*8*32     |  8*8   | 1      | 0       |
| Conv10     | 8\*8*64     |  3*3   | 1      | 1       |
| Conv11     | 8\*8*128    |  3*3   | 1      | 1       |
| ConvTrans2 | 16\*16*64   |  4*4   | 2      | 1       |
| Conv12     | 16\*16*64   |  3*3   | 1      | 1       |
| ConvTrans3 | 32\*32*32   |  4*4   | 2      | 1       |
| Conv13     | 32\*32*32   |  3*3   | 1      | 1       |
| ConvTrans4 | 64\*64*32   |  4*4   | 2      | 1       |
| ConvTrans5 | 128*128\*1  |  4*4   | 2      | 1       |



本实现使用了[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)数据集。

MVTec AD数据集

- 说明：
- 数据集大小：4.9G，共15个类、5354张高分辨图像
    - 训练集：3.4G，共3629张图像
    - 测试集：1.5G，共1725张图像
- 数据格式：二进制文件（PNG）和RGB
  MVTec AD其中一个类的目录结构如下：

```bash
.
└─metal_nut
  └─train
    └─good
      └─000.png
      └─001.png
      ...
  └─test
    └─bent
      └─000.png
      └─001.png
       ...
    └─color
      └─000.png
      └─001.png
       ...
    ...
  └─ground_truth
    └─bent
      └─000_mask.png
      └─001_mask.png
      ...
    └─color
      └─000_mask.png
      └─001_mask.png
      ...
    ...
```

其中验证集中good目录下的数据为无缺陷图片。

编织品纹理数据集我们采用像素级的评价指标，采用AUC来判断缺陷位置是否预测正确。MVTec AD数据集我们采用图像级的评价指标，当构建的图像中有预测出缺陷位置时我们认为这张图片时有缺陷的，采用图像级的预测方式：ok表示无缺陷图片预测正确率，nok表示有缺陷图片预测正确率，avg时整个数据集的预测正确率。

- 注：数据将在src/dataset.py中处理。





采用 [混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html) 的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。



- 硬件（Ascend/CPU）
    - 使用Ascend/CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)



通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

1. 修改config目录下对应数据集的yaml文件：

    ```yaml
    

    device_target: Ascend
    dataset: "none"
    dataset_path: ""
    aug_dir: ""   
    distribute: False

    grayscale: False   
    do_aug: True       
    online_aug: False  

    
    augment_num: 10000
    im_resize: 256
    crop_size: 128
    rotate_angle: 45.
    p_ratate: 0.3
    p_horizontal_flip: 0.3
    p_vertical_flip: 0.3

    
    z_dim: 100
    epochs: 200
    batch_size: 128
    lr: 2.0e-4
    decay: 1.0e-5
    flc: 32           
    stride: 32  
    load_ckpt_path: "" 

    
    image_level: True     
    ssim_threshold: -1.0  
    l1_threshold: -1.0    
    percent: 98
    checkpoint_path: ""   
    save_dir: "./output"  
    ```

2. 执行训练

    - Ascend处理器环境运行

      ```shell
      
      bash scripts/run_standalone_train.sh [CONFIG_PATH] [DEVICE_ID]

      
      bash scripts/run_eval.sh [CONFIG_PATH] [DEVICE_ID]

      
      bash scripts/run_infer_310.sh [MINDIR_PATH] [CONFIG_PATH] [SSIM_THRESHOLD] [L1_THRESHOLD] [DEVICE_ID]
      ```





- Ascend处理器单卡运行

  ```bash
  
  python train.py --config_path=[CONFIG_PATH]
  or
  bash scripts/run_standalone_train.sh [CONFIG_PATH] [DEVICE_ID]
  
  
  ```

- CPU处理器运行

  ```bash
  
  python train.py --config_path=[CONFIG_PATH] --device_target=CPU
  ```

  训练结束后，您可在`./checkpoint`下找到检查点文件。



- 在 ModelArts 上使用8卡训练

  ```python
  
  
  
  
  
  
  
  
  
  ```

- 在 ModelArts 上使用单卡训练

  ```python
  
  
  
  
  
  
  
  
  ```

训练结束后，您可在`/[桶名称]/result/checkpoint`下找到检查点文件。



**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**



- 在Ascend910环境运行时评估

  在运行以下命令之前，请检查用于推理的检查点路径。请将检查点路径设置为绝对全路径，例如`username/ssim-ae/ssim_autocoder_22-257_8.ckpt`。

  ```bash
  
  python eval.py --config_path=[CONFIG_PATH]
  or
  bash scripts/run_eval.sh [CONFIG_PATH] [DEVICE_ID]
  ```

  注：对于分布式训练后评估，请将checkpoint_path设置为最后保存的检查点文件，如`username/ssim-ae/ssim_autocoder_22-257_8.ckpt`。测试数据集的准确性如下：

  ```file
  
  ok: 0.9, nok: 0.9841269841269841, avg: 0.963855421686747
  ```



```shell
python export.py --config_path=[CONFIG_PATH]
```



在推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。batch_size只支持1。

- 使用MVTec AD bottle数据集进行推理

  在执行下面的命令之前，我们需要确认config文件中的配置和训练一致。其中ssim_threshold和l1_threshold需要手动添加，最好与910上自动获取的值一致。

  推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```shell
  bash scripts/run_infer_cpp.sh [MINDIR_PATH] [CONFIG_PATH] [SSIM_THRESHOLD] [L1_THRESHOLD] [DEVICE_TYPE] [DEVICE_ID]
  
  
  ok: 0.9, nok: 0.9841269841269841, avg: 0.963855421686747
  ```



| 参数          | Ascend                                                                          |
|-------------|---------------------------------------------------------------------------------|
| 模型版本        | SSIM-AE                                                                         |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8                                 |
| 上传日期        | 2021-12-30                                                                      |
| MindSpore版本 | 1.5.0                                                                           |
| 脚本          | [ssim-ae脚本](https://gitee.com/mindspore/models/tree/master/research/cv/SSIM-AE) |

| 数据集                | 训练参数                  | 速度(单卡)     | 总时长   | 损失函数 | 精度                                      | checkpoint文件大小 |
|--------------------|-----------------------|------------|-------|------|-----------------------------------------|----------------|
| MVTec AD bottle    | bottle_config.yaml    | 354ms/step | 1.6h  | SSIM | ok: 90%, nok: 98.4%, avg: 96.4% (图像级)   | 32M            |
| MVTec AD cable     | cable_config.yaml     | 359ms/step | 1.6h  | SSIM | ok: 0%, nok: 100%, avg: 61.3% (图像级)     | 32M            |
| MVTec AD capsule   | capsule_config.yaml   | 357ms/step | 1.6h  | SSIM | ok: 47.8%, nok: 91.7%, avg: 84.1% (图像级) | 32M            |
| MVTec AD carpet    | carpet_config.yaml    | 57ms/step  | 0.3h  | SSIM | ok: 50%, nok: 98.8%, avg: 87.1% (图像级)   | 13M            |
| MVTec AD grid      | grid_config.yaml      | 53/step    | 0.27h | SSIM | ok: 100%, nok: 94.7%, avg: 96.2% (图像级)  | 13M            |
| MVTec AD metal_nut | metal_nut_config.yaml | 355ms/step | 1.6h  | SSIM | ok: 27.2%, nok: 91.4%, avg: 79.1% (图像级) | 32M            |



dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。



请浏览官网[主页](https://gitee.com/mindspore/models)。