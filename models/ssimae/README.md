

<!-- TOC -->

- [Contents](
- [SSIM-AE Description](
- [Model Architecture](
- [Dataset](
- [Feature](
    - [Mixed Precision](
- [Environment Requirements](
- [Quick Start](
  - [Training Process](
      - [Training](
      - [ModelArt Training](
  - [Inference Process](
    - [Environment Inference on Ascend 910 AI Processor](
    - [Export Process](
    - [Environment Inference on Ascend 310 Processor](
- [Model Description](
- [Random Seed Description](
- [ModelZoo Home Page](

<!-- /TOC -->



Autoencoder has emerged as a popular method for unsupervised defect detection. Usually, the image reconstructed by autoencoder is compared with the source image at the pixel level. If the distance is greater than a certain threshold, the source image is considered as a defective image. However, the distance-based loss function causes a large error when the reconstruction of some edge regions in the image is inaccurate. Moreover, when the defects are roughly the same in intensity but differ greatly in structure, the distance-based loss function cannot detect these defects. Given that even more advanced autoencoder cannot deal with these problems, this paper proposes to use a perceptual loss function based on structural similarity which examines inter-dependencies between pixels, taking into account luminance, contrast and structural information.

[Paper](https://www.researchgate.net/publication/326222902): Improving Unsupervised Defect Segmentation by Applying Structural Similarity To Autoencoders



SSIM-AE consists of a series of symmetric convolutional and transposed convolutional layers. The network structure is as follows.

| Layer      | Output Size | Kernel | Stride | Padding |
| ---------- | ----------- | :----: | ------ | ------- |
| Input      | 128 x 128 x 1 |        |        |         |
| Conv1      | 64 x 64 x 32  |  4 x 4  | 2      | 1       |
| Conv2      | 32 x 32 x 32  |  4 x 4  | 2      | 1       |
| Conv3      | 32 x 32 x 32  |  3 x 3  | 1      | 1       |
| Conv4      | 16 x 16 x 64  |  4 x 4  | 2      | 1       |
| Conv5      | 16 x 16 x 64  |  3 x 3  | 1      | 1       |
| Conv6      | 8 x 8 x 128   |  4 x 4  | 2      | 1       |
| Conv7      | 8 x 8 x 64    |  3 x 3  | 1      | 1       |
| Conv8      | 8 x 8 x 32    |  3 x 3  | 1      | 1       |
| Conv9      | 1 x 1 x d     |  8 x 8  | 1      | 0       |
| ConvTrans1 | 8 x 8 x 32    |  8 x 8  | 1      | 0       |
| Conv10     | 8 x 8 x 64    |  3 x 3  | 1      | 1       |
| Conv11     | 8 x 8 x 128   |  3 x 3  | 1      | 1       |
| ConvTrans2 | 16 x 16 x 64  |  4 x 4  | 2      | 1       |
| Conv12     | 16 x 16 x 64  |  3 x 3  | 1      | 1       |
| ConvTrans3 | 32 x 32 x 32  |  4 x 4  | 2      | 1       |
| Conv13     | 32 x 32 x 32  |  3 x 3  | 1      | 1       |
| ConvTrans4 | 64 x 64 x 32  |  4 x 4  | 2      | 1       |
| ConvTrans5 | 128 x 128 x 1 |  4 x 4  | 2      | 1       |



Used dataset: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

MVTec AD Dataset

- Description:
- Dataset size: 4.9 GB, 5354 high-resolution images in 15 classes
    - Training set: 3.4 GB, 3629 images
    - Test set: 1.5 GB, 1725 images
- Data format: binary file (PNG) and RGB
  The directory structure of a class in MVTec AD is as follows:

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

Non-detective images are stored in the **good** directory in the validation set.

We adopt pixel-level evaluation metrics for the woven texture dataset. The AUC values are used to determine whether the defect location is predicted correctly. We adopt image-level evaluation metrics for the MVTec AD dataset. The defective image is recognized if its defect location is predicted by the image-level prediction. **ok** indicates the correct rate of non-defective image prediction. **nok** indicates the correct rate of defective image prediction. **avg** indicates the correct rate of whole dataset prediction.

- Note: Data will be processed in **src/dataset.py**.





[Mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) accelerates the training process of deep neural networks by using the single-precision (FP32) data and half-precision (FP16) data without compromising the precision of networks trained with single-precision (FP32) data. It not only accelerates the computing process and reduces the memory usage, but also supports a larger model or batch size to be trained on specific hardware.
Take the FP16 operator as an example. If the input data format is FP32, MindSpore automatically reduces the precision to process data. You can open the INFO log and search for the keyword "reduce precision" to view operators with reduced precision.



- Hardware (Ascend/CPU)
    - Set up the hardware environment with Ascend AI Processors or CPUs.
- Framework
  - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the following resources:
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)



After installing MindSpore from the official website, you can perform the following steps for training and evaluation:

1. Modify the **yaml** file of the corresponding dataset in the **config** directory.

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

2. Start training.

- On Ascend AI Processors:

  ```shell
  
  bash scripts/run_standalone_train.sh [CONFIG_PATH] [DEVICE_ID]

  
  bash scripts/run_eval.sh [CONFIG_PATH] [DEVICE_ID]

  
  bash scripts/run_infer_310.sh [MINDIR_PATH] [CONFIG_PATH] [SSIM_THRESHOLD] [L1_THRESHOLD] [DEVICE_ID]
  ```





- Single-device training on Ascend AI Processor

  ```bash
  
  python train.py --config_path=[CONFIG_PATH]
  or
  bash scripts/run_standalone_train.sh [CONFIG_PATH] [DEVICE_ID]
  
  
  ```

- Training on CPU

  ```bash
  
  python train.py --config_path=[CONFIG_PATH] --device_target=CPU
  ```

  After the training is complete, you can find the checkpoint file in `./checkpoint`.



- 8-device training on ModelArts

  ```python
  
  
  
  
  
  
  
  
  
  ```

- Single-device training on ModelArts

  ```python
  
  
  
  
  
  
  
  
  ```

After the training is complete, you can find the checkpoint file in `/[Bucket name]/result/checkpoint`.



**Set environment variables before inference by referring to [MindSpore C++ Inference Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md).**



- Evaluation

  Check the checkpoint path used for inference before running the command below. Set the checkpoint path to the absolute path, for example, `username/ssim-ae/ssim_autocoder_22-257_8.ckpt`.

  ```bash
  
  python eval.py --config_path=[CONFIG_PATH]
  or
  bash scripts/run_eval.sh [CONFIG_PATH] [DEVICE_ID]
  ```

  Note: For evaluation after distributed training , set checkpoint_path to the last saved checkpoint file, for example, `username/ssim-ae/ssim_autocoder_22-257_8.ckpt`. The accuracy of the test dataset is as follows:

  ```file
  
  ok: 0.9, nok: 0.9841269841269841, avg: 0.963855421686747
  ```



```shell
python export.py --config_path=[CONFIG_PATH]
```



   Export the model before inference. AIR models can be exported only in the Ascend 910 environment. MindIR models can be exported in any environment. The value of **batch_size** can only be **1**.

- Infer the bottle dataset of MVTec AD on Ascend 310.

  Before running the following command, ensure that the configurations in the **config** file is the same as the training parameters. You need to manually add the values of **ssim_threshold** and **l1_threshold**. The values are better to be consistent with values automatically obtained on the Ascend 910.

  The inference result is saved in the current directory. In the **acc.log** file, you can find similar result below.

  ```shell
  
  bash scripts/run_infer_310.sh [MINDIR_PATH] [CONFIG_PATH] [SSIM_THRESHOLD] [L1_THRESHOLD] [DEVICE_ID]
  
  
  ok: 0.9, nok: 0.9841269841269841, avg: 0.963855421686747
  ```



| Parameter         | Ascend                                                       |
| ------------- | ------------------------------------------------------------ |
| Model version     | SSIM-AE                                                      |
| Resources         | Ascend 910; 2.60 GHz CPU with 192 cores; 755 GB memory; EulerOS 2.8    |
| Upload date     | 2021-12-30                                                   |
| MindSpore version| 1.5.0                                                        |
| Script         | [ssim-ae script](https://gitee.com/mindspore/models/tree/master/research/cv/SSIM-AE)|

| Dataset   | Training Parameters| Speed (single device)| Total Duration| Loss Function| Accuracy| Checkpoint File Size|
| -------- |------- |----- |----- |-------- |------ |--------------- |
| MVTec AD bottle   | bottle_config.yaml | 354ms/step | 1.6 hours| SSIM | **ok**: 90%. **nok**: 98.4%. **avg**: 96.4%. (image level)| 32 MB|
| MVTec AD cable    | cable_config.yaml | 359 ms/step| 1.6 hours| SSIM | **ok**: 0%. **nok**: 100%. **avg**: 61.3%. (image level)| 32 MB|
| MVTec AD capsule  | capsule_config.yaml | 357 ms/step| 1.6 hours| SSIM | **ok**: 47.8%. **nok**: 91.7%. **avg**: 84.1%. (image level)| 32 MB|
| MVTec AD carpet   | carpet_config.yaml | 57 ms/step| 0.3 hours| SSIM | **ok**: 50%. **nok**: 98.8%. **avg**: 87.1%. (image level)| 13 MB|
| MVTec AD grid     | grid_config.yaml | 53 ms/step| 0.27 hours| SSIM | **ok**: 100%. **nok**: 94.7%. **avg**: 96.2%. (image level)| 13 MB|
| MVTec AD metal_nut   | metal_nut_config.yaml | 355 ms/step| 1.6 hours| SSIM | **ok**: 27.2%. **nok**: 91.4%. **avg**: 79.1%. (image level)| 32 MB|



The seed in the create_dataset function is set in **dataset.py**, and the random seed in **train.py** is used.



For details, please go to the [official website](https://gitee.com/mindspore/models).
