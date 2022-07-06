# Contents

[查看中文](./README_CN.md)

- [Contents](#contents)
- [Sphereface Description](#Sphereface-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Export Process](#export-process)
        - [Export](#Export)
    - [Inferenct Process](#Inferenct-process)
        - [Inferenct](#Inferenct)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
            - [Sphereface on CASIA-WebFace](#Sphereface-on-CAISA-WebFace)
        - [Inference Performance](#inference-performance)
            - [Sphereface on LFW](#Sphereface-on-LFW)
    - [How to use](#how-to-use)
        - [Inference](#inference-1)
        - [Continue Training on the Pretrained Model](#continue-training-on-the-pretrained-model)
        - [Transfer Learning](#transfer-learning)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Sphereface Description](#Sphereface-description)

Spherenet is an innovation in the field of face recognition using improved softamax proposed in 2017. He proposed angular softmax loss to improve the original softmax loss. On the basis of large margin softmax loss, two constraints of ||W||=1 and b=0 are added, so that the prediction result depends only on W and x. The angle between. His application scenario is in an open set environment. The paper also mainly solves this task, so that in a specific measurement space, the maximum intra-class distance is less than the minimum inter-class distance of different classes as much as possible. The paper Make the feature more discriminable.

[Paper](https://arxiv.org/abs/1704.08063):  Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, Le Song."SphereFace: Deep Hypersphere Embedding for Face Recognition."*Proceedings of the IEEE conference on computer vision and pattern recognition*.2017.

# [Model Architecture](#contents)

A variety of network architectures are given in the Sphereface paper. This code only implements its 20-layer network architecture. It uses multiple 3x3 convolutions with **steps of 1 and 2, and uses the residual structure and PReLu** to perform Feature extraction, and image classification in the last fully connected layer, through the training of the last layer, the weight can be used to distinguish the features of the image.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [CASIA-WebFace](https://download.csdn.net/download/fire_light_/10291726)

- Dataset size：4G，494,414 250*250 colorful images in 10575 classes
    - Train：3.7G，454,000 images  
    - Test：0.3G，40,000 images
- Data format：RGB images
    - Note：Data will be processed in src/dataset.py

Dataset used: [LFW](http://vis-www.cs.umass.edu/lfw/)

- Dataset size: 180M, 13,233 colorful images in 5749 classes
    - Train: 162M, 11,910 images
    - Test: 18M, 1,323 images
- Data format: RGB images.
    - Note: Data will be processed in src/eval.py

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend and GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:



- running on GPU

  ```python

  # run training example
  python train.py > train.log 2>&1 &

  # run distributed training example
  bash scripts/run_distribute_train_GPU.sh 8 0,1,2,3,4,5,6,7

  # run evaluation example
  python eval.py > eval.log 2>&1 &
  OR
  bash run_eval_standalone_GPU.sh [DEVICE_ID] [CKPT_FILES]
  # example: bash run_eval_standalone_GPU.sh 0 "/data/sphereface/6000-9923.ckpt"

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>.

We use CASIA-WebFace dataset by default. Your can also pass `$dataset_type` to the scripts so that select different datasets. For more details, please refer the specify script.

- ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows)

    - Train sphereface 1p on ModelArts

      ```python

      # (1) Add "config_path='/path_to_code/sphereface_config.yaml'" on the website UI interface.
      # (2) Perform a or b.
      #       a. Set "enable_modelarts=True" on sphereface_config.yaml file.
      #          Set "data_url='/user_name/CAISA/'" on sphereface_config.yaml file.
      #          Set "train_url='/user_name/Sphereface/output/'" on sphereface_config.yaml file.
      #          Set other parameters on sphereface_config.yaml file you need.
      #       b. Add "enable_modelarts=True" on the website UI interface.
      #          Add "train_url='/user_name/Sphereface/output/'" on the website UI interface.
      #          Add "data_url=‘/user_name/CAISA/’" on the website UI interface.
      #          Add other parameters on the website UI interface.
      # (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
      # (4) Set the code directory to "/path/sphereface" on the website UI interface.
      # (5) Set the startup file to "train.py" on the website UI interface.
      # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
      # (7) Create your job.

      ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text

├── model_zoo
    ├── README.md                          // All model related instructions
    ├── sphereface
        ├── README.md                    // Sphereface related instructions
        ├── ascend310_infer              // Implement 310 inference source code
        ├── scripts
        │   ├──run_distribute_train_GPU.sh          // Distributed to GPU training shell script
        │   ├──run_eval_standalone_GPU.sh           // Shell script evaluated by GPU
        │   ├──run_train_standalone_GPU.sh          // Shell script for single-card GPU training
        │   ├──run_distribute_train.sh              // Distributed to Ascend training shell script
        │   ├──run_eval_standalone_Ascend.sh        // Shell script evaluated by Ascend
        │   ├──run_train_standalone_Ascend.sh       // Shell script for single-card Ascend training
        ├── src
        │   ├──datasets                           // Data set processing
        │   │   ├──classfication.py               // Data classfication set processing
        │   │   ├──sampler.py                     // Data shuffle
        │   ├──losses                             // Loss function
        │   │   ├──crossentropy.py                // ASoftMax loss function
        │   ├──network                            // Net structure
        │   │   ├──spherenet.py                   // Sphereface 20 layer net structure
        ├── train.py               // Train script
        ├── eval.py               // Eval script
        ├── export.py            // Export the checkpoint file to air/mindir
        ├── sphereface_config.yaml  // Config file

```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for Sphereface, CASIA-WebFace dataset and LFW dataset.

  ```python

    device_target: 'Ascend'         #target device
    net: "sphereface20a"            #net's name
    dataset: "CASIA-WebFace"        #dataset's name
    is_distributed: 1               #whether distributed
    rank: 0                         #class
    group_size: 1                   #data group size
    label_smooth: 1                 #whether label smooth
    smooth_factor: 0.15             #label_smooth factore
    train_data_dir: "/data/sphereface/casia_landmark.txt"   #train data txt file
    train_img_dir : "/data/sphereface/CASIA-WebFace/"       #train data file
    train_pretrained: ""            #pretrained ckpt file(optional)
    image_size: "112, 96"           #crop image size
    num_classes: 10574              #train target class nums
    lr: 0.15                        #learing rate
    lr_scheduler: "cosine_annealing"    #learning rate adjustment model
    eta_min: 0                      #cos eta
    T_max: 20                       #cos half-cycle
    max_epoch: 20                   #train cycle
    per_batch_size: 32              #batch size
    warmup_epochs: 0                #warmup epoch
    weight_decay: 0.0005            #weight_decay factor
    momentum: 0.9                   #momentum factor
    ckpt_interval: 10000            #ckpt save needs iter
    save_ckpt_path: "./"            #ckpt save route
    is_save_on_master: 1            #whether save the ckpt
    ckpt_files: "/data/sphereface/Sphereface/scripts/device0/"  #eval ckpt route
    datatxt_src: '/data/sphereface/lfw_landmark.txt'            #eval data txt route
    pairtxt_src: '/data/sphereface/pairs.txt'                   #eval data pair txt route
    datasrc: '/data/sphereface/lfw/'                            #eval data route
    device_id: 7                #device id
    batch_size: 32              #eval batch size

  ```

For more configuration details, please refer the script `config.py`.

## [Training Process](#contents)

### Training


  ```

  The model checkpoint will be saved in the current directory.

- running on GPU

  ```bash

  bash scripts/run_train_standalone_GPU.sh 3

  ```

  The python command above will run in the background, you can view the results through the file `train.log`.

  After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

  ```bash

  2021-11-12 12:47:04,004:INFO:epoch[0], iter[1774], loss:8.9929695, mean_fps:0.00 imgs/sec
  2021-11-12 13:07:43,106:INFO:epoch[1], iter[3549], loss:8.491371, mean_fps:366.72 imgs/sec
  2021-11-12 13:28:21,060:INFO:epoch[2], iter[5324], loss:8.061005, mean_fps:367.06 imgs/sec
  2021-11-12 13:48:55,650:INFO:epoch[3], iter[7099], loss:7.3936157, mean_fps:368.06 imgs/sec
  ...

  ```

### Distributed Training

-

## [Evaluation Process](#contents)

### Evaluation

- evaluation on LFW dataset when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/train10-125_390.ckpt".

  ```python

  python eval.py > eval.log 2>&1 &
  OR
  bash run_eval_standalone.sh 0 "/data/sphereface/6000 9923.ckpt"

  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```bash

  # grep "accuracy:" eval.log
  LFWACC=0.9923 std=0.005 thd=0.3050

  ```

  Note that for evaluation after distributed training, please set the checkpoint_path to be the last saved checkpoint file such as "/data/sphereface/scripts/device0/0-20_1775.ckpt". The accuracy of the test dataset will be as follows:

  ```bash

  # grep "accuracy:" eval.log
  LFWACC=0.9928 std=0.0045 thd=0.2940

  ```

- evaluation on LFW dataset when running on GPU

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/train10-125_390.ckpt".

  ```bash

  python eval.py > eval.log 2>&1 &
  OR
  bash run_eval_standalone_GPU.sh 0 "/data/sphereface/6000 9923.ckpt"

  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```bash

  # grep "accuracy:" eval.log
  LFWACC=0.9913 std=0.004 thd=0.2840

  ```

  Note that for evaluation after distributed training, please set the checkpoint_path to be the last saved checkpoint file such as "/data/sphereface/scripts/device0/0-20_1775.ckpt". The accuracy of the test dataset will be as follows:

  ```bash

  # grep "accuracy:" eval.log
  LFWACC=0.9918 std=0.0045 thd=0.2800

  ```

## [Export Process](#contents)

### export

```shell

python export.py --file_format [EXPORT_FORMAT]

```

`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

## [Inference Process](#contents)

### Inference

Before performing inference, we need to export the model first. Air model can only be exported in Ascend 910 environment, mindir can be exported in any environment.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

-NOTE:Ascend310 inference use sphereface dataset .

Inference result is saved in shell
The accuracy of evaluating DenseNet121 on the test dataset of ImageNet will be as follows:

  ```log

  now the idx is %d 9
  LFWACC=0.9922 std=0.0044 thd=0.2800

  ```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### Sphereface on CASIA-WebFace

| Parameters                 | GPU|
| -------------------------- |----|
| Model Version              |Sphereface20a|
| Resource                   | V100*8; CPU 2.6GHZ 56Cores; Memory 377G; OS Euler2.8|
| uploaded Date              | 11/13/2021|
| MindSpore Version          | 1.5.0|
| Dataset                    | CASIA-WebFace|
| Training Parameters        | epoch=20, steps=1775, batch_size = 256, lr=0.15              |
| Optimizer                  |Momentum                                                    |
| Loss Function              | ASoftmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 3.2163                                                      |
| Speed                      | 1pc: 694 ms/step;  8pcs: 373 ms/step                          |
| Total time                 | 1pc: 412.39 mins;  8pcs: 220.69 mins                          |
| Parameters (M)             | 13.0                                                        |
| Checkpoint for Fine tuning |214.58M (.ckpt file)                                         |
| Scripts                    | [sphereface script](https://gitee.com/mindspore/models/tree/master/official/cv/sphereface) |

### Inference Performance

#### Sphereface on LFW

| Parameters          | GPU|
| ------------------- | ----|
| Model Version       | Sphereface20a               |
| Resource            | V100; OS Euler2.8|
| Uploaded Date       | 11/13/2021|
| MindSpore Version   | 1.5.0|
| Dataset             |LFW, 13,323 images     |
| batch_size          | 60                         |
| outputs             |probability                 |
| Accuracy            |1pc: 99.13%;  8pcs: 99.18%   |

## [How to use](#contents)

### Inference

If you need to use the trained model to perform inference on multiple hardware platforms, such as GPU, Ascend 910 or Ascend 310, you can refer to this [Link](https://www.mindspore.cn/docs/programming_guide/en/r1.6/quick_start/quick_video/inference.html). Following the steps below, this is a simple example:

- Running on  GPU

  ```python

  # Set context
  config.image_size = list(map(int,config.image_size.split(',')))
  context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,
                        save_graphs=False)

  # Load unseen dataset for inference
  ImgOut = getImg(datatxt_src,pairtxt_src,datasrc,network)

  # Define model
  network = spherenet(config.num_classes,True)

  # Load pre-trained model
   param_dict = load_checkpoint(model)
   param_dict_new = {}
       for key, values in param_dict.items():
           if key.startswith('moments.'):
               continue
           elif key.startswith('network.'):
               param_dict_new[key[8:]] = values
           else:
               param_dict_new[key] = values
       load_param_into_net(network, param_dict_new)


  # Make predictions on the unseen dataset
  for idx, (train, test) in enumerate(folds):
      best_thresh = find_best_threshold(thresholds, predicts[train])
      accuracy.append(eval_acc(best_thresh, predicts[test]))
      thd.append(best_thresh)
  print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))

  ```

### Continue Training on the Pretrained Model

- running  GPU

  ```python

  # Load dataset
  de_dataset = classification_dataset_imagenet(data_dir, image_size=[112,96],
                                                 per_batch_size=config.per_batch_size, max_epoch=config.max_epoch,
                                                 rank=0, group_size=config.group_size,
                                                 input_mode="txt", root=images_dir, shuffle=True)

  # Define model
  network = sphere20a(config.num_classes,feature=False)
  # Continue training if set pre_trained to be True
      if os.path.isfile(config.train_pretrained):
          param_dict = load_checkpoint(config.train_pretrained)
          param_dict_new = {}
          for key, values in param_dict.items():
              if key.startswith('moments.'):
                  continue
              elif key.startswith('network.'):
                  param_dict_new[key[8:]] = values
              else:
                  param_dict_new[key] = values
          load_param_into_net(network, param_dict_new)
          config.logger.info('load model %s success', str(config.train_predtrained))
  lr_schedule = lr_scheduler.get_lr()
  opt = Momentum(params=get_param_groups(network), learning_rate=Tensor(lr_schedule),
                   momentum=config.momentum, weight_decay=config.weight_decay, loss_scale=config.loss_scale)
  criterion = AngleLoss(classnum=config.num_classes,smooth_factor=config.smooth_factor)
  model = Model(train_net, optimizer=opt, metrics=None, loss_scale_manager=loss_scale_manager, amp_level="O0")

  # Set callbacks
  ckpt_max_num = config.max_epoch * config.steps_per_epoch // config.ckpt_interval
  ckpt_config = CheckpointConfig(save_checkpoint_steps=config.ckpt_interval,
                                       keep_checkpoint_max=ckpt_max_num)
  ckpt_cb = ModelCheckpoint(config=ckpt_config, directory=config.save_ckpt_path,
                                  prefix='%s' % config.rank)
  callbacks.append(ckpt_cb)

  # Start training
  model.train(config.max_epoch, de_dataset, callbacks=callbacks,dataset_sink_mode=False)

  ```

### Transfer Learning

To be added.

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)  

 Please check the official [homepage](https://gitee.com/mindspore/models).  
