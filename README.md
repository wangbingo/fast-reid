## 方案简介
本方案基于京东FastReID框架实现。
网络采用ResNest101, 损失函数为circle-loss + triplet-loss.

## 系统安装
$ conda install pytorch==1.6.0 torchvision tensorboard -c pytorch

$ git clone https://github.com/wangbingo/fast-reid.git

$ cd fast-reid

$ pip install -r requirements

$ cd fast-reid/fastreid/evaluation/rank_cylib && make all

## 数据准备
$ cp XXXX/train.zip ./ && cp XXXX/image_B.zip ./

$ unzip train.zip && unzip image_B.zip

$ cd fast-reid  &&  python tools/prepare_train-data.py &&
python tools/prepare_test-data.py

###将数据组织成market1501形式，便于直接利用框架

$ mkdir -p  fast-reid/datasets/Market-1501-v15.09.15 && \

 mv fast-reid/datasets/pclreid/train  fast-reid/datasets/Market-1501-v15.09.15 && \
 
 mv  fast-reid/datasets/Market-1501-v15.09.15/train fast-reid/datasets/Market-1501-v15.09.15/bounding_box_train

$ mv /content/fast-reid/datasets/pclreid/*  /content/fast-reid/datasets/Market-1501-v15.09.15

$ mv /content/fast-reid/datasets/Market-1501-v15.09.15/gallery  /content/fast-reid/datasets/Market-1501-v15.09.15/bounding_box_test


## 训练
$ cd fast-reid  && python ./tools/train_net.py --config-file ./configs/Market1501/sbs_S101.yml MODEL.DEVICE "cuda:0" 

###训练完成后，权重文件保存在fast-reid/logs/market1501/sbs_S101

由于权重文件较大，故存放在百度盘供提取。

链接: https://pan.baidu.com/s/10gM1TKk2LuNUw086DKCWUg  密码: 1vr6

## 测试（test）
$ cd fast-reid/ && sh demo/my_run_demo.sh

###测试完成后，结果文件保存在fast-reid/result_2020-10-14-08-56-35.json(时间为当前系统时间)


