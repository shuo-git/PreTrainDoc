## Fairseq+Fairscale运行（王硕）

### 代码

#### 位置

https://github.com/shuo-git/fairseq-pretrain

Fork Fairseq最新版本，已实现对Fairscale的集成。

#### 开发记录

###### commit 431620876aeaadddcd6bf075a7ac38b2fc3917c3

嵌入Huggingface mT5的tokenizer，可以直接处理plain text。现在实现的版本比较丑陋，需要在代码里手动预设mT5 tokenizer的位置，后续需要进一步优化。目前已验证可用的tokenizer路径为：[thu102] /data/private/ws/DATASET/Medical/mT5_tokenizer

### 环境

基础镜像：

```shell
docker pull gyxthu17/cpm-2:1.1
```

已下载好并存储在本地路径：

```shell
[thu102] /data/private/ws/cpm-2.tar
```

运行此`docker`需要安装`nvidia-docker`，详情请参照`RunMegatron.md`中相关介绍。在[thu102]服务器上安装好后，运行`nvidia-docker`碰到一些小问题，记不清了，根据报错信息搜一下比较好解决。

#### 安装细节

##### Fairseq安装

```shell
git clone git@github.com:shuo-git/fairseq-pretrain.git
cd fairseq-pretrain
pip install -e .
```

##### Faiseqscale安装

```shell
git clone git@github.com:facebookresearch/fairscale.git
cd fairscale
pip install -e .
pip install pytest # 否则fairscale会报错
pip uninstall numpy && pip install numpy # 重新安装适配的numpy版本，否则fairscale会报错
```

##### transformers & sentencepiece 安装

```shell
pip install transformers
pip install sentencepiece
export TOKENIZERS_PARALLELISM=false # 解决sentencepiece tokenizer可能死锁的警告
```

##### 测试

该环境在在清华服务器[thu102]上已经测试可以进行语言模型的training和validation。仅测试单机多卡，测试脚本如下：

```shell
docker_data=/data/private/ws/DATASET
docker_code=/home/ws
docker_image=fairseq-pretrain:v0.0
# 进入docker环境
nvidia-docker run --ipc=host --net=host --dns 8.8.8.8 -v $docker_data:$docker_data -v $docker_code:$docker_code -p 6000:6000  -it $docker_image bash
# 进入工作路径
cd /data/private/ws/DATASET/Medical
# 将原始数据处理为二进制文件（无需进行spm和bpe等操作）
fairseq-preprocess --only-source --trainpref head.txt --validpref head.txt --destdir test-data-bin --workers 4
# 语言模型训练，设置--ddp-backend fully_sharded时打开fairscale加速
CUDA_VISIBLE_DEVICES=6,7 fairseq-train test-data-bin --ddp-backend fully_sharded --fp16 --fp16-init-scale 4 --task language_modeling --tokens-per-sample 1024 --batch-size 8 --arch transformer_lm --optimizer adam --adam-betas "(0.9,0.98)" --lr 0.0001 --lr-scheduler polynomial_decay --warmup-updates 5 --total-num-update 10 --max-update 10 --log-format json --log-interval 1
# 假如不想保存checkpoint，上述命令可加"--no-save"
```

#### 后续

1. 解决环境中less/vim/终端中文乱码问题
2. 制作Dockerfile
3. 在智源服务器上运行起来
4. 多机多卡开发
5. FSDP有效性验证

