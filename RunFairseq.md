## Fairseq+Fairscale运行（王硕）

### 代码

#### 位置

https://github.com/shuo-git/fairseq-pretrain

Fork Fairseq最新版本，已实现对Fairscale的集成。

#### 开发记录

###### commit 7c2f5128

嵌入Huggingface mT5的tokenizer，可以直接处理plain text。现在实现的版本比较丑陋，需要在代码里手动预设mT5 tokenizer的位置，后续需要进一步优化。目前已验证可用的tokenizer路径为：[thu102] /data/private/ws/DATASET/Medical/mT5_tokenizer

###### commit 64819ef5

实现了Megatron-LM的learning rate schedule，使用方式为`--lr-scheduler cosine-megatron`。

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

##### 安装过程

已更新到Dockerfile中

###### Fairseq安装

```shell
git clone git@github.com:shuo-git/fairseq-pretrain.git
cd fairseq-pretrain
pip install -e .
```

###### Faiseqscale安装

```shell
git clone git@github.com:facebookresearch/fairscale.git
cd fairscale
pip install -e .
pip install pytest # 否则fairscale会报错
pip uninstall numpy && pip install numpy # 重新安装适配的numpy版本，否则fairscale会报错
```

###### transformers & sentencepiece 安装

```shell
pip install transformers
pip install sentencepiece
export TOKENIZERS_PARALLELISM=false # 解决sentencepiece tokenizer可能死锁的警告
```

###### 测试

该环境在在清华服务器[thu102]上已经测试可以进行语言模型的training和validation。仅测试单机多卡，测试脚本如下：

```shell
docker_data=/data/private/ws/DATASET
docker_code=/home/ws
docker_image=fairseq-pretrain:v0.4
# 进入docker环境
nvidia-docker run --ipc=host --net=host --dns 8.8.8.8 -v $docker_data:$docker_data -v $docker_code:$docker_code -p 6000:6000  -it $docker_image bash
# 进入工作路径
cd /data/private/ws/DATASET/Medical
# 将原始数据处理为二进制文件（无需进行spm和bpe等操作）
fairseq-preprocess --only-source --trainpref head.txt --validpref head.txt --destdir test-data-bin --workers 4
# 语言模型训练，设置--ddp-backend fully_sharded时打开fairscale加速
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train chunyu-dialog-bin --ddp-backend fully_sharded --fp16 --fp16-init-scale 4 --task language_modeling --tokens-per-sample 2048 --batch-size 8 --arch transformer_lm --optimizer adam --adam-betas "(0.9,0.98)" --lr 0.0001 --lr-scheduler polynomial_decay --warmup-updates 5 --total-num-update 10000 --max-update 10000 --log-format json --log-interval 1
# 假如不想保存checkpoint，上述命令可加"--no-save"
```

###### 训练

```shell
# 345M model training 超参设置参考 https://github.com/NVIDIA/Megatron-LM/blob/3860e995269df61d234ed910d4756e104e1ab844/examples/pretrain_gpt.sh
# --ddp-backend 设置为fully_sharded时等效开启ZeRO stage 1
# --no-reshard-after-forward 打开时等效ZeRO stage 2
data_bin=/dataset/98bda4fa/DATASET/chunyu-dialog/chunyu-dialog-bin
OMP_NUM_THREADS=20 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    fairseq-train $data_bin \
    --ddp-backend fully_sharded --no-reshard-after-forward \
    --fp16 --fp16-init-scale 4 --checkpoint-activations \
    --task language_modeling --tokens-per-sample 1024 --batch-size 32 --update-freq 2 \
    --arch transformer_lm_gpt2_small \
    --optimizer adam --adam-betas "(0.9,0.98)" \
    --weight-decay 1e-2 --clip-norm 1.0 \
    --lr 1.5e-4 --min-lr 1e-5 --lr-scheduler cosine-megatron --warmup-updates 3200 \
    --lr-period-updates 96800 --max-update 100000 \
    --save-interval 1 --keep-last-epochs 3 \
    --save-interval-updates 1000 --keep-interval-updates 100 \
    --log-format json --log-interval 1 | tee -a train.log
```

##### 制作Dockerfile

```dockerfile
FROM fairseq-pretrain:v0.0

ENV DEBIAN_FRONTEND noninteractive

COPY vimrc /root/.vimrc
COPY tmux.conf /root/.tmux.conf

# sources.list为对应ubuntu版本镜像源
COPY sources.list /etc/apt/sources.list
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    tree sysstat mtr ntpdate dos2unix zip unzip zsh
RUN apt-get remove -y librdmacm1 libibverbs1 ibverbs-providers && apt-get install -y librdmacm1 libibverbs1 ibverbs-providers
# RUN apt-get -y -o Dpkg::Options::="--force-overwrite" install ibverbs-providers

RUN pip --no-cache-dir install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip --no-cache-dir install jupyter notebook keras sklearn pandas matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN python -m ipykernel install --user --name base --display-name "Python3.8"

COPY prepare.sh /root/.jupyter/prepare.sh
RUN chmod +x /root/.jupyter/prepare.sh && mkdir -p /dataset /workspace /logs /model
EXPOSE 8888

RUN pip install future typing packaging -i https://pypi.tuna.tsinghua.edu.cn/simple
# 安装openmpi
# RUN mkdir /tmp/openmpi && \
#     cd /tmp/openmpi && \
#     wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
#     tar zxf openmpi-4.0.0.tar.gz && \
#     cd openmpi-4.0.0 && \
#     ./configure --enable-orterun-prefix-by-default && \
#     make -j $(nproc) all && \
#     make install && \
#     ldconfig && \
#     rm -rf /tmp/openmpi

# 安装fairseq
ADD fairseq-pretrain /root/fairseq-pretrain
WORKDIR /root/fairseq-pretrain
RUN pip install -e ./ -i https://pypi.tuna.tsinghua.edu.cn/simple
# 安装fairscale
ADD fairscale /root/fairscale
WORKDIR /root/fairscale
RUN pip install -e ./ -i https://pypi.tuna.tsinghua.edu.cn/simple && pip uninstall --yes numpy && pip install --no-input numpy pytest -i https://pypi.tuna.tsinghua.edu.cn/simple
# 安装transformers & sentencepiece
RUN pip install transformers sentencepiece -i https://pypi.tuna.tsinghua.edu.cn/simple

ADD mT5_tokenizer /root/mT5_tokenizer

# 安装ssh服务
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config &&\
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config &&\
    sed -i 's/Port 2222/Port 22/g' /etc/ssh/sshd_config &&\
    sed -i 's/PasswordAuthentication no/#PasswordAuthentication no/g' /etc/ssh/sshd_config
EXPOSE 22


# 安装horovod，支持分布式部署
# RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
#     pip install --no-cache-dir horovod==0.20.0 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
#     ldconfig
# CMD ["/bin/bash"]

# 配置oh-my-zsh
WORKDIR /root
ADD oh-my-zsh /root/.oh-my-zsh
ADD install.sh /root/install.sh
ADD environment.config /root/environment.config
# 登陆docker之后，如需要安装oh-my-zsh: 1. sh /root/install.sh; 2. cat /root/environment.config >> /root/.zshrc; 3. source /root/.zshrc
```

##### 导出镜像

```shell
docker save fairseq-pretrain:v0.4 | gzip > health-fairseq-v0.4.tar.gz
```

### 后续

1. ~~解决环境中less/vim/终端中文乱码问题~~
2. ~~制作Dockerfile~~
3. ~~在智源服务器上运行起来~~
4. ~~FSDP有效性验证~~
5. 多机多卡开发

