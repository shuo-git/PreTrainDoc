## Megatron-LM 运行(矫瑞)

### 获取代码

采用DeepSpeed示例提供的Megatron-LM GPT2代码。

```shell
$ git clone https://github.com/microsoft/DeepSpeedExamples.git
$ cd Megatron-LM-v1.1.5-3D_parallelism
```

调整Scheduler适配DeepSpeed（ds对调度器有一个assert，要求必须继承自``_LRScheduler``或者实现``__call__``方法，这里不改会报错呃呃）。

```python
# /Megatron-LM-v1.1.5-3D_parallelism/megatron/learning_rates.py

from torch.optim.lr_scheduler import _LRScheduler

# class AnnealingLR(object):
# change into
class AnnealingLR(_LRScheduler):
```

### 获取环境

```shell
$ docker pull nvcr.io/nvidia/pytorch:20.03-py3
```

该环境不包含DeepSpeed，进入后``pip install deepspeed``后即可运行。

下载`nvidia-docker`使得docker可以运行GPU，参考https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

```shell
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.
$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2
$ sudo systemctl restart docker
```

在docker中可以运行nvidia-smi即可。

### 数据处理

``tools/preprocess_data.py``是自带的数据处理脚本，将json格式输入转成.bin文件，转换过程通过自带的builder实现，我们只需要调整encode的部分就可以，简单的处理raw data的脚本：

```python
import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time

import torch

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args
        with open(args.vocab_file,'r') as f:
            self.voc = json.load(f)

    def initializer(self):
        Encoder.splitter = IdentitySplitter()

    def convert_token_to_id(self,token):
        if token in self.voc:
            return self.voc[token]
        else:
            return self.voc['<unk>']

    def encode(self, line):
        data = line.strip().split()
        ans = [self.convert_token_to_id(_) for _ in data]
        ans.append(self.voc['<|endoftext|>'])
        return ans

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input raw file')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')


    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input)
    with open(args.input, 'r') as f:
        fin = f.readlines()

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, fin, 25)
    #encoded_docs = map(encoder.encode, fin)

    level = "sentence"

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    out_bin_file = "{}_{}_{}.bin".format(args.output_prefix,'text', level)
    out_idx_file = "{}_{}_{}.idx".format(args.output_prefix,'text', level)
    builder = indexed_dataset.make_builder(out_bin_file,impl=args.dataset_impl,vocab_size=tokenizer.vocab_size)
    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for raw in encoded_docs:
        builder.add_item(torch.IntTensor(raw))
        builder.end_document()


    builder.finalize(out_idx_file)

if __name__ == '__main__':
    main()

```

一个处理后可以试验的文件夹在``thumt-119-3:/data/private/jr/DATASET/125w/data_hxc/train``。

### 单机多卡

```bash
$ nvidia-docker run --ipc=host -it -v /data/private/jr/DATASET/:/data/private/jr/DATASET -v /home/jr/connect/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism:/home/jr/connect/Megatron-LM -p 6000:6000 --net=host nvcr.io/nvidia/pytorch:20.12-py3 bash
$ pip install deepspeed
```

数据处理完成后可以调整``examples/pretrain_gpt2_distributed.sh``中的文件路径开始测试，注意train/val/test数据集过小可能会出现问题。

```sh
#! /bin/bash

# Runs the "345M" parameter model
export CUDA_VISIBLE_DEVICES=4,5
GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=10.10.10.102
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/data/private/jr/DATASET/125w/data_hxc/train/nist02-32k_text_sentence
CHECKPOINT_PATH=/data/private/jr/projects/MLM

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       ../pretrain_gpt2.py \
       --model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 8 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file /data/private/jr/DATASET/125w/data_hxc/train/vocab.json \
       --merge-file /data/private/jr/DATASET/125w/data_hxc/train/bpe32k \
       --data-impl mmap \
       --split 900,50,50 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --log-interval 10 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 

set +x
```

### 多机多卡

在多台服务器上同时执行单机多卡脚本，不同之处在于设置`NNODES`参数和各个节点的`RANK`。例如对于主机（119-3），设置

```
NNODES=2
NODE_RANK=0
```

对其他服务器（119-6），设置

```
NNODES=2
NODE_RANK=1
```

### DeepSpeed + 单机多卡

调整``examples/ds_pretrain_gpt2.sh``即可

```sh
#! /bin/bash

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=10.10.10.102
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export DLWS_NUM_WORKER=${NNODES}
export DLWS_NUM_GPU_PER_WORKER=${GPUS_PER_NODE}

DATA_PATH=/data/private/jr/DATASET/125w/data_hxc/train/nist02-32k_text_sentence
VOCAB_PATH=/data/private/jr/DATASET/125w/data_hxc/train/vocab.json
MERGE_PATH=/data/private/jr/DATASET/125w/data_hxc/train/bpe32k
CHECKPOINT_PATH=/data/private/jr/projects/MLM

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
config_json="$script_dir/ds_zero_stage_2_config.json"

# Megatron Model Parallelism
mp_size=4

NLAYERS=24
NHIDDEN=1024
BATCHSIZE=9
LOGDIR="tensorboard_data/${NLAYERS}l_${NHIDDEN}h_${NNODES}n_${GPUS_PER_NODE}g_${mp_size}mp_${BATCHSIZE}b_ds4"

#ZeRO Configs
stage=0
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000

#Actication Checkpointing and Contigious Memory
chkp_layers=1
PA=true
PA_CPU=false
CC=true
SYNCHRONIZE=true
PROFILE=false


gpt_options=" \
        --model-parallel-size ${mp_size} \
        --num-layers $NLAYERS \
        --hidden-size $NHIDDEN \
        --num-attention-heads 16 \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --batch-size $BATCHSIZE \
        --train-iters 500 \
        --lr-decay-iters 320000 \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_PATH \
        --merge-file $MERGE_PATH \
        --data-impl mmap \
        --split 900,50,50 \
        --distributed-backend nccl \
        --lr 1.5e-4 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup 0.01 \
        --checkpoint-activations \
        --log-interval 10 \
        --save-interval 10 \
        --eval-interval 1000 \
        --eval-iters 10 \
        --fp16 \
        --tensorboard-dir ${LOGDIR}
"
  
 deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${stage} \
                --zero-reduce-bucket-size ${rbs} \
                --zero-allgather-bucket-size ${agbs} 
            "

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-reduce-scatter"
fi

chkp_opt=" \
--checkpoint-activations \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --profile-backward"
fi

full_options="${gpt_options} ${deepspeed_options} ${chkp_opt}"

run_cmd="deepspeed --num_nodes ${DLWS_NUM_WORKER} --num_gpus ${DLWS_NUM_GPU_PER_WORKER} ../pretrain_gpt2.py $@ ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
```

### DeepSpeed + 多机多卡

添加一个hostfile形如

```
10.10.10.102 slots=8
10.10.10.105 slots=8
```

调整单机多卡脚本为

```sh
run_cmd="deepspeed --master_port ${MASTER_PORT} -i 10.10.10.102:4,5@10.10.10.105:4,5 --hostfile hostfile ../pretrain_gpt2.py $@ ${full_options}"
```

应该就可以，但是不太会在docker里实现不同服务器免密通信...

### 后续

- 数据对接
- dockerfile（应该只要加一句下载deepspeed就可以）

