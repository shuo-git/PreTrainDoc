## Distributed Training of Fairseq (沁泓)

### 多机并行

基本环境：

health-wangshuo-fairseq-pretrain version: v4

步骤

* 进入服务器后，执行下列指令，以补全dockerfile中缺失的环境配置
  * `sh /root/install.sh` （进入zsh）
  * `cat /root/environment.config >> /root/.zshrc`
  * `source /root/.zshrc`
* 参考`RunFairseq.md`，安装`fairseq-pretrain`，`fairscale`，`transformers`和`sentencepiece`
* 默认登录的是master服务器，通过`~/.ssh/authorized_keys`结合`ifconfig`，可以推出master和slaves的IP地址
  * 比如我尝试时，`authorized_keys`包含`10.42.88.24`和`10.42.90.31`，master IP是`10.42.88.23`，所以有一个slave，且IP是`10.42.90.31`（可以直接ssh登录slave来验证）
* master节点参考脚本如下（nnodes设成总结点数，master上node_rank设为0，master_addr为master的ip，master_port找一个空闲的port即可）

```
data_bin=/dataset/98bda4fa/DATASET/chunyu-dialog/chunyu-dialog-bin
OMP_NUM_THREADS=20
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=2 --node_rank=0 --master_addr="10.42.88.24" \
    --master_port=12345 \
    $(which fairseq-train) $data_bin \
    --ddp-backend fully_sharded --no-reshard-after-forward \
    --fp16 --fp16-init-scale 4 --checkpoint-activations \
    --task language_modeling --tokens-per-sample 1024 --batch-size 16 --update-freq 2 \
    --arch transformer_lm_gpt2_small \
    --optimizer adam --adam-betas "(0.9,0.98)" \
    --weight-decay 1e-2 --clip-norm 1.0 \
    --lr 1.5e-4 --min-lr 1e-5 --lr-scheduler cosine --warmup-updates 3200 \
    --lr-period-updates 96800 --max-update 100000 \
    --save-interval 1 --keep-last-epochs 3 \
    --save-interval-updates 1000 --keep-interval-updates 100 \
    --log-format json --log-interval 1 | tee -a train.log
```

* slave节点脚本在master基础上其他不变，node_rank设为该机器的rank

* 在master和slave上把脚本都跑起来，会自动通信



注意：中止进程之后，其他node上的进程还会占着显存，需要手动去kill



wps统计

| 配置     | WPS   | Batch size per GPU | 现存占用 per GPU |
| -------- | ----- | ------------------ | ---------------- |
| 单机8卡  | ~78k  | 1024x32            | ~31G             |
| 双机16卡 | ~120k | 1024x32            | ~31G             |



### 模型并行

* fairseq里的模型并行要安装megatron submodule

  `git submodule update --init fairseq/model_parallel/megatron`

* 在脚本里加入`--model-parallel-size`，表示total number of GPUs to parallelize model over。试了试设成2，卡死了


