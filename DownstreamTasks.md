## 下游任务评测（沁泓&泽远）

### 实验准备

#### 登陆智源服务器

登陆需要安装的包请见群里发的`智源算力平台SSH登录.pdf`

```shell
# Linux
ssh -o ProxyCommand="ncat --proxy-type http --proxy sshproxy.platform.baai.ac.cn:32321 %h %p" root@yourip
# Windows
ssh -o "ProxyCommand=nc -X connect -x sshproxy.platform.baai.ac.cn:32321 %h %p" root@yourip
# Mac
ssh -o ProxyCommand="corkscrew sshproxy.platform.baai.ac.cn 32321 %h %p" root@yourip
```

#### 模型

使用`/dataset/98bda4fa/ws/exp/gpt2small-chunyudialog-zeros12`的模型，建议使用`checkpoints/checkpoint_best.pt`

#### 数据

位置

`/dataset/98bda4fa/DATASET/medical-yzy`

预处理

```shell
dict=/dataset/98bda4fa/DATASET/chunyu-dialog/chunyu-dialog-bin/dict.txt
fairseq-preprocess --only-source \
  --srcdict $dict \
  --trainpref [YOURTRAIN] \
  --validpref [YOURVALID] \
  --testpref [YOURTEST] \
  --destdir [YOURDEST]/data-bin \
  --workers 4
```

注意需要事先将`~/mT5_tokenizer`拷贝到运行`fairseq-preprocess`的路径下。

#### 代码

请在分支`downstream`下进行开发，如有需求，可以自行新建分支。