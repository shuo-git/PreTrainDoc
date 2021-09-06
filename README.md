## 预训练项目总览

### 工程任务

#### 数据

* ~~讨论医疗数据格式设计~~
* ~~数据二进制化~~（参考https://fairseq.readthedocs.io/en/latest/getting_started.html#sharding-very-large-datasets）

#### 实验

* 345M模型 + 医疗对话数据
  * 训练参数确定
  * FSDP有效性验证
* 345M模型 + 医疗&通用数据
  * 下游任务评测
  * 不同数据使用方法效果实验，确定最终数据方案
* 多机多卡实现
  * 跑通多机多卡
  * FSDP有效性验证
* 1B模型 + 最终数据方案
  * 下游任务评测
  * 大模型影响实验
* 5B模型 + 最终数据方案
  * 下游任务评测
  * 大模型影响实验

### 研究任务

* 潜在研究方向
  * Datastore-based medical pre-trained model (沁泓介绍)
* 潜在研究任务
  * QA
* 医疗预训练模型存在的问题
  * 落地场景模糊
  * 可解释性/可靠性 差
  * In-domain高质量数据稀缺
