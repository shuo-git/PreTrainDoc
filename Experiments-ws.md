## 实验记录-王硕

### 20210907

* 模型名称：**gpt2small-chunyudialog-zeros12**
* 工作节点：Test-2
* 计算资源：V100 32 GB x **8**
* 实验备注：GPT2 small, chunyu-dialog, ZeRO stage 1&2
* 训练100k步，用时670886.0s (7.76 days)
* 实验路径：`/dataset/98bda4fa/ws/exp/gpt2small-chunyudialog-zeros12`

```shell
# 查看training loss curve
cat train.log | grep "train | {\"epoch\":"
# 查看valid loss curve
cat train.log | grep "valid |"
```

### 20210915 加入泽远收集数据



