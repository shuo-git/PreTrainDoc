## Brain++运行Fairseq

进入root模式，因为我们docker中的配置文件都在/root下（智源平台的历史遗留问题）

```shell
sudo -i
```

配置zsh

```shell
sh /root/install.sh
cat /root/environment.config >> /root/.zshrc
source ~/.zshrc
```

推荐一个zsh主题，通过`vim ~/.zshrc`设置

```shell
# Set name of the theme to load --- if set to "random", it will
# load a random theme each time oh-my-zsh is loaded, in which case,
# to know which specific one was loaded, run: echo $RANDOM_THEME
# See https://github.com/ohmyzsh/ohmyzsh/wiki/Themes
ZSH_THEME="ys"
```

此时默认`fairseq`的安装位置在`/root/fairseq-pretrain`，如果需要安装自己的`fairseq`，参考下面步骤

```shell
pip uninstall fairseq
cd $YOUR_FAIRSEQ_PATH
pip install -e .
```

原智源平台共享空间的数据路径在`/sharefs/wangshuo/98bda4fa79c096c65f5d26be51c7a3ee`

