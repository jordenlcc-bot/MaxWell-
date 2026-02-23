<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Maxwell Displacement Gating PINN

[](https://github.com/jordenlcc-bot/MaxWell-/blob/main/README.md#maxwell-displacement-gating-pinn)
基于 Maxwell 方程的位移门控物理神经网络实验（1D / 2D）。 项目介绍 这是我个人从零学习 AI 和物理时做的一个练习项目。 我本身不是博士，也不是学术圈的人，之前主要是在汽车行业做生意。 因为对 AI 感兴趣，我一步一步学代码、学 Maxwell 方程，做出了这个位移门控（DispField）的实验。 这个项目里有： 1D / 2D Maxwell PINN 的训练代码 Baseline 和带位移门控（DispField）的模型 一份 REPORT.md，总结了 L2 误差、稀疏度和推荐超参数 一份 paper_draft.md，是准备投稿用的草稿 这个仓库的目标是留下一个真实的学习过程： 我会把现在能跑通、验证过的结果都记录下来，给未来的自己和别人参考。 环境需求 Python 3.10+ PyTorch（支持 CUDA） 一张支持 CUDA 的显卡（例如 RTX 3050） 安装步骤 git clone [https://github.com/jordenlcc-bot/MaxWell-.git](https://github.com/jordenlcc-bot/MaxWell-.git) cd MaxWell- pip install -r requirements.txt
很好，排版已经差不多了，我们现在帮你补完“运行实验”这一块，再顺便把整份 README 整理成一条龙版本，你可以整份换过去。
直接全部 copy 这一段，去覆盖你现在的 README.md：

# Maxwell Displacement Gating PINN

基于 Maxwell 方程的位移门控物理神经网络实验（1D / 2D）。

## 项目介绍

这是我个人从零学习 AI 和物理时做的一个练习项目。
我本身不是博士，也不是学术圈的人，之前主要是在汽车行业做生意。
因为对 AI 感兴趣，我一步一步学代码、学 Maxwell 方程，做出了这个位移门控（DispField）的实验。

这个项目里有：

- 1D / 2D Maxwell PINN 的训练代码
- Baseline 和带位移门控（DispField）的模型
- 一份 `REPORT.md`，总结了 L2 误差、稀疏度和推荐超参数
- 一份 `paper_draft.md`，是准备投稿用的草稿

这个仓库的目标是留下一个真实的学习过程：
我会把现在能跑通、验证过的结果都记录下来，给未来的自己和别人参考。

## 环境需求

- Python 3.10+
- PyTorch（支持 CUDA）
- 一张支持 CUDA 的显卡（例如 RTX 3050）


## 安装步骤

```bash
git clone [https://github.com/jordenlcc-bot/MaxWell-.git](https://github.com/jordenlcc-bot/MaxWell-.git)
cd MaxWell-
pip install -r requirements.txt
运行 1D / 2D 实验
[](https://github.com/jordenlcc-bot/MaxWell-/blob/main/README.md#%E8%BF%90%E8%A1%8C-1d--2d-%E5%AE%9E%E9%AA%8C)
运行 1D 实验：
python run_experiment_1d.py
运行 2D 实验：
python run_experiment_2d.py
运行完成后，可以在 REPORT.md 中查看 1D / 2D 的 L2 误差和稀疏度结果。

好，我们现在把你的 README **整理成干净版**，你只要整份换掉就可以了。

> 步骤：打开 README.md → 全选删除 → 把下面这一整块 paste 进去 → 存档。

```markdown
# Maxwell Displacement Gating PINN

基于 Maxwell 方程的位移门控物理神经网络实验（1D / 2D）。

## 项目介绍

这是我个人从零学习 AI 和物理时做的一个练习项目。  
我本身不是博士，也不是学术圈的人，之前主要是在汽车行业做生意。  
因为对 AI 感兴趣，我一步一步学代码、学 Maxwell 方程，做出了这个位移门控（DispField）的实验。

这个项目里有：

- 1D / 2D Maxwell PINN 的训练代码  
- Baseline 和带位移门控（DispField）的模型  
- 一份 `REPORT.md`，总结了 L2 误差、稀疏度和推荐超参数  
- 一份 `paper_draft.md`，是准备投稿用的草稿  

这个仓库的目标是留下一个真实的学习过程：  
我会把现在能跑通、验证过的结果都记录下来，给未来的自己和别人参考。

## 环境需求

- Python 3.10+
- PyTorch（支持 CUDA）
- 一张支持 CUDA 的显卡（例如 RTX 3050）

## 安装步骤

```bash
git clone https://github.com/jordenlcc-bot/MaxWell-.git
cd MaxWell-
pip install -r requirements.txt
```


## 运行 1D / 2D 实验

运行 1D 实验：

```bash
python run_experiment_1d.py
```

运行 2D 实验：

```bash
python run_experiment_2d.py
```

运行完成后，可以在 `REPORT.md` 中查看 1D / 2D 的 L2 误差和稀疏度结果。

```

```

