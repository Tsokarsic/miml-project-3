1.运行主程序的方式  
本项目的主程序为 grokk_replica/train_grokk.py，调整训练变量的文件为config/train_grokk.yaml  
，调整模型结构的文件为config/model/grokk_model.yaml  
每次运行程序时，在两个config文件中调整相应的参数和模式（调整的方式在文件中均有具体注释来说明).  
再运行train_grokk文件即可得到相应的结果。  
2.每个py文件的主要内容和功能：  
项目的代码部分（即py文件）均集中在grokk_replica文件夹中，每个文件主要的功能如下：  
dataset.py：模运算数据集类的编写，包含2个固定数模加法的初始版本和多个素数相加的数据集,这里采用iterable dataset,每次从中随机抽取新的元素。  
optimizer.py：报告中ASAM优化器的具体实现。  
load_objs.py:加载数据集和模型的接口。  
transformer.py:项目中模型的基本结构，这里为方便统一将transformer，mlp，lstm三个模型的结构均写在此文件中。   
grokk_model.py：项目中用到的模型，包含调用基本模型的接口，损失函数和初始化方式等模型训练方式。  
utils.py：一些所用到的其他函数的编写  
train_grokk.py:项目的主函数，包括数据集的构建，模型和数据的导入和调用，完整的训练过程和结果的可视化（使用wandb)  
3.项目所需要安装的软件包  
本项目采用的python版本为3.11.1（anaconda配置环境下）  
项目环境和工具包的要求在requirements.txt中已经列出.  
其中wandb包的安装和使用详见https://wandb.ai/home, 需要先注册账号并在本项目的终端下登录，如不需使用wandb可视化结果，
将config/train_grokk.yaml中的use_wandb改为false，并注释掉train_grokk中最开始的import wandb即可。





