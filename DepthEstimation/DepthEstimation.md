<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Dense Depth Estimation in Monocular Endoscopy](#dense-depth-estimation-in-monocular-endoscopy)
  - [网络架构](#网络架构)
    - [深度估计网路](#深度估计网路)
    - [其它层](#其它层)
  - [数据集准备](#数据集准备)
  - [损失函数](#损失函数)
    - [Sparse Flow Loss](#sparse-flow-loss)
    - [Depth Consistency Loss](#depth-consistency-loss)
  - [学习率](#学习率)
  - [训练](#训练)
    - [batch size](#batch-size)
    - [结果](#结果)
    - [阶段性结论](#阶段性结论)

<!-- /code_chunk_output -->


# Dense Depth Estimation in Monocular Endoscopy

## 网络架构

### 深度估计网路

输入一张 rgb 图，最终输出一张估计的深度图。

<img src="/img/netArch.jpg" width=80%>

### 其它层

- [x] 深度校准层：利用 __特征点的深度图__ 来校准 __估计的深度图__。
- [x] 流图计算层：利用 __估计的深度图__ 计算出 __稠密的流图__。
- [x] 深度变形层：利用 __估计的深度图__ 互相投影出 __投影的深度层__。

<font color=OrangeRed> 流图计算层与深度变形层均使用 meshgrip 来支持 gpu 加速计算。</font>

## 数据集准备

|数据处理|得到|备注|
|---|---|---|
|初始数据|若干张单目内窥镜图像；相机内参| |
|__SfM预处理__ 特征点检测与匹配；相机运动估计；3D点云重建。|特征点的3D点云；每张图像对应的相机坐标系相对世界坐标系的位姿；。<font color=OrangeRed> 这三条信息等价于：每张图像中特征点的深度。进而可以用于深度估计的损失函数计算。 </font>|有些特征点虽然在图像中可见，但是如果其亮度过明/暗，深度过深/浅，则认为该特征点被污染了。|
|训练过程中，样本是成对使用的，因此需要将数据转换为计算损失函数所需的三条信息。|特征点的深度图；特征点的流图；每张图像中特征点的权重。|相邻几帧中，该特征点出现的次数越多，权重越高。|


## 损失函数

### Sparse Flow Loss

$$ U_k = \frac{Z_j (A_{0,0}U+A_{0,1}V+A_{0,2}) + B_{0,0}}{Z_j (A_{2,0}U+A_{2,1}V+A_{2,2}) + B_{2,0}} $$
$$ V_k = \frac{Z_j (A_{1,0}U+A_{1,1}V+A_{1,2}) + B_{1,0}}{Z_j (A_{2,0}U+A_{2,1}V+A_{2,2}) + B_{2,0}} $$

__推导：__
<img src="/img/derivation_flow.jpg" width=80%>

__特征点的流图__ 中包含了深度估计信息 $Z_j$，因此可用作损失函数。

__sparse flow loss__ 仅计算 frame j 中特征点的 flow loss。

### Depth Consistency Loss

利用相机坐标系的相对位姿关系，两张图像互作投影并插值，比较投影之后的深度图。

## 学习率

使用 cyclic learning rate 更新策略。学习率以 1 epoch 为周期波动。

## 训练

### batch size

* 文中设置 epoch = 100；每个 epoch 中的 iteration = 2000；每个 iteration 中的 __batch size = 8__。
* 显存占用 = 模型占用的显存 (自身参数 + 中间变量) + batch_size x 每个样本的显存占用 (每一层的输入与输出)。笔记本总显存：4GB；模型占用的显存：2.5GB。最终选择 __batch size = 2__。
* batch size 太小带来的 __问题__：参数收敛时会有震荡现象；需要更多次 iteration 才能达到较好的收敛效果。
* 我在每个 epoch 之后，都将模型参数保存为 __.pt 文件__。这样一来，每次重新运行程序时，都能在上一次训练的基础上，继续进行训练。

### 结果

损失函数值随时间下降图：
<img src="/img/loss.png" width=80%>
深蓝色线：overall loss；浅蓝色线：sparse flow loss；红色线：depth consistency loss。
sparse flow loss 权重 20；depth consistency loss 权重 5。

<table>
<tr><th> </th><th>frame j</th><th>frame k</th></tr>
<tr><th>原始图像</td><td rowspan="4"><img src="/img/frame_j.png" width=100%></td><td rowspan="4"><img src="/img/frame_k.png" width=100%></td></tr>
<tr><th>深度估计图</td>
<tr><th>稀疏流图</td>
<tr><th>稠密流图</td>
</table>

其中 __稀疏流图__ 为来自 SfM 的 ground truth。__稠密流图__ 计算自深度估计图，详见 Sparse Flow Loss。流图均用 __HSV__ 色彩空间来表现，H 色调表现流方向，S 饱和度均为255，V 明度表现流强度。

### 阶段性结论

* 由于当前训练集帧数很少，所以在 epoch = 20 时，就已经达到较好的收敛效果，不排除过拟合的情况。
* 针对不同的应用场景，可能需要分别训练模型参数。比如：内窥镜进给食道，膀胱镜进给尿道。
* 下一步需要由我们自己的视频生成数据集，进行训练。



