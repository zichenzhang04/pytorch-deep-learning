# Deep Learning in PyTorch
Last Updated on **Jul 2023**



## Simple Neural Networks



### Read in datasets

Linear regression model:

```Python
from torch.utils import data

#is_train表示是否希望数据迭代器对象在每个迭代周期内打乱数据
def load_array(data_arrays, batch_size, is_train=True):
    #data_arrays是元祖，*data_arrays将元祖解压为独立参数
    dataset = data.TensorDataset(*data_arrays)
    #挑选样本
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

#为了验证是否正常工作，让我们读取并打印第一个小批量样本。这里我们使用iter构造Python迭代器，并使用next从迭代器中获取第一项
batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))
```



### Define our model

Linear regression model:

```Python
from torch import nn

#我们将两个参数传递到nn.Linear中。 第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1
input_shape = 2
output_shape = 1
net = nn.Sequential(
  nn.Linear(input_shape, output_shape)
)
```

Softmax regression model:

```Python
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
```



### Define parameters

Linear regression model:

```python
#通过net[0]选择网络中的第一个图层，然后使用weight.data和bias.data方法访问参数。我们还可以使用替换方法normal_和fill_来重写参数值
net[0].weight.data.normal_(0, 0.01) #normal_(): normal distribution
net[0].bias.data.fill_(0) #fill_(): set values
```

Softmax regression model:

```python
def init_weights(m):
		#对线性层初始化
    if type(m) == nn.Linear:
      	#我们以均值0和标准差0.01随机初始化权重
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```



### Define loss functions

Linear regression model:

```python
#计算均方误差使用的是MSELoss类，也称为平方L2范数。默认情况下，它返回所有样本损失的平均值。
loss = nn.MSELoss()
```

Softmax regression model:

```Python
#交叉熵损失
loss = nn.CrossEntropyLoss(reduction='none')
```



### Define optimization algorithm

**Linear** and **Softmax** regression model: 小批量随机梯度下降算法是一种优化神经网络的标准工具，PyTorch在`optim`模块中实现了该算法的许多变种

```Python
#当我们实例化一个SGD实例时，我们要指定优化的参数 （可通过net.parameters()从我们的模型中获得）以及优化算法所需的超参数字典。小批量随机梯度下降只需要设置lr值，这里设置为0.03
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```



### Train our model

在每个迭代周期里，我们将完整遍历一次数据集（`train_data`）， 不停地从中获取一个小批量的输入和相应的标签。 对于每一个小批量，我们会进行以下步骤:

- 通过调用`net(X)`生成预测并计算损失`l`（前向传播）
- 通过进行反向传播来计算梯度
- 通过调用优化器来更新模型参数

Linear regression model:

```Python
num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad() #clear gradient
        l.backward() #no need to add sum() since Python has completed that for you
        trainer.step() #update the model (weights and biases)
    #calculate the loss at this loop step
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

Softmax regression model:

```python
"""训练模型一个迭代周期, updater是更新模型参数的常用函数，它接受批量大小作为参数。 它可以是d2l.sgd函数，也可以是框架的内置优化函数"""
def train_epoch(net, train_iter, loss, updater): 
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # metric[0]训练损失总和、metric[1]训练准确度总和、metric[2]样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        # 使用PyTorch内置的优化器和损失函数
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else: # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train(net, train_iter, test_iter, loss, num_epochs, updater):
  	#在动画中绘制数据的实用程序类Animator(可忽略)
    #animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    
num_epochs = 10
train(net, train_iter, test_iter, loss, num_epochs, trainer)
```

