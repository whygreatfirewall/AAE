import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



######################## 定义超参数 ###########################

EPOCH = 5              # 训练的总轮数
BATCH_SIZE = 50         # 每次训练的batch大小

# LR = 0.0001             # 学习率
LR_GEN = 0.0001         # autoencoder部分编码器
LR_GAN = 0.00005        # GAN部分编码器

N_HIDDEN = 1000         # 隐藏层节点个数
# N_D = 500               # 鉴别器隐藏层节点个数
Z_DIM = 2               # encoder提取的特征维度
X_DIM = 28 * 28         # 输入网络的图像特征维度
CLASS_NUM = 10          # 分类种类个数

DOWNLOAD_MNIST = False  # 是否在当前目录下下载MNIST手写数据集

EPS = 1e-15             # 定义一个小量，防止不可计算的0出现


######################## 数据预处理 ###########################

# 读入训练集
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),     # 只有将该数据放到Data.DataLoader处理，才能发挥ToTensor()/255的能力
    download=DOWNLOAD_MNIST
)

# 读入测试集
test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    # transform=torchvision.transforms.ToTensor(),   # 只有将该数据放到Data.DataLoader处理，才能发挥ToTensor()/255的能力
    download=DOWNLOAD_MNIST
)

# # plot one example
# print(train_data.data.size())
# print(train_data.targets.size())
# print(test_data.data.size())
# print(test_data.targets.size())
# plt.imshow(train_data.data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.targets[0])
# plt.show()

# 这里要注意，train_data这个数据集比较特殊，train_data.data是图片，train_data.label是标签。
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 构造测试数据
test_x = test_data.data.type(torch.FloatTensor)/255.   # value in range(0,1)
test_y = test_data.targets
print(torch.max(train_data.data.view(test_x.size()[0], -1), 1))
print(torch.max(train_data.data))
# print(test_x.size())
# print(test_y.size())





######################## 定义需要的网络 ###########################

# 定义编码器网络
class Encoder(nn.Module):
    def __init__(self, x_dim, N, z_dim):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, N),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(N, z_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

# 定义解码器网络
class Decoder(nn.Module):
    def __init__(self, x_dim, N, z_dim):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, N),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(N, x_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


# 定义监督者(Discriminator:)网路
class Discriminator(nn.Module):
    def __init__(self, N, z_dim):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(z_dim, N),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(N, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        discriminated = self.discriminator(x)
        return discriminated


# 定义softmax输出分类层
class out_softmax(nn.Module):
    def __init__(self, z_dim, label_dim):
        super(out_softmax, self).__init__()

        self.outlayer = nn.Linear(z_dim, label_dim)

    def forward(self, x):
        output = self.outlayer(x)
        return output




######################## 搭建网络 ###########################

# 搭建网络
encoder = Encoder(X_DIM, N_HIDDEN, Z_DIM)
decoder = Decoder(X_DIM, N_HIDDEN, Z_DIM)
D = Discriminator(N_HIDDEN, Z_DIM)
# 同时开启训练模式
encoder = encoder.train()
decoder = decoder.train()
D = D.train()
# 观察网络结构
print(encoder)
print(decoder)
print(D)
# 启用cuda加速
encoder = encoder.cuda()
decoder = decoder.cuda()
D = D.cuda()

######################## 声明优化器和损失函数 ###########################

# 优化器初始化
optimizer_encoder_gen = torch.optim.Adam(encoder.parameters(), lr=LR_GEN)
optimizer_encoder_gan = torch.optim.Adam(encoder.parameters(), lr=LR_GAN)
optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=LR_GEN)
optimizer_D = torch.optim.Adam(D.parameters(), lr=LR_GAN)

# optimizer_encoder_gen = torch.optim.SGD(encoder.parameters(), lr=LR_GEN)
# optimizer_encoder_gan = torch.optim.SGD(encoder.parameters(), lr=LR_GAN)
# optimizer_decoder = torch.optim.SGD(decoder.parameters(), lr=LR_GEN)
# optimizer_D = torch.optim.SGD(D.parameters(), lr=LR_GAN)

# 损失函数初始化
# loss_func = nn.CrossEntropyLoss()
# loss_func = nn.MSELoss()
loss_func_recon = nn.MSELoss()


######################## 搭建训练框架 ###########################

# 搭建训练框架
for epoch in range(EPOCH):
    for step, (x, b_label) in enumerate(train_loader):

        # 把这三个模型的累积梯度清空
        encoder.zero_grad()
        decoder.zero_grad()
        D.zero_grad()
        # 将训练数据加载进cuda显存中
        b_x = x.view(-1, 28*28).cuda()
        b_y = x.view(-1, 28*28).cuda()

        ################ Autoencoder部分 ######################

        # 前向传播
        b_z = encoder(b_x)
        output = decoder(b_z)
        # 计算损失
        loss_recon = loss_func_recon(output, b_y)
        # 后向传播
        optimizer_encoder_gen.zero_grad()
        optimizer_decoder.zero_grad()
        loss_recon.backward()
        optimizer_decoder.step()
        optimizer_encoder_gen.step()

        ################ 指定分布部分 ######################

        # 从正太分布中, 采样real gauss(真-高斯分布样本点)
        z_real_gauss = (5 * torch.randn(x.size()[0], Z_DIM)).cuda()

        ################ GAN的监督者(Discriminator)训练部分 ######################

        # 判别器判别一下真的样本, 得到真样本的鉴别值
        D_real_gauss = D(z_real_gauss)
        # 用encoder 生成假样本, 得到假样本的鉴别值
        encoder.eval()  # encoder切到测试形态, 这时候, encoder不参与优化
        z_fake_gauss = encoder(b_x)
        D_fake_gauss = D(z_fake_gauss)
        # 判别器总误差
        D_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))
        # 仅优化D判别器
        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()

        ################ GAN的生成者(Ganerate)训练部分 ######################

        # encoder充当生成器生成假样本, 得到假样本的鉴别值
        encoder.train()  # 切换训练形态, encoder参与优化
        z_fake_gauss = encoder(b_x)
        D_fake_gauss = D(z_fake_gauss)
        # 判别器总误差
        G_loss = -torch.mean(torch.log(D_fake_gauss + EPS))
        # 仅优化encoder生成器
        optimizer_encoder_gan.zero_grad()
        G_loss.backward()
        optimizer_encoder_gan.step()

        ################ 训练误差显示模块 ######################
        # 每100步显示训练结果
        if step % 100 == 0:
            # encoder = encoder.eval()     # 由于encoder网络中含有dropout层，故需要开启非训练（只读）模式
            # autoencoder_output = decoder(encoder(test_x))
            # pred_y = torch.max(test_output, 1)[1].data.squeeze()
            # accuracy = (pred_y == test_y).sum().item() / test_y.size(0)    #这里sum的用法不太一样
            print('Epoch: ', epoch,
                  '| autoencoder loss: %.4f' % loss_recon.data.item(),
                  '| D_loss:', D_loss.data.item(),
                  '| G_loss:', G_loss.data.item(),
                  )
            # encoder = encoder.train()    # 恢复encoder的训练状态


######################## 存储与读取模块 ###########################
torch.save(encoder.cpu(), '1Gaussion_encoder.pkl')
torch.save(decoder.cpu(), '1Gaussion_decoder.pkl')
torch.save(D.cpu(), '1Gaussion_D.pkl')

model1 = torch.load('1Gaussion_encoder.pkl')
model2 = torch.load('1Gaussion_decoder.pkl')
model3 = torch.load('1Gaussion_D.pkl')

