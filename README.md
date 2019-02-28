# CartoonGan
基于生成式对抗网络和pytorch实现随机生成卡通头像
============================================
这是我在学习pytorch和Gan的时候跟着教程完成的一个小demo，主要为了验证生成式对抗网络随机生成的神奇效果。
data数据集：数据集是我从一个网站上面爬下来的动漫图像，爬虫代码已经附上，直接运行即可。形如下面的图像
![数据图像](https://github.com/buptzhang0414/CartoonGan/blob/master/00ecf969dd1e8148a5121d05adcfb6c1.jpg)

或者直接在百度云上面下载
链接：
[百度云盘](https://pan.baidu.com/s/1TA0ejM21cBeVfjwqEAxlFA ) 
提取码：r1l7 

将下载好的数据集直接放在/data下面即可。

[model.py](https://github.com/buptzhang0414/CartoonGan/blob/master/CartoonGan/model.py )这里面定义了生成器和判别器的网络结构，使用pytorch实现很方便简单。

```python
class NetG(nn.Module):
    #生成器定义
    def __init__(self, opt):
        super(NetG, self).__init__()
        ngf = opt.ngf  # 生成器feature map数
        self.main = nn.Sequential(
            # 输入是一个nz维度的噪声，我们可以认为它是一个1*1*nz的feature map
            nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 上一步的输出形状：(ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 上一步的输出形状： (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 上一步的输出形状： (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 上一步的输出形状：(ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()  # 输出范围 -1~1 故而采用Tanh
            # 输出形状：3 x 96 x 96
        )

    def forward(self, input):
        return self.main(input)
```

```python
class NetD(nn.Module):
    
    #判别器定义
    

    def __init__(self, opt):
        super(NetD, self).__init__()
        ndf = opt.ndf
        self.main = nn.Sequential(
            # 输入 3 x 96 x 96
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf) x 32 x 32

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # 输出一个数(概率)
        )

    def forward(self, input):
        return self.main(input).view(-1)
```

[main.py](https://github.com/buptzhang0414/CartoonGan/blob/master/CartoonGan/main.py)里面定义了两个主要的函数，一个用于训练train,另一个用于测试generate，还有一些参数设置也在main.py里面定义了。

转到main.py目录下面，直接运行
```
python main.py train
```
即可直接按照默认进行训练。

测试的话可以直接
```
python main.py generate
```

![结果图像](https://github.com/buptzhang0414/CartoonGan/blob/master/result.png)

这是训练200代之后生成的结果图，可以看到基本的卡通头像轮廓已经有了，但是由于网络结构比较简单，所以生成的质量不是很高，这个demo仅仅是为了演示GAN的神奇生成作用。
