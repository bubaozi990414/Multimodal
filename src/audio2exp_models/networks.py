import torch
import torch.nn.functional as F
from torch import nn
#这段代码定义了一个名为 Conv2d 的 PyTorch 模型类，它继承自 nn.Module。
# 该模型类表示一个卷积神经网络，具有输入通道 (cin)、输出通道 (cout)、卷积核大小 (kernel_size)、卷积核步长 (stride) 和卷积核填充 (padding)。
# 该模型类还定义了 Residual 和 UseAct 两个参数，用于控制模型的残差连接和激活函数是否启用。
# 在__init__方法中，首先导入了 Torch 中的必要的库，然后使用 super() 方法父类的构造函数。
# 接着，定义了 self.conv_block 变量，它是一个 nn.Sequential 对象，它包含了一个 nn.Conv2d 对象和 nn.BatchNorm2d 对象。
# nn.ReLU() 对象是用于激活函数的。
# 在 forward 方法中，首先使用 self.conv_block 将输入 x 进行处理，然后判断是否需要使用残差连接。
# 如果使用残差连接，则将输入 x 加上输出 x 的结果。接着，决定是否需要使用激活函数。如果使用激活函数，则使用 nn.ReLU() 对象对输出进行处理;否则，直接返回输出。
# 需要注意的是，该代码中的 forward 方法没有返回值，因为 PyTorch 中的模型类通常不需要返回输出。
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, use_act = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual
        self.use_act = use_act

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        
        if self.use_act:
            return self.act(out)
        else:
            return out
#这段代码定义了一个名为 SimpleWrapperV2 的 PyTorch 模型类。
# 该模型类是一个包含一个音频编码器的神经网络，音频编码器由多个卷积层组成，并带有残差连接。
# 该模型类用于对音频信号进行编码，可以用于语音识别、音频分类等任务。
class SimpleWrapperV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            )
#这段代码实现了一个基于 PyTorch 语音识别模型的简单实现。它包含了一个预训练的音频编码器和一个线性分类器，用于对语音信号进行分类。
# 首先，代码中调用了 torch.load() 函数，加载了一个名为 wav2lip.pth 的 checkpoint 文件。该文件包含了一个预训练的音频编码器模型的权重和偏置。
# 接着，代码中定义了一个名为 self.audio_encoder 的 nn.Module 对象，它包含了一个 nn.Conv2d 对象和一些卷积层和残差连接。然后，代码中使用 nn.load_state_dict() 函数将 wav2lip_state_dict 中的权重和偏置加载到 self.audio_encoder 对象中。
# 最后，代码中定义了一个名为 self.mapping1 的 nn.Linear 对象，用于将输入特征向量映射到指定维度的向量空间中。然后，代码中使用 nn.init.constant_() 函数将 self.mapping1 对象的权重和偏置初始化为 0。
# 在 forward 方法中，代码首先将输入语音信号 x、参考语音信号 ref 和音量 ratio 传递给 self.audio_encoder 对象，并将其转换为一维向量。然后，将这三个向量传递给 self.mapping1 对象，得到最终的输出结果。输出结果被重构成与输入信号相同的维度，并加上参考信号 ref 的维度，以形成一个三维向量。最后，代码将输出结果返回。
        ## load the pre-trained audio_encoder
        #self.audio_encoder = self.audio_encoder.to(device)  
        '''
        wav2lip_state_dict = torch.load('/apdcephfs_cq2/share_1290939/wenxuazhang/checkpoints/wav2lip.pth')['state_dict']
        state_dict = self.audio_encoder.state_dict()

        for k,v in wav2lip_state_dict.items():
            if 'audio_encoder' in k:
                print('init:', k)
                state_dict[k.replace('module.audio_encoder.', '')] = v
        self.audio_encoder.load_state_dict(state_dict)
        '''

        self.mapping1 = nn.Linear(512+64+1, 64)
        #self.mapping2 = nn.Linear(30, 64)
        #nn.init.constant_(self.mapping1.weight, 0.)
        nn.init.constant_(self.mapping1.bias, 0.)

    def forward(self, x, ref, ratio):
        x = self.audio_encoder(x).view(x.size(0), -1)
        ref_reshape = ref.reshape(x.size(0), -1)
        ratio = ratio.reshape(x.size(0), -1)
        
        y = self.mapping1(torch.cat([x, ref_reshape, ratio], dim=1)) 
        out = y.reshape(ref.shape[0], ref.shape[1], -1) #+ ref # resudial
        return out
