from tqdm import tqdm
import torch
from torch import nn

#这段代码定义了一个名为 Audio2Exp 的神经网络模型类，它包含一个测试方法。
# 该模型类继承了 nn.Module，并定义了 init 方法来初始化模型的各个组件。
# 在 init 方法中，首先将模型的 netG 组件设置为当前设备上的一个网络。
# 然后设置 prepare_training_loss 参数为 False，表示当前模型不是为了训练而创建的，而是为了测试而创建的。
class Audio2Exp(nn.Module):
    def __init__(self, netG, cfg, device, prepare_training_loss=False):
        super(Audio2Exp, self).__init__()
        self.cfg = cfg
        self.device = device
        self.netG = netG.to(device)

    def test(self, batch):

        mel_input = batch['indiv_mels']                         # bs T 1 80 16
        bs = mel_input.shape[0]
        T = mel_input.shape[1]

        exp_coeff_pred = []

        for i in tqdm(range(0, T, 10),'audio2exp:'): # every 10 frames
            
            current_mel_input = mel_input[:,i:i+10]

            #ref = batch['ref'][:, :, :64].repeat((1,current_mel_input.shape[1],1))           #bs T 64
            ref = batch['ref'][:, :, :64][:, i:i+10]
            ratio = batch['ratio_gt'][:, i:i+10]                               #bs T

            audiox = current_mel_input.view(-1, 1, 80, 16)                  # bs*T 1 80 16

            curr_exp_coeff_pred  = self.netG(audiox, ref, ratio)         # bs T 64 

            exp_coeff_pred += [curr_exp_coeff_pred]

        # BS x T x 64
        results_dict = {
            'exp_coeff_pred': torch.cat(exp_coeff_pred, axis=1)
            }
        return results_dict

#在测试方法中，首先将传入的 batch 数据进行处理，得到每个 batch 中每个时间步的 Mel 映射。然后，创建一个大小为 BS x T x 64 的向量，其中 BS 表示每个 batch 的大小，T 表示每个时间步的数量。接着，使用一个循环遍历每个时间步，对每个时间步进行处理。在处理时，首先将当前时间步的 Mel 映射转换为一个大小为 BS x T x 80 x 16 的向量，其中 80 表示 Mel 映射的维度，16 表示 16 位精度的浮点数表示。然后，使用当前时间步的参考音频和目标音频的比值来生成当前时间步的预测 exp_coeff。最后，将预测的 exp_coeff 向量与当前时间步的 Mel 映射相加，得到一个大小为 BS x T x 64 的向量，该向量就是当前时间步的预测结果。
#最后，将预测结果保存到一个字典中，返回该字典。该字典中包含一个名为 "exp_coeff_pred" 的列，它是一个大小为 BS x T x 64 的向量，表示当前时间步的预测 exp_coeff。
