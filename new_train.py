#train.py
import os
import sys
import time
import datetime
import kornia
from torch import nn
from torch.utils.data import DataLoader
import torch
from MyDataset import MyDataset
from Utils.Loss_function import ssim_loss
import Utils.Loss_function as loss_function
from Utils.Loss_function import Fusionloss
from Utils.Draw_loss_curve import Draw_loss_curve
from Utils.new_net import Encoder, Base, Detail, Restormer_Decoder

# 仅保留融合阶段的参数
criteria_fusion = Fusionloss()

lr = 1e-4
weight_decay = 0
optim_step = 1
optim_gamma = 0.5
epochs = 10
clip_grad_norm_value = 0.01
batch_size = 8

# 仅保留融合阶段的损失参数
ir_vi_ssim = 8
L1_parametric = 2
grad_parametric = 15
decomp_II = 2


def train():
    dataset = MyDataset(train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    loader = {'train': dataloader, }
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    prev_time = time.time()
    print("***********Dataloader Finished***********")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    encoder = Encoder().to(device=device)
    decoder = Restormer_Decoder().to(device=device)
    base = Base().to(device=device)
    detail = Detail().to(device=device)

    # 优化器
    optimizer1 = torch.optim.Adam(
        encoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer2 = torch.optim.Adam(
        decoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer3 = torch.optim.Adam(
        base.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer4 = torch.optim.Adam(
        detail.parameters(), lr=lr, weight_decay=weight_decay)

    # 学习率调度器
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
    scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

    # 损失函数
    MSELoss = loss_function.MSELoss
    Local_SSIMLoss = loss_function.Local_SSIMLoss
    Global_SSIMLoss = loss_function.Global_SSIMLoss
    Gradient_loss = loss_function.Gradient_loss
    cc = loss_function.cc

    mean_loss = []

    # 训练循环 - 仅保留融合阶段
    for epoch in range(epochs):
        Temp_loss = 0
        for index, (ir_img, vi_img) in enumerate(dataloader):
            ir_img, vi_img = ir_img.to(device), vi_img.to(device)

            # 设置模型为训练模式
            encoder.train()
            base.train()
            detail.train()
            decoder.train()

            # 清零梯度
            encoder.zero_grad()
            base.zero_grad()
            detail.zero_grad()
            decoder.zero_grad()

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()

            # 仅执行融合阶段的操作
            ir_encoder = encoder(ir_img)
            vi_encoder = encoder(vi_img)
            ir_B = base(ir_encoder)
            ir_D = detail(ir_encoder)
            vi_B = base(vi_encoder)
            vi_D = detail(vi_encoder)

            # 生成融合特征
#            fusion_feature = decoder(ir_encoder, vi_encoder, ir_B, vi_B, ir_D, vi_D)
            fusion_feature = decoder(ir_B, vi_B, ir_D, vi_D)

            # 计算融合阶段损失
            cc_loss_B = cc(ir_B, vi_B)
            cc_loss_D = cc(ir_D, vi_D)
            loss_decomp = (cc_loss_D) ** 2 / (2.01 + cc_loss_B)

            ir_fusion_gradient_loss = Gradient_loss(kornia.filters.SpatialGradient()(ir_img),
                                                    kornia.filters.SpatialGradient()(fusion_feature))
            vi_fusion_gradient_loss = Gradient_loss(kornia.filters.SpatialGradient()(vi_img),
                                                    kornia.filters.SpatialGradient()(fusion_feature))

            loss_, _, _ = criteria_fusion(ir_img, vi_img, fusion_feature, L1_parametric, grad_parametric)
            ir_vi_SSIM = ssim_loss(ir_img, vi_img, fusion_feature)

            # 总损失
            loss = loss_ + decomp_II * loss_decomp + ir_vi_ssim * ir_vi_SSIM
            Temp_loss = Temp_loss + loss

            # 反向传播与参数更新
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(base.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(detail.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()

            # 打印训练进度
            batches_done = epoch * len(loader['train']) + index
            batches_left = epochs * len(loader['train']) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
                % (
                    epoch + 1,
                    epochs,
                    index + 1,
                    len(loader['train']),
                    loss.item(),
                    time_left,
                )
            )
            sys.stdout.flush()

        # 学习率更新
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
        scheduler4.step()

        # 限制最小学习率
        if optimizer1.param_groups[0]['lr'] <= 1e-6:
            optimizer1.param_groups[0]['lr'] = 1e-6
        if optimizer2.param_groups[0]['lr'] <= 1e-6:
            optimizer2.param_groups[0]['lr'] = 1e-6
        if optimizer3.param_groups[0]['lr'] <= 1e-6:
            optimizer3.param_groups[0]['lr'] = 1e-6
        if optimizer4.param_groups[0]['lr'] <= 1e-6:
            optimizer4.param_groups[0]['lr'] = 1e-6

        # 保存平均损失
        mean_loss.append(Temp_loss / (dataset.__len__() / batch_size))

        # 保存模型 checkpoint
        checkpoint = {
            'encoder': encoder.state_dict(),
            'base': base.state_dict(),
            'detail': detail.state_dict(),
            'decoder': decoder.state_dict(),
        }
        if not os.path.isdir('./model'):
            os.mkdir('./model')
        torch.save(checkpoint, os.path.join(f"./model/NewFuse_{epoch + 1}_" + timestamp + '.pth'))

    # 绘制损失曲线
    Draw_loss_curve(epochs, Mean_Loss=torch.tensor(mean_loss).cpu(), run_time=timestamp)


if __name__ == '__main__':
    train()
