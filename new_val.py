import os
import numpy as np
from torch import nn
import torch
from Utils.Image_read_and_save import img_save, image_read_cv2
from Utils.Valuation import Valuation
from Utils.new_net import Base, Encoder, Detail, Restormer_Decoder  # 确保导入的解码器支持四通道输入

# 四通道消融实验权重路径（重点修改）
model_pth = './model/NewFuse_10_10-10-15-10.pth'  # 四通道模型权重
Model_Name = 'MBSFuse_ablation_four_channel'  # 消融实验模型名称，用于区分结果


def val():
    for dataset_name in ['TNO', 'RoadScene', 'MSRS', 'MRI_CT']:
        print(f"The test result of {dataset_name} (four-channel ablation):")
        test_folder = os.path.join('test_images', dataset_name)
        # 结果保存路径添加消融实验标识，避免与原结果混淆
        test_out_folder = os.path.join('Results', f"{dataset_name}_ablation_four_channel")
        os.makedirs(test_out_folder, exist_ok=True)  # 确保保存目录存在

        # 设备选择
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 初始化模型（与原代码一致，但解码器将适配四通道输入）
        encoder = Encoder().to(device=device)  # 即使不用encoder输出，仍需加载权重保证Base/Detail输入维度匹配
        decoder = Restormer_Decoder().to(device=device)
        base = Base().to(device=device)
        detail = Detail().to(device=device)

        # 加载权重
        checkpoint = torch.load(model_pth, weights_only=True)
        encoder.load_state_dict(checkpoint['encoder'])
        base.load_state_dict(checkpoint['base'])
        detail.load_state_dict(checkpoint['detail'])
        decoder.load_state_dict(checkpoint['decoder'])

        # 切换为验证模式
        encoder.eval()
        base.eval()
        detail.eval()
        decoder.eval()

        with torch.no_grad():
            for img_name in os.listdir(os.path.join(test_folder, 'ir')):
                # 读取并预处理图像（与原代码一致）
                data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[
                              None, None, ...] / 255.0
                data_VIS = image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='GRAY')[
                               None, None, ...] / 255.0
                ir_img, vi_img = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS)
                ir_img, vi_img = ir_img.to(device), vi_img.to(device)

                # 特征提取（与原代码一致，但仅使用Base和Detail的输出）
                ir_encoder = encoder(ir_img)  # 用于生成Base/Detail输入，不直接传入解码器
                vi_encoder = encoder(vi_img)
                ir_B = base(ir_encoder)
                ir_D = detail(ir_encoder)
                vi_B = base(vi_encoder)
                vi_D = detail(vi_encoder)

                # 重点修改：解码器仅输入双分支四个特征（ir_B, vi_B, ir_D, vi_D）
                fusion_feature = decoder(ir_B, vi_B, ir_D, vi_D)

                # 后处理与保存（与原代码一致）
                data_normalized = (fusion_feature - torch.min(fusion_feature)) / (
                            torch.max(fusion_feature) - torch.min(fusion_feature))
                data_scaled = (data_normalized * 255).cpu().numpy()
                fi = np.squeeze(data_scaled).astype(np.uint8)
                img_save(fi.astype(np.uint8), img_name.split(sep='.')[0], test_out_folder)

        # 指标计算（与原代码一致，确保公平对比）
        eval_folder = test_out_folder
        ori_img_folder = test_folder
        metric_result = np.zeros((9))
        img_count = len(os.listdir(os.path.join(ori_img_folder, "ir")))

        for img_name in os.listdir(os.path.join(ori_img_folder, "ir")):
            ir = image_read_cv2(os.path.join(ori_img_folder, "ir", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder, "vi", img_name), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0] + ".png"), 'GRAY')

            metric_result += np.array([
                Valuation.EN(fi),
                Valuation.SD(fi),
                Valuation.SF(fi),
                Valuation.MI(fi, ir, vi),
                Valuation.SCD(fi, ir, vi),
                Valuation.VIFF(fi, ir, vi),
                Valuation.Qabf(fi, ir, vi),
                Valuation.MSE(fi, ir, vi),
                Valuation.SSIM(fi, ir, vi)
            ])

        metric_result /= img_count
        # 打印指标（格式与原代码一致，便于对比）
        print("\t\t\t EN\t\t  SD\t SF\t     MI\t     SCD\t VIF\t Qabf\t MSE\t SSIM")
        print(f"{Model_Name}" + '\t' +
              f"{np.round(metric_result[0], 2)}\t" +
              f"{np.round(metric_result[1], 2)}\t" +
              f"{np.round(metric_result[2], 2)}\t" +
              f"{np.round(metric_result[3], 2)}\t" +
              f"{np.round(metric_result[4], 2)}\t" +
              f"{np.round(metric_result[5], 2)}\t" +
              f"{np.round(metric_result[6], 2)}\t" +
              f"{np.round(metric_result[7], 2)}\t" +
              f"{np.round(metric_result[8], 2)}"
              )


if __name__ == '__main__':
    val()