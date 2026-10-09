import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import cv2
from configs.config import args

def print_options(args):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    # save to the disk
    if not os.path.exists(args.expr_dir):
        os.makedirs(args.expr_dir)

    file_name = os.path.join(args.expr_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Get_binarized_gradientmask_nopadding(nn.Module):
    def __init__(self, threshold=args.threshold):  # 增加一个阈值参数
        super(Get_binarized_gradientmask_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).to(args.device)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).to(args.device)
        self.threshold = threshold  # 保存阈值

    def forward(self, x):
        x0 = x[:, 0].float()  # 只取第一个通道的梯度
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)  # 计算梯度幅值

        # 对梯度幅值进行二值化
        binarized_x0 = torch.where(x0 > self.threshold, torch.tensor(1.0).to(args.device), torch.tensor(0.0).to(args.device))

        return binarized_x0

def calculate_iou(pred, target, num_classes=2):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection

        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / union)

    return ious

def save_visualizations(visualization_data, output_prefix):

    for batch_idx, (t1_batch, t2_batch, labels_batch, preds_batch) in enumerate(visualization_data):
        # 判断 t1_batch 是否为单个样本，即 batch_size == 1 的情况
        if len(t1_batch.shape) == 3:
            # 如果是单个样本，扩展维度，使其兼容后续处理
            t1_batch = np.expand_dims(t1_batch, axis=0)
            t2_batch = np.expand_dims(t2_batch, axis=0)
            labels_batch = np.expand_dims(labels_batch, axis=0)
            preds_batch = np.expand_dims(preds_batch, axis=0)

        batch_size = t1_batch.shape[0]

        for img_idx in range(batch_size):
            t1_np = t1_batch[img_idx]
            t2_np = t2_batch[img_idx]
            labels_np = labels_batch[img_idx]
            preds_np = preds_batch[img_idx]

            # # 对 t1_np 进行 RGB 映射 (169 -> R, 80 -> G, 45 -> B)
            # r_channel_t1 = t1_np[169 if 169 < t1_np.shape[0] else -1, :, :]  # 防止索引超出范围
            # g_channel_t1 = t1_np[80 if 80 < t1_np.shape[0] else -1, :, :]
            # b_channel_t1 = t1_np[45 if 45 < t1_np.shape[0] else -1, :, :]
            # rgb_image_t1 = np.stack([b_channel_t1, g_channel_t1, r_channel_t1], axis=-1)
            # # 对 t2_np 进行 RGB 映射 (81 -> R, 38 -> G, 20 -> B)
            # r_channel_t2 = t2_np[81 if 81 < t2_np.shape[0] else -1, :, :]
            # g_channel_t2 = t2_np[38 if 38 < t2_np.shape[0] else -1, :, :]
            # b_channel_t2 = t2_np[20 if 20 < t2_np.shape[0] else -1, :, :]
            # rgb_image_t2 = np.stack([b_channel_t2, g_channel_t2, r_channel_t2], axis=-1)
            # # 保存 T1 图像
            # t1_output_file = f'{output_prefix}_batch_{batch_idx + 1}_img_{img_idx + 1}_T1.jpg'
            # cv2.imwrite(t1_output_file, (rgb_image_t1 * 255).astype(np.uint8))
            # print(f'Saved T1 to {t1_output_file}')
            # # 保存 T2 图像
            # t2_output_file = f'{output_prefix}_batch_{batch_idx + 1}_img_{img_idx + 1}_T2.jpg'
            # cv2.imwrite(t2_output_file, (rgb_image_t2 * 255).astype(np.uint8))
            # print(f'Saved T2 to {t2_output_file}')
            # 保存标签图像
            label_output_file = f'{output_prefix}_batch_{batch_idx + 1}_img_{img_idx + 1}_Label.jpg'
            cv2.imwrite(label_output_file, (labels_np*255).astype(np.uint8))
            print(f'Saved Label to {label_output_file}')

            # 保存预测结果图像
            pred_output_file = f'{output_prefix}_batch_{batch_idx + 1}_img_{img_idx + 1}_Prediction.jpg'
            cv2.imwrite(pred_output_file, (preds_np*255).astype(np.uint8))
            print(f'Saved Prediction to {pred_output_file}')


