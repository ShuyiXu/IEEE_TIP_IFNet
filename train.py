# train.py

import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from data.hvcd_dataset import HVCD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix, classification_report
import time
from utils.loss import EdgeLoss
from utils.util import *


def main():

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print_options(args)
    setup_seed(args.seed)
    print(args)

    train_t1_dir = os.path.join(args.data_root, args.dataset, 'train', 'T1')
    train_t2_dir = os.path.join(args.data_root, args.dataset, 'train', 'T2')
    train_label_dir = os.path.join(args.data_root, args.dataset, 'train', 'label')

    test_t1_dir = os.path.join(args.data_root, args.dataset, 'test', 'T1')
    test_t2_dir = os.path.join(args.data_root, args.dataset, 'test', 'T2')
    test_label_dir = os.path.join(args.data_root, args.dataset, 'test', 'label')

    # 获取训练和测试集的索引
    num_train_samples = len(os.listdir(train_t1_dir))
    num_test_samples = len(os.listdir(test_t1_dir))

    train_indices = np.arange(1, num_train_samples + 1)
    test_indices = np.arange(1, num_test_samples + 1)
    # 加载数据集
    if args.dataset == 'HVCD':
        train_dataset = HVCD(train_t1_dir, train_t2_dir, train_label_dir, train_indices)
        test_dataset = HVCD(test_t1_dir, test_t2_dir, test_label_dir, test_indices)
    else:
        raise ValueError(f"Unsupported Dataset: {args.dataset}")

    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = IFNet(
        input_nc1=360,input_nc2=176,select_nc=3, output_nc=2,
        with_pos='learned', token_trans=True,
        enc_depth=1, dec_depth=1,
        with_decoder_pos='learned', if_upsample_2x=True,
        gumbel_temperature=args.gumbel_temperature
    ).to(args.device)
    model_name = args.model_name

    criterion1 = EdgeLoss()
    class_weights = torch.tensor([0.1, 1.0], dtype=torch.float32).to(args.device)
    criterion2 = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    num_epochs = args.epoches
    best_f1 = 0.0  # 初始化最佳F1 Score为0
    # best_train_visualization_data = None  # 存储最佳训练集可视化数据
    # best_test_visualization_data = None  # 存储最佳测试集可视化数据
    start_time = time.time()  # 记录开始时间
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        current_train_visualization_data = []  # 保存当前epoch的训练集可视化数据
        for i, (t1, t2, labels) in enumerate(train_dataloader):
            t1, t2, labels = t1.to(args.device), t2.to(args.device), labels.to(args.device)
            labels = labels.unsqueeze(1)
            outputs, edge_logits = model(t1, t2)
            edge_gt = Get_binarized_gradientmask_nopadding()(labels)
            optimizer.zero_grad()
            loss_io = criterion2(outputs, labels.squeeze(1))
            loss_edge = criterion1(edge_logits, edge_gt)
            loss = loss_io + args.lamda * loss_edge
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1 == 0:
                message = f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}'
                print(message)
                with open(os.path.join(args.expr_dir, 'loss.txt'), 'a') as opt_file:
                    opt_file.write(message + '\n')
            # 保存当前batch的原图、标签和预测结果，用于在F1最优时的可视化
            _, predicted = torch.max(outputs.data, 1)
            current_train_visualization_data.append((t1.cpu().numpy().squeeze(),
                                                     t2.cpu().numpy().squeeze(),
                                                     labels.cpu().numpy().squeeze(),
                                                     predicted.cpu().numpy().squeeze()))
            # torch.cuda.empty_cache()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataloader):.4f}',
              f'Learning Rate: {current_lr:.6f}')
        # 在测试集上评估模型
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []
        current_test_visualization_data = []  # 保存当前epoch的测试集可视化数据
        ious = []
        with torch.no_grad():
            for t1, t2, labels in test_dataloader:
                t1, t2, labels = t1.to(args.device), t2.to(args.device), labels.to(args.device)
                outputs, edge_logits= model(t1, t2)
                edge_gt = Get_binarized_gradientmask_nopadding()(labels.unsqueeze(1))
                loss_io = criterion2(outputs, labels)
                loss_edge = criterion1(edge_logits ,edge_gt)
                loss = loss_io + args.lamda * loss_edge
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.append(predicted.cpu().numpy().ravel())
                all_labels.append(labels.cpu().numpy().ravel())
                # 保存当前batch的原图、标签和预测结果
                current_test_visualization_data.append((t1.cpu().numpy().squeeze(),
                                                        t2.cpu().numpy().squeeze(),
                                                        labels.cpu().numpy().squeeze(),
                                                        predicted.cpu().numpy().squeeze()))
                # 计算IoU
                ious.extend(calculate_iou(predicted, labels))

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=1)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=1)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=1)
        kappa = cohen_kappa_score(all_labels, all_preds)
        mean_iou = np.nanmean(ious)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        class_report = classification_report(all_labels, all_preds, zero_division=1)

        print(f'Test Loss: {test_loss / len(test_dataloader):.4f}')
        print(f'Overall Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Cohen Kappa Score: {kappa:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Mean IoU: {mean_iou:.4f}')
        print(f'Confusion Matrix:\n{conf_matrix}')
        print(f'Classification Report:\n{class_report}')

        # 保存F1最优时的模型和可视化数据
        if f1 > best_f1:
            best_f1 = f1
            best_model_path = os.path.join(args.expr_dir, f'best_{model_name}_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved!")

            save_visualizations(current_train_visualization_data,
                                os.path.join(args.expr_dir, f'{model_name}_best_train'))
            save_visualizations(current_test_visualization_data, os.path.join(args.expr_dir, f'{model_name}_best_test'))

            results_file_path = os.path.join(args.expr_dir, 'evaluation_results.txt')
            with open(results_file_path, 'w') as file:
                file.write(f'Test Loss: {test_loss / len(test_dataloader):.4f}\n')
                file.write(f'Overall Accuracy: {accuracy:.4f}\n')
                file.write(f'Precision: {precision:.4f}\n')
                file.write(f'Recall: {recall:.4f}\n')
                file.write(f'Cohen Kappa Score: {kappa:.4f}\n')
                file.write(f'F1 Score: {f1:.4f}\n')
                file.write(f'Mean IoU: {mean_iou:.4f}\n')
                file.write(f'Confusion Matrix:\n{conf_matrix}\n')
                file.write(f'Classification Report:\n{class_report}\n')
                file.write(f'Best epoch:\n{epoch}\n')
    print('Training finished.')
    end_time = time.time()
    # 计算并输出运行时间
    total_time = end_time - start_time
    print(f"运行时间: {total_time} 秒")
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'Total Training Time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}')
    # # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 保存模型参数量和训练时间
    opt_file_path = os.path.join(args.expr_dir, 'opt.txt')
    with open(opt_file_path, 'a') as opt_file:
        opt_file.write('\n\n')
        opt_file.write(f'Total Parameters: {total_params:,}\n')
        opt_file.write(f'Trainable Parameters: {trainable_params:,}\n')
        opt_file.write(f'Total Training Time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\n')
        opt_file.write(f'Total Training Time (s): {(total_time/num_epochs):,}\n')
    print(f'Total Parameters: {total_params:,}')
    print(f'Trainable Parameters: {trainable_params:,}')

if __name__ == "__main__":
    main()
