# test.py

from torch.utils.data import DataLoader
from model import *
from data.hvcd_dataset import HVCD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix, classification_report
import time
from utils.loss import SoftLoULoss
from utils.util import *

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print_options(args)
    setup_seed(args.seed)
    print(args)

    test_t1_dir = os.path.join(args.data_root, args.dataset, 'test', 'T1')
    test_t2_dir = os.path.join(args.data_root, args.dataset, 'test', 'T2')
    test_label_dir = os.path.join(args.data_root, args.dataset, 'test', 'label')

    num_test_samples = len(os.listdir(test_t1_dir))
    test_indices = np.arange(1, num_test_samples + 1)

    if args.dataset == 'HVCD':
        test_dataset = HVCD(test_t1_dir, test_t2_dir, test_label_dir, test_indices)
    else:
        raise ValueError(f"Unsupported Dataset: {args.dataset}")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = IFNet(
        input_nc1=360,input_nc2=176,select_nc=3, output_nc=2,
        with_pos='learned', token_trans=True,
        enc_depth=1, dec_depth=1,
        with_decoder_pos='learned', if_upsample_2x=True
    ).to(args.device)
    model_name = args.model_name

    model.load_state_dict(torch.load(os.path.join(args.expr_dir, f'best_{model_name}_model.pth'), map_location=args.device))
    model.eval()
    criterion1 = SoftLoULoss(args.batch_size)
    class_weights = torch.tensor([0.1, 1.0], dtype=torch.float32).to(args.device)
    criterion2 = nn.CrossEntropyLoss(weight=class_weights)
    all_preds = []
    all_labels = []
    ious = []
    test_loss = 0.0
    start_time = time.time()
    with torch.no_grad():
        for t1, t2, labels in test_dataloader:
            t1 = t1.to(args.device)
            t2 = t2.to(args.device)
            labels = labels.to(args.device)
            outputs, edge_out = model(t1, t2)
            edge_gt = Get_binarized_gradientmask_nopadding()(labels.unsqueeze(1))
            loss_io = criterion2(outputs, labels)
            loss_edge = criterion1(edge_out, edge_gt)
            loss = loss_io + loss_edge
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu().numpy().ravel())
            all_labels.append(labels.cpu().numpy().ravel())
            ious.extend(calculate_iou(predicted, labels))
    end_time = time.time()
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=1)
    recall = recall_score(all_labels, all_preds, zero_division=1)
    f1 = f1_score(all_labels, all_preds, zero_division=1)
    kappa = cohen_kappa_score(all_labels, all_preds)
    mean_iou = np.nanmean(ious)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, zero_division=1)

    # ======================
    # Print Results
    # ======================
    print('=' * 50)
    print(f'Test Loss: {test_loss / len(test_dataloader):.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Cohen Kappa: {kappa:.4f}')
    print(f'Mean IoU: {mean_iou:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)
    print('=' * 50)

    print(f'Testing Time: {end_time - start_time:.2f} seconds')

if __name__ == "__main__":
    main()