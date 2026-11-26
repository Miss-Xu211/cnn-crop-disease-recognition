"""
PlantVillage 5 类图像分类：基础 CNN vs 嵌入 CBAM 的 CNN 对比实验
================================================================
本脚本完成整套实验流程:
    1. 用 ImageFolder 读取 data/plantvillage_5cls 下的 训练/验证/测试 集
    2. 训练集做数据增强（随机水平翻转、随机旋转、亮度对比度微调）
    3. 用完全相同的超参数分别训练 模型A(基础CNN) 和 模型B(CNN+CBAM)
    4. 每个 epoch 输出训练集/验证集的 loss 和准确率
    5. 训练结束后在测试集上计算 精确率(Precision)、召回率(Recall)、F1
    6. 绘制两个模型的 loss 曲线、准确率曲线并保存为图片
    7. 保存两个模型的最优权重 (.pth)

依赖安装（CPU 版，清华源，本作业不需要 CUDA）:
    pip install torch torchvision matplotlib scikit-learn datasets -i https://pypi.tuna.tsinghua.edu.cn/simple
    # 另外需要先运行一次 python prepare_data.py 准备好数据集

运行方式:
    python prepare_data.py       # 第一步：准备数据（只需执行一次）
    python train_compare.py      # 第二步：训练与对比实验
    python train_compare.py --epochs 5   # 想快速试跑时减少轮数
"""

import argparse
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # 无界面绘图后端，避免 PyCharm 中弹窗问题
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import BaselineCNN, CBAMCNN


# ===========================================================================
# 1. 参数配置
# ===========================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="基础CNN vs CNN+CBAM 对比实验")
    parser.add_argument("--data-dir", type=str, default="data/plantvillage_5cls",
                        help="ImageFolder 数据根目录（含 train/val/test）")
    parser.add_argument("--out-dir", type=str, default="outputs",
                        help="权重和曲线图的保存目录")
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    parser.add_argument("--img-size", type=int, default=128, help="输入图片边长")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader 线程数，Windows 建议 0，Linux 可设 4")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def set_seed(seed):
    """固定所有随机种子，保证两个模型拿到相同的数据顺序，对比更公平。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# 2. 数据加载与增强
# ===========================================================================
def build_dataloaders(data_dir, batch_size, img_size, num_workers):
    """
    构建训练/验证/测试 DataLoader。
    训练集使用数据增强，验证集和测试集只做 Resize + 归一化（不能增强，
    否则评估结果不稳定）。
    """
    # ImageNet 常用的均值/标准差，这里输入是 3 通道彩色图，直接用即可
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # 训练集增强：
    #   RandomHorizontalFlip -> 随机水平翻转（叶片左右镜像，标签不变）
    #   RandomRotation        -> 随机旋转 ±15 度
    #   ColorJitter           -> 亮度、对比度各自随机微调 20%
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 验证/测试集：只做确定性的预处理
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    data_root = Path(data_dir)
    train_set = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_set = datasets.ImageFolder(data_root / "val", transform=eval_tf)
    test_set = datasets.ImageFolder(data_root / "test", transform=eval_tf)

    # ImageFolder 按文件夹名的字母顺序生成类别索引，三个集合必须一致
    assert train_set.class_to_idx == val_set.class_to_idx == test_set.class_to_idx, \
        "训练/验证/测试集的类别目录不一致！"

    # pin_memory 是 GPU 加速选项，CPU 训练时设为 False
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False)

    print(f"类别映射: {train_set.class_to_idx}")
    print(f"样本数量 -> 训练: {len(train_set)}, 验证: {len(val_set)}, 测试: {len(test_set)}")
    return train_loader, val_loader, test_loader, train_set.classes


# ===========================================================================
# 3. 训练与评估
# ===========================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch，返回该轮的 平均loss 和 准确率。"""
    model.train()                       # 切换到训练模式（启用 Dropout/BN 更新）
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()           # 清空上一轮梯度
        outputs = model(images)         # 前向传播
        loss = criterion(outputs, labels)   # 交叉熵损失
        loss.backward()                 # 反向传播求梯度
        optimizer.step()                # 更新参数

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()                        # 评估不需要梯度，节省显存和计算
def evaluate(model, loader, criterion, device):
    """在验证/测试集上评估，返回 平均loss、准确率、全部预测和真实标签。"""
    model.eval()                        # 切换到评估模式（固定 BN 统计量）
    total_loss, all_preds, all_labels = 0.0, [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        all_preds.append(outputs.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return total_loss / len(loader.dataset), accuracy_score(labels, preds), preds, labels


# ===========================================================================
# 4. 单个模型的完整训练流程
# ===========================================================================
def run_experiment(model, model_name, loaders, args, device, out_dir):
    """
    训练并测试一个模型。
    返回:
        history      : 每个 epoch 的 loss/acc 记录（用于画曲线）
        test_metrics : 测试集上的各项指标
    """
    train_loader, val_loader, test_loader, class_names = loaders

    # 损失函数与优化器（两个模型使用完全相同的设置）
    criterion = nn.CrossEntropyLoss()                       # 多分类交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None            # 保存验证集准确率最高时的权重
    best_epoch = 0

    print("\n" + "=" * 70)
    print(f"开始训练: {model_name}    参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        # --- 训练一轮 ---
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # --- 验证一轮 ---
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # 记录曲线数据
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 按验证集准确率保存最优权重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[{model_name}] Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"TrainLoss {train_loss:.4f} | TrainAcc {train_acc:.4f} | "
              f"ValLoss {val_loss:.4f} | ValAcc {val_acc:.4f} | "
              f"{time.time() - start:.1f}s")

    # ---------------- 用验证集最优权重在测试集上评估 ----------------
    model.load_state_dict(best_state)
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    # 精确率、召回率、F1（macro 表示对每个类别分别计算后取平均，类别不均衡时更客观）
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds)

    test_metrics = {
        "model": model_name,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }

    print(f"\n----- {model_name} 测试集结果（使用第 {best_epoch} 轮的最优权重）-----")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print(f"Precision(macro): {precision:.4f} | Recall(macro): {recall:.4f} | F1(macro): {f1:.4f}")
    print("\n各类别详细报告:")
    print(classification_report(labels, preds, target_names=class_names, digits=4))
    print("混淆矩阵:")
    print(cm)

    # ---------------- 保存权重 ----------------
    weight_path = out_dir / f"{model_name}_best.pth"
    torch.save(best_state, weight_path)
    print(f"权重已保存: {weight_path.resolve()}")

    return history, test_metrics


# ===========================================================================
# 5. 绘制曲线
# ===========================================================================
def plot_curves(history_a, history_b, out_dir):
    """
    把两个模型的 loss 曲线、准确率曲线分别画在同一张图里，方便对比。
    曲线图的横轴是 epoch，纵轴分别是 loss 和 accuracy。
    """
    epochs = range(1, len(history_a["train_loss"]) + 1)

    # ---- 图1: Loss 曲线 ----
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_a["train_loss"], "o-", color="#1f77b4", label="Model A (CNN) - Train")
    plt.plot(epochs, history_a["val_loss"], "o--", color="#1f77b4", label="Model A (CNN) - Val")
    plt.plot(epochs, history_b["train_loss"], "s-", color="#d62728", label="Model B (CNN+CBAM) - Train")
    plt.plot(epochs, history_b["val_loss"], "s--", color="#d62728", label="Model B (CNN+CBAM) - Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_path = out_dir / "loss_curves.png"
    plt.savefig(loss_path, dpi=200)
    plt.close()

    # ---- 图2: 准确率曲线 ----
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_a["train_acc"], "o-", color="#1f77b4", label="Model A (CNN) - Train")
    plt.plot(epochs, history_a["val_acc"], "o--", color="#1f77b4", label="Model A (CNN) - Val")
    plt.plot(epochs, history_b["train_acc"], "s-", color="#d62728", label="Model B (CNN+CBAM) - Train")
    plt.plot(epochs, history_b["val_acc"], "s--", color="#d62728", label="Model B (CNN+CBAM) - Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    acc_path = out_dir / "accuracy_curves.png"
    plt.savefig(acc_path, dpi=200)
    plt.close()

    print(f"\n曲线图已保存:\n  {loss_path.resolve()}\n  {acc_path.resolve()}")


# ===========================================================================
# 6. 主流程
# ===========================================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    # 本作业统一使用 CPU 训练（安装的是 CPU 版 torch，代码无需 GPU）
    device = torch.device("cpu")
    print(f"使用设备: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 检查数据是否已准备好
    if not (Path(args.data_dir) / "train").exists():
        raise FileNotFoundError(
            f"未找到数据集 {Path(args.data_dir).resolve()}\n"
            f"请先运行: python prepare_data.py"
        )

    # 构建数据加载器（两个模型共用同一份数据划分）
    loaders = build_dataloaders(args.data_dir, args.batch_size, args.img_size, args.num_workers)
    class_names = loaders[3]
    num_classes = len(class_names)

    print("\n实验超参数:")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, "
          f"img_size={args.img_size}, optimizer=Adam, loss=CrossEntropyLoss")

    # ---------- 模型A: 基础 CNN ----------
    # 每个模型开始前都重置随机种子，保证两者读到的数据顺序、初始权重一致
    set_seed(args.seed)
    model_a = BaselineCNN(num_classes=num_classes).to(device)
    history_a, metrics_a = run_experiment(
        model_a, "baseline_cnn", loaders, args, device, out_dir
    )

    # ---------- 模型B: 同样的结构 + CBAM ----------
    set_seed(args.seed)
    model_b = CBAMCNN(num_classes=num_classes).to(device)
    history_b, metrics_b = run_experiment(
        model_b, "cbam_cnn", loaders, args, device, out_dir
    )

    # ---------- 绘图 ----------
    plot_curves(history_a, history_b, out_dir)

    # ---------- 汇总对比 ----------
    print("\n" + "=" * 70)
    print("最终对比结果")
    print("=" * 70)
    header = f"{'Model':<15}{'TestAcc':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}"
    print(header)
    print("-" * len(header))
    for m in [metrics_a, metrics_b]:
        print(f"{m['model']:<15}{m['test_acc']:>10.4f}{m['precision_macro']:>12.4f}"
              f"{m['recall_macro']:>10.4f}{m['f1_macro']:>10.4f}")



if __name__ == "__main__":
    main()
