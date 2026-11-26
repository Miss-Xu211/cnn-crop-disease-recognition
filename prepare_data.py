"""
数据准备脚本：把 PlantVillage 数据集整理成 ImageFolder 可以直接读取的目录结构
=============================================================================
为什么需要这个脚本？
    PlantVillage 原始数据在 HuggingFace 上是以 parquet 格式存放的，
    而 PyTorch 的 torchvision.datasets.ImageFolder 要求下面这种文件夹结构:

        data/plantvillage_5cls/
        ├── train/
        │   ├── Apple_Apple_scab/xxx.jpg
        │   ├── Corn_Common_rust/xxx.jpg
        │   └── ...
        ├── val/  (验证集，结构同上)
        └── test/ (测试集，结构同上)

本脚本做的事:
    1. 从 HuggingFace 镜像下载数据（parquet 文件，已下载则自动跳过）
    2. 只挑选 5 个类别，每类最多取 MAX_PER_CLASS 张，避免数据量过大训练太慢
    3. 按 7:2:1 的比例分层划分 训练集/验证集/测试集
    4. 把图片写成上面那种文件夹结构，之后 train_compare.py 就能直接用

依赖安装:
    pip install datasets pillow
    pip install torch torchvision matplotlib scikit-learn datasets   # 完整环境

运行方式:
    python prepare_data.py            # 第一次运行：下载 + 整理
    python prepare_data.py --force    # 删除旧数据并重新整理
"""

import argparse
import os
import shutil
import urllib.request
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# 使用国内镜像访问 HuggingFace，必须在 import datasets 之前设置才会生效
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset                      # noqa: E402
from sklearn.model_selection import train_test_split   # noqa: E402

# ----------------------------- 可调整参数 ----------------------------------
REPO_ID = "GVJahnavi/PlantVillage_dataset"   # HuggingFace 上的 PlantVillage 数据集
FILES = [                                    # 需要下载的 parquet 文件
    "data/train-00000-of-00002.parquet",
    "data/train-00001-of-00002.parquet",
    "data/test-00000-of-00001.parquet",
]
MAX_PER_CLASS = 1200        # 每个类别最多使用多少张图片（减小可加快训练）
RANDOM_SEED = 42            # 固定的随机种子，保证划分结果可复现
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.2, 0.1

# 本项目使用的 5 个类别（原始标签名 -> 干净的文件夹名）
SELECTED_CLASSES = {
    "Apple__Apple_scab": "Apple_Apple_scab",
    "Corn_(maize)___Common_rust_": "Corn_Common_rust",
    "Potato___Late_blight": "Potato_Late_blight",
    "Tomato___Early_blight": "Tomato_Early_blight",
    "Tomato___healthy": "Tomato_healthy",
}

# PlantVillage 原始 38 个类别的标签名，顺序与 parquet 中的 label 编号一一对应。
# 仅当 parquet 文件没有自带类别名时才会用到。
FULL_CLASS_NAMES = [
    "Apple__Apple_scab", "Apple__Black_rot", "Apple__Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

RAW_DIR = Path("data/_raw")                  # parquet 原始文件目录
TMP_DIR = Path("data/_tmp_all")              # 整理过程中的临时目录
OUT_DIR = Path("data/plantvillage_5cls")     # 最终 ImageFolder 根目录


def download_parquet_files():
    """下载 parquet 文件到本地，已存在且大小正常的文件自动跳过。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        dst = RAW_DIR / Path(fname).name
        if dst.exists() and dst.stat().st_size > 1024 * 1024:
            print(f"[跳过] 已存在: {dst}")
            continue
        # 依次尝试镜像站和官方站
        urls = [
            f"https://hf-mirror.com/datasets/{REPO_ID}/resolve/main/{fname}",
            f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{fname}",
        ]
        for url in urls:
            try:
                print(f"[下载] {url}")
                urllib.request.urlretrieve(url, dst)
                break
            except Exception as e:  # 网络错误时换下一个源
                print(f"  下载失败: {e}")
        else:
            raise RuntimeError(f"无法下载 {fname}，请检查网络后重试")
    return [str(RAW_DIR / Path(f).name) for f in FILES]


def extract_images(parquet_files):
    """
    流式读取 parquet（不会一次性把几十万张图读进内存），
    只保留选中的 5 个类别，按类别存到临时目录。
    """
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    class_dirs = {name: TMP_DIR / name for name in SELECTED_CLASSES.values()}
    for d in class_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # streaming=True 表示一条一条地读，内存占用很小
    ds = load_dataset("parquet", data_files=parquet_files, split="train", streaming=True)

    # 获取标签编号 -> 类别名 的映射
    names = getattr(ds.features["label"], "names", None) or FULL_CLASS_NAMES
    label_ids = {i: n for i, n in enumerate(names)}
    selected_ids = set()
    for idx, raw_name in label_ids.items():
        # 用"去掉多余下划线"的方式匹配，避免数据源命名细节不同导致匹配失败
        if raw_name.replace("_", "") in {k.replace("_", "") for k in SELECTED_CLASSES}:
            selected_ids.add(idx)
    if not selected_ids:
        raise RuntimeError("没有匹配到目标类别，请检查数据集标签名")

    counts = Counter()
    scanned = 0
    for example in ds:
        scanned += 1
        if scanned % 2000 == 0:
            print(f"  已扫描 {scanned} 张，当前各类数量: {dict(counts)}")

        label_id = int(example["label"])
        if label_id not in selected_ids:
            continue

        raw_name = label_ids[label_id]
        # 找到该原始标签对应的干净文件夹名
        clean_name = None
        for k, v in SELECTED_CLASSES.items():
            if k.replace("_", "") == raw_name.replace("_", ""):
                clean_name = v
                break
        if clean_name is None:
            continue

        if counts[clean_name] >= MAX_PER_CLASS:
            continue

        # 保存图片（统一转成 RGB 三通道）
        image = example["image"].convert("RGB")
        save_path = class_dirs[clean_name] / f"{clean_name}_{counts[clean_name]:05d}.jpg"
        image.save(save_path, "JPEG", quality=95)
        counts[clean_name] += 1

        # 每个类别都取够了就提前结束，节省时间
        if all(counts[name] >= MAX_PER_CLASS for name in class_dirs):
            break

    print(f"\n扫描完成，共读取 {scanned} 条，各类别保留数量:")
    for name in class_dirs:
        print(f"  {name:24s}: {counts[name]}")
    return class_dirs


def split_and_move(class_dirs):
    """
    对每个类别内部按 7:2:1 随机划分，然后移动到 train/val/test 目录。
    每个类别都单独划分，保证三个集合的类别分布一致（分层抽样）。
    """
    for split in ["train", "val", "test"]:
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    for class_name, src_dir in class_dirs.items():
        files = sorted(src_dir.glob("*.jpg"))
        if len(files) < 10:
            print(f"[警告] {class_name} 只有 {len(files)} 张，数量过少")

        # 第一次划分：70% 训练集，30% 临时集
        train_files, temp_files = train_test_split(
            files, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_SEED, shuffle=True
        )
        # 第二次划分：把 30% 临时集按 2:1 分成验证集(20%)和测试集(10%)
        val_ratio_in_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        val_files, test_files = train_test_split(
            temp_files, test_size=(1 - val_ratio_in_temp),
            random_state=RANDOM_SEED, shuffle=True
        )

        for split, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            dst_dir = OUT_DIR / split / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.move(str(f), str(dst_dir / f.name))

        print(f"  {class_name:24s} 训练 {len(train_files):4d} / 验证 {len(val_files):4d} / 测试 {len(test_files):4d}")


def main():
    parser = argparse.ArgumentParser(description="准备 PlantVillage 5 类 ImageFolder 数据集")
    parser.add_argument("--force", action="store_true", help="忽略已有结果，强制重新生成")
    args = parser.parse_args()

    # 已经生成过就直接退出，避免重复耗时
    if (OUT_DIR / "train").exists() and any((OUT_DIR / "train").iterdir()) and not args.force:
        print(f"[完成] 数据集已存在: {OUT_DIR.resolve()}")
        print("如需重新生成，请运行: python prepare_data.py --force")
        return

    if args.force:
        for d in [TMP_DIR, OUT_DIR]:
            if d.exists():
                shutil.rmtree(d)

    print("=" * 60)
    print("步骤 1/3: 下载 parquet 文件")
    parquet_files = download_parquet_files()

    print("=" * 60)
    print("步骤 2/3: 抽取 5 个类别的图片")
    class_dirs = extract_images(parquet_files)

    print("=" * 60)
    print("步骤 3/3: 按 7:2:1 划分训练/验证/测试集")
    split_and_move(class_dirs)

    shutil.rmtree(TMP_DIR)   # 清理临时目录

    # 输出最终统计
    print("=" * 60)
    print("数据集准备完成:", OUT_DIR.resolve())
    total = 0
    for split in ["train", "val", "test"]:
        n = sum(1 for _ in (OUT_DIR / split).rglob("*.jpg"))
        total += n
        print(f"  {split:5s}: {n} 张")
    print(f"  总计 : {total} 张，共 {len(SELECTED_CLASSES)} 个类别")


if __name__ == "__main__":
    main()
