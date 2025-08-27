import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import random

def check_data_format(file_path):
    print(f"Đang kiểm tra {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loại dữ liệu: {type(data)}")
    if isinstance(data, dict):
        print("Các khóa trong dict:", list(data.keys()))
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key} shape: {value.shape}, dtype: {value.dtype}")
            else:
                length = len(value) if hasattr(value, '__len__') else 'N/A'
                print(f"  {key} type: {type(value)}, length: {length}")
    elif isinstance(data, tuple):
        print(f"Tuple với {len(data)} phần tử")
        for i, item in enumerate(data):
            if isinstance(item, np.ndarray):
                print(f"  Phần tử {i} shape: {item.shape}, dtype: {item.dtype}")
                print(f"    min: {item.min()}, max: {item.max()}")
    elif isinstance(data, np.ndarray):
        print(f"Mảng shape: {data.shape}, dtype: {data.dtype}")
        print(f"Min: {data.min()}, Max: {data.max()}")
    return data

def visualize_samples(data, IMAGE_DIR):
    if isinstance(data, dict):
        images = data['features']
        labels = data['labels']
    else:
        images, labels = data
    if hasattr(images, 'numpy'):
        images = images.numpy()
    if hasattr(labels, 'numpy'):
        labels = labels.numpy()
    i = random.randint(0, len(images) - 1)
    img = images[i]
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    if img.max() > 1:
        img = img / 255.0
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title(f"Nhãn: {labels[i]}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'sample_images.png'))

def analyze_labels(data, file_key, IMAGE_DIR):
    if isinstance(data, dict):
        labels = data['labels']
    else:
        labels = data[1]
    if hasattr(labels, 'numpy'):
        labels = labels.numpy()
    class_counts = Counter(labels)
    unique_classes = len(class_counts)
    print(f"Số lớp duy nhất: {unique_classes}")
    print(f"Tổng số mẫu: {len(labels)}")
    print("Phân bố nhãn:")
    for cid in sorted(class_counts):
        cnt = class_counts[cid]
        pct = cnt / len(labels) * 100
        print(f"  Nhãn {cid}: {cnt} mẫu ({pct:.2f}%)")
    plt.figure(figsize=(12, 6))
    classes = sorted(class_counts)
    counts = [class_counts[c] for c in classes]
    plt.bar(classes, counts)
    plt.xlabel('Nhãn')
    plt.ylabel('Số lượng mẫu')
    plt.title(f'Phân bố nhãn — {file_key}')
    plt.xticks(classes, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, f'class_distribution_{file_key}.png'))
    return set(class_counts.keys())

def main():
    DATA_DIR = 'data'
    IMAGE_DIR = 'images'
    os.makedirs(IMAGE_DIR, exist_ok=True)
    raw_files = ['train.p', 'valid.p', 'test.p']
    files = [os.path.join(DATA_DIR, fn) for fn in raw_files]
    print('='*50)
    print('KIỂM TRA CÁC FILE DỮ LIỆU')
    print('='*50)
    all_data = {}
    total_classes = set()
    for path in files:
        print(f"\n{'-'*30}")
        if not os.path.exists(path):
            print(f"Không tìm thấy file {path}, bỏ qua.")
            continue
        data = check_data_format(path)
        key = os.path.basename(path)
        all_data[key] = data
        classes = analyze_labels(data, key, IMAGE_DIR)
        total_classes.update(classes)
        print(f"Số lớp trong {key}: {len(classes)}")
    print(f"\n{'-'*30}")
    print('TÓM TẮT:')
    print(f"Tổng số lớp duy nhất trên tất cả file: {len(total_classes)}")
    if total_classes:
        print(f"Khoảng nhãn: {min(total_classes)} đến {max(total_classes)}")
    else:
        print('Không có dữ liệu hợp lệ để xác định khoảng nhãn')
    if 'train.p' in all_data:
        visualize_samples(all_data['train.p'], IMAGE_DIR)

if __name__ == '__main__':
    main()
