import os
import json
import time
import copy
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 인텔 Arc GPU 가속기 인식
# ==========================================
try:
    import intel_extension_for_pytorch as ipex
    has_ipex = True
except ImportError:
    has_ipex = False

# ==========================================
# 모델 및 학습 함수 정의 (멀티프로세싱 보호 영역 밖)
# ==========================================
def build_model(experiment_type, num_classes, learning_rate, device):
    if experiment_type == "ResNet18_Finetune":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    elif experiment_type == "ResNet18_Freeze":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    elif experiment_type == "ResNet18_Scratch":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    elif experiment_type == "EfficientNetB0_Finetune":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model.to(device), optimizer

def train_and_eval(model, optimizer, experiment_name, train_loader, test_loader, train_size, test_size, epochs, device):
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'test_acc': [], 'test_precision': [], 'test_recall': []}

    print(f"\n==========================================")
    print(f"🚀 실험 시작: {experiment_name}")
    print(f"==========================================")

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        since = time.time()

        # ----------------- Training Phase -----------------
        model.train()
        running_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / train_size
        history['train_loss'].append(epoch_loss)

        # ----------------- Evaluation Phase -----------------
        model.eval()
        all_preds = []
        all_labels = []

        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]", leave=False)
        with torch.no_grad():
            for inputs, labels in test_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_acc = sum([p == l for p, l in zip(all_preds, all_labels)]) / test_size
        epoch_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        history['test_acc'].append(epoch_acc)
        history['test_precision'].append(epoch_precision)
        history['test_recall'].append(epoch_recall)

        time_elapsed = time.time() - since
        print(f"Epoch {epoch+1}/{epochs} 완료 | 소요시간: {time_elapsed:.0f}초 | "
              f"Train Loss: {epoch_loss:.4f} | Test Acc: {epoch_acc:.4f} | "
              f"Prec: {epoch_precision:.4f} | Rec: {epoch_recall:.4f}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, history, best_acc

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

# ==========================================
# 메인 실행 블록 (Windows 멀티프로세싱 보호)
# ==========================================
if __name__ == '__main__':
    # 1. 하이퍼파라미터 및 디바이스 설정
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # 인텔 Arc GPU 최우선 적용, 없으면 NVIDIA, 둘 다 없으면 CPU
    if has_ipex and hasattr(torch, 'xpu') and torch.xpu.is_available():
        DEVICE = torch.device("xpu")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print(f"[*] 최종 학습 디바이스: {DEVICE}")

    # 2. 데이터셋 다운로드 및 전처리
    print("\n[*] Kaggle에서 포켓몬 데이터셋을 확인 및 다운로드합니다...")
    dataset_path = kagglehub.dataset_download("lantian773030/pokemonclassification")
    data_dir = os.path.join(dataset_path, "PokemonData")
    if not os.path.exists(data_dir):
        data_dir = dataset_path

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(data_dir)
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes

    with open('class_names.json', 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_data = DatasetWrapper(train_dataset, transform=train_transforms)
    test_data = DatasetWrapper(test_dataset, transform=test_transforms)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"[*] Train 데이터 수: {train_size}, Test 데이터 수: {test_size}, 클래스 수: {num_classes}")

    # 3. 실험 실행
    experiment_list = [
        "ResNet18_Finetune",
        "ResNet18_Freeze",
        "ResNet18_Scratch",
        "EfficientNetB0_Finetune"
    ]

    all_results = {}
    global_best_acc = 0.0
    global_best_model_name = ""

    for exp in experiment_list:
        model, optimizer = build_model(exp, num_classes, LEARNING_RATE, DEVICE)
        trained_model, hist, best_acc = train_and_eval(
            model, optimizer, exp, train_loader, test_loader,
            train_size, test_size, EPOCHS, DEVICE
        )
        all_results[exp] = hist

        if best_acc > global_best_acc:
            global_best_acc = best_acc
            global_best_model_name = exp
            torch.save(trained_model.state_dict(), 'best_model.pth')

            with open('best_model_info.json', 'w') as f:
                json.dump({"best_architecture": exp}, f)

    print(f"\n🏆 최종 최고 성능 모델: {global_best_model_name} (Acc: {global_best_acc:.4f})")
    print("[*] 최고 성능 모델의 가중치가 'best_model.pth'로 저장되었습니다.")

    # 4. Learning Curve 시각화
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for exp in experiment_list:
        plt.plot(all_results[exp]['train_loss'], label=exp, marker='o', markersize=4)
    plt.title('Train Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    for exp in experiment_list:
        plt.plot(all_results[exp]['test_acc'], label=exp, marker='s', markersize=4)
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=300)
    print("[*] 학습 곡선이 'learning_curve.png'로 성공적으로 저장되었습니다.")