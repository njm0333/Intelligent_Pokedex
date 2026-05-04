import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

st.set_page_config(page_title="Pokemon Classifier", page_icon="🔍")

# ==========================================
# 1. 초기화 및 리소스 로드
# ==========================================
@st.cache_resource
def load_assets():
    """클래스 이름과 모델 아키텍처 정보를 로드합니다."""
    if not os.path.exists('class_names.json') or not os.path.exists('best_model_info.json') or not os.path.exists('best_model.pth'):
        return None, None, "필수 파일이 없습니다. 1_train.py를 먼저 실행하여 모델을 학습시켜주세요."

    with open('class_names.json', 'r', encoding='utf-8') as f:
        class_names = json.load(f)

    with open('best_model_info.json', 'r', encoding='utf-8') as f:
        info = json.load(f)
        best_arch = info.get("best_architecture")

    return class_names, best_arch, None

class_names, best_arch, error_msg = load_assets()

if error_msg:
    st.error(error_msg)
    st.stop()

num_classes = len(class_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 모델 재구성 및 가중치 삽입
# ==========================================
@st.cache_resource
def load_pytorch_model(arch_name, num_cls):
    """학습 스크립트에서 저장한 아키텍처 이름에 맞추어 모델 뼈대를 만들고 가중치를 입힙니다."""
    if "ResNet18" in arch_name:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_cls)
    elif "EfficientNetB0" in arch_name:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_cls)
    else:
        raise ValueError(f"알 수 없는 모델 구조입니다: {arch_name}")

    # 가중치 로드
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval() # 추론 모드 전환 필수 [cite: 1115]
    return model

model = load_pytorch_model(best_arch, num_classes)

# ==========================================
# 3. 이미지 전처리 정의
# ==========================================
# 학습 시 설정했던 Test Transform과 동일한 로직을 적용합니다. [cite: 1110, 1111, 1112, 1113, 1114]
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 4. Streamlit UI 렌더링
# ==========================================
st.title("Pokédex: 포켓몬 분류기 🔍")
st.markdown(f"**적용된 인공지능 모델:** `{best_arch}` (전이학습 성능 1위 모델)")
st.write("강의 슬라이드에 명시된 요구사항에 따라, 이미지를 입력하면 **Top-5 예측 결과**를 보여줍니다.")

uploaded_file = st.file_uploader("포켓몬 이미지를 업로드하세요 (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1) 이미지 화면에 출력
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="업로드된 포켓몬 이미지", use_container_width=True)

    st.write("🔄 모델 분석 중...")

    # 2) 텐서 변환 및 추론 [cite: 1115, 1116, 1117]
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        # 3) 확률값 계산을 위한 Softmax 적용 [cite: 1118]
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # 4) 가장 확률이 높은 상위 5개 추출 [cite: 8, 14, 1119]
        top5_prob, top5_catid = torch.topk(probabilities, 5)

    # 5) 강의 슬라이드 양식과 동일하게 결과 출력 [cite: 9, 10, 11, 12, 13, 1129, 1130, 1131, 1132, 1133]
    st.subheader("Top-5 predictions:")

    # 가독성을 위한 컨테이너 박스 생성
    with st.container():
        for i in range(5):
            idx = top5_catid[i].item()
            prob = top5_prob[i].item() * 100
            name = class_names[idx]

            # 1위 결과는 눈에 띄게 표시
            if i == 0:
                st.success(f"**{i+1}. {name} ({prob:.2f}%)** 🏆")
            else:
                st.write(f"{i+1}. {name} ({prob:.2f}%)")