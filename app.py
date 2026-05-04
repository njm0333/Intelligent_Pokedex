import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import requests
from io import BytesIO

# 4K 모니터 활용을 위해 layout="wide" 추가
st.set_page_config(page_title="Pokemon Classifier", page_icon="🔍", layout="wide")

# ==========================================
# 1. 초기화 및 리소스 로드
# ==========================================
@st.cache_resource
def load_assets():
    """클래스 이름, 모델 아키텍처 정보, 한국어 번역 사전을 로드합니다."""
    if not os.path.exists('class_names.json') or not os.path.exists('best_model_info.json') or not os.path.exists('best_model.pth'):
        return None, None, None, "필수 파일이 없습니다. 1_train.py를 먼저 실행하여 모델을 학습시켜주세요."

    with open('class_names.json', 'r', encoding='utf-8') as f:
        class_names = json.load(f)

    with open('best_model_info.json', 'r', encoding='utf-8') as f:
        info = json.load(f)
        best_arch = info.get("best_architecture")

    # 한국어 번역 사전 로드 (make_dict.py로 만든 파일)
    ko_dict = {}
    if os.path.exists('pokemon_ko_dict.json'):
        with open('pokemon_ko_dict.json', 'r', encoding='utf-8') as f:
            ko_dict = json.load(f)

    return class_names, best_arch, ko_dict, None

class_names, best_arch, pokemon_ko_dict, error_msg = load_assets()

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
    model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval() # 추론 모드 전환 필수
    return model

model = load_pytorch_model(best_arch, num_classes)

# ==========================================
# 3. 이미지 전처리 정의
# ==========================================
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 4. Streamlit UI 렌더링 (와이드 & 좌우 분할)
# ==========================================
st.title("Intelligent Pokedex")
st.markdown(f"**적용된 인공지능 모델:** `{best_arch}` ")
st.write("---")

# 화면을 좌/우 5:5 비율로 나눔
col1, col2 = st.columns(2)

image = None

with col1:
    st.subheader(" 이미지 입력")

    # 1. 파일 업로드 (드래그 앤 드롭 & 클립보드 지원)
    uploaded_file = st.file_uploader("이미지 파일 업로드", type=["jpg", "jpeg", "png"])

    st.write("**또는**")

    # 2. 이미지 URL 직접 입력
    image_url = st.text_input("웹 이미지 주소(URL) 붙여넣기")

# 입력받은 이미지 처리
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
elif image_url:
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        col1.error("이미지 주소를 불러올 수 없습니다. 주소가 정확한지 확인해주세요.")

# 이미지가 정상적으로 준비되었을 때만 우측 화면(col2)에 결과 표시
if image is not None:
    with col2:
        st.subheader("🔍 분석 결과")

        # 이미지 가로폭을 컬럼 크기에 딱 맞춤
        st.image(image, caption="분석 중인 포켓몬...", use_container_width=True)

        with st.spinner("AI가 도감을 뒤지는 중입니다... 🔄"):
            # 텐서 변환 및 추론
            input_tensor = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top5_prob, top5_catid = torch.topk(probabilities, 5)

        st.markdown("### 📊 Top-5 Predictions:")

        # 결과 출력
        with st.container():
            for i in range(5):
                idx = top5_catid[i].item()
                prob = top5_prob[i].item() * 100
                en_name = class_names[idx]

                # 사전에 등록된 한국어 이름이 있으면 가져옴 (없으면 원래 영어 이름 반환)
                ko_name = pokemon_ko_dict.get(en_name, pokemon_ko_dict.get(en_name.lower(), en_name))

                # 한국어 이름이 영어 이름과 다르면 (사전에 존재하면) 괄호 포함해서 출력
                if ko_name != en_name:
                    display_name = f"{en_name} ({ko_name})"
                else:
                    display_name = en_name

                if i == 0:
                    st.success(f"🥇 **1위: {display_name} ({prob:.2f}%)**")
                else:
                    st.write(f"**{i+1}위:** {display_name} ({prob:.2f}%)")
