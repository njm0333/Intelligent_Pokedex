import json
import os

# 텍스트 파일 이름
txt_filename = 'poke_multi_language.txt'
json_filename = 'pokemon_ko_dict.json'

pokemon_dict = {}

try:
    with open(txt_filename, 'r', encoding='utf-8') as f:
        for line in f:
            # 탭(\t)을 기준으로 텍스트 분리
            parts = line.strip().split('\t')

            # 제대로 분리되었고, 첫 번째 항목이 4자리 숫자(번호)인지 확인 [cite: 2199]
            if len(parts) >= 4 and parts[0].isdigit():
                ko_name = parts[1].strip()  # 두 번째 열: 한국어 [cite: 2199]
                en_name = parts[3].strip()  # 네 번째 열: 영어 [cite: 2199]

                # 영어 이름을 Key, 한국어 이름을 Value로 저장
                pokemon_dict[en_name] = ko_name

                # 데이터셋의 폴더명은 간혹 소문자이거나 공백 처리 방식이 다를 수 있으므로
                # 안전하게 소문자 버전도 추가로 매핑해둡니다.
                pokemon_dict[en_name.lower()] = ko_name

    # JSON 파일로 예쁘게 저장
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(pokemon_dict, f, ensure_ascii=False, indent=4)

    print(f"✅ 총 {len(pokemon_dict)//2}개의 포켓몬 한국어 매핑이 '{json_filename}'에 저장되었습니다!")

except FileNotFoundError:
    print(f"❌ '{txt_filename}' 파일을 찾을 수 없습니다. 파일 이름과 위치를 확인해주세요.")