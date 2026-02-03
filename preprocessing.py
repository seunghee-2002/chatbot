import json
import os
from tqdm import tqdm

# 1. 매핑 테이블 정의 (숫자 데이터를 텍스트로 변환하기 위함)
TYPE_MAP = {0: "전사", 1: "궁수", 2: "마법사", 3: "도적"}
AFF_MAP = {0: "매우낮음", 1: "낮음", 2: "보통", 3: "높음", 4: "매우높음"}
VISIT_MAP = {0: "첫방문", 1: "초면", 2: "아는사이", 3: "친구", 4: "단골"}
RES_MAP = {0: "첫방문", 1: "실패", 2: "성공", 3: "대성공"}
MATCH_MAP = {0: "나쁨", 1: "보통", 2: "좋음"}

def convert_to_instruction(example):
    """단일 데이터를 읽어 EXAONE 학습용 텍스트로 변환"""
    
    # 타입(adventurerType): 숫자면 맵 적용, 텍스트면 그대로 사용
    raw_type = example.get('adventurerType', 0)
    adv_type = TYPE_MAP.get(raw_type, raw_type) if isinstance(raw_type, int) else raw_type
    
    # 성격(personality): 이미 텍스트인 경우가 많으므로 그대로 가져오되 기본값 설정
    pers = example.get('personality', '평범형')
    
    # 수치형 데이터들 처리 (context 내부 포함)
    aff = AFF_MAP.get(example.get('affectionLevel', 2), "보통")
    visit = VISIT_MAP.get(example.get('visitCount', 0), "첫방문")
    
    ctx = example.get('context', {})
    last_res = RES_MAP.get(ctx.get('lastResult', 0), "첫방문")
    match = MATCH_MAP.get(ctx.get('lastTypeMatch', 1), "보통")
    
    # 최종 Instruction 문장 생성
    instruction = (
        f"타입: {adv_type}, 성격: {pers}, 호감도: {aff}, "
        f"방문: {visit}, 이전결과: {last_res}, 상성: {match} "
        "일 때의 적절한 모험가 인사말을 생성해줘."
    )
    
    return {"instruction": instruction, "output": example.get('greeting', "")}

def preprocess_all_data(dataset_dir="./Dataset", output_file="train_dataset.jsonl"):
    """폴더 내 모든 JSON을 읽어 하나의 JSONL로 합치고 현황 출력"""
    combined_data = []
    
    # 폴더 내 모든 .json 파일 목록 추출
    if not os.path.exists(dataset_dir):
        print(f"에러: {dataset_dir} 폴더를 찾을 수 없습니다.")
        return

    file_list = [f for f in os.listdir(dataset_dir) if f.endswith(".json")]
    print(f"총 {len(file_list)}개의 파일을 발견했습니다. 전처리를 시작합니다.")

    # tqdm을 사용하여 실시간 진행률 표시
    for filename in tqdm(file_list, desc="데이터 변환 중"):
        file_path = os.path.join(dataset_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 리스트 형태([ {...}, {...} ])인지 단일 객체({ ... })인지 확인
                if isinstance(data, list):
                    for item in data:
                        combined_data.append(convert_to_instruction(item))
                else:
                    combined_data.append(convert_to_instruction(data))
        except Exception as e:
            print(f"⚠️ {filename} 처리 중 오류 발생: {e}")

    # 최종 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in combined_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print("-" * 30)
    print(f"전처리 완료!")
    print(f"생성된 파일: {output_file}")
    print(f"총 학습 데이터 수: {len(combined_data)}개")
    print("-" * 30)

if __name__ == "__main__":
    preprocess_all_data()