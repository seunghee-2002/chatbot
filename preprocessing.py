import json
import os
from tqdm import tqdm

def convert_to_instruction(example):
    """새로운 데이터 형식을 EXAONE 학습용 텍스트로 변환"""
    
    # 필수 필드 추출 (기본값 설정)
    personality = example.get('성격', '평범형')
    adv_type = example.get('모험가타입', '전사')
    age = example.get('나이', '20대')
    gender = example.get('성별', '남성')
    grade = example.get('모험가등급', 'C급')
    visit_count = example.get('방문횟수', '첫방문')
    prev_item = example.get('이전_아이템', '없음')
    revisit_gap = example.get('재방문간격', '없음')
    last_quest = example.get('최근_의뢰', '첫방문')
    greeting = example.get('인사말', '')
    
    # Instruction 문장 생성
    instruction = (
        f"성격: {personality}, 모험가타입: {adv_type}, 나이: {age}, "
        f"성별: {gender}, 모험가등급: {grade}, 방문횟수: {visit_count}, "
        f"이전_아이템: {prev_item}, 재방문간격: {revisit_gap}, "
        f"최근_의뢰: {last_quest} "
        "일 때의 적절한 모험가 인사말을 생성해줘."
    )
    
    return {"instruction": instruction, "output": greeting}

def preprocess_all_data(dataset_dir="./dataset", output_file="train_dataset.jsonl"):
    """폴더 내 모든 JSON을 읽어 하나의 JSONL로 합치고 현황 출력"""
    combined_data = []
    
    # 폴더 존재 확인
    if not os.path.exists(dataset_dir):
        print(f"에러: {dataset_dir} 폴더를 찾을 수 없습니다.")
        return

    file_list = [f for f in os.listdir(dataset_dir) if f.endswith(".json")]
    print(f"총 {len(file_list)}개의 파일을 발견했습니다. 전처리를 시작합니다.")

    # tqdm으로 진행률 표시
    for filename in tqdm(file_list, desc="데이터 변환 중"):
        file_path = os.path.join(dataset_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 리스트 형태인지 단일 객체인지 확인
                if isinstance(data, list):
                    for item in data:
                        converted = convert_to_instruction(item)
                        # 인사말이 비어있지 않은 경우만 추가
                        if converted['output'].strip():
                            combined_data.append(converted)
                else:
                    converted = convert_to_instruction(data)
                    if converted['output'].strip():
                        combined_data.append(converted)
        except Exception as e:
            print(f"⚠️ {filename} 처리 중 오류 발생: {e}")

    # 최종 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in combined_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print("-" * 50)
    print(f"전처리 완료!")
    print(f"생성된 파일: {output_file}")
    print(f"총 학습 데이터 수: {len(combined_data)}개")
    print("-" * 50)
    
    # 샘플 데이터 출력
    if combined_data:
        print("\n[샘플 데이터 미리보기]")
        print(json.dumps(combined_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    preprocess_all_data()