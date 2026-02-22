from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

app = Flask(__name__)

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
lora_model_path = "./qwen-greeting-lora/checkpoint-248"

print("모델 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval()
print("모델 로딩 완료!")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()

    personality = data.get('성격', '평범형')
    adv_type    = data.get('모험가타입', '전사')
    age         = data.get('나이', '20대')
    gender      = data.get('성별', '남성')
    grade       = data.get('모험가등급', 'C급')
    visit_count = data.get('방문횟수', '첫방문')
    prev_item   = data.get('이전_아이템', '없음')
    revisit_gap = data.get('재방문간격', '없음')
    last_quest  = data.get('최근_의뢰', '첫방문')

    prompt = (
        f"성격: {personality}, 모험가타입: {adv_type}, 나이: {age}, "
        f"성별: {gender}, 모험가등급: {grade}, 방문횟수: {visit_count}, "
        f"이전_아이템: {prev_item}, 재방문간격: {revisit_gap}, "
        f"최근_의뢰: {last_quest} "
        "일 때의 적절한 모험가 인사말을 생성해줘."
    )

    messages = [
        {"role": "system", "content": "당신은 판타지 세계관의 무기 대여점을 방문하는 모험가의 인사말을 생성하는 전문가입니다."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(
        output[0][inputs['input_ids'].shape[-1]:],
        skip_special_tokens=True
    )

    return jsonify({"greeting": generated.strip()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)