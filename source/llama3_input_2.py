import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === 모델명 설정 ===
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# === tokenizer와 model 로드 ===
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# === pipeline 생성 ===
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16,
)

# === Cleaning Report 읽기 ===
def read_cleaning_report(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    cleaning_reports = []
    for line in lines:
        match = re.search(r'Round (\d+): cleaning_time\(s\): (\d+), distance\(m\): ([\d.]+), consumption\(W\):([\d.]+)  cleaning_coverage\(%\): (\d+), obstacle_avoid: (\d+), cleaning_retry\(count\): (\d+)', line)
        if match:
            report = {
                'round': int(match.group(1)),
                'cleaning_time_s': int(match.group(2)),
                'distance_m': float(match.group(3)),
                'consumption_w': float(match.group(4)),
                'cleaning_coverage_percent': int(match.group(5)),
                'obstacle_avoid_count': int(match.group(6)),
                'retry_count': int(match.group(7))
            }
            cleaning_reports.append(report)
    return cleaning_reports

# === Normal Input 생성 ===
def generate_normal_input(report):
    cleaning_time_min = report['cleaning_time_s'] // 60
    cleaning_time_sec = report['cleaning_time_s'] % 60
    return (
        f"Round {report['round']} Cleaning Report: "
        f"The vacuum cleaner operated for {cleaning_time_min} minutes and {cleaning_time_sec} seconds, "
        f"covering a distance of {report['distance_m']} meters. "
        f"It consumed approximately {report['consumption_w']} watts of energy during the cleaning process. "
        f"The final cleaning coverage achieved was {report['cleaning_coverage_percent']}%. "
        f"Obstacle avoidance events were {report['obstacle_avoid_count']}, "
        f"and cleaning retries were {report['retry_count']}. "
        f"<|culture|>"
    )

# === Llama로 단일 Level 수정 ===
def generate_single_level_from_normal(normal_input, base_value, level_type):
    system_prompt = f"""You are analyzing a vacuum cleaning result report.

Your task:
- Check the Current {level_type} Level: {base_value}.
- Based on the following cleaning result, decide whether the {level_type} Level should increase, decrease, or stay the same.
- Adjust the level within ±0.5 range (e.g., increase or decrease by 0.01 ~ 0.50).
- level step size is 0.01
- Explain your reason briefly.

Format your answer exactly like this:
{level_type} Level: (new value), Reason: (short reason)
"""

    user_prompt = f"""Cleaning Result:
{normal_input}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    output = generator(
        messages,
        max_length=512,
        temperature=0.7,
        num_return_sequences=1,
        do_sample=True,
    )

    response_text = None
    for item in output[0]['generated_text']:
        if item['role'] == 'assistant':
            response_text = item['content']
            break

    # Level 값 추출
    match = re.search(rf"{level_type} Level:\s*([\d.]+)", response_text)
    new_value = float(match.group(1)) if match else base_value
    new_value = round(new_value, 2)

    # Reason 추출
    reason_match = re.search(r"Reason:\s*(.+)", response_text)
    reason_text = reason_match.group(1).strip() if reason_match else "No reason provided"

    # ✅ 출력 추가
    print(f"\n✅ [{level_type}] Level 수정 결과")
    print(f"기존 {level_type} Level: {base_value}")
    print(f"수정된 {level_type} Level: {new_value}")
    print(f"Reason: {reason_text}")

    return new_value

# === Cultural Input 저장용 포맷 생성 ===
def generate_cultural_format(round_num, suction, obstacle, retry):
    return f"Round {round_num:02d}: suction:{suction:.2f} obstacle:{obstacle:.2f} retry:{retry:.2f}"

# === 실행 ===

# 초기 기본값
base_suction = 2.0
base_obstacle = 2.0
base_retry = 2.0

# 1. 청소 리포트 읽기
cleaning_reports = read_cleaning_report("/home/e2map/clean_map/Ikea/ikea_1_cleaning_report.txt")

# 2. Cultural 레벨 생성
cultural_inputs = []

for report in cleaning_reports:
    normal_input = generate_normal_input(report)
    print(normal_input)
    suction = generate_single_level_from_normal(normal_input, base_suction, "Suction")
    obstacle = generate_single_level_from_normal(normal_input, base_obstacle, "Obstacle Avoidance")
    retry = generate_single_level_from_normal(normal_input, base_retry, "Retry")
    #print(f"✅ Updated Levels - Suction: {suction:.2f}, Obstacle Avoidance: {obstacle:.2f}, Retry: {retry:.2f}")

    cultural_line = generate_cultural_format(report['round'], suction, obstacle, retry)
    cultural_inputs.append(cultural_line)

    # 다음 라운드는 이번 결과를 base로 사용
    base_suction = suction
    base_obstacle = obstacle
    base_retry = retry
# 3. 결과 txt로 저장
with open("ikea_1_cultural.txt", "w") as f:
    for line in cultural_inputs:
        f.write(line + "\n")

print("\n✅ Cultural input 생성 및 저장 완료!")