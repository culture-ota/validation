import torch
import re
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig
import torch.nn as nn

# === 모델명 설정 ===
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# === tokenizer와 model 로드 ===
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# === Special token 추가 ===
special_tokens_dict = {'additional_special_tokens': ['<|culture|>']}
num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

# === Regression Head 정의 ===
class RegressionHead(nn.Module):
    def __init__(self, hidden_size, output_size=3):
        super(RegressionHead, self).__init__()
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.linear(x)

# === culture token embedding 조회 함수 ===
def get_culture_token_embedding(tokenizer, model, token_str="<|culture|>"):
    token_id = tokenizer.convert_tokens_to_ids(token_str)
    embedding = model.model.embed_tokens.weight[token_id]
    return embedding

# === Fine-tuning 함수 (Regression 버전) ===
def fine_tune_cultural_token_embedding_regression(model, tokenizer, regression_head, cultural_input, target_levels, culture_token_str="<|culture|>", lr=5e-4):
    device = model.device
    model.train()
    regression_head.train()

    inputs = tokenizer(cultural_input, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    culture_token_id = tokenizer.convert_tokens_to_ids(culture_token_str)

    # === Special token embedding 준비
    with torch.no_grad():
        cultural_embedding = model.model.embed_tokens.weight[culture_token_id].float().detach().clone().to(device).requires_grad_(True)

    # === Optimizer 설정 (embedding + regression head)
    optimizer = torch.optim.AdamW(
        [cultural_embedding] + list(regression_head.parameters()),
        lr=lr,
        weight_decay=0.0
    )

    # === Forward
    embeddings = model.model.embed_tokens(input_ids)
    culture_token_positions = (input_ids == culture_token_id).nonzero(as_tuple=False)

    batch_idx, token_idx = culture_token_positions[0]
    embeddings[batch_idx, token_idx] = cultural_embedding

    cultural_token_output = embeddings[batch_idx, token_idx]

    preds = regression_head(cultural_token_output.float())

    target_tensor = torch.tensor(target_levels, dtype=torch.float, device=device)

    loss_fn = nn.MSELoss()
    loss = loss_fn(preds, target_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # === 업데이트된 cultural embedding 적용
    with torch.no_grad():
        model.model.embed_tokens.weight[culture_token_id] = cultural_embedding.to(dtype=model.model.embed_tokens.weight.dtype)

    print(f"Regression Loss: {loss.item():.6f}")

    return model

# === cultural input text 파일 읽기 ===
def read_cultural_inputs(text_path):
    suction_list = []
    obstacle_list = []
    retry_list = []

    with open(text_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            match = re.search(r'suction:([\d.]+) obstacle:([\d.]+) retry:([\d.]+)', line)
            if match:
                suction_list.append(float(match.group(1)))
                obstacle_list.append(float(match.group(2)))
                retry_list.append(float(match.group(3)))

    return suction_list, obstacle_list, retry_list

# === cultural input 생성 ===
def generate_cultural_input(suction_level, obstacle_level, retry_level):
    return f"<|culture|> suction:{suction_level:.2f} obstacle:{obstacle_level:.2f} retry:{retry_level:.2f}"

# === culture token embedding 저장 함수 추가 ===
def save_culture_token_embedding(embedding_tensor, save_path, round_num):
    embedding_array = embedding_tensor.detach().cpu().numpy()
    with open(save_path, 'a') as f:
        f.write(f"=== Round {round_num:02d} ===\n")
        f.write(' '.join(map(str, embedding_array.tolist())) + '\n\n')

# === 실행 ===

# 1. 텍스트 파일 경로
text_path = "/home/e2map/clean_map/Ikea/ikea_5_parameter_result.txt"
save_embedding_path = "/home/e2map/clean_map/Ikea/ikea_5_cultural_token.txt"

# 2. cultural level 리스트 읽기
suction_levels, obstacle_levels, retry_levels = read_cultural_inputs(text_path)

# 3. Regression Head 선언
regression_head = RegressionHead(hidden_size=model.config.hidden_size).to(model.device)

# 4. 20번 루프 학습
for i in range(20):
    suction = suction_levels[i]
    obstacle = obstacle_levels[i]
    retry = retry_levels[i]

    cultural_input = generate_cultural_input(suction, obstacle, retry)
    target_levels = [suction, obstacle, retry]

    print(f"\n=== Round {i+1:02d} Cultural Input ===")
    print(cultural_input)

    model = fine_tune_cultural_token_embedding_regression(
        model, tokenizer, regression_head,
        cultural_input, target_levels
    )

    culture_token_embedding = get_culture_token_embedding(tokenizer, model)
    print(f"\n✅ [After Round {i+1:02d}] Updated <|culture|> Token Embedding Vector:")
    print(culture_token_embedding.cpu())

    save_culture_token_embedding(culture_token_embedding, save_embedding_path, i + 1)
