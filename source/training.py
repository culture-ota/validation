# import os
# import fitz  # PyMuPDF
# import pandas as pd
# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
# from sklearn.model_selection import train_test_split
# from peft import LoraConfig, get_peft_model
# import torch
# import gc
# from transformers.generation.utils import Cache
# #
# # # 2. PDF 파일을 텍스트로 변환 (폴더 내 모든 PDF 파일 처리)
# # def pdf_to_text(folder_path):
# #     text_data = []
# #     for filename in os.listdir(folder_path):
# #         if filename.endswith(".pdf"):
# #             file_path = os.path.join(folder_path, filename)
# #             doc = fitz.open(file_path)
# #             text = ""
# #             for page in doc:
# #                 text += page.get_text()
# #             text_data.append(text)
# #     return text_data
# #
# # # PDF 파일이 들어있는 폴더 경로 설정
# # pdf_folder_path = "/data/seyeolyang/PycharmProjects/RAG/pdf"
# # texts = pdf_to_text(pdf_folder_path)
#
# # 3. 데이터 준비
# data = {'text': texts}
# df = pd.DataFrame(data)
# #train_df, eval_df = train_test_split(df, test_size=0.2)  # 데이터셋을 학습/검증 데이터로 분리
#
# train_dataset = Dataset.from_pandas(df)
# #train_dataset = Dataset.from_pandas(train_df)
# #eval_dataset = Dataset.from_pandas(eval_df)
#
# # 4. 모델과 Tokenizer 준비
# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# # Padding token 설정
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
#
# model = AutoModelForCausalLM.from_pretrained(model_name) #, torch_dtype=torch.float16) #)# device_map="auto")
#
# # Define LoRA configuration
# lora_config = LoraConfig(
#     r=8,                        # LoRA rank
#     lora_alpha=32,              # LoRA scaling
#     lora_dropout=0.1,           # Dropout rate
#     bias="none",                # Options: "none", "all", or "lora_only"
#
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
#                     "gate_proj", "up_proj", "down_proj", ],
# )
#
# # Apply LoRA to the model
# model = get_peft_model(model, lora_config)
# #model.to('cpu')
#
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)
#
# tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
# #tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
# #
# # 5. 학습 설정
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=1,  # 배치 크기를 1로 줄임
#     per_device_eval_batch_size=1,   # 배치 크기를 1로 줄임
#     #gradient_accumulation_steps=8,  # 배치 크기를 더 작게 설정하고 반복 횟수를 늘림
#
#     num_train_epochs=1,
#     weight_decay=0.01,
#     #use_cpu=True,  # GPU 사용하지 않도록 설정
#     fp16=True,  # fp16을 사용하여 훈련 설정
#     #optim = "adafactor"  # 메모리 사용량을 줄이기 위해 AdaFactor 사용
# )
#
# # 6. Trainer 설정 및 학습 시작
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_dataset,
#     #eval_dataset=tokenized_eval_dataset,
#     tokenizer=tokenizer,
# )
# # 메모리 청소
# # gc.collect()
# # torch.cuda.empty_cache()
#
# # Train the model
# trainer.train()
# #trainer_stats = trainer.train()
#
# #Evaluate the model
# #eval_results = trainer.evaluate()
# #print(f"Evaluation Results: {eval_results}")
#
# # 7. 예측 결과 확인
# # def generate_text(prompt, max_length=50):
# #     inputs = tokenizer(prompt, return_tensors="pt")
# #     outputs = model.generate(inputs.input_ids, max_length=max_length)
# #     return tokenizer.decode(outputs[0], skip_special_tokens=True)
# #
# # # 예시 입력을 사용하여 모델이 생성한 텍스트 출력
# # example_prompt = "This document discusses"
# # generated_text = generate_text(example_prompt)
# # print(f"Generated Text: {generated_text}")
#
#
# # 모델 저장
# trainer.save_model("./saved_model")
# tokenizer.save_pretrained("./saved_model")


import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model


train_dataset = load_dataset('csv', data_files='/data/seyeolyang/PycharmProjects/RAG/Crawling_filtering.csv')['train']

# train_dataset의 상위 5개를 DataFrame으로 변환하여 출력
df_head = train_dataset.to_pandas().head()
print(df_head)



# 4. 모델과 Tokenizer 준비
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Padding token 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name) #, torch_dtype=torch.float16) #)# device_map="auto")

# Define LoRA configuration
lora_config = LoraConfig(
    r=8,                        # LoRA rank
    lora_alpha=32,              # LoRA scaling
    lora_dropout=0.1,           # Dropout rate
    bias="none",                # Options: "none", "all", or "lora_only"

    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)


# def tokenize_function(examples):
#     texts = examples["text"]
#     return tokenizer(texts, padding="max_length", truncation=True)


def tokenize_function(examples):
    # examples["text"]가 리스트인지 확인하고, 아니면 리스트로 변환
    texts = examples["text"]

    # 리스트가 아닐 경우 리스트로 감싸줌
    if isinstance(texts, str):
        texts = [texts]
        print("not list")

    # 텍스트를 토큰화
    return tokenizer(texts, padding="max_length", truncation=True)
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

# 5. 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # 배치 크기를 1로 줄임
    per_device_eval_batch_size=16,   # 배치 크기를 1로 줄임
    num_train_epochs=1,
    weight_decay=0.01,
    fp16=True,  # fp16을 사용하여 훈련 설정
)

# 6. Trainer 설정 및 학습 시작
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# 모델 저장
trainer.save_model("./saved_model")
tokenizer.save_pretrained("./saved_model")