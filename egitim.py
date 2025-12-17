import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq

# 1. Veriyi Yükle
df = pd.read_csv("dataset.csv")
dataset = Dataset.from_pandas(df)

# [cite_start]2. Model ve Tokenizer (Transformer Mimarisi) [cite: 18]
MODEL_ID = "google/mt5-small" 
tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)

# [cite_start]3. Veriyi Hazırla (Preprocessing) [cite: 14]
def preprocess(examples):
    # T5 için girdi formatı
    inputs = ["summarize: " + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    # Hedef (Özet) kısımları
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")
    
    # -100 değeri, modelin padding kısımlarını (boşlukları) görmezden gelmesini sağlar
    model_inputs["labels"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    return model_inputs

tokenized_data = dataset.map(preprocess, batched=True)

# [cite_start]4. Eğitim Parametreleri [cite: 18, 20]
args = TrainingArguments(
    output_dir="./final_model",
    num_train_epochs=3, # Hızlı olması için 3 yapıldı
    per_device_train_batch_size=2,
    save_strategy="epoch",
    learning_rate=2e-5
)

# 5. Eğitimi Başlat
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_data,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

print("Eğitim başlıyor...")
trainer.train()

# [cite_start]6. Modeli Kaydet [cite: 18]
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
print("Model './final_model' klasörüne kaydedildi!")