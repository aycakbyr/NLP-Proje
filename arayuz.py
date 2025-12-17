import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration

# [cite_start]Eğittiğimiz modeli yüklüyoruz [cite: 18]
MODEL_YOLU = "./final_model"
tokenizer = T5Tokenizer.from_pretrained(MODEL_YOLU)
model = T5ForConditionalGeneration.from_pretrained(MODEL_YOLU)

def ozetle(metin):
    input_text = "summarize: " + metin
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=150, 
        min_length=30, 
        num_beams=4, # Daha iyi kelime seçimi yapar
        no_repeat_ngram_size=2, # Aynı kelimelerin tekrar etmesini engeller
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# [cite_start]Gradio Arayüzü [cite: 3]
demo = gr.Interface(
    fn=ozetle, 
    inputs=gr.Textbox(lines=10, label="Haber Metni"),
    outputs=gr.Textbox(label="Özet"),
    title="Transformer Metin Özetleme Projesi"
)

demo.launch()