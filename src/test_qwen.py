import time
import gc
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
IMAGE_PATH = "data/images/img_000.jpg"

def tester_qwen(mode="fp16"):
    print(f"\n--- Test Qwen2-VL-2B en mode : {mode.upper()} ---")
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if mode == "4bit":
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
    elif mode == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None

    # On limite la résolution max des pixels pour garder une VRAM basse sur 8 Go
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        quantization_config=quant_config,
        device_map="auto"
    )

    image = Image.open(IMAGE_PATH).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in detail (subject, background, colors, lighting)."}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to("cuda")

    debut = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=80)
    duree = time.time() - debut

    reponse_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    legende = processor.batch_decode(reponse_ids, skip_special_tokens=True)[0].strip()

    vram_go = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(f"Légende : {legende}")
    print(f"Temps   : {duree:.2f} secondes")
    print(f"VRAM    : {vram_go:.2f} Go")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    tester_qwen(mode="fp16")
    tester_qwen(mode="8bit")
    tester_qwen(mode="4bit")