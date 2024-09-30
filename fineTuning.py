import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from codecarbon import OfflineEmissionsTracker
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset, load_from_disk

device = 'cuda'
model_id = 'models/Mistral-7B-Instruct-v0.2'
dataset_id = './Neural-Story-v1-local'
output_dir = "mistral_7b-instruct-enzoP"
gpu_ids = [0, 1, 2, 3]

# Configura il tracker di emissioni
tracker = OfflineEmissionsTracker(
    project_name="Mistral7BInstruct0.2_fineTuning",
    gpu_ids=gpu_ids,
    tracking_mode='process',
    country_iso_code="ITA"
)

# Carica il tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

# Configurazione per Bits and Bytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Carica il modello
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.use_cache = False  # silenzia gli avvisi
model.gradient_checkpointing_enable()

# Configura il modello per il training a bassa precisione
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

#Carica il dataset
data = load_from_disk(dataset_id)
data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        report_to = []
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# Inizia il tracciamento delle emissioni e allena il modello
tracker.start()
try:
    trainer.train()
finally:
    tracker.stop()

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)

