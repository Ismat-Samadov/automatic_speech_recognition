"""
Sample Training Script for Azerbaijani ASR
Runs a quick training test with 100 samples using streaming mode
"""

import os
import ssl
import sys
import warnings

# ============================================================
# SSL Configuration (must be before imports)
# ============================================================
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

sys.modules['hf_xet'] = None
ssl._create_default_https_context = ssl._create_unverified_context

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

import requests
_orig_request = requests.Session.request

def _patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return _orig_request(self, method, url, **kwargs)

requests.Session.request = _patched_request

print("SSL verification disabled")

# ============================================================
# Configuration
# ============================================================
SAMPLE_SIZE = 100
MODEL_NAME = "openai/whisper-small"
DATASET_NAME = "LocalDoc/azerbaijani_asr"
OUTPUT_DIR = "./whisper-azerbaijani-sample"
LANGUAGE = "azerbaijani"
TASK = "transcribe"

CONFIG = {
    "batch_size": 4,
    "epochs": 1,
    "learning_rate": 1e-5,
    "max_steps": 50,
    "eval_steps": 25,
    "save_steps": 25,
    "warmup_steps": 10,
    "gradient_accumulation_steps": 1,
    "fp16": False,
}

print(f"\n{'='*60}")
print(f"Sample Training Mode - {SAMPLE_SIZE} samples")
print(f"Model: {MODEL_NAME}")
print(f"Output: {OUTPUT_DIR}")
print(f"{'='*60}\n")

# ============================================================
# Device Detection
# ============================================================
import torch

def detect_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Device: Apple Silicon (MPS)")
    else:
        device = "cpu"
        print("Device: CPU (no GPU detected)")
    return device

DEVICE = detect_device()

if DEVICE == "cpu":
    CONFIG["fp16"] = False
    print("Note: fp16 disabled for CPU")

# ============================================================
# Load Dataset with Streaming
# ============================================================
from datasets import load_dataset, Dataset, DatasetDict, Audio

print(f"\n{'='*60}")
print(f"Loading {SAMPLE_SIZE} samples via streaming...")
print(f"{'='*60}\n")

try:
    dataset_stream = load_dataset(DATASET_NAME, streaming=True, trust_remote_code=False)

    # Take samples from stream
    train_samples = list(dataset_stream["train"].take(SAMPLE_SIZE))
    eval_samples = list(dataset_stream["train"].skip(SAMPLE_SIZE).take(20))

    dataset = DatasetDict({
        "train": Dataset.from_list(train_samples),
        "test": Dataset.from_list(eval_samples),
    })

    print(f"Dataset loaded successfully!")
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Test: {len(dataset['test'])} samples")
    print(f"\nSample data:")
    print(f"  Keys: {list(dataset['train'][0].keys())}")

except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# ============================================================
# Load Model & Processor
# ============================================================
from transformers import WhisperProcessor, WhisperForConditionalGeneration

print(f"\n{'='*60}")
print(f"Loading model: {MODEL_NAME}")
print(f"{'='*60}\n")

processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None

print(f"Model loaded: {model.num_parameters():,} parameters")

# ============================================================
# Preprocess Dataset
# ============================================================
SAMPLING_RATE = 16000

# Detect column names
sample = dataset["train"][0]
audio_column = "audio" if "audio" in sample else "path"
text_column = "sentence" if "sentence" in sample else "text" if "text" in sample else "transcription"

print(f"\nColumn mapping:")
print(f"  Audio: {audio_column}")
print(f"  Text: {text_column}")

# Cast audio to correct sampling rate
dataset = dataset.cast_column(audio_column, Audio(sampling_rate=SAMPLING_RATE))

def prepare_dataset(batch):
    audio = batch[audio_column]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch[text_column]).input_ids
    return batch

print(f"\n{'='*60}")
print(f"Processing dataset...")
print(f"{'='*60}\n")

dataset = dataset.map(
    prepare_dataset,
    remove_columns=dataset.column_names["train"],
    num_proc=1,
)

print(f"Dataset processed!")

# ============================================================
# Data Collator
# ============================================================
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# ============================================================
# Evaluation Metrics
# ============================================================
import evaluate

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ============================================================
# Training Configuration
# ============================================================
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    num_train_epochs=CONFIG["epochs"],
    max_steps=CONFIG["max_steps"],
    warmup_steps=CONFIG["warmup_steps"],
    fp16=CONFIG["fp16"],
    eval_strategy="steps",
    eval_steps=CONFIG["eval_steps"],
    save_strategy="steps",
    save_steps=CONFIG["save_steps"],
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=10,
    report_to=["tensorboard"],
    use_cpu=(DEVICE == "cpu"),
    push_to_hub=False,
    remove_unused_columns=False,
)

# ============================================================
# Initialize Trainer
# ============================================================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

print(f"\n{'='*60}")
print(f"Starting training...")
print(f"{'='*60}\n")

# ============================================================
# Train
# ============================================================
train_result = trainer.train()

print(f"\n{'='*60}")
print(f"Training completed!")
print(f"{'='*60}")
print(f"\nTraining metrics:")
for key, value in train_result.metrics.items():
    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

# ============================================================
# Evaluate
# ============================================================
print(f"\n{'='*60}")
print(f"Running evaluation...")
print(f"{'='*60}\n")

eval_results = trainer.evaluate()

print(f"Evaluation results:")
for key, value in eval_results.items():
    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

# ============================================================
# Save Model
# ============================================================
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"\n{'='*60}")
print(f"Model saved to: {OUTPUT_DIR}")
print(f"{'='*60}\n")

print("Sample training complete!")
print(f"\nTo use the model:")
print(f"  from transformers import pipeline")
print(f"  pipe = pipeline('automatic-speech-recognition', model='{OUTPUT_DIR}')")
print(f"  result = pipe('audio.wav')")
