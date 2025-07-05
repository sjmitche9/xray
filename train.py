# train.py
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from model import ImageToTextModel
from dataset import DataCollatorImageText
import eval_metrics

# Optional: load evaluation metrics
bleu = eval_metrics.load("bleu")
rouge = eval_metrics.load("rouge")
meteor = eval_metrics.load("meteor")

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def compute_metrics(eval_preds):

    logits, labels = eval_preds
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return {
        "bleu": bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])["bleu"],
        "rougeL": rouge.compute(predictions=decoded_preds, references=decoded_labels)["rougeL"],
        "meteor": meteor.compute(predictions=decoded_preds, references=decoded_labels)["meteor"]
    }

def main():
    model = ImageToTextModel(vocab_size=tokenizer.vocab_size)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = load_from_disk("processed/train")
    val_dataset = load_from_disk("processed/val")

    collator = DataCollatorImageText(tokenizer)

    training_args = TrainingArguments(
        output_dir="checkpoints/",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=25,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        fp16=True,
        report_to="wandb",  # or "tensorboard"
        run_name="iu-xray-baseline",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,  # optional
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

if __name__ == "__main__":
    main()
