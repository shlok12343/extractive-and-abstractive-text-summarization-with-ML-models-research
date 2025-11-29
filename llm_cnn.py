from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import torch
import argparse
import os


def preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=128):
    """
    Tokenize CNN/DailyMail articles and summaries for seq2seq training.
    """
    inputs = examples["article"]
    targets = examples["highlights"]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train_llm(
    model_name: str = "facebook/bart-base",
    output_dir: str = "artifacts/llm_cnn",
    max_train_samples: int = 2000,
    max_val_samples: int = 200,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print("Loading CNN/DailyMail dataset...")
    raw_datasets = load_dataset("cnn_dailymail", "3.0.0")

    train_dataset = raw_datasets["train"]
    val_dataset = raw_datasets["validation"]

    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if max_val_samples is not None:
        val_dataset = val_dataset.select(range(min(max_val_samples, len(val_dataset))))

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    print(f"Loading tokenizer and model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


    print("Tokenizing datasets...")
    preprocess = lambda batch: preprocess_function(batch, tokenizer)
    tokenized_train = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_val = val_dataset.map(
        preprocess,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=num_train_epochs,
        logging_steps=50,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting LLM fine‑tuning on CNN/DailyMail...")
    trainer.train()

    print(f"Saving fine‑tuned model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


def evaluate_llm(
    model_dir: str = "artifacts/llm_cnn",
    split: str = "test[:20]",
    max_input_length: int = 512,
    max_target_length: int = 128,
) -> None:
    """
    Load a fine‑tuned LLM and evaluate it on CNN/DailyMail using ROUGE‑1 and ROUGE‑L.
    """
    if not os.path.isdir(model_dir):
        print(f"Model directory '{model_dir}' not found. Train the model first.")
        return

    print(f"Loading tokenizer and model from '{model_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()

    print(f"Loading CNN/DailyMail {split} for evaluation...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rougeL_scores = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for ex in dataset:
        article = ex["article"]
        gold_summary = ex["highlights"]

        inputs = tokenizer(
            article,
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=max_target_length,
                num_beams=4,
                early_stopping=True,
            )

        pred_summary = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        scores = scorer.score(gold_summary, pred_summary)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    if rouge1_scores:
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
        print("\nFine‑tuned LLM ROUGE scores on CNN/DailyMail:")
        print(f"ROUGE-1 F1: {avg_rouge1:.4f}")
        print(f"ROUGE-L F1: {avg_rougeL:.4f}")
    else:
        print("No evaluation samples processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="Train the LLM on CNN/DailyMail (`train`) or evaluate a saved model (`eval`).",
    )
    parser.add_argument(
        "--model_name",
        default="facebook/bart-base",
        help="Hugging Face model checkpoint to fine‑tune (e.g., 'facebook/bart-base', 't5-small').",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/llm_cnn",
        help="Directory to save / load the fine‑tuned model.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=20000,
        help="Max number of training examples to use from CNN/DailyMail.",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=2000,
        help="Max number of validation examples to use from CNN/DailyMail.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of fine‑tuning epochs.",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_llm(
            model_name=args.model_name,
            output_dir=args.output_dir,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            num_train_epochs=args.num_train_epochs,
        )
    else:
        evaluate_llm(model_dir=args.output_dir)


