# evaluate.py
import torch
from model import ImageToTextModel
from datasets import load_from_disk
from transformers import AutoTokenizer
import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")

def generate_beam(model, image, tokenizer, max_len=256, beam_width=3):
    model.eval()
    image = image.unsqueeze(0).to(DEVICE)
    features = model.cnn(image.unsqueeze(1))
    features = torch.mean(features.flatten(2), dim=2)
    memory = model.linear(features).unsqueeze(0)

    sequences = [[tokenizer.cls_token_id]]
    scores = [0]

    for _ in range(max_len):
        all_candidates = []
        for seq, score in zip(sequences, scores):
            input_ids = torch.tensor(seq).unsqueeze(0).to(DEVICE)
            tgt = model.embedding(input_ids).permute(1, 0, 2)
            out = model.decoder(tgt, memory)
            logits = model.output(out.permute(1, 0, 2))[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            topk = torch.topk(log_probs, beam_width)
            for i in range(beam_width):
                token = topk.indices[i].item()
                candidate = seq + [token]
                candidate_score = score + topk.values[i].item()
                all_candidates.append((candidate_score, candidate))

        # Prune to beam_width best
        ordered = sorted(all_candidates, key=lambda tup: tup[0], reverse=True)
        sequences = [seq for _, seq in ordered[:beam_width]]
        scores = [score for score, _ in ordered[:beam_width]]

        # Stop if all beams ended
        if all(seq[-1] == tokenizer.sep_token_id for seq in sequences):
            break

    final = sequences[0]
    return tokenizer.decode(final, skip_special_tokens=True)

def evaluate_all():
    model = ImageToTextModel()
    model.load_state_dict(torch.load("checkpoints/pytorch_model.bin", map_location=DEVICE))
    model.to(DEVICE)

    test_dataset = load_from_disk("processed/test")

    predictions, references = [], []

    for example in test_dataset:
        image = torch.tensor(example["image"], dtype=torch.float32)
        target = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        generated = generate_beam(model, image, tokenizer)

        predictions.append(generated)
        references.append(target)

    print("\n📊 Evaluating...")

    bleu = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])
    rouge = rouge_metric.compute(predictions=predictions, references=references)
    meteor = meteor_metric.compute(predictions=predictions, references=references)

    print(f"BLEU:   {bleu['bleu']:.4f}")
    print(f"ROUGE-L: {rouge['rougeL']:.4f}")
    print(f"METEOR: {meteor['meteor']:.4f}")

if __name__ == "__main__":
    evaluate_all()