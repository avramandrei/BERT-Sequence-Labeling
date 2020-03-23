import torch
import argparse
import os
import pickle
from load import load_data_from_file
from transformers import *


def main():
    device = torch.device(args.device)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with open(os.path.join(args.model_path, "label_encoder.pk"), "rb") as file:
        label_encoder = pickle.load(file)

    test_loader, _ = load_data_from_file(args.test_path,
                                         1,
                                         args.token_column, args.predict_column,
                                         args.lang_model_name,
                                         512,
                                         args.separator,
                                         args.pad_label, args.null_label,
                                         device,
                                         label_encoder,
                                         False)

    tokenizer = AutoTokenizer.from_pretrained(args.lang_model_name)

    model = torch.load(os.path.join(args.model_path, "model.pt"), map_location=args.device)
    model.fine_tune = False
    model.eval()

    list_labels = []

    for i, (test_x, _, mask, _) in enumerate(test_loader):
        print("Predicting tags for sequence: {}/{}...".format(i, len(test_loader.dataset)))
        logits = model.forward(test_x, mask)
        preds = torch.argmax(logits, 2)

        end = torch.argmax(mask, dim=1)

        labels = label_encoder.inverse_transform(preds[0][1:end].tolist())
        list_labels.append(labels)

    with(open(os.path.join(args.test_path), "r", encoding='utf-8')) as in_file, \
         open(os.path.join(args.output_path, "predict.conllu"), "w", encoding='utf-8') as out_file:
        sentence_idx = 0
        label_idx = 0

        for line in in_file:
            if not line.startswith("#"):
                if line not in [" ", "\n"]:
                    tokens = line.split(args.separator)

                    token = tokens[args.token_column]
                    subtokens = tokenizer.encode(token, add_special_tokens=False)

                    tokens[args.predict_column] = list_labels[sentence_idx][label_idx]

                    label_idx += len(subtokens)

                    for token in tokens[:-1]:
                        out_file.write("{}{}".format(token, args.separator))

                    out_file.write("{}".format(tokens[-1] + "\n" if "\n" not in tokens[-1] else tokens[-1]))
                else:
                    # print(label_idx, len(curr_labels))
                    assert label_idx == len(list_labels[sentence_idx])

                    out_file.write("\n")
                    sentence_idx += 1
                    label_idx = 0
            else:
                out_file.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_path")
    parser.add_argument("model_path", type=str)
    parser.add_argument("token_column", type=int)
    parser.add_argument("predict_column", type=int)
    parser.add_argument("lang_model_name", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--separator", type=str, default="\t")
    parser.add_argument("--pad_label", type=str, default="<pad>")
    parser.add_argument("--null_label", type=str, default="<X>")
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    main()
