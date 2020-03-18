import argparse
from load import load_data
import torch


def main():
    device = torch.device(args.device)

    train_loader = load_data(args.train_path,
                             args.batch_size,
                             args.tokens_column, args.predict_column,
                             args.lang_model,
                             args.max_len,
                             args.separator,
                             args.pad_label,
                             device)

    dev_loader = load_data(args.train_path,
                           args.batch_size,
                           args.tokens_column, args.predict_column,
                           args.lang_model,
                           args.max_len,
                           args.separator,
                           args.pad_label,
                           device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", type=str)
    parser.add_argument("dev_path", type=str)
    parser.add_argument("predict_column", type=int)
    parser.add_argument("--tokens_column", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lang_model", type=str, default="bert-base-cased")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--separator", type=str, default="\t")
    parser.add_argument("--pad_label", type=str, default="<pad>")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    #tokenizer = BertTokenizer.from_pretrained(args.lang_model)

    # print(tokenizer.encode())

    main()