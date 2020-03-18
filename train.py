import argparse
from load import load_data
import torch
from model import LangModelWithDense
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Precision, Recall
from transformers import *


def main():
    device = torch.device(args.device)

    train_loader, label_encoder = load_data(args.train_path,
                                            args.batch_size,
                                            args.tokens_column, args.predict_column,
                                            args.lang_model_name,
                                            args.max_len,
                                            args.separator,
                                            args.pad_label,
                                            device)

    dev_loader, _ = load_data(args.train_path,
                              args.batch_size,
                              args.tokens_column, args.predict_column,
                              args.lang_model_name,
                              args.max_len,
                              args.separator,
                              args.pad_label,
                              device,
                              label_encoder)

    lang_model = AutoModel.from_pretrained(args.lang_model_name)
    input_size = 768 if "base" in args.lang_model_name else 1024

    model = LangModelWithDense(lang_model, input_size, len(label_encoder.classes_), False)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = torch.nn.CrossEntropyLoss()

    max_epochs = 100
    validate_every = 100
    checkpoint_every = 100

    trainer = create_supervised_trainer(model, optimizer, criterion)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy()})

    trainer.run(train_loader, max_epochs=max_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", type=str)
    parser.add_argument("dev_path", type=str)
    parser.add_argument("predict_column", type=int)
    parser.add_argument("--tokens_column", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lang_model_name", type=str, default="bert-base-cased")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--separator", type=str, default="\t")
    parser.add_argument("--pad_label", type=str, default="<pad>")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    #tokenizer = BertTokenizer.from_pretrained(args.lang_model)

    # print(tokenizer.encode())

    main()