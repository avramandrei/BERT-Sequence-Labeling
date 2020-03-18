import argparse
from load import load_data
import torch
from model import LangModelWithDense
from tqdm import tqdm
from transformers import *


def train_model(model,
                train_loader, dev_loader,
                epochs,
                optimizer, criterion,
                num_classes, target_classes,
                tokenizer,
                save_path, device):

    train_tqdm = tqdm(train_loader)
    dev_tqdm = tqdm(dev_loader)

    for epoch in range(epochs):
        model.train()

        for i, (train_x, train_y, mask) in enumerate(train_tqdm):
            optimizer.zero_grad()

            output = model.forward(train_x, mask)

            curr_loss = criterion(output.reshape(-1, num_classes).to(device), train_y.reshape(-1))
            curr_loss.backward()
            optimizer.step()



        model.eval()
        loss = 0
        acc = 0

        micro_prec = 0
        micro_recall = 0
        micro_f1 = 0

        macro_prec = 0
        macro_recall = 0
        macro_f1 = 0

        for i, (dev_x, dev_y, mask) in enumerate(dev_tqdm):
            dev_x, dev_y, mask = train_x, train_y, mask = cut_padding(dev_x, dev_y, mask, device,
                                                                      tokenizer.pad_token_id)

            output = model.forward(dev_x, mask)
            curr_loss = criterion(output.reshape(-1, num_classes).to(device), dev_y.reshape(-1))

            # --------------------------------------- Evaluate model ------------------------------------------------- #

            loss = (loss * i + curr_loss.item()) / (i + 1)

            pred = torch.tensor([torch.argmax(x) for x in output.view(-1, num_classes)])
            dev_y = dev_y.reshape(-1)  # reshape to linear vector
            curr_acc = accuracy_score(dev_y.cpu(), pred.cpu())

            curr_micro_prec = precision_score(dev_y.cpu(), pred.cpu(), labels=target_classes, average='micro')
            curr_micro_recall = recall_score(dev_y.cpu(), pred.cpu(), labels=target_classes, average='micro')
            curr_micro_f1 = f1_score(dev_y.cpu(), pred.cpu(), labels=target_classes, average='micro')

            curr_macro_prec = precision_score(dev_y.cpu(), pred.cpu(), labels=target_classes, average='macro')
            curr_macro_recall = recall_score(dev_y.cpu(), pred.cpu(), labels=target_classes, average='macro')
            curr_macro_f1 = f1_score(dev_y.cpu(), pred.cpu(), labels=target_classes, average='macro')

            acc = (acc * i + curr_acc) / (i + 1)
            micro_prec = (micro_prec * i + curr_micro_prec) / (i + 1)
            micro_recall = (micro_recall * i + curr_micro_recall) / (i + 1)
            micro_f1 = (micro_f1 * i + curr_micro_f1) / (i + 1)

            macro_prec = (macro_prec * i + curr_macro_prec) / (i + 1)
            macro_recall = (macro_recall * i + curr_macro_recall) / (i + 1)
            macro_f1 = (macro_f1 * i + curr_macro_f1) / (i + 1)

            dev_tqdm.set_description("Epoch: {}/{}, Dev Loss: {:.4f}, Dev Accuracy: {:.4f}, "
                                  "Dev Micro F1: {:.4f}, Dev Macro F1: {:.4f}".
                                  format(epoch, epochs, loss, acc, micro_f1, macro_f1))
            dev_tqdm.refresh()

        if macro_f1 > best_macro_f1:
            print("Macro F1 score improved from {:.4f} -> {:.4f}. Saving model...".format(best_macro_f1, macro_f1))
            best_macro_f1 = macro_f1
            torch.save(model, save_path + ".pt")
            with open(save_path + ".txt", "w") as file:
                file.write("Acc: {}, Macro Prec: {}, Macro Rec: {}, Macro F1: "
                           "{}, Micro Prec: {}, Micro Rec: {}, Micro F1: {}".format(acc,
                                                                                    macro_prec,
                                                                                    macro_recall,
                                                                                    macro_f1,
                                                                                    micro_prec,
                                                                                    micro_recall,
                                                                                    micro_f1))



def main():
    device = torch.device(args.device)

    train_loader, dev_loader, label_encoder = load_data(args.train_path,
                                                        args.dev_path,
                                                        args.batch_size,
                                                        args.tokens_column, args.predict_column,
                                                        args.lang_model_name,
                                                        args.max_len,
                                                        args.separator,
                                                        args.pad_label,
                                                        device)

    lang_model = AutoModel.from_pretrained(args.lang_model_name)
    input_size = 768 if "base" in args.lang_model_name else 1024

    model = LangModelWithDense(lang_model, input_size, len(label_encoder.classes_), False)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = torch.nn.CrossEntropyLoss()



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