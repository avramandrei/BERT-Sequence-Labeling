from transformers import *
import torch
from sklearn.preprocessing import LabelEncoder


def load_data(path, batch_size, tokens_column, predict_column, lang_model, max_len, separator, pad_label, device):
    tokenizer = AutoTokenizer.from_pretrained(lang_model)
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    list_all_tokens = []
    list_all_labels = []
    list_all_masks = []

    with open(path, "r", encoding='utf-8') as file:
        list_tokens = []
        list_labels = []

        for line in file:
            if "#" not in line and "-DOCSTART- -X- O O" not in line:
                if line is not "\n":
                    tokens = line.split(separator)

                    token = tokens[tokens_column]
                    label = tokens[predict_column].replace("\n", "")

                    subtokens = tokenizer.encode(token)

                    list_tokens += subtokens
                    list_labels += [label] * len(subtokens)
                else:
                    assert len(list_tokens) == len(list_labels)

                    if len(list_tokens) == 0:
                        continue

                    list_all_tokens.append(torch.tensor([cls_token_id] + list_tokens + [sep_token_id]))
                    list_all_labels.append([pad_label] + list_labels + [pad_label])
                    list_all_masks.append(torch.tensor([1] * (len(list_tokens) + 2)))

                    assert len(list_tokens) == len(list_labels)

                    list_tokens = []
                    list_labels = []

    assert len(list_all_tokens) == len(list_all_labels) == len(list_all_masks)

    label_encoder = LabelEncoder()
    label_encoder.fit(sum(list_all_labels, []))
    list_all_encoded_labels = [torch.tensor(label_encoder.transform(list_labels)) for list_labels in list_all_labels]

    X = torch.nn.utils.rnn.pad_sequence(list_all_tokens, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    y = torch.nn.utils.rnn.pad_sequence(list_all_encoded_labels, batch_first=True, padding_value=0).to(device)
    masks = torch.nn.utils.rnn.pad_sequence(list_all_masks, batch_first=True, padding_value=0).to(device)

    dataset = torch.utils.data.TensorDataset(X, y, masks)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader