import torch.nn as nn
import torch


class LangModelWithDense(nn.Module):
    def __init__(self, lang_model, input_size, num_classes, fine_tune):

        super(LangModelWithDense, self).__init__()
        self.num_classes = num_classes
        self.fine_tune = fine_tune

        self.lang_model = lang_model

        self.linear = nn.Linear(input_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        if not self.fine_tune:
            with torch.no_grad():
                self.lang_model.eval()
                embeddings = self.lang_model(x)[0]
        else:
            embeddings = self.lang_model(x)[0]

        outputs = torch.zeros((batch_size, seq_len, self.num_classes))
        for i in range(seq_len):
            output = self.dropout(self.linear(embeddings[:, i, :]))

            outputs[:, i, :] = output

        return outputs.reshape(-1, self.num_classes * seq_len)