import time
import torch
import mlflow
import pickle
import collections
import pandas as pd
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient


def load_data(train_dir):
    df_train = pd.read_csv(train_dir)
    train_x, train_y = df_train["sentence"].tolist(), df_train["label"].tolist()

    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    
    return train_x, train_y, encoder


def collate_batch(batch):
    text_pipeline = lambda x: vocab(x)
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(_label)
        text_list.append(text_pipeline(_text))

    label_list, text_list = torch.tensor(label_list, dtype=torch.int64), torch.tensor(text_list)
    return label_list.to(device), text_list.to(device)


def text_padding(pos_x, y, max_len, pad_token):
    iterator = []
    for i in range(0, len(pos_x)):
        if len(pos_x[i]) > max_len:
            iterator.append((y[i], pos_x[i][0:max_len]))
        else:
            iterator.append((y[i], pos_x[i][0:] + ([pad_token]*(max_len-len(pos_x[i])))))

    return iterator


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, max_len):
        super(TextCNN, self).__init__()
        num_channels = 100
        kernel_size = [3, 4, 5]
        dropout_keep = 0.5

        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[0] + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[1] + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=num_channels, kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[2] + 1)
        )

        self.dropout = nn.Dropout(dropout_keep)
        self.fc = nn.Linear(num_channels * len(kernel_size), num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded_sent = self.embeddings(x).permute(0, 2, 1)
        
        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)

        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        final_out = self.fc(final_feature_map)

        return self.softmax(final_out)


def train(dataloader, model):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 10

    for idx, (label, text) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = F.cross_entropy(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)

        if idx % log_interval == 0 and idx > 0:
            print('| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f}'.format(epoch, idx, len(dataloader), total_acc/total_count))
            mlflow.log_metric("train loss", loss.item())
            mlflow.log_metric("train acc", total_acc/total_count)
            
            total_acc, total_count = 0, 0


def test(dataloader, model):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            predicted_label = model(text)
            loss = F.cross_entropy(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

    return total_acc / total_count


if __name__ == "__main__":
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # exp_info = MlflowClient().get_experiment_by_name("nlp")
    # exp_id = exp_info.experiment_id if exp_info else MlflowClient().create_experiment("nlp")
    # with mlflow.start_run(experiment_id=exp_id) as run:

    mlflow.set_experiment('nlp')
    with mlflow.start_run() as run:
        Hyper Parameters
        train_dir = "train.csv"
        epochs = 3
        max_len = 30
        hidden_dim = 300
        lr = 0.001
        batch_size = 4
        total_acc = None
        device = torch.device("cpu")

        # Flow
        print("1. Load Data")
        train_x, train_y, encoder = load_data(train_dir)
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=321, stratify=train_y)

        print("2. Pre Processing")
        train_x = [sentence.split(" ") for sentence in train_x]
        val_x = [sentence.split(" ") for sentence in val_x]

        vocab = build_vocab_from_iterator(train_x, specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])

        with open('vocab.pickle', 'wb') as f:
            pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

        train_iter = text_padding(train_x, train_y, max_len, pad_token="<unk>")
        val_iter = text_padding(val_x, val_y, max_len, pad_token="<unk>")

        train_dataloader = DataLoader(train_iter, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        val_dataloader = DataLoader(val_iter, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

        print("3. Build Model")
        model = TextCNN(len(vocab), hidden_dim, num_class=len(list(set(train_y))), max_len=max_len).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

        print("4. Train")
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train(train_dataloader, model)
            acc_val = test(val_dataloader, model)
            if total_acc is not None and total_acc > acc_val:
                scheduler.step()
            else:
                total_acc = acc_val

            print('-' * 59)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid accuracy {:8.3f} '.format(epoch, time.time()-start_time, total_acc))
            print('-' * 59)

        # MLflow
        import random
        random_no = random.randrange(0, len(train_x))
        train_y = encoder.inverse_transform(train_y)

        mlflow.log_param("train", train_dir)
        mlflow.log_param("train num", len(train_x))
        mlflow.log_param("class num", len(set(train_y)))
        mlflow.log_param("class", collections.Counter(train_y))
        mlflow.log_param("train example", train_x[random_no])
        mlflow.log_param("train text max length", max([len(x) for x in train_x]))
        mlflow.log_param("train text average length", sum([len(x) for x in train_x])/len(train_x))
        mlflow.log_param("epochs", epochs)

        mlflow.pytorch.log_model(model, "model", pip_requirements=[f"torch=={torch.__version__}"])