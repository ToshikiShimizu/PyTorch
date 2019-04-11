# coding:utf-8
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from torchvision import models
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.model_selection import train_test_split
class MultiLabelResNet():
    def __init__(self, bs, epoch = 10):
        self.batch_size = bs
        self.epoch = epoch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
    def score(self, yy, y_pred):
        return label_ranking_average_precision_score(yy.data.to("cpu").numpy(), y_pred.data.to("cpu").numpy())

    def get_loader(self, X, Y):
        X = np.stack([X, X, X], axis=-1).reshape(-1,3,8,8)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)
        ds = TensorDataset(X, Y)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        return loader

    def fit(self, X, Y, X_val, Y_val):
        if X_val is None or Y_val is None:
            Valid = False
        else:
            Valid = True
        self.n_in = X.shape[1]
        self.n_out = Y.shape[1]
        loader = self.get_loader(X, Y)
        if Valid:
            valid_loader = self.get_loader(X_val, Y_val)
        self.net = models.resnet18(pretrained=False)
        self.net.fc = nn.Linear(512,10)
        self.net = self.net.to(self.device)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.net.parameters())

        # 最適化を実行
        self.losses = []
        self.train_precisions = []
        self.valid_precisions = []
        for epoch in range(self.epoch):
            running_loss = 0.0
            train_precision = []
            for xx, yy in loader:
                y_pred = self.net(xx)
                loss = loss_fn(y_pred, yy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_precision.append(self.score(yy, y_pred))
            self.train_precisions.append(np.mean(train_precision))
            self.losses.append(running_loss)
            if Valid:
                for xx, yy in valid_loader:
                    valid_precision = []
                    y_pred = self.net(xx)
                    valid_precision.append(self.score(yy, y_pred))
                self.valid_precisions.append(np.mean(valid_precision))
    def predict(self, X):
        X = np.stack([X, X, X], axis=-1).reshape(-1,3,8,8)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        return self.net(X).data.to("cpu").numpy()



if __name__=='__main__':
    digits = load_digits()
    X = digits.data
    Y = digits.target
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(Y.reshape(-1,1))
    print (X.shape)
    print (Y.shape)
    X, X_val, Y, Y_val = train_test_split(X, Y,train_size=0.8) 
    multi_label_resnet = MultiLabelResNet(64,2)
    multi_label_resnet.fit(X, Y, X_val, Y_val)
    pred = multi_label_resnet.predict(X)
    print (multi_label_resnet.train_precisions)
    print (multi_label_resnet.valid_precisions)
