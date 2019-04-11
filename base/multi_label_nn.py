# coding:utf-8
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
class MultiLabelNN():
    def __init__(self, bs, epoch = 10):
        self.batch_size = bs
        self.epoch = epoch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        pass
    def fit(self, X, Y):
        self.n_in = X.shape[1]
        self.n_out = Y.shape[1]
        # NumPyのndarrayをPyTorchのTensorに変換
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)
        # Datasetを作成
        ds = TensorDataset(X, Y)

        # 異なる順番で64個ずつデータを返すDataLoaderを作成
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.net = nn.Sequential(
            nn.Linear(self.n_in, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_out)
        )
        self.net = self.net.to(self.device)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.net.parameters())

        # 最適化を実行
        self.losses = []
        for epoch in range(self.epoch):
            running_loss = 0.0
            for xx, yy in loader:
                # xx, yyは64個分のみ受け取れる
                y_pred = self.net(xx)
                loss = loss_fn(y_pred, yy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            self.losses.append(running_loss)
    def predict(self, X):
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
    multi_label_nn = MultiLabelNN(64)
    multi_label_nn.fit(X, Y)
    pred = multi_label_nn.predict(X)
    print (pred)
    print (Y)
    print ((np.argmax(pred,axis=1)==np.argmax(Y,axis=1)).mean())
