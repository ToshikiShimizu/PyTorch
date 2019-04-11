# coding:utf-8
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import torch
from torch import nn
from sklearn.datasets import load_digits
import numpy as np
class BaseNN():
    def __init__(self):
        pass
    def fit(self, X, Y):
        # NumPyのndarrayをPyTorchのTensorに変換
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.int64)
        # Datasetを作成
        ds = TensorDataset(X, Y)

        # 異なる順番で64個ずつデータを返すDataLoaderを作成
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters())

        # 最適化を実行
        losses = []
        for epoch in range(10):
            running_loss = 0.0
            for xx, yy in loader:
                # xx, yyは64個分のみ受け取れる
                y_pred = self.net(xx)
                loss = loss_fn(y_pred, yy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            losses.append(running_loss)
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        return self.net(X).data.numpy()

if __name__=='__main__':
    digits = load_digits()
    X = digits.data
    Y = digits.target
    print (X.shape)
    print (Y.shape)
    base_nn = BaseNN()
    base_nn.fit(X, Y)
    pred = base_nn.predict(X)
    print (pred)
    print (Y)
    print ((np.argmax(pred,axis=1)==Y).mean())
