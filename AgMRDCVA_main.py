import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from networks import Encoder
from utils import normalize_data, calculate_laplacian_matrix

# ================= 1. 环境配置与设备检测 =================
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Device: {device}")


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(42)


# ================= 2. 数据处理函数 =================
def create_dataset(testindex, attribute_matrix):
    data_files = [
        'data_ow_bubble.mat', 'data_ow_plug.mat', 'data_ow_slug.mat',
        'data_ow_wave.mat', 'data_ow_st.mat', 'data_ow_ann.mat',
        'data_wo_bubble.mat', 'data_wo_plug.mat', 'data_wo_slug.mat', 'data_wo_ann.mat'
    ]

    attr_mat = attribute_matrix.values
    train_data_list, train_attr_list, train_label_list = [], [], []
    test_data_list, test_attr_list, test_label_list = [], [], []

    for idx, data_file in enumerate(data_files):
        path = f'./OGW_mat_data/{data_file}'
        if not os.path.exists(path):
            print(f"Warning: {data_file} not found, skipping...")
            continue

        data = loadmat(path)[data_file.split('.')[0]]
        n = data.shape[0]

        # 训练集与测试集分割 (前200个样本测试)
        train_data_list.append(data[200:])
        test_data_list.append(data[:200])

        # 标签处理
        train_attr_list.append(np.tile(attr_mat[idx, :], (n - 200, 1)))
        train_label_list.append(np.full((n - 200, 1), idx))

        test_attr_list.append(np.tile(attr_mat[idx, :], (200, 1)))
        test_label_list.append(np.full((200, 1), idx))

    train_data = np.vstack(train_data_list)
    train_attr = np.vstack(train_attr_list)
    train_label = np.vstack(train_label_list)

    if testindex == 'data_ogw_test':
        test_data = np.vstack(test_data_list)
        test_label = np.vstack(test_label_list)
    else:
        # 过渡过程处理
        Tdata1 = loadmat(f'./OGW_mat_data/{testindex}.mat')[testindex]
        test_data = Tdata1
        test_label = np.zeros((Tdata1.shape[0], 1))  # 过渡标签仅占位

    return train_data, train_label, train_attr, test_data, test_label


def sliding_window(data, att_labels, class_labels, window_size, p):
    n_samples = data.shape[0]
    if window_size > n_samples:
        raise ValueError("Window size exceeds data length")

    windows_data, windows_att, windows_cls = [], [], []
    for i in range(n_samples - window_size + 1):
        windows_data.append(data[i:i + window_size])
        # 使用窗口内的第 p 个时刻作为标签点
        windows_att.append(att_labels[i + p])
        windows_cls.append(class_labels[i + p])

    return np.array(windows_data), np.array(windows_att), np.array(windows_cls)


# ================= 3. 模型训练 =================
def train_model(CVA_net1, CVA_net2, train_loader, optimizer1, optimizer2, epochs,
                criterion_att, criterion_class, params, p, f):
    CVA_net1.train()
    CVA_net2.train()
    loss_history = []

    # 损失权重解构
    p_man, p_cls, p_att, p_cov = params['man'], params['cls'], params['att'], params['cov']

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, Y_att, Y_cls in train_loader:
            X_batch, Y_att, Y_cls = X_batch.to(device), Y_att.to(device), Y_cls.to(device)
            batch_sz = X_batch.size(0)

            # 拆分过去与未来矩阵
            Xp, Xf = X_batch[:, :p, :], X_batch[:, -f:, :]

            # 前向传播
            out1, out_att, logits = CVA_net1(Xp)
            out2, _, _ = CVA_net2(Xf)

            # 1. 相关性损失 (最大化对角线相关性)
            combined = torch.cat((out1, out2), dim=0)
            corr = torch.corrcoef(combined.T)  # 特征间相关性
            corr_pf = corr[:out1.size(1), out1.size(1):]
            loss_corr = -torch.trace(corr_pf)  # 负迹最大化

            # 2. 局部流形正则化 (拉普拉斯)
            L = calculate_laplacian_matrix(Xp[:, -1, :].cpu().numpy(), n_neighbors=5)
            if isinstance(L, np.ndarray):
                L = torch.from_numpy(L).to(device).float()
            else:
                L = L.to(device).float()
            loss_man = torch.norm(out1.T @ L @ out1, p="fro") / batch_sz

            # 3. 属性与分类损失
            loss_cls = criterion_class(logits, Y_cls.squeeze().long())
            loss_att = criterion_att(out_att, Y_att)

            # 4. 防止平凡解 (协方差正则化)
            I = torch.eye(out1.size(1)).to(device)
            cov = (out1.T @ out1) / batch_sz
            loss_cov = torch.norm(cov - I, p="fro")

            # 总损失
            total_loss = loss_corr + p_man * loss_man + p_cls * loss_cls + p_att * loss_att + p_cov * loss_cov

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            total_loss.backward()
            optimizer1.step()
            optimizer2.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Total Loss: {avg_loss:.4f}")

    # 绘图
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color='royalblue')
    plt.title('Training Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


# ================= 4. 测试与工具函数 =================
def test_model(CVA_net1, X_train, X_test, p):
    CVA_net1.eval()
    with torch.no_grad():
        Xp_train = X_train[:, :p, :].to(device)
        Xp_test = X_test[:, :p, :].to(device)

        test_f, test_a, test_logits = CVA_net1(Xp_test)
        train_f, _, _ = CVA_net1(Xp_train)

    return train_f.cpu().numpy(), test_f.cpu().numpy(), test_logits.cpu()


def plot_confusion_matrix(y_true, y_pred, num_classes):
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc:.4f}")

    conf = confusion_matrix(y_true, y_pred)
    conf_norm = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    labels = [str(i + 1) for i in range(num_classes)]
    sns.heatmap(conf_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label");
    plt.ylabel("True Label");
    plt.title("Normalized Confusion Matrix")
    plt.show()


def pre_attribute_model(model, traindata, train_attributelabel, testdata):
    print('Attribute predictor: '+model)
    model_dict = {'SVR': SVR(kernel='rbf'), 'rf': RandomForestRegressor(n_estimators=50),
                  'Ridge': Ridge(alpha=1), 'Lasso': Lasso(alpha=0.1)}
    res_list = []
    for i in range(train_attributelabel.shape[1]):
        clf = model_dict[model]
        if max(train_attributelabel[:, i]) != 0:
            clf.fit(traindata, train_attributelabel[:, i])  # 训练数据和每个属性回归
            res = clf.predict(testdata)     # 用已拟合模型对测试数据进行属性预测
        else:
            res = np.zeros(testdata.shape[0])
        res_list.append(res.T)
    test_pre_attribute = np.mat(np.row_stack(res_list)).T

    return test_pre_attribute


# ================= 5. 主程序入口 =================
if __name__ == '__main__':
    p, f = 10, 10
    epoch = 100
    testindex = 'data_ogw_test'
    hparams = {'man': 0.6, 'cls': 0.2, 'att': 0.1, 'cov': 0.1}
    torch.set_default_dtype(torch.float32)

    # 数据准备
    attr_df = pd.read_excel('./OGW_attribute.xlsx', index_col='no')
    train_d, train_l, train_a, test_d, test_l = create_dataset(testindex, attr_df)
    train_d, test_d = normalize_data(train_d, test_d)

    # 滑动窗口
    win_train_x, win_train_a, win_train_l = sliding_window(train_d, train_a, train_l, p + f, p)
    win_test_x, _, win_test_l = sliding_window(test_d, test_l, test_l, p + f, p)

    # 封装 DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(win_train_x),
                                  torch.FloatTensor(win_train_a),
                                  torch.FloatTensor(win_train_l))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

    X_train_tensor = torch.FloatTensor(win_train_x)
    X_test_tensor = torch.FloatTensor(win_test_x)

    # 模型与优化器
    net1, net2 = Encoder().to(device), Encoder().to(device)
    opt1 = optim.Adam(net1.parameters(), lr=0.001, weight_decay=1e-5)
    opt2 = optim.Adam(net2.parameters(), lr=0.001, weight_decay=1e-5)

    print(">>> AgMRDCVA training...")
    train_model(net1, net2, train_loader, opt1, opt2, epoch,
                nn.MSELoss(), nn.CrossEntropyLoss(), hparams, p, f)

    print(">>> Testing...")
    train_feat, test_feat, test_logits = test_model(net1, X_train_tensor, X_test_tensor, p)

    if testindex == 'data_ogw_test':
        preds = torch.argmax(test_logits, dim=1).numpy()
        plot_confusion_matrix(win_test_l, preds, num_classes=len(np.unique(win_train_l)))
    else:
        test_attr_pre = pre_attribute_model('SVR', train_feat, win_train_a, test_feat)

        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 兼容多系统
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        im = ax1.imshow(test_attr_pre.T, aspect='auto', cmap='Blues')
        plt.colorbar(im, ax=ax1)

        max_logits = torch.max(test_logits, dim=1)[0].numpy()
        ax2.plot(max_logits, color='firebrick', lw=1.5)

        plt.tight_layout()
        plt.show()