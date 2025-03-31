from scipy.io import loadmat
from torch.utils.data import DataLoader
from CVA_model import Encoder
import torch.optim
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import seaborn as sns
import random
from utils import *
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dataset(testindex, attribute_matrix):
    data_files = [
        'data_ow_bubble.mat', 'data_ow_plug.mat', 'data_ow_slug.mat',
        'data_ow_wave.mat', 'data_ow_st.mat', 'data_ow_ann.mat',
        'data_wo_bubble.mat', 'data_wo_plug.mat', 'data_wo_slug.mat', 'data_wo_ann.mat'
    ]

    attribute_matrix = attribute_matrix.values

    train_data_list = []
    train_attribute_label_list = []
    train_label_list = []
    
    test_data_list = []
    test_attribute_label_list = []
    test_label_list = []

    for idx, data_file in enumerate(data_files):
        try:
            data = loadmat(f'./OGW_mat_data/{data_file}')[data_file.split('.')[0]]

            # 200 samples of each class are used for testing and the remaining samples are used for training
            n = data.shape[0]
            train_data_list.append(data[200:])
            attribute = [attribute_matrix[idx, :]] * (n - 200)  # Attributes for each sample
            train_attribute_label_list.append(attribute)
            train_label_list.append([[idx]] * (n - 200))        # class label for each sample
            
            test_data_list.append(data[:200])
            attribute = [attribute_matrix[idx, :]] * 200
            test_attribute_label_list.append(attribute)
            test_label_list.append([[idx]] * 200)

        except FileNotFoundError:
            print(f"Error: {data_file} not found!")
            return None

    train_data = np.row_stack(train_data_list)
    train_attribute_label = np.row_stack(train_attribute_label_list)
    train_label = np.row_stack(train_label_list)

    # test data
    if testindex == 'data_ogw_test':        # typical states
        test_data = np.row_stack(test_data_list)
        test_label = np.row_stack(test_label_list)

    else:       # transition states
        Tdata1 = loadmat(f'./OGW_mat_data/{testindex}.mat')[testindex]
        test_data = np.row_stack([Tdata1])
        test_label = np.row_stack(test_label_list)[:700]        # class label is useless for transition states

    return train_data, train_label, train_attribute_label, test_data, test_label
    
def sliding_window(data, att_labels, class_labels, window_size):
    n_samples, n_features = data.shape

    if window_size > n_samples:
        raise ValueError("The window size cannot be larger than the number of data samples")

    windows_data = []
    windows_att_labels = []
    windows_class_labels = []

    for i in range(n_samples - window_size + 1):
        # data for the current window
        window_data = data[i:i + window_size]
        # label of the current window (take the label of the last row of Xp)
        windows_att_label = att_labels[i + p]
        windows_class_label = class_labels[i + p]

        windows_data.append(window_data)
        windows_att_labels.append(windows_att_label)
        windows_class_labels.append(windows_class_label)

    windows_data = np.array(windows_data)
    windows_att_labels = np.array(windows_att_labels)
    windows_class_labels = np.array(windows_class_labels)

    return windows_data, windows_att_labels, windows_class_labels

def train_model(batch_size, CVA_net1, CVA_net2, train_loader, optimizer1, optimizer2, epochs, criterion_att,
                criterion_class, para_man, para_cls, para_att, para_cov, p=20, f=20, epsilon=1e-8):

    CVA_net1.train()
    CVA_net2.train()
    loss_history = []
    I = torch.eye(batch_size)

    for epoch in range(epochs):
        epoch_loss = 0.0

        for data in train_loader:
            X_data, Y_attribute, Y_class = data

            # Divide CVA past matrix and future matrix
            Xp = X_data[:, :p, :]
            Xf = X_data[:, -f:, :]

            # Forward pass
            output1, output_attri, class_logits = CVA_net1(Xp)
            output2, _, _ = CVA_net2(Xf)

            # Global serial correlation loss
            corr = torch.corrcoef(torch.cat((output1, output2), dim=0))
            corr_pf = corr[:batch_size, batch_size:]
            corr_diag = torch.diag(corr_pf).norm(p="fro")
            loss_corr = 1 / (corr_diag + epsilon)

            # Local manifold regularization loss
            Xp_batch = Xp[:, -1]
            L = calculate_laplacian_matrix(Xp_batch, n_neighbors=5)
            L = L.to(torch.float32)
            LPP_M = (output1.T) @ L @ output1
            loss_man = torch.norm(LPP_M, p="fro") / batch_size

            # Attribute embedding and attribute guidance loss
            loss_cls = criterion_class(class_logits, Y_class.squeeze().long())
            loss_att = criterion_att(output_attri, Y_attribute)

            # prevent trivial solutions
            Cov = torch.matmul(output1, output1.t()) / batch_size
            loss_cov = torch.norm(Cov - I, p="fro")

            # Total loss
            total_loss = loss_corr + para_man * loss_man + para_cls * loss_cls + para_att * loss_att + para_cov * loss_cov

            # Backpropagation and optimization
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            total_loss.backward()
            optimizer1.step()
            optimizer2.step()

            epoch_loss += total_loss.item()

        loss_history.append(epoch_loss / len(train_loader))

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Total_loss: {round(epoch_loss/len(train_loader), 4)} ")

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss', color='blue')
    plt.title('Training Loss over Steps')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_model(CVA_net1, X_train, X_test, p):
    CVA_net1.eval()
    with torch.no_grad():
        Xp_train = X_train[:, :p, :]
        Xp_test = X_test[:, :p, :]
        test_feature, test_attribute, test_class_logits = CVA_net1(Xp_test)
        test_feature = test_feature.detach().numpy()
        train_feature, train_attribute, train_class_logits = CVA_net1(Xp_train)
        train_feature = train_feature.detach().numpy()
    return train_feature, test_feature, test_class_logits

def plot_confusion_matrix(y_true, y_pred):
    # accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    num_classes = conf_matrix.shape[0]
    for i in range(num_classes):
        class_accuracy = conf_matrix[i, i] / conf_matrix[i].sum()  # 类别 i 的准确率
        print(f"Accuracy for Class {i}: {class_accuracy:.4f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], yticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def pre_attribute_model(model, train_feature, train_attributelabel, test_feature):
    model_dict = {'SVR': SVR(kernel='rbf'), 'rf': RandomForestRegressor(n_estimators=50),
                  'Ridge': Ridge(alpha=1), 'Lasso': Lasso(alpha=0.1)}
    res_list = []
    for i in range(train_attributelabel.shape[1]):
        clf = model_dict[model]
        if max(train_attributelabel[:, i]) != 0:
            clf.fit(train_feature, train_attributelabel[:, i])
            res = clf.predict(test_feature)
        else:
            res = np.zeros(test_feature.shape[0])
        res_list.append(res.T)
    test_pre_attribute = np.mat(np.row_stack(res_list)).T

    return test_pre_attribute

def sliding_window_average(arr, window_size):
    return np.array([np.convolve(row, np.ones(window_size)/window_size, mode='valid') for row in arr])


if __name__ == '__main__':

    set_random_seed()
    CVA_net1 = Encoder()
    CVA_net2 = Encoder()
    p = 20
    f = 20

    testindex = 'data_ogw_test'     # data_ogw_test, data_transition_1
    print("Test case: " + str(testindex))

    # Create dataset
    attribute_matrix = pd.read_excel('./OGW_attribute.xlsx', index_col='no')
    train_data, train_label, train_attribute_label, test_data, test_label = create_dataset(testindex, attribute_matrix)
    train_data, test_data = normalize_data(train_data, test_data)

    # CVA sliding window processing
    windows_train_data, windows_train_attribute_label, windows_train_label = sliding_window(train_data, train_attribute_label, train_label, p + f)
    windows_test_data, _, windows_test_label = sliding_window(test_data, test_label, test_label, p + f)

    # Convert data to tensors
    X_train, Y_train_att, Y_train_class = windows_train_data, windows_train_attribute_label, windows_train_label
    X_train, Y_train_att, Y_train_class = torch.FloatTensor(X_train), torch.FloatTensor(Y_train_att), torch.FloatTensor(windows_train_label)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train_att, Y_train_class)
    X_test = torch.FloatTensor(windows_test_data)

    # Parameter setting
    epochs = 100
    learning_rate = 0.01
    batch_size = 64
    para_man = 0.6
    para_cls = 0.2
    para_att = 0.1
    para_cov = 0.1

    criterion_att = torch.nn.MSELoss()
    criterion_class = torch.nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(CVA_net1.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
    optimizer2 = torch.optim.Adam(CVA_net2.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print("================= AgMRDCVA Training =================")
    train_model(batch_size, CVA_net1, CVA_net2, train_loader, optimizer1, optimizer2, epochs, criterion_att,
                criterion_class, para_man, para_cls, para_att, para_cov, p, f, epsilon=1e-8)

    print("================= AgMRDCVA Testing =================")
    train_feature, test_feature, test_class_logits = test_model(CVA_net1, X_train, X_test, p)

    if testindex == 'data_ogw_test':
        print('Identification for typical flow states')
        predicted_classes = torch.argmax(test_class_logits, dim=1)
        plot_confusion_matrix(windows_test_label, predicted_classes)

    else:
        print('Attribute evolution monitoring for transition states')
        print("Attribute prediction...")
        test_pre_attribute = pre_attribute_model('SVR', train_feature, windows_train_attribute_label, test_feature)
        test_pre_attribute = np.array(test_pre_attribute)

        Maximum_logits = torch.max(test_class_logits, dim=1)[0]
        Maximum_logits = Maximum_logits.unsqueeze(1)
        Maximum_logits = Maximum_logits.detach().numpy()
        Maximum_logits = sliding_window_average(Maximum_logits.T, p)

        test_pre_attribute = sliding_window_average(test_pre_attribute.T, p) / 5

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))

        axs[0].imshow(test_pre_attribute, interpolation='nearest', cmap='Purples', origin='upper', aspect='auto')
        axs[0].set_title('Flow state evolution')
        axs[0].set_xlabel("Samples")
        axs[0].set_ylabel("State attributes")
        axs[0].set_yticks(range(11))
        axs[0].set_yticklabels(['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11'])

        axs[1].plot(Maximum_logits[0, :], color='b')
        axs[1].set_title('Class logits')

        plt.colorbar(axs[0].images[0], ax=axs[0])
        plt.tight_layout()
        plt.show()