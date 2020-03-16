from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import os
import torch
from torch import nn
from torch.autograd import Variable
import csv
from itertools import chain
from sklearn import metrics
import math
import torch.nn.functional as F
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
"====================================================================================================="
def readSeq(ID):
    Path = "datapreprocessing/data/fasta/"
    filename = Path + ID + ".fasta"
    fr = open(filename)
    next(fr)
    Seq = fr.read().replace("\n", "")
    return Seq
def load(fileName):
    data = []
    # numFeat = len(open(fileName).readline().split('\t'))
    # print(numFeat)
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(' ')
        lineArr = (curLine[0], curLine[1])
        data.append(lineArr)
    return data
def Indd_stacking2(clf, train_data, test_data, OneEncoded, n_folds=5):
    train_num, test_num = len(train_data), len(test_data)     # 训练集的数量6204，测试集的数量1552
    second_level_train_set = np.zeros((train_num,))    # np.zeros((n,))为n维数组，形式为1*n，对应于第二层训练集
    second_level_test_set = np.zeros((test_num,))   # np.zeros((n,))为n维数组，形式为1*n，对应于第二层测试集
    test_nfolds_sets = np.zeros((test_num, n_folds))    #
    kf = KFold(n_splits=n_folds)    # 5折交叉实验
    train_data_ = [(np.array([OneEncoded[train_data[i][0]], OneEncoded[train_data[i][1]]]), train_data[i][2]) for i in range(len(train_data))]
    test_data_ = [(np.array([OneEncoded[test_data[i][0]],  OneEncoded[test_data[i][1]]]), test_data[i][2]) for i in range(len(test_data))]
    # 设置超参数
    # num_epochs = 2     # 迭代的次数
    net = clf()
    criterion = nn.CrossEntropyLoss()   # 定义损失函数为交叉熵
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)  # 使用梯度下降法进行优化
    for i, (train_index, test_index) in enumerate(kf.split(train_data_)):   # 将训练集的6204条数据划分为5份，每份
        print("n_folds = %d"%i)
        tra = np.array(train_data_)[train_index]
        tst = np.array(train_data_)[test_index]
        tra_ = [(torch.from_numpy(tra[i][0]), torch.from_numpy(np.array(tra[i][1]))) for i in range(len(tra))]
        tst_ = [(torch.from_numpy(tst[i][0]), torch.from_numpy(np.array(tst[i][1]))) for i in range(len(tst))]
        test_ = [(torch.from_numpy(test_data_[i][0]), torch.from_numpy(np.array(test_data_[i][1]))) for i in range(len(test_data_))]
        tra_ = DataLoader(tra_, batch_size=256, shuffle=True)
        tst_ = DataLoader(tst_, batch_size=128, shuffle=False)
        test_ = DataLoader(test_, batch_size=64, shuffle=False)
        second_level_train_set[test_index], test_nfolds_sets[:, i] = train(net, tra_, tst_, test_, num_epochs, optimizer, criterion)
    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set
def train(net, train_data, valid_data, x_test, num_epochs, optimizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
    output_valid_data = []
    output_x_test = []
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for a, a_label in train_data:
            a_label = a_label.long()
            a = torch.tensor(a, dtype=torch.float32)
            if torch.cuda.is_available():
                a = Variable(a.cuda())  # (64, 2, 128, 128)
                a_label = Variable(a_label.cuda())  # (64, 128, 128)
            else:
                a = Variable(a)     # 理想中，a的维度是2*128*128
                a_label = Variable(a_label)
            # forward
            output, output_label = net(a)
            loss = criterion(output, a_label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if valid_data is not None:
            Acc = 0
            Recall = 0
            Specificity = 0
            Precision = 0
            MCC = 0
            F1 = 0
            AUC = 0
            net = net.eval()
            for a, a_label in valid_data:
                a_label = a_label.long()
                a = torch.tensor(a, dtype=torch.float32)
                if torch.cuda.is_available():
                    a = Variable(a.cuda(), volatile=True)   # (64, 2, 128, 128)
                    a_label = Variable(a_label.cuda(), volatile=True)   # (64, 128, 128)
                else:
                    a = Variable(a, volatile=True)
                    a_label = Variable(a_label, volatile=True)
                output, output_label = net(a)
                _, pred_label_valid_data = output.max(1)
                if epoch == num_epochs-1:
                    output_valid_data.append(pred_label_valid_data)
                loss = criterion(output, a_label)
                loss.item()
                y_pred = output.max(1)[1]
                Acc += metrics.accuracy_score(a_label.cpu(), y_pred.cpu())
                Recall += metrics.recall_score(a_label.cpu(), y_pred.cpu())
                Precision += metrics.precision_score(a_label.cpu(), y_pred.cpu())
                F1 += metrics.f1_score(a_label.cpu(), y_pred.cpu())
                AUC += metrics.roc_auc_score(a_label.cpu(), y_pred.cpu())
            epoch_str = (
                    "Epoch %d. Accuracy: %f, Recall: %f, Precision: %f, F1: %f, AUC: %f"
                    % (epoch, Acc / len(valid_data), Recall / len(valid_data),
                       Precision / len(valid_data), F1 / len(valid_data),
                       AUC / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        print(epoch_str)
        output_x_test = []
        if x_test is not None:
            x_test_loss = 0
            x_test_acc = 0
            net = net.eval()
            for a, a_label in x_test:
                a_label = a_label.long()
                a = torch.tensor(a, dtype=torch.float32)
                if torch.cuda.is_available():
                    a = Variable(a.cuda(), volatile=True)   # (128, 2, 128, 128)
                    a_label = Variable(a_label.cuda(), volatile=True)   # (64, 128, 128)
                else:
                    a = Variable(a, volatile=True)
                    a_label = Variable(a_label, volatile=True)
                output, output_label = net(a)
                _, pred_label_x_test = output.max(1)
                if epoch == num_epochs-1:
                    output_x_test.append(pred_label_x_test)
    output_valid_data = [output_valid_data[i] for i in range(len(output_valid_data))]
    output_valid_data = np.array(list(chain(*output_valid_data)))
    output_x_test = [output_x_test[i] for i in range(len(output_x_test))]
    output_x_test = np.array(list(chain(*output_x_test)))
    return output_valid_data, output_x_test
# AC
def Auto_Covariance(seq):
    lg = 30     # will affect 'ac_array' down below
    AC_array = [[0 for u in range(lg)] for v in range(7)]
    mean_feature = [0, 0, 0, 0, 0, 0, 0]
    locate_feature = transfer_feature()     # 提取蛋白质序列中每个氨基酸残基的标准化后的特征
    for j in range(len(mean_feature)):
        for i in range(len(seq)):
            if (seq[i]=='X' or seq[i]=='U' or seq[i]==' ' or seq[i]=='B'):
                continue
            mean_feature[j] += locate_feature[seq[i]][j]
    for k in range(len(mean_feature)):
        mean_feature[k] /= len(seq)
    for lag in range(lg):
        for ac_fea in range(len(mean_feature)):
            AC_array[ac_fea][lag] = acsum(seq, lag, mean_feature, ac_fea, locate_feature)
    Auto_Covariance_feature = []
    for o in range(len(AC_array)):
        for p in range(len(AC_array[0])):
            Auto_Covariance_feature += [AC_array[o][p]]
    Auto_Covariance_feature = np.array(Auto_Covariance_feature)
    return Auto_Covariance_feature
def acsum(protein_array, lag, mean_feature, ac_fea, locate_feature):
    phychem_sum = 0
    for i in range (len(protein_array)-lag):
        if(protein_array[i]=='X' or protein_array[i+lag]=='X'
                or protein_array[i]=='U' or protein_array[i+lag]=='U'
                or protein_array[i]==' ' or protein_array[i+lag]==' '
                or protein_array[i]=='B' or protein_array[i+lag]=='B'):
            continue
        phychem_sum += (locate_feature[protein_array[i]][ac_fea]-mean_feature[ac_fea]) * (locate_feature[protein_array[i+lag]][ac_fea]-mean_feature[ac_fea])
    phychem_sum /= (len(protein_array)-lag+0.0000001)
    return phychem_sum
def CodingAC(protein):
    # 获取字典形式后根据序列进行编码，用字典进行保存所有的蛋白质的编码形式并返回
    count = 0
    proteinAC = dict()
    for p_key in protein.keys():
        if len(protein[p_key]) == 0:
            print(p_key)
        Ac = Auto_Covariance(protein[p_key])
        proteinAC[p_key] = Ac
        count = count + 1
        if count%200 == 0:
            print(count)
    return proteinAC
# CT
def conjoint_triad(seq):
    local_operate_array = aac_7_number_description(seq)
    vector_3_matrix = [[a,b,c,0] for a in range(1,8) for b in range(1,8) for c in range(1,8)]
    for m in range(len(local_operate_array)-2):
        vector_3_matrix[(local_operate_array[m]-1)*49+(local_operate_array[m+1]-1)*7+(local_operate_array[m+2]-1)][3] += 1
    CT_array=[]
    for q in range(343):
        CT_array+=[vector_3_matrix[q][3]]
    CT_array = np.array(CT_array)
    return CT_array
def CodingCT(protein):
    # 获取字典形式后根据序列进行编码，用字典进行保存所有的蛋白质的编码形式并返回
    count = 0
    proteinCT = dict()
    for p_key in protein.keys():
        if len(protein[p_key]) == 0:
            print(p_key)
        Ct = conjoint_triad(protein[p_key])
        proteinCT[p_key] = Ct
        count = count + 1
        if count%200 == 0:
            print(count)
    return proteinCT
# LD
def local_descriptors(seq):
    local_operate_array = aac_7_number_description(seq)
    A_point = math.floor(len(seq) / 4) - 1
    B_point = A_point * 2 + 1
    C_point = A_point * 3 + 2
    part_vector = []
    part_vector += construct_63_vector(local_operate_array[0:A_point])
    part_vector += construct_63_vector(local_operate_array[A_point:B_point])
    part_vector += construct_63_vector(local_operate_array[B_point:C_point])
    part_vector += construct_63_vector(local_operate_array[C_point:])
    part_vector += construct_63_vector(local_operate_array[0:B_point])
    part_vector += construct_63_vector(local_operate_array[B_point:])
    part_vector += construct_63_vector(local_operate_array[A_point:C_point])
    part_vector += construct_63_vector(local_operate_array[0:C_point])
    part_vector += construct_63_vector(local_operate_array[A_point:])
    part_vector += construct_63_vector(local_operate_array[math.floor(A_point / 2):math.floor(C_point - A_point / 2)])
    part_vector = np.array(part_vector)
    return part_vector
def construct_63_vector(part_array):
    simple_7 = [0 for n in range(7)]
    marix_7_7 = [[0 for n in range(7)] for m in range(7)]
    simple_21 = [0 for n in range(21)]
    simple_35 = [0 for n in range(35)]
    for i in range(len(part_array)):
        simple_7[part_array[i] - 1] += 1
        if (i < (len(part_array) - 1) and part_array[i] != part_array[i + 1]):
            if (part_array[i] > part_array[i + 1]):
                j, k = part_array[i + 1], part_array[i]
            else:
                j, k = part_array[i], part_array[i + 1]
            marix_7_7[j - 1][k - 1] += 1
    i = 0
    for j in range(7):
        for k in range(j + 1, 7):
            simple_21[i] = marix_7_7[j][k]
            i += 1
    residue_count = [0, 0, 0, 0, 0, 0, 0]
    for q in range(len(part_array)):
        residue_count[part_array[q] - 1] += 1
        if (residue_count[part_array[q] - 1] == 1):
            simple_35[5 * (part_array[q] - 1)] = q + 1
        elif (residue_count[part_array[q] - 1] == math.floor(simple_7[part_array[q] - 1] / 4)):
            simple_35[5 * (part_array[q] - 1) + 1] = q + 1
        elif (residue_count[part_array[q] - 1] == math.floor(simple_7[part_array[q] - 1] / 2)):
            simple_35[5 * (part_array[q] - 1) + 2] = q + 1
        elif (residue_count[part_array[q] - 1] == math.floor(simple_7[part_array[q] - 1] * 0.75)):
            simple_35[5 * (part_array[q] - 1) + 3] = q + 1
        elif (residue_count[part_array[q] - 1] == simple_7[part_array[q] - 1]):
            simple_35[5 * (part_array[q] - 1) + 4] = q + 1
    for o in range(7):
        simple_7[o] /= len(part_array)
    for p in range(21):
        simple_21[p] /= len(part_array)
    for m in range(35):
        simple_35[m] /= len(part_array)
    simple_63_vector = simple_7 + simple_21 + simple_35
    return simple_63_vector
def aac_7_number_description(protein_array):
    # 将蛋白质序列中的氨基酸进行分类
    local_operate_array=[]
    for i in range(len(protein_array)):
        if (protein_array[i] in 'AGV'):
            local_operate_array+=[1]
        elif (protein_array[i] in 'ILFP'):
            local_operate_array+=[2]
        elif (protein_array[i] in 'YMTS'):
            local_operate_array+=[3]
        elif (protein_array[i] in 'HNQW'):
            local_operate_array+=[4]
        elif (protein_array[i] in 'RK'):
            local_operate_array+=[5]
        elif (protein_array[i] in 'DE'):
            local_operate_array+=[6]
        elif (protein_array[i]=='C'):
            local_operate_array+=[7]
        else :
            local_operate_array+=[7]
    return local_operate_array
def CodingLD(protein):
    # 获取字典形式后根据序列进行编码，用字典进行保存所有的蛋白质的编码形式并返回
    count = 0
    proteinLD = dict()
    for p_key in protein.keys():
        if len(protein[p_key]) == 0:
            print(p_key)
        Ld = local_descriptors(protein[p_key])
        proteinLD[p_key] = Ld
        count = count + 1
        if count%200 == 0:
            print(count)
    return proteinLD
# PseAAC
def PseAAC(seq):
    nambda = 15  #
    omega = 0.05  #
    locate_feature = transfer_feature()  # 提取蛋白质序列中每个氨基酸残基的标准化后的特征
    AA_frequency = {'A': [0], 'C': [0], 'D': [0], 'E': [0], 'F': [0], 'G': [0], 'H': [0], 'I': [0], 'K': [0], 'L': [0],
                    'M': [0], 'N': [0], 'P': [0], 'Q': [0], 'R': [0], 'S': [0], 'T': [0], 'V': [0], 'W': [0], 'Y': [0]}
    A_class_feature = [0 for v in range(20)]  # 全为0的列表
    B_class_feature = []
    sum_frequency = 0
    sum_occurrence_frequency = 0
    for i in range(len(seq)):
        if (seq[i] == 'X' or seq[i] == 'U' or seq[i] == 'B'):
            continue
        AA_frequency[seq[i]][0] += 1
    for j in AA_frequency:
        sum_frequency += AA_frequency[j][0]
    for m in AA_frequency:
        if (sum_frequency == 0):
            s = [0 for b in range(35)]
            return s
        else:
            AA_frequency[m][0] /= sum_frequency
    for o in AA_frequency:
        sum_occurrence_frequency += AA_frequency[o][0]

    for k in range(1, nambda + 1):
        B_class_feature += [thet(seq, locate_feature, k)]
    Pu_under = sum_occurrence_frequency + omega * sum(B_class_feature)
    for l in range(nambda):
        B_class_feature[l] = (B_class_feature[l] * omega / Pu_under) * 100
    number_range = range(len(AA_frequency))
    for charater, number in zip(AA_frequency, number_range):
        A_class_feature[number] = AA_frequency[charater][0] / Pu_under * 100
    class_feature = A_class_feature + B_class_feature
    class_feature = np.array(class_feature)
    return class_feature
def thet(seq, locate_feature, t):
    sum_comp = 0
    for i in range(len(seq) - t):
        sum_comp += comp(seq[i], seq[i + t], locate_feature)
    if (len(seq) - t) == 0:
        sum_comp /= (len(seq) - t + 1)
    else:
        sum_comp /= (len(seq) - t)
    return sum_comp
def comp(Ri, Rj, locate_feature):
    theth = 0
    if (Ri == 'X' or Rj == 'X' or Ri == 'U' or Rj == 'U' or Ri == 'B' or Rj == 'B'):
        return 0
    else:
        for i in range(3):
            theth += pow(locate_feature[Ri][i] - locate_feature[Rj][i], 2)
        theth = theth / 3
        return theth
def transfer_feature():
    opposite_path = os.path.abspath('')
    with open(os.path.join(opposite_path, 'normalized_feature.csv')) as C:
        normalized_feature=csv.reader(C)
        feature_hash = {}
        amino_acid = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
        for charater in amino_acid:
            feature_hash[charater] = []
        for row in normalized_feature:
            i = 0
            for charater in amino_acid:
                feature_hash[charater] += [float(row[i])]
                i += 1
    return feature_hash
def CodingPseAAC(protein):
    # 获取字典形式后根据序列进行编码，用字典进行保存所有的蛋白质的编码形式并返回
    count = 0
    proteinPseAAC = dict()
    for p_key in protein.keys():
        PseA = PseAAC(protein[p_key])
        proteinPseAAC[p_key] = PseA
        count = count + 1
        if count%200 == 0:
            print(count)
    return proteinPseAAC

# Models
class layer_normalization(nn.Module):
    def __init__(self, features, epsilon=1e-8):
        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta
class multihead_attention(nn.Module):
    def __init__(self, num_units, num_heads=4, dropout_rate=0, causality=False):
        super(multihead_attention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.output_dropout = nn.Dropout(p=self.dropout_rate)
        self.normalization = layer_normalization(self.num_units)
    def forward(self, queries, keys, values):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        # Linear projections
        if torch.cuda.is_available():
            queries = queries.cuda()
            keys = keys.cuda()
            values = values.cuda()
        Q = self.Q_proj(queries)  # (N, T_q, C)     torch.Size([512, 64, 64])
        K = self.K_proj(keys)  # (N, T_q, C)        torch.Size([512, 64, 64])
        V = self.V_proj(values)  # (N, T_q, C)      torch.Size([512, 64, 64])
        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)     torch.Size([2048, 64, 16])
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)     torch.Size([2048, 64, 16])
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)     torch.Size([2048, 64, 16])
        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)
        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)
        # Key Masking
        key_masks = torch.sign(torch.abs(torch.sum(keys, dim=-1)))  # (N, T_k)
        key_masks = key_masks.repeat(self.num_heads, 1)  # (h*N, T_k)
        key_masks = torch.unsqueeze(key_masks, 1).repeat(1, queries.size()[1], 1)  # (h*N, T_q, T_k)
        if torch.cuda.is_available():
            padding = Variable(torch.ones(*outputs.size()) * (-2 ** 32 + 1)).cuda()
        else:
            padding = Variable(torch.ones(*outputs.size()) * (-2 ** 32 + 1))
        # print(padding.device)
        condition = key_masks.eq(0.).float()
        # print(condition.device)
        outputs = padding * condition + outputs * (1. - condition)
        # Causality = Future blinding
        if self.causality:
            diag_vals = torch.ones(*outputs[0, :, :].size())  # (T_q, T_k)
            tril = torch.tril(diag_vals, diagonal=0)  # (T_q, T_k)
            # print(tril)
            masks = Variable(torch.unsqueeze(tril, 0).repeat(outputs.size()[0], 1, 1))  # (h*N, T_q, T_k)
            padding = Variable(torch.ones(*masks.size()) * (-2 ** 32 + 1))
            condition = masks.eq(0.).float()
            outputs = padding * condition + outputs * (1. - condition)
        # Activation
        outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)
        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks
        # Dropouts
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)
        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)
        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        # Residual connection
        outputs += queries
        # Normalize
        outputs = self.normalization(outputs)  # (N, T_q, C)
        return outputs
def block(in_channel, out_channel, stride=1):
    layer = nn.Sequential(nn.BatchNorm2d(in_channel),
        nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False))
    return layer
class cnn_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(cnn_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2
        self.conv1 = block(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = block(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # if not self.same_shape:
        self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        x = self.conv2(x)
        x = F.relu(self.bn2(x), True)
        x = self.conv3(x)
        return x
class cnn2_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(cnn2_block, self).__init__()
        self.conv1 = block(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = block(out_channel, out_channel)
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        x = self.conv2(x)
        # x = self.bn1(x)
        return x

class Model4(nn.Module):
    """210"""
    def __init__(self):
        super(Model4, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*16, 2*16, 1, bias=False)
        self.linear_sim = nn.Linear(4*16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.dense_dim3 = 128
        self.clf_layer = nn.Sequential(nn.Linear(210*2+4*16+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2),  # nn.ELU(),
                                         nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                         nn.Linear(self.dense_dim2, self.dense_dim3),
                                         nn.Linear(self.dense_dim3, 2), nn.BatchNorm1d(2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(210, 1*16*16)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 210])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 16, 16])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 16, 16])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model41(nn.Module):
    """210"""
    def __init__(self):
        super(Model41, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*16, 2*16, 1, bias=False)
        self.linear_sim = nn.Linear(4*16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.clf_layer = nn.Sequential(nn.Linear(210*2+4*16+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2),  # nn.ELU(),
                                         nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                         nn.BatchNorm1d(self.dense_dim2), nn.Dropout(0.2),
                                         nn.Linear(self.dense_dim2, 2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(210, 1*16*16)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        # print('aggregate_layers--input_: {}'.format(input_.shape))
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 210])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 16, 16])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 16, 16])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model42(nn.Module):
    """210"""
    def __init__(self):
        super(Model42, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*16, 2*16, 1, bias=False)
        self.linear_sim = nn.Linear(4*16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.dense_dim3 = 128
        self.clf_layer = nn.Sequential(nn.Linear(210*2+4*16+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1),
                                         nn.Linear(self.dense_dim1, self.dense_dim2), nn.Dropout(0.3), nn.ELU(),
                                         nn.Linear(self.dense_dim2, self.dense_dim3), nn.ELU(),
                                         nn.Linear(self.dense_dim3, 2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(210, 1*16*16)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 210])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 16, 16])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 16, 16])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model43(nn.Module):
    """210"""
    def __init__(self):
        super(Model43, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*16, 2*16, 1, bias=False)
        self.linear_sim = nn.Linear(4*16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 128
        self.clf_layer = nn.Sequential(nn.Linear(210*2+4*16+3, self.dense_dim1),
                                       nn.Linear(self.dense_dim1, self.dense_dim2),
                                         nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(210, 1*16*16)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 210])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 16, 16])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 16, 16])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model44(nn.Module):
    """210"""
    def __init__(self):
        super(Model44, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*16, 2*16, 1, bias=False)
        self.linear_sim = nn.Linear(4*16, 1)
        self.dense_dim1 = 512
        self.clf_layer = nn.Sequential(nn.Linear(210*2+4*16+3, self.dense_dim1), nn.Linear(self.dense_dim1, 2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(210, 1*16*16)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 210])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 16, 16])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 16, 16])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model45(nn.Module):
    """210"""
    def __init__(self):
        super(Model45, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*16, 2*16, 1, bias=False)
        self.linear_sim = nn.Linear(4*16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 64
        self.dense_dim3 = 16
        self.clf_layer = nn.Sequential(nn.Linear(210*2+4*16+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1),
                                         nn.Linear(self.dense_dim1, self.dense_dim2), nn.BatchNorm1d(self.dense_dim2),
                                         nn.Linear(self.dense_dim2, self.dense_dim3),
                                         nn.Linear(self.dense_dim3, 2), nn.BatchNorm1d(2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(210, 1*16*16)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 210])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 16, 16])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 16, 16])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model46(nn.Module):
    """210"""
    def __init__(self):
        super(Model46, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*16, 2*16, 1, bias=False)
        self.linear_sim = nn.Linear(4*16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 128
        self.clf_layer = nn.Sequential(nn.Linear(210*2+4*16+3, self.dense_dim1), nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(), nn.BatchNorm1d(self.dense_dim2),
                                         nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(210, 1*16*16)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 210])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 16, 16])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 16, 16])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label

class Model5(nn.Module):
    """630"""
    def __init__(self):
        super(Model5, self).__init__()
        self.attention = multihead_attention(64)
        self.pro_len = 64
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*64, 2*64, 1, bias=False)
        self.linear_sim = nn.Linear(4*64, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.dense_dim3 = 128
        self.clf_layer = nn.Sequential(nn.Linear(630*2+4*64+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2),  # nn.ELU(),
                                         nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                         nn.Linear(self.dense_dim2, self.dense_dim3),
                                         nn.Linear(self.dense_dim3, 2), nn.BatchNorm1d(2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(630, 64*64)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 64, 64)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 630])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 64, 64])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 64, 64])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 64, 64])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 64, 64])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 64*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model51(nn.Module):
    """630"""
    def __init__(self):
        super(Model51, self).__init__()
        self.attention = multihead_attention(64)
        self.pro_len = 64
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*64, 2*64, 1, bias=False)
        self.linear_sim = nn.Linear(4*64, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.dense_dim3 = 128
        self.clf_layer = nn.Sequential(nn.Linear(630*2+4*64+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1),
                                         nn.Linear(self.dense_dim1, 2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(630, 64*64)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 64, 64)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 630])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 64, 64])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 64, 64])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 64, 64])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 64, 64])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 64*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model52(nn.Module):
    """630"""
    def __init__(self):
        super(Model52, self).__init__()
        self.attention = multihead_attention(64)
        self.pro_len = 64
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*64, 2*64, 1, bias=False)
        self.linear_sim = nn.Linear(4*64, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.dense_dim3 = 128
        self.clf_layer = nn.Linear(630*2+4*64+3, 2)
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(630, 64*64)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 64, 64)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 630])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 64, 64])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 64, 64])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 64, 64])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 64, 64])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 64*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model53(nn.Module):
    """630"""
    def __init__(self):
        super(Model53, self).__init__()
        self.attention = multihead_attention(64)
        self.pro_len = 64
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*64, 2*64, 1, bias=False)
        self.linear_sim = nn.Linear(4*64, 1)
        self.dense_dim1 = 1024
        self.dense_dim2 = 512
        self.dense_dim3 = 128
        self.clf_layer = nn.Sequential(nn.Linear(630*2+4*64+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1),   # nn.ELU(),
                                         nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                         nn.Linear(self.dense_dim2, self.dense_dim3), nn.Dropout(0.2),
                                         nn.Linear(self.dense_dim3, 2), nn.BatchNorm1d(2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(630, 64*64)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 64, 64)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 630])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 64, 64])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 64, 64])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 64, 64])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 64, 64])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 64*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model54(nn.Module):
    """630"""
    def __init__(self):
        super(Model54, self).__init__()
        self.attention = multihead_attention(64)
        self.pro_len = 64
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*64, 2*64, 1, bias=False)
        self.linear_sim = nn.Linear(4*64, 1)
        self.dense_dim1 = 1024
        self.dense_dim2 = 512
        self.dense_dim3 = 128
        self.clf_layer = nn.Sequential(nn.Linear(630*2+4*64+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2),  # nn.ELU(),
                                         nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                         nn.Linear(self.dense_dim2, self.dense_dim3), nn.ELU(),nn.BatchNorm1d(self.dense_dim3),
                                         nn.Linear(self.dense_dim3, 2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(630, 64*64)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 64, 64)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 630])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 64, 64])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 64, 64])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 64, 64])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 64, 64])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 64*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model55(nn.Module):
    """630"""
    def __init__(self):
        super(Model55, self).__init__()
        self.attention = multihead_attention(64)
        self.pro_len = 64
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*64, 2*64, 1, bias=False)
        self.linear_sim = nn.Linear(4*64, 1)
        self.dense_dim1 = 1024
        self.dense_dim2 = 256
        self.clf_layer = nn.Sequential(nn.Linear(630*2+4*64+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2),  # nn.ELU(),
                                         nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                         nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(630, 64*64)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 64, 64)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 630])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 64, 64])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 64, 64])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 64, 64])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 64, 64])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 64*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model56(nn.Module):
    """630"""
    def __init__(self):
        super(Model56, self).__init__()
        self.attention = multihead_attention(64)
        self.pro_len = 64
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*64, 2*64, 1, bias=False)
        self.linear_sim = nn.Linear(4*64, 1)
        self.dense_dim1 = 1024
        self.dense_dim2 = 512
        self.dense_dim3 = 64
        self.clf_layer = nn.Sequential(nn.Linear(630*2+4*64+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2),
                                         nn.Linear(self.dense_dim1, self.dense_dim2), nn.BatchNorm1d(self.dense_dim2),
                                         nn.Linear(self.dense_dim2, self.dense_dim3), nn.ELU(),
                                         nn.Linear(self.dense_dim3, 2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(630, 64*64)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 64, 64)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 630])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 64, 64])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 64, 64])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 64, 64])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 64, 64])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 64*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label

class Model6(nn.Module):
    """343"""
    def __init__(self):
        super(Model6, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*16, 2*16, 1, bias=False)
        self.linear_sim = nn.Linear(4*16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.dense_dim3 = 128
        self.clf_layer = nn.Sequential(nn.Linear(343*2+4*16+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2),  # nn.ELU(),
                                         nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                         nn.Linear(self.dense_dim2, self.dense_dim3),
                                         nn.Linear(self.dense_dim3, 2), nn.BatchNorm1d(2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(343, 1*16*16)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 210])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 16, 16])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 16, 16])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model61(nn.Module):
    """343"""
    def __init__(self):
        super(Model61, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*16, 2*16, 1, bias=False)
        self.linear_sim = nn.Linear(4*16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.dense_dim3 = 128
        self.clf_layer = nn.Sequential(nn.Linear(343*2+4*16+3, self.dense_dim1), nn.Dropout(0.2),  # nn.ELU(),
                                         nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                         nn.BatchNorm1d(self.dense_dim2), nn.ELU(),
                                         nn.Linear(self.dense_dim2, self.dense_dim3), nn.ELU(),
                                         nn.Linear(self.dense_dim3, 2), nn.BatchNorm1d(2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(343, 1*16*16)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 210])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 16, 16])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 16, 16])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model62(nn.Module):
    """343"""
    def __init__(self):
        super(Model62, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*16, 2*16, 1, bias=False)
        self.linear_sim = nn.Linear(4*16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.dense_dim3 = 128
        self.clf_layer = nn.Sequential(nn.Linear(343*2+4*16+3, self.dense_dim1), nn.Dropout(0.2), nn.Linear(self.dense_dim1, 2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(343, 1*16*16)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 210])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 16, 16])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 16, 16])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model63(nn.Module):
    """343"""
    def __init__(self):
        super(Model63, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*16, 2*16, 1, bias=False)
        self.linear_sim = nn.Linear(4*16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.clf_layer = nn.Sequential(nn.Linear(343*2+4*16+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2),  # nn.ELU(),
                                         nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                         nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(343, 1*16*16)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 210])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 16, 16])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 16, 16])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model64(nn.Module):
    """343"""
    def __init__(self):
        super(Model64, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*16, 2*16, 1, bias=False)
        self.linear_sim = nn.Linear(4*16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.clf_layer = nn.Sequential(nn.Linear(343*2+4*16+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1),   # nn.ELU(),
                                         nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(), nn.Dropout(0.2),
                                         nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(343, 1*16*16)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 210])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 16, 16])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 16, 16])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model65(nn.Module):
    """343"""
    def __init__(self):
        super(Model65, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*16, 2*16, 1, bias=False)
        self.linear_sim = nn.Linear(4*16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.clf_layer = nn.Sequential(nn.Linear(343*2+4*16+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2), nn.ELU(),
                                         nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(), nn.Dropout(0.2),
                                         nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(343, 1*16*16)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 210])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 16, 16])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 16, 16])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model66(nn.Module):
    """343"""
    def __init__(self):
        super(Model66, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*16, 2*16, 1, bias=False)
        self.linear_sim = nn.Linear(4*16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 128
        self.clf_layer = nn.Sequential(nn.Linear(343*2+4*16+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1),   # nn.ELU(),
                                         nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(), nn.Dropout(0.2),
                                         nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(343, 1*16*16)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 210])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 16, 16])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 16, 16])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label

class Model7(nn.Module):
    """35"""
    def __init__(self):
        super(Model7, self).__init__()
        self.attention = multihead_attention(8)
        self.pro_len = 8
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*8, 2*8, 1, bias=False)
        self.linear_sim = nn.Linear(4*8, 1)
        self.clf_layer = nn.Linear(4*8+3, 2)
        self.dense_dim1 = 64
        self.clf_layer = nn.Sequential(nn.Linear(35*2+4*8+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1), nn.ELU(),
                                         nn.Linear(self.dense_dim1, 2), nn.BatchNorm1d(2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(35, 8*8)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 8, 8)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        # print('aggregate_layers--input_: {}'.format(input_.shape))
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 630])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 64, 64])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 64, 64])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 64, 64])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 64, 64])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 64*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model71(nn.Module):
    """35"""
    def __init__(self):
        super(Model71, self).__init__()
        self.attention = multihead_attention(8)
        self.pro_len = 8
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*8, 2*8, 1, bias=False)
        self.linear_sim = nn.Linear(4*8, 1)
        self.clf_layer = nn.Linear(4*8+3, 2)
        self.dense_dim1 = 32
        self.clf_layer = nn.Sequential(nn.Linear(35*2+4*8+3, self.dense_dim1), nn.ELU(),
                                         nn.Linear(self.dense_dim1, 2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(35, 8*8)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 8, 8)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 630])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 64, 64])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 64, 64])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 64, 64])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 64, 64])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 64*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model72(nn.Module):
    """35"""
    def __init__(self):
        super(Model72, self).__init__()
        self.attention = multihead_attention(8)
        self.pro_len = 8
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*8, 2*8, 1, bias=False)
        self.linear_sim = nn.Linear(4*8, 1)
        self.clf_layer = nn.Linear(4*8+3, 2)
        self.dense_dim1 = 64
        self.clf_layer = nn.Linear(35*2+4*8+3, 2)
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(35, 8*8)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 8, 8)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out
    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 630])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 64, 64])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 64, 64])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 64, 64])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 64, 64])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 64*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model73(nn.Module):
    """35"""
    def __init__(self):
        super(Model73, self).__init__()
        self.attention = multihead_attention(8)
        self.pro_len = 8
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*8, 2*8, 1, bias=False)
        self.linear_sim = nn.Linear(4*8, 1)
        self.clf_layer = nn.Linear(4*8+3, 2)
        self.dense_dim1 = 64
        self.dense_dim2 = 32
        self.clf_layer = nn.Sequential(nn.Linear(35*2+4*8+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1), nn.ELU(),
                                         nn.Linear(self.dense_dim1, self.dense_dim2),
                                         nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(35, 8*8)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 8, 8)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        # print('aggregate_layers--input_: {}'.format(input_.shape))
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 630])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 64, 64])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 64, 64])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 64, 64])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 64, 64])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 64*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model74(nn.Module):
    """35"""
    def __init__(self):
        super(Model74, self).__init__()
        self.attention = multihead_attention(8)
        self.pro_len = 8
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*8, 2*8, 1, bias=False)
        self.linear_sim = nn.Linear(4*8, 1)
        self.clf_layer = nn.Linear(4*8+3, 2)
        self.dense_dim1 = 64
        self.dense_dim2 = 16
        self.clf_layer = nn.Sequential(nn.Linear(35*2+4*8+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1), nn.ELU(),
                                         nn.Linear(self.dense_dim1, self.dense_dim2),
                                         nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(35, 8*8)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 8, 8)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        # print('aggregate_layers--input_: {}'.format(input_.shape))
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 630])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 64, 64])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 64, 64])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 64, 64])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 64, 64])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 64*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model75(nn.Module):
    """35"""
    def __init__(self):
        super(Model75, self).__init__()
        self.attention = multihead_attention(8)
        self.pro_len = 8
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*8, 2*8, 1, bias=False)
        self.linear_sim = nn.Linear(4*8, 1)
        self.clf_layer = nn.Linear(4*8+3, 2)
        self.dense_dim1 = 64
        self.dense_dim2 = 16
        self.clf_layer = nn.Sequential(nn.Linear(35*2+4*8+3, self.dense_dim1),
                                         nn.BatchNorm1d(self.dense_dim1),
                                         nn.Linear(self.dense_dim1, self.dense_dim2),
                                         nn.BatchNorm1d(self.dense_dim2), nn.ELU(),
                                         nn.Linear(self.dense_dim2, 2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(35, 8*8)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 8, 8)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        # print('aggregate_layers--input_: {}'.format(input_.shape))
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 630])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 64, 64])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 64, 64])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 64, 64])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 64, 64])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 64*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label
class Model76(nn.Module):
    """35"""
    def __init__(self):
        super(Model76, self).__init__()
        self.attention = multihead_attention(8)
        self.pro_len = 8
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2*8, 2*8, 1, bias=False)
        self.linear_sim = nn.Linear(4*8, 1)
        self.clf_layer = nn.Linear(4*8+3, 2)
        self.dense_dim1 = 64
        self.dense_dim2 = 16
        self.clf_layer = nn.Sequential(nn.Linear(35*2+4*8+3, self.dense_dim1), nn.ELU(),
                                         nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                         nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))
    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32)
        netliearn = nn.Linear(35, 8*8)
        input_0 = netliearn(input_)     # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 8, 8)   # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)    #     [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        # print('aggregate_layers--input_: {}'.format(input_.shape))
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_
    def forward(self, X):
        proteinEncoded1, proteinEncoded2 = X.split(1, 1)    # proteinEncoded1: torch.Size([512, 1, 630])
        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 64, 64])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)  # proGatedCNN_2: torch.Size([512, 64, 64])
        conc_pro12 = X.view(X.shape[0], X.shape[2]*2)      # 512,1,1260
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()
        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)      # pro1_att: torch.Size([512, 64, 64])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())      # pro1_self: torch.Size([512, 64, 64])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)     # pro1_rep: torch.Size([512, 64*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()   # 余弦相似度torch.Size([512, 1])
        sen_bilinear_sim =self.Bilinear_sim(pro1_rep, pro2_rep)     # 双线性（Bilinear) torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))   # torch.Size([512, 1])
        merged = torch.cat((pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(), sen_linear_sim.cuda()), 1)      # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())   # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label


def getnewList(newlist):
    d = []
    for element in newlist:
        if not isinstance(element, list):
            d.append(element)
        else:
            d.extend(getnewList(element))

    return d

def text_save(filename, data):  #filename为写入CSV文件的路径，data为要写入数据列表.
    count = 0
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
        count = count + 1
    file.close()
    print("save = %d"%count)
    print("success")

def loadtt(fileName):
    data = []
    # numFeat = len(open(fileName).readline().split('\t'))
    # print(numFeat)
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(' ')
        lineArr = [curLine[0], curLine[1], int(curLine[2])]
        data.append(lineArr)
    return data

"====================================================================================================="
import random
"""
Step1 读取蛋白质相互作用数据
1 读取蛋白质互作对
2 对每个蛋白质进行编码
3 将编码后的蛋白质信息传入卷积进行特征提取

"""

# 1
disease = "Cardiac"
Ensnum_epoch = 500
num_epochs = 200

Po_pairs = load("Datasets/" + disease + "/Positive_pairs.txt")
Ne_pairs = load("Datasets/" + disease + "/Negative_pairs.txt")
# Po_pairs = [(Po_pairs[i][0], Po_pairs[i][1], 1) for i in range(len(Po_pairs))]  # 如果相互作用标记为1
# Ne_pairs = [(Ne_pairs[i][0], Ne_pairs[i][1], 0) for i in range(len(Ne_pairs))]  # 不相互作用标记为0
# Allpairs = Po_pairs + Ne_pairs  # 所有的数据
# X = [[Allpairs[i][0], Allpairs[i][1], Allpairs[i][2]] for i in range(len(Allpairs))]
# train_data, test_data = train_test_split(X, test_size=0.2)    # 这里的X
train_data = loadtt("Datasets/" + disease + "/train.txt")
test_data = loadtt("Datasets/" + disease + "/test.txt")
save_figdata = [[i, test_data[i][2]] for i in range(len(test_data))]

# 2
# 将涉及到的蛋白质保存
proteinID1 = [Po_pairs[i][0] for i in range(len(Po_pairs))]
proteinID2 = [Po_pairs[i][1] for i in range(len(Po_pairs))]
proteinID3 = [Ne_pairs[i][0] for i in range(len(Ne_pairs))]
proteinID4 = [Ne_pairs[i][1] for i in range(len(Ne_pairs))]
proteinID = list(set(proteinID1 + proteinID2 + proteinID3 + proteinID4))

# 获取每个蛋白质的序列并存入字典
proteinSeq = []
# print(proteinID[0])
# print(readSeq(proteinID[0]))
for ID in proteinID:
    Seq = readSeq(ID)
    proteinSeq.append(Seq)
protein = dict(zip(proteinID, proteinSeq))      # 形式如{ID: Seq}


# 获取字典形式后根据序列进行编码，用字典进行保存
proteinAC = CodingAC(protein)   # M4,M41,M42
proteinLD = CodingLD(protein)   # M5,M51,M52
proteinCT = CodingCT(protein)   # M6,M61,M62
proteinPseAAC = CodingPseAAC(protein)   # M7,M71,M72


proteindict = proteinAC, proteinLD, proteinCT, proteinPseAAC

# 3
"""以下是整个程序大致框架，需要填充每个学习器"""
"""get_stacking函数需要改，因为原分类器和pytorch有不同"""
train_sets = []
test_sets = []
models4list = [Model4, Model41, Model42, Model43]
models5list = [Model5, Model51, Model52, Model53]
models6list = [Model6, Model61, Model62, Model63]
models7list = [Model7, Model71, Model72, Model73]
num_model = 0
for i in range(len(proteindict)):   # 有三种不同编码形式
    if i == 0:
        for clf in models4list:  # models 使用列表形式表示，并且假设了三种模型
            print("================AC  num_model = %d  ================"%num_model)
            train_set, test_set = Indd_stacking2(clf, train_data, test_data, proteindict[i])
            train_sets.append(train_set)
            test_sets.append(test_set)
            num_model = num_model + 1
    elif i == 1:
        for clf in models5list:  # models 使用列表形式表示，并且假设了三种模型
            print("================LD  num_model = %d  ================"%num_model)
            train_set, test_set = Indd_stacking2(clf, train_data, test_data, proteindict[i])
            train_sets.append(train_set)
            test_sets.append(test_set)
            num_model = num_model + 1
    elif i == 2:
        for clf in models6list:  # models 使用列表形式表示，并且假设了三种模型
            print("================CT  num_model = %d  ================"%num_model)
            train_set, test_set = Indd_stacking2(clf, train_data, test_data, proteindict[i])
            train_sets.append(train_set)
            test_sets.append(test_set)
            num_model = num_model + 1
    else:
        for clf in models7list:  # models 使用列表形式表示，并且假设了三种模型
            print("================PseAAC  num_model = %d  ================"%num_model)
            train_set, test_set = Indd_stacking2(clf, train_data, test_data, proteindict[i])
            train_sets.append(train_set)
            test_sets.append(test_set)
            num_model = num_model + 1



# 特征为每个模型输出的结果
meta_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)
meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)

# 修改数据类型
meta_train = torch.tensor(meta_train, dtype=torch.float32)
meta_test = torch.tensor(meta_test, dtype=torch.float32)

print(meta_train.shape)     # 理想(6204,9)
print(meta_test.shape)      # 理想(1552,9)

meta_train = [(meta_train[i], torch.from_numpy(np.array(train_data[i][2]))) for i in range(len(meta_train))]
meta_test = [(meta_test[i], torch.from_numpy(np.array(test_data[i][2]))) for i in range(len(meta_test))]


# 保存第二层训练器的训练集和测试集
metatrain = []
for i in range(len(meta_train)):
    a = meta_train[i][0].numpy().tolist()
    al = meta_train[i][1].numpy().tolist()
    a.append(al)
    metatrain.append(a)

metatest = []
for i in range(len(meta_test)):
    a = meta_test[i][0].numpy().tolist()
    al = meta_test[i][1].numpy().tolist()
    a.append(al)
    metatest.append(a)

metatrain = np.mat(metatrain)
metatest = np.mat(metatest)

np.savetxt("Datasets/" + disease + "/metaData/ours_meta_train.txt", metatrain, fmt='%0.4f')
np.savetxt("Datasets/" + disease + "/metaData/ours_meta_test.txt", metatest, fmt='%0.4f')


# 迭代器
meta_train = DataLoader(meta_train, batch_size=512, shuffle=True)
meta_test = DataLoader(meta_test, batch_size=256, shuffle=True)



# 使用 Sequential 定义 4 层神经网络
# 因为输入的维度是模型的个数，输出的分类是二元分类；
in_netClaDNN = 16

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(in_netClaDNN, 16)
        self.l2 = nn.Linear(16, 2)
        self.l3 = nn.Linear(2, 2)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(2)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.bn1(x))
        x = self.bn2(self.l2(x))
        x = F.softmax(self.l3(x))
        return x
netClaDNN = Net()


# 定义 loss 函数
criterion = nn.CrossEntropyLoss()   # 定义损失函数为交叉熵
optimizer = torch.optim.SGD(netClaDNN.parameters(), 1e-1) # 优化使用随机梯度下降，学习率 0.1

# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []

print("================Ensemble=================")
epochspred = []
epochspredlabel = []
for e in range(Ensnum_epoch):
    train_loss = 0
    train_acc = 0
    netClaDNN.train()
    for a, a_label in meta_train:
        a_label = a_label.long()
        a = Variable(a)
        a_label = Variable(a_label)

        # 前向传播
        out = netClaDNN(a)
        loss = criterion(out, a_label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == a_label).sum().item()
        acc = num_correct / a.shape[0]
        train_acc += acc

    losses.append(train_loss / len(meta_train))
    acces.append(train_acc / len(meta_train))

    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    Acc = 0
    Recall = 0
    Specificity = 0
    Precision = 0
    MCC = 0
    F1 = 0
    AUC = 0
    netClaDNN.eval()  # 将模型改为预测模式

    onepred = []
    onepredlabel = []
    for a, a_label in meta_test:
        a_label = a_label.long()
        a = Variable(a)
        a_label = Variable(a_label)
        out = netClaDNN(a)
        loss = criterion(out, a_label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        y_pscore = out[:, 1]
        _, pred = out.max(1)
        num_correct = (pred == a_label).sum().item()
        acc = num_correct / a.shape[0]
        eval_acc += acc

        y_pred = pred
        Acc += metrics.accuracy_score(a_label.cpu(), y_pred.cpu())
        Recall += metrics.recall_score(a_label.cpu(), y_pred.cpu())
        Precision += metrics.precision_score(a_label.cpu(), y_pred.cpu())
        F1 += metrics.f1_score(a_label.cpu(), y_pred.cpu())
        AUC += metrics.roc_auc_score(a_label.cpu(), y_pred.cpu())

        onepred.append(y_pscore.tolist())
        onepred = getnewList(onepred)

        onepredlabel.append(pred.numpy().tolist())
        onepredlabel = getnewList(onepredlabel)

    epoch_str = (
            "Epoch %d. Accuracy: %f, Recall: %f, Precision: %f, F1: %f, AUC: %f"
            % (e, Acc / len(meta_test), Recall / len(meta_test),
               Precision / len(meta_test), F1 / len(meta_test),
               AUC / len(meta_test)))
    print(epoch_str)
    eval_losses.append(eval_loss / len(meta_test))
    eval_acces.append(eval_acc / len(meta_test))
    # print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
    #         .format(e, train_loss / len(meta_train), train_acc / len(meta_train),
    #                 eval_loss / len(meta_test), eval_acc / len(meta_test)))
    epochspred.append(onepred)
    epochspredlabel.append(onepredlabel)

epochspred = np.mat(epochspred).T
save_figdatapro = np.mat(save_figdata)
save_figdatapro = np.hstack((save_figdatapro, epochspred))
np.savetxt("Datasets/" + disease + "/figdata/ours_pro.txt", save_figdatapro, fmt='%0.6f')

epochspredlabel = np.mat(epochspredlabel).T
save_figdatalabel = np.mat(save_figdata)
save_figdatalabel = np.hstack((save_figdatalabel, epochspredlabel))
np.savetxt("Datasets/" + disease + "/figdata/ours_label.txt", save_figdatalabel, fmt='%0.6f')






