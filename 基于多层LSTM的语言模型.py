import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import give_valid_test
from time import time
import _pickle as cpickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def make_batch(train_path, word2number_dict, batch_size, n_step):
    """
    将数据转化为用于训练的格式
    :param train_path: 训练样本文件路径
    :param word2number_dict: 词表
    :param batch_size: 每一批的量
    :param n_step: 输入维度
    :return: 输入集合，输出集合
    """
    # 存放输入输出的所有batch
    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8')  # 打开训练样本文件

    # 构造一个batch
    input_batch = []
    target_batch = []
    for sen in text:
        word = sen.strip().split(" ")  # 按空格分词，并序列化表示
        word = ["<sos>"] + word
        word = word + ["<eos>"]

        if len(word) <= n_step:  # 如果语句长度小于输入+输出的长度，则进行填充
            word = ["<pad>"] * (n_step + 1 - len(word)) + word

        for word_index in range(len(word) - n_step):
            # 构造输入输出序列，并通过词表转换为数字表示
            input = [word2number_dict[n] for n in word[word_index: word_index + n_step]]
            target = word2number_dict[word[word_index + n_step]]

            # 添加到batch中，一个完整的input_batch为batch_size*n_step，target_batch为batch_size
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    return all_input_batch, all_target_batch  # (batch num, batch size, n_step) (batch num, batch size)


def make_dict(train_path):
    """
    构造单词表
    :param train_path: 训练样本文件路径
    :return: 单词和数字一一对应的两个词表
    """
    text = open(train_path, 'r', encoding='utf-8')  # 打开样本文件
    word_list = set()  # 使用集合

    for line in text:
        line = line.strip().split(" ")
        word_list = word_list.union(set(line))  # 将每句话的单词构成集合，然后通过“并”操作放入单词集合中

    word_list = list(sorted(word_list))  # 将集合转化为list类型的单词列表

    # 构造单词和数字一一对应的词表
    word2number_dict = {w: i + 2 for i, w in enumerate(word_list)}
    number2word_dict = {i + 2: w for i, w in enumerate(word_list)}

    # 向词表添加特殊意义符号，包括用于填充的<pad>，未知词<unk_word>，开始标志<sos>，结束标志<eos>
    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk_word>"] = 1
    number2word_dict[1] = "<unk_word>"
    word2number_dict["<sos>"] = 2
    number2word_dict[2] = "<sos>"
    word2number_dict["<eos>"] = 3
    number2word_dict[3] = "<eos>"

    return word2number_dict, number2word_dict


class MultilayerLSTM(nn.Module):
    """
    多层LSTM模型
    n_class: 单词类别数量
    emb_size: 词嵌入维度
    n_hidden: 隐藏单元数量
    n_layer: LSTM层数
    """
    def __init__(self, num_layer=1):
        super(MultilayerLSTM, self).__init__()
        if type(num_layer) is not int:
            raise TypeError('The parameter type of \'num_layer\' must be \'int\'')
        if num_layer < 1:
            raise ValueError('The number of layers must be at least 1')
        self.num_layer = num_layer

        # 词嵌入层
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)

        '''第1层LSTM，使用ModuleList和ParameterList保存每一层LSTM的参数，并初始填入第1层的内容'''
        # 输入门线性层
        self.all_W_i = nn.ModuleList([nn.Linear(n_hidden + emb_size, n_hidden, bias=False)])
        self.all_b_i = nn.ParameterList([nn.Parameter(torch.ones([n_hidden]))])
        self.all_W_c = nn.ModuleList([nn.Linear(n_hidden + emb_size, n_hidden, bias=False)])
        self.all_b_c = nn.ParameterList([nn.Parameter(torch.ones([n_hidden]))])

        # 遗忘门线性层
        self.all_W_f = nn.ModuleList([nn.Linear(n_hidden + emb_size, n_hidden, bias=False)])
        self.all_b_f = nn.ParameterList([nn.Parameter(torch.ones([n_hidden]))])

        # 输出门线性层
        self.all_W_o = nn.ModuleList([nn.Linear(n_hidden + emb_size, n_hidden, bias=False)])
        self.all_b_o = nn.ParameterList([nn.Parameter(torch.ones([n_hidden]))])

        '''第2到num_layer层LSTM，结构与第一层相同，依次填入列表中'''
        while len(self.all_W_i) < self.num_layer:
            # 输入门线性层
            self.all_W_i.append(nn.Linear(n_hidden + n_hidden, n_hidden, bias=False))
            self.all_b_i.append(nn.Parameter(torch.ones([n_hidden])))
            self.all_W_c.append(nn.Linear(n_hidden + n_hidden, n_hidden, bias=False))
            self.all_b_c.append(nn.Parameter(torch.ones([n_hidden])))

            # 遗忘门线性层
            self.all_W_f.append(nn.Linear(n_hidden + n_hidden, n_hidden, bias=False))
            self.all_b_f.append(nn.Parameter(torch.ones([n_hidden])))

            # 输出门线性层
            self.all_W_o.append(nn.Linear(n_hidden + n_hidden, n_hidden, bias=False))
            self.all_b_o.append(nn.Parameter(torch.ones([n_hidden])))

        # 激活函数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # 模型最终的线性输出层
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        X = self.C(X)  # X输入时为[batch_size * n_step]，通过词嵌入变为[batch_size * n_step * emb_size]
        X = X.transpose(0, 1)  # 变换为[n_step * batch_size * emb_size]

        # 初始化每一层上一时刻的隐藏层输入和记忆输入
        h_t_1 = [None]  # 预先设置一个空位，用于存放第一层LSTM的数据输入
        c_t_1 = []
        while len(h_t_1) <= self.num_layer:
            h_t_1.append(torch.zeros(batch_size, n_hidden).to(device))
            c_t_1.append(torch.zeros(batch_size, n_hidden).to(device))

        # 循环对每一个位置的词进行计算，x_t为[batch_size * emb_size]
        for x_t in X:
            h_t_1[0] = x_t  # 第一层LSTM的数据输入，将该输入视为第0层LSTM（实际不存在）的隐藏层输出，方便将每层的操作统一化编写

            # 循环执行每一层LSTM的计算操作
            for layer in range(self.num_layer):
                # 将上一时刻的隐藏层输出与下层LSTM的隐藏层输出拼接成一个矩阵[batch_size * (n_hidden + n_hidden)]
                # 其中第1层LSTM的一个输入为h_t_1[0]，即x_t，此时拼接的矩阵为[batch_size * (n_hidden + emb_size)]
                matrix = torch.cat((h_t_1[layer + 1], h_t_1[layer]), 1)

                # 输入门
                i_t = self.sigmoid(self.all_W_i[layer](matrix) + self.all_b_i[layer])
                _c_t = self.tanh(self.all_W_c[layer](matrix) + self.all_b_c[layer])

                # 遗忘门
                f_t = self.sigmoid(self.all_W_f[layer](matrix) + self.all_b_f[layer])

                # 记忆更新
                c_t = f_t * c_t_1[layer] + i_t * _c_t

                # 输出门
                o_t = self.sigmoid(self.all_W_o[layer](matrix) + self.all_b_o[layer])
                h_t = o_t * self.tanh(c_t)

                # 更新上一时刻隐藏层输出和记忆
                h_t_1[layer + 1] = h_t
                c_t_1[layer] = c_t

        model = self.W(h_t_1[-1]) + self.b
        return model


def train_LSTMlm():
    # 实例化模型
    model = MultilayerLSTM(num_layer=num_layer)
    model.to(device)
    print(model)

    # 使用交叉熵损失函数，Adam优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # 开始训练
    begin_time = time()  # 计时开始
    batch_number = len(all_input_batch)
    for epoch in range(all_epoch):
        count_batch = 0
        for input_batch, target_batch in zip(all_input_batch, all_target_batch):
            optimizer.zero_grad()

            # input_batch : [batch_size, n_step, n_class]
            output = model(input_batch)

            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            ppl = math.exp(loss.item())
            if (count_batch + 1) % 100 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                      'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

            loss.backward()
            optimizer.step()

            count_batch += 1
        print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
              'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

        # valid after training one epoch
        all_valid_batch, all_valid_target = give_valid_test.give_valid(data_root, word2number_dict, n_step)
        all_valid_batch = torch.LongTensor(all_valid_batch).to(device)  # list to tensor
        all_valid_target = torch.LongTensor(all_valid_target).to(device)

        total_valid = len(all_valid_target) * 128  # valid and test batch size is 128
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(all_valid_batch, all_valid_target):
                valid_output = model(valid_batch)
                valid_loss = criterion(valid_output, valid_target)
                total_loss += valid_loss.item()
                count_loss += 1

            print(f'Valid {total_valid} samples after epoch:', '%04d' % (epoch + 1), 'loss =',
                  '{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

        if (epoch + 1) % save_checkpoint_epoch == 0:
            torch.save(model, f"models/{num_layer}layer_LSTM_model_epoch{epoch + 1}.ckpt")
    run_time = time() - begin_time
    print('训练耗时：{:.2f}s'.format(run_time))


def test_LSTMlm(select_model_path):
    model = torch.load(select_model_path, map_location="cpu")  # load the selected model
    model.to(device)

    # load the test data
    all_test_batch, all_test_target = give_valid_test.give_test(data_root, word2number_dict, n_step)
    all_test_batch = torch.LongTensor(all_test_batch).to(device)  # list to tensor
    all_test_target = torch.LongTensor(all_test_target).to(device)
    total_test = len(all_test_target) * 128  # valid and test batch size is 128
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count_loss = 0
    for test_batch, test_target in zip(all_test_batch, all_test_target):
        test_output = model(test_batch)
        test_loss = criterion(test_output, test_target)
        total_loss += test_loss.item()
        count_loss += 1

    print(f"Test {total_test} samples with {select_model_path}……………………")
    print('loss =', '{:.6f}'.format(total_loss / count_loss),
          'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))


if __name__ == '__main__':
    num_layer = 3  # LSTM层数，可选大于等于1的整数
    n_step = 5  # 输入语句的长度，同时也是循环单元的数量
    n_hidden = 128  # number of hidden units in one cell
    batch_size = 128  # batch size
    learn_rate = 0.0005
    all_epoch = 5  # the all epoch for training
    emb_size = 256  # embeding size
    save_checkpoint_epoch = all_epoch  # save a checkpoint per save_checkpoint_epoch epochs !!! Note the save path !!!
    data_root = 'penn_small'
    train_path = os.path.join(data_root, 'train.txt')  # the path of train dataset

    print("print parameter ......")
    print("n_step:", n_step)
    print("n_hidden:", n_hidden)
    print("batch_size:", batch_size)
    print("learn_rate:", learn_rate)
    print("all_epoch:", all_epoch)
    print("emb_size:", emb_size)
    print("save_checkpoint_epoch:", save_checkpoint_epoch)
    print("train_data:", data_root)

    # 构造词表
    word2number_dict, number2word_dict = make_dict(train_path)
    # print(word2number_dict)

    # 单词的类别数量，即词表长度
    print("The size of the dictionary is:", len(word2number_dict))
    n_class = len(word2number_dict)  # n_class (= dict size)

    # 构造用于训练的输入输出数据
    print("generating train_batch ......")
    all_input_batch, all_target_batch = make_batch(train_path, word2number_dict, batch_size, n_step)  # make the batch
    train_batch_list = [all_input_batch, all_target_batch]

    # 将数据转换为张量表示
    print("The number of the train batch is:", len(all_input_batch))
    all_input_batch = torch.LongTensor(all_input_batch).to(device)  # list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)
    # print(all_input_batch.shape)
    # print(all_target_batch.shape)
    all_input_batch = all_input_batch.reshape(-1, batch_size, n_step)
    all_target_batch = all_target_batch.reshape(-1, batch_size)

    # 训练模型
    print("\nTrain the LSTMLM……………………")
    train_LSTMlm()

    # 测试模型
    print("\nTest the LSTMLM……………………")
    select_model_path = f"models/{num_layer}layer_LSTM_model_epoch{save_checkpoint_epoch}.ckpt"
    test_LSTMlm(select_model_path)
