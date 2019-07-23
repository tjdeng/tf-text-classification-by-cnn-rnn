import numpy as np
import pickle
import re
def data_processing(positive_data_file, negative_data_file):
    # 处理输入的影评，每一行是一些词
    # x = [N,max_len,300]
    neg_dir = negative_data_file
    pos_dir = positive_data_file
    with open(neg_dir, "r", encoding='Windows-1252') as f:
        data = f.read().split('\n')  # 读取每一行的单词
        neg_words = [0] * len(data)
        for d in range(len(data)):
            neg_words[d] = data[d].split(' ')  # 以空格划分单词
    print(neg_words[1])
    max_len = 0
    for d in neg_words:
        if len(d) > max_len:
            max_len = len(d)
    print(max_len)
    with open(pos_dir, "r", encoding='Windows-1252') as f:
        data = f.read().split('\n')
        pos_words = [0] * len(data)
        for d in range(len(data)):
            pos_words[d] = data[d].split(' ')
    print(pos_words[1])
    for d in pos_words:
        if len(d) > max_len:
            max_len = len(d)
    print(max_len)

    # word_to_vector
    vectors_dir = r'rt-polaritydata/test2.w2v'
    with open(vectors_dir, "r", encoding='Windows-1252') as f:
        data = f.read()
        # data = str(data)
        data = data.split('\n')
        i = 0
        word_to_vec = {}
        vec = []
        print(len(data))
        for d in range(len(data)):
            if i:
                dd = data[d].split(' ')
                word = dd[0]
                vecs = dd[1:]
                vec = []
                for v in vecs:
                    if v and word is not '':
                        vec.append(float(v))
                word_to_vec[word] = vec
            i += 1

    # text word to vector
    worddim = 300
    null_fill = [0.0] * worddim
    x_neg = np.zeros((len(neg_words), max_len, worddim))
    print(x_neg.shape)
    for line in range(len(neg_words)):
        for word_ind in range(max_len):
            if word_ind >= len(neg_words[line]):
                x_neg[line][word_ind] = np.array(null_fill)
            else:
                if neg_words[line][word_ind] in word_to_vec and word_to_vec[neg_words[line][word_ind]]:
                    x_neg[line][word_ind] = np.array(word_to_vec[neg_words[line][word_ind]])
                else:
                    x_neg[line][word_ind] = np.array(null_fill)

    null_fill = [0.0] * worddim
    x_pos = np.zeros((len(pos_words), max_len, worddim))
    print(x_pos.shape)
    for line in range(len(pos_words)):
        for word_ind in range(max_len):
            if word_ind >= len(pos_words[line]):
                x_pos[line][word_ind] = np.array(null_fill)
            else:
                if pos_words[line][word_ind] in word_to_vec and word_to_vec[pos_words[line][word_ind]]:
                    x_pos[line][word_ind] = np.array(word_to_vec[pos_words[line][word_ind]])
                else:
                    x_pos[line][word_ind] = np.array(null_fill)

    # x of shape(14012, 447, 300) and y of shape(14012,)
    from sklearn.preprocessing import OneHotEncoder
    y_neg = np.zeros((len(neg_words)))
    y_pos = np.ones((len(pos_words)))
    # y_ = np.concatenate((y_neg,y_pos),axis=0)
    # x_ = np.concatenate((x_neg,x_pos),axis=0)
    train_data = np.concatenate((x_neg[len(neg_words) // 10:], x_pos[len(neg_words) // 10:]), axis=0)  # 将积极和消极的数据连接起来
    train_label = np.concatenate((y_neg[len(neg_words) // 10:], y_pos[len(neg_words) // 10:]), axis=0)

    ohe = OneHotEncoder()
    ohe.fit([[0], [1]])
    train_label = np.array(ohe.transform(np.transpose([train_label, ])).toarray())  # trainsform into onehot label

    np.random.seed(231)
    np.random.shuffle(train_data)
    np.random.seed(231)
    np.random.shuffle(train_label)

    test_data = np.concatenate((x_neg[:len(neg_words) // 10], x_pos[:len(pos_words) // 10]), axis=0)
    test_label = np.concatenate((y_neg[:len(neg_words) // 10], y_pos[:len(pos_words) // 10]), axis=0)
    test_label = np.array(ohe.transform(np.transpose([test_label, ])).toarray())  # trainsform into onehot label
    np.random.seed(131)
    np.random.shuffle(test_data)
    np.random.seed(131)
    np.random.shuffle(test_label)
    print(test_data.shape)
    print(test_label.shape)
    print(train_data.shape)

    return [train_data, train_label, test_data, test_label]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        print("\nnum_epochs:{}/{}".format(epoch, num_epochs))
        print("")
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def padding_sentences(input_sentences, padding_token, padding_sentence_length = None):
    sentences = [sentence.split(' ') for sentence in input_sentences]
    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max([len(sentence) for sentence in sentences])
    for sentence in sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
    return (sentences, max_sentence_length)

def saveDict(input_dict, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(input_dict, f)

def loadDict(dict_file):
    output_dict = None
    with open(dict_file, 'rb') as f:
        output_dict = pickle.load(f)
    return output_dict

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='Windows-1252').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='Windows-1252').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]