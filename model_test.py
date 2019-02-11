from keras.preprocessing.text import Tokenizer
from keras.models import Model,load_model
from keras import backend as K 
import keras
import tensorflow as tf 
import numpy as np
import os

# Set number of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set memory of gpu
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

MAX_SEQUENCE_LENGTH = 200

def generate_tokenizer(file_path_bad, file_path_good, file_path_bad2=None, file_path_good2=None):
    """Generate tokenizer to convert text to digit sequences"""
    f1 = open(file_path_bad)
    f2 = open(file_path_good)
    bad_data = f1.read().split("\n")
    good_data = f2.read().split("\n")
    if file_path_bad2 and file_path_good2:
        f3 = open(file_path_bad2)
        f4 = open(file_path_good2)
        bad_data2 = f3.read().split("\n")
        good_data2 = f4.read().split("\n")
        data = bad_data + good_data + bad_data2 + good_data2
    else:
        data = bad_data + good_data
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', lower=False,split=" ", char_level=False)
    tokenizer.fit_on_texts(data)
    return tokenizer
    

def generate_sequences(tokenizer, file_target):
    """Generate digit sequences"""
    f = open(file_target)
    target_data = f.read().split("\n")

    sequences = tokenizer.texts_to_sequences(target_data)
    result_sequences = []
    # Padding sequences
    for seq in sequences:
        while len(seq) < MAX_SEQUENCE_LENGTH:
            seq.insert(0, 0)
        else:
            seq = seq[len(seq)-MAX_SEQUENCE_LENGTH:]
        seq = [str(i) for i in seq]
        result_sequences.append(seq)
    return result_sequences

def model_predict(model, templates, data):
    """Use model to predict test files"""
    pairs_1 = []
    pairs_2 = []
    score = []
    for j in range(len(templates)):
        pairs_1.append(data)
        pairs_2.append(templates[j])
    pairs_1 = np.array(pairs_1)
    pairs_2 = np.array(pairs_2)
    score_list = model.predict([pairs_1, pairs_2])
    score = sum(score_list)/len(templates)
    return score

def Metrics(TP, FP, TN, FN):
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*(P*R)/(P+R)
    FNR = FN/(FN+TP)
    FPR = FP/(FP+TN)
    return (P, R, F1, FNR, FPR)

def test_SQL():
    """Single vulnerability detection test: SQL"""
    tokenizer = generate_tokenizer('data/train/bad_opcode_SQL.txt', 'data/train/good_opcode_SQL.txt')
    templates = generate_sequences(tokenizer, 'templates/templates_SQL.txt')
    bad_data = generate_sequences(tokenizer, 'data/test/bad_opcode_SQL.txt')
    good_data = generate_sequences(tokenizer, 'data/test/good_opcode_SQL.txt')
    model = load_model('model/bilstm_model_SQL.h5')
    TP = 0
    FN = 0
    for data in bad_data:
        score = model_predict(model, templates, data)
        # print(score)
        if score > 0.5:
            TP += 1
        else:
            FN += 1
    print('------------------------------------------')
    TN = 0
    FP = 0
    for data in good_data:
        score = model_predict(model, templates, data)
        # print(score)
        if score <= 0.5:
            TN += 1
        else:
            FP += 1
    P, R, F1, FNR, FPR = Metrics(TP, FP, TN, FN)
    print("TP, FP, TN, FN :")
    print(TP, FP, TN, FN)
    print("P, R, F1, FNR, FPR :")
    print(P, R, F1, FNR, FPR)

def test_XSS():
    """Single vulnerability detection test: XSS"""
    tokenizer = generate_tokenizer('data/train/bad_opcode_XSS.txt', 'data/train/good_opcode_XSS.txt')
    templates = generate_sequences(tokenizer, 'templates/templates_XSS.txt')
    bad_data = generate_sequences(tokenizer, 'data/test/bad_opcode_XSS.txt')
    good_data = generate_sequences(tokenizer, 'data/test/good_opcode_XSS.txt')
    model = load_model('model/bilstm_model_XSS.h5')
    TP = 0
    FN = 0
    for data in bad_data:
        score = model_predict(model, templates, data)
        if score > 0.6:
            TP += 1
        else:
            FN += 1
    print('------------------------------------------')
    TN = 0
    FP = 0
    for data in good_data:
        score = model_predict(model, templates, data)
        if score <= 0.6:
            TN += 1
        else:
            FP += 1
    P, R, F1, FNR, FPR = Metrics(TP, FP, TN, FN)
    print("TP, FP, TN, FN :")
    print(TP, FP, TN, FN)
    print("P, R, F1, FNR, FPR :")
    print(P, R, F1, FNR, FPR)

def test_MUL():
    """Multiple vulnerability detection test: SQL and XSS"""
    tokenizer = generate_tokenizer('data/train/bad_opcode_SQL.txt', 'data/train/good_opcode_SQL.txt',
        'data/train/bad_opcode_XSS.txt', 'data/train/good_opcode_XSS.txt')
    templates_SQL = generate_sequences(tokenizer, 'templates/templates_SQL.txt')
    templates_XSS = generate_sequences(tokenizer, 'templates/templates_XSS.txt')
    bad_data_SQL = generate_sequences(tokenizer, 'data/test/bad_opcode_SQL.txt')
    bad_data_XSS = generate_sequences(tokenizer, 'data/test/bad_opcode_XSS.txt')
    good_data = generate_sequences(tokenizer, 'data/test/good_opcode_MUL.txt')
    model = load_model('model/bilstm_model_MUL.h5')
    TP = 0
    FN = 0
    for data in bad_data_SQL:
        score = model_predict(model, templates_SQL, data)
        # print(score)
        if score > 0.5:
            TP += 1
        else:
            FN += 1
    print('------------------------------------------')
    for data in bad_data_XSS:
        score = model_predict(model, templates_XSS, data)
        # print(score)
        if score > 0.5:
            TP += 1
        else:
            FN += 1
    print('------------------------------------------')
    TN = 0
    FP = 0
    for data in good_data:
        score_SQL = model_predict(model, templates_SQL, data)
        score_XSS = model_predict(model, templates_XSS, data)
        # print(score_SQL, score_XSS)
        if score_SQL <= 0.5 and score_XSS <= 0.5:
            TN += 1
        else:
            FP += 1
    P, R, F1, FNR, FPR = Metrics(TP, FP, TN, FN)
    print("TP, FP, TN, FN :")
    print(TP, FP, TN, FN)
    print("P, R, F1, FNR, FPR :")
    print(P, R, F1, FNR, FPR)


if __name__ == '__main__':
    print("=========Predict SQL Vulnerabilities=========")
    test_SQL()
    print("=========Predict XSS Vulnerabilities=========")
    test_XSS()
    print("=========Predict MUL Vulnerabilities=========")
    test_MUL()