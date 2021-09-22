# _*_coding: utf-8-*_
# @Project : 基于用户画像的商品推荐挑战赛
# @FileNAme: all_code.py
# @Author  : Rocket,Qian
# @Time    : 2021/9/19 18:28
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import GRU
import tensorflow as tf
from gensim.models import Word2Vec
from config import parser
import os
import warnings

warnings.filterwarnings('ignore')

args = parser.parse_args()
# 读取数据，简单处理list数据
train_all = pd.read_csv(args.train_file, header=None)
test_all = pd.read_csv(args.test_file, header=None)
# train_first = pd.read_csv(r'E:\Competition\基于用户画像的商品推荐挑战赛\dataset\train.txt', header=None)
train_all.columns = ['pid', 'label', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'model', 'make']
test_all.columns = ['pid', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'model', 'make']
print('导入数据成功')
train = train_all[train_all['tagid'].notnull()]
test = test_all[test_all['tagid'].notnull()]

flag = 0
if flag == 1:
    train = pd.concat([train_first, train])
    # train.to_csv(r'E:\Competition\基于用户画像的商品推荐挑战赛\train_sum.csv', index=False)

train['label'] = train['label'].astype(int)

data = pd.concat([train, test])
data['label'] = data['label'].fillna(-1)
data['tagid'] = data['tagid'].apply(lambda x: eval(x))
data['tagid'] = data['tagid'].apply(lambda x: [str(i) for i in x])

embed_size = args.embed_size
MAX_WORDS_NUM = args.MAX_WORDS_NUM
MAX_SEQUENCE_LENGTH = args.MAX_SEQUENCE_LENGTH
w2v_model = Word2Vec(sentences=data['tagid'].tolist(), vector_size=embed_size, window=args.window, min_count=1,
                     epochs=args.epochs, hs=1)

X_train = data[:train.shape[0]]['tagid']
X_test = data[train.shape[0]:]['tagid']

tokenizer = text.Tokenizer(num_words=MAX_WORDS_NUM)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', truncating='pre')
X_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', truncating='pre')
word_index = tokenizer.word_index
nb_words = len(word_index) + 1

embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    try:
        embedding_vector = w2v_model.wv.get_vector(word)
    except KeyError:
        continue
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
y_cat = train['label'].values

# GPU设置
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# 定义模型
def my_model():
    embedding_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(nb_words,
                         embed_size,
                         input_length=MAX_SEQUENCE_LENGTH,
                         weights=[embedding_matrix],
                         trainable=False
                         )
    embed = embedder(embedding_input)
    l = GRU(args.GRU1_hidden_size, return_sequences=True)(embed)
    flat = BatchNormalization()(l)
    drop = Dropout(args.dropout)(flat)
    l2 = GRU(args.GRU2_hidden_size)(drop)
    output = Dense(1, activation='sigmoid')(l2)
    model = Model(inputs=embedding_input, outputs=output)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])
    return model


# 五折交叉验证
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
triain_pre = np.zeros([len(train), 1])
test_predictions = np.zeros([len(test), 1])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
    print("fold n{}".format(fold_ + 1))
    model = my_model()
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
    bst_model_path = "./{}.h5".format(fold_)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    X_tra, X_val = X_train[trn_idx], X_train[val_idx]
    y_tra, y_val = y_cat[trn_idx], y_cat[val_idx]

    model.fit(X_tra, y_tra,
              validation_data=(X_val, y_val),
              epochs=args.nn_epochs, batch_size=args.batch_size, shuffle=True,
              callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)
    triain_pre[val_idx] = model.predict(X_val)
    test_predictions += model.predict(X_test) / folds.n_splits
    del model


submit = test[['pid']]
submit['tmp'] = test_predictions
submit.columns = ['user_id', 'tmp']

submit['rank'] = submit['tmp'].rank()
submit['category_id'] = 1
submit.loc[submit['rank'] <= int(submit.shape[0] * 0.859), 'category_id'] = 0

submit_null = test_all[test_all['tagid'].isna()][['pid']]
submit_null['category_id'] = 1

submit_notnull = submit[['user_id', 'category_id']]
submit_notnull.columns = ['pid', 'category_id']

sub = pd.concat([submit_null, submit_notnull])
sub.sort_values(by='pid', ascending=True, inplace=True)
sub.to_csv(
    r'E:\Competition\基于用户画像的商品推荐挑战赛\result\0920GRUemd64b400win1_0859.csv',
    index=False)
