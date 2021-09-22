# _*_coding: utf-8-*_
# @Project : 基于用户画像的商品推荐挑战赛
# @FileNAme: config.py
# @Author  : Rocket,Qian
# @Time    : 2021/9/10 20:25
import argparse
path = r'E:\Competition\基于用户画像的商品推荐挑战赛\dataset'
parser = argparse.ArgumentParser(description="基于用户画像的商品推荐挑战赛")

# ========================= Dataset Configs ==========================
parser.add_argument('--train_file', type=str, default=path + r'\data2\train.txt')
parser.add_argument('--test_file', type=str, default=path + r'\data2\test.txt')

# ========================= Word2Vec Configs ==========================
parser.add_argument('--embed_size', type=int, default=64, help='embedding_size of every tagid')
parser.add_argument('--MAX_WORDS_NUM', type=int, default=224253, help='all word of fusai data')
parser.add_argument('--MAX_SEQUENCE_LENGTH', type=int, default=256)
parser.add_argument('--window', type=int, default=1)
parser.add_argument('--epochs', type=int, default=10, help='Num epochs of word2vec')

# ========================= Model Configs ==========================
parser.add_argument('--GRU1_hidden_size', type=int, default=128, help='GRU1 hidden_size ')
parser.add_argument('--GRU2_hidden_size', type=int, default=256, help='GRU2 hidden_size ')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio')
parser.add_argument('--nn_epochs', type=int, default=128, help='number of total epochs to train')
parser.add_argument('--batch_size', type=int, default=400, help='number of total epochs to train')


args = parser.parse_args()
print(args.dropout)
print(args.train_file)
print(args.batch_size)
