# 基于用户画像的商品推荐
## Usage
### Requirments
numpy==1.19.5  
tensorflow_gpu==2.4.1  
gensim==4.0.1  
pandas==0.25.3  
scikit_learn==0.24.2  

### 代码说明
#### 数据处理
- 根据tagid是否缺失把train和test(复赛数据)分出两部分数据集
- 将复赛数据集的train和test的tagid未缺失用户的tagid序列用来做Word2Vector

#### 模型说明
- 两层GRU
- 五折交叉验证

#### 结果输出
- test中tagid缺失的用户label直接预测为1
- 线下train_tagidNotnull_F1Score为0.6773461

## 另外：
- 硬件方面用的自己的一块3060显卡，在batch_size为512的时候，仅有3.6G显存；
- 就算这块显卡再不济，也比我的MacBook Pro计算速度快100倍，比免费版Google colab快30倍。
- 跑一次模型从30min~2h不等，跟embedding size、batch_size、hidden 层数有关。
