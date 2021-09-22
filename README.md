# 基于用户画像的商品推荐
## Usage
### Requirments

- Tensorflow-GPU 2.4.1
- CUDA 1.10

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