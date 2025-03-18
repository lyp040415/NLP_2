import os
import jieba
import re
import numpy as np
from gensim import corpora, models
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import pandas as pd
import warnings
from sklearn.exceptions import FitFailedWarning

# 忽略交叉验证的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

# 配置参数
K_VALUES = [20, 100, 500, 1000, 3000]  # 段落长度
T_VALUES = [5, 10, 20, 50, 100]             # 主题数量
UNITS = ['word', 'char']               # 基本单元
N_SAMPLES = 1000                       # 抽取样本数
N_FOLDS = 10                           # 交叉验证次数

# 加载停用词（需自行准备停用词文件）
with open('cn_stopwords.txt', 'r', encoding='utf-8') as f:
    STOPWORDS = set([line.strip() for line in f])

# 预处理函数
def preprocess(text, unit='word'):
    """文本预处理：清洗、分词/分字、去停用词"""
    # 去除非中文字符和标点
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    
    # 分词/分字处理
    if unit == 'word':
        tokens = jieba.lcut(text)
    elif unit == 'char':
        tokens = list(text)
    
    # 去除停用词
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens

def build_dataset(novel_dir, k=20, unit='word'):
    """构建数据集：从所有小说中抽取固定长度的段落"""
    all_paragraphs = []
    labels = []
    
    for novel_file in os.listdir(novel_dir):
        if not novel_file.endswith('.txt'):
            continue
            
        # 读取小说内容
        try:
            with open(os.path.join(novel_dir, novel_file), 'r', encoding='utf-8') as f:
                text = f.read().replace('\n', '')
        except UnicodeDecodeError:
            # 如果utf-8失败，尝试gb18030
            try:
                with open(os.path.join(novel_dir, novel_file), 'r', encoding='gb18030') as f:
                    text = f.read().replace('\n', '')
            except UnicodeDecodeError:
                # 如果仍然失败，忽略无法解码的字符
                with open(os.path.join(novel_dir, novel_file), 'r', encoding='gb18030', errors='ignore') as f:
                    text = f.read().replace('\n', '')
        
        # 预处理文本
        tokens = preprocess(text, unit=unit)
        
        # 分割为k长度的段落
        paragraphs = [tokens[i*k : (i+1)*k] for i in range(len(tokens)//k)]
        labels += [novel_file] * len(paragraphs)
        all_paragraphs += paragraphs
    
    # 随机抽取N_SAMPLES个样本
    combined = list(zip(all_paragraphs, labels))
    np.random.shuffle(combined)
    sampled = combined[:N_SAMPLES]
    paragraphs, labels = zip(*sampled)
    return paragraphs, labels

# 主实验流程
def run_experiment(novel_dir):
    results = []
    
    for unit in UNITS:
        print(f"\n===== 处理单元: {unit} =====")
        
        for k in K_VALUES:
            print(f"处理段落长度 K={k}...")
            
            # 1. 构建数据集
            paragraphs, labels = build_dataset(novel_dir, k=k, unit=unit)
            
            # 2. 创建词典和词袋
            dictionary = corpora.Dictionary(paragraphs)
            corpus = [dictionary.doc2bow(p) for p in paragraphs]
            
            for t in T_VALUES:
                print(f"正在处理 T={t}", end=' ', flush=True)
                
                # 3. 训练LDA模型
                lda = models.LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=t,
                    passes=10,
                    random_state=42
                )
                
                # 4. 获取主题分布特征
                X = np.zeros((len(corpus), t))
                for i, doc in enumerate(corpus):
                    topics = lda.get_document_topics(doc, minimum_probability=0)
                    X[i] = [prob for _, prob in topics]
                
                # 5. 交叉验证分类（静默模式）
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
                accuracies = []
                
                # 移除所有fold的打印语句
                for train_idx, test_idx in skf.split(X, labels):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
                    
                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    accuracies.append(accuracy_score(y_test, y_pred))
                
                # 记录结果
                results.append({
                    'Unit': unit,
                    'K': k,
                    'T': t,
                    'Accuracy': np.mean(accuracies),
                    'Std': np.std(accuracies)
                })
                print(f"=> Acc: {np.mean(accuracies):.4f}")  # 单行输出当前结果
    
    return pd.DataFrame(results)

# 运行实验
novel_dir = r"D:\课程\大四\大四下\自然语言处理\第二次作业\jyxstxtqj_downcc.com"
results_df = run_experiment(novel_dir)

# 结果展示
print("\n===== 最终实验结果 =====")
for _, row in results_df.sort_values(['Unit','K','T']).iterrows():
    print(
        f"基本单元:{row['Unit']}，K:{row['K']}，T:{row['T']}，"
        f"Acc:{row['Accuracy']:.4f}±{row['Std']:.4f}"
    )