import os
import re
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 1. 文本预处理
def preprocess_text(text):
    # 去除标点符号、数字等非汉字字符
    text = re.sub(r'[^\u4e00-\u9fff]', '', text)
    return text

# 2. 均匀抽取段落
def extract_paragraphs(text, K):
    paragraphs = [text[i:i+K] for i in range(0, len(text), K)]
    return paragraphs[:1000]  # 只取前1000个段落

# 3. 读取语料库并处理
def load_corpus(directory, K):
    corpus = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='gb18030') as file:
                text = file.read()
                text = preprocess_text(text)
                paragraphs = extract_paragraphs(text, K)
                corpus.extend(paragraphs)
                labels.extend([filename] * len(paragraphs))
    return corpus, labels

# 4. LDA模型训练与主题分布表示
def train_lda(corpus, T, unit='word'):
    if unit == 'word':
        vectorizer = CountVectorizer(max_features=1000)
    else:
        vectorizer = CountVectorizer(analyzer='char', max_features=1000)
    X = vectorizer.fit_transform(corpus)
    lda = LatentDirichletAllocation(n_components=T, random_state=42)
    X_lda = lda.fit_transform(X)
    return X_lda

# 5. 分类模型训练与评估
def evaluate_classification(X, y, classifier='nb'):
    if classifier == 'nb':
        clf = MultinomialNB()
    elif classifier == 'svm':
        clf = SVC()
    elif classifier == 'rf':
        clf = RandomForestClassifier()
    scores = cross_val_score(clf, X, y, cv=10)
    return np.mean(scores)

# 6. 主函数
def main():
    directory = "D:\\课程\\大四\\大四下\\自然语言处理\\第二次作业\\jyxstxtqj_downcc.com"
    K_values = [20, 100, 500, 1000, 3000]
    T_values = [5, 10, 20, 50, 100]
    units = ['word', 'char']
    results = {}

    for K in K_values:
        corpus, labels = load_corpus(directory, K)
        for T in T_values:
            for unit in units:
                X_lda = train_lda(corpus, T, unit)
                accuracy = evaluate_classification(X_lda, labels, classifier='nb')
                results[(K, T, unit)] = accuracy
                print(f"K={K}, T={T}, unit={unit}, accuracy={accuracy}")

    # 输出结果
    for key, value in results.items():
        print(f"K={key[0]}, T={key[1]}, unit={key[2]}, accuracy={value}")

if __name__ == "__main__":
    main()