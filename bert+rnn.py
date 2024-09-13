import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 初始化 BERT tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = TFBertModel.from_pretrained('bert-base-chinese')

# 读取数据
def load_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ', 1)  # 分割文本和标签
            if len(parts) == 2:
                text, label = parts
                texts.append(text)
                labels.append(int(label))  # 标签转换为整数
    return texts, labels

# 文件路径
file_path = 'data.txt'
texts, labels = load_data(file_path)

def preprocess_texts(texts, tokenizer, max_length=64):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='np')
    return encodings

max_length = 64
encodings = preprocess_texts(texts, tokenizer, max_length)
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

# 将数据拆分为训练集和测试集
X_train_ids, X_test_ids, y_train, y_test = train_test_split(input_ids, labels, test_size=0.2, random_state=42)
X_train_mask, X_test_mask = train_test_split(attention_mask, test_size=0.2, random_state=42)

# 2. 构建模型
class BertLSTMClassifier(tf.keras.Model):
    def __init__(self, bert_model, num_classes, lstm_units=64):
        super(BertLSTMClassifier, self).__init__()
        self.bert = bert_model
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=False))
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense = tf.keras.layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, training=training)
        sequence_output = output.last_hidden_state  # Extract last hidden state
        x = self.lstm(sequence_output)
        x = self.dropout(x, training=training)
        x = self.dense(x)
        return x

# 创建模型实例
model = BertLSTMClassifier(bert_model, num_classes=1)

# 3. 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. 训练模型
history = model.fit(
    (np.array(X_train_ids), np.array(X_train_mask)),
    np.array(y_train),
    epochs=3,
    batch_size=2,
    validation_split=0.2
)

# 5. 评估模型
loss, accuracy = model.evaluate((X_test_ids, X_test_mask), np.array(y_test))
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

# 6. 预测并生成分类报告
y_pred = model.predict((X_test_ids, X_test_mask))
y_pred_classes = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_classes))
