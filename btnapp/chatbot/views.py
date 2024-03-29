from cProfile import label
import imp
import re
from urllib import response
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
import json
from visstep.models import StepCount_Data

# query generation model
import pandas as pd
from datetime import datetime
import datetime
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import konlpy
from konlpy.tag import *
import pickle

# kobert package
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

# GPU 사용
device = torch.device("cpu")

#BERT 모델, Vocabulary 불러오기 필수
bertmodel, vocab = get_pytorch_kobert_model()

# Create your views here.

NUM_WORDS = 500

@csrf_exempt
def chat(request):
    if request.method == 'POST':
        
        input1 = request.POST['input1']

        # 텍스트의 라벨 판별
        label = predict(input1)[-1]
        
        # 유저 입력이 어떠한 종류의 query인지 판별
        result = get_query(input1, label)

        date_1 = []
        date_2 = []
        stepcount_1 = []
        stepcount_2 = []
        legend_value = []
        answer = "걸음 수"
        if label == 'Compare':
            input1 = input1.replace(" ", "")
            if "이번주" in input1:
                legend_value = ['이번주', '저번주']
                answer = "이번주와 저번주 비교 걸음 수입니다."
                date_1 = ["월", "화", "수", "목", "금", "토", "일"]
                date_2 = ["월", "화", "수", "목", "금", "토", "일"]
                for i in StepCount_Data.objects.raw("select * from stepcountData where date BETWEEN date('now', '-7 days', 'weekday 1') and date('now')"):
                    stepcount_1.append(i.stepCount)
                for i in StepCount_Data.objects.raw("select * from stepcountData where date BETWEEN date('now', '-14 days', 'weekday 1') and date('now', '-7 days', 'weekday 0')"):
                    stepcount_2.append(i.stepCount)

            elif "이번달" in input1:
                legend_value = ['이번달', '저번달']
                answer = "이번달과 저번달 비교 걸음 수입니다."
                for i in StepCount_Data.objects.raw("select * from stepcountData where date BETWEEN date('now', 'start of month') and date('now')"):
                    date_1.append(str(int(str(i.date)[8:])))
                    stepcount_1.append(i.stepCount)
                for i in StepCount_Data.objects.raw("select * from stepcountData where date BETWEEN date('now', 'start of month', '-1 month') and date('now', 'start of month', '-1 days')"):
                    date_2.append(str(int(str(i.date)[8:])))
                    stepcount_2.append(i.stepCount)

            elif ("2021" and "2022") in input1 or ("올해" and "작년") in input1:
                if "월" in input1:
                    legend_value, answer, date_1, date_2, stepcount_1, stepcount_2 = compare_year_month(input1)
                else:
                    legend_value = ['2022년', '2021년']
                    answer = "올해와 작년 비교 걸음 수입니다."
                    date_1 = ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"]
                    date_2 = ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"]
                    stepcount_1, stepcount_2 = compare_year()
            else: 
                answer = "올해와 작년 비교 걸음 수입니다."
                date_1 = ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"]
                date_2 = ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"]
                stepcount_1, stepcount_2 = compare_year()

        elif label == 'Specify':            
            if "-14 days" in result:
                answer = "지난주 걸음 수입니다."
                label = "weeks"
            elif "start of month" in result:
                answer = "저번 달 걸음 수입니다"
                label = "month"
            else:
                result = specify_month(input1, result)
                answer = str(StepCount_Data.objects.raw(result)[0].date)[:4] + "년 " + str(int(str(StepCount_Data.objects.raw(result)[0].date)[5:7])) + "월 걸음 수입니다."
                label = "month"

            for i in StepCount_Data.objects.raw(result):
                date_1.append(str(int(str(i.date)[8:])))
                stepcount_1.append(i.stepCount)

        else:
            
            if "-6 days" in result:
                date_1, stepcount_1 = today_date(result)
                answer = "최근 1주일 걸음 수입니다."
                label = "weeks"
            elif "-13 days" in result:
                date_1, stepcount_1 = today_date(result)
                answer = "최근 2주일 걸음 수입니다."
                label = "weeks"
            elif "-20 days" in result:
                date_1, stepcount_1 = today_date(result)
                answer = "최근 3주일 걸음 수입니다."
                label = "month"
            elif "-27 days" in result:
                date_1, stepcount_1 = today_date(result)
                answer = "최근 4주일 걸음 수입니다."
                label = "month"
            elif "start of month" in result:
                date_1, stepcount_1 = today_date(result)
                answer = "이번 달 걸음 수입니다."
                label = "month"

            elif "-2 month" in result:
                answer, date_1, stepcount_1 = avg_weeks(result)
                label = "avg_weeks"
                answer = "최근 2개월 걸음 수입니다."
                
            elif "-3 month" in result:
                answer, date_1, stepcount_1 = avg_weeks(result)
                label = "avg_weeks"
                answer = "최근 3개월 걸음 수입니다."

            elif "-6 month" in result:
                answer, date_1, stepcount_1 = avg_months(result)
                label = "avg_months"
                answer = "최근 6개월 걸음 수입니다."

            else:
                answer, date_1, stepcount_1 = avg_months(result)
                label = "avg_months"
                answer = "최근 12개월 걸음 수입니다."
            

        output = dict()
        output['response'] = answer
        output['date_1'] = date_1
        output['date_2'] = date_2
        output['stepcount_1'] = stepcount_1
        output['stepcount_2'] = stepcount_2
        output['label'] = label 
        output['legend_value'] = legend_value
        return HttpResponse(json.dumps(output), status=200)
    else:
        return render(request, 'chatbot/chat.html')

def today_date(result):
    date_1 = []
    stepcount_1 = []
    for i in StepCount_Data.objects.raw(result):
        date_1.append(str(int(str(i.date)[8:])))
        stepcount_1.append(i.stepCount)
    return date_1, stepcount_1

def avg_weeks(result):
    date_1 = []
    stepcount_1 = []
    answer = "걸음 수"
    tmp_date = []
    tmp_stepcount = 0
    tmp = 0
    last_data = StepCount_Data.objects.raw(result)[-1].date

    for i in StepCount_Data.objects.raw(result):
        if i.date.weekday() != 0:
            tmp_date.append(str(int(str(i.date)[5:7])) + "월 " + str(int(str(i.date)[8:])) + "일")
            tmp_stepcount = tmp_stepcount + int(i.stepCount)
            tmp = tmp + 1
            if i.date == last_data:
                date_1.append(tmp_date[0])
                stepcount_1.append(tmp_stepcount/tmp)
        else:
            if tmp_stepcount != 0:
                date_1.append(tmp_date[0])
                stepcount_1.append(tmp_stepcount/tmp)
                tmp_date = []
                tmp_stepcount = 0
                tmp = 0
                tmp_date.append(str(int(str(i.date)[5:7])) + "월 " + str(int(str(i.date)[8:])) + "일")
                tmp_stepcount = tmp_stepcount + int(i.stepCount)
                tmp = tmp + 1
            else:
                date_1.append(str(int(str(i.date)[5:7])) + "월 " + str(int(str(i.date)[8:])) + "일")
                stepcount_1.append(i.stepCount)

    return answer, date_1, stepcount_1

def avg_months(result):
    date_1 = []
    stepcount_1 = []
    answer = "걸음 수"
    tmp_stepcount = 0
    tmp = 0
    last_data = StepCount_Data.objects.raw(result)[-1].date

    for i in StepCount_Data.objects.raw(result):
        if i.date.day != 1:
            if len(date_1) == 0:
                date_1.append(str(i.date)[0:4] + "년 " + str(int(str(i.date)[5:7])) + "월")
            elif date_1[-1] != str(i.date)[0:4] + "년 " + str(int(str(i.date)[5:7])) + "월":
                date_1.append(str(i.date)[0:4] + "년 " + str(int(str(i.date)[5:7])) + "월")
            tmp_stepcount = tmp_stepcount + int(i.stepCount)
            tmp = tmp + 1
            if i.date == last_data:
                stepcount_1.append(tmp_stepcount/tmp)

        else:
            if tmp_stepcount != 0:
                stepcount_1.append(tmp_stepcount/tmp)
                tmp_stepcount = 0
                tmp = 0
                tmp_stepcount = tmp_stepcount + int(i.stepCount)
                tmp = tmp + 1
            else:
                date_1.append(str(i.date)[0:4] + "년 " + str(int(str(i.date)[5:7])) + "월")
                stepcount_1.append(i.stepCount)

    return answer, date_1, stepcount_1

def specify_month(input1, result):
    input1 = input1.replace(" ", "")
    if "2021" in input1 or "작년" in input1:
        result = result
    elif "2022" in input1:
        month_num = re.findall(r'\d+', input1[input1.find("월") - 2 : input1.find("월")])[0]
        if int(month_num) >= 6:
            if int(month_num) < 10:
                month_num = "0" + month_num
            if int(datetime.date.today().month) == month_num:
                result = "select * from stepcountData where date BETWEEN '2022-" + month_num + "-01' and date('now')"
            else:
                result = "select * from stepcountData where date BETWEEN '2022-" + month_num + "-01' and date('2022-" + month_num + "-01', '+1 month', '-1 days')"
        else:
            result = result

    elif "올해" in input1:
        result = str(result).replace("2021", "2022")
        if int(datetime.date.today().month) == int(result[result.find('2022') + 5: result.find('2022') + 7]):
            result = "select * from stepcountData where date between date('now','start of month') and date('now') ORDER BY (date) ASC"
    else:
        query_month = int(result[result.find('2021') + 5: result.find('2021') + 7])
        if int(datetime.date.today().month) < query_month:
            result = result
        elif int(datetime.date.today().month) == query_month:
            result = "select * from stepcountData where date between date('now','start of month') and date('now') ORDER BY (date) ASC"
        else:
            result = str(result).replace("2021", "2022")

    return result

def compare_year_month(input1):
    date_1 = []
    date_2 = []
    stepcount_1 = []
    stepcount_2 = []
    legend_value = []

    month_num = re.findall(r'\d+', input1[input1.find("월") - 2 : input1.find("월")])[0]
    legend_value = ['2022년 ' + month_num  + '월', '2021년 ' + month_num  + '월']
    answer = "올해와 작년 " + month_num + "월 비교 걸음 수입니다."
    if len(month_num) == 1:
        month_num = "0" + month_num
    for i in StepCount_Data.objects.raw("select * from stepcountData where date BETWEEN '2021-" + month_num + "-01' and date('2021-" + month_num + "-01', '+1 month', '-1 days')"):
        date_2.append(str(int(str(i.date)[8:])))
        stepcount_2.append(i.stepCount)
    if int(datetime.date.today().month) == int(month_num):
        for i in StepCount_Data.objects.raw("select * from stepcountData where date BETWEEN '2022-" + month_num + "-01' and date('now')"):
            date_1.append(str(int(str(i.date)[8:])))
            stepcount_1.append(i.stepCount)
    else:
        for i in StepCount_Data.objects.raw("select * from stepcountData where date BETWEEN '2022-" + month_num + "-01' and date('2022-" + month_num + "-01', '+1 month', '-1 days')"):
            date_1.append(str(int(str(i.date)[8:])))
            stepcount_1.append(i.stepCount)


    return legend_value, answer, date_1, date_2, stepcount_1, stepcount_2

def compare_year():
    stepcount_1 = []
    stepcount_2 = []

    tmp_month = 0
    tmp_stepcount = 0
    tmp_day = 0
    for i in StepCount_Data.objects.raw("select * from stepcountData where date BETWEEN '2022-01-01' and date('now')"):
        if tmp_month == 0:
            tmp_month = i.date.month
        if tmp_month == i.date.month:
            tmp_stepcount = tmp_stepcount + i.stepCount
            tmp_day = i.date.day
            if str(i.date) == str(datetime.date.today()):
                stepcount_1.append(tmp_stepcount/tmp_day)
                tmp_month = 0
                tmp_stepcount = 0
                tmp_day = 0
        else:
            tmp_month = i.date.month
            stepcount_1.append(tmp_stepcount/tmp_day)
            tmp_stepcount = i.stepCount
        
    for i in StepCount_Data.objects.raw("select * from stepcountData where date BETWEEN '2021-01-01' and date('2021-01-01', '+12 month', '-1 days')"):
        if tmp_month == 0:
            tmp_month = i.date.month
        if tmp_month == i.date.month:
            tmp_stepcount = tmp_stepcount + i.stepCount
            tmp_day = i.date.day
            if str(i.date) == str(datetime.date(2021, 12, 31)):
                stepcount_2.append(tmp_stepcount/tmp_day)
        else:
            tmp_month = i.date.month
            stepcount_2.append(tmp_stepcount/tmp_day)
            tmp_stepcount = i.stepCount
    return stepcount_1, stepcount_2

# kobert model
def new_softmax(a) : 
    c = np.max(a) # 최댓값
    exp_a = np.exp(a-c) # 각각의 원소에 최댓값을 뺀 값에 exp를 취한다. (이를 통해 overflow 방지)
    sum_exp_a = np.sum(exp_a)
    y = (exp_a / sum_exp_a) * 100
    return np.round(y, 3)

def predict(predict_sentence: str):

    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    model.load_state_dict(torch.load('./../../classification_model/classification_model_state_dict.pt',  map_location=device))

    max_len = 64
    batch_size = 64
    
    # 토큰화
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)
    
    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)
        
        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()
            min_v = min(logits)
            total = 0
            probability = []
            logits = np.round(new_softmax(logits), 3).tolist()
            for logit in logits:
                # print(logit)
                probability.append(np.round(logit, 3))

            if np.argmax(logits) == 0:  emotion = "Today"
            elif np.argmax(logits) == 1: emotion = "Specify"
            elif np.argmax(logits) == 2: emotion = 'Compare'
            

            probability.append(emotion)
            # print(probability)
    return probability

# kobert class
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=3,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

# class QueryResult:
#     label: str
#     query: str
#     contains_comparison: bool = False

def get_query(user_input1: str, label):
    tf.enable_eager_execution()

    max_len = 40
    vocab_size = 515
    tokenizer = Tokenizer() 

    # result = QueryResult()
    # result.label = label

    if label == 'Today':
        with open('./static/today_model/today_tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        model = Seq2seq(sos=tokenizer.word_index['\t'], eos=tokenizer.word_index['\n'])
        model.load_weights("./static/today_model/text_to_sql_today")
    elif label == 'Specify':
        with open('./static/specify_model/specify_tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        model = Seq2seq(sos=tokenizer.word_index['\t'], eos=tokenizer.word_index['\n'])
        model.load_weights("./static/specify_model/text_to_sql_specify")
    else:
        with open('./static/compare_model/compare_tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        model = Seq2seq(sos=tokenizer.word_index['\t'], eos=tokenizer.word_index['\n'])
        model.load_weights("./static/compare_model/text_to_sql_compare")

    @tf.function
    def test_step(model, inputs):
        return model(inputs, training=False)
    
    okt = Okt()
    
    tmp_seq = [" ".join(okt.morphs(user_input1))]
    # print("tmp_seq : ", tmp_seq)

    test_data = list()
    test_data = tokenizer.texts_to_sequences(tmp_seq)
    # print("tokenized data : ", test_data)

    prd_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,value=0,padding='pre',maxlen=128)

    prd_data = tf.data.Dataset.from_tensor_slices(prd_data).batch(1).prefetch(1024)

    for seq in prd_data :
        prediction = test_step(model, seq)
        predicted_seq = tokenizer.sequences_to_texts(prediction.numpy())
        # print(predicted_seq)
        # print("predict tokens : ", prediction.numpy())

    predicted_seq = str(predicted_seq[0])
    # predicted_seq = str(predicted_seq[0]).replace(" _ ", "_")
    predicted_seq = predicted_seq.replace("e (", "e(")
    predicted_seq = predicted_seq.replace("' ", "'")
    predicted_seq = predicted_seq.replace(" '", "'")
    predicted_seq = predicted_seq.replace(" - ", "-")
    predicted_seq = predicted_seq.replace("+ ", "+")
    predicted_seq = predicted_seq.replace("- ", "-")
    # print(predicted_seq)

    result = "select * from stepcountData where date " + predicted_seq + " ORDER BY (date) ASC"
    return result

class Encoder(tf.keras.Model):
  def __init__(self):
    super(Encoder, self).__init__()
    # 1000개의 단어들을 128크기의 vector로 Embedding해줌.
    self.emb = tf.keras.layers.Embedding(NUM_WORDS, 128)
    # return_state는 return하는 Output에 최근의 state를 더해주느냐에 대한 옵션
    # 즉, Hidden state와 Cell state를 출력해주기 위한 옵션이라고 볼 수 있다.
    # default는 False이므로 주의하자!
    # return_sequence=True로하는 이유는 Attention mechanism을 사용할 때 우리가 key와 value는
    # Encoder에서 나오는 Hidden state 부분을 사용했어야 했다. 그러므로 모든 Hidden State를 사용하기 위해 바꿔준다.
    self.lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)

  def call(self, x, training=False, mask=None):
    x = self.emb(x)
    H, h, c = self.lstm(x)
    return H, h, c


class Decoder(tf.keras.Model):
  def __init__(self):
    super(Decoder, self).__init__()
    self.emb = tf.keras.layers.Embedding(NUM_WORDS, 128)
    # return_sequence는 return 할 Output을 full sequence 또는 Sequence의 마지막에서 출력할지를 결정하는 옵션
    # False는 마지막에만 출력, True는 모든 곳에서의 출력
    self.lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)
    # LSTM 출력에다가 Attention value를 dense에 넘겨주는 것이 Attention mechanism이므로
    self.att = tf.keras.layers.Attention()
    self.dense = tf.keras.layers.Dense(NUM_WORDS, activation='softmax')

  def call(self, inputs, training=False, mask=None):
    # x : shifted output, s0 : Decoder단의 처음들어오는 Hidden state
    # c0 : Decoder단의 처음들어오는 cell state H: Encoder단의 Hidden state(Key와 value로 사용)
    x, s0, c0, H = inputs
    x = self.emb(x)

    # initial_state는 셀의 첫 번째 호출로 전달 될 초기 상태 텐서 목록을 의미
    # 이전의 Encoder에서 만들어진 Hidden state와 Cell state를 입력으로 받아야 하므로
    # S : Hidden state를 전부다 모아놓은 것이 될 것이다.(Query로 사용)
    S, h, c = self.lstm(x, initial_state=[s0, c0])

    # Query로 사용할 때는 하나 앞선 시점을 사용해줘야 하므로
    # s0가 제일 앞에 입력으로 들어가는데 현재 Encoder 부분에서의 출력이 batch 크기에 따라서 length가 현재 1이기 때문에 2차원형태로 들어오게 된다.
    # 그러므로 이제 3차원 형태로 확장해 주기 위해서 newaxis를 넣어준다.
    # 또한 decoder의 S(Hidden state) 중에 마지막은 예측할 다음이 없으므로 배제해준다.
    S_ = tf.concat([s0[:, tf.newaxis, :], S[:, :-1, :]], axis=1)

    # Attention 적용
    # 아래 []안에는 원래 Query, Key와 value 순으로 입력해야하는데 아래처럼 두가지만 입력한다면
    # 마지막 것을 Key와 value로 사용한다.
    A = self.att([S_, H])

    y = tf.concat([S, A], axis=-1)
    return self.dense(y), h, c


class Seq2seq(tf.keras.Model):
  def __init__(self, sos, eos):
    super(Seq2seq, self).__init__()
    self.enc = Encoder()
    self.dec = Decoder()
    self.sos = sos
    self.eos = eos

  def call(self, inputs, training=False, mask=None):
    if training is True:
      # 학습을 하기 위해서는 우리가 입력과 출력 두가지를 다 알고 있어야 한다.
      # 출력이 필요한 이유는 Decoder단의 입력으로 shited_ouput을 넣어주게 되어있기 때문이다.
      x, y = inputs

      # LSTM으로 구현되었기 때문에 Hidden State와 Cell State를 출력으로 내준다.
      H, h, c = self.enc(x)

      # Hidden state와 cell state, shifted output을 초기값으로 입력 받고
      # 출력으로 나오는 y는 Decoder의 결과이기 때문에 전체 문장이 될 것이다.
      y, _, _ = self.dec((y, h, c, H))
      return y

    else:
      x = inputs
      H, h, c = self.enc(x)

      # Decoder 단에 제일 먼저 sos를 넣어주게끔 tensor화시키고
      y = tf.convert_to_tensor(self.sos)
      # shape을 맞춰주기 위한 작업이다.
      y = tf.reshape(y, (1, 1))

      # 최대 64길이 까지 출력으로 받을 것이다.
      seq = tf.TensorArray(tf.int32, 128)

      # tf.keras.Model에 의해서 call 함수는 auto graph모델로 변환이 되게 되는데,
      # 이때, tf.range를 사용해 for문이나 while문을 작성시 내부적으로 tf 함수로 되어있다면
      # 그 for문과 while문이 굉장히 효율적으로 된다.
      for idx in tf.range(128):
        y, h, c = self.dec([y, h, c, H])
        # 아래 두가지 작업은 test data를 예측하므로 처음 예측한값을 다시 다음 step의 입력으로 넣어주어야하기에 해야하는 작업이다.
        # 위의 출력으로 나온 y는 softmax를 지나서 나온 값이므로
        # 가장 높은 값의 index값을 tf.int32로 형변환해주고
        # 위에서 만들어 놓았던 TensorArray에 idx에 y를 추가해준다.
        y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.int32)
        # 위의 값을 그대로 넣어주게 되면 Dimension이 하나밖에 없어서
        # 실제로 네트워크를 사용할 때 Batch를 고려해서 사용해야 하기 때문에 (1,1)으로 설정해 준다.
        y = tf.reshape(y, (1, 1))
        seq = seq.write(idx, y)

        if y == self.eos:
          break
      # stack은 그동안 TensorArray로 받은 값을 쌓아주는 작업을 한다.    
      return tf.reshape(seq.stack(), (1, 128))