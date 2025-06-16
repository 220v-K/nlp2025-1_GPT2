#!/usr/bin/env python3

'''
SST 과 CFIMDB 위에서 GPT2SentimentClassifier를 훈련하고 평가.
'''

import random, numpy as np, argparse
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.metrics import f1_score, accuracy_score

from models.gpt2 import GPT2Model
from optimizer import AdamW
from tqdm import tqdm

TQDM_DISABLE = False


# 필요한 모든 random seed 설정.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


# GPU 사용 가능 여부와 개수 확인
def check_gpu():
  if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    print(f"GPU 사용 가능, 총 {n_gpu}개의 GPU가 감지되었습니다.")
    for i in range(n_gpu):
      print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    return True, n_gpu
  else:
    print("GPU를 사용할 수 없습니다. CPU를 사용합니다.")
    return False, 0


class GPT2SentimentClassifier(torch.nn.Module):
  '''
  이 모듈은 GPT-2를 사용하여 클로즈 스타일(빈칸 채우기) 작업으로 감정 분류를 수행한다.

  SST 데이터셋의 감정 범주 = 5 가지(0 - "부정"에서 4 - "긍정"까지).
  따라서, forward() 함수는 5개의 클래스 각각에 대해 하나의 로짓(logit)을 반환해야 한다.
  '''

  def __init__(self, config):
    super(GPT2SentimentClassifier, self).__init__()
    self.num_labels = config.num_labels
    self.gpt = GPT2Model.from_pretrained()

    # 사전학습 모드에서는 GPT 파라미터들을 업데이트할 필요가가 없다.
    assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
    for param in self.gpt.parameters():
      if config.fine_tune_mode == 'last-linear-layer':
        param.requires_grad = False
      elif config.fine_tune_mode == 'full-model':
        param.requires_grad = True

    '''
    TODO: BERT 임베딩의 감정 분류를 위해 필요한 인스턴스 변수를 생성하시오.
    '''
    ##----- 새로 작성한 코드 -----
    # 감정 분류를 위한 선형 레이어 (분류 헤드)
    # 입력 크기는 GPT 모델의 hidden_size, 출력 크기는 라벨의 수(num_labels)
    # Dropout + 분류 헤드
    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
    self.classification_head = torch.nn.Linear(config.hidden_size, self.num_labels)
    ##-------------------------


  def forward(self, input_ids, attention_mask):
    '''문장들의 batch를 받아서 감정 클래스에 대한 로짓을 반환'''

    '''
    TODO: 최종 GPT contextualized embedding은 마지막 토큰의 hidden state이다.
        힌트: 현재 훈련 반복루프에서 손실 함수로 `F.cross_entropy`를 사용하고 있음을 고려하여
        적절한 반환값이 무엇인지 생각해보시오.
    '''
    ##----- 새로 작성한 코드 -----
    gpt_output = self.gpt(input_ids=input_ids, attention_mask=attention_mask) # GPT Model을 통한 output
    x = gpt_output['last_token']           # [batch, hidden_size] (GPT output의 last token's hidden state 사용)
    x = self.dropout(x)                    # Dropout 적용
    logits = self.classification_head(x)   # [batch, num_labels]  (classifier head 적용, logit output.)
    
    return logits
    ##-------------------------


class SentimentDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [x[0] for x in data]
    labels = [x[1] for x in data]
    sent_ids = [x[2] for x in data]

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])
    labels = torch.LongTensor(labels)

    return token_ids, attention_mask, labels, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data


class SentimentTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [x[0] for x in data]
    sent_ids = [x[1] for x in data]

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    return token_ids, attention_mask, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data


# 데이터 로드: (sentence, label)의 리스트.
def load_data(filename, flag='train'):
  num_labels = {}
  data = []
  if flag == 'test':
    with open(filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].lower().strip()
        sent_id = record['id'].lower().strip()
        data.append((sent, sent_id))
  else:
    with open(filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].lower().strip()
        sent_id = record['id'].lower().strip()
        label = int(record['sentiment'].strip())
        if label not in num_labels:
          num_labels[label] = len(num_labels)
        data.append((sent, label, sent_id))
    print(f"load {len(data)} data from {filename}")

  if flag == 'train':
    return data, len(num_labels)
  else:
    return data


# dev 사례들로 모델을 평가한다.
def model_eval(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true = []
  y_pred = []
  sents = []
  sent_ids = []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                                   batch['labels'], batch['sents'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    b_labels = b_labels.flatten()
    y_true.extend(b_labels)
    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)

  return acc, f1, y_pred, y_true, sents, sent_ids


# test 사례들로 모델을 평가한다.
def model_test_eval(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_pred = []
  sents = []
  sent_ids = []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                         batch['sents'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  return y_pred, sents, sent_ids


def save_model(model, optimizer, args, config, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'model_config': config,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  use_gpu, n_gpu = check_gpu()
  device = torch.device('cuda') if use_gpu and args.use_gpu else torch.device('cpu')
  
  # 데이터와 해당 데이터셋 및 데이터로더를 만든다.
  train_data, num_labels = load_data(args.train, 'train')
  dev_data = load_data(args.dev, 'valid')

  train_dataset = SentimentDataset(train_data, args)
  dev_dataset = SentimentDataset(dev_data, args)

  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                collate_fn=train_dataset.collate_fn)
  dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                              collate_fn=dev_dataset.collate_fn)

  # Init model.
  config = {'hidden_dropout_prob': args.hidden_dropout_prob,
            'num_labels': num_labels,
            'hidden_size': 768,
            'data_dir': '.',
            'fine_tune_mode': args.fine_tune_mode}

  config = SimpleNamespace(**config)

  model = GPT2SentimentClassifier(config)
  
  # 여러 GPU 사용 설정
  if use_gpu and args.use_gpu and n_gpu > 1:
    print(f"{n_gpu}개의 GPU를 병렬로 사용합니다.")
    model = torch.nn.DataParallel(model)
  
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)
  best_dev_acc = 0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask, b_labels = (batch['token_ids'],
                                 batch['attention_mask'], batch['labels'])

      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      b_labels = b_labels.to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / (num_batches)

    train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
    dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, config, args.filepath)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def test(args):
  with torch.no_grad():
    use_gpu, n_gpu = check_gpu()
    device = torch.device('cuda') if use_gpu and args.use_gpu else torch.device('cpu')
    
    saved = torch.load(args.filepath)
    config = saved['model_config']
    model = GPT2SentimentClassifier(config)
    
    # 여러 GPU 사용 설정
    if use_gpu and args.use_gpu and n_gpu > 1:
      print(f"{n_gpu}개의 GPU를 병렬로 사용합니다.")
      model = torch.nn.DataParallel(model)
    
    model.load_state_dict(saved['model'])
    model = model.to(device)
    print(f"load model from {args.filepath}")

    dev_data = load_data(args.dev, 'valid')
    dev_dataset = SentimentDataset(dev_data, args)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    test_data = load_data(args.test, 'test')
    test_dataset = SentimentTestDataset(test_data, args)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_dataloader, model, device)
    print('DONE DEV')

    test_pred, test_sents, test_sent_ids = model_test_eval(test_dataloader, model, device)
    print('DONE Test')

    # 평가 결과를 txt 파일에 저장
    eval_results_file = f"predictions/{args.fine_tune_mode}-{args.dataset_name}-eval-results.txt"
    with open(eval_results_file, "w+") as f:
      f.write(f"===== {args.dataset_name} Evaluation Results =====\n")
      f.write(f"Fine-tune mode: {args.fine_tune_mode}\n")
      f.write(f"Dev Accuracy: {dev_acc:.4f}\n")
      f.write(f"Dev F1 Score: {dev_f1:.4f}\n")
      f.write(f"=========================================\n\n")
    
    print(f"평가 결과가 {eval_results_file}에 저장되었습니다.")

    with open(args.dev_out, "w+") as f:
      print(f"dev acc :: {dev_acc :.3f}")
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(dev_sent_ids, dev_pred):
        f.write(f"{p}, {s} \n")

    with open(args.test_out, "w+") as f:
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(test_sent_ids, test_pred):
        f.write(f"{p}, {s} \n")


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--fine-tune-mode", type=str,
                      help='last-linear-layer: the GPT parameters are frozen and the task specific head parameters are updated; full-model: GPT parameters are updated as well',
                      choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
  parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                      default=1e-3)

  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = get_args()
  seed_everything(args.seed)

  # 먼저 결과 디렉토리가 있는지 확인하고, 없으면 생성합니다
  import os
  if not os.path.exists('predictions'):
    os.makedirs('predictions')
    print("predictions 디렉토리를 생성했습니다.")
  
  # 모든 실험 결과를 요약할 파일
  summary_file = f"predictions/all_results_summary.txt"
  with open(summary_file, "w+") as f:
    f.write("===== 모든 데이터셋 실험 결과 =====\n")
    f.write(f"Fine-tune mode: {args.fine_tune_mode}\n\n")

  print('Training Sentiment Classifier on SST...')
  config = SimpleNamespace(
    filepath='sst-classifier.pt',
    lr=args.lr,
    use_gpu=args.use_gpu,
    epochs=args.epochs,
    batch_size=args.batch_size,
    hidden_dropout_prob=args.hidden_dropout_prob,
    train='data/ids-sst-train.csv',
    dev='data/ids-sst-dev.csv',
    test='data/ids-sst-test-student.csv',
    fine_tune_mode=args.fine_tune_mode,
    dev_out='predictions/' + args.fine_tune_mode + '-sst-dev-out.csv',
    test_out='predictions/' + args.fine_tune_mode + '-sst-test-out.csv',
    dataset_name='sst'
  )

  train(config)

  print('Evaluating on SST...')
  test(config)

  print('Training Sentiment Classifier on cfimdb...')
  config = SimpleNamespace(
    filepath='cfimdb-classifier.pt',
    lr=args.lr,
    use_gpu=args.use_gpu,
    epochs=args.epochs,
    batch_size=8,
    hidden_dropout_prob=args.hidden_dropout_prob,
    train='data/ids-cfimdb-train.csv',
    dev='data/ids-cfimdb-dev.csv',
    test='data/ids-cfimdb-test-student.csv',
    fine_tune_mode=args.fine_tune_mode,
    dev_out='predictions/' + args.fine_tune_mode + '-cfimdb-dev-out.csv',
    test_out='predictions/' + args.fine_tune_mode + '-cfimdb-test-out.csv',
    dataset_name='cfimdb'
  )

  train(config)

  print('Evaluating on cfimdb...')
  test(config)
  
  # 모든 실험이 끝난 후 각 데이터셋의 결과를 요약 파일에 추가
  with open(summary_file, "a") as f:
    # SST 결과 추가
    sst_result_file = f"predictions/{args.fine_tune_mode}-sst-eval-results.txt"
    if os.path.exists(sst_result_file):
      with open(sst_result_file, "r") as sst_f:
        f.write("SST 결과:\n")
        f.write(sst_f.read())
        f.write("\n\n")
    
    # CFIMDB 결과 추가
    cfimdb_result_file = f"predictions/{args.fine_tune_mode}-cfimdb-eval-results.txt"
    if os.path.exists(cfimdb_result_file):
      with open(cfimdb_result_file, "r") as cfimdb_f:
        f.write("CFIMDB 결과:\n")
        f.write(cfimdb_f.read())
        f.write("\n\n")
  
  print(f"모든 실험 결과가 {summary_file}에 요약되었습니다.")
