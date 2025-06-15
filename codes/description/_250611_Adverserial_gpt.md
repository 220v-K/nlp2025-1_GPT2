ParaphraseDataModule
Quora CSV → ParaphraseDetectionDataset → DataLoader.

LightningParaphrase

GPT-2 encoder + Linear head (2-way).

training_step에서 R-Drop(두 forward → KL)과 APT 가중 교차엔트로피를 함께 사용.

lexical similarity는 배치 내부 토큰 집합의 Jaccard로 계산하여 가중치 결정.

validation/test는 CE만으로 평가하며 Accuracy·F1·AUROC 로깅.

논문 방법 적용 위치

논문의 핵심인 Adversarial Paraphrasing Task 아이디어를 손실 가중치로 구현:
‣ Paraphrase(1)이면서 lex-sim 낮음, 또는 Non-paraphrase(0)이면서 lex-sim 높음 ⇒ 가중치 증가.
이를 통해 모델이 어휘 중복에 의존하지 않고 의미적 추론을 학습하게 함.

로그·체크포인트 전략

TensorBoardLogger: step-별 loss·acc·f1·auc.

TextFileLogger(ReferenceCode 방식 차용): 50 step마다 loss/메모리/학습률을 txt 기록.

ModelCheckpoint: modelsize_타임스탬프_epoch-val_acc.pt 형식, 상위 3개 + 마지막 저장.

차원 관리 고려

GPT-2 last_hidden_state → (B, L, D) → 마지막 non-pad 토큰 인덱스로 (B, D) 추출 후 head.

R-Drop 대비 forward 출력 크기 일치 확인.

Jaccard 계산 시 padding ID 제외하여 여집합 오류 방지.