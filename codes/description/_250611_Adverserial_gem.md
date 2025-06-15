구현 요약
1. 전체 아키텍처 및 데이터 흐름
프레임워크: PyTorch Lightning을 사용하여 훈련, 검증, 테스트 파이프라인을 체계적으로 구성했습니다. 이는 코드의 재사용성과 가독성을 높여줍니다.
모델: GPT-2를 백본으로 사용하며, [Collaborator]에 정의된 GPT2Model을 임포트하여 활용했습니다. 입력 문장 쌍은 프롬프트 형식('Is "{s1}" a paraphrase of "{s2}"?')으로 변환된 후, GPT-2 인코더를 통과합니다. 인코더의 마지막 토큰 은닉 상태(hidden state) 위에 선형 분류 헤드를 추가하여 "패러프레이즈 여부"를 이진 분류합니다.
데이터 흐름:
quora-*.csv 파일에서 데이터를 로드합니다 (load_paraphrase_data).
ParaphraseDataModule이 데이터를 캡슐화하고, 각 단계(fit, test)에 맞는 DataLoader를 생성합니다.
DataLoader는 배치 단위로 토큰화된 데이터를 ParaphraseLitModuleAPT 모델에 전달합니다.
학습이 완료된 후, 가장 성능이 좋았던 체크포인트를 이용해 test 단계를 실행하고, 예측 결과를 predictions/ 폴더에 CSV 파일로 저장합니다.
2. 논문 방법론 적용 (Adversarial Paraphrasing Task 원칙)
핵심 원칙: 논문은 의미적으로는 동일하지만 어휘적/구문적으로는 매우 다른 "적대적" 예시(Adversarial Examples)를 통해 모델을 훈련시켜, 피상적인 단어 일치에 의존하지 않는 강건한(robust) 패러프레이즈 탐지기 개발을 제안합니다.
구현: 논문에서 제시한 커스텀 데이터셋(AP_H, AP_T5)을 직접 사용할 수 없으므로, 논문의 핵심 목표인 모델의 강건성 향상을 코드 레벨에서 구현하는 데 초점을 맞췄습니다.
이를 위해 [PreviousCode]에 구현된 R-Drop 기법을 채택했습니다. R-Drop은 동일한 입력에 대해 두 번의 드롭아웃(dropout)을 적용한 후, 두 출력 분포 간의 KL-Divergence를 최소화하는 정규화(regularization) 기법입니다.
이는 모델이 드롭아웃이라는 작은 변화에 일관된 예측을 하도록 강제하여, 특정 어휘나 구문에 과적합되는 것을 방지하고 더 일반화된 의미적 특징을 학습하도록 유도합니다. 이 접근 방식은 논문의 "어휘적 다양성에 강건한 모델"이라는 목표와 정확히 일치합니다.
손실 함수는 CrossEntropy + α * R-Drop(KL-Divergence)로 구성되며, simCSE 관련 로직은 요구사항에 따라 제거했습니다.
3. 로깅, 체크포인트, 및 측정 지표 전략
로깅:
TensorBoard: TensorBoardLogger를 사용하여 학습 및 검증 과정의 손실(loss), 정확도(accuracy), F1 점수, 학습률(learning rate) 등 주요 지표를 시각화하고 추적합니다. 로그는 outputs/logs/ 폴더에 시간 기반으로 생성된 고유 디렉토리에 저장됩니다.
텍스트 파일: [ReferenceCode]에서 영감을 받은 TextFileLogger 콜백을 구현하여, 터미널 출력과 유사한 학습 진행 상황을 단계별로 텍스트 파일에 기록합니다. 이를 통해 학습 과정을 상세히 복기할 수 있습니다.
체크포인트: ModelCheckpoint 콜백을 사용하여 검증 정확도(val/acc)가 가장 높은 모델의 가중치를 outputs/checkpoints/ 폴더에 자동으로 저장합니다. save_last=True 옵션으로 마지막 에포크의 모델도 함께 저장하여 학습 재개에 용이하도록 했습니다.
측정 지표: torchmetrics 라이브러리를 사용하여 Accuracy와 F1-Score를 효율적으로 계산하고, 각 단계(step) 및 에포크(epoch)마다 로깅하여 모델 성능을 다각도로 평가합니다.
4. 텐서 차원 관리
입력: GPT2Tokenizer는 문장 쌍을 프롬프트 형식의 단일 시퀀스로 변환하고, 패딩(padding)을 통해 배치 내 모든 시퀀스의 길이를 통일합니다. 입력 텐서 input_ids와 attention_mask의 차원은 (Batch_Size, Sequence_Length)입니다.
GPT-2 인코더: 인코더는 (Batch_Size, Sequence_Length, Hidden_Size) 차원의 은닉 상태 텐서를 출력합니다.
표현 추출: 어텐션 마스크를 사용하여 각 시퀀스의 마지막 실제 토큰 위치를 계산하고, 해당 위치의 은닉 상태를 추출합니다. 이 표현 벡터의 차원은 (Batch_Size, Hidden_Size)입니다.
분류 헤드: 이 표현 벡터는 (Hidden_Size, 2) 차원의 선형 레이어를 통과하여 최종 로짓(logits)을 생성하며, 이 로짓의 차원은 (Batch_Size, 2)가 됩니다. 모든 과정에서 텐서 차원이 원활하게 정렬되도록 파이프라인을 설계했습니다.