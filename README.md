# 2025-1학기 자연어처리개론 기말 팀프로젝트



## 가상환경 재현

`environment.yml` 또는 `requirements.txt` 를 통해 가상환경 재현.
(conda 가상환경 생성 후 pip install 추천드립니다.)

```bash
$ pip install -r requirements.txt
or
$ conda env create -f environment.yml
```



## 폴더 구조 설명

```
└─ nlp2025-1
   ├─ [lot of codes]
   ├─ predictions/
   ├─ data/							[Dataset]
   ├─ models/
   └─ modules/
└─ paraphrase_detection
   ├─ outputs/
   │  ├─ logs_txt(250522-23)/	[text 파일로 정리한 학습 log]
   │  ├─ logs_txt/				[text 파일로 정리한 학습 log (0523 이후)]
   │  ├─ logs/						[tensorboard log 파일]
   │  └─ checkpoints/
   ├─ codes/						[Paraphrase Detection 관련 code]
   ├─ data/							[Dataset (BackTranslation Augmented data 포함)]
   └─ checkpoints/ 				[github에서 제거]
```

**PART-I**의 과제는 `nlp2025-1` 폴더에 완성되어 있음.



`nlp2025-1` 폴더는 sonnet generation 위주
`paraphrase_detection` 폴더는 paraphrase detection 위주의 코드를 저장.

`paraphrase_detection/codes` 에는 `_(date)_(method)` 형식으로 trial code를 모두 기록함.



=> (sonnet generation 폴더링 설명 넣을 것.)



## 학습 재현 방법 (paraphrase detection)

`paraphrase_detection/codes` 경로에서

```bash
$ python (코드 파일명).py [hyperparameter]
```



예를 들어, `_250611_Adverserial_gem.py` 를 20 epoch, lr 1e-5, batch size 128로 실행

```bash
$ python _250611_Adverserial_gem.py --epochs 20 --lr 1e-5 --batch_size 128
```



각 method에 따른 pipeline마다 설정할 수 있는 hyperparameter가 다르니, 코드를 확인한 후 실행 바람.

(특히, `_250613_default_BackTranslation.py` 에서는 학습 dataset을 다르게 지정해주어야 함. `quora-train-augmented-bt-merged.csv`)

=> BackTranslation으로 증강한 dataset은 용량 문제로 github에 업로드하지 못하였으니,
재현 시 아래 링크에서 다운로드 받아 `data/` 경로에 넣어주세요.

https://drive.google.com/file/d/1IgWqM0psvTXcBK95QPZ8ntjKZ3JJavKS/view?usp=sharing



**\**참고**

여러 코드에서 이 설정을 사용하고 있으나, A6000 48gb가 아닌 다른 gpu를 사용한다면 주석처리 할 것을 권함.

```python
torch.set_float32_matmul_precision('high')
torch.set_float32_matmul_precision('medium')
```



학습 환경은 A6000 48gb GPU를 사용하였으며,
많은 실험에서 epoch당 약 15분~30분 이내의 시간이 소요되었다. (gpt2 model size 기준.)



## 학습 재현 방법 (sonnet generation)

=> (sonnet generation 학습 재현 방법 넣을 것.)
