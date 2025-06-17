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
nlp2025-1_GPT2/
├── codes/
│   ├── LICENSE
│   ├── _250522_SimCSE.py
│   ├── _250523_default_paraphrase.py
│   ├── _250523_simcse_etc.py
│   ├── _250611_Adverserial_gem.py
│   ├── _250612_BackTranslation.py
│   ├── _250612_BackTranslation_2.py
│   ├── _250612_Multitask.py
│   ├── _250613_default_BackTranslation.py
│   ├── _250614_PeFT.py
│   ├── _250614_pooling.py
│   ├── _250615_contrastive.py
│   ├── _250615_multitask2.py
│   ├── _BT_augment.py
│   ├── _BT_merge_augmented.py
│   ├── classifier.py
│   ├── config.py
│   ├── datasets.py
│   ├── env.yml
│   ├── evaluation.py
│   ├── optimizer.py
│   ├── optimizer_test.npy
│   ├── optimizer_test.py
│   ├── paraphrase_detection.py
│   ├── prepare_submit.py
│   ├── run_test_from_ckpt.py
│   ├── sanity_check.py
│   ├── sonnet_generation_BS.py
│   ├── sonnet_generation_CE.py
│   ├── sonnet_generation_CS.py
│   ├── sonnet_generation_MBR.py
│   ├── sonnet_generation_Prefix.py
│   ├── sonnet_generation_base.py
│   ├── sonnet_generation_fine.py
│   ├── test_250522_default.py
│   ├── test_250522_simcse.py
│   ├── test_250523_simcse_etc.py
│   ├── utils.py
│   ├── models/
│   ├── modules/
│   ├── description/
│   └── __pycache__/
├── data/
│   ├── ETPC.xml
│   ├── TRUE_sonnets_held_out_dev.txt
│   ├── ids-cfimdb-dev.csv
│   ├── ids-cfimdb-test-student.csv
│   ├── ids-cfimdb-train.csv
│   ├── ids-sst-dev.csv
│   ├── ids-sst-test-student.csv
│   ├── ids-sst-train.csv
│   ├── quora-dev.csv
│   ├── quora-test-student.csv
│   ├── quora-train.csv
│   ├── sonnets.txt
│   ├── sonnets_held_out.txt
│   └── sonnets_held_out_dev.txt
├── models/
│   ├── __pycache__/
│   ├── base_gpt.py
│   └── gpt2.py
├── modules/
│   ├── __pycache__/
│   ├── attention.py
│   └── gpt2_layer.py
├── outputs/
│   ├── logs/
│   ├── logs_txt(250522-23)/
│   └── logs_txt/
├── predictions/
│   └── empty.txt
├── sonnet_scr/
│   ├── empty.txt
│   ├── sonnet_generation_BS.py
│   ├── sonnet_generation_CE.py
│   ├── sonnet_generation_CS.py
│   ├── sonnet_generation_MBR.py
│   ├── sonnet_generation_Prefix.py
│   ├── sonnet_generation_base.py
│   └── sonnet_generation_fine.py
├── 자연어처리개론 기말프로젝트 최종보고서.docx
├── 자연어처리개론 기말프로젝트 최종보고서.pdf
├── requirements.txt
├── environment.yml
├── ...etc(기존 코드 등등)
```

**PART-I**의 과제는 완성되어 있음.

`codes` 폴더에 sonnet generation / paraphrase detection의 개선 시도 코드가 작성되어 있음.
paraphrase_detection의 코드는 `_(date)_(method)` 형식으로 naming.
sonnet generation의 코드는 `sonnet_generation__(method)` 형식으로 naming.


## 학습 재현 방법 (paraphrase detection)

`codes` 경로에서

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



## 학습 재현 방법 (Sonnet Generation)

> **⚠️ 현재 `sonnet_scr/` 폴더에 모든 코드가 임시로 들어 있는 상태입니다.**  
> 학습/실행이 정상적으로 동작하려면 `sonnet_scr/` 폴더 안의 파일들을 프로젝트 최상단(`./`)으로 옮겨야 합니다.  
> 예: `sonnet_scr/sonnet_generation_*.py` → `./sonnet_generation_*.py`

---

### 1. 파인튜닝 없이 평가하기

파인튜닝 없이 다양한 생성 전략을 평가하려면 아래 코드 파일을 실행합니다:

- `sonnet_generation_base.py` (Baseline)
- `sonnet_generation_Prefix.py` (Prefix-Tuning)
- `sonnet_generation_CS.py` (Contrastive Search)
- `sonnet_generation_BS.py` (Beam Search)
- `sonnet_generation_MBR.py` (MBR)
- `sonnet_generation_CE.py` (Candidate Ensemble + MBR)

```bash
$ python sonnet_generation_*.py --use_gpu
```

> 각 코드 내부에는 아래와 같은 GPU 설정이 포함되어 있습니다.  
> GPU 환경에 맞게 수정 또는 제거해 주세요:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
```

- GPU가 여러 장인 경우: 사용할 GPU 번호를 위에 입력  
- GPU가 1장뿐이거나 자동 할당을 원할 경우: 위 줄 삭제

---

### 2. 파인튜닝 수행

전체 GPT-2 파라미터를 학습하려면 다음 명령어를 실행합니다:

```bash
$ python sonnet_generation_fine.py --use_gpu
```

- 해당 코드는 기존 `train()` 함수를 개선하여 다음 기법을 포함합니다:
  - Linear warmup scheduler (`get_linear_schedule_with_warmup`)
  - Gradient clipping (`clip_grad_norm_`)
  - Dev CHRF 평가 기반 최고 성능 상위 3개 모델만 저장, 나머지 자동 삭제

> 최종적으로 가장 높은 성능을 기록한 모델은 다음과 같습니다:
>
> `./74_200-1e-05-sonnet.pt`

→ 위 코드는 `sonnet_generation_fine.py`만 실행하면 자동 생성됩니다.

---

### 3. 학습된 모델로 평가 실행하기

학습된 모델을 사용해 평가하려면, `main()` 함수 내 다음 항목을 아래처럼 조정하세요:

```python
# train(args)
generate_submission_sonnets(args)
generate_sonnets_from_checkpoint(args)
```

즉, `train(args)`는 주석 처리하고, `generate_*()` 함수는 주석 해제합니다.

또한 평가에 사용할 체크포인트 경로를 아래와 같이 지정하세요:

```python
parser.add_argument("--checkpoint", type=str,
    default='./74_200-1e-05-sonnet.pt',
    help='불러올 모델 체크포인트 파일 경로')
```
