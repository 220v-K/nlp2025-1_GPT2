#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_250522_SimCSE_paraphrase_detection.py로 생성된 체크포인트를 사용하여 dev set에 대해 테스트를 진행하는 스크립트
체크포인트 경로: /home/2020112534/NLP_final/nlp2025-1/checkpoints/250522_simcse/best_model.pt

실행:
  python test_250522_simcse_checkpoint.py --use_gpu
"""

import argparse
import os
import torch
import logging
import numpy as np
import random
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from datasets import (
    ParaphraseDetectionDataset,
    load_paraphrase_data
)
from evaluation import model_eval_paraphrase
from models.gpt2 import GPT2Model

TQDM_DISABLE = False

def seed_everything(seed=11711):
    """시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup_logging():
    """로깅 설정"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"logs/test_250522_simcse_{timestamp}.log"
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_gpu():
    """GPU 사용 가능 여부 확인"""
    use_gpu = torch.cuda.is_available()
    n_gpu = torch.cuda.device_count()
    return use_gpu, n_gpu

# _250522_SimCSE_paraphrase_detection.py의 ParaphraseGPT 클래스 재정의
class ParaphraseGPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            args.model_size, d=args.d, l=args.l, num_heads=args.num_heads
        )
        self.paraphrase_detection_head = nn.Linear(args.d, 2)
        for p in self.gpt.parameters():
            p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        gpt_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = gpt_out['last_hidden_state']
        last_idx = attention_mask.sum(dim=1) - 1
        last_token_state = hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device),
            last_idx
        ]
        logits = self.paraphrase_detection_head(last_token_state)
        return logits

def add_arguments(args):
    """모델 크기에 따라 결정되는 인수들을 추가."""
    if args.model_size == 'gpt2':
        args.d, args.l, args.num_heads = 768, 12, 12
    elif args.model_size == 'gpt2-medium':
        args.d, args.l, args.num_heads = 1024, 24, 16
    elif args.model_size == 'gpt2-large':
        args.d, args.l, args.num_heads = 1280, 36, 20
    else:
        raise ValueError(f'{args.model_size} unsupported.')
    return args

def evaluate_checkpoint_on_dev(args):
    """체크포인트를 로드하고 dev set에 대해 평가를 수행합니다."""
    logger = setup_logging()
    logger.info("=== 250522 SimCSE 체크포인트 Dev Set 평가 시작 ===")
    
    # GPU 설정
    use_gpu, n_gpu = check_gpu()
    device = torch.device('cuda') if use_gpu and args.use_gpu else torch.device('cpu')
    logger.info(f"사용 디바이스: {device}, GPU 개수: {n_gpu}")
    
    # 체크포인트 경로 확인
    checkpoint_path = args.checkpoint_path
    if not os.path.exists(checkpoint_path):
        logger.error(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return
    
    logger.info(f"체크포인트 로드 중: {checkpoint_path}")
    
    try:
        # PyTorch 체크포인트에서 모델 로드
        saved = torch.load(checkpoint_path, map_location=device)
        
        # 저장된 args 가져오기
        model_args = saved['args']
        logger.info(f"모델 하이퍼파라미터:")
        logger.info(f"  모델 크기: {model_args.model_size}")
        logger.info(f"  학습률: {model_args.lr}")
        logger.info(f"  배치 크기: {model_args.batch_size}")
        logger.info(f"  에폭: {model_args.epochs}")
        logger.info(f"  차원: {model_args.d}")
        logger.info(f"  레이어: {model_args.l}")
        logger.info(f"  헤드 수: {model_args.num_heads}")
        
        # SimCSE 관련 파라미터도 출력
        if hasattr(model_args, 'simcse_epochs'):
            logger.info(f"  SimCSE 에폭: {model_args.simcse_epochs}")
            logger.info(f"  SimCSE 배치 크기: {model_args.simcse_batch}")
            logger.info(f"  SimCSE 학습률: {model_args.simcse_lr}")
            logger.info(f"  SimCSE 온도: {model_args.simcse_tau}")
            logger.info(f"  최대 길이: {model_args.max_len}")
        
        # 모델 생성
        model = ParaphraseGPT(model_args)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        model.eval()
        logger.info("모델 로드 성공")
            
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return
    
    # Dev 데이터 로드
    logger.info("Dev 데이터 로드 중...")
    try:
        dev_raw = load_paraphrase_data(args.para_dev, split="dev")
        
        # 데이터셋을 위한 args 객체 생성 (model_args 사용)
        dev_dataset = ParaphraseDetectionDataset(dev_raw, model_args)
        dev_dataloader = DataLoader(
            dev_dataset, 
            shuffle=False, 
            batch_size=args.batch_size,
            collate_fn=dev_dataset.collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Dev 데이터셋 크기: {len(dev_dataset)}")
        
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        return
    
    # Dev set 평가
    logger.info("Dev set 평가 중...")
    try:
        with torch.no_grad():
            dev_acc, dev_f1, dev_pred, dev_true, dev_ids = model_eval_paraphrase(
                dev_dataloader, model, device
            )
        
        logger.info(f'=== Dev Set 평가 결과 ===')
        logger.info(f'정확도 (Accuracy): {dev_acc:.4f}')
        logger.info(f'F1 스코어: {dev_f1:.4f}')
        
        # 상세 분석
        total_samples = len(dev_true)
        correct_predictions = sum([1 for p, t in zip(dev_pred, dev_true) if p == t])
        incorrect_predictions = total_samples - correct_predictions
        
        logger.info(f'전체 샘플: {total_samples}')
        logger.info(f'정확한 예측: {correct_predictions}')
        logger.info(f'잘못된 예측: {incorrect_predictions}')
        logger.info(f'오류율: {incorrect_predictions/total_samples*100:.2f}%')
        
        # 클래스별 분석
        true_positives = sum([1 for p, t in zip(dev_pred, dev_true) if p == 1 and t == 1])
        true_negatives = sum([1 for p, t in zip(dev_pred, dev_true) if p == 0 and t == 0])
        false_positives = sum([1 for p, t in zip(dev_pred, dev_true) if p == 1 and t == 0])
        false_negatives = sum([1 for p, t in zip(dev_pred, dev_true) if p == 0 and t == 1])
        
        logger.info(f'=== 혼동 행렬 ===')
        logger.info(f'True Positives: {true_positives}')
        logger.info(f'True Negatives: {true_negatives}')
        logger.info(f'False Positives: {false_positives}')
        logger.info(f'False Negatives: {false_negatives}')
        
        # Precision, Recall 계산
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        logger.info(f'Precision: {precision:.4f}')
        logger.info(f'Recall: {recall:.4f}')
        
        # 추가적인 분석: 클래스별 정확도
        class_0_total = sum([1 for t in dev_true if t == 0])
        class_1_total = sum([1 for t in dev_true if t == 1])
        class_0_correct = sum([1 for p, t in zip(dev_pred, dev_true) if p == 0 and t == 0])
        class_1_correct = sum([1 for p, t in zip(dev_pred, dev_true) if p == 1 and t == 1])
        
        logger.info(f'=== 클래스별 성능 ===')
        logger.info(f'클래스 0 (Non-paraphrase) 정확도: {class_0_correct/class_0_total:.4f} ({class_0_correct}/{class_0_total})')
        logger.info(f'클래스 1 (Paraphrase) 정확도: {class_1_correct/class_1_total:.4f} ({class_1_correct}/{class_1_total})')
        
        # SimCSE 효과 분석을 위한 추가 정보
        logger.info(f'=== SimCSE 모델 특성 ===')
        logger.info(f'이 모델은 SimCSE contrastive pre-training을 거친 모델입니다.')
        if hasattr(model_args, 'simcse_epochs'):
            logger.info(f'SimCSE 사전 훈련 에폭: {model_args.simcse_epochs}')
            logger.info(f'SimCSE 온도 파라미터: {model_args.simcse_tau}')
        
    except Exception as e:
        logger.error(f"평가 중 오류 발생: {e}")
        return
    
    # 결과 저장
    try:
        os.makedirs('predictions', exist_ok=True)
        
        # Dev set 예측 결과 저장
        dev_output_path = args.para_dev_out
        logger.info(f"Dev set 예측 결과 저장 중: {dev_output_path}")
        
        with open(dev_output_path, 'w') as f:
            f.write('id,Predicted_Is_Paraphrase\n')
            for pid, pred in zip(dev_ids, dev_pred):
                f.write(f'{pid},{pred}\n')
        
        # 상세 평가 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'predictions/250522_simcse_dev_evaluation_results_{timestamp}.txt'
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f'=== 250522 SimCSE 체크포인트 Dev Set 평가 결과 ===\n')
            f.write(f'체크포인트: {checkpoint_path}\n')
            f.write(f'평가 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'\n=== 모델 하이퍼파라미터 ===\n')
            f.write(f'모델 크기: {model_args.model_size}\n')
            f.write(f'학습률: {model_args.lr}\n')
            f.write(f'배치 크기: {model_args.batch_size}\n')
            f.write(f'에폭: {model_args.epochs}\n')
            f.write(f'차원: {model_args.d}\n')
            f.write(f'레이어: {model_args.l}\n')
            f.write(f'헤드 수: {model_args.num_heads}\n')
            
            # SimCSE 관련 파라미터
            if hasattr(model_args, 'simcse_epochs'):
                f.write(f'\n=== SimCSE 파라미터 ===\n')
                f.write(f'SimCSE 에폭: {model_args.simcse_epochs}\n')
                f.write(f'SimCSE 배치 크기: {model_args.simcse_batch}\n')
                f.write(f'SimCSE 학습률: {model_args.simcse_lr}\n')
                f.write(f'SimCSE 온도: {model_args.simcse_tau}\n')
                f.write(f'최대 길이: {model_args.max_len}\n')
            
            f.write(f'\n=== 성능 지표 ===\n')
            f.write(f'정확도 (Accuracy): {dev_acc:.4f}\n')
            f.write(f'F1 스코어: {dev_f1:.4f}\n')
            f.write(f'Precision: {precision:.4f}\n')
            f.write(f'Recall: {recall:.4f}\n')
            f.write(f'\n=== 기본 통계 ===\n')
            f.write(f'전체 샘플: {total_samples}\n')
            f.write(f'정확한 예측: {correct_predictions}\n')
            f.write(f'잘못된 예측: {incorrect_predictions}\n')
            f.write(f'오류율: {incorrect_predictions/total_samples*100:.2f}%\n')
            f.write(f'\n=== 혼동 행렬 ===\n')
            f.write(f'True Positives: {true_positives}\n')
            f.write(f'True Negatives: {true_negatives}\n')
            f.write(f'False Positives: {false_positives}\n')
            f.write(f'False Negatives: {false_negatives}\n')
            f.write(f'\n=== 클래스별 성능 ===\n')
            f.write(f'클래스 0 (Non-paraphrase) 정확도: {class_0_correct/class_0_total:.4f} ({class_0_correct}/{class_0_total})\n')
            f.write(f'클래스 1 (Paraphrase) 정확도: {class_1_correct/class_1_total:.4f} ({class_1_correct}/{class_1_total})\n')
            f.write(f'\n=== 모델 특성 ===\n')
            f.write(f'이 모델은 SimCSE contrastive pre-training을 거친 모델입니다.\n')
            f.write(f'SimCSE는 문장 표현 학습을 위한 대조 학습 방법론입니다.\n')
        
        logger.info(f"상세 평가 결과가 {results_file}에 저장되었습니다.")
        
        # 성능 요약 파일도 별도로 저장
        summary_file = f'predictions/250522_simcse_dev_summary_{timestamp}.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f'체크포인트: {os.path.basename(checkpoint_path)}\n')
            f.write(f'모델 타입: SimCSE + GPT-2 Paraphrase Detection\n')
            f.write(f'정확도: {dev_acc:.4f}\n')
            f.write(f'F1: {dev_f1:.4f}\n')
            f.write(f'Precision: {precision:.4f}\n')
            f.write(f'Recall: {recall:.4f}\n')
            if hasattr(model_args, 'simcse_epochs'):
                f.write(f'SimCSE 에폭: {model_args.simcse_epochs}\n')
                f.write(f'SimCSE 온도: {model_args.simcse_tau}\n')
        
        logger.info(f"성능 요약이 {summary_file}에 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"결과 저장 중 오류 발생: {e}")
        return
    
    logger.info("=== Dev Set 평가 완료 ===")
    return dev_acc, dev_f1

def get_args():
    parser = argparse.ArgumentParser(description="250522 SimCSE 체크포인트를 사용한 Dev Set 평가")
    
    # 필수 인자들
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default="/home/2020112534/NLP_final/nlp2025-1/checkpoints/250522_simcse/best_model.pt",
        help="평가할 SimCSE 체크포인트 경로"
    )
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv", help="Dev 데이터 경로")
    parser.add_argument("--para_dev_out", type=str, default="predictions/250522-simcse-dev-output.csv", help="Dev 예측 결과 저장 경로")
    
    # 설정 인자들
    parser.add_argument("--seed", type=int, default=11711, help="랜덤 시드")
    parser.add_argument("--use_gpu", action='store_true', help="GPU 사용 여부")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    
    return parser.parse_args()

def main():
    args = get_args()
    seed_everything(args.seed)
    
    # CUDA 설정
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 평가 실행
    evaluate_checkpoint_on_dev(args)

if __name__ == "__main__":
    main()