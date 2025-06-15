#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Lightning 체크포인트를 사용하여 dev set에 대해 테스트를 진행하는 스크립트
체크포인트 경로: /home/2020112534/NLP_final/nlp2025-1/checkpoints/250523_simcse_etc/best-epoch=9-val_acc=0.8682.ckpt

실행:
  python test_checkpoint_dev.py --use_gpu
"""

import argparse
import os
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import pytorch_lightning as pl

from datasets import (
    ParaphraseDetectionDataset,
    load_paraphrase_data
)
from evaluation import model_eval_paraphrase

# 기존 250523_simcse_etc_paraphrase.py에서 모델 클래스 import
from _250523_simcse_etc_paraphrase import ParaphraseLightningModel

TQDM_DISABLE = False

def seed_everything(seed=11711):
    """시드 고정"""
    import random
    import numpy as np
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
    log_filename = f"logs/test_checkpoint_dev_{timestamp}.log"
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

def evaluate_checkpoint_on_dev(args):
    """체크포인트를 로드하고 dev set에 대해 평가를 수행합니다."""
    logger = setup_logging()
    logger.info("=== 체크포인트 Dev Set 평가 시작 ===")
    
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
        # PyTorch Lightning 체크포인트에서 모델 로드
        model = ParaphraseLightningModel.load_from_checkpoint(
            checkpoint_path,
            map_location=device
        )
        model = model.to(device)
        model.eval()
        logger.info("모델 로드 성공")
        
        # 하이퍼파라미터 정보 출력
        logger.info(f"모델 하이퍼파라미터:")
        for key, value in model.hparams.items():
            logger.info(f"  {key}: {value}")
            
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return
    
    # Dev 데이터 로드
    logger.info("Dev 데이터 로드 중...")
    try:
        dev_raw = load_paraphrase_data(args.para_dev, split="dev")
        
        # args 객체 생성 (데이터셋에서 필요한 속성들)
        data_args = argparse.Namespace(
            model_size=model.hparams.get('model_size', 'gpt2'),
            max_length=model.hparams.get('max_length', 128)
        )
        
        dev_dataset = ParaphraseDetectionDataset(dev_raw, data_args)
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
        results_file = f'predictions/dev_evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f'=== Dev Set 평가 결과 ===\n')
            f.write(f'체크포인트: {checkpoint_path}\n')
            f.write(f'정확도 (Accuracy): {dev_acc:.4f}\n')
            f.write(f'F1 스코어: {dev_f1:.4f}\n')
            f.write(f'Precision: {precision:.4f}\n')
            f.write(f'Recall: {recall:.4f}\n')
            f.write(f'전체 샘플: {total_samples}\n')
            f.write(f'정확한 예측: {correct_predictions}\n')
            f.write(f'잘못된 예측: {incorrect_predictions}\n')
            f.write(f'오류율: {incorrect_predictions/total_samples*100:.2f}%\n')
            f.write(f'\n=== 혼동 행렬 ===\n')
            f.write(f'True Positives: {true_positives}\n')
            f.write(f'True Negatives: {true_negatives}\n')
            f.write(f'False Positives: {false_positives}\n')
            f.write(f'False Negatives: {false_negatives}\n')
        
        logger.info(f"상세 평가 결과가 {results_file}에 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"결과 저장 중 오류 발생: {e}")
        return
    
    logger.info("=== Dev Set 평가 완료 ===")
    return dev_acc, dev_f1

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Lightning 체크포인트를 사용한 Dev Set 평가")
    
    # 필수 인자들
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default="/home/2020112534/NLP_final/nlp2025-1/checkpoints/250523_simcse_etc/best-epoch=9-val_acc=0.8682.ckpt",
        help="평가할 PyTorch Lightning 체크포인트 경로"
    )
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv", help="Dev 데이터 경로")
    parser.add_argument("--para_dev_out", type=str, default="predictions/checkpoint-dev-output.csv", help="Dev 예측 결과 저장 경로")
    
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