import argparse
import datetime
from pathlib import Path

import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import csv

# 로컬 모듈
from _250611_Adverserial_gem import (
    ParaphraseDataModule,
    ParaphraseLitModuleAPT,
)


def get_test_args():
    """테스트 스크립트 전용 인자 파서."""
    parser = argparse.ArgumentParser(
        description="Paraphrase Detection 모델 체크포인트 테스트 스크립트"
    )

    # 필수 인자
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="테스트에 사용할 .ckpt 파일 경로",
    )

    # 데이터 & 출력 경로
    parser.add_argument(
        "--test_file",
        type=str,
        default="../data/quora-dev.csv",
        help="테스트(또는 dev) 데이터가 저장된 TSV/CSV 파일 경로",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../predictions",
        help="예측 결과가 저장될 디렉토리",
    )

    # 기타 하이퍼파라미터 (필요 시 수정 가능)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1, help="사용할 GPU 개수 (0이면 CPU)")

    return parser.parse_args()


def main():
    args = get_test_args()

    # 1) 체크포인트로부터 모델 로드
    model: ParaphraseLitModuleAPT = ParaphraseLitModuleAPT.load_from_checkpoint(
        args.ckpt_path
    )

    # 2) 출력 파일 경로 설정 (타임스탬프 기반)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model.hparams.para_dev_out = str(Path(args.output_dir) / f"dev_pred_{timestamp}.csv")
    model.hparams.para_test_out = str(Path(args.output_dir) / f"test_pred_{timestamp}.csv")

    # 3) DataModule 준비 (train 데이터는 사용하지 않으므로 dummy 값)
    dm_hparams = argparse.Namespace(
        # 데이터 경로
        para_train=args.test_file,  # 더미
        para_dev=args.test_file,
        para_test=args.test_file,
        # 데이터 로딩 및 토크나이저
        model_size=model.hparams.model_size,
        max_length=getattr(model.hparams, "max_length", 128),
        # 배치 관련
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule = ParaphraseDataModule(dm_hparams)

    # 4) Trainer 생성 후 테스트 실행
    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus,
        precision=getattr(model.hparams, "precision", "16-mixed"),
        logger=False,
    )

    print("--- Starting Testing ---")
    trainer.test(model=model, datamodule=datamodule)
    print(f"\nPredictions saved to: {model.hparams.para_dev_out} & {model.hparams.para_test_out}")

    # --- Dev metrics 계산 및 저장 --------------------------------------
    y_true, y_pred = [], []
    # 1) 예측 결과 읽기
    with open(model.hparams.para_dev_out, "r") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            y_pred.append(int(row["Predicted_Is_Paraphrase"].strip()))
            y_true.append(None)  # placeholder, will fill later

    # 2) 정답 레이블 읽기 (dev 파일)
    gt_map = {}
    with open(args.test_file, "r") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            gt_map[row["id"].strip()] = int(float(row["is_duplicate"].strip()))

    # dev 예측 파일과 동일 순서 보장 위해 다시 iterate
    y_true = []
    with open(model.hparams.para_dev_out, "r") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            sent_id = row["id"].strip() if "id" in row else row["id"]
            y_true.append(gt_map[sent_id])

    # 3) 메트릭 계산
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    cls_report = classification_report(y_true, y_pred)

    metrics_txt_path = Path(args.output_dir) / f"dev_metrics_{timestamp}.txt"
    with open(metrics_txt_path, "w") as fp:
        fp.write("Dev set evaluation metrics\n")
        fp.write(f"Accuracy: {acc:.4f}\n")
        fp.write(f"Precision (macro): {prec:.4f}\n")
        fp.write(f"Recall (macro): {rec:.4f}\n")
        fp.write(f"F1-score (macro): {f1:.4f}\n\n")
        fp.write("Classification Report:\n")
        fp.write(cls_report)
    print(f"Dev metrics saved to: {metrics_txt_path}")


if __name__ == "__main__":
    main() 