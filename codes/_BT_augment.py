import argparse
import csv
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

# ---------------------------------------------------------------------------
# 0. 시드 고정 유틸리티
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 11711):
    """재현성을 위해 각종 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 1. Back-Translation 모듈
# ---------------------------------------------------------------------------

class BackTranslator:
    """Hugging Face MarianMT 모델을 사용하여 문장을 지정 언어 → 영어로 역번역합니다."""

    def __init__(self, target_lang: str = "es", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_lang = target_lang

        model_en_to_target = f"Helsinki-NLP/opus-mt-en-{target_lang}"
        model_target_to_en = f"Helsinki-NLP/opus-mt-{target_lang}-en"

        print(f"Loading model for en → {target_lang}…")
        self.tokenizer_en_to_target = MarianTokenizer.from_pretrained(model_en_to_target)
        self.model_en_to_target = MarianMTModel.from_pretrained(model_en_to_target).to(
            self.device
        )

        # GPU 사용 시 메모리 절약을 위해 FP16 모드로 변환
        if self.device.startswith("cuda"):
            self.model_en_to_target.half()

        print(f"Loading model for {target_lang} → en…")
        self.tokenizer_target_to_en = MarianTokenizer.from_pretrained(model_target_to_en)
        self.model_target_to_en = MarianMTModel.from_pretrained(model_target_to_en).to(
            self.device
        )

        if self.device.startswith("cuda"):
            self.model_target_to_en.half()

        print(f"Models for language '{target_lang}' loaded on {self.device}.")

    @torch.inference_mode()
    def translate(self, sentences: List[str], batch_size: int = 64) -> List[str]:
        """문장 리스트를 batch 단위로 역번역한다."""
        augmented: List[str] = []
        for i in tqdm(
            range(0, len(sentences), batch_size), desc=f"Back-translating ({self.target_lang})"
        ):
            batch = sentences[i : i + batch_size]
            inputs = self.tokenizer_en_to_target(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)

            # 자동 혼합 정밀도(MP) 활성화
            with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
                translated_ids = self.model_en_to_target.generate(**inputs)
            translated_texts = self.tokenizer_en_to_target.batch_decode(
                translated_ids, skip_special_tokens=True
            )

            inputs_back = self.tokenizer_target_to_en(
                translated_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)

            with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
                back_ids = self.model_target_to_en.generate(**inputs_back)
            back_texts = self.tokenizer_target_to_en.batch_decode(back_ids, skip_special_tokens=True)
            augmented.extend(back_texts)
        return augmented


# ---------------------------------------------------------------------------
# 2. 증강 파이프라인
# ---------------------------------------------------------------------------

def run_augmentation(args):
    """Quora paraphrase 데이터에서 positive 샘플을 역번역 증강한다."""
    print("=" * 60)
    print("STEP 1: Starting Data Augmentation using Back-Translation")
    print("=" * 60)
    print(f"Loading original data from: {args.para_train}")

    df = pd.read_csv(args.para_train, sep="\t")
    df_pos = df[df["is_duplicate"] == 1].copy()
    print(f"Found {len(df_pos)} positive pairs to augment.")

    # 출력 파일 경로 (언어별)
    out_path = Path(args.augmented_train_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lang = args.lang  # 단일 언어

    # 기존 파일이 있다면 삭제 후 재생성
    if out_path.exists():
        out_path.unlink()

    # --------------------------------------------------
    # 1) 원본 데이터를 먼저 저장 (헤더 포함)
    # --------------------------------------------------
    df.to_csv(
        out_path,
        sep="\t",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
        mode="w",
        header=True,
    )

    # --------------------------------------------------
    # 2) 증강 수행 (단일 언어) & 주기적 저장
    # --------------------------------------------------
    translator = BackTranslator(target_lang=lang)
    augmented_sents = translator.translate(df_pos["sentence2"].tolist(), args.bt_batch_size)

    buffer = []
    augmented_cnt = 0
    for idx, row in df_pos.iterrows():
        aug_s2 = augmented_sents[df_pos.index.get_loc(idx)]
        if row["sentence2"].strip().lower() == aug_s2.strip().lower():
            continue  # 변형되지 않은 문장은 제외

        new_row = {
            "id": f"{row['id']}_bt_{lang}",
            "sentence1": row["sentence1"],
            "sentence2": aug_s2,
            "is_duplicate": 1,
        }
        for q_col in ("qid1", "qid2"):
            if q_col in row.index:
                new_row[q_col] = row[q_col]

        buffer.append(new_row)
        augmented_cnt += 1

        # N개 단위로 즉시 저장하여 중간 결과 보존
        if len(buffer) >= args.save_every:
            pd.DataFrame(buffer).to_csv(
                out_path,
                sep="\t",
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                mode="a",
                header=False,
            )
            buffer.clear()

    # 남은 버퍼 저장
    if buffer:
        pd.DataFrame(buffer).to_csv(
            out_path,
            sep="\t",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
            mode="a",
            header=False,
        )

    print("=" * 60)
    print(
        f"Original size: {len(df):,} | Augmented: {augmented_cnt:,} | Final written: {len(df)+augmented_cnt:,}"
    )
    print(f"Language-specific augmented data saved to {out_path.resolve()}")

    print("Data Augmentation Finished.")


# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser("Back-Translation Augmentation Script")
    p.add_argument("--para_train", type=str, default="../data/quora-train.csv")
    p.add_argument(
        "--augmented_train_path",
        type=str,
        default=None,
        help="경로를 지정하지 않으면 <para_train>-augmented-bt.csv 로 저장합니다.",
    )
    # 증강 대상 언어 (단일)
    p.add_argument("--lang", type=str, default="es", help="중간 번역에 사용할 언어 코드 (예: es, de, fr …)")
    p.add_argument("--bt_batch_size", type=int, default=64)
    p.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="증강 샘플을 N개 생성할 때마다 파일에 즉시 저장합니다.",
    )
    p.add_argument("--seed", type=int, default=11711)
    p.add_argument("--force", action="store_true", help="기존 증강 파일이 있어도 덮어쓴다.")
    args = p.parse_args()

    # 증강 파일 경로 자동 설정 (언어별)
    if args.augmented_train_path is None:
        pt = Path(args.para_train)
        args.augmented_train_path = pt.parent / f"{pt.stem}-augmented-bt_{args.lang}.csv"
    return args


def main():
    args = get_args()
    seed_everything(args.seed)

    out = Path(args.augmented_train_path)
    if not args.force and out.exists():
        print(f"Found existing augmented file at {out}. Use --force to regenerate.")
        return
    run_augmentation(args)


if __name__ == "__main__":
    main() 