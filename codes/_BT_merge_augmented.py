# paraphrase_detection/codes/_BT_merge_augmented.py
import argparse, csv, random
from pathlib import Path
import pandas as pd

def seed_everything(seed=11711):
    random.seed(seed)

def get_args():
    p=argparse.ArgumentParser("Merge language-specific BT CSVs")
    p.add_argument("--para_train", type=str, default="../data/quora-train.csv")
    p.add_argument("--langs", nargs="+", default=["ar","de","es","fr","ru","zh"])
    p.add_argument("--output_path", type=str, default=None,
                   help="저장 경로(default: <para_train>-augmented-bt.csv)")
    p.add_argument("--seed", type=int, default=11711)
    return p.parse_args()

def main():
    args=get_args()
    seed_everything(args.seed)
    base=Path(args.para_train)
    df_base=pd.read_csv(base,sep="\t")

    augmented=[]
    for lang in args.langs:
        f=base.parent/f"{base.stem}-augmented-bt_{lang}.csv"
        if not f.exists():
            print(f"[WARN] {f.name} 없음 – 건너뜀")
            continue
        df=pd.read_csv(f,sep="\t")
        # 증강 부분만 가져오고 싶다면 다음 라인 사용
        # df=df[~df['id'].astype(str).str.endswith(f"_bt_{lang}")]
        augmented.append(df[df["id"].astype(str).str.endswith(f"_bt_{lang}")])

    df_final=pd.concat([df_base]+augmented,ignore_index=True).sample(frac=1).reset_index(drop=True)
    out=Path(args.output_path) if args.output_path else base.parent/f"{base.stem}-augmented-bt-merged.csv"
    df_final.to_csv(out,sep="\t",index=False,quoting=csv.QUOTE_NONNUMERIC)
    print(f"통합 CSV 저장 완료 → {out.resolve()}  (총 {len(df_final):,} rows)")

if __name__=="__main__":
    main()