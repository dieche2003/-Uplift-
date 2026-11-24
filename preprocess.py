"""阶段1：Criteo uplift v2.1 数据校验、EDA、分层抽样及数据划分。

运行位置：D:\\xiangmu-guanggao
依赖：python -m pip install pandas numpy scikit-learn pyarrow
运行：python preprocess.py
本文件不训练模型，也不根据测试集选择任何建模方案。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(r"D:\xiangmu-guanggao")
RAW_FILE = ROOT / "raw" / "criteo-uplift-v2.1.csv"
DATA_DIR = ROOT / "data"
FEATURES = [f"f{i}" for i in range(12)]
LABELS = ["treatment", "exposure", "visit", "conversion"]
COLUMNS = FEATURES + LABELS
RANDOM_STATE = 42
SAMPLE_FRACTION = 0.30


def load_and_validate(path: Path) -> pd.DataFrame:
    """先检查表头，再读取完整数据；标签校验完成后才转为整数。"""
    if not path.is_file():
        raise FileNotFoundError(f"未找到原始数据，请检查路径：{path}")
    header = pd.read_csv(path, compression="infer", nrows=0).columns.tolist()
    if len(header) != len(COLUMNS) or set(header) != set(COLUMNS):
        raise ValueError(f"字段不符合预期。预期：{COLUMNS}；实际：{header}")

    # 大数据采用 float32 特征以节省内存；标签暂用浮点数，避免整数转换
    # 在校验前截断非法值（例如 0.5）。不对原始特征拟合任何变换。
    dtypes = {name: "float32" for name in FEATURES}
    dtypes.update({name: "float64" for name in LABELS})
    df = pd.read_csv(path, compression="infer", dtype=dtypes)
    if df.empty:
        raise ValueError("原始数据为空。")
    print("\n原始数据前10行：")
    print(df.head(10).to_string(index=False))
    print(f"\n字段列表：{df.columns.tolist()}")
    print(f"数据形状：{df.shape}")
    print("\n各字段缺失值数量：")
    print(df.isna().sum().to_string())
    if df.isna().any().any():
        raise ValueError("发现缺失值，请先核实数据完整性；本阶段不自动填补。")
    for name in LABELS:
        if not df[name].isin([0, 1]).all():
            raise ValueError(f"{name} 存在非0/1取值。")
        df[name] = df[name].astype("uint8")
    if df["treatment"].nunique() != 2:
        raise ValueError("treatment 必须同时包含0和1。")
    # 逐列检查，避免一次性为所有特征生成大型中间数组。
    for name in FEATURES:
        if not np.isfinite(df[name].to_numpy()).all():
            raise ValueError(f"{name} 存在无穷大或非法数值。")
    return df


def describe_data(df: pd.DataFrame) -> None:
    """报告原始分布；均值等于二元标签的正例比例。"""
    for name in LABELS:
        counts = df[name].value_counts().reindex([0, 1], fill_value=0)
        table = pd.DataFrame({"数量": counts, "比例": counts / len(df)})
        print(f"\n{name} 分布：\n{table.to_string()}")
    print(f"\nvisit 均值：{df['visit'].mean():.6f} ({df['visit'].mean():.3%})")
    print(f"conversion 均值：{df['conversion'].mean():.6f} "
          f"({df['conversion'].mean():.3%})")
    print("参考比例为 visit≈4.7%、conversion≈0.29%；不强行修改数据以匹配参考值。")
    for name in ["visit", "conversion"]:
        positives = int(df[name].sum())
        ratio = (len(df) - positives) / positives if positives else np.inf
        print(f"{name} 正例数={positives:,}，负例/正例={ratio:.2f}。")
    print("标签正例较少，尤其conversion稀疏；后续估计和小规模Top-K分析可能波动较大。")
    print("\nITT口径核查：treatment为随机分配，exposure为实际曝光：")
    print(pd.crosstab(df["treatment"], df["exposure"]).to_string())
    print("保留全部随机分配样本，不按exposure过滤；实际曝光不能替代随机分配。")


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """用于比较抽样前后组别比例及结果变量均值。"""
    table = df.groupby("treatment", observed=True).agg(
        n=("visit", "size"), visit_mean=("visit", "mean"),
        conversion_mean=("conversion", "mean"),
        exposure_mean=("exposure", "mean"),
    ).reindex([0, 1])
    table["share"] = table["n"] / len(df)
    return table


def feature_smd(df: pd.DataFrame) -> pd.Series:
    """计算处理前特征标准化均值差：(处理组均值-对照组均值)/合并标准差。

    使用float64累积统计量以改善数值精度；常数且均值相同的特征记为0。
    """
    result = {}
    treatment = df["treatment"].to_numpy()
    for name in FEATURES:
        values = df[name].to_numpy(dtype="float64")
        control, treated = values[treatment == 0], values[treatment == 1]
        gap = treated.mean() - control.mean()
        scale = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
        result[name] = gap / scale if scale > 0 else (0.0 if gap == 0 else np.inf)
    return pd.Series(result, name="SMD")


def check_sampling(original: pd.DataFrame, sampled: pd.DataFrame) -> None:
    """描述抽样保真度及特征平衡性，不把结果变量差异当作随机化检验。"""
    before, after = group_summary(original), group_summary(sampled)
    print("\n抽样前分组统计（0=对照组，1=处理组）：\n", before.to_string())
    print("\n抽样后分组统计：\n", after.to_string())
    print("\n抽样后减去抽样前的组内visit均值变化：")
    print((after["visit_mean"] - before["visit_mean"]).to_string())
    gap = after.loc[1, "visit_mean"] - after.loc[0, "visit_mean"]
    print(f"抽样后处理组-对照组visit均值差：{gap:.6f}（{gap * 100:.4f}个百分点）")
    print("该差异可能包含广告处理效应，不能据此证明或否定RCT随机性。")

    smd = pd.concat([feature_smd(original).rename("原始SMD"),
                     feature_smd(sampled).rename("抽样SMD")], axis=1)
    print("\n处理前特征平衡性（SMD）：\n", smd.to_string())
    maximum = smd["抽样SMD"].abs().max()
    if maximum < 0.1:
        print("抽样后所有特征|SMD|<0.1，未发现明显的已观测特征不平衡。")
    else:
        print("部分特征|SMD|≥0.1，建议核查；不通过反复换随机种子筛选样本。")
    print("0.1是描述性经验阈值；特征平衡不能证明随机化，未观测变量仍无法检验。")
    print("本流程仅按treatment组内随机抽样，未按结果标签或实际曝光筛选样本。")


def split_sample(sampled: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """先保留20%测试集，再将余下80%按7:1拆为训练与验证集。"""
    train_val, test = train_test_split(
        sampled, test_size=0.2, random_state=RANDOM_STATE,
        stratify=sampled["treatment"],
    )
    train, validation = train_test_split(
        train_val, test_size=0.125, random_state=RANDOM_STATE,
        stratify=train_val["treatment"],
    )
    parts = {"train": train, "validation": validation, "test": test}
    if sum(len(part) for part in parts.values()) != len(sampled):
        raise RuntimeError("数据划分数量不一致。")
    # 保留读取时的唯一行索引，校验同一条原始记录不会进入多个集合。
    indices = pd.Index(np.concatenate([part.index.to_numpy() for part in parts.values()]))
    if not indices.is_unique:
        raise RuntimeError("训练、验证、测试集存在原始行重叠。")
    return parts


def save_subsets(sampled: pd.DataFrame, parts: dict[str, pd.DataFrame]) -> None:
    """保存30%总样本及三份互斥划分，验证parquet写入后的行数与字段。"""
    import pyarrow.parquet as pq

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, subset in {"sampled": sampled, **parts}.items():
        path = DATA_DIR / f"{name}.parquet"
        subset.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        metadata = pq.ParquetFile(path)
        if metadata.metadata.num_rows != len(subset) or metadata.schema_arrow.names != COLUMNS:
            raise RuntimeError(f"保存后校验失败：{path}")
        print(f"\n已保存：{path}，行数={len(subset):,}")
        print(group_summary(subset).to_string())
        if name != "sampled":
            print(f"占抽样子集比例：{len(subset) / len(sampled):.4%}")
            if subset["conversion"].sum() == 0:
                print("提示：本集合没有conversion正例，后续conversion分析需特别处理。")


def main() -> None:
    # 先确认parquet依赖可用，避免读完大文件后才发现无法存储。
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError("请先运行：python -m pip install pyarrow") from exc
    original = load_and_validate(RAW_FILE)[COLUMNS]
    describe_data(original)
    # 按treatment分层保留30%；保留原始组别比例，不强行重采样成精确85:15。
    _, sampled = train_test_split(
        original, test_size=SAMPLE_FRACTION, random_state=RANDOM_STATE,
        stratify=original["treatment"],
    )
    del _  # 立即释放未选中的70%样本，降低峰值内存占用。
    print(f"\n抽样规模：{len(sampled):,}/{len(original):,} "
          f"({len(sampled) / len(original):.4%})，随机种子={RANDOM_STATE}")
    check_sampling(original, sampled)
    del original
    parts = split_sample(sampled)
    save_subsets(sampled, parts)
    print("\n阶段1完成：仅预处理与EDA，未训练任何模型。")


if __name__ == "__main__":
    main()
