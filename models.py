"""阶段2传统预测基线及阶段3 uplift训练。运行位置：D:\\xiangmu-guanggao。

安装依赖：python -m pip install pandas numpy scikit-learn lightgbm joblib pyarrow scikit-uplift
阶段2：python -X utf8 -u models.py --stage baselines
阶段3：python -X utf8 -u models.py --stage uplift
基线只使用f0~f11；uplift另外使用treatment分配信息。均不使用exposure。
"""

import json
import platform
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import binary_auc


ROOT = Path(r"D:\xiangmu-guanggao")
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
RESULT_DIR = ROOT / "result"
FEATURES = [f"f{i}" for i in range(12)]
TARGET = "visit"
RANDOM_STATE = 42


def load_prediction_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """只从磁盘读取允许使用的特征及标签，避免误用处理/曝光信息。"""
    if not path.is_file():
        raise FileNotFoundError(f"未找到数据集：{path}")
    frame = pd.read_parquet(path, columns=FEATURES + [TARGET])
    if frame.empty or frame.isna().any().any():
        raise ValueError(f"数据为空或包含缺失值：{path}")
    if set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError(f"visit必须同时包含0和1：{path}")
    for name in FEATURES:
        if not np.isfinite(frame[name].to_numpy()).all():
            raise ValueError(f"特征{name}包含非法数值：{path}")
    # 使用显式列名及固定列顺序；后续预测也必须传入这12个字段。
    return frame.loc[:, FEATURES], frame[TARGET]


def resolve_validation_file(data_dir: Path) -> Path:
    """兼容本阶段指定的val.parquet及阶段1产出的validation.parquet。"""
    for filename in ["val.parquet", "validation.parquet"]:
        path = data_dir / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"{data_dir}下未找到val.parquet或validation.parquet。")


def train_lr(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """标准化与LR绑定保存；标准化参数仅从训练集估计。

    使用L2正则化的LR，max_iter严格设为1000；不做类别重采样或权重修改。
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000, solver="lbfgs", C=1.0,
            random_state=RANDOM_STATE,
        )),
    ])
    model.fit(X_train, y_train)
    if model.named_steps["classifier"].n_iter_.max() >= 1000:
        raise RuntimeError("LR达到迭代上限，请检查收敛情况，不能直接报告为收敛模型。")
    return model


def train_lightgbm(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_validation: pd.DataFrame, y_validation: pd.Series,
) -> lgb.LGBMClassifier:
    """固定超参数，验证集AUC决定早停轮数；测试集不参与训练或早停。

    不对合并后的训练+验证集重新拟合，从而保持清晰、可复用的评估流程。
    限定8个线程以避免训练占满机器资源。
    """
    model = lgb.LGBMClassifier(
        objective="binary", metric="auc", n_estimators=1000, learning_rate=0.05,
        num_leaves=31, max_depth=-1, min_child_samples=100,
        colsample_bytree=0.9, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=8,
        deterministic=True, force_col_wise=True, verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_validation, y_validation)], eval_metric="auc",
        # 仅监控验证集AUC；关闭默认binary_logloss指标对早停的影响。
        callbacks=[lgb.early_stopping(50, first_metric_only=True),
                   lgb.log_evaluation(100)],
    )
    return model


def positive_probability(model, features: pd.DataFrame) -> np.ndarray:
    """统一取出visit=1概率，并核实特征输入未混入任何额外字段。"""
    if features.columns.tolist() != FEATURES:
        raise ValueError("预测输入必须严格按顺序只包含f0~f11。")
    positive_index = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
    return model.predict_proba(features)[:, positive_index]


def run_baselines(data_dir: Path = DATA_DIR, model_dir: Path = MODEL_DIR) -> pd.DataFrame:
    """完整基线流程；返回对比表，供后续阶段调用。

    训练和验证完成后才加载测试集，测试AUC不参与方案选择。
    joblib保存LR完整Pipeline与LightGBM，另保存指标与运行配置。
    """
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    validation_path = resolve_validation_file(data_dir)
    print(f"验证集实际路径：{validation_path}", flush=True)
    print(f"唯一输入特征：{FEATURES}；预测标签：{TARGET}", flush=True)
    X_train, y_train = load_prediction_split(data_dir / "train.parquet")
    X_val, y_val = load_prediction_split(validation_path)
    print(f"训练集：{len(y_train):,}；验证集：{len(y_val):,}", flush=True)

    print("\n开始训练LR（含训练集拟合的标准化）……", flush=True)
    models = {"LR": train_lr(X_train, y_train)}
    print("开始训练LightGBM（验证集AUC早停）……", flush=True)
    models["LightGBM"] = train_lightgbm(X_train, y_train, X_val, y_val)
    validation_scores = {
        name: binary_auc(y_val, positive_probability(model, X_val))
        for name, model in models.items()
    }
    train_rows, validation_rows = len(y_train), len(y_val)
    del X_train, y_train, X_val, y_val

    # 测试集仅在模型确定后评估；不依据测试分数反向调整参数。
    X_test, y_test = load_prediction_split(data_dir / "test.parquet")
    records = []
    for name, model in models.items():
        prediction = positive_probability(model, X_test)
        test_auc = binary_auc(y_test, prediction)
        filename = f"{name.lower()}_baseline.joblib"
        path = model_dir / filename
        joblib.dump(model, path, compress=3)
        # 小批量核对序列化前后的概率一致性，确认保存模型可复用。
        restored = joblib.load(path)
        np.testing.assert_allclose(
            positive_probability(restored, X_test.iloc[:128]), prediction[:128],
            rtol=1e-10, atol=1e-12,
        )
        records.append({
            "model": name, "target": TARGET, "n_features": len(FEATURES),
            "validation_auc": validation_scores[name], "test_auc": test_auc,
            "test_rows": len(y_test), "model_file": filename,
        })
        print(f"已保存{name}模型：{path}", flush=True)

    result = pd.DataFrame(records)
    result.to_csv(model_dir / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "features": FEATURES, "target": TARGET, "random_state": RANDOM_STATE,
        "train_rows": train_rows, "validation_rows": validation_rows,
        "test_rows": len(y_test), "validation_file": str(validation_path),
        "lightgbm_best_iteration": int(models["LightGBM"].best_iteration_),
        "lr_iterations": int(models["LR"].named_steps["classifier"].n_iter_.max()),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "lightgbm": lgb.__version__, "pandas": pd.__version__},
        "parameters": {
            "lr": models["LR"].named_steps["classifier"].get_params(),
            "lightgbm": models["LightGBM"].get_params(),
        },
    }
    (model_dir / "baseline_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("\n传统预测基线对比（AUC）：", flush=True)
    print(result[["model", "validation_auc", "test_auc"]].to_string(
        index=False, float_format=lambda value: f"{value:.6f}",
    ), flush=True)
    print("\nAUC衡量visit预测排序能力，不代表广告增量或ROI。", flush=True)
    return result


# ==================== 阶段3：uplift模块 ====================


def load_uplift_split(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """复用基线特征校验，额外读取ITT分组变量；不读取exposure。

    X保持12列；treatment单独传给sklift，避免S-learner重复添加分组列。
    """
    X, y = load_prediction_split(path)
    treatment = pd.read_parquet(path, columns=["treatment"])["treatment"]
    if not treatment.index.equals(X.index) or len(treatment) != len(X):
        raise ValueError("treatment与特征行索引不一致。")
    if set(treatment.unique()) != {0, 1}:
        raise ValueError("treatment必须同时包含0、1且不能有缺失值。")
    for group in [0, 1]:
        if set(y[treatment == group].unique()) != {0, 1}:
            raise ValueError(f"treatment={group}组的visit必须同时包含0和1。")
    return X, y, treatment


def make_uplift_estimator() -> lgb.LGBMClassifier:
    """两个learner使用相同的预设LightGBM配置，不用测试集调整轮数。

    本阶段仅使用训练集拟合，固定500棵树，无早停、无超参数搜索。
    不进行85:15组别重采样，也不做类别加权，以保留原始条件概率口径。
    colsample_bytree=1确保S-learner各轮均可使用treatment特征。
    """
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.05,
        num_leaves=31, max_depth=-1, min_child_samples=100,
        colsample_bytree=1.0, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=8, deterministic=True,
        force_col_wise=True, verbosity=-1,
    )


def train_s_learner(X: pd.DataFrame, y: pd.Series, treatment: pd.Series):
    """SoloModel(dummy)：内部拟合P(visit=1|X,treatment)。

    调用者只传12列X；sklift内部添加treatment，实际基学习器共13列。
    预测时分别设置treatment=1和0，输出两种条件概率之差。
    """
    from sklift.models import SoloModel

    if X.columns.tolist() != FEATURES:
        raise ValueError("S-learner的X只能包含f0~f11，treatment必须单独传入。")
    model = SoloModel(estimator=make_uplift_estimator(), method="dummy")
    model.fit(X, y, treatment=treatment)
    if model.estimator.feature_name_ != FEATURES + ["treatment"]:
        raise RuntimeError("S-learner内部特征与预期不符。")
    return model


def train_t_learner(X: pd.DataFrame, y: pd.Series, treatment: pd.Series):
    """TwoModels(vanilla)：在treatment=1和0组分别拟合独立模型。

    两个基学习器均仅使用f0~f11；不采用DDR等其他方法。
    predict(X)返回处理组概率减去对照组概率，允许出现负值。
    """
    from sklift.models import TwoModels

    if X.columns.tolist() != FEATURES:
        raise ValueError("T-learner的X必须只包含f0~f11。")
    model = TwoModels(
        estimator_trmnt=make_uplift_estimator(),
        estimator_ctrl=make_uplift_estimator(), method="vanilla",
    )
    model.fit(X, y, treatment=treatment)
    for estimator in [model.estimator_trmnt, model.estimator_ctrl]:
        if estimator.feature_name_ != FEATURES:
            raise RuntimeError("T-learner内部特征与预期不符。")
    return model


def predict_uplift(model, X: pd.DataFrame) -> dict[str, np.ndarray]:
    """使用sklift的predict计算uplift，并核对两种条件概率之差。

    不传入测试集的实际treatment：每个测试样本均预测两种处理状态。
    返回副本，防止sklift下一次predict更新内部缓存影响已有结果。
    """
    if X.columns.tolist() != FEATURES:
        raise ValueError("uplift预测输入必须严格按顺序只包含f0~f11。")
    uplift = np.asarray(model.predict(X)).copy()
    p1 = np.asarray(model.trmnt_preds_).copy()
    p0 = np.asarray(model.ctrl_preds_).copy()
    for probability in [p1, p0]:
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise RuntimeError("模型输出了非法概率。")
    np.testing.assert_allclose(uplift, p1 - p0, rtol=1e-10, atol=1e-12)
    return {"p_treatment": p1, "p_control": p0, "uplift": uplift}


def run_uplift_models(
    data_dir: Path = DATA_DIR, model_dir: Path = MODEL_DIR,
    result_dir: Path = RESULT_DIR,
) -> pd.DataFrame:
    """阶段3入口：训练、保存、测试集打分；本阶段不计算uplift评估指标。"""
    import hashlib
    import sklift

    data_dir, model_dir, result_dir = Path(data_dir), Path(model_dir), Path(result_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    X_train, y_train, treatment_train = load_uplift_split(data_dir / "train.parquet")
    group_sizes = treatment_train.value_counts().sort_index()
    print(f"训练集：{len(y_train):,}；ITT分组数量：{group_sizes.to_dict()}", flush=True)
    print("固定LightGBM配置：500棵树；不使用测试集选参数或早停。", flush=True)
    print("开始训练S-learner（SoloModel）……", flush=True)
    models = {"s_learner": train_s_learner(X_train, y_train, treatment_train)}
    print("开始训练T-learner（TwoModels vanilla）……", flush=True)
    models["t_learner"] = train_t_learner(X_train, y_train, treatment_train)
    train_rows = len(y_train)
    del X_train, y_train, treatment_train

    # 在测试预测前保存，避免将sklift的测试预测缓存数组打包进模型。
    for name, model in models.items():
        path = model_dir / f"{name}_uplift.joblib"
        joblib.dump(model, path, compress=3)
        print(f"已保存：{path}", flush=True)

    test_path = data_dir / "test.parquet"
    X_test, y_test, treatment_test = load_uplift_split(test_path)
    # test_row_id是test.parquet中的0起始行位置，不是原始CSV的用户ID。
    # 保持测试文件原顺序；visit、treatment仅用于后续评估，不用于打分。
    predictions = pd.DataFrame({
        "test_row_id": np.arange(len(X_test), dtype="int64"),
        "treatment": treatment_test.to_numpy(), "visit": y_test.to_numpy(),
    })
    summaries = []
    for name, model in models.items():
        print(f"正在对测试集预测：{name}……", flush=True)
        scores = predict_uplift(model, X_test)
        for column, values in scores.items():
            predictions[f"{name}_{column}"] = values
        restored = joblib.load(model_dir / f"{name}_uplift.joblib")
        roundtrip = predict_uplift(restored, X_test.iloc[:128])
        for column in scores:
            np.testing.assert_allclose(
                roundtrip[column], scores[column][:128], rtol=1e-10, atol=1e-12,
            )
        summaries.append({
            "model": name, "test_rows": len(X_test),
            "mean_predicted_uplift": float(scores["uplift"].mean()),
            "min_predicted_uplift": float(scores["uplift"].min()),
            "max_predicted_uplift": float(scores["uplift"].max()),
            "negative_score_share": float((scores["uplift"] < 0).mean()),
        })

    output_path = result_dir / "uplift_test_predictions.parquet"
    predictions.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
    pd.testing.assert_frame_equal(pd.read_parquet(output_path), predictions)
    summary = pd.DataFrame(summaries)
    summary.to_csv(result_dir / "uplift_prediction_summary.csv", index=False, encoding="utf-8-sig")
    # 记录测试文件指纹，便于后续确认得分与测试文件来自同一次划分。
    digest = hashlib.sha256()
    with test_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    metadata = {
        "target": TARGET, "estimand": "ITT", "features": FEATURES,
        "treatment_column": "treatment", "exposure_used": False,
        "train_file": str(data_dir / "train.parquet"), "test_file": str(test_path),
        "test_file_sha256": digest.hexdigest(), "train_rows": train_rows,
        "test_rows": len(predictions), "training_group_sizes": group_sizes.to_dict(),
        "test_row_id_definition": "zero-based positional row number in test.parquet",
        "uplift_definition": "P(visit=1|X,treatment=1)-P(visit=1|X,treatment=0)",
        "validation_used": False, "test_used_for_tuning": False,
        "methods": {"s_learner": "SoloModel(dummy)", "t_learner": "TwoModels(vanilla)"},
        "base_estimator_parameters": make_uplift_estimator().get_params(),
        "versions": {"sklift": sklift.__version__, "lightgbm": lgb.__version__,
                     "sklearn": sklearn.__version__, "pandas": pd.__version__},
    }
    (result_dir / "uplift_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n测试集预测已保存：{output_path}", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("以上为预测值分布，不是Qini、AUUC或实际业务收益。阶段3完成。", flush=True)
    return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="阶段2基线 / 阶段3 uplift模型训练")
    parser.add_argument("--stage", choices=["baselines", "uplift"], default="baselines")
    arguments = parser.parse_args()
    if arguments.stage == "uplift":
        run_uplift_models()
    else:
        run_baselines()
