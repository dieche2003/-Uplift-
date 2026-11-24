"""阶段2/4/5：传统AUC、uplift指标/Qini图及广告ROI敏感性模拟。

运行位置：D:\\xiangmu-guanggao
运行命令：python -X utf8 -u evaluate.py
阶段5命令：python -X utf8 -u evaluate.py --stage roi
本文件只读取已有模型及预测，不重新训练，不按测试结果选择模型参数。
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(r"D:\xiangmu-guanggao")
FEATURES = [f"f{i}" for i in range(12)]


def binary_auc(y_true, positive_probability) -> float:
    """使用正例概率计算ROC AUC，而不是使用0/1预测类别。

    后续uplift评估可继续扩展本文件；普通预测AUC不能解释为增量收益。
    """
    labels = np.asarray(y_true)
    scores = np.asarray(positive_probability)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("真实标签与预测概率必须是一维数组，且长度相同。")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("AUC要求评估集同时包含0、1两类标签。")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("预测概率必须是[0, 1]内的有限数值。")
    return float(roc_auc_score(labels, scores))


def load_aligned_test(root: Path):
    """校验文件指纹、行位置和标签，防止测试集与预测结果错配。"""
    import pandas as pd

    test_path = root / "data" / "test.parquet"
    metadata = json.loads((root / "result" / "uplift_metadata.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    with test_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != metadata["test_file_sha256"]:
        raise ValueError("测试文件与阶段3记录的指纹不一致，请核查数据版本。")
    if metadata["target"] != "visit" or metadata["estimand"] != "ITT":
        raise ValueError("预测元数据的目标或ITT口径不一致。")
    test = pd.read_parquet(test_path, columns=FEATURES + ["treatment", "visit"])
    predictions = pd.read_parquet(root / "result" / "uplift_test_predictions.parquet")
    if len(test) != len(predictions) or len(test) != metadata["test_rows"]:
        raise ValueError("测试集与预测文件的行数不一致。")
    if not np.array_equal(predictions["test_row_id"], np.arange(len(test))):
        raise ValueError("test_row_id不连续或预测顺序发生变化。")
    for name in ["treatment", "visit"]:
        if set(test[name].unique()) != {0, 1}:
            raise ValueError(f"测试集{name}必须同时包含0和1，且不能有缺失值。")
        if not np.array_equal(predictions[name], test[name]):
            raise ValueError(f"预测文件中的{name}与测试集未逐行对齐。")
    for prefix in ["s_learner", "t_learner"]:
        scores = predictions[f"{prefix}_uplift"].to_numpy()
        if not np.isfinite(scores).all() or (np.abs(scores) > 1).any():
            raise ValueError(f"{prefix}存在非法uplift得分。")
        np.testing.assert_allclose(
            scores,
            predictions[f"{prefix}_p_treatment"] - predictions[f"{prefix}_p_control"],
            rtol=1e-10, atol=1e-12,
        )
    return test, predictions, digest.hexdigest()


def calculate_uplift_metrics(test, predictions):
    """Qini、AUUC均使用sklift的归一化定义，允许负指标，不裁剪。

    Qini系数=(实际Qini面积-随机线面积)/(理想Qini面积-随机线面积)。
    AUUC采用uplift_auc_score的对应定义，不是原始曲线面积。
    negative_effect=True允许理想Qini曲线包含负效应部分。
    """
    import pandas as pd
    from sklift.metrics import qini_auc_score, uplift_auc_score

    # 阶段1为节省内存将标签存为uint8；sklift理想Qini排序包含1-0等
    # 有符号减法。必须先转int64，避免0-1在uint8下溢为255，污染归一化分母。
    y = test["visit"].to_numpy(dtype="int64")
    treatment = test["treatment"].to_numpy(dtype="int64")
    rows = []
    for name, prefix in [("S-learner", "s_learner"), ("T-learner", "t_learner")]:
        scores = predictions[f"{prefix}_uplift"].to_numpy()
        qini = float(qini_auc_score(y, scores, treatment, negative_effect=True))
        auuc = float(uplift_auc_score(y, scores, treatment))
        if not np.isfinite([qini, auuc]).all():
            raise ValueError("归一化指标非有限值，请检查理想曲线分母是否为0。")
        rows.append({"model": name, "qini_coefficient": qini, "auuc_normalized": auuc})
    return pd.DataFrame(rows)


def evaluate_baseline_models(test, model_dir: Path):
    """加载阶段2模型，在同一测试集上重新核算AUC，不重新拟合。"""
    import joblib
    import pandas as pd

    X = test.loc[:, FEATURES]
    rows = []
    for name, filename in [("LR", "lr_baseline.joblib"), ("LightGBM", "lightgbm_baseline.joblib")]:
        model = joblib.load(model_dir / filename)
        if list(model.feature_names_in_) != FEATURES:
            raise ValueError(f"{name}保存模型的特征与预期不符。")
        positive_index = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
        score = binary_auc(test["visit"], model.predict_proba(X)[:, positive_index])
        rows.append({"model": name, "test_auc": score})
    result = pd.DataFrame(rows)
    # 核对阶段2记录，避免错误引用其他实验的基线结果。
    saved = pd.read_csv(model_dir / "baseline_metrics.csv").set_index("model")
    for row in rows:
        np.testing.assert_allclose(row["test_auc"], saved.loc[row["model"], "test_auc"], atol=1e-12, rtol=0)
    return result


def import_sklift_qini_plot():
    """兼容sklift 0.5.1引用已迁移的sklearn绘图支持检查函数。

    仅在导入sklift.viz时临时提供原函数的别名，随后恢复sklearn命名空间；
    不修改第三方安装文件，也不替换sklift的曲线、指标或绘图算法。
    """
    import sklearn.utils

    added_alias = not hasattr(sklearn.utils, "check_matplotlib_support")
    if added_alias:
        from sklearn.utils._optional_dependencies import check_matplotlib_support
        sklearn.utils.check_matplotlib_support = check_matplotlib_support
    try:
        from sklift.viz import plot_qini_curve
    finally:
        if added_alias:
            delattr(sklearn.utils, "check_matplotlib_support")
    return plot_qini_curve


def draw_qini_comparison(test, predictions, metrics, figure_dir: Path) -> Path:
    """sklift生成两条Qini曲线，统一为覆盖比例横轴，输出600dpi PNG与PDF。

    纵轴沿用sklift的处理组尺度：Y_t(k)-Y_c(k)*N_t(k)/N_c(k)，
    并非投放利润或对全部用户的增量人数；图不代表置信区间。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, ticker

    plot_qini_curve = import_sklift_qini_plot()
    # 与指标计算采用相同的有符号标签类型，确保图例Qini与汇总表一致。
    y = test["visit"].to_numpy(dtype="int64")
    treatment = test["treatment"].to_numpy(dtype="int64")
    installed = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]
    chinese = next((name for name in candidates if name in installed), None)
    if chinese is None:
        raise RuntimeError("未找到中文字体，请安装微软雅黑、黑体或Noto Sans CJK SC后重试。")
    figure_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "sans-serif", "font.sans-serif": [chinese, "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10.5,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "legend.fontsize": 10, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, ax = plt.subplots(figsize=(8.5, 5.8))
        styles = [("S-learner", "s_learner", "#0072B2", "-"),
                  ("T-learner", "t_learner", "#D55E00", "--")]
        endpoint = None
        for name, prefix, color, linestyle in styles:
            qini = metrics.set_index("model").loc[name, "qini_coefficient"]
            display = plot_qini_curve(
                y, predictions[f"{prefix}_uplift"], treatment,
                random=False, perfect=False, negative_effect=True, ax=ax,
                name=name, color=color, linestyle=linestyle, linewidth=1.7,
                label=f"{name}（Qini = {qini:.4f}）",
            )
            # 仅转换显示坐标，sklift指标仍按全量原始曲线计算。
            display.line_.set_xdata(display.x_actual / len(test))
            if endpoint is not None:
                np.testing.assert_allclose(display.y_actual[-1], endpoint)
            endpoint = float(display.y_actual[-1])
        ax.plot([0, 1], [0, endpoint], color="#666666", linestyle=":",
                linewidth=1.5, label="随机定向参考线")
        ax.relim()
        ax.autoscale_view()
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_xlabel("按预测 uplift 降序选取的测试样本比例")
        ax.set_ylabel("累计增量访问量（处理组尺度）")
        ax.set_title("S-learner 与 T-learner 的 Qini 曲线（测试集）", pad=12)
        ax.grid(axis="both", linestyle="--", alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.985), frameon=False)
        fig.text(0.12, 0.035,
                 f"注：visit 为结果变量，treatment 为 ITT 分组；测试样本数 n = {len(test):,}。\n"
                 "Qini 系数采用 sklift 归一化定义；图中曲线为点估计，未展示置信区间。",
                 fontsize=9, va="bottom")
        fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.21)
        path = figure_dir / "qini_curve.png"
        fig.savefig(path, dpi=600, facecolor="white")
        fig.savefig(figure_dir / "qini_curve.pdf", facecolor="white")
        plt.close(fig)
    print(f"绘图字体：{chinese}；已保存：{path}", flush=True)
    return path


def run_evaluation(root: Path = ROOT):
    """阶段4入口：严格对齐、指标计算、基线复核、汇总存储、论文风格绘图。"""
    import pandas as pd
    import sklift

    root = Path(root)
    test, predictions, test_digest = load_aligned_test(root)
    print(f"测试集及预测对齐检查通过：{len(test):,}条记录。", flush=True)
    uplift_metrics = calculate_uplift_metrics(test, predictions)
    baseline_metrics = evaluate_baseline_models(test, root / "model")
    baseline_metrics["model_type"] = "传统预测"
    uplift_metrics["model_type"] = "uplift"
    summary = pd.concat([baseline_metrics, uplift_metrics], ignore_index=True)
    summary = summary[["model", "model_type", "test_auc", "qini_coefficient", "auuc_normalized"]]
    result_dir = root / "result"
    uplift_metrics.to_csv(result_dir / "uplift_metrics.csv", index=False, encoding="utf-8-sig")
    # 不适用的指标保存为空值，打印时用破折号，不能填写为0或横向比较。
    summary.to_csv(result_dir / "model_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    print("\n测试集汇总指标：", flush=True)
    print(summary.to_string(index=False, na_rep="—", float_format=lambda value: f"{value:.6f}"), flush=True)
    figure_path = draw_qini_comparison(test, predictions, uplift_metrics, root / "figure")
    metadata = {
        "target": "visit", "estimand": "ITT", "test_rows": len(test),
        "test_file_sha256": test_digest, "sklift_version": sklift.__version__,
        "qini_function": "sklift.metrics.qini_auc_score(negative_effect=True)",
        "auuc_function": "sklift.metrics.uplift_auc_score",
        "normalization": "(actual area - random area) / (perfect area - random area)",
        "metric_label_dtype": "int64 (prevents uint8 subtraction underflow in sklift)",
        "baseline_auc_recomputed": True, "confidence_intervals_computed": False,
        "figure_path": str(figure_path), "png_dpi": 600,
    }
    (result_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("\n阶段4完成。AUC与uplift指标分属不同任务；本阶段未计算ROI。", flush=True)
    return summary


# ==================== 阶段5：业务ROI敏感性模拟 ====================


def estimate_policy_outcomes(population, target_count: int) -> dict:
    """用RCT分组均值差估计对目标人群全部分配treatment时的ITT增量。

    population为Top-K子群；随机策略则使用完整测试人群估计期望率。
    不能对exposure=1的人再做visit比较：实际曝光是处理后变量。
    exposure为二元指标，不包含曝光频次，本模拟假设每个exposure=1
    对应一次计费曝光；若实际有重复曝光，应另乘真实平均计费次数。
    """
    treated = population.loc[population["treatment"] == 1]
    control = population.loc[population["treatment"] == 0]
    if min(len(treated), len(control)) < 2 or target_count <= 0:
        raise ValueError("目标规模必须为正，且估计人群的处理/对照组均需至少2条记录。")
    visit_t, visit_c = float(treated["visit"].mean()), float(control["visit"].mean())
    exposure_t = float(treated["exposure"].mean())
    if exposure_t <= 0:
        raise ValueError("估计曝光率为0，无法定义按曝光成本计算的ROI。")
    if control["exposure"].sum() != 0:
        raise ValueError("发现对照组曝光，需重新定义无投放反事实和增量成本。")
    effect = visit_t - visit_c
    return {
        "target_count": int(target_count), "estimation_rows": len(population),
        "observed_treated": len(treated), "observed_control": len(control),
        "visit_rate_treated": visit_t, "visit_rate_control": visit_c,
        "itt_visit_rate_difference": effect, "exposure_rate_if_treated": exposure_t,
        "expected_incremental_visits": target_count * effect,
        "expected_billed_exposures": target_count * exposure_t,
        "incremental_visits_per_exposure": effect / exposure_t,
    }


def build_roi_scenarios(
    test, predictions, top_k=(0.05, 0.10, 0.20, 0.30),
    visit_values=(5.0, 10.0, 15.0), exposure_costs=(0.5, 1.0, 2.0),
):
    """3种策略×4个K×9组价值/成本，共108行；比较相同目标人数。

    ROI=(增量访问价值-曝光成本)/曝光成本，保留负增量及负ROI。
    随机策略为全体测试人群均值的解析期望，而非一次随机抽样的结果。
    两种策略曝光率不同，等人数不等于等预算；不直接相减实际visit总数。
    """
    import pandas as pd

    if len(test) != len(predictions):
        raise ValueError("测试集与预测长度不一致。")
    if any(not 0 < k <= 1 for k in top_k):
        raise ValueError("Top-K比例必须在(0, 1]之间。")
    if any(v <= 0 for v in visit_values) or any(c <= 0 for c in exposure_costs):
        raise ValueError("访问价值与曝光成本必须为正。")
    # 固定排序及并列分数处理：稳定排序，分数并列时保持测试原始行序。
    orders = {
        strategy: np.argsort(-predictions[column].to_numpy(), kind="stable")
        for strategy, column in [
            ("S-learner", "s_learner_uplift"), ("T-learner", "t_learner_uplift"),
        ]
    }
    records = []
    for k in top_k:
        n_target = int(np.floor(len(test) * k))
        populations = {"Random": test}
        populations.update({name: test.iloc[order[:n_target]] for name, order in orders.items()})
        for strategy, population in populations.items():
            outcome = estimate_policy_outcomes(population, n_target)
            for value in visit_values:
                for cost in exposure_costs:
                    benefit = outcome["expected_incremental_visits"] * value
                    spend = outcome["expected_billed_exposures"] * cost
                    records.append({
                        "strategy": strategy, "top_k": k, "visit_value": value,
                        "exposure_cost": cost, **outcome,
                        "incremental_visit_value": benefit, "advertising_cost": spend,
                        "net_incremental_value": benefit - spend,
                        "roi": (benefit - spend) / spend,
                        "break_even_visit_value": spend / outcome["expected_incremental_visits"]
                        if outcome["expected_incremental_visits"] > 0 else np.nan,
                    })
    return pd.DataFrame(records)


def compare_roi_with_random(scenarios):
    """生成72行定向策略与随机策略的匹配对比；ROI差值以比例单位保存。"""
    keys = ["top_k", "visit_value", "exposure_cost"]
    fields = ["roi", "expected_incremental_visits", "expected_billed_exposures",
              "advertising_cost", "net_incremental_value"]
    random = scenarios.loc[scenarios["strategy"] == "Random", keys + fields]
    targeted = scenarios.loc[scenarios["strategy"] != "Random"].copy()
    matched = targeted.merge(random, on=keys, suffixes=("", "_random"), validate="many_to_one")
    matched["roi_difference_vs_random"] = matched["roi"] - matched["roi_random"]
    matched["net_value_difference_vs_random"] = (
        matched["net_incremental_value"] - matched["net_incremental_value_random"]
    )
    return matched


def draw_roi_sensitivity(scenarios, figure_dir: Path) -> Path:
    """9个分面展示全部价值/成本组合，纵轴为ROI百分比，包含零收益线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, ticker

    fonts = {font.name for font in font_manager.fontManager.ttflist}
    chinese = next((f for f in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"] if f in fonts), None)
    if chinese is None:
        raise RuntimeError("未找到可用中文字体。")
    figure_dir.mkdir(parents=True, exist_ok=True)
    values = sorted(scenarios["visit_value"].unique())
    costs = sorted(scenarios["exposure_cost"].unique())
    styles = [("S-learner", "#0072B2", "-", "o"),
              ("T-learner", "#D55E00", "--", "s"),
              ("Random", "#666666", ":", "^")]
    with plt.rc_context({
        "font.family": "sans-serif", "font.sans-serif": [chinese, "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 9,
        "axes.titlesize": 10, "axes.linewidth": 0.7, "pdf.fonttype": 42,
    }):
        fig, axes = plt.subplots(len(values), len(costs), figsize=(11, 8.5), squeeze=False)
        for row, value in enumerate(values):
            for col, cost in enumerate(costs):
                ax = axes[row, col]
                subset = scenarios[(scenarios["visit_value"] == value) & (scenarios["exposure_cost"] == cost)]
                for strategy, color, style, marker in styles:
                    data = subset[subset["strategy"] == strategy].sort_values("top_k")
                    ax.plot(data["top_k"], data["roi"], color=color, linestyle=style,
                            marker=marker, markersize=4, linewidth=1.5,
                            label="随机投放（期望）" if strategy == "Random" else f"{strategy} 定向")
                ax.axhline(0, color="#999999", linewidth=0.8, alpha=0.8)
                ax.set_title(f"访问价值 v={value:g}；曝光成本 c={cost:g}", pad=7)
                ax.set_xticks([.05, .1, .2, .3])
                ax.xaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0))
                ax.yaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
                ax.grid(linestyle="--", alpha=0.22, linewidth=0.6)
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_ylabel("增量 ROI")
                if row == len(values) - 1:
                    ax.set_xlabel("目标人群比例 Top-K")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.suptitle("广告投放增量 ROI 敏感性分析：Uplift 定向与随机策略", fontsize=14, y=0.975)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
        fig.text(.07, .025,
                 "注：v、c 使用相同货币单位；比较相同目标人数，非相同预算。随机策略为全体人群的期望结果。\n"
                 "每个曝光阳性样本假设计费一次；增量访问按 treatment 的 ITT 均值差估计。各分面纵轴刻度独立。\n"
                 "ROI 为假设价值与成本下的点估计，未计其他运营成本，未展示置信区间。", fontsize=8.5, va="bottom")
        fig.subplots_adjust(left=.075, right=.98, top=.85, bottom=.15, wspace=.35, hspace=.4)
        path = figure_dir / "roi_sensitivity.png"
        fig.savefig(path, dpi=600, facecolor="white")
        fig.savefig(figure_dir / "roi_sensitivity.pdf", facecolor="white")
        plt.close(fig)
    return path


def write_roi_report(scenarios, comparisons, result_dir: Path) -> Path:
    """生成可直接改写为论文业务分析小节的Markdown，保留完整CSV供排版。"""
    reference = scenarios[(scenarios["visit_value"] == 10) & (scenarios["exposure_cost"] == 1)]
    lines = [
        "# 广告投放增量ROI敏感性分析", "",
        "本分析以visit为结果变量，以treatment为ITT处理定义。按模型预测uplift降序选择Top-K人群。",
        "随机策略在相同人数下使用全体测试人群的期望访问率差及曝光率，不额外抽取一次随机样本。", "",
        "## 估计与收益口径", "",
        "记目标人群人数为N，处理/对照组visit均值分别为μ₁、μ₀，处理组exposure均值为e₁。",
        "预期增量访问=N×(μ₁−μ₀)；预期计费曝光=N×e₁；模拟曝光成本=N×e₁×c。",
        "ROI=[N×(μ₁−μ₀)×v−N×e₁×c]/(N×e₁×c)。",
        "这里没有把观测分组的85:15人数差异当成收益，组内均值差已按人数归一化。", "",
        "exposure仅为二元变量，缺少曝光次数。假设每个曝光阳性样本计费一次；如实际平均计费m次，成本需乘m。",
        "本分析未按exposure筛选样本，不估计实际曝光者的因果效应。比较为等目标人数，非等预算。",
        "假设处理机制和样本可推广到模拟投放；visit价值为情景参数，并非数据集实测收入。", "",
        "## 参考情景：单次增量访问价值10、单次曝光成本1", "",
        "| 策略 | Top-K | 目标人数 | 预期增量访问 | 预期计费曝光 | 增量访问价值 | 曝光成本 | 净增量价值 | ROI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in reference.itertuples(index=False):
        strategy = "随机投放（期望）" if row.strategy == "Random" else row.strategy
        lines.append(f"| {strategy} | {row.top_k:.0%} | {row.target_count:,} | "
                     f"{row.expected_incremental_visits:,.2f} | {row.expected_billed_exposures:,.2f} | "
                     f"{row.incremental_visit_value:,.2f} | {row.advertising_cost:,.2f} | "
                     f"{row.net_incremental_value:,.2f} | {row.roi:.2%} |")
    ref_lookup = reference.set_index(["strategy", "top_k"])
    s5 = ref_lookup.loc[("S-learner", .05)]
    t5 = ref_lookup.loc[("T-learner", .05)]
    r5 = ref_lookup.loc[("Random", .05)]
    lines += ["", "## 当前测试集的情景结果", "",
              f"在v=10、c=1、Top-5%的情景下，S-learner、T-learner与随机策略的ROI分别为"
              f"{s5.roi:.2%}、{t5.roi:.2%}与{r5.roi:.2%}。",
              f"S-learner预期增量访问为{s5.expected_incremental_visits:,.2f}，高于T-learner的"
              f"{t5.expected_incremental_visits:,.2f}，但预期计费曝光也更高"
              f"（{s5.expected_billed_exposures:,.2f}对{t5.expected_billed_exposures:,.2f}），因此ROI反而较低。",
              "本例说明，增量访问规模、净增量价值与ROI是不同业务目标，不能仅依据Qini或增量访问总量判断成本效率。",
              "同一K下采用全局统一的v、c时，ROI排名取决于单位预期曝光的增量访问；9种参数组合不构成9次独立效果验证。", "",
              "## 解释边界", "",
              "在当前测试集和固定模型下，Qini或AUUC较高不必然意味着ROI较高；目标人群的曝光概率也影响成本。",
              "同一人群的ROI随v提高而增大、随c提高而减小。随机策略的期望ROI不随K变化，但收益总量与成本均随K变化。",
              "所有数值为点估计，未进行显著性检验，不构成已验证的真实盈利。固定K用于报告，不根据测试收益重新训练模型。",
              "visit正例稀疏及85:15组别不均衡会影响子群估计精度；conversion未用于本模拟，不能将访问价值解释为实测转化收入。", "",
              f"完整情景表共{len(scenarios)}行：roi_sensitivity_results.csv。",
              f"定向与随机匹配对比共{len(comparisons)}行：roi_strategy_comparison.csv。",
              "论文图题建议：不同访问价值与曝光成本下的广告投放增量ROI敏感性分析。", ""]
    path = result_dir / "roi_business_analysis.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_roi_analysis(root: Path = ROOT):
    """阶段5入口，只生成业务模拟，不重新训练或重算阶段4图表。"""
    import pandas as pd

    root = Path(root)
    test, predictions, digest = load_aligned_test(root)
    exposure = pd.read_parquet(root / "data" / "test.parquet", columns=["exposure"])
    if not exposure.index.equals(test.index) or not exposure["exposure"].isin([0, 1]).all():
        raise ValueError("曝光数据行索引或二元取值异常。")
    test["exposure"] = exposure["exposure"]
    scenarios = build_roi_scenarios(test, predictions)
    comparisons = compare_roi_with_random(scenarios)
    result_dir = root / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(result_dir / "roi_sensitivity_results.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(result_dir / "roi_strategy_comparison.csv", index=False, encoding="utf-8-sig")
    figure = draw_roi_sensitivity(scenarios, root / "figure")
    report = write_roi_report(scenarios, comparisons, result_dir)
    metadata = {
        "target": "visit", "estimand": "ITT", "test_rows": len(test),
        "test_file_sha256": digest, "top_k": [.05, .1, .2, .3],
        "visit_values": [5, 10, 15], "exposure_costs": [.5, 1, 2],
        "currency_unit": "unspecified common currency unit for value and cost",
        "effect_estimator": "within-policy subgroup treatment/control visit mean difference",
        "random_strategy": "analytic expectation using full test population rates",
        "comparison_basis": "equal target population size, not equal budget",
        "cost_model": "N_target * mean(exposure | treatment=1, subgroup) * unit_cost",
        "billed_exposures_per_exposed_sample_assumption": 1,
        "roi_formula": "(incremental_visits * visit_value - advertising_cost) / advertising_cost",
        "confidence_intervals_computed": False, "other_operating_costs_included": False,
        "conversion_used": False, "scenario_rows": len(scenarios),
        "comparison_rows": len(comparisons), "figure_path": str(figure), "png_dpi": 600,
    }
    (result_dir / "roi_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("\n全部ROI情景（比例按百分数打印）：", flush=True)
    output = scenarios[["strategy", "top_k", "visit_value", "exposure_cost", "roi"]].copy()
    output["top_k"] = output["top_k"].map(lambda x: f"{x:.0%}")
    output["roi"] = output["roi"].map(lambda x: f"{x:.2%}")
    print(output.to_string(index=False), flush=True)
    print(f"\n情景数={len(scenarios)}；匹配对比数={len(comparisons)}", flush=True)
    print(f"图片：{figure}\n论文业务分析表：{report}", flush=True)
    print("阶段5完成。以上为等人数、单次计费曝光假设下的模拟ROI，不是实测利润。", flush=True)
    return scenarios


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="阶段4模型评估 / 阶段5 ROI敏感性模拟")
    parser.add_argument("--stage", choices=["metrics", "roi"], default="metrics")
    arguments = parser.parse_args()
    if arguments.stage == "roi":
        run_roi_analysis()
    else:
        run_evaluation()
