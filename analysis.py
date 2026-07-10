import sys
import platform

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")

_font_map = {
    "Windows": ["SimHei", "Microsoft YaHei"],
    "Darwin": ["PingFang SC", "Heiti SC"],
    "Linux": ["WenQuanYi Micro Hei", "Noto Sans CJK SC"],
}
matplotlib.rcParams["font.sans-serif"] = _font_map.get(
    platform.system(), ["sans-serif"]
)
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path("gongkao")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_all_data():
    """加载全部国考和省考数据"""
    dfs = []

    guokao_path = DATA_DIR / "国考.csv"
    if guokao_path.exists():
        df_gk = pd.read_csv(guokao_path)
        df_gk["exam_type"] = "国考"
        dfs.append(df_gk)
        print(f"国考数据: {len(df_gk)} 条")
    else:
        print(f"警告: 未找到 {guokao_path}")

    shengkao_dir = DATA_DIR / "省考"
    if shengkao_dir.exists():
        for csv_file in shengkao_dir.glob("*.csv"):
            province_name = csv_file.stem
            df_sk = pd.read_csv(csv_file)
            df_sk["exam_type"] = "省考"
            dfs.append(df_sk)
        print(f"省考数据: {len(dfs) - 1} 个省份")
    else:
        print(f"警告: 未找到 {shengkao_dir}")

    if not dfs:
        print("错误: 没有加载到任何数据，请确保 gongkao/ 目录存在且包含 CSV 文件")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    print(f"总数据量: {len(df)} 条, {len(df.columns)} 列")
    return df


def overview(df):
    """数据概览"""
    print("=" * 60)
    print("一、数据概览")
    print("=" * 60)

    print(f"\n总记录数: {len(df):,}")
    print(f"年份范围: {int(df['year'].min())} - {int(df['year'].max())}")
    print(f"省份数量: {df['province'].nunique()}")

    print("\n【考试类型分布】")
    print(df["exam_type"].value_counts())

    print("\n【各年数据量】")
    print(df["year"].value_counts().sort_index())

    print("\n【缺失值统计】")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"缺失数": missing, "占比%": missing_pct})
    print(missing_df[missing_df["缺失数"] > 0])


def yearly_trend(df):
    """年度趋势分析"""
    print("\n" + "=" * 60)
    print("二、年度招录趋势分析")
    print("=" * 60)

    yearly = df.groupby(["year", "exam_type"])["employ_count"].sum().unstack(fill_value=0)
    yearly["合计"] = yearly.sum(axis=1)
    print("\n【各年招录人数】")
    print(yearly.to_string())

    # 各年报考人数
    yearly_enroll = df.groupby(["year", "exam_type"])["enrollment_count"].sum().unstack(fill_value=0)
    print("\n【各年报考人数】")
    print(yearly_enroll.to_string())

    # 平均竞争比
    df_valid = df[df["competition_ratio"].notna() & (df["competition_ratio"] > 0)]
    avg_comp = df_valid.groupby(["year", "exam_type"])["competition_ratio"].mean().unstack()
    print("\n【各年平均竞争比】")
    print(avg_comp.round(2).to_string())

    # 图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    yearly.plot(kind="bar", ax=axes[0])
    axes[0].set_title("各年度招录人数")
    axes[0].set_xlabel("年份")
    axes[0].set_ylabel("招录人数")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=0)

    avg_comp.plot(kind="bar", ax=axes[1])
    axes[1].set_title("各年度平均竞争比")
    axes[1].set_xlabel("年份")
    axes[1].set_ylabel("平均竞争比")
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "yearly_trend.png", dpi=150)
    plt.close()
    print("\n图表已保存: output/yearly_trend.png")


def province_analysis(df):
    """省份分析"""
    print("\n" + "=" * 60)
    print("三、省份分布分析")
    print("=" * 60)

    # 各省招录人数
    prov = df.groupby("province")["employ_count"].sum().sort_values(ascending=False)
    print(f"\n【招录人数 Top 10 省份】")
    print(prov.head(10).to_string())

    print(f"\n【招录人数 Bottom 10 省份】")
    print(prov.tail(10).to_string())

    # 图表
    fig, ax = plt.subplots(figsize=(14, 8))
    prov_sorted = prov.sort_values(ascending=True)
    colors = ["#2ecc71" if v == prov_sorted.max() else "#e74c3c" if v == prov_sorted.min() else "#3498db"
              for v in prov_sorted.values]
    ax.barh(prov_sorted.index.astype(str), prov_sorted.values, color=colors)
    ax.set_title("各省份招录总人数")
    ax.set_xlabel("招录人数")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "province_distribution.png", dpi=150)
    plt.close()
    print("\n图表已保存: output/province_distribution.png")


def degree_analysis(df):
    """学历要求分析"""
    print("\n" + "=" * 60)
    print("四、学历要求分析")
    print("=" * 60)

    degree_map = {
        "大专及以上": ["大专及以上", "大专以上", "专科及以上"],
        "仅限本科": ["仅限本科", "本科(仅限本科)"],
        "本科及以上": ["本科及以上", "本科以上", "本科或硕士研究生", "本科或硕士", "本科及以上学历"],
        "仅限硕士": ["仅限硕士", "仅限硕士研究生", "硕士研究生及以上仅限硕士"],
        "硕士及以上": ["硕士研究生及以上", "硕士及以上", "研究生（硕士）及以上"],
        "仅限博士": ["仅限博士", "仅限博士研究生"],
    }

    def classify_degree(d):
        if pd.isna(d):
            return "未知"
        d = str(d).strip()
        for cat, keywords in degree_map.items():
            for kw in keywords:
                if kw in d:
                    return cat
        # fallback
        if "硕士" in d:
            return "硕士及以上"
        if "本科" in d:
            return "本科及以上"
        if "大专" in d or "专科" in d:
            return "大专及以上"
        if "博士" in d:
            return "仅限博士"
        if "研究生" in d:
            return "硕士及以上"
        return "其他"

    df["degree_cat"] = df["degree_requirement"].apply(classify_degree)

    print("\n【学历要求分布】")
    degree_counts = df["degree_cat"].value_counts()
    degree_pct = (degree_counts / len(df) * 100).round(2)
    print(pd.DataFrame({"数量": degree_counts, "占比%": degree_pct}))

    # 图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 饼图
    top5 = degree_counts.head(5)
    other = degree_counts.iloc[5:].sum()
    if other > 0:
        pie_data = pd.concat([top5, pd.Series({"其他": other})])
    else:
        pie_data = top5
    axes[0].pie(pie_data.values, labels=pie_data.index, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("学历要求占比")

    # 各类型平均竞争比
    comp_by_degree = df[df["competition_ratio"].notna()].groupby("degree_cat")["competition_ratio"].mean().sort_values(ascending=True)
    comp_by_degree.plot(kind="barh", ax=axes[1], color="#e67e22")
    axes[1].set_title("各学历要求平均竞争比")
    axes[1].set_xlabel("平均竞争比")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "degree_analysis.png", dpi=150)
    plt.close()
    print("\n图表已保存: output/degree_analysis.png")


def department_analysis(df):
    """部门系统分析"""
    print("\n" + "=" * 60)
    print("五、招录部门系统分析")
    print("=" * 60)

    # Top 部门系统
    sys_counts = df["department_system"].value_counts()
    print(f"\n【Top 15 招录部门系统】")
    print(sys_counts.head(15).to_string())

    # Top 部门系统招录人数
    sys_employ = df.groupby("department_system")["employ_count"].sum().sort_values(ascending=False)
    print(f"\n【Top 10 部门系统招录人数】")
    print(sys_employ.head(10).to_string())

    # 各部门系统平均竞争比
    sys_comp = df[df["competition_ratio"].notna()].groupby("department_system")["competition_ratio"].agg(["mean", "count"])
    sys_comp = sys_comp[sys_comp["count"] >= 10].sort_values("mean", ascending=False)
    print(f"\n【竞争最激烈的部门系统 Top 10】")
    print(sys_comp.head(10).round(2).to_string())

    # 图表
    fig, ax = plt.subplots(figsize=(14, 8))
    top_sys = sys_employ.head(15)
    ax.barh(top_sys.index.astype(str)[::-1], top_sys.values[::-1], color="#9b59b6")
    ax.set_title("招录人数 Top 15 部门系统")
    ax.set_xlabel("招录人数")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "department_system.png", dpi=150)
    plt.close()
    print("\n图表已保存: output/department_system.png")


def competition_analysis(df):
    """竞争分析"""
    print("\n" + "=" * 60)
    print("六、竞争比分析")
    print("=" * 60)

    df_comp = df[df["competition_ratio"].notna() & (df["competition_ratio"] > 0)].copy()

    print(f"\n有竞争比数据的岗位: {len(df_comp):,} / {len(df):,}")

    print("\n【竞争比统计】")
    stats = df_comp["competition_ratio"].describe()
    print(stats.to_string())

    print(f"\n竞争比分布:")
    bins = [0, 3, 10, 30, 50, 100, 200, float("inf")]
    labels = ["<3:1", "3-10:1", "10-30:1", "30-50:1", "50-100:1", "100-200:1", ">200:1"]
    df_comp["comp_range"] = pd.cut(df_comp["competition_ratio"], bins=bins, labels=labels, right=True)
    print(df_comp["comp_range"].value_counts().sort_index())

    # 图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df_comp["competition_ratio"].clip(upper=500).hist(bins=50, ax=axes[0], color="#3498db", edgecolor="white")
    axes[0].axvline(df_comp["competition_ratio"].median(), color="red", linestyle="--", label=f'中位数: {df_comp["competition_ratio"].median():.1f}')
    axes[0].set_title("竞争比分布 (截断至500)")
    axes[0].set_xlabel("竞争比")
    axes[0].legend()

    comp_range = df_comp["comp_range"].value_counts().sort_index()
    comp_range.plot(kind="bar", ax=axes[1], color="#e74c3c")
    axes[1].set_title("竞争比区间分布")
    axes[1].set_xlabel("竞争比区间")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "competition_analysis.png", dpi=150)
    plt.close()
    print("\n图表已保存: output/competition_analysis.png")


def main():
    print("公务员考试数据分析")
    print("=" * 60)

    df = load_all_data()

    overview(df)
    yearly_trend(df)
    province_analysis(df)
    degree_analysis(df)
    department_analysis(df)
    competition_analysis(df)

    print("\n" + "=" * 60)
    print("分析完成！结果保存在 output/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    main()
