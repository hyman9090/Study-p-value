"""現實場景學 p-value：班級同學身高

場景：
    香港 18 歲男生平均身高據報約 168 cm。
    你量咗你班 20 個男同學嘅身高，想知：
    「呢班同學嘅平均身高，同全港平均 168 cm 有冇顯著分別？」

假設：
    H0（虛無假設）：班級平均身高 = 168 cm（同全港一樣，差異只係抽樣誤差）
    H1（對立假設）：班級平均身高 ≠ 168 cm

方法：one-sample t-test（雙尾），顯著水平 α = 0.05
p-value 意思：假設 H0 係真，見到而家咁極端（或更極端）數據嘅機率。
"""

import sys
from math import sqrt
from statistics import mean, stdev

from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")  # Windows console 預設 cp950，轉 UTF-8 先印到廣東話字

# 你量返嚟嘅數據：20 個同學嘅身高（cm）
HEIGHTS = [
    168, 172, 175, 169, 171,
    170, 174, 167, 173, 176,
    168, 171, 169, 172, 170,
    173, 168, 174, 171, 170,
]

ALPHA = 0.05  # 顯著水平


def manual_t_test(data: list[float], mu0: float) -> tuple[float, float, int]:
    """手算 t 統計量同 p-value，逐步顯示點樣計出嚟。"""
    n = len(data)
    x_bar = mean(data)          # 樣本平均值
    s = stdev(data)             # 樣本標準差（除以 n-1）
    se = s / sqrt(n)            # 標準誤差 SE = s / √n
    t_stat = (x_bar - mu0) / se  # t = (x̄ - μ0) / SE
    df = n - 1                  # 自由度

    # p-value：t 分佈下，|t| 咁大或更大嘅機率（雙尾，所以乘 2）
    p_value = 2 * stats.t.sf(abs(t_stat), df)

    print(f"  n（樣本數）        = {n}")
    print(f"  x̄（樣本平均）      = {x_bar:.2f} cm")
    print(f"  s（樣本標準差）    = {s:.3f} cm")
    print(f"  SE（標準誤差）     = s/√n = {s:.3f}/√{n} = {se:.4f}")
    print(f"  t 統計量           = ({x_bar:.2f} - {mu0}) / {se:.4f} = {t_stat:.3f}")
    print(f"  df（自由度）       = {df}")
    print(f"  p-value（雙尾）    = {p_value:.6f}")
    return t_stat, p_value, df


def run_test(mu0: float) -> None:
    print(f"\n=== H0: 班級平均身高 = {mu0} cm ===")
    print("【手算步驟】")
    _, p_value, _ = manual_t_test(HEIGHTS, mu0)

    # scipy 內建函數驗證：答案要同手算一樣
    result = stats.ttest_1samp(HEIGHTS, mu0)
    print(f"【scipy 驗證】t = {result.statistic:.3f}, p = {result.pvalue:.6f}")

    print("【結論】")
    if p_value < ALPHA:
        print(f"  p = {p_value:.4f} < α = {ALPHA} → 拒絕 H0")
        print(f"  有顯著證據話班級平均身高唔係 {mu0} cm。")
    else:
        print(f"  p = {p_value:.4f} ≥ α = {ALPHA} → 唔能夠拒絕 H0")
        print(f"  冇足夠證據話班級平均身高同 {mu0} cm 有分別。")


def main() -> None:
    print("場景：20 個班級同學身高，同假設嘅全港平均比較")
    print(f"數據：{HEIGHTS}")

    # 同一數據，兩個唔同 H0，睇 p-value 點變
    run_test(168.0)  # 預期：拒絕 H0（同學明顯高啲）
    run_test(170.0)  # 預期：唔能夠拒絕 H0（差異可以係巧合）

    print("\n重點：p-value 細 → 數據同 H0 好唔夾 → 拒絕 H0")
    print("     p-value 大 → 差異可能只係抽樣巧合 → 唔拒絕 H0")


if __name__ == "__main__":
    main()
