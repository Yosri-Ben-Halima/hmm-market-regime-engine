"""Institutional-grade tear sheet renderer."""

from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.stats import norm
from scipy import stats

from ..config import REGIME_LABELS, REGIME_COLORS, BG, GRID, TXT, BLUE
from ..models.hmm import transition_matrix

import logging

log = logging.getLogger(__name__)


def _ax(ax, title="", xl="", yl="", leg=True, fs=7):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TXT, labelsize=fs)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, lw=0.35, alpha=0.7)
    if title:
        ax.set_title(title, color=TXT, fontsize=fs + 1, fontweight="bold", pad=5)
    if xl:
        ax.set_xlabel(xl, color="#6b7280", fontsize=fs)
    if yl:
        ax.set_ylabel(yl, color="#6b7280", fontsize=fs)
    if leg:
        ax.legend(fontsize=fs - 1, facecolor="#111827", edgecolor=GRID, labelcolor=TXT)


def _shade(ax, df):
    grp = df["regime"].ne(df["regime"].shift()).cumsum()
    for _, g in df.groupby(grp):
        k = int(g["regime"].iloc[0])
        ax.axvspan(
            g.index[0], g.index[-1], alpha=0.10, color=REGIME_COLORS[k], linewidth=0
        )


def build_tearsheet(df, fc, rm, ticker, out="tearsheet.png"):
    log.info("Rendering tear sheet…")
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "text.color": TXT,
            "font.family": "monospace",
        }
    )
    fig = plt.figure(figsize=(22, 26), facecolor=BG)
    fig.suptitle(
        f"  S&P 500 ({ticker}) │ Regime Forecasting Engine │ {datetime.today():%Y-%m-%d}",
        color=TXT,
        fontsize=13,
        fontweight="bold",
        x=0.01,
        ha="left",
        y=0.997,
    )
    gs = gridspec.GridSpec(
        6,
        4,
        figure=fig,
        hspace=0.55,
        wspace=0.35,
        top=0.975,
        bottom=0.03,
        left=0.05,
        right=0.98,
    )

    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor(BG)
    ax0.axis("off")
    cur = fc["current_regime"]
    cp = df[["p_bull", "p_bear", "p_crisis"]].iloc[-1].values
    kpis = [
        ("Last Close", f"${fc['current_price']:.2f}", TXT),
        (
            "Ann. Return",
            f"{rm['mu'] * 100:.1f}%",
            "#22c55e" if rm["mu"] > 0 else "#ef4444",
        ),
        ("Ann. Vol", f"{rm['sigma'] * 100:.1f}%", "#60a5fa"),
        ("Sharpe", f"{rm['sharpe']:.2f}", "#22c55e" if rm["sharpe"] > 0 else "#ef4444"),
        (
            "Sortino",
            f"{rm['sortino']:.2f}",
            "#22c55e" if rm["sortino"] > 0 else "#ef4444",
        ),
        ("Max DD", f"{rm['max_dd']:.1f}%", "#ef4444"),
        ("VaR 95%", f"{rm['var95'] * 100:.2f}%", "#f97316"),
        ("CVaR 95%", f"{rm['cvar95'] * 100:.2f}%", "#dc2626"),
        ("Skewness", f"{rm['skewness']:.3f}", "#a78bfa"),
        ("Ex. Kurtosis", f"{rm['kurtosis']:.2f}", "#fbbf24"),
        ("Active Regime", REGIME_LABELS[cur], REGIME_COLORS[cur]),
        ("Confidence", f"{cp[cur] * 100:.1f}%", REGIME_COLORS[cur]),
    ]
    N = len(kpis)
    for i, (lbl, val, col) in enumerate(kpis):
        x = i / N
        ax0.text(
            x + 0.002,
            0.75,
            lbl,
            transform=ax0.transAxes,
            color="#6b7280",
            fontsize=7,
            va="top",
        )
        ax0.text(
            x + 0.002,
            0.20,
            val,
            transform=ax0.transAxes,
            color=col,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        if i < N - 1:
            ax0.axvline((i + 1) / N - 0.002, color=GRID, lw=0.5, ymin=0.05, ymax=0.95)

    ax1 = fig.add_subplot(gs[1, :])
    _shade(ax1, df)
    ax1.plot(df.index, df["close"], color=BLUE, lw=1.2, label="Close")
    ax1.plot(df.index, df["ema_50"], color="#fbbf24", lw=0.7, ls="--", label="EMA50")
    ax1.plot(df.index, df["ema_200"], color="#f472b6", lw=0.7, ls="--", label="EMA200")
    handles = [
        mpatches.Patch(color=REGIME_COLORS[k], alpha=0.4, label=REGIME_LABELS[k])
        for k in range(3)
    ]
    handles += ax1.get_legend_handles_labels()[0]
    _ax(
        ax1,
        title=f"{ticker}  Close · EMA Overlay · Regime Classification",
        yl="Price ($)",
        leg=False,
    )
    ax1.legend(
        handles=handles,
        fontsize=6,
        facecolor="#111827",
        edgecolor=GRID,
        labelcolor=TXT,
        ncol=6,
    )

    ax2 = fig.add_subplot(gs[2, :2])
    ax2.stackplot(
        df.index,
        df["p_bull"],
        df["p_bear"],
        df["p_crisis"],
        colors=["#22c55e", "#facc15", "#ef4444"],
        alpha=0.75,
        labels=["Bull", "Transition", "Crisis"],
    )
    _ax(ax2, title="Regime Posterior Probabilities  (HMM)", yl="Probability")

    ax3 = fig.add_subplot(gs[2, 2:])
    if "ewma_vol_daily" in df.columns:
        ewma_ann = df["ewma_vol_daily"] * np.sqrt(252) * 100
        ax3.plot(df.index, ewma_ann, color="#f97316", lw=1.1, label="EWMA Vol (λ=0.94)")
    ax3.fill_between(df.index, df["rvol_20"], alpha=0.18, color="#60a5fa")
    ax3.plot(
        df.index, df["rvol_20"], color="#60a5fa", lw=0.8, ls="--", label="20D RVol"
    )
    ax3.plot(df.index, df["rvol_60"], color="#a78bfa", lw=0.6, ls=":", label="60D RVol")
    _ax(ax3, title="Volatility: EWMA (λ=0.94) vs Realized (Ann. %)", yl="%")

    ax8 = fig.add_subplot(gs[3, :2])
    bands = fc["bands"]
    days = np.arange(fc["horizon"] + 1)
    col_r = REGIME_COLORS[cur]
    ax8.fill_between(days, bands[5], bands[95], alpha=0.10, color=col_r, label="5–95%")
    ax8.fill_between(
        days, bands[10], bands[90], alpha=0.15, color=col_r, label="10–90%"
    )
    ax8.fill_between(
        days, bands[25], bands[75], alpha=0.22, color=col_r, label="25–75%"
    )
    ax8.plot(days, bands[50], color=col_r, lw=2.0, label="Median")
    ax8.plot(days, bands["mean"], color=BLUE, lw=1.2, ls="--", label="Mean")
    ax8.axhline(fc["current_price"], color=GRID, lw=0.5, ls=":")
    _ax(
        ax8,
        title=f"21D Monte Carlo ({fc['n_paths']:,} paths)  │  Vol: EWMA+t  │  Regime: {REGIME_LABELS[cur]}",
        xl="Trading Days Ahead",
        yl="Price ($)",
    )

    ax9 = fig.add_subplot(gs[3, 2])
    ro = fc["regime_occ"]
    ax9.stackplot(
        days,
        ro[:, 0],
        ro[:, 1],
        ro[:, 2],
        colors=["#22c55e", "#facc15", "#ef4444"],
        alpha=0.80,
        labels=["Bull", "Transition", "Crisis"],
    )
    _ax(ax9, title="Forecast Regime Occupancy", xl="Days Ahead")

    ax10 = fig.add_subplot(gs[3, 3])
    tr = fc["terminal_rets"] * 100
    ax10.hist(tr, bins=60, density=True, color=col_r, alpha=0.65, edgecolor="none")
    xf = np.linspace(tr.min(), tr.max(), 200)
    ax10.plot(xf, norm.pdf(xf, tr.mean(), tr.std()), color=BLUE, lw=1.2, label="Normal")
    ax10.axvline(
        np.percentile(tr, 5), color="#ef4444", lw=0.9, ls="--", label="5th pct"
    )
    ax10.axvline(
        np.percentile(tr, 95), color="#22c55e", lw=0.9, ls="--", label="95th pct"
    )
    ax10.axvline(0, color=GRID, lw=0.5)
    _ax(ax10, title="21D Terminal Return Dist.", xl="Return (%)")

    ax11 = fig.add_subplot(gs[4, 0])
    r = df["log_ret"].values * 100
    ax11.hist(r, bins=80, density=True, color=BLUE, alpha=0.45, edgecolor="none")
    xr = np.linspace(r.min(), r.max(), 200)
    ax11.plot(
        xr, norm.pdf(xr, r.mean(), r.std()), color="#fbbf24", lw=1.0, label="Normal"
    )
    ax11.axvline(rm["var95"] * 100, color="#ef4444", lw=0.8, ls="--", label="VaR 95%")
    _ax(ax11, title="Daily Return Distribution", xl="Log-Return (%)")

    ax12 = fig.add_subplot(gs[4, 1])
    (osm, osr), (slope, intercept, _) = stats.probplot(
        df["log_ret"].values, dist="norm"
    )
    ax12.scatter(osm, osr, s=1.5, color=BLUE, alpha=0.4)
    ax12.plot(osm, slope * np.array(osm) + intercept, color="#fbbf24", lw=1)
    _ax(
        ax12,
        title="Normal Q-Q Plot",
        xl="Theoretical Quantiles",
        yl="Sample Quantiles",
        leg=False,
    )

    ax13 = fig.add_subplot(gs[4, 2])
    for k in range(3):
        m = df["regime"] == k
        ax13.scatter(
            df.loc[m, "rvol_20"],
            df.loc[m, "log_ret"] * 100,
            s=2,
            color=REGIME_COLORS[k],
            alpha=0.3,
            label=REGIME_LABELS[k],
        )
    _ax(
        ax13,
        title="Regime Scatter: Vol vs Return",
        xl="20D RVol (%)",
        yl="Daily Return (%)",
    )

    ax14 = fig.add_subplot(gs[4, 3])
    T = transition_matrix(df["regime"].values)
    im = ax14.imshow(T, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(3):
        for j in range(3):
            ax14.text(
                j,
                i,
                f"{T[i, j]:.3f}",
                ha="center",
                va="center",
                color="#f9fafb",
                fontsize=8,
                fontweight="bold",
            )
    ax14.set_xticks(range(3))
    ax14.set_yticks(range(3))
    ax14.set_xticklabels([REGIME_LABELS[k] for k in range(3)], fontsize=6, color=TXT)
    ax14.set_yticklabels([REGIME_LABELS[k] for k in range(3)], fontsize=6, color=TXT)
    ax14.set_title(
        "Empirical Transition Matrix", color=TXT, fontsize=8, fontweight="bold", pad=5
    )
    for sp in ax14.spines.values():
        sp.set_color(GRID)
    plt.colorbar(im, ax=ax14, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(
        color=TXT, labelcolor=TXT
    )

    ax15 = fig.add_subplot(gs[5, :])
    ax15.axis("off")
    rows = []
    for k in range(3):
        m = df["regime"] == k
        r_ = df.loc[m, "log_ret"]
        rows.append(
            [
                REGIME_LABELS[k],
                f"{m.sum() / len(df) * 100:.1f}%",
                f"{r_.mean() * 252 * 100:+.2f}%",
                f"{r_.std() * np.sqrt(252) * 100:.2f}%",
                f"{(r_.mean() * 252) / (r_.std() * np.sqrt(252) + 1e-10):.2f}",
                f"{np.percentile(r_, 5) * 100:.2f}%",
                f"{cp[k] * 100:.1f}%",
            ]
        )
    cols = [
        "Regime",
        "Frequency",
        "Ann.Return",
        "Ann.Vol",
        "Sharpe",
        "VaR 95%",
        "P(Current)",
    ]
    tbl = ax15.table(
        cellText=rows, colLabels=cols, cellLoc="center", loc="center", bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r_, c_), cell in tbl.get_celld().items():
        cell.set_facecolor("#0d1117" if r_ > 0 else "#1f2937")
        cell.set_edgecolor(GRID)
        cell.set_text_props(color=TXT if r_ > 0 else "#9ca3af")
        if r_ > 0:
            cell.set_facecolor(REGIME_COLORS[r_ - 1] + "1a")
    ax15.set_title(
        "Regime Performance Summary", color=TXT, fontsize=8, fontweight="bold", pad=4
    )

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    log.info(f"  → Saved: {out}")
    plt.close(fig)
    return out
