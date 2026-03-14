# Buff-mini – گزارش کامل سیستم و عملکرد

## 1) هدف و دامنه
هدف سیستم: ساخت یک موتور تحقیقاتی قطعی و تکرارپذیر برای کشف و ارزیابی ستاپ‌های معاملاتی در داده تاریخی کریپتو، با تاکید بر هزینه‌ها، اعتبارسنجی چندمرحله‌ای، و جلوگیری از نتایج کاذب.

دامنه اجرای فعلی (طبق آخرین ران کامل):
- نماد اصلی: BTC/USDT
- تایم‌فریم‌های کشف: 15m, 30m, 1h, 2h, 4h
- تایم‌فریم‌های Promotion: 2 تایم‌فریم برتر
- تایم‌فریم نهایی اعتبارسنجی: 1 تایم‌فریم
- خانواده‌های ستاپ فعال در Stage‑52: structure_pullback_continuation, liquidity_sweep_reversal, squeeze_flow_breakout

## 2) داده‌ها و منابع
داده‌های اصلی:
- OHLCV با پوشش حدود 4 سال (1m)، داده موجود و sanity‑checked
- مشتقات (در حد رایگان):
  - funding
  - long/short ratio
  - taker flow
  - OI (short‑horizon / محدود)

منابع داده:
- Binance, Bybit, Deribit (اولویت Binance)

Snapshot:
- داده منجمد: DATA_FROZEN_v1
- مسیر snapshot: data/snapshots/DATA_FROZEN_v1.json

## 3) معماری و جریان کلی (End‑to‑End)
جریان سطح‌بالا:
1) Stage‑28: ساخت context پایه، ارزیابی اولیه، تولید خلاصه‌ها
2) Stage‑39: تولید اولیه کاندیدها (signal generation)
3) Stage‑47: Signal Genesis 2.0 (ستاپ‑محور)
4) Stage‑48: Tradability Learning v2 (gate قبل از replay)
5) Stage‑51..57: اعمال scope، گیت‌های tradability و اعتبارسنجی (replay/WF/MC/seed)
6) Stage‑60..72: قفل‌کردن chain، ارتقای ML/validation/uncertainty/search/replay و verdict کمپینی

## 4) الگوریتم‌ها و روش‌ها
تولید ستاپ (Setup‑based):
- هر کاندید شامل context/trigger/confirmation/invalidation + entry/stop/target/hold + RR model + cost proxy
- خانواده‌های ستاپ برای این ران: 3 خانواده اصلی
- Stage‑70 برای گسترش جستجو: حداقل 12 خانواده ساختاریافته

گیت‌های tradability (Stage‑A / Stage‑B):
- شروط شامل RR حداقل، cost edge مثبت، احتمال TP قبل از SL، حد نگهداری

مدل‌های ML (Tradability Model v2 / ML Stack v3):
- Base: logreg, hgbt, rf
- Calibration: Platt
- ویژگی‌ها: بازده‌های کوتاه، range/ATR، volume z, funding, taker ratios, long/short, RR, stop/target/hold, cost_edge

اعتبارسنجی:
- Replay (Full)
- Walk‑Forward (WF)
- Monte Carlo (MC)
- Cross‑seed validation
- Uncertainty gate (abstain policy)

کارایی:
- مسیر replay: precheck → micro_replay → full_replay → WF → MC
- Cache بر اساس data_hash / setup_signature / timeframe / cost_model

یادگیری خودکار:
- registry برای motifs و allocation priors
- campaign memory فقط برای allocation (cold‑start هر ران)

## 5) ابزارها و اجزای نرم‌افزار
- زبان: Python 3.11
- کتابخانه‌ها: pandas, numpy, scikit‑learn
- optional (اگر نصب باشند): xgboost, lightgbm, catboost
- گزارش‌گیری: summary JSON + report MD برای هر Stage
- تست‌ها: pytest با پوشش گسترده

## 6) خروجی‌ها و آرتیفکت‌ها
مهم‌ترین خروجی‌های این ران:
- گزارش trace کامل: docs/full_trace_report.md
- trace JSON کامل: docs/full_trace_summary.json
- خلاصه هر Stage: docs/stageXX_summary.json
- آرتیفکت‌های ران: runs/<run_id>/stageXX/…

## 7) عملکرد آخرین ران کامل (Path‑1)
Run ID:
- stage28_run_id: 20260313_154858_044fff9053df_stage28
- chain_id: 7331827e6135cda0d46c7470

خلاصه رشد کاندیدها:
- Stage‑39: raw candidates = 64
- Stage‑47: upgraded raw = 196، shortlisted = 91
- Stage‑52: 275 candidates (5 تایم‌فریم × 3 خانواده)

Tradability و ML:
- Stage‑53: label_coverage = 0.8727، survivors A/B = 139
- Stage‑48: tradable_rate = 0.1694، net_return_after_cost_mean = -0.0032

گیت‌های Stage‑57 (علت شکست نهایی):
- Replay: trade_count = 139، exp_lcb = 0.0079، maxDD = 1.0، failure_dominance = 0.8306 → FAIL
- Walk‑Forward: usable_windows = 0 → FAIL
- Monte‑Carlo: conservative_downside_bound = -0.005 → FAIL
- Cross‑seed: surviving_seeds = 0 → FAIL

Verdict نهایی:
- Stage‑72: NO_EDGE_IN_SCOPE (fail streak = 14)

## 8) تحلیل نهایی (چرا به صفر می‌رسیم)
سیستم سیگنال تولید می‌کند و حتی replay محدود دارد، اما:
- drawdown بسیار بالا (maxDD = 1.0)
- تمرکز شکست‌ها بسیار بالا (failure dominance = 0.83)
- WF/MC عملاً پاس نشده یا اجرا نشده
- پایداری cross‑seed صفر

بنابراین در گیت‌های robustness شکست می‌خوریم و مسیر به edge معتبر نمی‌رسد.

## 9) منابع برای تحلیل و عیب‌یابی توسط LLM
برای تحلیل عمیق، این فایل‌ها کلیدی هستند:
- docs/full_trace_summary.json
- docs/stage57_summary.json
- docs/stage48_tradability_learning_summary.json
- docs/stage53_summary.json
- docs/stage55_summary.json
- docs/stage71_summary.json

