# نتایج دور داوری (شاخهٔ `paper_TC`)

اجرا: ۲۱ آبان ۱۴۰۵ (۱۲ نوامبر ۲۰۲۶)، ساعت ۱۵:۱۱ تا ۱۹:۵۹ به وقت UTC، روی سرور vast.ai با RTX 4080.
فهرست کارها از `docs/paper/review/run_review_packages.sh` و دستور بعدی کاربر آمده است.
اینجا فقط خروجی‌های متنی و شکل‌هاست. پوشه‌های اجرا با پیش‌بینی‌ها داخل git نیستند.

**کد:**
- `paper2_rq2_depth_v1` و `paper2_rq2_mlp_lag_v1` ساعت ۱۸:۰۷ با کامیت 342d203 شروع شدند.
- بقیه با کامیت 25c6175 اجرا شدند.

**`paper2_slx_buffer_v1` کنار گذاشته شد:** در بافر ۲۰ کیلومتر، foldهای ۱، ۲، ۵ و ۶ هیچ ایستگاهی ندارند و اجرا با خطای `no station data for CO in 1402 after the hold-out` متوقف شد. ۳ اجرای بافر ۱۰ کیلومتر آن در `runs_review.csv` هست، ولی استفاده نمی‌شود.

| پوشه | فرمان |
|---|---|
| `evaluate_step4/` | `paper evaluate --hypothesis PR_c1404_r30 PR_c1404_r38 PR_depth_L3` |
| `review/` | `paper review`: همبستگی‌نگار، مبنای درون‌یابی، کالیبراسیون seedها، سال‌ها |
| `evaluate_step5/` | `paper evaluate --hypothesis PR_depth_L1 … PR_depth_L5 PR_graph_r23 PR_graph_r38 PR_c1404_r30 PR_c1404_r38` |
| `reach/` | `paper reach --exp paper2_base_v1 paper2_slx_short_v1 paper2_slx_reach_v1 paper2_rq2_mlp_v1 paper2_rq2_mlp_lag_v1 paper2_rq2_sage_v1 paper2_rq2_depth_v1` |
| `evaluate_buffer/` | `paper evaluate --hypothesis PR_buffer02 PR_buffer10 PR_buffer20` (بعد از `paper2_rq2_buffer_v2`) |
| `logs/` | لاگ همهٔ فرمان‌ها. ترتیب و کد خروج هر فرمان در `review_round_status.log` است |
| `runs_review.csv` | ردیف‌های `runs/registry.csv` برای ۳۹ اجرای این دور |

شکل‌های ۳ تا ۶ در `docs/paper/figures/` با `paper figures` در همین دور دوباره ساخته شدند.

**هشدارها (خطا نیستند):**
- `RuntimeWarning: Mean of empty slice` در هر سه `evaluate`: از نمونه‌های بوت‌استرپی است که ماه یا کانون معتبر ندارند.
- `torch.jit.script is deprecated` در اجرای `paper2_rq2_buffer_v2`.
