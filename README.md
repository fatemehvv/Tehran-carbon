# Tehran-carbon

**همجوشی داده‌های چندوجهی برای تحلیل پیش‌بینانهٔ کاهش ردپای کربن در تهران**
(پایان‌نامهٔ کارشناسی ارشد، فاطمه وفائیان؛ استاد راهنما: دکتر وحید خطیبی)

همهٔ پرسش‌های پژوهش در یک ریپو و با یک ساختار پیاده می‌شوند:

| پرسش | موضوع | وضعیت |
|---|---|---|
| RQ1 | معماری بهینهٔ همجوشی ترافیک/آلودگی/انرژی (early / late / gated + حذف حوزه‌ها) | کد آماده؛ پیش‌ثبت در `docs/preregistration_rq1.md` |
| RQ2 | آیا GNN (SAGE / GCN / GAT) کانون‌ها را بهتر از مدل‌های سنتی پیدا می‌کند؟ | کد آماده |
| RQ3 | یادگیری تقویتی برای مداخلات تطبیقی | طراحی در `docs/rq3_plan.md` |

## ساختار

```
main.py                 تنها نقطهٔ اجرا (python main.py <command>)
configs/config.yaml     تنها فایل تنظیمات: defaults + آزمایش‌های نسخه‌دار + قواعد فرضیه‌ها
data/raw/               داده‌های خام (داخل git نیست؛ فقط MANIFEST.csv با sha256 هر فایل)
data/interim, processed کش‌ها (خودکار ساخته می‌شوند)
runs/                   هر اجرا یک پوشه؛ runs/registry.csv = تاریخچهٔ همهٔ اجراها (داخل git)
gee/                    اسکریپت‌های Google Earth Engine (NO2، AlphaEarth، NO2 روزانه)
docs/                   پیش‌ثبت RQ1، طرح RQ3
src/tehran_carbon/
  config.py             Config: خواندن فایل، inherits، sweep، --set، هش
  tracking.py           Run (پوشهٔ اجرا) و RunRegistry (تاریخچه، جلوگیری از تغییر آزمایش اجراشده)
  data/
    grid.py             شبکهٔ L0 (۷۲×۱۲۶ سلول ۱/۱۲۰ درجه)
    sources/            یک کلاس برای هر حوزهٔ داده: satellite, context, traffic, energy, pollution, embeddings
    dataset.py          CellMonthDataset: ساخت دادهٔ سلول–ماه، هدف z، دامنهٔ برچسب، کش
    splits.py           SpatialSplit: CV بلوکی ۲۲ کیلومتری + بافر
    graphs.py           GridGraph (چندمقیاسی، PyG)، DistrictGraph
  models/               encoders، fusion (early/late/gated)، backbones (none/sage/gcn/gat)، model، baselines
  training/trainer.py   NeuralPredictor (استانداردسازی، چند seed، early stopping)
  experiments/          CrossValidation (CV مکانی + انتقال زمانی + مدل نهایی)
  evaluation/           metrics، BlockBootstrap، HypothesisTest
  inventory/l1.py       L1Inventory (IPCC Tier 1 + مونت‌کارلو)
  viz/maps.py           نقشهٔ مشاهده/پیش‌بینی و وزن حوزه‌ها
tests/                  pytest (روی سرور اجرا شود)
```

## نصب روی سرور

```bash
git clone https://github.com/fatemehvv/Tehran-carbon.git && cd Tehran-carbon
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
# فقط CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu
# یا GPU: نسخهٔ CUDA مناسب سرور از https://pytorch.org
pip install -r requirements.txt
```

## داده

`data/raw` داخل git نیست (حدود ۶۰ مگابایت). از لپ‌تاپ روی سرور کپی کنید و درستی آن را بسنجید:

```bash
scp -r data/raw  user@server:~/Tehran-carbon/data/      # از روی لپ‌تاپ
python main.py manifest                                  # روی سرور: باید بگوید matches MANIFEST.csv
```

اگر دادهٔ تازه‌ای اضافه شد: `python main.py manifest --write` (هش داده عوض می‌شود و کش‌ها از نو ساخته می‌شوند).

## تنظیمات و نسخه‌بندی

- همهٔ عددها، مسیرها، نام ستون‌ها و ابرپارامترها در `configs/config.yaml` است و کد هیچ مقدار ثابتی ندارد.
- هر آزمایش نام نسخه‌دار دارد (`rq1_gated_v1`) و فقط تفاوتش با `defaults` را می‌نویسد.
- **آزمایشی را که اجرا شده ویرایش نکنید.** نسخهٔ تازه بسازید:
  ```yaml
  rq1_gated_v2: {inherits: rq1_gated_v1, train: {lr: 1.0e-3}}
  ```
  اگر آزمایش اجراشده تغییر کند، `train` متوقف می‌شود (config drift).
- برای یک آزمون سریع می‌توانید `--set train.lr=1e-3` بدهید؛ این اجرا در تاریخچه `adhoc` علامت می‌خورد و در تصمیم فرضیه‌ها شرکت نمی‌کند.
- `sweep` یک آزمایش را به همهٔ ترکیب‌ها باز می‌کند (مثلاً `rq1_ablation_v1` = ۷ ترکیب حوزه × ۳ نوع همجوشی = ۲۱ اجرا).

## ترتیب اجرا

```bash
pytest                                            # آزمون‌ها (آزمون‌های داده چند دقیقه طول می‌کشند)
python main.py show --exp rq1_ablation_v1         # کانفیگ‌های حل‌شده و هش‌ها
python main.py prepare --exp rq1_gated_v1         # ساخت و کش داده + نمایش طرح CV
python main.py train --exp rq1_gated_v1 --smoke   # آزمون نصب: ۱ fold، ۳ epoch

python main.py inventory --exp inv_l1_v1          # L1 هر دو سال + مقایسه با اعداد RQ1 v2

python main.py train --exp rq1_ablation_v1        # RQ1: ۲۱ اجرا
python main.py train --exp rq1_context_only_v1
python main.py train --exp rq1_base_forward_v1
python main.py train --exp rq1_base_gbm_v1
python main.py train --exp rq1_pollution_no2_v1
python main.py evaluate --hypothesis H1           # قاعدهٔ پیش‌ثبت‌شده → H1_decision.csv

python main.py train --exp rq2_mlp_v1             # RQ2
python main.py train --exp rq2_sage_v1
python main.py train --exp rq2_base_slx_gbm_v1    # و بقیهٔ rq2_base_*
python main.py evaluate --hypothesis H2

python main.py runs --rq RQ1                      # تاریخچه
python main.py train --resume <run_id>            # ادامهٔ اجرای قطع‌شده (foldهای تمام‌شده دوباره اجرا نمی‌شوند)
```

**خروجی هر اجرا:** `runs/<زمان>_<آزمایش>_<هش>/` شامل:
- `config.yaml` و `meta.json` (کامیت git، نسخهٔ پکیج‌ها و هش داده)
- `metrics.json`
- `preds/`: پیش‌بینی‌های out-of-fold و وزن حوزه‌ها
- `models/final.pt`
- `figures/`
- `log.txt`

## مدل پیش‌آموزش‌دیده (AlphaEarth)

1. `gee/export_alphaearth_L0.js` را در Earth Engine اجرا کنید.
2. دو فایل `alphaearth_L0_2023.tif` و `alphaearth_L0_2024.tif` را در `data/raw/embeddings/` بگذارید.
3. `python main.py manifest --write` را اجرا کنید.
4. `python main.py train --exp rq1_gated_emb_v1` را اجرا کنید.

embeddingها فریز شده‌اند. با PCA کوچک می‌شوند و به گروه context اضافه می‌شوند. فقط اگر زیر همان CV مکانی نتیجه را بهتر کنند نگه داشته می‌شوند.

## افزودن چیز تازه

| چه چیزی | چه کاری |
|---|---|
| منبع داده | یک کلاس `Source` در `data/sources/` + یک بلوک در `data.inputs` + نام لایه‌ها در `data.modalities` |
| نوع همجوشی یا backbone | یک کلاس با `@FUSIONS.register("name")` یا `@BACKBONES.register("name")` |
| مدل سنتی | یک کلاس با `@BASELINES.register("name")` + پارامترها در `model.baseline.params` |
| آزمایش | یک ورودی تازه در `experiments` |

## محدودیت دادهٔ فعلی

فایل آلودگی ۱۴۰۳ فقط CO، NO2 و NOX دارد. PM2.5، PM10 و SO2 فقط برای ۱۴۰۲ هست. برای همین در طرح اصلی حوزهٔ آلودگی فقط CO است: تنها آلاینده‌ای که هم‌گونهٔ هدف نیست و در هر دو سال وجود دارد. اگر دادهٔ ساعتی کامل ۱۴۰۳ از شرکت کنترل کیفیت هوا به دست آمد، کافی است لایه‌های آن را به `data.inputs.pollution.pollutants` و `data.modalities.pollution` اضافه کنید.
