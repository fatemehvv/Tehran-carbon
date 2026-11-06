# فصل‌های ۳ تا ۵ پایان‌نامه

- `thesis_chapters_3-5.docx`: نسخهٔ فعلی (روش، یافته‌ها، جمع‌بندی) در قالب دانشگاه.
- `figures/`: شکل‌های فارسی همین فصل‌ها (PNG، ۳۰۰ dpi).
- `build/`: ساخت دوبارهٔ فایل Word از متن (`content.py`) و قالب نسخهٔ پیشین (`template_v1.docx`).

متن فصل‌ها در `build/content.py` است. پس از ویرایش آن:

```bash
cd docs/thesis/build
python fa_figures.py ../../tables template_media ../figures      # فقط اگر شکل‌ها تغییر کرده‌اند (template_media = پوشهٔ word/media قالب)
python build.py template_v1.docx ../figures out pages.json
(cd out && zip -XrD ../../thesis_chapters_3-5.docx .)
```

`pages.json` شمارهٔ صفحهٔ فهرست‌ها را نگه می‌دارد (`pages.py` آن را از PDF همین فایل می‌سازد). Word هنگام باز کردن
فایل می‌پرسد که فیلدها به‌روز شوند یا نه؛ «Yes» را بزنید تا فهرست مطالب، فهرست شکل‌ها و جدول‌ها و شماره‌ها دقیق شوند.

نیاز: Python با `matplotlib` (نسخهٔ ۳٫۱۰ یا بالاتر، برای نوشتار فارسی)، `pandas`، `Pillow` و فونت Vazirmatn.
