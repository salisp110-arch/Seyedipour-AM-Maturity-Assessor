# app.py
# -*- coding: utf-8 -*-
import os, io, json, base64, requests, re
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Optional
from urllib.parse import quote  # برای URL-encode مسیر گیت‌هاب

# ---------- بررسی وجود plotly ----------
def _has_pkg(pkg, version=None):
    try:
        __import__(pkg)
        return True
    except Exception:
        return False

if not _has_pkg("plotly", "5.22.0"):
    st.error("برای اجرای داشبورد نیاز به بستهٔ plotly دارید. لطفاً نصب کنید: pip install plotly==5.22.0")
    st.stop()
import plotly.graph_objects as go
import plotly.express as px

# ---------- scikit-learn اختیاری (خوشه‌بندی) ----------
try:
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ---------- پیکربندی مسیرها ----------
st.set_page_config(page_title="پرسشنامه و داشبورد مدیریت دارایی", layout="wide")
BASE = Path("."); DATA_DIR = BASE/"data"; ASSETS_DIR = BASE/"assets"
DATA_DIR.mkdir(exist_ok=True); ASSETS_DIR.mkdir(exist_ok=True)

# ---------- استایل و فونت ----------
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font@v30.1.0/dist/font-face.css">
<style>
:root{ --app-font: Vazir, Tahoma, Arial, sans-serif; }
html, body, * { font-family: var(--app-font) !important; direction: rtl; }
.block-container{ padding-top: .6rem; padding-bottom: 3rem; }
h1,h2,h3,h4{ color:#16325c; }
.page-head{ display:flex; gap:16px; align-items:center; margin-bottom:10px; }
.page-title{ margin:0; font-weight:800; color:#16325c; font-size:20px; line-height:1.4; }

.question-card{
  background: rgba(255,255,255,0.78); backdrop-filter: blur(6px);
  padding: 16px 18px; margin: 10px 0 16px 0; border-radius: 14px;
  border: 1px solid #e8eef7; box-shadow: 0 6px 16px rgba(36,74,143,0.08), inset 0 1px 0 rgba(255,255,255,0.7);
}
.q-head{ font-weight:800; color:#16325c; font-size:15px; margin-bottom:8px; }
.q-desc{ color:#222; font-size:14px; line-height:1.9; margin-bottom:10px; }
.q-num{ display:inline-block; background:#e8f0fe; color:#16325c; font-weight:700; border-radius:8px; padding:2px 8px; margin-left:6px; font-size:12px;}
.q-question{ color:#0f3b8f; font-weight:700; margin:.2rem 0 .4rem 0; }

.kpi{
<style>
  border-radius:14px; padding:16px 18px; border:1px solid #e6ecf5;
  background:linear-gradient(180deg,#ffffff 0%,#f6f9ff 100%); box-shadow:0 8px 20px rgba(0,0,0,0.05);
  min-height:96px;
}
.kpi .title{ color:#456; font-size:13px; margin-bottom:6px; }
.kpi .value{ color:#0f3b8f; font-size:22px; font-weight:800; }
.kpi .sub{ color:#6b7c93; font-size:12px; }

.panel{
  background: linear-gradient(180deg,#f2f7ff 0%, #eaf3ff 100%);
  border:1px solid #d7e6ff; border-radius:16px; padding:16px 18px; margin:12px 0 18px 0;
  box-shadow: 0 10px 24px rgba(31,79,176,.12), inset 0 1px 0 rgba(255,255,255,.8);
}
.panel h3, .panel h4{ margin-top:0; color:#17407a; }

.mapping table{ font-size:12px; }
.mapping .row_heading, .mapping .blank{ display:none; }

.stTabs [role="tab"]{ direction: rtl; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_white"
TARGET = 45  # 🎯 خط هدف ثابت

# ---------- موضوعات (اگر topics.json نبود، خودکار بساز) ----------
TOPICS_PATH = BASE/"topics.json"
EMBEDDED_TOPICS = [
    {"id":1, "name":"هدف و زمینه (Purpose & Context)",
     "desc":"Purpose و Context نقطه شروع سیستم مدیریت دارایی هستند. Purpose همان مأموریت و ارزش‌هایی است که سازمان برای ذی‌نفعان خلق می‌کند. Context محیطی است که سازمان در آن فعالیت دارد: شامل شرایط اجتماعی، سیاسی، اقتصادی، فناورانه و داخلی. این دو باید در SAMP و اهداف مدیریت دارایی منعکس شوند تا اقدامات سازمان همسو با مأموریت اصلی باشد. ابزارهایی مانند SWOT و PESTLE برای تحلیل محیط و شناسایی ریسک‌ها و فرصت‌ها استفاده می‌شوند. سازمان‌هایی که Purpose و Context را به‌طور منظم بازنگری می‌کنند، بهتر می‌توانند منابع خود را بهینه کنند، ریسک‌ها را کاهش دهند و فرصت‌ها را شناسایی نمایند."},
    {"id":2, "name":"مدیریت ذی‌نفعان",
     "desc":"مدیریت ذی‌نفعان به معنای داشتن یک رویکرد ساختاریافته و مستند برای شناسایی، درگیر کردن و مدیریت نیازها و انتظارات افرادی است که می‌توانند بر سازمان اثر بگذارند یا از آن اثر بپذیرند. این ذی‌نفعان می‌توانند داخلی یا خارجی باشند. هدف، ایجاد شفافیت و اطمینان از این است که ارزش‌های مورد انتظار ذی‌نفعان در فعالیت‌های مدیریت دارایی منعکس شود. ابزارهایی مانند Stakeholder Mapping و ماتریس نفوذ-علاقه به سنجش اهمیت و تعریف راهکار ارتباط مؤثر کمک می‌کنند. پایش مستمر و سازوکارهای رسمی مشارکت، مدیریت ریسک و مشروعیت اجتماعی را تقویت می‌کند."},
    {"id":3, "name":"هزینه‌یابی و ارزش‌گذاری دارایی",
     "desc":"هزینه‌یابی دارایی شامل شناسایی و ثبت کل هزینه‌های سرمایه‌ای (Capex) و عملیاتی (Opex) در طول چرخه عمر است. ارزش‌گذاری دارایی فرآیند سنجش ارزش مالی دارایی‌ها طبق استانداردهای حسابداری است. این دو حوزه برای تصمیم‌گیری سرمایه‌گذاری و گزارش‌دهی مالی حیاتی‌اند. ابزارهایی مانند NPV، IRR، Payback و LCC به‌کار می‌روند."},
    {"id":4, "name":"خط مشی مدیریت دارایی",
     "desc":"خط مشی مدیریت دارایی سندی رسمی است که تعهد سازمان به مدیریت دارایی را بیان می‌کند و با چشم‌انداز، مأموریت و اهداف کلان همسو می‌شود. این سیاست چارچوبی جهت‌دار برای هم‌سویی برنامه‌های استراتژیک و اهداف دارایی فراهم می‌کند و معمولاً بخشی از SAMP است و با سایر خط‌مشی‌های کلان یکپارچه می‌شود. سازمان‌های پیشرو این سیاست را به‌طور منظم بازبینی و به کارکنان ابلاغ می‌کنند."},
    {"id":5, "name":"سیستم مدیریت دارایی (AMS)",
     "desc":"سیستم مدیریت دارایی مجموعه‌ای از عناصر مرتبط برای ایجاد، به‌روزرسانی و پایدارسازی سیاست‌ها، اهداف و فرآیندهای مدیریت دارایی است و باید با سایر سیستم‌های مدیریتی مانند ISO 9001/14001/45001 همسو باشد. این سیستم شامل فرآیندهایی برای ارزیابی اثربخشی، شناسایی عدم انطباق‌ها و اجرای بهبود مستمر است. ISO 55001 چارچوب طراحی و ممیزی ارائه می‌دهد."},
    {"id":6, "name":"اطمینان و ممیزی",
     "desc":"اطمینان و ممیزی فرآیندهای ساختاریافته‌ای برای ارزیابی اثربخشی دارایی‌ها، فعالیت‌های مدیریت دارایی و خود AMS هستند. الگوی «سه خط دفاع» معمولاً برای تفکیک مسئولیت‌های عملیاتی، کنترل ریسک و ممیزی مستقل استفاده می‌شود. ممیزی‌های داخلی و خارجی، ورودی‌های کلیدی برای بازنگری مدیریت و بهبود AMS محسوب می‌شوند."},
    {"id":7, "name":"استانداردهای فنی و قوانین",
     "desc":"باید اطمینان حاصل شود که تمامی فعالیت‌ها با قوانین، مقررات و استانداردهای فنی مرتبط (ملی، بین‌المللی یا صنعتی) سازگارند. علاوه بر قوانین الزام‌آور، «کدهای عملی» و استانداردهای صنعتی معیار قضاوت خوب محسوب می‌شوند. فرآیندهای شناسایی، پایش و اعمال الزامات در SAMP و برنامه‌های چرخه عمر ضروری است. ممیزی مستقل ابزار کلیدی اطمینان از انطباق است."},
    {"id":8, "name":"آرایش سازمانی",
     "desc":"آرایش سازمانی نحوه سازمان‌دهی افراد از نظر ساختار، مسئولیت‌ها و خطوط ارتباطی است. جایگاه مدیریت دارایی در چارت سازمانی نشانه مهمی از جدیت سازمان در این حوزه است. تعریف نقش‌ها و مسئولیت‌های مدیریت دارایی در سطح ارشد برای همکاری بین‌رشته‌ای ضروری است."},
    {"id":9, "name":"فرهنگ سازمانی",
     "desc":"فرهنگ سازمانی نحوه فکر کردن و رفتار افراد در جهت اهداف مدیریت دارایی است. فرهنگ باید فعالانه مدیریت شود تا همکاری، شفافیت، مسئولیت‌پذیری و یادگیری مستمر تقویت شود. حمایت مشهود مدیریت ارشد و سازگاری رفتارها پایه‌های فرهنگ مطلوب‌اند."},
    {"id":10, "name":"مدیریت شایستگی",
     "desc":"شایستگی یعنی توانایی به‌کارگیری دانش و مهارت برای دستیابی به نتایج مورد انتظار. مدیریت شایستگی شامل ارزیابی، ثبت و توسعه مهارت‌های افراد از سطح هیئت‌مدیره تا کارگاه است. چارچوب‌هایی مانند IAM Competence Framework و ISO 55012 برای تعریف و پایش شایستگی‌ها به کار می‌آیند."},
    {"id":11, "name":"مدیریت تغییر سازمانی",
     "desc":"رویکردی ساختاریافته برای هدایت افراد در برابر تغییرات فرآیندها، فناوری، ساختار یا فرهنگ. مدل‌هایی مانند ADKAR یا ۸گام کاتر کمک می‌کنند. عوامل کلیدی موفقیت: رهبری متعهد، مشارکت ذی‌نفعان، ارتباطات شفاف و برنامه آموزشی."},
    {"id":12, "name":"تحلیل تقاضا",
     "desc":"ابزاری برای درک نیازهای آینده ذی‌نفعان و تغییرات احتمالی آنها. خروجی تحلیل تقاضا ورودی مهمی برای مدیریت ریسک، برنامه‌ریزی سرمایه‌ای و عملیاتی است. شامل پیش‌بینی سناریو، تحلیل روند و مدل‌های کمی."},
    {"id":13, "name":"توسعه پایدار",
     "desc":"پاسخگویی به نیازهای امروز بدون به خطر انداختن توان نسل‌های آینده. تعیین معیارهای پایداری، LCA، کاهش کربن و هم‌سویی با SDGs/BS8900-1 توصیه می‌شود."},
    {"id":14, "name":"استراتژی و اهداف مدیریت دارایی",
     "desc":"در SAMP تعریف می‌شوند و اصول سیاست مدیریت دارایی را به اقدامات عملی تبدیل می‌کنند. اهداف باید SMART باشند و نیاز ذی‌نفعان، ریسک، چرخه عمر و قابلیت‌های سازمان لحاظ شوند."},
    {"id":15, "name":"برنامه‌ریزی مدیریت دارایی",
     "desc":"تهیه برنامه‌های عملیاتی برای تحقق SAMP شامل فعالیت‌ها، منابع، هزینه‌ها، زمان‌بندی‌ها و مسئولیت‌ها. ادغام با سایر برنامه‌های سازمانی و بازنگری منظم اهمیت دارد."},
    {"id":16, "name":"استراتژی و برنامه‌ریزی توقف‌ها و تعمیرات اساسی",
     "desc":"STO شامل برنامه‌ریزی، زمان‌بندی و اجرای کارهایی است که در زمان بهره‌برداری قابل انجام نیست. این فعالیت‌ها پرهزینه و پرریسک‌اند و نیازمند هماهنگی واحدها هستند."},
    {"id":17, "name":"برنامه‌ریزی اضطراری و تحلیل تاب‌آوری",
     "desc":"توانایی مقاومت در برابر اختلالات و بازگشت سریع. ابزارها: چرخه تاب‌آوری، ISO 22301، تحلیل سناریو."},
    {"id":18, "name":"استراتژی و مدیریت منابع",
      "desc":"تعیین نحوه تأمین و مدیریت منابع انسانی، تجهیزاتی، خدمات و مواد لازم؛ شامل استخدام، برون‌سپاری، شراکت، مدیریت پیمانکاران و هم‌راستایی با SAMP."},
    {"id":19, "name":"مدیریت زنجیره تأمین",
     "desc":"تضمین تأمین به‌موقع و باکیفیت تجهیزات/مواد/خدمات؛ انتخاب و ارزیابی پیمانکاران، مدیریت قراردادها و ریسک تأمین‌کنندگان."},
    {"id":20, "name":"تحقق ارزش چرخه عمر",
     "desc":"اطمینان از بیشترین ارزش کل در کل چرخه عمر (ایجاد، بهره‌برداری، نگهداری، بهبود، نوسازی و کنارگذاری). ابزارها: تحلیل ارزش، LCC، TCO، CBA."},
    {"id":21, "name":"هزینه‌یابی و ارزش‌گذاری دارایی (تمرکز مالی)",
     "desc":"ثبت دقیق Capex/Opex و ارزش‌گذاری برای تصمیم‌گیری سرمایه‌ای و گزارش‌دهی مالی با استفاده از ابزارهای کمی."},
    {"id":22, "name":"تصمیم‌گیری",
     "desc":"در قلب AM؛ روش متناسب با ریسک/پیچیدگی؛ چارچوب تصمیم‌گیری، مشارکت بین‌رشته‌ای و ابزارهای کمی و ماتریس ریسک."},
    {"id":23, "name":"ایجاد و تملک دارایی",
     "desc":"برنامه‌ریزی تا تحویل به بهره‌برداری با درنظرگرفتن RAMS و هزینه‌های کل؛ روش‌های قراردادی مانند PPP/BOT/اجاره نیز رایج است."},
    {"id":24, "name":"مهندسی سیستم‌ها",
     "desc":"رویکرد میان‌رشته‌ای با تمرکز بر RAMS؛ V-Model از نیازمندی تا آزمون/اعتبارسنجی و مدیریت واسط‌ها؛ ISO 15288 راهنماست."},
    {"id":25, "name":"قابلیت اطمینان یکپارچه",
     "desc":"به‌کارگیری اصول/تکنیک‌های قابلیت اطمینان در سراسر چرخه عمر (RCM, FMECA, تحلیل خرابی، افزونگی) برای کاهش ریسک خرابی."},
    {"id":26, "name":"عملیات دارایی",
     "desc":"سیاست‌ها/فرآیندهای بهره‌برداری برای سطح خدمت با رعایت HSE، قابلیت اطمینان و عملکرد مالی؛ توجه به خطای انسانی، اتوماسیون و پایش."},
    {"id":27, "name":"اجرای نگهداری",
     "desc":"مدیریت برنامه‌ریزی، زمان‌بندی، اجرا و تحلیل نگهداری؛ بازرسی/پایش وضعیت، PM، CM و استفاده از EAMS و روش‌های پیش‌بینانه."},
    {"id":28, "name":"مدیریت و پاسخ به رخدادها",
     "desc":"تشخیص، تحلیل، اقدام اصلاحی و بازیابی پس از خرابی‌ها/حوادث؛ FRACAS، RCA، 5Why، ایشیکاوا؛ سازوکار واکنش سریع متناسب با ریسک."},
    {"id":29, "name":"بازتخصیص و کنارگذاری دارایی",
     "desc":"گزینه‌های بازاستفاده/نوسازی/فروش/بازیافت/کنارگذاری با توجه به اثرات اقتصادی، زیست‌محیطی و اجتماعی؛ اقتصاد دایره‌ای."},
    {"id":30, "name":"استراتژی داده و اطلاعات",
     "desc":"مشخص می‌کند داده‌های دارایی چگونه جمع‌آوری، ذخیره، تحلیل، نگهداری و حذف می‌شوند؛ هم‌سویی با SAMP، کیفیت داده، امنیت و یکپارچگی."},
    {"id":31, "name":"مدیریت دانش",
     "desc":"شناسایی، ثبت، سازمان‌دهی، اشتراک‌گذاری و نگهداری دانش ضمنی/صریح؛ درس‌آموخته‌ها، جانشین‌پروری، BIM و دوقلوی دیجیتال."},
    {"id":32, "name":"استانداردهای داده و اطلاعات",
     "desc":"استانداردهای طبقه‌بندی، ویژگی‌ها، مقیاس وضعیت، دسته‌بندی خرابی، KPIها و کیفیت داده؛ استفاده از BIM/DT/ISO 8000."},
    {"id":33, "name":"مدیریت داده و اطلاعات",
     "desc":"تضمین دقت، به‌روز بودن، امنیت و دسترس‌پذیری؛ تعیین مسئولیت‌ها، فرکانس به‌روزرسانی و کیفیت؛ سطح اعتماد به داده مشخص شود."},
    {"id":34, "name":"سیستم‌های داده و اطلاعات",
     "desc":"سیستم‌های پشتیبان جمع‌آوری/یکپارچه‌سازی/تحلیل؛ یکپارچگی سیستم‌ها و هزینه-فایدهٔ داده‌ها برای تصمیم‌گیری بهتر."},
    {"id":35, "name":"مدیریت پیکربندی",
     "desc":"فرآیند شناسایی، ثبت و کنترل ویژگی‌های عملکردی/فیزیکی دارایی‌ها، نرم‌افزارها و اسناد؛ کنترل تغییر، گزارش وضعیت و ممیزی."},
    {"id":36, "name":"مدیریت ریسک",
     "desc":"طبق ISO 31000: اثر عدم قطعیت بر اهداف؛ تهدید/فرصت؛ Criticality، ماتریس ریسک، رجیستر، Bow-tie، FTA، ETA؛ ۴T، اشتهای ریسک و تحمل ریسک."},
    {"id":37, "name":"پایش",
     "desc":"سنجش ارزش تحقق‌یافته با شاخص‌های مالی/غیرف مالی، سطح خدمت و وضعیت دارایی‌ها؛ بازخورد برای بهینه‌سازی سرمایه‌گذاری/عملیات/نگهداری."},
    {"id":38, "name":"بهبود مستمر",
     "desc":"تحلیل عملکرد برای شناسایی فرصت‌ها و ایجاد تغییرات تدریجی؛ چرخه PDCA پراستفاده‌ترین ابزار است."},
    {"id":39, "name":"مدیریت تغییر",
     "desc":"سیستمی برای شناسایی، ارزیابی، اجرا و اطلاع‌رسانی تغییرات ناشی از قوانین جدید، فناوری نو، تغییرات کارکنان یا شرایط بحرانی."},
    {"id":40, "name":"نتایج و پیامدها",
     "desc":"ترکیبی از خروجی‌ها و اثرات کوتاه/بلندمدت مالی/غیرمالی؛ چارچوب‌های Value Framework و 6 Capitals برای سنجش ارزش به‌کار می‌روند."}
]
if not TOPICS_PATH.exists():
    TOPICS_PATH.write_text(json.dumps(EMBEDDED_TOPICS, ensure_ascii=False, indent=2), encoding="utf-8")
TOPICS = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))
if len(TOPICS)!=40:
    st.warning("⚠️ تعداد موضوعات باید دقیقاً ۴۰ باشد.")

# ---------- نقش‌ها و رنگ‌ها ----------
ROLES = ["مدیران ارشد","مدیران اجرایی","سرپرستان / خبرگان","متخصصان فنی","متخصصان غیر فنی"]
ROLE_COLORS = {"مدیران ارشد":"#d62728","مدیران اجرایی":"#1f77b4","سرپرستان / خبرگان":"#2ca02c",
               "متخصصان فنی":"#ff7f0e","متخصصان غیر فنی":"#9467bd","میانگین سازمان":"#111"}

# ---------- گزینه‌های پاسخ ----------
LEVEL_OPTIONS = [
    ("اطلاعی در این مورد ندارم.",0),
    ("سازمان نیاز به این موضوع را شناسایی کرده ولی جزئیات آن را نمی‌دانم.",1),
    ("سازمان در حال تدوین دستورالعمل‌های مرتبط است و فعالیت‌هایی به‌صورت موردی انجام می‌شود.",2),
    ("بله، این موضوع در سازمان به‌صورت کامل و استاندارد پیاده‌سازی و اجرایی شده است.",3),
    ("بله، چند سال است که نتایج اجرای آن بر اساس شاخص‌های استاندارد ارزیابی می‌شود و از بهترین تجربه‌ها برای بهبود مستمر استفاده می‌گردد.",4),
]
REL_OPTIONS = [("هیچ ارتباطی ندارد.",1),("ارتباط کم دارد.",3),("تا حدی مرتبط است.",5),("ارتباط زیادی دارد.",7),("کاملاً مرتبط است.",10)]

# ---------- وزن‌های فازی ----------
ROLE_MAP_EN2FA={"Senior Managers":"مدیران ارشد","Executives":"مدیران اجرایی","Supervisors/Sr Experts":"سرپرستان / خبرگان","Technical Experts":"متخصصان فنی","Non-Technical Experts":"متخصصان غیر فنی"}
NORM_WEIGHTS = {
    1:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    2:{"Senior Managers":0.2692,"Executives":0.3846,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    3:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    4:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    5:{"Senior Managers":0.2692,"Executives":0.3846,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    6:{"Senior Managers":0.1923,"Executives":0.2692,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    7:{"Senior Managers":0.0385,"Executives":0.1923,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.3846,"Non-Technical Experts":0.1154},
    8:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    9:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.0385,"Non-Technical Experts":0.1923},
    10:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    11:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    12:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    13:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    14:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    15:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    16:{"Senior Managers":0.1154,"Executives":0.1923,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.2692,"Non-Technical Experts":0.0385},
    17:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    18:{"Senior Managers":0.2692,"Executives":0.3846,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    19:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    20:{"Senior Managers":0.2692,"Executives":0.3846,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    21:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    22:{"Senior Managers":0.2692,"Executives":0.3846,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    23:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    24:{"Senior Managers":0.0385,"Executives":0.1923,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.3846,"Non-Technical Experts":0.1154},
    25:{"Senior Managers":0.0385,"Executives":0.1923,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.3846,"Non-Technical Experts":0.1154},
    26:{"Senior Managers":0.1154,"Executives":0.1923,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.2692,"Non-Technical Experts":0.0385},
    27:{"Senior Managers":0.1154,"Executives":0.1923,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.2692,"Non-Technical Experts":0.0385},
    28:{"Senior Managers":0.1154,"Executives":0.1923,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.2692,"Non-Technical Experts":0.0385},
    29:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.0385,"Technical Experts":0.1154,"Non-Technical Experts":0.2692},
    30:{"Senior Managers":0.1154,"Executives":0.3846,"Supervisors/Sr Experts":0.0385,"Technical Experts":0.2692,"Non-Technical Experts":0.1923},
    31:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    32:{"Senior Managers":0.0385,"Executives":0.2692,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.3846,"Non-Technical Experts":0.1923},
    33:{"Senior Managers":0.0385,"Executives":0.1923,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.3846,"Non-Technical Experts":0.2692},
    34:{"Senior Managers":0.0385,"Executives":0.2692,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.3846,"Non-Technical Experts":0.1923},
    35:{"Senior Managers":0.0385,"Executives":0.1923,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.3846,"Non-Technical Experts":0.2692},
    36:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    37:{"Senior Managers":0.0385,"Executives":0.2692,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.1923,"Non-Technical Experts":0.1154},
    38:{"Senior Managers":0.0385,"Executives":0.2692,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.1923,"Non-Technical Experts":0.1154},
    39:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    40:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.0385,"Non-Technical Experts":0.1923},
}

# ---------- GitHub backend (اختیاری/پایدار) ----------
def _get_secret(name, default=""):
    try:
        val = st.secrets.get(name, None)
        if val is not None:
            return str(val)
    except Exception:
        pass
    return os.getenv(name, default)

GH_TOKEN  = _get_secret("GITHUB_TOKEN", "").strip()
GH_REPO   = _get_secret("GH_REPO", "").strip()          # e.g. "owner/repo"
GH_BRANCH = _get_secret("GH_BRANCH", "main").strip()
GH_DIR    = _get_secret("GH_DIR", "data").strip() or "data"
USE_GH    = bool(GH_TOKEN and GH_REPO)

def _gh_headers():
    return {
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

def _encode_path(path: str) -> str:
    # هر بخش مسیر را جداگانه encode می‌کنیم تا حروف فارسی/فاصله‌ها اوکی شوند
    return "/".join(quote(seg) for seg in path.split("/"))

def _gh_contents_url(path):
    return f"https://api.github.com/repos/{GH_REPO}/contents/{_encode_path(path)}"

def _gh_get_file(path):
    try:
        url = _gh_contents_url(path)
        params = {"ref": GH_BRANCH}
        r = requests.get(url, headers=_gh_headers(), params=params, timeout=20)
        if r.status_code == 200:
            data = r.json()
            sha = data.get("sha")
            content_b64 = data.get("content", "")
            content_bytes = base64.b64decode(content_b64)
            return sha, content_bytes
        elif r.status_code == 404:
            return None, None
        else:
            return None, None
    except Exception:
        return None, None

def _gh_put_file(path, content_bytes, message, sha=None):
    url = _gh_contents_url(path)
    body = {
        "message": message,
        "branch": GH_BRANCH,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
    }
    if sha:
        body["sha"] = sha
    r = requests.put(url, headers=_gh_headers(), json=body, timeout=25)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GitHub PUT failed: {r.status_code} {r.text[:200]}")

def _gh_list_companies():
    companies = set()
    try:
        url = _gh_contents_url(GH_DIR)
        params = {"ref": GH_BRANCH}
        r = requests.get(url, headers=_gh_headers(), params=params, timeout=20)
        if r.status_code == 200:
            items = r.json()
            for it in items:
                if it.get("type") == "dir" and it.get("name"):
                    # بررسی وجود responses.csv
                    sub = f"{GH_DIR}/{it['name']}/responses.csv"
                    _, content = _gh_get_file(sub)
                    if content:
                        companies.add(it["name"])
    except Exception:
        pass
    return sorted(companies)

# ---------- کمک‌توابع داده ----------
def _sanitize_company_name(name: str) -> str:
    """نام شرکت برای مسیر فایل (حذف / و \\ و کنترل فاصله‌ها)"""
    s = (name or "").strip()
    s = s.replace("/", "／").replace("\\", "＼")
    s = re.sub(r"\s+", " ", s)
    s = s.strip(".")
    return s

def ensure_company(company: str):
    if not USE_GH:
        folder = DATA_DIR/_sanitize_company_name(company)
        folder.mkdir(parents=True, exist_ok=True)

def _standard_cols():
    cols=["timestamp","company","respondent","role"]
    for t in TOPICS:
        cols += [f"t{t['id']}_maturity", f"t{t['id']}_rel", f"t{t['id']}_adj"]
    return cols

def load_company_df(company: str) -> pd.DataFrame:
    company = _sanitize_company_name(company)
    cols = _standard_cols()

    # --- تلاش از GitHub ---
    if USE_GH:
        gh_path = f"{GH_DIR}/{company}/responses.csv"
        sha, content = _gh_get_file(gh_path)
        if content:
            try:
                from io import BytesIO
                return pd.read_csv(BytesIO(content))
            except Exception:
                # اگر CSV خراب/ناخوانا بود، به لوکال برگرد
                pass

    # --- حالت محلی (fallback یا حالت عادی) ---
    p = DATA_DIR/company/"responses.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=cols)

def save_response(company: str, rec: dict):
    # company اصلی برای داخل CSV می‌ماند؛ برای مسیر ذخیره از sanitized استفاده می‌کنیم
    folder_name = _sanitize_company_name(company)

    # ابتدا دیتافریم موجود را (با fallback) بخوانیم
    df_old = load_company_df(folder_name)
    df_new = pd.concat([df_old, pd.DataFrame([rec])], ignore_index=True)

    # --- تلاش برای نوشتن در GitHub ---
    if USE_GH:
        gh_path = f"{GH_DIR}/{folder_name}/responses.csv"
        sha, _existing = _gh_get_file(gh_path)
        csv_bytes = df_new.to_csv(index=False).encode("utf-8")
        msg = f"Add response: {folder_name} @ {rec.get('timestamp','')}"
        try:
            _gh_put_file(gh_path, csv_bytes, msg, sha=sha)
            return
        except Exception:
            st.warning("ذخیره در GitHub ناموفق بود؛ به‌صورت محلی ذخیره شد. مدیر سیستم PAT/دسترسی GitHub را بررسی کند.")

    # --- ذخیره لوکال (fallback) ---
    (DATA_DIR/folder_name).mkdir(parents=True, exist_ok=True)
    df_new.to_csv(DATA_DIR/folder_name/"responses.csv", index=False)

def get_company_logo_path(company:str)->Optional[Path]:
    folder=DATA_DIR/_sanitize_company_name(company)
    for ext in ("png","jpg","jpeg"):
        p=folder/f"logo.{ext}"
        if p.exists(): return p
    return None

# ---------- توابع رسم ----------
def _angles_deg_40():
    base=np.arange(0,360,360/40.0); return (base+90)%360

def plot_radar(series_dict, tick_numbers, tick_mapping_df, target=45, annotate=False, height=900, point_size=7):
    N=len(tick_numbers); angles=_angles_deg_40()
    fig=go.Figure()
    for label,vals in series_dict.items():
        arr=list(vals)
        if len(arr)!=N: arr=(arr+[None]*N)[:N]
        fig.add_trace(go.Scatterpolar(
            r=arr+[arr[0]], theta=angles.tolist()+[angles[0]], thetaunit="degrees",
            mode="lines+markers"+("+text" if annotate else ""), name=label,
            text=[f"{v:.0f}" if v is not None else "" for v in arr+[arr[0]]] if annotate else None,
            marker=dict(size=point_size, line=dict(width=1), color=ROLE_COLORS.get(label, "#333"))))
    # خط هدف
    fig.add_trace(go.Scatterpolar(r=[target]*(N+1), theta=angles.tolist()+[angles[0]], thetaunit="degrees",
        mode="lines", name=f"هدف {target}", line=dict(dash="dash",width=3,color="#444"), hoverinfo="skip"))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, font=dict(family="Vazir, Tahoma"),
        height=height,
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100], dtick=10, gridcolor="#e6ecf5"),
            angularaxis=dict(thetaunit="degrees", direction="clockwise", rotation=0,
                             tickmode="array", tickvals=angles.tolist(),
                             ticktext=tick_numbers, gridcolor="#edf2fb"),
            bgcolor="white"
        ),
        paper_bgcolor="#ffffff",
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        margin=dict(t=40,b=120,l=10,r=10)
    )
    c1, c2 = st.columns([3,2])
    with c1: st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### نگاشت شماره ↔ نام موضوع")
        st.dataframe(tick_mapping_df, use_container_width=True, height=min(700, 22*(len(tick_numbers)+2)))

def plot_bars_multirole(per_role, labels, title, target=45, height=600):
    fig=go.Figure()
    for lab,vals in per_role.items():
        fig.add_trace(go.Bar(x=labels, y=vals, name=lab, marker_color=ROLE_COLORS.get(lab, "#555")))
    fig.update_layout(template=PLOTLY_TEMPLATE, font=dict(family="Vazir, Tahoma"),
        title=title, xaxis_title="موضوع", yaxis_title="نمره (0..100)",
        xaxis=dict(tickfont=dict(size=10)), barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        margin=dict(t=40,b=120,l=10,r=10), paper_bgcolor="#ffffff", height=height)
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=target-5, y1=target+5,
                  fillcolor="rgba(255,0,0,0.06)", line_width=0)
    fig.add_hline(y=target, line_dash="dash", line_color="red", annotation_text=f"هدف {TARGET}")
    st.plotly_chart(fig, use_container_width=True)

def plot_bars_top_bottom(series, topic_names, top=10):
    s = pd.Series(series, index=[f"{i+1:02d} — {n}" for i,n in enumerate(topic_names)])
    top_s = s.sort_values(ascending=False).head(top)
    bot_s = s.sort_values(ascending=True).head(top)
    colA, colB = st.columns(2)
    with colA:
        fig = px.bar(top_s[::-1], orientation="h", template=PLOTLY_TEMPLATE, title=f"Top {top} (میانگین سازمان)")
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        fig = px.bar(bot_s[::-1], orientation="h", template=PLOTLY_TEMPLATE, title=f"Bottom {top} (میانگین سازمان)")
        st.plotly_chart(fig, use_container_width=True)

def plot_lines_multirole(per_role, title, target=45):
    x=[f"{i+1:02d}" for i in range(len(list(per_role.values())[0]))]; fig=go.Figure()
    for lab,vals in per_role.items():
        fig.add_trace(go.Scatter(x=x, y=vals, mode="lines+markers", name=lab, line=dict(width=2, color=ROLE_COLORS.get(lab, "#444"))))
    fig.update_layout(template=PLOTLY_TEMPLATE, font=dict(family="Vazir, Tahoma"),
        title=title, xaxis_title="موضوع", yaxis_title="نمره (0..100)", paper_bgcolor="#ffffff", hovermode="x unified")
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=target-5, y1=target+5,
                  fillcolor="rgba(255,0,0,0.06)", line_width=0)
    fig.add_hline(y=target, line_dash="dash", line_color="red", annotation_text=f"هدف {TARGET}")
    st.plotly_chart(fig, use_container_width=True)

def org_weighted_topic(per_role_norm_fa, topic_id:int):
    w=NORM_WEIGHTS.get(topic_id,{}); num=0.; den=0.
    en2fa=ROLE_MAP_EN2FA
    for en_key,weight in w.items():
        fa=en2fa[en_key]; lst=per_role_norm_fa.get(fa,[]); idx=topic_id-1
        if idx<len(lst) and pd.notna(lst[idx]): num+=weight*lst[idx]; den+=weight
    return np.nan if den==0 else num/den

# ---------- تب‌ها ----------
tabs = st.tabs(["📝 پرسشنامه","📊 داشبورد"])

# ======================= پرسشنامه =======================
with tabs[0]:
    # هدر
    st.markdown('<div class="page-head">', unsafe_allow_html=True)
    col1, col2 = st.columns([1,6])
    holding_logo_path = ASSETS_DIR/"holding_logo.png"
    with col1:
        if holding_logo_path.exists(): st.image(str(holding_logo_path), width=110)
    with col2:
        st.markdown('<div class="page-title">پرسشنامه تعیین سطح بلوغ هلدینگ انرژی گستر سینا و شرکت‌های تابعه در مدیریت دارایی فیزیکی</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # برندینگ هلدینگ (تعویض لوگو)
    with st.expander("⚙️ برندینگ هلدینگ (اختیاری)"):
        holding_logo_file = st.file_uploader("لوگوی هلدینگ انرژی گستر سینا", type=["png","jpg","jpeg"], key="upl_holding_logo")
        if holding_logo_file:
            (ASSETS_DIR/"holding_logo.png").write_bytes(holding_logo_file.getbuffer())
            st.success("لوگوی هلدینگ به‌روزرسانی شد. صفحه را یک‌بار رفرش کنید.")

    # باکس راهنما
    st.info("برای هر موضوع ابتدا توضیح فارسی آن را بخوانید، سپس با توجه به دو پرسش ذیل هر موضوع، یکی از گزینه‌های زیر هر پرسش را انتخاب بفرمایید.")

    company_input = st.text_input("نام شرکت")
    respondent = st.text_input("نام و نام خانوادگی (اختیاری)")
    role = st.selectbox("نقش / رده سازمانی", ROLES)

    answers={}
    for t in TOPICS:
        st.markdown(f'''
        <div class="question-card">
          <div class="q-head"><span class="q-num">{t["id"]:02d}</span>{t["name"]}</div>
          <div class="q-desc">{t["desc"].replace("\n","<br>")}</div>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown(f'<div class="q-question">۱) به نظر شما، موضوع «{t["name"]}» در سازمان شما در چه سطحی قرار دارد؟</div>', unsafe_allow_html=True)
        m_choice = st.radio("", options=[opt for (opt,_) in LEVEL_OPTIONS], key=f"mat_{t['id']}", label_visibility="collapsed")
        st.markdown(f'<div class="q-question">۲) موضوع «{t["name"]}» چقدر به حیطه کاری شما ارتباط مستقیم دارد؟</div>', unsafe_allow_html=True)
        r_choice = st.radio("", options=[opt for (opt,_) in REL_OPTIONS], key=f"rel_{t['id']}", label_visibility="collapsed")
        answers[t['id']] = (m_choice, r_choice)

    if st.button("ثبت پاسخ"):
        company = (company_input or "").strip()
        role_val = (role or "").strip()
        if not company: st.error("نام شرکت را وارد کنید.")
        elif not role_val: st.error("نقش/رده سازمانی را انتخاب کنید.")
        elif len(answers)!=len(TOPICS): st.error("لطفاً همهٔ ۴۰ موضوع را پاسخ دهید.")
        else:
            ensure_company(company)
            rec={"timestamp":datetime.now().isoformat(timespec="seconds"),
                 "company":company,"respondent":respondent,"role":role_val}
            m_map = dict(LEVEL_OPTIONS); r_map = dict(REL_OPTIONS)
            for t in TOPICS:
                m_label, r_label = answers[t['id']]
                m = m_map.get(m_label, 0)
                r = r_map.get(r_label, 1)
                rec[f"t{t['id']}_maturity"]=m; rec[f"t{t['id']}_rel"]=r; rec[f"t{t['id']}_adj"]=m*r
            save_response(company, rec)
            st.success("✅ پاسخ شما با موفقیت ذخیره شد.")

# ======================= داشبورد =======================
with tabs[1]:
    st.subheader("📊 داشبورد نتایج")
    password = st.text_input("🔑 رمز عبور داشبورد را وارد کنید", type="password")
    if password != "Emacraven110":
        st.error("دسترسی محدود است. رمز عبور درست را وارد کنید."); st.stop()

    # نمایش وضعیت backend
    st.caption(
        f"حالت ذخیره‌سازی: {'GitHub' if USE_GH else 'Local CSV'}"
        + (f" — {GH_REPO} · {GH_BRANCH} · {GH_DIR}" if USE_GH else "")
    )

    # لیست شرکت‌ها (GitHub + لوکال)
    companies_local = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    companies_github = _gh_list_companies() if USE_GH else []
    companies = sorted(set(companies_local) | set(companies_github))
    if not companies: st.warning("هنوز هیچ پاسخی ثبت نشده است."); st.stop()

    company_sel = st.selectbox("انتخاب شرکت", companies)
    company = (company_sel or "").strip()

    colL, colH, colC = st.columns([1,1,6])
    holding_logo_path = ASSETS_DIR/"holding_logo.png"
    with colH:
        if holding_logo_path.exists(): st.image(str(holding_logo_path), width=90, caption="هلدینگ")
    with colL:
        st.caption("لوگوی شرکت (لوکال):")
        comp_logo_file = st.file_uploader("آپلود/به‌روزرسانی لوگو", key="uplogo", type=["png","jpg","jpeg"])
        if comp_logo_file:
            (DATA_DIR/_sanitize_company_name(company)/"logo.png").write_bytes(comp_logo_file.getbuffer())
            st.success("لوگوی شرکت ذخیره شد.")
        comp_logo_path = get_company_logo_path(company)
        if comp_logo_path: st.image(str(comp_logo_path), width=90, caption=company)

    df = load_company_df(company)
    if df.empty:
        st.warning("برای این شرکت پاسخی وجود ندارد."); st.stop()

    # نرمال‌سازی 0..100
    for t in TOPICS:
        c=f"t{t['id']}_adj"
        df[c]=pd.to_numeric(df[c], errors="coerce")
        df[c]=df[c].apply(lambda x: (x/40)*100 if pd.notna(x) else np.nan)

    # میانگین نقش‌ها
    role_means={}
    for r in ROLES:
        sub=df[df["role"]==r]
        role_means[r]=[sub[f"t{t['id']}_adj"].mean() if not sub.empty else np.nan for t in TOPICS]

    # میانگین سازمان (وزن‌دهی فازی)
    per_role_norm_fa={r:role_means[r] for r in ROLES}
    org_series=[org_weighted_topic(per_role_norm_fa, t["id"]) for t in TOPICS]

    # KPI
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    nanmean_org=np.nanmean(org_series)
    org_avg=float(nanmean_org) if np.isfinite(nanmean_org) else 0.0
    pass_rate=np.mean([1 if (v>=TARGET) else 0 for v in org_series if pd.notna(v)])*100 if any(pd.notna(v) for v in org_series) else 0
    simple_means=[np.nanmean([role_means[r][i] for r in ROLES if pd.notna(role_means[r][i])]) for i in range(40)]
    has_any=any(np.isfinite(x) for x in simple_means)
    if has_any:
        best_idx=int(np.nanargmax(simple_means)); worst_idx=int(np.nanargmin(simple_means))
        best_label=f"{best_idx+1:02d} — {TOPICS[best_idx]['name']}"; worst_label=f"{worst_idx+1:02d} — {TOPICS[worst_idx]['name']}"
    else:
        best_label="-"; worst_label="-"

    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(f"""<div class="kpi"><div class="title">میانگین سازمان (فازی)</div>
    <div class="value">{org_avg:.1f}</div><div class="sub">از 100</div></div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="kpi"><div class="title">نرخ عبور از هدف</div>
    <div class="value">{pass_rate:.0f}%</div><div class="sub">نقاط ≥ {TARGET}</div></div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="kpi"><div class="title">بهترین موضوع</div>
    <div class="value">{best_label}</div><div class="sub">میانگین ساده نقش‌ها</div></div>""", unsafe_allow_html=True)
    k4.markdown(f"""<div class="kpi"><div class="title">ضعیف‌ترین موضوع</div>
    <div class="value">{worst_label}</div><div class="sub">میانگین ساده نقش‌ها</div></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # فیلترها/تنظیمات
    st.markdown('<div class="panel"><h4>فیلترها و تنظیمات نمایش</h4>', unsafe_allow_html=True)
    annotate_radar = st.checkbox("نمایش اعداد روی نقاط رادار", value=False)
    col_sz1, col_sz2 = st.columns(2)
    with col_sz1:
        radar_point_size = st.slider("اندازه نقاط رادار", 4, 12, 7, key="rad_pt")
    with col_sz2:
        radar_height = st.slider("ارتفاع رادار (px)", 600, 1100, 900, 50, key="rad_h")
    bar_height = st.slider("ارتفاع نمودار میله‌ای (px)", 400, 900, 600, 50, key="bar_h")

    roles_selected = st.multiselect("نقش‌های قابل نمایش", ROLES, default=ROLES)
    topic_range = st.slider("بازهٔ موضوع‌ها", 1, 40, (1,40))
    label_mode = st.radio("حالت برچسب محور X / زاویه", ["شماره (01..40)","نام کوتاه","نام کامل"], horizontal=True)
    idx0, idx1 = topic_range[0]-1, topic_range[1]
    topics_slice = TOPICS[idx0:idx1]
    names_full = [t['name'] for t in topics_slice]
    names_short = [n if len(n)<=14 else n[:13]+"…" for n in names_full]
    labels_bar = [f"{i+idx0+1:02d}" for i,_ in enumerate(topics_slice)] if label_mode=="شماره (01..40)" else (names_short if label_mode=="نام کوتاه" else names_full)
    tick_numbers = [f"{i+idx0+1:02d}" for i,_ in enumerate(topics_slice)]
    tick_mapping_df = pd.DataFrame({"شماره":tick_numbers, "نام موضوع":names_full})
    role_means_filtered={r: role_means[r][idx0:idx1] for r in roles_selected}
    org_series_slice = org_series[idx0:idx1]
    st.markdown('</div>', unsafe_allow_html=True)

    # رادار چندنقشی + جدول نگاشت
    st.markdown('<div class="panel"><h4>رادار ۴۰‌بخشی (خوانا)</h4>', unsafe_allow_html=True)
    if role_means_filtered:
        plot_radar(role_means_filtered, tick_numbers, tick_mapping_df, target=TARGET,
                   annotate=annotate_radar, height=radar_height, point_size=radar_point_size)
    else:
        st.info("نقشی برای نمایش انتخاب نشده است.")
    st.markdown('</div>', unsafe_allow_html=True)

    # رادار میانگین سازمان (وزن‌دهی فازی)
    st.markdown('<div class="panel"><h4>رادار میانگین سازمان (وزن‌دهی فازی)</h4>', unsafe_allow_html=True)
    plot_radar({"میانگین سازمان": org_series_slice}, tick_numbers, tick_mapping_df,
               target=TARGET, annotate=annotate_radar, height=radar_height, point_size=radar_point_size)
    st.markdown('</div>', unsafe_allow_html=True)

    # میله‌ای گروهی (نقش‌ها)
    st.markdown('<div class="panel"><h4>نمودار میله‌ای گروهی (نقش‌ها)</h4>', unsafe_allow_html=True)
    plot_bars_multirole({r:role_means[r][idx0:idx1] for r in roles_selected},
                        labels_bar, "مقایسه رده‌ها (0..100)", target=TARGET, height=bar_height)
    st.markdown('</div>', unsafe_allow_html=True)

    # Top / Bottom
    st.markdown('<div class="panel"><h4>Top/Bottom — میانگین سازمان</h4>', unsafe_allow_html=True)
    plot_bars_top_bottom(org_series_slice, names_full, top=10)
    st.markdown('</div>', unsafe_allow_html=True)

    # Heatmap و Boxplot
    st.markdown('<div class="panel"><h4>Heatmap و Boxplot</h4>', unsafe_allow_html=True)
    heat_df = pd.DataFrame({"موضوع":labels_bar})
    for r in roles_selected: heat_df[r]=role_means[r][idx0:idx1]
    hm = heat_df.melt(id_vars="موضوع", var_name="نقش", value_name="امتیاز")
    fig_heat = px.density_heatmap(hm, x="نقش", y="موضوع", z="امتیاز",
                                  color_continuous_scale="RdYlGn", height=560, template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_heat, use_container_width=True)
    fig_box = px.box(hm.dropna(), x="نقش", y="امتیاز", points="all", color="نقش",
                     color_discrete_map=ROLE_COLORS, template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # همبستگی و خوشه‌بندی (اختیاری)
    st.markdown('<div class="panel"><h4>ماتریس همبستگی و خوشه‌بندی</h4>', unsafe_allow_html=True)
    corr_base = heat_df.set_index("موضوع")[roles_selected]
    if not corr_base.empty:
        corr = corr_base.T.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                             aspect="auto", height=620, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_corr, use_container_width=True)
    if SKLEARN_OK and not corr_base.empty:
        try:
            X_raw = corr_base.values
            imp_med = SimpleImputer(strategy="median"); X_med = imp_med.fit_transform(X_raw)
            if np.isnan(X_med).any():
                imp_zero = SimpleImputer(strategy="constant", fill_value=0.0); X = imp_zero.fit_transform(X_raw)
            else:
                X = X_med
            if np.allclose(X, 0) or np.nanstd(X) == 0:
                st.info("دادهٔ کافی/متغیر برای خوشه‌بندی وجود ندارد.")
            else:
                k = st.slider("تعداد خوشه‌ها (K)", 2, 6, 3)
                K = min(k, X.shape[0]) if X.shape[0] >= 2 else 2
                if X.shape[0] >= 2:
                    km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
                    clusters = km.labels_
                    cl_df = pd.DataFrame({"موضوع":corr_base.index,"خوشه":clusters}).sort_values("خوشه")
                    st.dataframe(cl_df, use_container_width=True)
                else:
                    st.info("برای خوشه‌بندی حداقل به ۲ موضوع نیاز است.")
        except Exception as e:
            st.warning(f"خوشه‌بندی انجام نشد: {e}")
    else:
        st.caption("برای فعال‌شدن خوشه‌بندی، scikit-learn را نصب کنید (اختیاری).")

    st.markdown('</div>', unsafe_allow_html=True)

    # دانلود
    st.markdown('<div class="panel"><h4>دانلود</h4>', unsafe_allow_html=True)
    st.download_button("⬇️ دانلود CSV پاسخ‌های شرکت",
                       data=load_company_df(company).to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{_sanitize_company_name(company)}_responses.csv", mime="text/csv")
    st.caption("برای دانلود تصویر نمودارها، می‌توانید بستهٔ اختیاری `kaleido` را نصب کنید.")
    st.markdown('</div>', unsafe_allow_html=True)
