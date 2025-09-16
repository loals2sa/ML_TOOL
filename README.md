## Install & Run (quick)

### 1. Make installer & runner executable (one-time)
```bash
chmod +x install.sh run.sh

2. Run installer (installs dependencies)

./install.sh

3. Run the tool

./run.sh
# or directly
python3 ppo
# or if ppo is executable:
./ppo

If you prefer manual installation:

pip install -r requirements.txt
python3 ppo

---

## ملاحظات سريعة
- السكربت `install.sh` يحاول تغطية Termux / Debian-like Linux. على أجهزة أخرى (مثلاً بعض توزيعات Linux أو macOS) قد تحتاج أوامر تثبيت مختلفة — لكن تثبيت بايثون وpip يفي بالغرض.
- لو تضع الملف `ppo.py` بدل `ppo` غيّر اسم في `run.sh` أو أنشئ رابط: `mv ppo ppo.py` أو `ln -s ppo ppo.py`.
- إذا تريد، أجهز لك نسخة ZIP تحتوي `ppo`, `install.sh`, `run.sh`, `requirements.txt`, `README.md` جاهزين — أعطني تأكيد أسم الملفات النهائية (مثلاً `ppo` أو `ppo.py`) وأنا أكتب لك أمر واحد تنفذه لصنع الأرشيف محليًا.
