# Fouad Tool

Simple terminal banner & text color tool.

## Requirements
- Python 3.6+
- pip

## Install dependencies
```bash
pip install -r requirements.txt
```

## Run
From the project folder:
```bash
# run with python
python3 ppo

# or make executable and run
chmod +x ppo
./ppo
```

## Files
- `ppo` — main python script (the tool)
- `README.md` — this file
- `requirements.txt` — list of dependencies
- `instructions/INSTALL.md` — install & run instructions


---

B — محتوى requirements.txt (انسخه إلى ملف requirements.txt)

pyfiglet
colorama


---

C — محتوى instructions/INSTALL.md (انسخه إلى instructions/INSTALL.md)

# Install & Run (detailed)

## 1) On Android (Termux)
1. Install Termux and open it.
2. Update & install git, python:
```bash
pkg update -y
pkg install git python -y
```
3. Move your script into a folder:
```bash
mkdir -p ~/myproject
mv /sdcard/Download/ppo ~/myproject/ppo   # adjust source path
cd ~/myproject
```
4. Install dependencies:
```bash
pip install -r requirements.txt
```
5. Run:
```bash
python3 ppo
# or
chmod +x ppo
./ppo
```

## 2) On Linux / macOS
1. Clone repo:
```bash
git clone https://github.com/USERNAME/REPO.git
cd REPO
```
2. Install dependencies:
```bash
pip3 install -r requirements.txt
```
3. Run:
```bash
python3 ppo
```

## 3) If missing modules:
```bash
pip install pyfiglet colorama
```


---

D — أوامر Git سريعة (Termux / any terminal)

انسخ ونفّذ من داخل مجلد المشروع (~/myproject مثلاً):

# init repo locally
git init
git add .
git commit -m "Initial commit: Fouad tool"

# create repo on GitHub via browser, then add remote (replace USERNAME/REPO)
git remote add origin https://github.com/USERNAME/REPO.git
git branch -M main
git push -u origin main
