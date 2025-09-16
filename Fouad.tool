#!/usr/bin/env python3
# Fouad Tool — Clean Version (Sticky Banner)
# by fouad

import sys, os, platform, subprocess, time

# ---------- CONFIG ----------
OWNER_NAME = "fouad"
INSTAGRAM_HANDLE = "1.pvl"
TELEGRAM_HANDLE  = "blackHatFouad0"
FIGLET_FONT = "slant"
# ----------------------------------------

def ensure_import(module_name, pip_name=None):
    try:
        return __import__(module_name)
    except ImportError:
        pkg = pip_name or module_name
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return __import__(module_name)

def clear_screen():
    try:
        if platform.system().lower() == "windows":
            os.system("cls")
        else:
            os.system("clear")
    except:
        pass

# ---------- Multicolor FOUAD Banner ----------
def render_big_fouad(word="FOUAD", font=FIGLET_FONT):
    pyfiglet = ensure_import("pyfiglet")
    try:
        colorama = ensure_import("colorama"); colorama.init()
    except:
        pass
    fig = pyfiglet.Figlet(font=font)
    letters = [fig.renderText(ch) for ch in word]
    split_lines = [a.splitlines() for a in letters]
    max_h = max(len(ls) for ls in split_lines) if split_lines else 0
    COLORS = ["\033[91m","\033[93m","\033[92m","\033[94m","\033[95m"]
    BOLD = "\033[1m"; RESET = "\033[0m"
    out_lines = []
    for row in range(max_h):
        parts = []
        for i, ls in enumerate(split_lines):
            col = COLORS[i % len(COLORS)]
            if row < len(ls):
                parts.append(col + BOLD + ls[row] + RESET)
            else:
                parts.append(" " * (len(ls[0]) if ls else 0))
        out_lines.append("".join(parts))
    return "\n".join(out_lines)

def show_banner_screen(extra=""):
    clear_screen()
    print(render_big_fouad("FOUAD"))
    print(f"© {OWNER_NAME}    Instagram: @{INSTAGRAM_HANDLE}    Telegram: @{TELEGRAM_HANDLE}\n")
    if extra:
        print(extra)

# ---------- Writer ----------
def run_writer():
    pyfiglet = ensure_import("pyfiglet")
    try:
        colorama = ensure_import("colorama"); colorama.init()
    except:
        pass

    RESET="\033[0m"; BOLD="\033[1m"
    COLOR_MAP={
        "Black": "\033[30m", "Red": "\033[31m", "Green": "\033[32m", "Yellow": "\033[33m",
        "Blue": "\033[34m", "Magenta": "\033[35m", "Cyan": "\033[36m", "White": "\033[37m",
        "Bright Black": "\033[90m", "Bright Red": "\033[91m", "Bright Green": "\033[92m", "Bright Yellow": "\033[93m",
        "Bright Blue": "\033[94m", "Bright Magenta": "\033[95m", "Bright Cyan": "\033[96m", "Bright White": "\033[97m",
    }
    FONT_MAP={
        0:("plain",None),1:("standard","standard"),2:("slant","slant"),
        3:("big","big"),4:("block","block"),5:("banner3-D","banner3-D"),
        6:("doom","doom"),7:("isometric1","isometric1"),
        8:("digital","digital"),9:("small","small")
    }

    def show_fonts():
        return "\n[ fonts — pick by number ]\n" + "\n".join(f" {k:2}) {v[0]}" for k,v in FONT_MAP.items())

    def figlet_text(txt, font_choice):
        if font_choice==0: return txt
        name=FONT_MAP.get(font_choice,("standard","standard"))[1]
        return pyfiglet.Figlet(font=name).renderText(txt)

    def apply_colors(text, mode, choice):
        if mode=="1":  # single
            code=COLOR_MAP.get(choice.title(), "\033[91m")
            return BOLD+code+text+RESET
        elif mode=="2":  # all colors
            codes=list(COLOR_MAP.values())
            out=[]; idx=0
            for ch in text:
                if ch=="\n": out.append("\n"); continue
                out.append(BOLD+codes[idx%len(codes)]+ch+RESET); idx+=1
            return "".join(out)
        else:  # pattern
            names=choice.replace(","," ").split()
            codes=[COLOR_MAP.get(n.title()) for n in names if COLOR_MAP.get(n.title())]
            if not codes: codes=[COLOR_MAP["Bright Red"]]
            out=[]; idx=0
            for ch in text:
                if ch=="\n": out.append("\n"); continue
                out.append(BOLD+codes[idx%len(codes)]+ch+RESET); idx+=1
            return "".join(out)

    while True:
        # Step 1: enter text
        show_banner_screen("[ STEP 1 ] Enter your text")
        text=input("\nYour text: » ").strip()
        if not text: continue

        # Step 2: choose font
        show_banner_screen("[ STEP 2 ] Choose a font\n"+show_fonts())
        try:
            font_choice=int(input("\nFont number (default 2=slant): » ") or "2")
        except:
            font_choice=2
        art=figlet_text(text,font_choice)

        # Step 3: choose colors
        colors_list="\n".join(f"{c}{BOLD}{name}{RESET}" for name,c in COLOR_MAP.items())
        show_banner_screen("[ STEP 3 ] Choose color mode\n\n1) Single Color\n2) All Colors (Rainbow)\n3) Pattern (multiple colors)\n\n"+colors_list)
        mode=input("\nMode [1/2/3] (default 1): » ").strip() or "1"
        if mode=="1":
            choice=input("Enter color name (e.g. Red, Blue): » ").strip() or "Bright Red"
        elif mode=="2":
            choice="all"
        else:
            choice=input("Enter multiple colors (e.g. Red Green Blue): » ").strip() or "Red Blue"

        # Step 4: show result
        result=apply_colors(art, mode, choice)
        show_banner_screen("[ RESULT ]\n")
        print(result)

        again=input("\nDo you want to try again? (y/n): » ").strip().lower()
        if again!="y": break

# ---------- Run ----------
if __name__=="__main__":
    try:
        show_banner_screen("[ START ]")
        run_writer()
        print("\nThanks for using Fouad Tool!\n")
    except KeyboardInterrupt:
        print("\n[ aborted ]"); sys.exit(
