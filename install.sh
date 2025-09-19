#!/bin/bash

# Fouad Tool - Advanced Cybersecurity Framework
# Installation Script

echo "🔥 FOUAD TOOL - WORLD-CLASS CYBERSECURITY FRAMEWORK 🔥"
echo "Author: Fouad Zulof | Instagram: @1.pvl"
echo "======================================================"
echo ""
echo "🌍 THE MOST ADVANCED CYBERSECURITY TOOL IN THE WORLD 🌍"
echo "🤖 AI-Powered | ☁️ Cloud-Native | 🚀 Enterprise-Grade"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.7"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 3.7+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Install pip if not available
if ! command -v pip3 &> /dev/null; then
    echo "📦 Installing pip3..."
    sudo apt update
    sudo apt install -y python3-pip
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Make the script executable
chmod +x redteam/fouad.py

# Create desktop entry for GUI mode
echo "🖥️ Creating desktop entry..."
cat > ~/.local/share/applications/fouad-tool.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Fouad Tool
Comment=Advanced Cybersecurity Framework
Exec=python3 $(pwd)/redteam/fouad.py --gui
Icon=applications-internet
Terminal=false
Categories=Network;Security;
EOF

# Create symlink for easy access
echo "🔗 Creating system-wide symlink..."
sudo ln -sf $(pwd)/redteam/fouad.py /usr/local/bin/fouad-tool

# Create output directory
echo "📁 Creating output directory..."
mkdir -p ~/fouad_tool_output

# Install common cybersecurity tools (optional)
echo ""
echo "🛠️ Would you like to install common cybersecurity tools? (y/n)"
read -r install_tools

if [[ $install_tools =~ ^[Yy]$ ]]; then
    echo "📦 Installing cybersecurity tools..."
    
    # Update package list
    sudo apt update
    
    # Install world-class cybersecurity tools
    sudo apt install -y \
        nmap \
        netcat-openbsd \
        tcpdump \
        tshark \
        curl \
        wget \
        whois \
        dnsutils \
        traceroute \
        ping \
        base64 \
        john \
        hydra \
        crunch \
        fcrackzip \
        socat \
        metasploit-framework \
        sqlmap \
        nikto \
        gobuster \
        ffuf \
        wpscan \
        whatweb \
        sslscan \
        enum4linux \
        smbclient \
        responder \
        impacket-scripts \
        bloodhound \
        bettercap \
        theharvester \
        sublist3r \
        amass \
        dirsearch \
        xsstrike \
        commix \
        recon-ng \
        maltego \
        crackmapexec \
        kerbrute \
        masscan \
        zmap \
        netdiscover \
        arp-scan \
        nbtscan \
        binwalk \
        ghidra \
        radare2 \
        gdb \
        strace \
        ltrace \
        medusa \
        patator \
        cewl \
        rsmangler \
        cupp \
        nuclei \
        httpx \
        assetfinder \
        skipfish \
        wapiti \
        arachni \
        w3af \
        openvas \
        burpsuite \
        zaproxy \
        apktool \
        jadx \
        frida \
        python3-shodan \
        python3-censys \
        python3-whois \
        python3-dnspython \
        python3-netaddr \
        python3-requests \
        python3-beautifulsoup4 \
        python3-scapy \
        python3-paramiko \
        python3-cryptography \
        python3-numpy \
        python3-pandas \
        python3-matplotlib \
        python3-seaborn \
        python3-plotly \
        python3-dash \
        python3-jupyter \
        python3-tensorflow \
        python3-torch \
        python3-sklearn \
        python3-scipy \
        python3-nltk \
        python3-spacy \
        python3-transformers \
        python3-openai \
        python3-anthropic \
        python3-cohere \
        python3-huggingface-hub \
        python3-langchain \
        python3-chromadb \
        python3-pinecone-client \
        python3-weaviate-client \
        python3-qdrant-client \
        python3-redis \
        python3-elasticsearch \
        python3-pymongo \
        python3-psycopg2 \
        python3-pymysql \
        python3-sqlalchemy \
        python3-alembic \
        python3-aiohttp \
        python3-websockets \
        python3-asyncio-mqtt \
        python3-paho-mqtt \
        python3-netmiko \
        python3-napalm \
        python3-pyats \
        python3-boto3 \
        python3-azure-identity \
        python3-azure-mgmt-resource \
        python3-google-cloud-security \
        python3-kubernetes \
        python3-docker \
        python3-ansible \
        python3-terraform \
        python3-virustotal-api \
        python3-abuseipdb \
        python3-greynoise \
        python3-misp-python \
        python3-yaml \
        python3-xmltodict \
        python3-lxml \
        python3-openpyxl \
        python3-xlsxwriter \
        python3-reportlab \
        python3-jinja2 \
        python3-markdown \
        python3-psutil \
        python3-py-cpuinfo \
        python3-gputil \
        python3-netifaces \
        python3-pycryptodome \
        python3-hashlib2 \
        python3-bcrypt \
        python3-argon2-cffi \
        python3-selenium \
        python3-requests-html \
        python3-httpx \
        python3-pytest \
        python3-pytest-asyncio \
        python3-pytest-cov \
        python3-black \
        python3-flake8 \
        python3-mypy \
        python3-bandit \
        python3-safety \
        python3-sphinx \
        python3-mkdocs \
        python3-mkdocs-material \
        python3-graphviz \
        python3-pygraphviz \
        python3-cython \
        python3-numba \
        python3-joblib \
        python3-multiprocessing-logging
    
    echo "✅ Cybersecurity tools installed!"
fi

echo ""
echo "🎉 WORLD-CLASS INSTALLATION COMPLETED! 🎉"
echo ""
echo "🌍 FOUAD TOOL - THE MOST ADVANCED CYBERSECURITY FRAMEWORK IN THE WORLD 🌍"
echo ""
echo "📋 Usage:"
echo "  Terminal mode: fouad-tool"
echo "  GUI mode: fouad-tool --gui"
echo "  Or: python3 redteam/fouad.py [--gui]"
echo ""
echo "🚀 World-Class Features:"
echo "  🤖 AI-Powered Threat Analysis"
echo "  ☁️ Cloud-Native Security"
echo "  🏢 Enterprise Integration"
echo "  📊 Advanced Analytics"
echo "  🎯 Real-Time Monitoring"
echo "  🔍 Threat Intelligence"
echo "  🛡️ Red/Blue Team Automation"
echo "  📱 Mobile & IoT Security"
echo ""
echo "📁 Output directory: ~/fouad_tool_output"
echo "📝 Logs: fouad_tool.log"
echo "⚙️ Config: ~/.fouad_tool_config.json"
echo "🗄️ Database: fouad_tool.db"
echo ""
echo "🔥 THE WORLD'S MOST ADVANCED CYBERSECURITY TOOL IS READY! 🔥"
echo "🌍 Stay in the shadows, dominate the digital world! 🌍"
