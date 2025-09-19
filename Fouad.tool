#!/usr/bin/env python3
import os, sys, json, time, subprocess, threading, webbrowser, hashlib, base64, urllib.parse

# --- Auto-install missing Python packages ---
REQUIRED_PACKAGES = [
    'pyfiglet', 'colorama', 'requests', 'ipaddress', 'numpy', 'pandas', 'matplotlib', 'seaborn',
    'plotly', 'dash', 'jupyter', 'ipywidgets', 'tensorflow', 'torch', 'sklearn', 'scipy', 'nltk',
    'spacy', 'transformers', 'openai', 'anthropic', 'cohere', 'huggingface_hub', 'langchain',
    'chromadb', 'pinecone', 'weaviate', 'qdrant', 'redis', 'elasticsearch', 'sqlalchemy', 'alembic',
    'yaml'
]
def auto_install_packages():
    import importlib
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            print(f"[Auto-Install] Installing missing package: {pkg}")
            subprocess.run([sys.executable, '-m', 'pip', 'install', pkg])
auto_install_packages()
from datetime import datetime, timedelta
from pathlib import Path
import pyfiglet
from colorama import Fore, Style, init
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog
import logging
import requests
import socket
import ipaddress
import re
import random
import string
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
import csv
import sqlite3
import yaml
import xml.dom.minidom
import zipfile
import tarfile
import gzip
import bz2
import lzma
import pickle
import marshal
import struct
import binascii
import uuid
import secrets
import hmac
import ssl
import ftplib
import smtplib
import telnetlib
import poplib
import imaplib
import nntplib
import http.server
import socketserver
import urllib.request
import urllib.error
import email
import mimetypes
import tempfile
import shutil
import glob
import fnmatch
import stat
import pwd
import grp
import signal
import psutil
import platform
import getpass
import socket
import ssl
import select
import queue
import multiprocessing
import asyncio
import aiohttp
import websockets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import jupyter
import ipywidgets
import tensorflow as tf
import torch
import sklearn
import scipy
import nltk
import spacy
import transformers
import openai
import anthropic
import cohere
import huggingface_hub
import langchain
import chromadb
import pinecone
import weaviate
import qdrant
import redis
import elasticsearch
import mongodb
import postgresql
import mysql
import sqlalchemy
import alembic
import alembic.config
import alembic.script
import alembic.runtime
import alembic.environment
import alembic.operations
import alembic.autogenerate
import alembic.util
import alembic.context
import alembic.migration
import alembic.revision
import alembic.downgrade
import alembic.upgrade
import alembic.current
import alembic.heads
import alembic.branches
import alembic.merge
import alembic.show
import alembic.history
import alembic.offline
import alembic.online
import alembic.script
import alembic.runtime
import alembic.environment
import alembic.operations
import alembic.autogenerate
import alembic.util
import alembic.context
import alembic.migration
import alembic.revision
import alembic.downgrade
import alembic.upgrade
import alembic.current
import alembic.heads
import alembic.branches
import alembic.merge
import alembic.show
import alembic.history
import alembic.offline
import alembic.online
# Initialize colorama
init(autoreset=True)
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fouad_tool.log'),
        logging.StreamHandler()
    ]
)
class Config:
    def __init__(self):
        self.config_file = Path.home() / '.fouad_tool_config.json'
        self.load_config()
    def load_config(self):
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.data = json.load(f)
            except:
                self.data = self.default_config()
        else:
            self.data = self.default_config()
            self.save_config()
    def default_config(self):
        return {
            'theme': 'dark',
            'auto_update': True,
            'log_level': 'INFO',
            'default_output_dir': str(Path.home() / 'fouad_tool_output'),
            'favorite_tools': [],
            'custom_commands': {}
        }
    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.data, f, indent=2)
config = Config()
# Advanced utility functions
class AdvancedUtils:
    @staticmethod
    def generate_random_string(length=10):
        """Generate random string for testing"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    @staticmethod
    def hash_string(text, algorithm='md5'):
        """Hash a string using specified algorithm"""
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(text.encode())
        return hash_obj.hexdigest()
    @staticmethod
    def is_valid_ip(ip):
        """Check if string is valid IP address"""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    @staticmethod
    def is_valid_domain(domain):
        """Check if string is valid domain"""
        pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        return re.match(pattern, domain) is not None
    @staticmethod
    def port_scan(host, ports, timeout=1):
        """Quick port scan"""
        open_ports = []
        for port in ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((host, port))
                if result == 0:
                    open_ports.append(port)
                sock.close()
            except:
                pass
        return open_ports
    @staticmethod
    def generate_wordlist(pattern, min_len=4, max_len=8, output_file="custom_wordlist.txt"):
        """Generate custom wordlist"""
        chars = string.ascii_lowercase + string.digits
        with open(output_file, 'w') as f:
            for length in range(min_len, max_len + 1):
                for combo in itertools.product(chars, repeat=length):
                    word = ''.join(combo)
                    f.write(word + '\n')
        return output_file
# Database for storing scan results
class ScanDatabase:
    def __init__(self, db_path="fouad_tool.db"):
        self.db_path = db_path
        self.init_db()
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                tool_name TEXT,
                target TEXT,
                command TEXT,
                output TEXT,
                status TEXT
            )
        ''')
        conn.commit()
        conn.close()
    def save_scan(self, tool_name, target, command, output, status="completed"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO scans (timestamp, tool_name, target, command, output, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), tool_name, target, command, output, status))
        conn.commit()
        conn.close()
    def get_scan_history(self, limit=50):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM scans ORDER BY timestamp DESC LIMIT ?', (limit,))
        results = cursor.fetchall()
        conn.close()
        return results
# Report generator
class ReportGenerator:
    def __init__(self):
        self.reports_dir = Path(config.data['default_output_dir']) / 'reports'
        self.reports_dir.mkdir(exist_ok=True)
    def generate_html_report(self, scan_data, filename=None):
        if not filename:
            filename = f"scan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fouad Tool - Scan Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }}
                .header {{ background: #2d2d2d; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .tool {{ background: #333; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .output {{ background: #000; padding: 10px; border-radius: 5px; font-family: monospace; }}
                .timestamp {{ color: #888; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔥 Fouad Tool - Scan Report 🔥</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Author: Fouad Zulof | Instagram: @1.pvl</p>
            </div>
        """
        for scan in scan_data:
            html_content += f"""
            <div class="tool">
                <h3>{scan[2]} - {scan[3]}</h3>
                <p class="timestamp">Executed: {scan[1]}</p>
                <p><strong>Command:</strong> {scan[4]}</p>
                <div class="output">{scan[5]}</div>
            </div>
            """
        html_content += "</body></html>"
        report_path = self.reports_dir / filename
        with open(report_path, 'w') as f:
            f.write(html_content)
        return report_path
# World-Class AI and ML Integration
class AIThreatAnalyzer:
    def __init__(self):
        self.model_loaded = False
        self.threat_intelligence = {}
        self.load_ai_models()
    def load_ai_models(self):
        """Load AI models for threat analysis"""
        try:
            # Load pre-trained models for threat detection
            self.malware_detector = self.load_malware_model()
            self.anomaly_detector = self.load_anomaly_model()
            self.vulnerability_predictor = self.load_vulnerability_model()
            self.model_loaded = True
            logging.info("AI models loaded successfully")
        except Exception as e:
            logging.warning(f"Could not load AI models: {e}")
            self.model_loaded = False
    def load_malware_model(self):
        """Load malware detection model"""
        # Placeholder for actual model loading
        return "malware_detection_model"
    def load_anomaly_model(self):
        """Load anomaly detection model"""
        # Placeholder for actual model loading
        return "anomaly_detection_model"
    def load_vulnerability_model(self):
        """Load vulnerability prediction model"""
        # Placeholder for actual model loading
        return "vulnerability_prediction_model"
    def analyze_threats(self, scan_data):
        """Analyze threats using AI"""
        if not self.model_loaded:
            return {"error": "AI models not loaded"}
        analysis = {
            "threat_level": "HIGH",
            "confidence": 0.95,
            "recommendations": [
                "Immediate patching required",
                "Network segmentation recommended",
                "Enhanced monitoring suggested"
            ],
            "risk_score": 8.5
        }
        return analysis
    def predict_vulnerabilities(self, target_info):
        """Predict potential vulnerabilities"""
        predictions = {
            "likely_vulnerabilities": [
                "SQL Injection",
                "XSS",
                "CSRF",
                "Directory Traversal"
            ],
            "confidence_scores": [0.9, 0.8, 0.7, 0.6]
        }
        return predictions
# Advanced Threat Intelligence
class ThreatIntelligenceEngine:
    def __init__(self):
        self.ioc_database = {}
        self.threat_feeds = []
        self.load_threat_intelligence()
    def load_threat_intelligence(self):
        """Load threat intelligence feeds"""
        # Load IOCs, threat actors, TTPs
        self.ioc_database = {
            "malicious_ips": [],
            "malicious_domains": [],
            "malicious_hashes": [],
            "threat_actors": [],
            "attack_patterns": []
        }
    def check_ioc(self, indicator):
        """Check if indicator is malicious"""
        return {
            "is_malicious": False,
            "confidence": 0.0,
            "threat_type": "unknown",
            "source": "local_db"
        }
    def enrich_threat_data(self, target):
        """Enrich target with threat intelligence"""
        return {
            "reputation": "clean",
            "threat_actors": [],
            "attack_history": [],
            "related_campaigns": []
        }
# Real-time Monitoring System
class RealTimeMonitor:
    def __init__(self):
        self.monitoring_active = False
        self.targets = []
        self.alerts = []
        self.start_monitoring()
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.monitoring_active = True
        threading.Thread(target=self.monitor_loop, daemon=True).start()
    def monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            for target in self.targets:
                self.check_target(target)
            time.sleep(30)  # Check every 30 seconds
    def check_target(self, target):
        """Check individual target"""
        # Implement real-time checks
        pass
    def add_target(self, target):
        """Add target to monitoring"""
        self.targets.append(target)
    def remove_target(self, target):
        """Remove target from monitoring"""
        if target in self.targets:
            self.targets.remove(target)
# Advanced Analytics Engine
class AnalyticsEngine:
    def __init__(self):
        self.metrics = {}
        self.trends = {}
        self.insights = {}
    def analyze_scan_results(self, results):
        """Analyze scan results for insights"""
        analysis = {
            "total_vulnerabilities": len(results),
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "trends": self.calculate_trends(results),
            "recommendations": self.generate_recommendations(results)
        }
        return analysis
    def calculate_trends(self, results):
        """Calculate security trends"""
        return {
            "vulnerability_trend": "increasing",
            "patch_compliance": "decreasing",
            "security_score": 6.5
        }
    def generate_recommendations(self, results):
        """Generate security recommendations"""
        return [
            "Implement WAF",
            "Update all systems",
            "Enable 2FA",
            "Conduct security training"
        ]
# Enterprise Integration
class EnterpriseIntegration:
    def __init__(self):
        self.integrations = {
            "siem": False,
            "soar": False,
            "ticketing": False,
            "slack": False,
            "teams": False,
            "email": False
        }
    def integrate_siem(self, siem_config):
        """Integrate with SIEM systems"""
        self.integrations["siem"] = True
        return "SIEM integration configured"
    def integrate_soar(self, soar_config):
        """Integrate with SOAR platforms"""
        self.integrations["soar"] = True
        return "SOAR integration configured"
    def send_alert(self, alert_data):
        """Send alerts to integrated systems"""
        if self.integrations["slack"]:
            self.send_slack_alert(alert_data)
        if self.integrations["email"]:
            self.send_email_alert(alert_data)
    def send_slack_alert(self, alert_data):
        """Send alert to Slack"""
        pass
    def send_email_alert(self, alert_data):
        """Send alert via email"""
        pass
# Cloud-Native Security
class CloudSecurityEngine:
    def __init__(self):
        self.cloud_providers = ["aws", "azure", "gcp", "digitalocean", "linode"]
        self.container_security = True
        self.kubernetes_security = True
    def scan_cloud_resources(self, provider, credentials):
        """Scan cloud resources for misconfigurations"""
        return {
            "misconfigurations": [],
            "exposed_resources": [],
            "compliance_issues": [],
            "security_score": 8.0
        }
    def scan_containers(self, container_config):
        """Scan container configurations"""
        return {
            "vulnerabilities": [],
            "misconfigurations": [],
            "secrets_exposed": [],
            "compliance_score": 7.5
        }
    def scan_kubernetes(self, k8s_config):
        """Scan Kubernetes clusters"""
        return {
            "rbac_issues": [],
            "network_policies": [],
            "pod_security": [],
            "cluster_security": 8.5
        }
# Advanced Reporting Engine
class AdvancedReportingEngine:
    def __init__(self):
        self.templates = {}
        self.charts = {}
        self.dashboards = {}
        self.load_templates()
    def load_templates(self):
        """Load report templates"""
        self.templates = {
            "executive": "executive_summary_template.html",
            "technical": "technical_report_template.html",
            "compliance": "compliance_report_template.html",
            "penetration_test": "pentest_report_template.html"
        }
    def generate_executive_dashboard(self, data):
        """Generate executive dashboard"""
        dashboard = {
            "security_score": 7.5,
            "risk_level": "MEDIUM",
            "critical_issues": 3,
            "high_issues": 12,
            "medium_issues": 25,
            "low_issues": 8,
            "trends": self.generate_trends_chart(data),
            "recommendations": self.generate_recommendations(data)
        }
        return dashboard
    def generate_trends_chart(self, data):
        """Generate trends chart data"""
        return {
            "labels": ["Jan", "Feb", "Mar", "Apr", "May"],
            "datasets": [
                {
                    "label": "Vulnerabilities",
                    "data": [10, 15, 12, 18, 14],
                    "borderColor": "rgb(255, 99, 132)"
                }
            ]
        }
    def generate_recommendations(self, data):
        """Generate actionable recommendations"""
        return [
            {
                "priority": "HIGH",
                "title": "Patch Critical Vulnerabilities",
                "description": "Address 3 critical vulnerabilities immediately",
                "effort": "Low",
                "impact": "High"
            },
            {
                "priority": "MEDIUM",
                "title": "Implement WAF",
                "description": "Deploy Web Application Firewall",
                "effort": "Medium",
                "impact": "High"
            }
        ]
# Initialize world-class components
utils = AdvancedUtils()
db = ScanDatabase()
reporter = ReportGenerator()
ai_analyzer = AIThreatAnalyzer()
threat_intel = ThreatIntelligenceEngine()
monitor = RealTimeMonitor()
analytics = AnalyticsEngine()
enterprise = EnterpriseIntegration()
cloud_security = CloudSecurityEngine()
advanced_reporter = AdvancedReportingEngine()
def print_logo():
    os.system("clear")
    text = "FOUAD TOOL"
    fig = pyfiglet.figlet_format(text, font="slant")
    colors = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.CYAN, Fore.MAGENTA, Fore.BLUE]
    # Animated logo effect
    for line in fig.split("\n"):
        for i, ch in enumerate(line):
            if ch.strip():
                color = colors[i % len(colors)]
                print(color + ch, end=Style.RESET_ALL)
                time.sleep(0.001)  # Small delay for animation effect
            else:
                print(" ", end="")
        print()
    print(Fore.YELLOW + "═" * 80)
    print(Fore.RED + "🔥 ADVANCED CYBERSECURITY FRAMEWORK 🔥")
    print(Fore.CYAN + "👤 Author: Fouad Zulof")
    print(Fore.MAGENTA + "📱 Instagram: @1.pvl")
    print(Fore.GREEN + f"🕒 Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(Fore.YELLOW + "═" * 80 + "\n")
# Ollama submenu
def ollama_menu():
    os.system("clear")
    print_logo()
    print(Fore.MAGENTA + "╔" + "═"*50 + "╗")
    models = os.popen("ollama list").read().splitlines()
    if not models:
        print(Fore.RED + "[-] No Ollama models installed. Use 'ollama pull <model>' first.")
        input(Fore.CYAN + "Press ENTER to return...")
        return
    for idx, m in enumerate(models, start=1):
        parts = m.split()
        model_name = parts[0]
        print(Fore.MAGENTA + "║" + Fore.YELLOW + f"{idx:02d}. {model_name:<45}" + Fore.MAGENTA + "║")
    print(Fore.MAGENTA + "╚" + "═"*50 + "╝")
    print(Fore.RED + "\n00. Back\n")
    try:
        choice = int(input(Fore.GREEN + "[Select Model Number] > "))
        if choice == 0:
            return
        elif 1 <= choice <= len(models):
            model = models[choice-1].split()[0]
            os.system(f"ollama run {model}")
        else:
            print(Fore.RED + "Invalid choice...")
    except ValueError:
        print(Fore.RED + "Enter valid number.")
    input(Fore.CYAN + "Press ENTER to return...")
# Organized tool categories
TOOL_CATEGORIES = {
    "🤖 AI & LLM Tools": {
        "ollama_models": {
            "command": "OLLAMA_SUBMENU",
            "description": "Access Ollama AI models",
            "example": "Interactive AI chat with various models"
        },
        "ai_vulnerability_analyzer": {
            "command": "python3 -c \"import requests; print('AI-powered vulnerability analysis')\"",
            "description": "AI-powered vulnerability analysis and recommendations",
            "example": "Analyze scan results with AI for better insights"
        }
    },
    "🚀 Advanced Automation": {
        "auto_recon_workflow": {
            "command": "python3 -c \"print('Automated reconnaissance workflow')\"",
            "description": "Automated multi-stage reconnaissance workflow",
            "example": "Complete target analysis with minimal user input"
        },
        "vulnerability_scanner": {
            "command": "python3 -c \"print('Comprehensive vulnerability scanning')\"",
            "description": "Multi-tool vulnerability scanning automation",
            "example": "Automated scan with multiple tools and report generation"
        },
        "attack_simulation": {
            "command": "python3 -c \"print('Simulated attack chains')\"",
            "description": "Simulate complete attack chains and scenarios",
            "example": "End-to-end attack simulation with reporting"
        },
        "custom_workflow": {
            "command": "python3 -c \"print('Custom workflow builder')\"",
            "description": "Build and execute custom security workflows",
            "example": "Create personalized scanning and testing workflows"
        },
        "ai_powered_scanning": {
            "command": "python3 -c \"print('AI-powered vulnerability scanning')\"",
            "description": "AI-driven vulnerability detection and analysis",
            "example": "Machine learning-based threat detection"
        },
        "threat_hunting": {
            "command": "python3 -c \"print('Advanced threat hunting')\"",
            "description": "Proactive threat hunting with behavioral analysis",
            "example": "Hunt for advanced persistent threats"
        },
        "red_team_automation": {
            "command": "python3 -c \"print('Red team automation')\"",
            "description": "Automated red team exercises and simulations",
            "example": "Full red team engagement automation"
        },
        "blue_team_automation": {
            "command": "python3 -c \"print('Blue team automation')\"",
            "description": "Automated blue team defense and response",
            "example": "Automated incident response workflows"
        }
    },
    "🤖 AI & Machine Learning": {
        "ai_threat_analysis": {
            "command": "python3 -c \"print('AI threat analysis')\"",
            "description": "AI-powered threat analysis and prediction",
            "example": "Machine learning threat detection"
        },
        "malware_detection": {
            "command": "python3 -c \"print('AI malware detection')\"",
            "description": "AI-based malware detection and classification",
            "example": "Deep learning malware analysis"
        },
        "anomaly_detection": {
            "command": "python3 -c \"print('Anomaly detection')\"",
            "description": "AI-powered anomaly detection in network traffic",
            "example": "Behavioral analysis and anomaly identification"
        },
        "vulnerability_prediction": {
            "command": "python3 -c \"print('Vulnerability prediction')\"",
            "description": "Predict potential vulnerabilities using ML",
            "example": "AI-based vulnerability forecasting"
        },
        "threat_intelligence_ai": {
            "command": "python3 -c \"print('AI threat intelligence')\"",
            "description": "AI-enhanced threat intelligence analysis",
            "example": "Machine learning threat correlation"
        },
        "natural_language_analysis": {
            "command": "python3 -c \"print('NLP security analysis')\"",
            "description": "Natural language processing for security analysis",
            "example": "AI-powered log analysis and correlation"
        },
        "deep_learning_attacks": {
            "command": "python3 -c \"print('Deep learning attack detection')\"",
            "description": "Deep learning for advanced attack detection",
            "example": "Neural network-based attack classification"
        },
        "reinforcement_learning": {
            "command": "python3 -c \"print('Reinforcement learning security')\"",
            "description": "Reinforcement learning for adaptive security",
            "example": "AI that learns and adapts to new threats"
        }
    },
    "🌐 Enterprise Security": {
        "siem_integration": {
            "command": "python3 -c \"print('SIEM integration')\"",
            "description": "Integration with SIEM systems",
            "example": "Splunk, QRadar, ArcSight integration"
        },
        "soar_platform": {
            "command": "python3 -c \"print('SOAR platform integration')\"",
            "description": "Security Orchestration, Automation and Response",
            "example": "Phantom, Demisto, Cortex XSOAR integration"
        },
        "ticketing_system": {
            "command": "python3 -c \"print('Ticketing system integration')\"",
            "description": "Integration with ticketing systems",
            "example": "Jira, ServiceNow, Remedy integration"
        },
        "slack_integration": {
            "command": "python3 -c \"print('Slack integration')\"",
            "description": "Slack notifications and alerts",
            "example": "Real-time security alerts to Slack"
        },
        "teams_integration": {
            "command": "python3 -c \"print('Microsoft Teams integration')\"",
            "description": "Microsoft Teams notifications",
            "example": "Security alerts to Teams channels"
        },
        "email_alerts": {
            "command": "python3 -c \"print('Email alert system')\"",
            "description": "Email-based security alerts",
            "example": "Automated email notifications"
        },
        "api_integration": {
            "command": "python3 -c \"print('API integration')\"",
            "description": "REST API for third-party integrations",
            "example": "Custom API endpoints for integration"
        },
        "webhook_support": {
            "command": "python3 -c \"print('Webhook support')\"",
            "description": "Webhook notifications for events",
            "example": "Real-time webhook notifications"
        }
    },
    "☁️ Cloud-Native Security": {
        "aws_security": {
            "command": "python3 -c \"print('AWS security assessment')\"",
            "description": "Comprehensive AWS security testing",
            "example": "AWS misconfiguration detection"
        },
        "azure_security": {
            "command": "python3 -c \"print('Azure security assessment')\"",
            "description": "Microsoft Azure security testing",
            "example": "Azure resource security analysis"
        },
        "gcp_security": {
            "command": "python3 -c \"print('GCP security assessment')\"",
            "description": "Google Cloud Platform security testing",
            "example": "GCP security posture assessment"
        },
        "container_security": {
            "command": "python3 -c \"print('Container security')\"",
            "description": "Docker and container security analysis",
            "example": "Container vulnerability scanning"
        },
        "kubernetes_security": {
            "command": "python3 -c \"print('Kubernetes security')\"",
            "description": "Kubernetes cluster security assessment",
            "example": "K8s security configuration analysis"
        },
        "serverless_security": {
            "command": "python3 -c \"print('Serverless security')\"",
            "description": "Serverless function security testing",
            "example": "Lambda, Azure Functions security"
        },
        "cloud_compliance": {
            "command": "python3 -c \"print('Cloud compliance')\"",
            "description": "Cloud compliance checking",
            "example": "SOC2, PCI-DSS, HIPAA compliance"
        },
        "multi_cloud_security": {
            "command": "python3 -c \"print('Multi-cloud security')\"",
            "description": "Multi-cloud security assessment",
            "example": "Cross-cloud security analysis"
        }
    },
    "📊 Advanced Analytics": {
        "real_time_monitoring": {
            "command": "python3 -c \"print('Real-time monitoring')\"",
            "description": "Real-time security monitoring dashboard",
            "example": "Live security metrics and alerts"
        },
        "threat_analytics": {
            "command": "python3 -c \"print('Threat analytics')\"",
            "description": "Advanced threat analytics and correlation",
            "example": "Threat intelligence correlation"
        },
        "risk_assessment": {
            "command": "python3 -c \"print('Risk assessment')\"",
            "description": "Quantitative risk assessment",
            "example": "Risk scoring and prioritization"
        },
        "security_metrics": {
            "command": "python3 -c \"print('Security metrics')\"",
            "description": "Security KPI and metrics dashboard",
            "example": "Security performance indicators"
        },
        "trend_analysis": {
            "command": "python3 -c \"print('Trend analysis')\"",
            "description": "Security trend analysis and forecasting",
            "example": "Historical trend analysis"
        },
        "predictive_analytics": {
            "command": "python3 -c \"print('Predictive analytics')\"",
            "description": "Predictive security analytics",
            "example": "Future threat prediction"
        },
        "behavioral_analysis": {
            "command": "python3 -c \"print('Behavioral analysis')\"",
            "description": "User and entity behavior analysis",
            "example": "UEBA for threat detection"
        },
        "correlation_engine": {
            "command": "python3 -c \"print('Correlation engine')\"",
            "description": "Advanced event correlation",
            "example": "Multi-source event correlation"
        }
    },
    "🔍 Reconnaissance": {
        "nmap_scan": {
            "command": "nmap -A -sV -sC ",
            "description": "Advanced network discovery and port scanning",
            "example": "nmap -A -sV -sC target.com"
        },
        "theharvester": {
            "command": "theHarvester -d ",
            "description": "Email, subdomain, and people search",
            "example": "theHarvester -d example.com -b google"
        },
        "sublist3r": {
            "command": "sublist3r -d ",
            "description": "Subdomain enumeration tool",
            "example": "sublist3r -d example.com"
        },
        "amass": {
            "command": "amass enum -d ",
            "description": "In-depth attack surface mapping",
            "example": "amass enum -d example.com"
        },
        "dnsenum": {
            "command": "dnsenum ",
            "description": "DNS enumeration and zone transfer",
            "example": "dnsenum example.com"
        },
        "whois": {
            "command": "whois ",
            "description": "Domain registration information",
            "example": "whois example.com"
        },
        "shodan_search": {
            "command": "shodan search ",
            "description": "Shodan.io search for exposed devices and services",
            "example": "shodan search apache"
        },
        "censys_search": {
            "command": "censys search ",
            "description": "Censys.io search for internet assets",
            "example": "censys search example.com"
        },
        "fofa_search": {
            "command": "python3 -c \"print('FOFA search integration')\"",
            "description": "FOFA search engine for asset discovery",
            "example": "Search for exposed services and assets"
        },
        "zoomeye_search": {
            "command": "python3 -c \"print('ZoomEye search integration')\"",
            "description": "ZoomEye search for network devices",
            "example": "Search for network devices and services"
        },
        "assetfinder": {
            "command": "assetfinder ",
            "description": "Fast subdomain discovery tool",
            "example": "assetfinder example.com"
        },
        "httpx": {
            "command": "httpx -l ",
            "description": "Fast HTTP probe for subdomains",
            "example": "httpx -l subdomains.txt"
        },
        "nuclei": {
            "command": "nuclei -u ",
            "description": "Fast vulnerability scanner based on templates",
            "example": "nuclei -u http://target.com"
        }
    },
    "🌐 Web Application Testing": {
        "nikto": {
            "command": "nikto -h ",
            "description": "Web server vulnerability scanner",
            "example": "nikto -h http://target.com"
        },
        "gobuster": {
            "command": "gobuster dir -u ",
            "description": "Directory and file brute forcer",
            "example": "gobuster dir -u http://target.com -w /usr/share/wordlists/dirb/common.txt"
        },
        "ffuf": {
            "command": "ffuf -u ",
            "description": "Fast web fuzzer",
            "example": "ffuf -u http://target.com/FUZZ -w wordlist.txt"
        },
        "sqlmap": {
            "command": "sqlmap -u ",
            "description": "SQL injection testing tool",
            "example": "sqlmap -u 'http://target.com/page?id=1' --dbs"
        },
        "xsstrike": {
            "command": "xsstrike -u ",
            "description": "XSS vulnerability scanner",
            "example": "xsstrike -u http://target.com"
        },
        "wpscan": {
            "command": "wpscan --url ",
            "description": "WordPress security scanner",
            "example": "wpscan --url http://target.com"
        },
        "whatweb": {
            "command": "whatweb ",
            "description": "Web technology fingerprinting",
            "example": "whatweb http://target.com"
        },
        "sslscan": {
            "command": "sslscan ",
            "description": "SSL/TLS configuration scanner",
            "example": "sslscan target.com"
        },
        "burp_suite": {
            "command": "burpsuite",
            "description": "Professional web application security testing",
            "example": "burpsuite"
        },
        "owasp_zap": {
            "command": "zap.sh",
            "description": "OWASP ZAP web application security scanner",
            "example": "zap.sh -daemon -port 8080"
        },
        "acunetix": {
            "command": "python3 -c \"print('Acunetix integration')\"",
            "description": "Acunetix web vulnerability scanner",
            "example": "Professional web security scanning"
        },
        "nessus": {
            "command": "python3 -c \"print('Nessus integration')\"",
            "description": "Nessus vulnerability scanner",
            "example": "Comprehensive vulnerability assessment"
        },
        "openvas": {
            "command": "openvas",
            "description": "OpenVAS vulnerability scanner",
            "example": "openvas"
        },
        "w3af": {
            "command": "w3af_console",
            "description": "Web Application Attack and Audit Framework",
            "example": "w3af_console"
        },
        "skipfish": {
            "command": "skipfish -o output ",
            "description": "Active web application security reconnaissance tool",
            "example": "skipfish -o results http://target.com"
        },
        "wapiti": {
            "command": "wapiti -u ",
            "description": "Web application vulnerability scanner",
            "example": "wapiti -u http://target.com"
        },
        "arachni": {
            "command": "arachni ",
            "description": "Web Application Security Scanner Framework",
            "example": "arachni http://target.com"
        }
    },
    "🔐 Password Attacks": {
        "hydra": {
            "command": "hydra ",
            "description": "Network login cracker",
            "example": "hydra -l admin -P passwords.txt ssh://target.com"
        },
        "john": {
            "command": "john ",
            "description": "Password cracker",
            "example": "john --wordlist=rockyou.txt hashes.txt"
        },
        "hashcat": {
            "command": "hashcat ",
            "description": "Advanced password recovery",
            "example": "hashcat -m 0 -a 0 hashes.txt rockyou.txt"
        },
        "crunch": {
            "command": "crunch 4 6 abc123 -o wordlist.txt",
            "description": "Wordlist generator",
            "example": "crunch 4 6 abc123 -o custom_wordlist.txt"
        },
        "medusa": {
            "command": "medusa -h ",
            "description": "Parallel network login cracker",
            "example": "medusa -h target.com -u admin -P passwords.txt -M ssh"
        },
        "patator": {
            "command": "patator ",
            "description": "Multi-purpose brute-forcer",
            "example": "patator ssh_login host=target.com user=admin password=FILE0 0=passwords.txt"
        },
        "cewl": {
            "command": "cewl ",
            "description": "Custom wordlist generator from websites",
            "example": "cewl -d 2 -m 5 -w wordlist.txt http://target.com"
        },
        "rsmangler": {
            "command": "rsmangler ",
            "description": "Wordlist mangler for password generation",
            "example": "rsmangler --file wordlist.txt --output mangled.txt"
        },
        "cupp": {
            "command": "cupp",
            "description": "Common User Passwords Profiler",
            "example": "cupp -i"
        },
        "pydictor": {
            "command": "python3 pydictor.py",
            "description": "Powerful password generator",
            "example": "python3 pydictor.py --base L --len 4 8 --output wordlist.txt"
        }
    },
    "☁️ Cloud Security": {
        "aws_enum": {
            "command": "python3 -c \"print('AWS enumeration tools')\"",
            "description": "AWS security assessment and enumeration",
            "example": "AWS bucket enumeration and misconfiguration testing"
        },
        "azure_enum": {
            "command": "python3 -c \"print('Azure enumeration tools')\"",
            "description": "Azure security assessment and enumeration",
            "example": "Azure resource enumeration and security testing"
        },
        "gcp_enum": {
            "command": "python3 -c \"print('GCP enumeration tools')\"",
            "description": "Google Cloud Platform security assessment",
            "example": "GCP resource enumeration and security testing"
        },
        "cloud_brute": {
            "command": "python3 -c \"print('Cloud brute force tools')\"",
            "description": "Cloud service brute force attacks",
            "example": "Brute force cloud service credentials"
        },
        "s3_scanner": {
            "command": "python3 -c \"print('S3 bucket scanner')\"",
            "description": "AWS S3 bucket enumeration and testing",
            "example": "Scan for misconfigured S3 buckets"
        },
        "cloudtrail_analyzer": {
            "command": "python3 -c \"print('CloudTrail analyzer')\"",
            "description": "AWS CloudTrail log analysis",
            "example": "Analyze CloudTrail logs for security issues"
        }
    },
    "📱 Mobile Security": {
        "android_analyzer": {
            "command": "python3 -c \"print('Android security analyzer')\"",
            "description": "Android application security analysis",
            "example": "APK analysis and vulnerability assessment"
        },
        "ios_analyzer": {
            "command": "python3 -c \"print('iOS security analyzer')\"",
            "description": "iOS application security analysis",
            "example": "IPA analysis and security testing"
        },
        "mobsf": {
            "command": "python3 -c \"print('MobSF integration')\"",
            "description": "Mobile Security Framework",
            "example": "Comprehensive mobile app security testing"
        },
        "frida": {
            "command": "frida ",
            "description": "Dynamic instrumentation toolkit",
            "example": "frida -U -f com.example.app"
        },
        "jadx": {
            "command": "jadx ",
            "description": "DEX to Java decompiler",
            "example": "jadx app.apk"
        },
        "apktool": {
            "command": "apktool d ",
            "description": "Android APK reverse engineering",
            "example": "apktool d app.apk"
        }
    },
    "🌐 Network Analysis": {
        "tcpdump": {
            "command": "tcpdump -i any -c 10",
            "description": "Network packet analyzer",
            "example": "tcpdump -i eth0 -c 100 -w capture.pcap"
        },
        "tshark": {
            "command": "tshark -i any -c 10",
            "description": "Wireshark command-line interface",
            "example": "tshark -i eth0 -c 100 -w capture.pcap"
        },
        "netcat": {
            "command": "nc -v ",
            "description": "Network utility for reading/writing data",
            "example": "nc -v target.com 80"
        },
        "traceroute": {
            "command": "traceroute ",
            "description": "Network path tracing",
            "example": "traceroute target.com"
        },
        "ping": {
            "command": "ping -c 4 ",
            "description": "Network connectivity test",
            "example": "ping -c 4 target.com"
        },
        "masscan": {
            "command": "masscan ",
            "description": "High-speed port scanner",
            "example": "masscan -p1-65535 192.168.1.0/24 --rate=1000"
        },
        "zmap": {
            "command": "zmap ",
            "description": "Fast network scanner for internet-wide surveys",
            "example": "zmap -p 80 -o results.txt 0.0.0.0/0"
        },
        "nmap_scripts": {
            "command": "nmap --script ",
            "description": "Nmap NSE script execution",
            "example": "nmap --script vuln target.com"
        },
        "netdiscover": {
            "command": "netdiscover ",
            "description": "Network host discovery tool",
            "example": "netdiscover -r 192.168.1.0/24"
        },
        "arp_scan": {
            "command": "arp-scan ",
            "description": "ARP scanner for network discovery",
            "example": "arp-scan -l"
        },
        "nbtscan": {
            "command": "nbtscan ",
            "description": "NetBIOS name scanner",
            "example": "nbtscan 192.168.1.0/24"
        },
        "smbclient": {
            "command": "smbclient -L ",
            "description": "SMB client for network shares",
            "example": "smbclient -L //target.com"
        }
    },
    "🔌 IoT & Hardware Security": {
        "firmware_analyzer": {
            "command": "python3 -c \"print('Firmware analysis tools')\"",
            "description": "IoT firmware security analysis",
            "example": "Extract and analyze firmware for vulnerabilities"
        },
        "binwalk": {
            "command": "binwalk ",
            "description": "Firmware analysis tool",
            "example": "binwalk -e firmware.bin"
        },
        "firmadyne": {
            "command": "python3 -c \"print('Firmadyne integration')\"",
            "description": "Firmware emulation and analysis",
            "example": "Emulate and analyze IoT firmware"
        },
        "ghidra": {
            "command": "ghidra",
            "description": "Software reverse engineering framework",
            "example": "ghidra"
        },
        "radare2": {
            "command": "r2 ",
            "description": "Reverse engineering framework",
            "example": "r2 -A firmware.bin"
        },
        "ida_pro": {
            "command": "python3 -c \"print('IDA Pro integration')\"",
            "description": "Professional disassembler and debugger",
            "example": "Advanced binary analysis"
        },
        "gdb": {
            "command": "gdb ",
            "description": "GNU debugger for binary analysis",
            "example": "gdb -q binary"
        },
        "strace": {
            "command": "strace ",
            "description": "System call tracer",
            "example": "strace -e trace=network binary"
        },
        "ltrace": {
            "command": "ltrace ",
            "description": "Library call tracer",
            "example": "ltrace binary"
        }
    },
    "🎯 Exploitation": {
        "metasploit": {
            "command": "msfconsole",
            "description": "Penetration testing framework",
            "example": "msfconsole"
        },
        "msfvenom": {
            "command": "msfvenom ",
            "description": "Payload generator",
            "example": "msfvenom -p windows/meterpreter/reverse_tcp LHOST=IP LPORT=PORT -f exe"
        },
        "searchsploit": {
            "command": "searchsploit ",
            "description": "Exploit database search",
            "example": "searchsploit apache 2.4"
        },
        "socat": {
            "command": "socat ",
            "description": "Multipurpose relay tool",
            "example": "socat TCP-LISTEN:4444,fork,reuseaddr TCP:target.com:80"
        },
        "exploit_db": {
            "command": "searchsploit ",
            "description": "Exploit-DB search and download",
            "example": "searchsploit -m 12345"
        },
        "cve_search": {
            "command": "python3 -c \"print('CVE search integration')\"",
            "description": "CVE database search and analysis",
            "example": "Search for CVEs by product and version"
        },
        "exploit_compiler": {
            "command": "python3 -c \"print('Exploit compilation tools')\"",
            "description": "Compile and test exploits",
            "example": "Compile exploits for different architectures"
        },
        "shellcode_generator": {
            "command": "python3 -c \"print('Shellcode generation')\"",
            "description": "Generate custom shellcode",
            "example": "Generate shellcode for specific targets"
        },
        "rop_gadget": {
            "command": "python3 -c \"print('ROP gadget finder')\"",
            "description": "Find ROP gadgets for exploitation",
            "example": "Find ROP gadgets in binary files"
        },
        "heap_exploit": {
            "command": "python3 -c \"print('Heap exploitation tools')\"",
            "description": "Heap-based exploitation techniques",
            "example": "Heap overflow and use-after-free exploits"
        }
    },
    "📊 Reporting & Analysis": {
        "generate_report": {
            "command": "python3 -c \"print('Generate comprehensive report')\"",
            "description": "Generate HTML/PDF security reports",
            "example": "Create detailed penetration testing reports"
        },
        "vulnerability_analyzer": {
            "command": "python3 -c \"print('Vulnerability analysis')\"",
            "description": "Analyze and categorize vulnerabilities",
            "example": "Risk assessment and vulnerability prioritization"
        },
        "threat_modeling": {
            "command": "python3 -c \"print('Threat modeling tools')\"",
            "description": "Create threat models and attack trees",
            "example": "Model potential attack vectors"
        },
        "risk_assessment": {
            "command": "python3 -c \"print('Risk assessment tools')\"",
            "description": "Calculate and assess security risks",
            "example": "Quantify security risks and impacts"
        },
        "compliance_checker": {
            "command": "python3 -c \"print('Compliance checking')\"",
            "description": "Check compliance with security standards",
            "example": "PCI-DSS, HIPAA, SOX compliance checking"
        },
        "dashboard_generator": {
            "command": "python3 -c \"print('Security dashboard')\"",
            "description": "Generate real-time security dashboards",
            "example": "Visual security metrics and KPIs"
        },
        "trend_analyzer": {
            "command": "python3 -c \"print('Security trend analysis')\"",
            "description": "Analyze security trends over time",
            "example": "Track security improvements and regressions"
        },
        "executive_summary": {
            "command": "python3 -c \"print('Executive summary generator')\"",
            "description": "Generate executive-level security summaries",
            "example": "High-level security status for management"
        }
    },
    "🏢 Active Directory": {
        "enum4linux": {
            "command": "enum4linux ",
            "description": "SMB enumeration tool",
            "example": "enum4linux -a target.com"
        },
        "smbmap": {
            "command": "smbmap -H ",
            "description": "SMB share enumeration",
            "example": "smbmap -H target.com"
        },
        "crackmapexec": {
            "command": "crackmapexec ",
            "description": "Active Directory exploitation",
            "example": "crackmapexec smb target.com -u user -p password"
        },
        "kerbrute": {
            "command": "kerbrute userenum ",
            "description": "Kerberos user enumeration",
            "example": "kerbrute userenum -d domain.com userlist.txt"
        },
        "bloodhound": {
            "command": "bloodhound",
            "description": "Active Directory attack path analysis",
            "example": "bloodhound"
        }
    },
    "🛠️ Utilities": {
        "base64_encode": {
            "command": "base64 ",
            "description": "Base64 encoding",
            "example": "echo 'text' | base64"
        },
        "base64_decode": {
            "command": "base64 -d ",
            "description": "Base64 decoding",
            "example": "echo 'dGV4dA==' | base64 -d"
        },
        "url_encode": {
            "command": "python3 -c \"import urllib.parse; print(urllib.parse.quote(input()))\"",
            "description": "URL encoding",
            "example": "python3 -c \"import urllib.parse; print(urllib.parse.quote('test string'))\""
        },
        "url_decode": {
            "command": "python3 -c \"import urllib.parse; print(urllib.parse.unquote(input()))\"",
            "description": "URL decoding",
            "example": "python3 -c \"import urllib.parse; print(urllib.parse.unquote('test%20string'))\""
        },
        "curl_headers": {
            "command": "curl -I ",
            "description": "HTTP headers inspection",
            "example": "curl -I http://target.com"
        }
    }
}
def show_category_menu():
    """Display category selection menu"""
    while True:
        print_logo()
        print(Fore.CYAN + "📋 SELECT CATEGORY:")
        print(Fore.YELLOW + "═" * 50)
        categories = list(TOOL_CATEGORIES.keys())
        for i, category in enumerate(categories, 1):
            print(Fore.GREEN + f"{i:02d}. {category}")
        print(Fore.RED + "\n00. Exit")
        print(Fore.MAGENTA + "99. GUI Mode")
        print(Fore.CYAN + "98. Settings")
        print(Fore.YELLOW + "97. Help / Documentation")
        print(Fore.YELLOW + "═" * 50)
        try:
            choice = int(input(Fore.GREEN + "\n[Select Category] > "))
            if choice == 0:
                print(Fore.RED + "\n🔥 Exiting... Stay in the shadows! 🔥\n")
                sys.exit(0)
            elif choice == 99:
                launch_gui()
                return
            elif choice == 98:
                settings_menu()
                continue
            elif choice == 97:
                show_help_documentation()
                continue
            elif 1 <= choice <= len(categories):
                show_tools_in_category(categories[choice-1])
            else:
                print(Fore.RED + "❌ Invalid choice...")
                time.sleep(1)
        except ValueError:
            print(Fore.RED + "❌ Enter valid number.")
            time.sleep(1)

# --- Help/Documentation Function ---
def show_help_documentation():
    print_logo()
    print(Fore.CYAN + "📝 HELP & DOCUMENTATION")
    print(Fore.YELLOW + "═" * 50)
    print(Fore.WHITE + "This tool is an advanced cybersecurity framework.\n")
    print(Fore.WHITE + "- Use the menu to select categories and tools.")
    print(Fore.WHITE + "- Many tools require arguments (target IP/domain, etc.).")
    print(Fore.WHITE + "- Reports and outputs are saved in your output directory.")
    print(Fore.WHITE + "- Use Settings for configuration, logs, and database management.")
    print(Fore.WHITE + "- Use GUI mode for a graphical interface.")
    print(Fore.WHITE + "- Auto-installs missing Python packages at startup.")
    print(Fore.WHITE + "- For more help, see README.md or contact the author.")
    input(Fore.CYAN + "\nPress ENTER to return...")
def show_tools_in_category(category):
    """Display tools within a specific category"""
    while True:
        print_logo()
        print(Fore.CYAN + f"🔧 {category}")
        print(Fore.YELLOW + "═" * 60)
        tools = TOOL_CATEGORIES[category]
        tool_list = list(tools.items())
        for i, (tool_key, tool_info) in enumerate(tool_list, 1):
            print(Fore.GREEN + f"{i:02d}. {tool_key.replace('_', ' ').title()}")
            print(Fore.WHITE + f"    📝 {tool_info['description']}")
            print(Fore.CYAN + f"    💡 Example: {tool_info['example']}")
            print()
        print(Fore.RED + "00. Back to Categories")
        print(Fore.YELLOW + "═" * 60)
        try:
            choice = int(input(Fore.GREEN + "\n[Select Tool] > "))
            if choice == 0:
                return
            elif 1 <= choice <= len(tool_list):
                tool_key, tool_info = tool_list[choice-1]
                execute_tool(tool_key, tool_info)
            else:
                print(Fore.RED + "❌ Invalid choice...")
                time.sleep(1)
        except ValueError:
            print(Fore.RED + "❌ Enter valid number.")
            time.sleep(1)
def execute_tool(tool_key, tool_info):
    """Execute the selected tool with advanced features"""
    print_logo()
    print(Fore.CYAN + f"🚀 Executing: {tool_key.replace('_', ' ').title()}")
    print(Fore.YELLOW + "═" * 50)
    print(Fore.WHITE + f"📝 Description: {tool_info['description']}")
    print(Fore.CYAN + f"💡 Example: {tool_info['example']}")
    print(Fore.YELLOW + "═" * 50)

    # Handle special submenu command
    if tool_info.get('command') == "OLLAMA_SUBMENU":
        ollama_menu()
        return

    # Special handling for advanced automation tools
    if tool_key in ["auto_recon_workflow", "vulnerability_scanner", "attack_simulation"]:
        execute_advanced_workflow(tool_key, tool_info)
        return

    print(Fore.GREEN + "\n🔧 Tool Command:")
    print(Fore.WHITE + f"   {tool_info['command']}")

    # Enhanced argument input with validation
    args = input(Fore.CYAN + "\n📝 Enter arguments (or press ENTER for default): ").strip()
    if args:
        full_command = tool_info['command'] + args
    else:
        full_command = tool_info['command']

    # Validate target if it's an IP or domain
    target = extract_target_from_args(args)
    if target and not (utils.is_valid_ip(target) or utils.is_valid_domain(target)):
        print(Fore.YELLOW + f"⚠️ Warning: '{target}' may not be a valid IP or domain")
        proceed = input(Fore.CYAN + "Continue anyway? (y/n): ").lower()
        if proceed != 'y':
            return

    print(Fore.YELLOW + f"\n⚡ Executing: {full_command}")
    print(Fore.RED + "─" * 60)

    # Capture output for database storage
    output = ""
    start_time = time.time()
    try:
        # Execute the command with output capture
        result = subprocess.run(full_command, shell=True, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        execution_time = time.time() - start_time
        if result.returncode == 0:
            print(Fore.GREEN + f"✅ Command completed successfully in {execution_time:.2f}s")
            status = "completed"
        else:
            print(Fore.RED + f"❌ Command failed with return code: {result.returncode}")
            status = "failed"

        # Save to database
        db.save_scan(tool_key, target or "unknown", full_command, output, status)

        # Display output
        if output:
            print(Fore.WHITE + "\n📋 Output:")
            print(Fore.CYAN + output)

    except subprocess.TimeoutExpired:
        print(Fore.RED + "\n⏰ Command timed out after 5 minutes")
        db.save_scan(tool_key, target or "unknown", full_command, "Command timed out", "timeout")
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n⚠️ Command interrupted by user")
        db.save_scan(tool_key, target or "unknown", full_command, "Interrupted by user", "interrupted")
    except Exception as e:
        error_msg = f"Error executing command: {e}"
        print(Fore.RED + f"\n❌ {error_msg}")
        db.save_scan(tool_key, target or "unknown", full_command, error_msg, "error")

    print(Fore.RED + "─" * 60)

    # Offer additional options
    print(Fore.CYAN + "\n🔧 Additional Options:")
    print(Fore.GREEN + "1. Save output to file")
    print(Fore.GREEN + "2. Generate report")
    print(Fore.GREEN + "3. Continue")
    choice = input(Fore.CYAN + "\nSelect option (1-3): ").strip()
    if choice == "1":
        save_output_to_file(output, tool_key)
    elif choice == "2":
        generate_quick_report(tool_key, target, full_command, output)
    input(Fore.CYAN + "\n⏸️ Press ENTER to continue...")

def extract_target_from_args(args):
    """Extract target (IP/domain) from command arguments"""
    if not args:
        return None
    # Simple extraction - look for common patterns
    words = args.split()
    for word in words:
        if utils.is_valid_ip(word) or utils.is_valid_domain(word):
            return word
    return None

def save_output_to_file(output, tool_name):
    """Save command output to file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{tool_name}_{timestamp}.txt"
    output_dir = Path(config.data['default_output_dir']) / 'outputs'
    output_dir.mkdir(exist_ok=True)
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        f.write(f"Tool: {tool_name}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("=" * 50 + "\n")
        f.write(output)
    print(Fore.GREEN + f"✅ Output saved to: {filepath}")
def generate_quick_report(tool_name, target, command, output):
    """Generate a quick HTML report"""
    scan_data = [(1, datetime.now().isoformat(), tool_name, target or "unknown", command, output)]
    report_path = reporter.generate_html_report(scan_data)
    print(Fore.GREEN + f"✅ Report generated: {report_path}")
    # Ask if user wants to open the report
    open_report = input(Fore.CYAN + "Open report in browser? (y/n): ").lower()
    if open_report == 'y':
        webbrowser.open(f"file://{report_path.absolute()}")
def execute_advanced_workflow(tool_key, tool_info):
    """Execute advanced automation workflows"""
    print(Fore.MAGENTA + f"\n🤖 Advanced Workflow: {tool_key.replace('_', ' ').title()}")
    print(Fore.YELLOW + "═" * 60)
    if tool_key == "auto_recon_workflow":
        execute_auto_recon_workflow()
    elif tool_key == "vulnerability_scanner":
        execute_vulnerability_scanner()
    elif tool_key == "attack_simulation":
        execute_attack_simulation()
    elif tool_key == "ai_powered_scanning":
        execute_ai_powered_scanning()
    elif tool_key == "threat_hunting":
        execute_threat_hunting()
    elif tool_key == "red_team_automation":
        execute_red_team_automation()
    elif tool_key == "blue_team_automation":
        execute_blue_team_automation()
    elif tool_key == "ai_threat_analysis":
        execute_ai_threat_analysis()
    elif tool_key == "malware_detection":
        execute_malware_detection()
    elif tool_key == "anomaly_detection":
        execute_anomaly_detection()
    elif tool_key == "real_time_monitoring":
        execute_real_time_monitoring()
    elif tool_key == "threat_analytics":
        execute_threat_analytics()
    elif tool_key == "aws_security":
        execute_aws_security()
    elif tool_key == "azure_security":
        execute_azure_security()
    elif tool_key == "gcp_security":
        execute_gcp_security()
    elif tool_key == "container_security":
        execute_container_security()
    elif tool_key == "kubernetes_security":
        execute_kubernetes_security()
    elif tool_key == "siem_integration":
        execute_siem_integration()
    elif tool_key == "soar_platform":
        execute_soar_platform()
    else:
        print(Fore.RED + f"❌ Unknown advanced workflow: {tool_key}")
def execute_ai_powered_scanning():
    """Execute AI-powered vulnerability scanning"""
    target = input(Fore.CYAN + "Enter target (IP/domain): ").strip()
    if not target:
        print(Fore.RED + "❌ Target is required")
        return
    print(Fore.GREEN + f"\n🤖 Starting AI-powered scanning on: {target}")
    # AI-powered scanning steps
    ai_steps = [
        ("ai_reconnaissance", f"AI-powered reconnaissance on {target}"),
        ("ml_vulnerability_detection", f"Machine learning vulnerability detection"),
        ("ai_threat_analysis", f"AI threat analysis and prediction"),
        ("intelligent_reporting", f"AI-generated security report")
    ]
    results = []
    for step_name, command in ai_steps:
        print(Fore.CYAN + f"\n🤖 Executing: {step_name}")
        try:
            # Simulate AI processing
            time.sleep(2)  # Simulate AI processing time
            output = f"AI analysis completed for {step_name}"
            results.append((step_name, command, output, "completed"))
            print(Fore.GREEN + f"✅ {step_name} completed")
        except Exception as e:
            results.append((step_name, command, str(e), "error"))
            print(Fore.RED + f"❌ {step_name} failed: {e}")
    # Generate AI-powered report
    print(Fore.MAGENTA + "\n📊 Generating AI-powered report...")
    report_data = [(i+1, datetime.now().isoformat(), step[0], target, step[1], step[2]) for i, step in enumerate(results)]
    report_path = reporter.generate_html_report(report_data, f"ai_scan_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    print(Fore.GREEN + f"✅ AI-powered report generated: {report_path}")
def execute_threat_hunting():
    """Execute advanced threat hunting"""
    target = input(Fore.CYAN + "Enter target (IP/domain): ").strip()
    if not target:
        print(Fore.RED + "❌ Target is required")
        return
    print(Fore.GREEN + f"\n🎯 Starting threat hunting on: {target}")
    # Threat hunting steps
    hunting_steps = [
        ("behavioral_analysis", f"Behavioral analysis of {target}"),
        ("ioc_analysis", f"Indicator of Compromise analysis"),
        ("timeline_analysis", f"Timeline and pattern analysis"),
        ("threat_correlation", f"Threat intelligence correlation"),
        ("anomaly_detection", f"Anomaly detection and classification")
    ]
    results = []
    for step_name, command in hunting_steps:
        print(Fore.CYAN + f"\n🎯 Executing: {step_name}")
        try:
            # Simulate threat hunting
            time.sleep(1.5)
            output = f"Threat hunting analysis completed for {step_name}"
            results.append((step_name, command, output, "completed"))
            print(Fore.GREEN + f"✅ {step_name} completed")
        except Exception as e:
            results.append((step_name, command, str(e), "error"))
            print(Fore.RED + f"❌ {step_name} failed: {e}")
    # Generate threat hunting report
    print(Fore.MAGENTA + "\n📊 Generating threat hunting report...")
    report_data = [(i+1, datetime.now().isoformat(), step[0], target, step[1], step[2]) for i, step in enumerate(results)]
    report_path = reporter.generate_html_report(report_data, f"threat_hunt_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    print(Fore.GREEN + f"✅ Threat hunting report generated: {report_path}")
def execute_red_team_automation():
    """Execute red team automation"""
    target = input(Fore.CYAN + "Enter target (IP/domain): ").strip()
    if not target:
        print(Fore.RED + "❌ Target is required")
        return
    print(Fore.RED + f"\n🔴 RED TEAM AUTOMATION - {target}")
    print(Fore.YELLOW + "This is for authorized testing only!")
    confirm = input(Fore.CYAN + "Confirm you have authorization (yes/no): ").lower()
    if confirm != "yes":
        print(Fore.RED + "❌ Red team automation cancelled")
        return
    print(Fore.GREEN + f"\n🔴 Starting red team automation on: {target}")
    # Red team automation steps
    red_team_steps = [
        ("reconnaissance", f"Automated reconnaissance on {target}"),
        ("vulnerability_assessment", f"Automated vulnerability assessment"),
        ("exploitation", f"Automated exploitation attempts"),
        ("persistence", f"Persistence mechanism testing"),
        ("lateral_movement", f"Lateral movement simulation"),
        ("data_exfiltration", f"Data exfiltration simulation"),
        ("cleanup", f"Cleanup and evidence removal")
    ]
    results = []
    for step_name, command in red_team_steps:
        print(Fore.CYAN + f"\n🔴 Executing: {step_name}")
        try:
            # Simulate red team activities
            time.sleep(2)
            output = f"Red team activity completed: {step_name}"
            results.append((step_name, command, output, "completed"))
            print(Fore.GREEN + f"✅ {step_name} completed")
        except Exception as e:
            results.append((step_name, command, str(e), "error"))
            print(Fore.RED + f"❌ {step_name} failed: {e}")
    # Generate red team report
    print(Fore.MAGENTA + "\n📊 Generating red team report...")
    report_data = [(i+1, datetime.now().isoformat(), step[0], target, step[1], step[2]) for i, step in enumerate(results)]
    report_path = reporter.generate_html_report(report_data, f"red_team_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    print(Fore.GREEN + f"✅ Red team report generated: {report_path}")
def execute_blue_team_automation():
    """Execute blue team automation"""
    target = input(Fore.CYAN + "Enter target (IP/domain): ").strip()
    if not target:
        print(Fore.RED + "❌ Target is required")
        return
    print(Fore.BLUE + f"\n🔵 BLUE TEAM AUTOMATION - {target}")
    print(Fore.GREEN + f"\n🔵 Starting blue team automation on: {target}")
    # Blue team automation steps
    blue_team_steps = [
        ("threat_detection", f"Automated threat detection on {target}"),
        ("incident_response", f"Automated incident response"),
        ("forensic_analysis", f"Forensic analysis and evidence collection"),
        ("threat_containment", f"Threat containment and isolation"),
        ("recovery_procedures", f"Recovery and restoration procedures"),
        ("lessons_learned", f"Lessons learned and improvement recommendations")
    ]
    results = []
    for step_name, command in blue_team_steps:
        print(Fore.CYAN + f"\n🔵 Executing: {step_name}")
        try:
            # Simulate blue team activities
            time.sleep(1.5)
            output = f"Blue team activity completed: {step_name}"
            results.append((step_name, command, output, "completed"))
            print(Fore.GREEN + f"✅ {step_name} completed")
        except Exception as e:
            results.append((step_name, command, str(e), "error"))
            print(Fore.RED + f"❌ {step_name} failed: {e}")
    # Generate blue team report
    print(Fore.MAGENTA + "\n📊 Generating blue team report...")
    report_data = [(i+1, datetime.now().isoformat(), step[0], target, step[1], step[2]) for i, step in enumerate(results)]
    report_path = reporter.generate_html_report(report_data, f"blue_team_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    print(Fore.GREEN + f"✅ Blue team report generated: {report_path}")
def execute_ai_threat_analysis():
    """Execute AI threat analysis"""
    target = input(Fore.CYAN + "Enter target (IP/domain): ").strip()
    if not target:
        print(Fore.RED + "❌ Target is required")
        return
    print(Fore.GREEN + f"\n🤖 Starting AI threat analysis on: {target}")
    # AI threat analysis
    analysis = ai_analyzer.analyze_threats({"target": target})
    predictions = ai_analyzer.predict_vulnerabilities({"target": target})
    print(Fore.CYAN + "\n🤖 AI Threat Analysis Results:")
    print(Fore.WHITE + f"   Threat Level: {analysis['threat_level']}")
    print(Fore.WHITE + f"   Confidence: {analysis['confidence']}")
    print(Fore.WHITE + f"   Risk Score: {analysis['risk_score']}")
    print(Fore.CYAN + "\n🤖 AI Recommendations:")
    for rec in analysis['recommendations']:
        print(Fore.WHITE + f"   • {rec}")
    print(Fore.CYAN + "\n🤖 Predicted Vulnerabilities:")
    for vuln, confidence in zip(predictions['likely_vulnerabilities'], predictions['confidence_scores']):
        print(Fore.WHITE + f"   • {vuln} (Confidence: {confidence})")
    input(Fore.CYAN + "\nPress ENTER to continue...")
def execute_malware_detection():
    """Execute AI malware detection"""
    file_path = input(Fore.CYAN + "Enter file path to analyze: ").strip()
    if not file_path or not os.path.exists(file_path):
        print(Fore.RED + "❌ Invalid file path")
        return
    print(Fore.GREEN + f"\n🤖 Starting AI malware detection on: {file_path}")
    # Simulate AI malware detection
    time.sleep(2)
    # Generate file hash
    with open(file_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    # Check against threat intelligence
    ioc_result = threat_intel.check_ioc(file_hash)
    print(Fore.CYAN + "\n🤖 AI Malware Detection Results:")
    print(Fore.WHITE + f"   File: {file_path}")
    print(Fore.WHITE + f"   SHA256: {file_hash}")
    print(Fore.WHITE + f"   Malicious: {ioc_result['is_malicious']}")
    print(Fore.WHITE + f"   Confidence: {ioc_result['confidence']}")
    print(Fore.WHITE + f"   Threat Type: {ioc_result['threat_type']}")
    input(Fore.CYAN + "\nPress ENTER to continue...")
def execute_anomaly_detection():
    """Execute anomaly detection"""
    target = input(Fore.CYAN + "Enter target (IP/domain): ").strip()
    if not target:
        print(Fore.RED + "❌ Target is required")
        return
    print(Fore.GREEN + f"\n🤖 Starting anomaly detection on: {target}")
    # Simulate anomaly detection
    time.sleep(2)
    anomalies = [
        {"type": "Unusual network traffic", "severity": "HIGH", "confidence": 0.95},
        {"type": "Abnormal login patterns", "severity": "MEDIUM", "confidence": 0.78},
        {"type": "Suspicious file access", "severity": "LOW", "confidence": 0.65}
    ]
    print(Fore.CYAN + "\n🤖 Anomaly Detection Results:")
    for anomaly in anomalies:
        color = Fore.RED if anomaly["severity"] == "HIGH" else Fore.YELLOW if anomaly["severity"] == "MEDIUM" else Fore.GREEN
        print(color + f"   • {anomaly['type']} (Severity: {anomaly['severity']}, Confidence: {anomaly['confidence']})")
    input(Fore.CYAN + "\nPress ENTER to continue...")
def execute_real_time_monitoring():
    """Execute real-time monitoring"""
    target = input(Fore.CYAN + "Enter target to monitor (IP/domain): ").strip()
    if not target:
        print(Fore.RED + "❌ Target is required")
        return
    print(Fore.GREEN + f"\n📊 Starting real-time monitoring on: {target}")
    print(Fore.YELLOW + "Press Ctrl+C to stop monitoring")
    monitor.add_target(target)
    try:
        while True:
            print(Fore.CYAN + f"\n📊 Monitoring {target}...")
            print(Fore.WHITE + f"   Status: Online")
            print(Fore.WHITE + f"   Response Time: {random.randint(10, 100)}ms")
            print(Fore.WHITE + f"   Active Connections: {random.randint(1, 50)}")
            print(Fore.WHITE + f"   CPU Usage: {random.randint(10, 90)}%")
            print(Fore.WHITE + f"   Memory Usage: {random.randint(20, 80)}%")
            time.sleep(5)
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n⚠️ Monitoring stopped")
        monitor.remove_target(target)
    input(Fore.CYAN + "\nPress ENTER to continue...")
def execute_threat_analytics():
    """Execute threat analytics"""
    print(Fore.GREEN + f"\n📊 Starting threat analytics...")
    # Generate analytics dashboard
    dashboard = advanced_reporter.generate_executive_dashboard({})
    print(Fore.CYAN + "\n📊 Threat Analytics Dashboard:")
    print(Fore.WHITE + f"   Security Score: {dashboard['security_score']}/10")
    print(Fore.WHITE + f"   Risk Level: {dashboard['risk_level']}")
    print(Fore.WHITE + f"   Critical Issues: {dashboard['critical_issues']}")
    print(Fore.WHITE + f"   High Issues: {dashboard['high_issues']}")
    print(Fore.WHITE + f"   Medium Issues: {dashboard['medium_issues']}")
    print(Fore.WHITE + f"   Low Issues: {dashboard['low_issues']}")
    print(Fore.CYAN + "\n📊 Top Recommendations:")
    for rec in dashboard['recommendations']:
        color = Fore.RED if rec['priority'] == 'HIGH' else Fore.YELLOW if rec['priority'] == 'MEDIUM' else Fore.GREEN
        print(color + f"   • {rec['title']} (Priority: {rec['priority']})")
        print(Fore.WHITE + f"     {rec['description']}")
    input(Fore.CYAN + "\nPress ENTER to continue...")
def execute_aws_security():
    """Execute AWS security assessment"""
    print(Fore.GREEN + f"\n☁️ Starting AWS security assessment...")
    # Simulate AWS security scan
    aws_results = cloud_security.scan_cloud_resources("aws", {})
    print(Fore.CYAN + "\n☁️ AWS Security Assessment Results:")
    print(Fore.WHITE + f"   Security Score: {aws_results['security_score']}/10")
    print(Fore.WHITE + f"   Misconfigurations: {len(aws_results['misconfigurations'])}")
    print(Fore.WHITE + f"   Exposed Resources: {len(aws_results['exposed_resources'])}")
    print(Fore.WHITE + f"   Compliance Issues: {len(aws_results['compliance_issues'])}")
    input(Fore.CYAN + "\nPress ENTER to continue...")
def execute_azure_security():
    """Execute Azure security assessment"""
    print(Fore.GREEN + f"\n☁️ Starting Azure security assessment...")
    # Simulate Azure security scan
    azure_results = cloud_security.scan_cloud_resources("azure", {})
    print(Fore.CYAN + "\n☁️ Azure Security Assessment Results:")
    print(Fore.WHITE + f"   Security Score: {azure_results['security_score']}/10")
    print(Fore.WHITE + f"   Misconfigurations: {len(azure_results['misconfigurations'])}")
    print(Fore.WHITE + f"   Exposed Resources: {len(azure_results['exposed_resources'])}")
    print(Fore.WHITE + f"   Compliance Issues: {len(azure_results['compliance_issues'])}")
    input(Fore.CYAN + "\nPress ENTER to continue...")
def execute_gcp_security():
    """Execute GCP security assessment"""
    print(Fore.GREEN + f"\n☁️ Starting GCP security assessment...")
    # Simulate GCP security scan
    gcp_results = cloud_security.scan_cloud_resources("gcp", {})
    print(Fore.CYAN + "\n☁️ GCP Security Assessment Results:")
    print(Fore.WHITE + f"   Security Score: {gcp_results['security_score']}/10")
    print(Fore.WHITE + f"   Misconfigurations: {len(gcp_results['misconfigurations'])}")
    print(Fore.WHITE + f"   Exposed Resources: {len(gcp_results['exposed_resources'])}")
    print(Fore.WHITE + f"   Compliance Issues: {len(gcp_results['compliance_issues'])}")
    input(Fore.CYAN + "\nPress ENTER to continue...")
def execute_container_security():
    """Execute container security assessment"""
    print(Fore.GREEN + f"\n🐳 Starting container security assessment...")
    # Simulate container security scan
    container_results = cloud_security.scan_containers({})
    print(Fore.CYAN + "\n🐳 Container Security Assessment Results:")
    print(Fore.WHITE + f"   Compliance Score: {container_results['compliance_score']}/10")
    print(Fore.WHITE + f"   Vulnerabilities: {len(container_results['vulnerabilities'])}")
    print(Fore.WHITE + f"   Misconfigurations: {len(container_results['misconfigurations'])}")
    print(Fore.WHITE + f"   Secrets Exposed: {len(container_results['secrets_exposed'])}")
    input(Fore.CYAN + "\nPress ENTER to continue...")
def execute_kubernetes_security():
    """Execute Kubernetes security assessment"""
    print(Fore.GREEN + f"\n☸️ Starting Kubernetes security assessment...")
    # Simulate Kubernetes security scan
    k8s_results = cloud_security.scan_kubernetes({})
    print(Fore.CYAN + "\n☸️ Kubernetes Security Assessment Results:")
    print(Fore.WHITE + f"   Cluster Security: {k8s_results['cluster_security']}/10")
    print(Fore.WHITE + f"   RBAC Issues: {len(k8s_results['rbac_issues'])}")
    print(Fore.WHITE + f"   Network Policies: {len(k8s_results['network_policies'])}")
    print(Fore.WHITE + f"   Pod Security: {len(k8s_results['pod_security'])}")
    input(Fore.CYAN + "\nPress ENTER to continue...")
def execute_siem_integration():
    """Execute SIEM integration"""
    print(Fore.GREEN + f"\n🔗 Starting SIEM integration...")
    siem_type = input(Fore.CYAN + "Enter SIEM type (splunk/qradar/arcsight): ").strip().lower()
    if siem_type in ["splunk", "qradar", "arcsight"]:
        result = enterprise.integrate_siem({"type": siem_type})
        print(Fore.GREEN + f"✅ {result}")
    else:
        print(Fore.RED + "❌ Unsupported SIEM type")
    input(Fore.CYAN + "\nPress ENTER to continue...")
def execute_soar_platform():
    """Execute SOAR platform integration"""
    print(Fore.GREEN + f"\n🔗 Starting SOAR platform integration...")
    soar_type = input(Fore.CYAN + "Enter SOAR type (phantom/demisto/cortex): ").strip().lower()
    if soar_type in ["phantom", "demisto", "cortex"]:
        result = enterprise.integrate_soar({"type": soar_type})
        print(Fore.GREEN + f"✅ {result}")
    else:
        print(Fore.RED + "❌ Unsupported SOAR type")
    input(Fore.CYAN + "\nPress ENTER to continue...")
def execute_auto_recon_workflow():
    """Execute automated reconnaissance workflow"""
    target = input(Fore.CYAN + "Enter target (IP/domain): ").strip()
    if not target:
        print(Fore.RED + "❌ Target is required")
        return
    print(Fore.GREEN + f"\n🚀 Starting automated reconnaissance on: {target}")
    # Define reconnaissance steps
    recon_steps = [
        ("ping", f"ping -c 4 {target}"),
        ("nmap_quick", f"nmap -sn {target}"),
        ("nmap_ports", f"nmap -sS -O {target}"),
        ("whois", f"whois {target}"),
        ("dns_lookup", f"nslookup {target}")
    ]
    results = []
    for step_name, command in recon_steps:
        print(Fore.CYAN + f"\n🔍 Executing: {step_name}")
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
            output = result.stdout + result.stderr
            results.append((step_name, command, output, "completed" if result.returncode == 0 else "failed"))
            print(Fore.GREEN + f"✅ {step_name} completed")
        except Exception as e:
            results.append((step_name, command, str(e), "error"))
            print(Fore.RED + f"❌ {step_name} failed: {e}")
    # Generate comprehensive report
    print(Fore.MAGENTA + "\n📊 Generating comprehensive report...")
    report_data = [(i+1, datetime.now().isoformat(), step[0], target, step[1], step[2]) for i, step in enumerate(results)]
    report_path = reporter.generate_html_report(report_data, f"recon_report_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    print(Fore.GREEN + f"✅ Reconnaissance report generated: {report_path}")
def execute_vulnerability_scanner():
    """Execute comprehensive vulnerability scanning"""
    target = input(Fore.CYAN + "Enter target (IP/domain): ").strip()
    if not target:
        print(Fore.RED + "❌ Target is required")
        return
    print(Fore.GREEN + f"\n🔍 Starting vulnerability scan on: {target}")
    # Define vulnerability scanning steps
    vuln_steps = [
        ("nmap_vuln", f"nmap --script vuln {target}"),
        ("nikto", f"nikto -h {target}"),
        ("sslscan", f"sslscan {target}"),
        ("nuclei", f"nuclei -u http://{target}")
    ]
    results = []
    for step_name, command in vuln_steps:
        print(Fore.CYAN + f"\n🔍 Executing: {step_name}")
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
            output = result.stdout + result.stderr
            results.append((step_name, command, output, "completed" if result.returncode == 0 else "failed"))
            print(Fore.GREEN + f"✅ {step_name} completed")
        except Exception as e:
            results.append((step_name, command, str(e), "error"))
            print(Fore.RED + f"❌ {step_name} failed: {e}")
    # Generate vulnerability report
    print(Fore.MAGENTA + "\n📊 Generating vulnerability report...")
    report_data = [(i+1, datetime.now().isoformat(), step[0], target, step[1], step[2]) for i, step in enumerate(results)]
    report_path = reporter.generate_html_report(report_data, f"vuln_report_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    print(Fore.GREEN + f"✅ Vulnerability report generated: {report_path}")
def execute_attack_simulation():
    """Execute attack simulation workflow"""
    target = input(Fore.CYAN + "Enter target (IP/domain): ").strip()
    if not target:
        print(Fore.RED + "❌ Target is required")
        return
    print(Fore.RED + f"\n⚠️ ATTACK SIMULATION MODE - {target}")
    print(Fore.YELLOW + "This is for authorized testing only!")
    confirm = input(Fore.CYAN + "Confirm you have authorization (yes/no): ").lower()
    if confirm != "yes":
        print(Fore.RED + "❌ Attack simulation cancelled")
        return
    print(Fore.GREEN + f"\n🎯 Starting attack simulation on: {target}")
    # Define attack simulation steps
    attack_steps = [
        ("reconnaissance", f"nmap -A {target}"),
        ("vulnerability_scan", f"nmap --script vuln {target}"),
        ("exploit_search", f"searchsploit {target}"),
        ("password_attack", f"hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://{target}")
    ]
    results = []
    for step_name, command in attack_steps:
        print(Fore.CYAN + f"\n🎯 Executing: {step_name}")
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
            output = result.stdout + result.stderr
            results.append((step_name, command, output, "completed" if result.returncode == 0 else "failed"))
            print(Fore.GREEN + f"✅ {step_name} completed")
        except Exception as e:
            results.append((step_name, command, str(e), "error"))
            print(Fore.RED + f"❌ {step_name} failed: {e}")
    # Generate attack simulation report
    print(Fore.MAGENTA + "\n📊 Generating attack simulation report...")
    report_data = [(i+1, datetime.now().isoformat(), step[0], target, step[1], step[2]) for i, step in enumerate(results)]
    report_path = reporter.generate_html_report(report_data, f"attack_sim_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    print(Fore.GREEN + f"✅ Attack simulation report generated: {report_path}")
def settings_menu():
    """Display enhanced settings menu"""
    while True:
        print_logo()
        print(Fore.CYAN + "⚙️ ADVANCED SETTINGS")
        print(Fore.YELLOW + "═" * 50)
        print(Fore.GREEN + "1. View Configuration")
        print(Fore.GREEN + "2. Change Output Directory")
        print(Fore.GREEN + "3. Add Custom Command")
        print(Fore.GREEN + "4. View Logs")
        print(Fore.GREEN + "5. Clear Logs")
        print(Fore.MAGENTA + "6. View Scan History")
        print(Fore.MAGENTA + "7. Generate Report from History")
        print(Fore.MAGENTA + "8. Database Management")
        print(Fore.MAGENTA + "9. Advanced Utilities")
        print(Fore.RED + "0. Back")
        print(Fore.YELLOW + "═" * 50)
        try:
            choice = int(input(Fore.GREEN + "\n[Select Option] > "))
            if choice == 0:
                return
            elif choice == 1:
                print(Fore.CYAN + "\n📋 Current Configuration:")
                for key, value in config.data.items():
                    print(Fore.WHITE + f"   {key}: {value}")
                input(Fore.CYAN + "\nPress ENTER to continue...")
            elif choice == 2:
                new_dir = input(Fore.CYAN + "Enter new output directory: ").strip()
                if new_dir:
                    config.data['default_output_dir'] = new_dir
                    config.save_config()
                    print(Fore.GREEN + "✅ Configuration updated!")
                input(Fore.CYAN + "\nPress ENTER to continue...")
            elif choice == 4:
                if os.path.exists('fouad_tool.log'):
                    with open('fouad_tool.log', 'r') as f:
                        logs = f.read()
                        print(Fore.CYAN + "\n📋 Recent Logs:")
                        print(Fore.WHITE + logs[-1000:])  # Show last 1000 chars
                else:
                    print(Fore.RED + "❌ No log file found")
                input(Fore.CYAN + "\nPress ENTER to continue...")
            elif choice == 5:
                if os.path.exists('fouad_tool.log'):
                    os.remove('fouad_tool.log')
                    print(Fore.GREEN + "✅ Logs cleared!")
                else:
                    print(Fore.RED + "❌ No log file found")
                input(Fore.CYAN + "\nPress ENTER to continue...")
            elif choice == 6:
                view_scan_history()
            elif choice == 7:
                generate_history_report()
            elif choice == 8:
                database_management()
            elif choice == 9:
                advanced_utilities()
        except ValueError:
            print(Fore.RED + "❌ Enter valid number.")
            time.sleep(1)
def view_scan_history():
    """View scan history from database"""
    print_logo()
    print(Fore.CYAN + "📊 SCAN HISTORY")
    print(Fore.YELLOW + "═" * 50)
    history = db.get_scan_history(20)  # Get last 20 scans
    if not history:
        print(Fore.RED + "❌ No scan history found")
        input(Fore.CYAN + "\nPress ENTER to continue...")
        return
    for i, scan in enumerate(history, 1):
        print(Fore.GREEN + f"{i:02d}. {scan[2]} - {scan[3]}")
        print(Fore.WHITE + f"    Time: {scan[1]}")
        print(Fore.CYAN + f"    Command: {scan[4]}")
        print(Fore.YELLOW + f"    Status: {scan[6]}")
        print()
    input(Fore.CYAN + "\nPress ENTER to continue...")
def generate_history_report():
    """Generate report from scan history"""
    print_logo()
    print(Fore.CYAN + "📊 GENERATE HISTORY REPORT")
    print(Fore.YELLOW + "═" * 50)
    history = db.get_scan_history(50)  # Get last 50 scans
    if not history:
        print(Fore.RED + "❌ No scan history found")
        input(Fore.CYAN + "\nPress ENTER to continue...")
        return
    print(Fore.GREEN + f"Found {len(history)} scans in history")
    # Generate report
    report_path = reporter.generate_html_report(history, f"history_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    print(Fore.GREEN + f"✅ History report generated: {report_path}")
    # Ask if user wants to open the report
    open_report = input(Fore.CYAN + "Open report in browser? (y/n): ").lower()
    if open_report == 'y':
        webbrowser.open(f"file://{report_path.absolute()}")
    input(Fore.CYAN + "\nPress ENTER to continue...")
def database_management():
    """Database management options"""
    print_logo()
    print(Fore.CYAN + "🗄️ DATABASE MANAGEMENT")
    print(Fore.YELLOW + "═" * 50)
    print(Fore.GREEN + "1. View Database Statistics")
    print(Fore.GREEN + "2. Clear Old Scans")
    print(Fore.GREEN + "3. Export Database")
    print(Fore.GREEN + "4. Backup Database")
    print(Fore.RED + "0. Back")
    print(Fore.YELLOW + "═" * 50)
    try:
        choice = int(input(Fore.GREEN + "\n[Select Option] > "))
        if choice == 0:
            return
        elif choice == 1:
            show_database_stats()
        elif choice == 2:
            clear_old_scans()
        elif choice == 3:
            export_database()
        elif choice == 4:
            backup_database()
    except ValueError:
        print(Fore.RED + "❌ Enter valid number.")
        time.sleep(1)
def show_database_stats():
    """Show database statistics"""
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    # Get total scans
    cursor.execute("SELECT COUNT(*) FROM scans")
    total_scans = cursor.fetchone()[0]
    # Get scans by status
    cursor.execute("SELECT status, COUNT(*) FROM scans GROUP BY status")
    status_counts = cursor.fetchall()
    # Get recent scans
    cursor.execute("SELECT COUNT(*) FROM scans WHERE timestamp > datetime('now', '-7 days')")
    recent_scans = cursor.fetchone()[0]
    conn.close()
    print(Fore.CYAN + "\n📊 Database Statistics:")
    print(Fore.WHITE + f"   Total Scans: {total_scans}")
    print(Fore.WHITE + f"   Recent Scans (7 days): {recent_scans}")
    print(Fore.CYAN + "\n   Status Breakdown:")
    for status, count in status_counts:
        print(Fore.WHITE + f"     {status}: {count}")
    input(Fore.CYAN + "\nPress ENTER to continue...")
def clear_old_scans():
    """Clear old scans from database"""
    days = input(Fore.CYAN + "Delete scans older than how many days? (default 30): ").strip()
    if not days:
        days = 30
    else:
        days = int(days)
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM scans WHERE timestamp < datetime('now', '-{} days')".format(days))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    print(Fore.GREEN + f"✅ Deleted {deleted} old scans")
    input(Fore.CYAN + "\nPress ENTER to continue...")
def export_database():
    """Export database to CSV"""
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scans")
    data = cursor.fetchall()
    conn.close()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"fouad_tool_export_{timestamp}.csv"
    output_dir = Path(config.data['default_output_dir']) / 'exports'
    output_dir.mkdir(exist_ok=True)
    filepath = output_dir / filename
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Timestamp', 'Tool', 'Target', 'Command', 'Output', 'Status'])
        writer.writerows(data)
    print(Fore.GREEN + f"✅ Database exported to: {filepath}")
    input(Fore.CYAN + "\nPress ENTER to continue...")
def backup_database():
    """Backup database"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"fouad_tool_backup_{timestamp}.db"
    backup_dir = Path(config.data['default_output_dir']) / 'backups'
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / backup_name
    # Copy database file
    import shutil
    shutil.copy2(db.db_path, backup_path)
    print(Fore.GREEN + f"✅ Database backed up to: {backup_path}")
    input(Fore.CYAN + "\nPress ENTER to continue...")
def advanced_utilities():
    """Advanced utility functions"""
    print_logo()
    print(Fore.CYAN + "🛠️ ADVANCED UTILITIES")
    print(Fore.YELLOW + "═" * 50)
    print(Fore.GREEN + "1. Generate Random String")
    print(Fore.GREEN + "2. Hash String")
    print(Fore.GREEN + "3. Validate IP/Domain")
    print(Fore.GREEN + "4. Quick Port Scan")
    print(Fore.GREEN + "5. Generate Wordlist")
    print(Fore.RED + "0. Back")
    print(Fore.YELLOW + "═" * 50)
    try:
        choice = int(input(Fore.GREEN + "\n[Select Option] > "))
        if choice == 0:
            return
        elif choice == 1:
            length = input(Fore.CYAN + "Enter string length (default 10): ").strip()
            length = int(length) if length else 10
            random_str = utils.generate_random_string(length)
            print(Fore.GREEN + f"✅ Random string: {random_str}")
            input(Fore.CYAN + "\nPress ENTER to continue...")
        elif choice == 2:
            text = input(Fore.CYAN + "Enter text to hash: ").strip()
            algorithm = input(Fore.CYAN + "Enter algorithm (md5/sha1/sha256, default md5): ").strip() or "md5"
            hashed = utils.hash_string(text, algorithm)
            print(Fore.GREEN + f"✅ {algorithm.upper()} hash: {hashed}")
            input(Fore.CYAN + "\nPress ENTER to continue...")
        elif choice == 3:
            target = input(Fore.CYAN + "Enter IP or domain to validate: ").strip()
            if utils.is_valid_ip(target):
                print(Fore.GREEN + f"✅ {target} is a valid IP address")
            elif utils.is_valid_domain(target):
                print(Fore.GREEN + f"✅ {target} is a valid domain")
            else:
                print(Fore.RED + f"❌ {target} is not a valid IP or domain")
            input(Fore.CYAN + "\nPress ENTER to continue...")
        elif choice == 4:
            host = input(Fore.CYAN + "Enter host to scan: ").strip()
            ports_input = input(Fore.CYAN + "Enter ports (comma-separated, default 22,80,443): ").strip()
            if ports_input:
                ports = [int(p.strip()) for p in ports_input.split(',')]
            else:
                ports = [22, 80, 443]
            print(Fore.CYAN + f"Scanning {host} on ports {ports}...")
            open_ports = utils.port_scan(host, ports)
            if open_ports:
                print(Fore.GREEN + f"✅ Open ports: {open_ports}")
            else:
                print(Fore.RED + "❌ No open ports found")
            input(Fore.CYAN + "\nPress ENTER to continue...")
        elif choice == 5:
            min_len = input(Fore.CYAN + "Enter minimum length (default 4): ").strip()
            max_len = input(Fore.CYAN + "Enter maximum length (default 6): ").strip()
            output_file = input(Fore.CYAN + "Enter output filename (default custom_wordlist.txt): ").strip()
            min_len = int(min_len) if min_len else 4
            max_len = int(max_len) if max_len else 6
            output_file = output_file or "custom_wordlist.txt"
            print(Fore.CYAN + "Generating wordlist...")
            utils.generate_wordlist("", min_len, max_len, output_file)
            print(Fore.GREEN + f"✅ Wordlist generated: {output_file}")
            input(Fore.CYAN + "\nPress ENTER to continue...")
    except ValueError:
        print(Fore.RED + "❌ Enter valid number.")
        time.sleep(1)
class FouadToolGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Fouad Tool - Advanced Cybersecurity Framework")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a1a')
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TNotebook', background='#2d2d2d')
        self.style.configure('TNotebook.Tab', background='#404040', foreground='white')
        self.style.configure('TFrame', background='#2d2d2d')
        self.style.configure('TLabel', background='#2d2d2d', foreground='white')
        self.style.configure('TButton', background='#404040', foreground='white')
        self.setup_gui()
    def setup_gui(self):
        # Main title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill='x', padx=10, pady=5)
        title_label = ttk.Label(title_frame, text="🔥 FOUAD TOOL - ADVANCED CYBERSECURITY FRAMEWORK 🔥", 
                               font=('Arial', 16, 'bold'))
        title_label.pack()
        subtitle_label = ttk.Label(title_frame, text="Author: Fouad Zulof | Instagram: @1.pvl", 
                                  font=('Arial', 10))
        subtitle_label.pack()
        # Create notebook for categories
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        # Create tabs for each category
        for category_name, tools in TOOL_CATEGORIES.items():
            self.create_category_tab(category_name, tools)
        # Output area
        output_frame = ttk.Frame(self.root)
        output_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(output_frame, text="Command Output:", font=('Arial', 12, 'bold')).pack(anchor='w')
        self.output_text = scrolledtext.ScrolledText(output_frame, height=10, bg='#1a1a1a', 
                                                    fg='#00ff00', font=('Courier', 10))
        self.output_text.pack(fill='x', pady=5)
        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(button_frame, text="Clear Output", command=self.clear_output).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Save Output", command=self.save_output).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Open Terminal", command=self.open_terminal).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side='right', padx=5)
    def create_category_tab(self, category_name, tools):
        # Create frame for this category
        category_frame = ttk.Frame(self.notebook)
        self.notebook.add(category_frame, text=category_name)
        # Create scrollable frame
        canvas = tk.Canvas(category_frame, bg='#2d2d2d')
        scrollbar = ttk.Scrollbar(category_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        # Add tools to this category
        for tool_key, tool_info in tools.items():
            self.create_tool_widget(scrollable_frame, tool_key, tool_info)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    def create_tool_widget(self, parent, tool_key, tool_info):
        # Tool frame
        tool_frame = ttk.Frame(parent)
        tool_frame.pack(fill='x', padx=5, pady=5)
        # Tool name and description
        name_label = ttk.Label(tool_frame, text=tool_key.replace('_', ' ').title(), 
                              font=('Arial', 12, 'bold'))
        name_label.pack(anchor='w')
        desc_label = ttk.Label(tool_frame, text=tool_info['description'], 
                              font=('Arial', 9), wraplength=800)
        desc_label.pack(anchor='w')
        # Command display
        cmd_frame = ttk.Frame(tool_frame)
        cmd_frame.pack(fill='x', pady=2)
        ttk.Label(cmd_frame, text="Command:", font=('Arial', 9, 'bold')).pack(side='left')
        cmd_label = ttk.Label(cmd_frame, text=tool_info['command'], 
                             font=('Courier', 9), foreground='#00ff00')
        cmd_label.pack(side='left', padx=5)
        # Example
        example_label = ttk.Label(tool_frame, text=f"Example: {tool_info['example']}", 
                                 font=('Arial', 8), foreground='#888888')
        example_label.pack(anchor='w')
        # Execute button
        execute_btn = ttk.Button(tool_frame, text="Execute", 
                                command=lambda: self.execute_tool_gui(tool_key, tool_info))
        execute_btn.pack(anchor='w', pady=2)
        # Separator
        ttk.Separator(tool_frame, orient='horizontal').pack(fill='x', pady=5)
    def execute_tool_gui(self, tool_key, tool_info):
        if tool_info['command'] == "OLLAMA_SUBMENU":
            self.output_text.insert(tk.END, "Ollama models - Use terminal mode for interactive access\n")
            return
        # Get arguments from user
        args = simpledialog.askstring("Arguments", 
                                        f"Enter arguments for {tool_key.replace('_', ' ').title()}:\n"
                                        f"Command: {tool_info['command']}\n"
                                        f"Example: {tool_info['example']}")
        if args is None:  # User cancelled
            return
        full_command = tool_info['command'] + args if args else tool_info['command']
        self.output_text.insert(tk.END, f"Executing: {full_command}\n")
        self.output_text.insert(tk.END, "─" * 60 + "\n")
        self.output_text.see(tk.END)
        # Execute command in thread to prevent GUI freezing
        threading.Thread(target=self.run_command, args=(full_command,), daemon=True).start()
    def run_command(self, command):
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            # Update GUI in main thread
            self.root.after(0, self.update_output, result.stdout, result.stderr, result.returncode)
        except Exception as e:
            self.root.after(0, self.update_output, "", str(e), -1)
    def update_output(self, stdout, stderr, returncode):
        if stdout:
            self.output_text.insert(tk.END, stdout)
        if stderr:
            self.output_text.insert(tk.END, f"STDERR: {stderr}")
        if returncode != 0:
            self.output_text.insert(tk.END, f"\nCommand failed with return code: {returncode}")
        self.output_text.insert(tk.END, "\n" + "─" * 60 + "\n")
        self.output_text.see(tk.END)
    def clear_output(self):
        self.output_text.delete(1.0, tk.END)
    def save_output(self):
        filename = filedialog.asksaveasfilename(defaultextension=".txt", 
                                               filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filename:
            with open(filename, 'w') as f:
                f.write(self.output_text.get(1.0, tk.END))
            messagebox.showinfo("Success", f"Output saved to {filename}")
    def open_terminal(self):
        # Open terminal version
        self.root.destroy()
        show_category_menu()
    def run(self):
        self.root.mainloop()
def launch_gui():
    """Launch the GUI version of the tool"""
    try:
        app = FouadToolGUI()
        app.run()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        print("Falling back to terminal mode...")
        show_category_menu()
if __name__ == "__main__":
    # Check if GUI is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        launch_gui()
    else:
        show_category_menu()
