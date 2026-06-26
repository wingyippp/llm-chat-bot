"""
Qwen30B 大模型对话程序 - Windows GUI版本
使用 llama.cpp 进行推理，支持自动从国内镜像下载模型
"""

import sys
import os
import subprocess
import threading
import json
import urllib.request
import ssl
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QSpinBox, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont


class ModelDownloadThread(QThread):
    progress_updated = pyqtSignal(int, int, str)
    download_completed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model_url, model_path):
        super().__init__()
        self.model_url = model_url
        self.model_path = model_path
        self.running = True
        
    def run(self):
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            response = urllib.request.urlopen(self.model_url, context=ctx)
            total_size = int(response.headers.get('Content-Length', 0))
            
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            downloaded = 0
            chunk_size = 1024 * 1024
            
            with open(self.model_path, 'wb') as f:
                while self.running:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    speed_msg = f"{downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB"
                    self.progress_updated.emit(downloaded, total_size, speed_msg)
            
            if downloaded == total_size:
                self.download_completed.emit(self.model_path)
            else:
                self.error_occurred.emit("下载不完整")
                
        except Exception as e:
            self.error_occurred.emit(str(e))
            
    def stop(self):
        self.running = False


class InferenceThread(QThread):
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished_response = pyqtSignal()
    
    def __init__(self, llama_path, model_path, prompt, params):
        super().__init__()
        self.llama_path = llama_path
        self.model_path = model_path
        self.prompt = prompt
        self.params = params
        self.process = None
        
    def run(self):
        try:
            cmd = [
                str(self.llama_path),
                "-m", str(self.model_path),
                "-p", self.prompt,
                "-n", str(self.params.get('max_tokens', 2048)),
                "--temp", str(self.params.get('temperature', 0.7)),
                "-c", str(self.params.get('context_size', 4096)),
                "-ngl", str(self.params.get('gpu_layers', 0)),
                "--repeat-penalty", str(self.params.get('repeat_penalty', 1.1)),
                "--no-display-prompt",
                "-r", "用户:"
            ]
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='ignore',
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            full_response = ""
            for line in self.process.stdout:
                if line:
                    full_response += line
                    self.response_received.emit(line)
            
            self.process.wait()
            
            if self.process.returncode == 0:
                self.finished_response.emit()
            else:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                if stderr:
                    self.error_occurred.emit(f"推理错误: {stderr}")
                else:
                    self.finished_response.emit()
                    
        except Exception as e:
            self.error_occurred.emit(str(e))
            
    def stop(self):
        if self.process:
            self.process.terminate()


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qwen30B 大模型对话程序")
        self.setGeometry(100, 100, 900, 700)
        
        # 模型路径检测
        if getattr(sys, 'frozen', False):
            self.app_dir = Path(sys.executable).parent
            model_path_1 = self.app_dir / "model" / "Qwen3-30B-A3B-Q4_K_M.gguf"
            model_path_2 = self.app_dir / "_internal" / "model" / "Qwen3-30B-A3B-Q4_K_M.gguf"
            llama_path_1 = self.app_dir / "llama-cli.exe"
            llama_path_2 = self.app_dir / "_internal" / "llama-cli.exe"
            
            self.model_path = model_path_1 if model_path_1.exists() else model_path_2
            self.llama_path = llama_path_1 if llama_path_1.exists() else llama_path_2
        else:
            self.app_dir = Path(__file__).parent
            self.llama_path = self.app_dir / "llama-cli.exe"
            self.model_path = self.app_dir / "model" / "Qwen3-30B-A3B-Q4_K_M.gguf"
        
        self.model_url = "https://hf-mirror.com/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf"
        
        self.chat_history = []
        self.current_thread = None
        self.download_thread = None
        
        self.init_ui()
        QTimer.singleShot(100, self.check_model)
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 标题
        title = QLabel("Qwen30B 大模型对话程序")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 模型状态
        self.model_status = QLabel("模型状态: 检查中...")
        self.model_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.model_status)
        
        # 对话显示区域
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Microsoft YaHei", 13))
        self.chat_display.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd; padding: 10px;")
        layout.addWidget(self.chat_display, stretch=1)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 下载按钮
        self.download_button = QPushButton("下载模型 (约18GB)")
        self.download_button.setVisible(False)
        self.download_button.clicked.connect(self.download_model)
        layout.addWidget(self.download_button)
        
        # 输入区域
        input_layout = QHBoxLayout()
        
        self.input_field = QTextEdit()
        self.input_field.setFont(QFont("Microsoft YaHei", 13))
        self.input_field.setMaximumHeight(100)
        self.input_field.setPlaceholderText("输入您的问题...")
        input_layout.addWidget(self.input_field, stretch=1)
        
        self.send_button = QPushButton("发送")
        self.send_button.setFont(QFont("Microsoft YaHei", 13))
        self.send_button.setMinimumWidth(80)
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        self.stop_button = QPushButton("停止")
        self.stop_button.setFont(QFont("Microsoft YaHei", 13))
        self.stop_button.setMinimumWidth(80)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_generation)
        input_layout.addWidget(self.stop_button)
        
        layout.addLayout(input_layout)
        
        # 参数设置
        params_layout = QHBoxLayout()
        
        params_layout.addWidget(QLabel("温度:"))
        self.temp_spin = QSpinBox()
        self.temp_spin.setRange(0, 100)
        self.temp_spin.setValue(70)
        params_layout.addWidget(self.temp_spin)
        
        params_layout.addWidget(QLabel("上下文:"))
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(512, 16384)
        self.ctx_spin.setValue(4096)
        params_layout.addWidget(self.ctx_spin)
        
        params_layout.addWidget(QLabel("GPU层:"))
        self.gpu_spin = QSpinBox()
        self.gpu_spin.setRange(0, 99)
        self.gpu_spin.setValue(0)
        params_layout.addWidget(self.gpu_spin)
        
        self.clear_button = QPushButton("清除对话")
        self.clear_button.clicked.connect(self.clear_chat)
        params_layout.addWidget(self.clear_button)
        
        params_layout.addStretch()
        layout.addLayout(params_layout)
        
        # 初始禁用输入
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        
    def check_model(self):
        if self.model_path.exists():
            size_gb = self.model_path.stat().st_size / (1024**3)
            self.model_status.setText(f"模型就绪 ({size_gb:.1f} GB)")
            self.input_field.setEnabled(True)
            self.send_button.setEnabled(True)
            self.add_message("系统", "欢迎使用 Qwen30B 大模型对话程序！")
        else:
            self.model_status.setText("模型未找到")
            self.download_button.setVisible(True)
            self.download_button.setEnabled(True)
            self.add_message("系统", "模型文件未找到。请点击下方'下载模型'按钮从国内镜像下载。\n模型大小约 18GB，下载时间取决于网络速度。")
            
    def download_model(self):
        self.download_button.setEnabled(False)
        self.download_button.setText("正在下载...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.download_thread = ModelDownloadThread(
            self.model_url, 
            str(self.model_path)
        )
        self.download_thread.progress_updated.connect(self.on_download_progress)
        self.download_thread.download_completed.connect(self.on_download_completed)
        self.download_thread.error_occurred.connect(self.on_download_error)
        self.download_thread.start()
        
    def on_download_progress(self, downloaded, total, speed):
        if total > 0:
            progress = int((downloaded / total) * 100)
            self.progress_bar.setValue(progress)
            self.progress_bar.setFormat(f"{speed} ({progress}%)")
        else:
            self.progress_bar.setFormat(speed)
            
    def on_download_completed(self, model_path):
        self.progress_bar.setVisible(False)
        self.download_button.setVisible(False)
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.model_status.setText("模型已下载 ✓")
        QMessageBox.information(self, "下载完成", f"模型已下载到:\n{model_path}\n\n可以开始对话了！")
        
    def on_download_error(self, error):
        self.progress_bar.setVisible(False)
        self.download_button.setEnabled(True)
        self.download_button.setText("下载模型")
        QMessageBox.critical(self, "下载错误", f"模型下载失败:\n{error}")
        
    def add_message(self, role, content):
        self.chat_display.append(f"{role}: {content}\n")
        
    def send_message(self):
        user_input = self.input_field.toPlainText().strip()
        if not user_input:
            return
            
        self.add_message("用户", user_input)
        self.chat_history.append({"role": "user", "content": user_input})
        
        self.input_field.clear()
        self.send_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        prompt = self.build_prompt()
        
        params = {
            'temperature': self.temp_spin.value() / 100,
            'context_size': self.ctx_spin.value(),
            'gpu_layers': self.gpu_spin.value(),
            'max_tokens': 2048
        }
        
        self.current_thread = InferenceThread(
            self.llama_path,
            self.model_path,
            prompt,
            params
        )
        self.current_thread.response_received.connect(self.on_response)
        self.current_thread.error_occurred.connect(self.on_error)
        self.current_thread.finished_response.connect(self.on_finished)
        self.current_thread.start()
        
    def build_prompt(self):
        prompt = "你是一个有帮助的AI助手，请用中文回答用户的问题。\n\n"
        
        for msg in self.chat_history[-10:]:
            if msg['role'] == 'user':
                prompt += f"用户: {msg['content']}\n"
            else:
                prompt += f"助手: {msg['content']}\n"
        
        prompt += "助手: "
        return prompt
        
    def on_response(self, text):
        # 直接显示，不做过滤
        self.chat_display.insertPlainText(text)
        
    def on_error(self, error):
        self.add_message("系统", f"错误: {error}")
        self.reset_ui()
        
    def on_finished(self):
        self.chat_display.append("\n")
        self.reset_ui()
        
    def reset_ui(self):
        self.send_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
    def stop_generation(self):
        if self.current_thread:
            self.current_thread.stop()
            self.add_message("系统", "已停止生成")
            self.reset_ui()
            
    def clear_chat(self):
        self.chat_display.clear()
        self.chat_history.clear()
        self.add_message("系统", "对话已清除，可以开始新的对话。")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()