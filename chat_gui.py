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
    QTextEdit, QPushButton, QLabel, QSpinBox, QProgressBar, QMessageBox, QComboBox
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
        self.prompt_file = None
        
    def run(self):
        try:
            # 使用临时文件传递 prompt，避免命令行编码问题
            import tempfile
            
            self.prompt_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False)
            self.prompt_file.write(self.prompt)
            self.prompt_file.close()
            
            cmd = [
                str(self.llama_path),
                "-m", str(self.model_path),
                "-f", self.prompt_file.name,  # 使用文件而不是 -p 参数
                "-n", str(self.params.get('max_tokens', 2048)),
                "--temp", str(self.params.get('temperature', 0.7)),
                "-c", str(self.params.get('context_size', 4096)),
                "-ngl", str(self.params.get('gpu_layers', 0)),
                "--repeat-penalty", str(self.params.get('repeat_penalty', 1.1)),
                "--no-display-prompt",
                "-r", "用户:"
            ]
            
            # 如果使用 CPU 模式，强制指定不使用 GPU 设备，避免 Vulkan 初始化
            if self.params.get('gpu_layers', 0) == 0:
                cmd.append("--device")
                cmd.append("none")
            
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
            
            # 清理临时文件
            try:
                import os
                os.unlink(self.prompt_file.name)
            except:
                pass
            
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
        # 清理临时文件
        if self.prompt_file:
            try:
                import os
                os.unlink(self.prompt_file.name)
            except:
                pass


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qwen30B 大模型对话程序")
        self.setGeometry(100, 100, 900, 700)
        
        # 回退到CPU模式的标志
        self.fallback_to_cpu = False
        
        # 模型配置：名称、文件名、下载链接、大小描述
        self.models_config = {
            'qwen30b': {
                'name': 'Qwen3-30B-A3B (约18GB)',
                'filename': 'Qwen3-30B-A3B-Q4_K_M.gguf',
                'url': 'https://hf-mirror.com/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf',
                'description': '大模型，效果好但速度慢'
            },
            'qwen7b': {
                'name': 'Qwen2.5-7B-Instruct (约4.5GB)',
                'filename': 'Qwen2.5-7B-Instruct-Q4_K_M.gguf',
                'url': 'https://hf-mirror.com/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf',
                'description': '中型模型，适合CPU，速度快'
            }
        }
        
        # 模型路径检测
        if getattr(sys, 'frozen', False):
            self.app_dir = Path(sys.executable).parent
            llama_paths = [
                self.app_dir / "llama-cli.exe",
                self.app_dir / "_internal" / "llama-cli.exe",
            ]
            
            self.llama_path = None
            for path in llama_paths:
                if path.exists():
                    self.llama_path = path
                    break
            
            if not self.llama_path:
                self.llama_path = llama_paths[1]
        else:
            self.app_dir = Path(__file__).parent
            self.llama_path = self.app_dir / "llama-cli.exe"
        
        # 默认使用 Qwen2.5-7B 模型（速度更快）
        self.current_model_key = 'qwen7b'
        self.model_path = self.app_dir / "model" / self.models_config[self.current_model_key]['filename']
        self.model_url = self.models_config[self.current_model_key]['url']
        
        # 读取知识库文件夹中的所有 txt 文件
        self.knowledge_base = self.load_knowledge_base()
        
        self.chat_history = []
        self.current_thread = None
        self.download_thread = None
        
        self.init_ui()
        QTimer.singleShot(100, self.check_model)
        QTimer.singleShot(200, self.detect_gpu_and_set_default)
        # 默认使用 CPU 模式，跳过 Vulkan 测试以节省启动时间
        self.fallback_to_cpu = True
        self.gpu_spin.setValue(0)
    
    def load_knowledge_base(self):
        # 尝试从 knowledge 文件夹读取所有 txt 文件
        knowledge_paths = [
            self.app_dir / "knowledge",
            self.app_dir / "_internal" / "knowledge",
        ]
        
        all_content = ""
        for kb_dir in knowledge_paths:
            if kb_dir.exists() and kb_dir.is_dir():
                for txt_file in sorted(kb_dir.glob("*.txt")):
                    try:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            all_content += f"=== {txt_file.name} ===\n"
                            all_content += content
                            all_content += "\n\n"
                    except Exception as e:
                        pass
        
        return all_content
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 标题
        title = QLabel("Qwen30B 大模型对话程序")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 模型选择
        model_layout = QHBoxLayout()
        model_label = QLabel("选择模型:")
        model_label.setFont(QFont("Microsoft YaHei", 11))
        model_layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        for key, config in self.models_config.items():
            self.model_combo.addItem(f"{config['name']} - {config['description']}", key)
        self.model_combo.setCurrentText(f"{self.models_config[self.current_model_key]['name']} - {self.models_config[self.current_model_key]['description']}")
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
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
        self.download_button = QPushButton("下载模型")
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
        
    def detect_gpu_and_set_default(self):
        # 如果已经设置为 CPU 模式，不再覆盖
        if self.fallback_to_cpu:
            self.gpu_spin.setValue(0)
            self.model_status.setText(f"{self.model_status.text()} | 使用CPU模式")
            return
        
        vram_gb = self.get_gpu_vram()
        gpu_name = self.get_gpu_name()
        
        if vram_gb is None:
            self.gpu_spin.setValue(0)
            self.model_status.setText(f"{self.model_status.text()} | 未检测到GPU，使用CPU模式")
            return
        
        if vram_gb < 8:
            gpu_layers = 0
            self.model_status.setText(f"{self.model_status.text()} | GPU: {gpu_name} ({vram_gb:.0f}GB)，显存不足，使用CPU模式")
        elif vram_gb < 12:
            gpu_layers = 25
            self.model_status.setText(f"{self.model_status.text()} | GPU: {gpu_name} ({vram_gb:.0f}GB)，建议GPU层: {gpu_layers}")
        elif vram_gb < 16:
            gpu_layers = 40
            self.model_status.setText(f"{self.model_status.text()} | GPU: {gpu_name} ({vram_gb:.0f}GB)，建议GPU层: {gpu_layers}")
        elif vram_gb < 24:
            gpu_layers = 60
            self.model_status.setText(f"{self.model_status.text()} | GPU: {gpu_name} ({vram_gb:.0f}GB)，建议GPU层: {gpu_layers}")
        else:
            gpu_layers = 99
            self.model_status.setText(f"{self.model_status.text()} | GPU: {gpu_name} ({vram_gb:.0f}GB)，建议GPU层: {gpu_layers}")
        
        self.gpu_spin.setValue(gpu_layers)
        
    def check_vulkan_support(self):
        if not self.llama_path.exists():
            return
        
        if not self.model_path.exists():
            return
        
        vulkan_dll_exists = True
        try:
            import ctypes
            ctypes.WinDLL('vulkan-1.dll')
        except Exception:
            vulkan_dll_exists = False
        
        if not vulkan_dll_exists:
            self.fallback_to_cpu = True
            self.model_status.setText(f"{self.model_status.text()} | Vulkan DLL 不存在，使用CPU模式")
            self.gpu_spin.setValue(0)
            self.add_message("系统", "系统未安装 Vulkan 运行时，已自动切换到 CPU 模式。")
            return
        
        self.add_message("系统", "正在检测 Vulkan GPU 加速兼容性...")
        
        import tempfile
        
        test_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False)
        test_file.write("test")
        test_file.close()
        
        test_thread = threading.Thread(target=self._run_vulkan_test, args=(test_file.name,))
        test_thread.start()
    
    def _run_vulkan_test(self, test_file_path):
        try:
            result = subprocess.run(
                [str(self.llama_path), '-m', str(self.model_path), '-f', test_file_path, 
                 '-n', '1', '-ngl', '1', '--no-display-prompt', '-c', '512'],
                capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=60
            )
            
            try:
                os.unlink(test_file_path)
            except:
                pass
            
            stderr_lower = result.stderr.lower() if result.stderr else ""
            
            if result.returncode != 0 or 'ggml_vulkan' in stderr_lower and 'error' in stderr_lower:
                self.fallback_to_cpu = True
                QTimer.singleShot(0, lambda: self._update_vulkan_status(False, "Vulkan 运行时错误，使用CPU模式"))
            else:
                QTimer.singleShot(0, lambda: self._update_vulkan_status(True, "Vulkan 支持 ✓"))
                
        except subprocess.TimeoutExpired:
            try:
                os.unlink(test_file_path)
            except:
                pass
            self.fallback_to_cpu = True
            QTimer.singleShot(0, lambda: self._update_vulkan_status(False, "Vulkan 检测超时，使用CPU模式"))
        except Exception as e:
            try:
                os.unlink(test_file_path)
            except:
                pass
            self.fallback_to_cpu = True
            QTimer.singleShot(0, lambda: self._update_vulkan_status(False, f"Vulkan 检测异常: {str(e)[:30]}"))
    
    def _update_vulkan_status(self, success, message):
        if success:
            self.model_status.setText(f"{self.model_status.text()} | {message}")
        else:
            self.fallback_to_cpu = True
            self.model_status.setText(f"{self.model_status.text()} | {message}")
            self.gpu_spin.setValue(0)
            self.add_message("系统", "检测到 Vulkan GPU 加速存在兼容性问题，已自动切换到 CPU 模式。")
        
    def get_gpu_vram(self):
        try:
            import subprocess
            # 使用 nvidia-smi 获取 NVIDIA 显卡显存
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    # nvidia-smi 返回 MB，转换为 GB
                    vram_mb = int(lines[0].strip())
                    return vram_mb / 1024
        except:
            pass
        
        # 如果 nvidia-smi 失败，尝试使用 WMI
        try:
            result = subprocess.run(
                ['powershell', '-Command', 
                 'Get-WmiObject -Class Win32_VideoController | Select-Object AdapterRAM | ConvertTo-Json'],
                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
            )
            import json
            data = json.loads(result.stdout)
            
            max_vram = 0
            if isinstance(data, list):
                for gpu in data:
                    vram = gpu.get('AdapterRAM', 0) / (1024**3)
                    if vram > max_vram:
                        max_vram = vram
            else:
                max_vram = data.get('AdapterRAM', 0) / (1024**3)
            
            # 如果 WMI 返回的值太小或为 0，认为显存不足
            return max_vram if max_vram >= 4 else None
        except:
            return None
        
    def get_gpu_name(self):
        try:
            import subprocess
            # 优先使用 nvidia-smi 获取 NVIDIA 显卡名称
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except:
            pass
        
        # 如果 nvidia-smi 失败，使用 WMI
        try:
            result = subprocess.run(
                ['powershell', '-Command', 
                 'Get-WmiObject -Class Win32_VideoController | Select-Object Name | ConvertTo-Json'],
                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
            )
            import json
            data = json.loads(result.stdout)
            
            if isinstance(data, list):
                for gpu in data:
                    name = gpu.get('Name', '')
                    if name and 'NVIDIA' in name:
                        return name
                return data[0].get('Name', '未知') if data else '未知'
            else:
                return data.get('Name', '未知')
        except:
            return '未知'
        
    def check_model(self):
        if self.model_path.exists():
            size_gb = self.model_path.stat().st_size / (1024**3)
            self.model_status.setText(f"模型就绪 ({size_gb:.1f} GB)")
            self.input_field.setEnabled(True)
            self.send_button.setEnabled(True)
            model_name = self.models_config[self.current_model_key]['name']
            self.add_message("系统", f"欢迎使用 {model_name} 大模型对话程序！")
        else:
            self.model_status.setText("模型未找到")
            self.download_button.setVisible(True)
            self.download_button.setEnabled(True)
            model_info = self.models_config[self.current_model_key]
            self.add_message("系统", f"模型文件未找到。请点击下方'下载模型'按钮从国内镜像下载。\n当前选择: {model_info['name']}")
            
    def on_model_changed(self, index):
        self.current_model_key = self.model_combo.currentData()
        config = self.models_config[self.current_model_key]
        self.model_path = self.app_dir / "model" / config['filename']
        self.model_url = config['url']
        
        self.add_message("系统", f"已切换到: {config['name']}")
        self.check_model()
            
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
        self.run_inference()
        
    def run_inference(self):
        self.send_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        prompt = self.build_prompt()
        
        gpu_layers = 0 if self.fallback_to_cpu else self.gpu_spin.value()
        params = {
            'temperature': self.temp_spin.value() / 100,
            'context_size': self.ctx_spin.value(),
            'gpu_layers': gpu_layers,
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
        # 构建系统提示词
        prompt = "你是一个有帮助的AI助手，请用中文回答用户的问题。\n\n"
        
        # 如果有知识库，将其添加到提示词中
        if self.knowledge_base:
            prompt += "以下是公司的规章制度知识库，当用户询问关于公司实习生管理、实习补贴、实习流程等相关问题时，请根据此知识库的内容进行回答：\n\n"
            prompt += "---知识库开始---\n"
            prompt += self.knowledge_base
            prompt += "\n---知识库结束---\n\n"
            prompt += "请注意：当用户问及知识库相关问题时，请根据知识库内容准确回答；对于知识库以外的其他问题，请根据你的通用知识进行回答。\n\n"
        
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
        
        # 检测是否是GPU相关错误，如果是则尝试回退到CPU模式
        gpu_error_keywords = ['ggml_vulkan', 'Missing op', 'Vulkan', 'GPU']
        if any(keyword in error for keyword in gpu_error_keywords) and not self.fallback_to_cpu:
            self.fallback_to_cpu = True
            self.add_message("系统", "检测到GPU错误，自动切换到CPU模式重新尝试...")
            
            # 重新运行推理，使用CPU模式（不会重复添加用户消息）
            self.run_inference()
            return
        
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