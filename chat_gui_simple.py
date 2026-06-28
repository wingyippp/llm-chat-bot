"""
Qwen30B 大模型对话程序 - Windows GUI版本
使用 Ollama API 进行推理（无需单独下载模型）
"""

import sys
import os
import requests
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QSpinBox, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QColor


class OllamaThread(QThread):
    """调用 Ollama API 的线程"""
    
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished_response = pyqtSignal()
    
    def __init__(self, api_url, model, messages, stream=True):
        super().__init__()
        self.api_url = api_url
        self.model = model
        self.messages = messages
        self.stream = stream
        self.running = True
        
    def run(self):
        """调用 Ollama API"""
        try:
            full_response = ""
            
            if self.stream:
                # 流式响应
                with requests.post(
                    self.api_url,
                    json={"model": self.model, "messages": self.messages, "stream": True},
                    stream=True,
                    timeout=300
                ) as response:
                    for line in response.iter_lines():
                        if not self.running:
                            break
                        if line:
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    content = data["message"]["content"]
                                    full_response += content
                                    self.response_received.emit(content)
                            except:
                                pass
            else:
                # 非流式响应
                response = requests.post(
                    self.api_url,
                    json={"model": self.model, "messages": self.messages},
                    timeout=300
                )
                if response.status_code == 200:
                    data = response.json()
                    if "message" in data and "content" in data["message"]:
                        full_response = data["message"]["content"]
                        self.response_received.emit(full_response)
            
            self.finished_response.emit()
            
        except Exception as e:
            self.error_occurred.emit(str(e))
            
    def stop(self):
        """停止请求"""
        self.running = False


class ChatWindow(QMainWindow):
    """主对话窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qwen30B 大模型对话程序")
        self.setGeometry(100, 100, 900, 700)
        
        # Ollama 配置
        self.api_url = "http://localhost:11434/api/chat"
        self.model = "qwen3.6:35b-a3b"  # 默认模型
        
        # 对话历史
        self.chat_history = []
        self.current_thread = None
        
        # 初始化UI
        self.init_ui()
        
        # 检查 Ollama 服务
        QTimer.singleShot(100, self.check_ollama)
        
    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 标题
        title_label = QLabel("Qwen30B 大模型对话程序")
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 状态栏
        self.status_label = QLabel("状态: 检查 Ollama 服务...")
        self.status_label.setFont(QFont("Microsoft YaHei", 10))
        layout.addWidget(self.status_label)
        
        # 模型选择
        model_container = QHBoxLayout()
        model_container.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_container.addWidget(self.model_combo)
        
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.clicked.connect(self.load_models)
        model_container.addWidget(self.refresh_button)
        
        model_container.addStretch()
        layout.addLayout(model_container)
        
        # 对话显示区
        self.chat_display = QTextEdit()
        self.chat_display.setFont(QFont("Microsoft YaHei", 11))
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.chat_display, stretch=1)
        
        # 输入区
        input_container = QHBoxLayout()
        
        self.input_field = QTextEdit()
        self.input_field.setFont(QFont("Microsoft YaHei", 11))
        self.input_field.setMaximumHeight(100)
        self.input_field.setPlaceholderText("请输入您的问题...")
        self.input_field.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 8px;
            }
        """)
        input_container.addWidget(self.input_field, stretch=1)
        
        # 发送按钮
        self.send_button = QPushButton("发送")
        self.send_button.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        self.send_button.setMinimumWidth(80)
        self.send_button.setMinimumHeight(100)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.send_button.clicked.connect(self.send_message)
        input_container.addWidget(self.send_button)
        
        # 停止按钮
        self.stop_button = QPushButton("停止")
        self.stop_button.setFont(QFont("Microsoft YaHei", 12))
        self.stop_button.setMinimumWidth(80)
        self.stop_button.setMinimumHeight(100)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.stop_button.clicked.connect(self.stop_generation)
        input_container.addWidget(self.stop_button)
        
        layout.addLayout(input_container)
        
        # 参数设置区
        params_container = QHBoxLayout()
        
        # 温度
        params_container.addWidget(QLabel("温度:"))
        self.temp_spin = QSpinBox()
        self.temp_spin.setRange(0, 100)
        self.temp_spin.setValue(70)
        self.temp_spin.setSingleStep(5)
        params_container.addWidget(self.temp_spin)
        
        # 上下文长度
        params_container.addWidget(QLabel("上下文:"))
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(512, 16384)
        self.ctx_spin.setValue(4096)
        self.ctx_spin.setSingleStep(512)
        params_container.addWidget(self.ctx_spin)
        
        # 清除按钮
        self.clear_button = QPushButton("清除对话")
        self.clear_button.clicked.connect(self.clear_chat)
        params_container.addWidget(self.clear_button)
        
        params_container.addStretch()
        layout.addLayout(params_container)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #ffffff; }
            QLabel { color: #333; }
        """)
        
    def check_ollama(self):
        """检查 Ollama 服务"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                self.status_label.setText("状态: Ollama 服务正常 ✓")
                self.status_label.setStyleSheet("color: green;")
                self.load_models()
            else:
                self.status_label.setText("状态: Ollama 服务异常")
                self.status_label.setStyleSheet("color: red;")
        except requests.exceptions.ConnectionError:
            self.status_label.setText("状态: Ollama 未运行 (请运行 'ollama serve')")
            self.status_label.setStyleSheet("color: red;")
            QMessageBox.warning(
                self,
                "Ollama 未运行",
                "请先启动 Ollama 服务:\n\n"
                "1. 打开终端\n"
                "2. 运行命令: ollama serve\n"
                "3. 保持终端打开\n"
                "4. 刷新页面"
            )
        except Exception as e:
            self.status_label.setText(f"状态: 错误 - {str(e)}")
            self.status_label.setStyleSheet("color: red;")
            
    def load_models(self):
        """加载可用模型"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                
                self.model_combo.clear()
                self.model_combo.addItems(models)
                
                # 选择默认模型
                if self.model in models:
                    self.model_combo.setCurrentText(self.model)
                elif models:
                    self.model_combo.setCurrentIndex(0)
                    
        except Exception as e:
            self.status_label.setText(f"状态: 加载模型失败 - {str(e)}")
            
    def on_model_changed(self, model_name):
        """模型选择改变"""
        self.model = model_name
        
    def send_message(self):
        """发送消息"""
        user_input = self.input_field.toPlainText().strip()
        if not user_input:
            return
        
        # 显示用户消息
        self.display_message("用户", user_input, QColor("#2196F3"))
        self.chat_history.append({"role": "user", "content": user_input})
        
        # 清空输入
        self.input_field.clear()
        
        # 禁用按钮
        self.send_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("状态: 正在生成...")
        
        # 构建消息（包含系统提示和历史）
        messages = [{"role": "system", "content": "你是一个有帮助的AI助手，请用中文回答用户的问题。"}]
        messages.extend(self.chat_history[-10:])  # 最近10轮对话
        
        # 启动线程
        self.current_thread = OllamaThread(
            self.api_url,
            self.model,
            messages,
            stream=True
        )
        self.current_thread.response_received.connect(self.on_response)
        self.current_thread.error_occurred.connect(self.on_error)
        self.current_thread.finished_response.connect(self.on_finished)
        self.current_thread.start()
        
    def display_message(self, role, content, color=None):
        """显示消息"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        html = f'<div style="margin: 10px 0;">'
        html += f'<span style="font-weight: bold; color: {color.name() if color else "#333"};">{role}:</span><br/>'
        html += f'<span style="color: #333; white-space: pre-wrap;">{content}</span>'
        html += '</div><hr style="border: 1px solid #eee;"/>'
        
        self.chat_display.insertHtml(html)
        self.chat_display.ensureCursorVisible()
        
    def on_response(self, content):
        """收到响应（流式）"""
        # 更新最后一条助手消息
        pass  # 流式更新暂时不处理
        
    def on_error(self, error):
        """发生错误"""
        self.display_message("系统", f"错误: {error}", QColor("#f44336"))
        self.reset_ui()
        
    def on_finished(self):
        """完成响应"""
        # 保存助手回复
        # 从显示区提取最后一条助手回复
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # 简化处理：直接从 chat_history 获取
        # 注意：Ollama 流式响应时需要特殊处理
        self.reset_ui()
        
    def reset_ui(self):
        """重置UI状态"""
        self.send_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("状态: 准备就绪 ✓")
        self.status_label.setStyleSheet("color: green;")
        self.current_thread = None
        
    def stop_generation(self):
        """停止生成"""
        if self.current_thread:
            self.current_thread.stop()
            self.display_message("系统", "已停止生成", QColor("#FF9800"))
            self.reset_ui()
            
    def clear_chat(self):
        """清除对话"""
        self.chat_display.clear()
        self.chat_history.clear()
        self.display_message("系统", "对话已清除，可以开始新的对话。", QColor("#9E9E9E"))


def main():
    """主入口"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()