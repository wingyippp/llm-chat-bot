import urllib.request
import ssl
import os
import sys

url = "https://hf-mirror.com/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
output_path = "c:\\coding_agent\\qwen30b_chat\\model\\Qwen2.5-7B-Instruct-Q4_K_M.gguf"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

print(f"开始下载: {url}")
print(f"保存到: {output_path}")
print()

with urllib.request.urlopen(url, context=ctx) as response:
    total_size = int(response.headers.get('Content-Length', 0))
    downloaded = 0
    chunk_size = 1024 * 1024
    
    with open(output_path, 'wb') as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                speed = downloaded / (1024 * 1024)
                print(f"\r下载进度: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end='')
            else:
                print(f"\r下载进度: {downloaded / (1024*1024):.1f} MB", end='')
            
            sys.stdout.flush()

print()
print("下载完成！")