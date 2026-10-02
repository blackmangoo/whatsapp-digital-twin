@echo off
title WhatsApp Twin LLM Server
echo ===================================================
echo  Starting WhatsApp Digital Twin Server
echo  Base Model: Qwen 2.5 7B Instruct (q4_k_m)
echo  Fine-tuned Adapter: lora_model.gguf
echo  Port: 8080
echo ===================================================
echo.

set LLAMA_BIN=%LOCALAPPDATA%\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe

if not exist "%LLAMA_BIN%" (
    echo [ERROR] llama-server not found at %LLAMA_BIN%
    echo Trying PATH command 'llama-server'...
    set LLAMA_BIN=llama-server
)

"%LLAMA_BIN%" --hf-repo Qwen/Qwen2.5-7B-Instruct-GGUF:q4_k_m --lora lora_model.gguf --port 8080 -c 2048 --host 127.0.0.1

pause
