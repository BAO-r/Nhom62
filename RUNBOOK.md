📚 Hướng Dẫn Vận Hành: Hệ Thống Đa Tác Tử Phân Tích & Tổng Hợp Literature Review

Hệ thống đa tác tử (Multi-Agent System) tự động phân tích, phản biện và tổng hợp báo cáo **Tổng quan nghiên cứu (Literature Review) từ các tài liệu PDF khoa học đa nguồn. Hệ thống chạy hoàn toàn Local sử dụng CrewAI và Ollama, không phụ thuộc vào LangChain hay dịch vụ API trả phí.

------------------------------------------------------------------------------------------

 🛠️ 1. Yêu Cầu Tiền Trạm (Prerequisites)

* Hệ điều hành: Windows 10/11, macOS, hoặc Linux.
* Python: Phiên bản `3.10` đến `3.12`.
* RAM: Tối thiểu 16GB (Khuyên dùng 32GB nếu chạy model 7B/8B).
* GPU: NVIDIA GPU với VRAM >= 8GB (để chạy Ollama mượt mà).

------------------------------------------------------------------------------------------

 🚀 2. Cài Đặt Môi Trường
 
 Bước 1: Cài đặt & Khởi chạy Ollama
1. Tải và cài đặt Ollama từ [ollama.com](https://ollama.com/).
2. Mở Terminal / Command Prompt và tải mô hình AI (khuyên dùng `qwen2.5:7b` cho tiếng Việt):
   ```bash
   ollama pull qwen2.5:7b
3. Chạy mô hình AI-Ollama
   ```bash
   ollama serve
   ollama run qwen2.5:7b
   
 Bước 2: Cài đặt môi trường ảo cho python
1. Tạo môi trường ảo (Virtual Environment)
   ```bash
   python -m venv venv
2. Kích hoạt môi trường ảo
   ```bash
   venv/Scripts/activate
3. Cài đặt thư viện
   ```bash
   pip install crewai pypdf pathlib

------------------------------------------------------------------------------------------








