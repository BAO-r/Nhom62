import os
import sys
from pathlib import Path
from pypdf import PdfReader
from crewai import Agent, Crew, Process, Task

os.environ["OPENAI_API_BASE"] = "http://localhost:11434"

OLLAMA_MODEL = "ollama/qwen2.5:7b"

def read_pdfs_from_directory(dir_path: str, max_chars_per_pdf: int = 12000) -> str:
    """
    Quét và đọc nội dung tất cả file PDF trong thư mục.
    Cắt bớt ký tự nếu file quá dài để tránh tràn Context Window / VRAM của Ollama.
    """
    folder = Path(dir_path)
    if not folder.exists() or not folder.is_dir():
        print(f"❌ Lỗi: Thư mục '{dir_path}' không tồn tại!")
        sys.exit(1)

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ Cảnh báo: Không tìm thấy file PDF nào trong '{dir_path}'.")
        sys.exit(1)

    print(f"🔍 Đã tìm thấy {len(pdf_files)} file PDF. Đang tiến hành đọc...")

    combined_text = ""
    for idx, file_path in enumerate(pdf_files, 1):
        try:
            reader = PdfReader(file_path)
            file_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    file_text += text + "\n"
            
            # Cắt bớt văn bản nếu quá dài để bảo vệ VRAM
            if len(file_text) > max_chars_per_pdf:
                file_text = file_text[:max_chars_per_pdf] + "\n[... Nội dung bị cắt bớt để tối ưu bộ nhớ ...]"

            combined_text += f"\n--- BẮT ĐẦU TÀI LIỆU {idx}: {file_path.name} ---\n"
            combined_text += file_text
            combined_text += f"\n--- KẾT THÚC TÀI LIỆU {idx} ---\n"
            print(f"  [✓] Đã đọc thành công: {file_path.name}")
        except Exception as e:
            print(f"  [✗] Lỗi khi đọc file {file_path.name}: {e}")

    return combined_text

def build_and_run_crew(pdf_contents: str):
   
    data_extractor = Agent(
        role="Chuyên gia Trích xuất và Tóm tắt Văn bản Khoa học",
        goal="Trích xuất luận điểm chính, phương pháp, kết quả và hạn chế từ các tài liệu được cung cấp.",
        backstory=(
            "Bạn là một nhà nghiên cứu kỳ cựu với khả năng đọc nhanh và cô đọng nội dung "
            "các bài báo khoa học phức tạp một cách chính xác, minh bạch."
        ),
        llm=OLLAMA_MODEL,     
        verbose=True,
        allow_delegation=False,
        max_iter=3            
    )

    critical_reviewer = Agent(
        role="Nhà Phản biện và Đánh giá Phương pháp Nghiên cứu",
        goal="Tìm ra các khoảng trống nghiên cứu (Research Gaps), điểm mâu thuẫn và hạn chế giữa các bài báo.",
        backstory=(
            "Bạn là một giám khảo phản biện khắt khe. Nhiệm vụ của bạn là so sánh các nghiên cứu, "
            "phát hiện sự xung đột về kết quả và chỉ ra những điểm mà các nghiên cứu chưa giải quyết được."
        ),
        llm=OLLAMA_MODEL,     
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

    synthesizer = Agent(
        role="Chuyên gia Tổng hợp Báo cáo Tổng quan Nghiên cứu (Literature Review)",
        goal="Tổng hợp thông tin phân tích và phản biện thành một Báo cáo Literature Review hoàn chỉnh.",
        backstory=(
            "Bạn là một GS.TS chuyên viết và xuất bản các bài báo tổng quan (Review Paper) trên "
            "các tạp chí hàng đầu. Bạn có khả năng hệ thống hóa tri thức một cách mạch lạc."
        ),
        llm=OLLAMA_MODEL,    
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

    task_extract = Task(
        description=(
            "Phân tích toàn bộ dữ liệu từ các tài liệu PDF dưới đây:\n\n"
            "{pdf_data}\n\n"
            "Hãy lập danh sách tóm tắt từng tài liệu gồm: (1) Tên/Mục tiêu nghiên cứu, "
            "(2) Phương pháp sử dụng, (3) Kết quả chính."
        ),
        expected_output="Bản tóm tắt chi tiết cấu trúc hóa của từng tài liệu khoa học.",
        agent=data_extractor
    )

    task_review = Task(
        description=(
            "Dựa trên bản tóm tắt thu được, hãy tiến hành phản biện:\n"
            "1. So sánh sự giống và khác nhau giữa các nghiên cứu.\n"
            "2. Chỉ ra các mâu thuẫn hoặc điểm chưa thống nhất.\n"
            "3. Xác định các khoảng trống nghiên cứu (Research Gaps) còn tồn tại."
        ),
        expected_output="Báo cáo phân tích phản biện, mâu thuẫn và khoảng trống nghiên cứu.",
        agent=critical_reviewer
    )

    task_synthesize = Task(
        description=(
            "Dựa trên dữ liệu đã trích xuất và phân tích phản biện, hãy tổng hợp thành một báo cáo "
            "Tổng quan nghiên cứu (Literature Review) hoàn chỉnh theo cấu trúc:\n"
            "1. ĐẶT VẤN ĐỀ & TỔNG QUAN\n"
            "2. PHƯƠNG PHÁP & CÁC HƯỚNG CẬP NHẬT CHÍNH\n"
            "3. PHÂN TÍCH PHẢN BIỆN & KHOẢNG TRỐNG NGHIÊN CỨU\n"
            "4. KẾT LUẬN & HƯỚNG PHÁT TRIỂN\n\n"
            "Lưu ý: Viết dưới dạng văn bản hoàn chỉnh (Plain Text / Markdown), KHÔNG xuất định dạng JSON."
        ),
        expected_output="Báo cáo Literature Review hoàn chỉnh bằng tiếng Việt.",
        agent=synthesizer
    )

    literature_crew = Crew(
        agents=[data_extractor, critical_reviewer, synthesizer],
        tasks=[task_extract, task_review, task_synthesize],
        process=Process.sequential,
        verbose=True
    )

    result = literature_crew.kickoff(inputs={"pdf_data": pdf_contents})
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_dir = sys.argv[1]
    else:
        pdf_dir = input("Nhập đường dẫn thư mục chứa các file PDF: ").strip()

    pdf_text = read_pdfs_from_directory(pdf_dir)

    print("\n" + "="*50)
    print("🚀 BẮT ĐẦU HỆ THỐNG ĐA TÁC TỬ PHÂN TÍCH & TỔNG HỢP")
    print("="*50 + "\n")

    final_report = build_and_run_crew(pdf_text)

    print("\n" + "="*50)
    print("📄 BÁO CÁO TỔNG QUAN NGHIÊN CỨU (LITERATURE REVIEW)")
    print("="*50 + "\n")
    print(final_report)
