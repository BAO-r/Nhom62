import os
from pathlib import Path
import tempfile
from pypdf import PdfReader
from crewai import Agent, Crew, Process, Task
import streamlit as st

# Thiết lập kết nối Ollama local
os.environ["OPENAI_API_BASE"] = "http://localhost:11434"
os.environ["OPENAI_API_KEY"] = "ollama"

OLLAMA_MODEL = "ollama/qwen2.5:7b"

# --- Cấu hình giao diện Streamlit ---
st.set_page_config(
    page_title="Multi-Agent Literature Review",
    page_icon="📚",
    layout="wide"
)

st.title("📚 HỆ THỐNG ĐA TÁC TỬ TỰ ĐỘNG PHÂN TÍCH, PHẢN BIỆN VÀ TỔNG HỢP BÁO CÁO TÀI LIỆU ĐA NGUỒN")

st.markdown("Tải lên các file PDF bài báo, tài liệu của bạn để hệ thống tự động tổng hợp thành báo cáo **Literature Review**.")

# Khung tải file ở Sidebar hoặc màn hình chính
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải file")
    uploaded_files = st.file_uploader(
        "Chọn các file PDF cần phân tích", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    max_chars = st.slider("Giới hạn ký tự mỗi file PDF", 5000, 20000, 10000, 1000)
    
    run_button = st.button("🚀 Bắt đầu phân tích", type="primary")

def read_uploaded_pdfs(files, max_chars_per_pdf: int = 10000) -> str:
    """Đọc nội dung các file PDF được tải lên từ Streamlit."""
    combined_text = ""
    for idx, uploaded_file in enumerate(files, 1):
        try:
            # Đọc trực tiếp file upload bằng pypdf
            reader = PdfReader(uploaded_file)
            file_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    file_text += text + "\n"
            
            # Cắt bớt nếu quá dài để bảo vệ VRAM
            if len(file_text) > max_chars_per_pdf:
                file_text = file_text[:max_chars_per_pdf] + "\n[... Nội dung bị cắt bớt để tối ưu bộ nhớ ...]"

            combined_text += f"\n--- BẮT ĐẦU TÀI LIỆU {idx}: {uploaded_file.name} ---\n"
            file_text_clean = file_text if file_text else "[Không thể trích xuất văn bản từ PDF này]"
            combined_text += file_text_clean
            combined_text += f"\n--- KẾT THÚC TÀI LIỆU {idx} ---\n"
        except Exception as e:
            st.error(f"Lỗi khi đọc file {uploaded_file.name}: {e}")
            
    return combined_text

def build_and_run_crew(pdf_contents: str):
    # 1. Định nghĩa các Agents (Bổ sung Tác tử 1)
    document_parser = Agent(
        role="Document Parser Agent",
        goal="Đọc và làm sạch văn bản từ file PDF hoặc báo cáo thô[cite: 1].",
        backstory="Bạn là một chuyên gia phân tích và tiền xử lý tài liệu, có khả năng làm sạch văn bản thô, loại bỏ các nhiễu và chuẩn hóa định dạng từ các tài liệu khoa học.",
        llm=OLLAMA_MODEL,    
        verbose=True,
        allow_delegation=False,
    )

    data_extractor = Agent(
        role="Chuyên gia Trích xuất và Tóm tắt Văn bản Khoa học",
        goal="Trích xuất chính xác luận điểm chính, phương pháp, kết quả và hạn chế từ các tài liệu được cung cấp.",
        backstory="Bạn là một nhà nghiên cứu kỳ cựu với khả năng đọc nhanh và cô đọng nội dung các bài báo khoa học.",
        llm=OLLAMA_MODEL,    
        verbose=True,
        allow_delegation=False,
    )

    critical_reviewer = Agent(
        role="Nhà Phản biện và Đánh giá Phương pháp Nghiên cứu",
        goal="Tìm ra các khoảng trống nghiên cứu (Research Gaps), điểm mâu thuẫn và hạn chế giữa các bài báo.",
        backstory="Bạn là một giám khảo phản biện khắt khe, chuyên so sánh nghiên cứu và phát hiện xung đột.",
        llm=OLLAMA_MODEL,    
        verbose=True,
        allow_delegation=False,
    )

    synthesizer = Agent(
        role="Chuyên gia Tổng hợp Báo cáo Tổng quan Nghiên cứu (Literature Review)",
        goal="Tổng hợp thông tin phân tích và phản biện thành một Báo cáo Literature Review hoàn chỉnh.",
        backstory="You are a top-tier professor specialized in writing and publishing leading review papers.",
        llm=OLLAMA_MODEL,   
        verbose=True,
        allow_delegation=False,
    )

    # 2. Định nghĩa các Tasks (Bổ sung task_parse cho Tác tử 1)
    task_parse = Task(
        description=(
            "Đọc và làm sạch nội dung thô từ các tài liệu PDF dưới đây:\n\n"
            "{pdf_data}\n\n"
            "Hãy loại bỏ các ký tự nhiễu, định dạng lại văn bản cho sạch sẽ, rõ ràng và chuẩn hóa cấu trúc để phục vụ cho các tác vụ trích xuất tiếp theo[cite: 1]."
        ),
        expected_output="Văn bản đã được làm sạch và chuẩn hóa từ tài liệu thô.",
        agent=document_parser
    )

    task_extract = Task(
        description=(
            "Dựa trên kết quả văn bản đã được làm sạch từ tác vụ trước, hãy lập danh sách tóm tắt cấu trúc hóa từng tài liệu bao gồm: "
            "(1) Tên/Mục tiêu nghiên cứu, (2) Phương pháp sử dụng, (3) Kết quả chính."
        ),
        expected_output="Bản tóm tắt chi tiết cấu trúc hóa của từng tài liệu khoa học.",
        agent=data_extractor
    )

    task_review = Task(
        description=(
            "Dựa trên kết quả trích xuất từ tác vụ trước, hãy thực hiện phản biện:\n"
            "1. So sánh sự giống và khác nhau giữa các nghiên cứu.\n"
            "2. Chỉ ra các mâu thuẫn hoặc điểm chưa thống nhất giữa các kết quả.\n"
            "3. Xác định các khoảng trống nghiên cứu (Research Gaps) còn tồn tại."
        ),
        expected_output="Báo cáo phân tích phản biện, mâu thuẫn và khoảng trống nghiên cứu.",
        agent=critical_reviewer
    )

    task_synthesize = Task(
        description=(
            "Dựa trên toàn bộ dữ liệu trích xuất và phân tích phản biện, hãy tổng hợp thành một báo cáo "
            "Tổng quan nghiên cứu (Literature Review) hoàn chỉnh bằng tiếng Việt theo đúng cấu trúc:\n"
            "1. ĐẶT VẤN ĐỀ & TỔNG QUAN\n"
            "2. PHƯƠNG PHÁP & CÁC HƯỚNG CẬP NHẬT CHÍNH\n"
            "3. PHÂN TÍCH PHẢN BIỆN & KHOẢNG TRỐNG NGHIÊN CỨU\n"
            "4. KẾT LUẬN & HƯỚNG PHÁT TRIỂN\n\n"
            "Yêu cầu: Trình bày dạng văn bản Markdown rõ ràng, tuyệt đối không trả về định dạng JSON."
        ),
        expected_output="Báo cáo Literature Review hoàn chỉnh dạng Markdown.",
        agent=synthesizer,
        output_file="literature_review_report.md"
    )

    # 3. Khởi tạo Crew với 4 tác tử
    literature_crew = Crew(
        agents=[document_parser, data_extractor, critical_reviewer, synthesizer],
        tasks=[task_parse, task_extract, task_review, task_synthesize],
        process=Process.sequential,
        verbose=True,
        memory=False,
        max_rpm=5
    )

    return literature_crew.kickoff(inputs={"pdf_data": pdf_contents})

# --- Xử lý sự kiện khi bấm nút chạy ---
if run_button:
    if not uploaded_files:
        st.warning("⚠️ Vui lòng tải lên ít nhất một file PDF trước khi bắt đầu!")
    else:
        with st.spinner("🔄 Đang xử lý file PDF và chạy các Multi-Agent... Quá trình này có thể mất vài phút tùy thuộc vào phần cứng của bạn."):
            # 1. Đọc nội dung PDF
            pdf_text = read_uploaded_pdfs(uploaded_files, max_chars_per_pdf=max_chars)
            
            if pdf_text.strip():
                # 2. Chạy CrewAI
                try:
                    final_report = build_and_run_crew(pdf_text)
                    
                    st.success("✅ Đã hoàn thành báo cáo Literature Review thành công!")
                    
                    # Hiển thị kết quả ra màn hình dạng Markdown
                    st.markdown("---")
                    st.markdown("### 📄 Báo cáo Tổng quan Nghiên cứu")
                    st.markdown(str(final_report))
                    
                    # Nút tải xuống file báo cáo
                    st.download_button(
                        label="📥 Tải xuống báo cáo (.md)",
                        data=str(final_report),
                        file_name="literature_review_report.md",
                        mime="text/markdown"
                    )
                except Exception as e:
                    st.error(f"❌ Đã xảy ra lỗi trong quá trình chạy CrewAI: {e}")
            else:
                st.error("❌ Không thể trích xuất được nội dung từ các file PDF đã chọn.")
