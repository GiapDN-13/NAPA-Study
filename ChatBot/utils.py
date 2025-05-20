import warnings 
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_api_keys():
    """Tải các API key từ file .env."""
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    langchain_project = os.getenv("LANGCHAIN_PROJECT", "default-project") # Thêm project, mặc định là 'default-project'
    
    if not google_api_key:
        print("CẢNH BÁO: GOOGLE_API_KEY không được tìm thấy trong file .env.")
    if not langchain_api_key:
        print("CẢNH BÁO: LANGCHAIN_API_KEY không được tìm thấy trong file .env. LangSmith tracing có thể không hoạt động.")
    else:
        # Tự động thiết lập các biến môi trường cần thiết cho LangSmith nếu key được cung cấp
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
        if langchain_project:
            os.environ["LANGCHAIN_PROJECT"] = langchain_project
        print(f"LangSmith tracing được kích hoạt cho project: {os.getenv('LANGCHAIN_PROJECT')}")


    return google_api_key, langchain_api_key

def load_knowledge_base(directory_path="knowledge_base"):
    """
    Tải tất cả các file .md và .txt từ thư mục được chỉ định.
    Sử dụng UnstructuredMarkdownLoader cho file .md và TextLoader cho file .txt.
    """
    print(f"Đang tải tài liệu từ thư mục: {directory_path}")
    
    # Loader cho file Markdown
    md_loader = DirectoryLoader(
        directory_path,
        glob="**/*.md", # Lấy tất cả file .md trong thư mục và thư mục con
        loader_cls=UnstructuredMarkdownLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True,
        use_multithreading=True
    )
    
    # Loader cho file Text
    txt_loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt", # Lấy tất cả file .txt
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True,
        use_multithreading=True
    )
    
    md_documents = md_loader.load()
    txt_documents = txt_loader.load()
    
    all_documents = md_documents + txt_documents
    
    if not all_documents:
        print(f"Không tìm thấy tài liệu nào trong thư mục: {directory_path}")
    else:
        print(f"Đã tải thành công {len(all_documents)} tài liệu.")
            
    return all_documents

def split_documents(documents, chunk_size=1000, chunk_overlap=400):
    """
    Chia tài liệu thành các đoạn nhỏ hơn (chunks).
    Tối ưu cho tiếng Việt với chunk size lớn hơn và overlap nhiều hơn
    để giữ được ngữ cảnh tốt hơn.
    """
    print(f"Đang chia tài liệu thành các chunks (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=[
            "\n## ",  # Tiêu đề chính
            "\n### ",  # Tiêu đề phụ
            "\n- ",    # Danh sách
            "\n    - ", # Danh sách con
            "\n",      # Dòng mới
            ". ",      # Câu
            "! ",      # Câu
            "? ",      # Câu
            ", ",      # Phẩy
            " ",       # Khoảng trắng
            ""         # Ký tự
        ]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Đã chia thành công {len(chunks)} chunks.")
    return chunks

if __name__ == '__main__':
    # Test các hàm trong utils.py
    google_key, lc_key = load_api_keys()
    print(f"Google API Key (loaded): {'Có' if google_key else 'Không'}")
    print(f"LangSmith API Key (loaded): {'Có' if lc_key else 'Không'}")

    # Tạo một số file mẫu để test nếu chưa có
    if not os.path.exists("knowledge_base"):
        os.makedirs("knowledge_base")
    if not os.path.exists("knowledge_base/test.md"):
        with open("knowledge_base/test.md", "w", encoding="utf-8") as f:
            f.write("# Test Markdown\nĐây là nội dung test cho file markdown.")
    if not os.path.exists("knowledge_base/test.txt"):
        with open("knowledge_base/test.txt", "w", encoding="utf-8") as f:
            f.write("Test Text\nĐây là nội dung test cho file text.")

    docs = load_knowledge_base()
    if docs:
        chunks = split_documents(docs)