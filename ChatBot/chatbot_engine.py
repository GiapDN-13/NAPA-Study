import os
from pydantic import SecretStr
from utils import load_api_keys, load_knowledge_base, split_documents

# LangChain components
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS 
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA 


# --- Constants ---
MODEL_NAME_GEMINI = "gemini-1.5-flash-latest"
EMBEDDING_MODEL_NAME_GEMINI = "models/embedding-001" 
VECTOR_STORE_PATH = "faiss_index_ecommerce" 
KNOWLEDGE_BASE_DIR = "knowledge_base"

class ChatbotEngine:
    def __init__(self):
        raw_google_api_key, _ = load_api_keys()
        if not raw_google_api_key:
            raise ValueError("Google API Key is required to run the chatbot engine.")
        self.google_api_key = SecretStr(raw_google_api_key)

        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        
        self._initialize_llm()
        self._initialize_embeddings()
        self._initialize_vector_store()
        self._initialize_qa_chain()

    def _initialize_llm(self):
        """Khởi tạo mô hình ngôn ngữ lớn (LLM)."""
        print(f"Đang khởi tạo LLM: {MODEL_NAME_GEMINI}")
        try:
            if not self.google_api_key:
                raise ValueError("Google API Key is not set for LLM initialization.")
            self.llm = ChatGoogleGenerativeAI(
                model=MODEL_NAME_GEMINI,
                google_api_key=self.google_api_key,
                temperature=0.2,
                top_p=0.8,
                top_k=40,
                convert_system_message_to_human=True
            )
            print("LLM đã khởi tạo thành công.")
        except Exception as e:
            print(f"Lỗi khi khởi tạo LLM: {e}")
            raise

    def _initialize_embeddings(self):
        """Khởi tạo mô hình embeddings."""
        print(f"Đang khởi tạo mô hình Embeddings: {EMBEDDING_MODEL_NAME_GEMINI}")
        try:
            if not self.google_api_key:
                raise ValueError("Google API Key is not set for Embeddings initialization.")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL_NAME_GEMINI,
                google_api_key=self.google_api_key
            )
            print("Mô hình Embeddings đã khởi tạo thành công.")
        except Exception as e:
            print(f"Lỗi khi khởi tạo Embeddings: {e}")
            raise

    def _load_and_process_documents(self):
        """Tải và xử lý tài liệu từ knowledge base."""
        documents = load_knowledge_base(KNOWLEDGE_BASE_DIR)
        if not documents:
            print("Không có tài liệu nào được tải. Vector store có thể sẽ trống.")
            return []
        chunks = split_documents(documents)
        return chunks

    def _initialize_vector_store(self):
        """Khởi tạo vector store. Tải từ disk nếu tồn tại, nếu không thì tạo mới."""
        print("Đang khởi tạo Vector Store...")
        
        if not self.embeddings:
            print("LỖI: Embeddings chưa được khởi tạo. Không thể khởi tạo Vector Store.")
            self.vector_store = None
            return

        processed_chunks = self._load_and_process_documents()

        if not processed_chunks:
            print("Không có chunks nào để xử lý cho vector store. Bỏ qua việc tạo/tải vector store.")
            self.vector_store = None
            return

        # Kiểm tra xem index đã tồn tại chưa
        if os.path.exists(VECTOR_STORE_PATH) and os.path.isdir(VECTOR_STORE_PATH):
            try:
                print(f"Đang tải Vector Store từ: {VECTOR_STORE_PATH}")
                self.vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True 
                )
                print("Vector Store đã tải thành công từ disk.")
            except Exception as e:
                print(f"Lỗi khi tải Vector Store từ disk: {e}. Sẽ tạo mới.")
                self.vector_store = FAISS.from_documents(processed_chunks, self.embeddings)
                self.vector_store.save_local(VECTOR_STORE_PATH)
                print(f"ĐAng tạo và lưu Vector Store mới tại: {VECTOR_STORE_PATH}")
        else:
            print(f"Không tìm thấy Vector Store tại {VECTOR_STORE_PATH}. Đang tạo mới...")
            self.vector_store = FAISS.from_documents(processed_chunks, self.embeddings)
            self.vector_store.save_local(VECTOR_STORE_PATH) 
            print(f"Đang tạo và lưu Vector Store mới tại: {VECTOR_STORE_PATH}")
        
        if self.vector_store:
            print("Vector Store đã sẵn sàng.")
        else:
            print("LƯU Ý: Vector Store chưa được khởi tạo thành công.")


    def _initialize_qa_chain(self):
        """Khởi tạo chuỗi (chain) hỏi đáp RetrievalQA."""
        if not self.llm or not self.vector_store:
            print("LLM hoặc Vector Store chưa được khởi tạo. Không thể tạo QA chain.")
            return

        print("Đang khởi tạo QA Chain...")
        # Tạo một retriever từ vector store với cấu hình tối ưu
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={
                "k": 5,
                "score_threshold": 0.7
            }
        )

        # Tạo Prompt Template tối ưu
        prompt_template_str = """Bạn là một trợ lý AI chuyên nghiệp cho một nền tảng thương mại điện tử.
        Nhiệm vụ của bạn là trả lời các câu hỏi của khách hàng một cách chính xác, đầy đủ và thân thiện.

        Hướng dẫn quan trọng:
        1. Chỉ sử dụng thông tin từ Context được cung cấp để trả lời
        2. Nếu thông tin trong Context không đủ, hãy thừa nhận điều đó và đề xuất cách liên hệ hỗ trợ
        3. Trả lời bằng tiếng Việt, rõ ràng và dễ hiểu
        4. Nếu câu hỏi không liên quan đến thương mại điện tử, hãy lịch sự chuyển hướng
        5. Luôn giữ giọng điệu chuyên nghiệp và hữu ích
        6. Khi trả lời về sản phẩm, hãy cung cấp đầy đủ thông tin theo thứ tự:
           - Tên sản phẩm và mã sản phẩm
           - Mô tả chung
           - Thông số kỹ thuật (nếu có)
           - Giá bán
           - Phụ kiện đi kèm (nếu có)
           - Thông tin bảo hành (nếu có)

        Context:
        {context}

        Câu hỏi của khách hàng:
        {question}

        Hãy trả lời câu hỏi trên một cách chính xác và hữu ích:
        """
        
        QA_PROMPT = PromptTemplate(
            template=prompt_template_str,
            input_variables=["context", "question"]
        )

        # Tạo RetrievalQA chain với cấu hình tối ưu
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True
        )
        print("QA Chain đã khởi tạo thành công.")

    def ask(self, query: str):
        """Hỏi chatbot và nhận câu trả lời."""
        if not self.qa_chain:
            return "Xin lỗi, hệ thống chatbot hiện chưa sẵn sàng. Vui lòng thử lại sau."

        print(f"\nĐang xử lý câu hỏi: {query}")
        try:
            response = self.qa_chain.invoke({"query": query}) 
            
            answer = response.get("result", "Không tìm thấy câu trả lời.")
            source_documents = response.get("source_documents", [])
            
            print(f"Câu trả lời thô từ LLM: {answer}")
            if source_documents:
                print(f"Dựa trên {len(source_documents)} nguồn tài liệu:")
            else:
                print("Không tìm thấy nguồn tài liệu cụ thể cho câu trả lời này.")
                
            return answer
        except Exception as e:
            print(f"Lỗi khi xử lý câu hỏi: {e}")
            return f"Đã có lỗi xảy ra khi xử lý yêu cầu của bạn. Chi tiết: {str(e)}"


if __name__ == '__main__':
    # Test ChatbotEngine
    try:
        print("--- Bắt đầu kiểm tra ChatbotEngine ---")
        engine = ChatbotEngine()
        
        if engine.qa_chain:
            print("\n--- Thử nghiệm hỏi đáp ---")
            
            test_queries = [
                "Quy trình đặt hàng như thế nào?",
                "Phí vận chuyển ra sao?",
                "Làm thế nào để đổi trả sản phẩm bị lỗi?",
                "Máy xay sinh tố BlendMaster 5000 có bảo hành không?",
                "Shop có bán áo màu hồng không?"
            ]
            
            for q in test_queries:
                print(f"\n[USER]: {q}")
                answer = engine.ask(q)
                print(f"[BOT]: {answer}")
        else:
            print("Không thể thực hiện hỏi đáp vì QA chain chưa được khởi tạo.")
            
    except ValueError as ve:
        print(f"Lỗi Value: {ve}")
    except Exception as e:
        print(f"Lỗi không mong muốn: {e}")
    finally:
        print("\n--- Kết thúc kiểm tra ChatbotEngine ---")