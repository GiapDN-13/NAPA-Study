import streamlit as st
from chatbot_engine import ChatbotEngine, MODEL_NAME_GEMINI
import warnings

# --- Custom CSS ---
def local_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background: #e6f3ff;
        border-left: 5px solid #2196F3;
    }
    .bot-message {
        background: #f0f0f0;
        border-left: 5px solid #4CAF50;
    }
    .chat-header {
        color: #1E3A8A;
        font-weight: 600;
    }
    .sidebar-content {
        padding: 1rem;
        background-color: #f0f7ff;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Page Configuration ---
st.set_page_config(
    page_title="E-commerce Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

local_css()

# --- Hide warnings ---
warnings.filterwarnings(
    "ignore",
    message="Convert_system_message_to_human will be deprecated!",
    category=UserWarning,
    module="langchain_google_genai"
)

# --- App Header ---
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://img.icons8.com/color/96/000000/shopping-cart--v2.png", width=80)
with col2:
    st.title("Chatbot Hỗ trợ Khách hàng E-commerce")
    st.caption(f"Được cung cấp bởi Google Gemini ({MODEL_NAME_GEMINI}) và LangChain")

# --- Chatbot Engine Initialization ---
@st.cache_resource
def load_chatbot_engine():
    """Tải và cache ChatbotEngine."""
    print("Đang khởi tạo ChatbotEngine cho ứng dụng Streamlit...")
    try:
        engine = ChatbotEngine()
        print("ChatbotEngine đã sẵn sàng trong Streamlit.")
        return engine
    except ValueError as ve:
        st.error(f"Lỗi cấu hình API: {ve}. Vui lòng kiểm tra file .env và API key của Google.")
        return None
    except Exception as e:
        st.error(f"Lỗi không mong muốn khi khởi tạo ChatbotEngine: {e}")
        return None

chatbot_engine = load_chatbot_engine()

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Welcome card ---
if not st.session_state.messages:
    st.info("""
    ### 👋 Chào mừng bạn!
    
    Tôi là trợ lý ảo của cửa hàng. Tôi có thể giúp bạn:
    - Tìm kiếm sản phẩm
    - Trả lời về quy trình đặt hàng
    - Giải đáp thắc mắc về vận chuyển
    - Hướng dẫn chính sách đổi trả
    
    Hãy nhập câu hỏi của bạn vào ô chat bên dưới.
    """)

# --- Display chat history ---
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"<div class='chat-message user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"<div class='chat-message bot-message'>{message['content']}</div>", unsafe_allow_html=True)

# --- User Input Processing ---
user_query = st.chat_input("Nhập câu hỏi của bạn ở đây...")

if user_query and chatbot_engine:
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"<div class='chat-message user-message'>{user_query}</div>", unsafe_allow_html=True)

    # Get response from chatbot
    with st.spinner("Bot đang suy nghĩ... 🤔"):
        try:
            bot_response = chatbot_engine.ask(user_query)
        except Exception as e:
            bot_response = f"Xin lỗi, đã có lỗi xảy ra khi tôi cố gắng trả lời bạn: {e}"
            st.error(bot_response)

    # Add bot response to history and display
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(f"<div class='chat-message bot-message'>{bot_response}</div>", unsafe_allow_html=True)

elif user_query and not chatbot_engine:
    st.error("Chatbot engine chưa được khởi tạo thành công. Vui lòng kiểm tra lỗi ở trên và thử làm mới trang.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("<h3 class='chat-header'>Công cụ & Thông tin</h3>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
    st.markdown("### Về Chatbot này")
    st.info(
        "Chatbot này được thiết kế để trả lời các câu hỏi phổ biến về nền tảng e-commerce của chúng tôi. "
        "Sử dụng mô hình ngôn ngữ lớn và kỹ thuật Retrieval Augmented Generation (RAG) "
        "để cung cấp thông tin chính xác từ cơ sở kiến thức."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
    st.markdown("### Trạng thái hệ thống")
    if chatbot_engine and chatbot_engine.vector_store:
        st.success("✅ Vector Store đã được tải!")
    else:
        st.warning("⚠️ Vector Store chưa sẵn sàng.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
    st.markdown("### Quản lý cuộc trò chuyện")
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)