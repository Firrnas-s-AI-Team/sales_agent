import streamlit as st
import os
import glob
import json
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import JSONLoader
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Directories
BASE_DIR = r"E:\Singularity\Sales_Aget_project\Dataset\New folder\LLM_\LLM_\Office-OF"
base_imag = r'E:\Singularity\Sales_Aget_project\Dataset\New folder\LLM_\LLM_'

# Function to get categories
def get_categories():
    return [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

# Function to get products
def get_products(category):
    category_path = os.path.join(BASE_DIR, category)
    return [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))] if os.path.exists(category_path) else []

# Function to load product details
def load_product_details(category, product):
    product_path = os.path.join(BASE_DIR, category, product)
    json_files = glob.glob(os.path.join(product_path, "*.json"))
    if json_files:
        loader = JSONLoader(
            file_path=json_files[0],
            jq_schema=".[]",
            text_content=False,
            metadata_func=lambda record, index: {"source": "json_data", "index": index}
        )
        return loader.load()
    return {"error": "Product details not found."}

# Streamlit UI
st.set_page_config(page_title="Mobica Sales Chatbot", layout="wide")
st.title("🛋️ Welcome to Mobica - Premium Furniture Solutions")
st.sidebar.header("🔍 Explore Our Products")

categories = get_categories()
category = st.sidebar.selectbox("📂 Select a Category", categories)
products = get_products(category) if category else []
product = st.sidebar.selectbox("📦 Select a Product", products)

docs, product_images = [], {}
if product:
    docs = load_product_details(category, product)
    for doc in docs:
        content = json.loads(doc.page_content)
        image_paths = content.get('Image Paths', [])
        if image_paths:
            product_images[content['Product Name']] = [os.path.join(base_imag, img) for img in image_paths]

# Chatbot Section
st.subheader("🤖 Mobica Sales Chatbot")
chat_history = ChatMessageHistory()
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents([Document(page_content=json.dumps(json.loads(doc.page_content))) for doc in docs], embedding=embeddings)
retriever = vectorstore.as_retriever()

groq_api_key = os.getenv('Groq_API_Key')
llm = ChatGroq(groq_api_key=groq_api_key, model='mixtral-8x7b-32768')

# prompt_template ="""أنت شغال وكيل مبيعات جاوب على كل الأسئلة اللي ليها علاقة  بالمنتجات.
# من خلال السياق هاديك مواصفات كل منتج في السياق 

# السياق: {context}
# السؤال: {question}
# جاوب علي جميع الأسالة باللغة العربية المصرية 
# """

prompt_template = """
إنت دلوقتي شغال وكيل مبيعات محترف، مطلوب منك:
١- تجاوب على كل الأسئلة اللي ليها علاقة بالمنتجات
٢- تستخدم اللغة العربية المصرية في كل إجاباتك
٣- تحافظ على المصطلحات التقنية زي (finished electrostatic powder coated) بالإنجليزي
٤- تشرح المواصفات التقنية بأسلوب سهل وبسيط
٥- تتكلم بأسلوب مندوب المبيعات المصري

✅ **تعامل مع الأسئلة كالتالي:**
- **إذا كان السؤال فارغ أو غير واضح، لا تسترجع أي بيانات من السياق، فقط اطلب توضيحاً بأسلوب بسيط، مثال:**  
  - "معلش يا فندم، ممكن توضح سؤالك أكتر؟"  
  - "ياريت تكتب استفسارك علشان أقدر أساعدك!"   
في هذه الحالة، لا تضيف أي تفاصيل عن المنتجات أو المواصفات

- **إذا كان السؤال واضحاً ويتعلق بالمنتجات، جاوب بطريقة منظمة بالنمط التالي:**  
  - **المميزات الأساسية**  
  - **المقاسات المتاحة**  
  - **الإضافات الاختيارية** 


- **إذا سأل العميل عن السعر، وضّح أن الأسعار غير متاحة حالياً بأي من العبارات التالية:**  
  - "يا فندم، الأسعار مش متاحة حالياً، لكن ممكن أقولك كل التفاصيل عن المنتج."

طريقة تنظيم الإجابة:
• نظم المواصفات في نقط واضحة تحت عناوين رئيسية:
  - المواصفات الأساسية
  - المقاسات أو الأبعاد المتاحة
  - المميزات
  - الإضافات الاختيارية

• اكتب المقاسات أو الأبعاد والأرقام بشكل منظم
• استخدم الرموز والعلامات للتوضيح:
  - النقط (•) للمميزات
  - الشرطة (-) للتفاصيل
  - الأقواس () للمصطلحات التقنية

السياق بتاع المنتجات: {context}
السؤال بتاعك: {question}

لازم تجاوب باللغة العربية المصرية وتستخدم تعبيرات زي:
- "طيب ممكن أقولك"
وتحط المصطلحات التقنية بس بالإنجليزي
"""
 
prompt = ChatPromptTemplate.from_template(prompt_template)
output_parser = StrOutputParser()

rag_chain = ({
    "context": retriever,
    "question": RunnablePassthrough(),
    "chat_history": RunnableLambda(lambda _: "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history.messages])),
} | prompt | llm | output_parser)

question = st.text_input("💬 Ask a Question about the Product:")
if question:
    response = rag_chain.invoke({"question": question})
    st.markdown(f"**🗨️ Response:** {response}")
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)
    
    for product_name in product_images:
        if product_name.lower() in response.lower():
            for img_path in product_images[product_name]:
                if os.path.exists(img_path):
                    st.image(img_path, caption=product_name, use_container_width=True)
