import streamlit as st
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM
import torch

# Load JSON Data
json_path = r"C:\Users\Abdallah\Downloads\Agent_langchain\OFSOCF_pdf_1_all_pages.json"
with open(json_path, "r", encoding="utf-8") as f:
    product_data = json.load(f)

# Initialize Embeddings and Vector Store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# Convert JSON data to text format for indexing
def convert_json_to_text(product_data):
    return [json.dumps(product, indent=2) for product in product_data]

product_texts = convert_json_to_text(product_data)
vectorstore = FAISS.from_texts(product_texts, embeddings)

# Search Function
def search_products(query: str):
    search_results = vectorstore.similarity_search(query, k=6)
    if not search_results:
        return "Sorry, I couldn't find any relevant information.", None
    
    result_text = "Here are the results that match your query:\n\n"
    product_images = []
    
    for idx, result in enumerate(search_results):
        result_text += f"{idx + 1}. {result.page_content}\n\n"
        
        # Extract image path if available
        product_json = json.loads(result.page_content)
        if "Image Paths" in product_json and product_json["Image Paths"]:
            product_images.append(product_json["Image Paths"][0])
    
    return result_text, product_images

search_tool = Tool(
    name="Product Search",
    func=search_products,
    description="Use this tool to search within the product catalog."
)

# Agent Prompts
retrieval_agent_prompt = """
You are a retrieval agent. Your only task is to use the provided 'Product Search' tool to retrieve information relevant to the user's query.
Do not interpret, summarize, or finalize the answer. Just return the content from the tool.
If nothing is relevant, return what the tool returns.

Example Queries and Responses:
Query: I have an office that is 50*60 and I want to furnish it as an office. What do you recommend?
Response: Based on your office size, I recommend modular workstations, ergonomic chairs, and storage solutions from our catalog that maximize space efficiency and comfort.

Query: What is the best ergonomic chair for long working hours?
Response: The best ergonomic chair for long working hours is the "LORDO" chair, featuring an elastic mesh backrest, lumbar support, and adjustable armrests for maximum comfort.

Query: Can you suggest a meeting table that fits 10 people?
Response: The "NASDAQ" meeting table, available in 250cm and 300cm lengths, is an excellent option for a 10-person meeting room.
"""

final_agent_prompt = """
You are a knowledgeable and friendly sales agent.
You have retrieved the following relevant sections from the document:
{retrieved_content}

Now, based on the user's query: "{user_query}", provide a concise and helpful answer.
If the retrieved content does not directly answer the question, provide the best possible response based on the given information.
If no relevant information is found, politely say you couldn't find anything.
"""

# Initialize LLM
llm = OllamaLLM(model="qwen2.5:3b-instruct")

# Agent 1: Retrieval Agent
retrieval_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retrieval_agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent_type="zero-shot-react-description",
    memory=retrieval_memory,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True,
    prompt=retrieval_agent_prompt,
    temperature=0.0
)

# Agent 2: Final Answer Generator
def generate_final_answer(retrieved_content: str, user_query: str) -> str:
    prompt = final_agent_prompt.format(retrieved_content=retrieved_content, user_query=user_query)
    return llm(prompt)

# Streamlit UI
st.title("Product Search Assistant")

query = st.text_input("Enter your search query:")
if st.button("Search"):
    retrieved_content = retrieval_agent.run(query)
    product_images = []
    if isinstance(retrieved_content, tuple):
        retrieved_content, product_images = retrieved_content
    final_answer = generate_final_answer(retrieved_content, query)
    st.write("### Answer:")
    st.write(final_answer)
    
    if product_images:
        st.write("### Product Images:")
        for img_path in product_images:
            st.image(img_path, caption="Product Image", use_column_width=True)
