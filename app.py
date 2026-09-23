import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# 1. Page Configuration
st.set_page_config(page_title="Free Search Agent", page_icon="🤖")
st.title("🌐 My Free Agentic App")
st.caption("Powered by Llama 3 & DuckDuckGo Search (Modern LangChain)")

groq_api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")
st.info("💡 Get a free key from ://groq.com and paste it in the sidebar.")

user_query = st.text_input("What would you like the agent to research today?")

if user_query:
    if not groq_api_key:
        st.warning("Please enter your free Groq API key in the sidebar to proceed.")
    else:
        try:
            with st.spinner("Agent is searching the web and thinking..."):
                # Initialize modern LLM Brain
                llm = ChatGroq(
                    groq_api_key=groq_api_key, 
                    model_name="llama3-8b-8192",
                    temperature=0.3
                )
                
                # Setup Free Tool
                tools = [DuckDuckGoSearchRun()]
                
                # Define standard ReAct prompt structure required by modern LangChain
                template = """Answer the following questions as best you can. You have access to the following tools:

                {tools}

                Use the following format:

                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question

                Begin!

                Question: {input}
                Thought:{agent_scratchpad}"""

                prompt = PromptTemplate.from_template(template)
                
                # Build the modern agent loop
                agent = create_react_agent(llm, tools, prompt)
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
                
                # Run execution
                response = agent_executor.invoke({"input": user_query})
                
                st.success("Analysis Complete:")
                st.write(response["output"])
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
