import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType

# 1. Page Configuration for Web Interface
st.set_page_config(page_title="Free Search Agent", page_icon="🤖")
st.title("🌐 My Free Agentic App")
st.caption("Powered by Llama 3 & DuckDuckGo Search (100% Free Tier)")

# 2. Secure Input for API Key (or hardcode it while testing)
# For security when sharing, users can input their key here.
groq_api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")

st.info("💡 To use this agent, get a free key from ://groq.com and turn off script blockers if the verification box fails.")

# 3. Main Agent Execution
user_query = st.text_input("What would you like the agent to research today?")

if user_query:
    if not groq_api_key:
        st.warning("Please enter your free Groq API key in the sidebar to proceed.")
    else:
        try:
            with st.spinner("Agent is searching the web and thinking..."):
                # Initialize the free LLM Brain
                llm = ChatGroq(
                    groq_api_key=groq_api_key, 
                    model_name="llama3-8b-8192",
                    temperature=0.3
                )
                
                # Initialize the Free Web Search Tool
                search_tool = DuckDuckGoSearchRun()
                tools = [search_tool]
                
                # Assemble the Agent Executor loop
                agent = initialize_agent(
                    tools=tools,
                    llm=llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True
                )
                
                # Run the query through the tool ecosystem
                response = agent.run(user_query)
                
                # Output results to the web screen
                st.success("Analysis Complete:")
                st.write(response)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
