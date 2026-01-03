from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompts import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()
tools = [TavilySearch()]
llm = ChatOpenAI(temperature=0, model="deepseek-chat")
"""creating another instance of structured llm which is only responsible to """
structured_llm = llm.with_structured_output(AgentResponse)
react_prompt = hub.pull("hwchase17/react")
react_prompt_with_format_instructions = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tool_names"],
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
).partial(format_instructions="")
""" agent created from the create_react-agent is not actually the agent but we can consider as a chain of reasoning
where this will give the input to LLM and then the llm will decide whom to call next be it llm again or any tools

This is going to act as a reasoning chain and return a Runnable. It provides the info of tools and all the input 
that user has provided to the prompt and then LLm is going to reason on the input that, hey you are going to 
run the tavily tool and the output of tavily is going to fed as an input to the LLM and so on so forth.
"""
agent = create_react_agent(
    llm=llm, tools=tools, prompt=react_prompt_with_format_instructions
)

"""This is like a while loop. As many times calls have to go through between tools and llms, this executor will make
sure to complete the task
AgentExecutor is the orchestrator who decides based on the llm output that what needs to be done next.
"""
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
##chain = agent_executor
chain = agent_executor | extract_output | structured_llm


def main():
    print("Hello from search-agent!!")
    chain.invoke(
        input={
            "input": "search for 2 job postings for an ai engineer using langchain in the Mumbai area on linkedin and list their details"
        }
    )


if __name__ == "__main__":
    main()
