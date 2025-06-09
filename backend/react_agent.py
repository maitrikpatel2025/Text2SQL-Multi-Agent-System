"""A simple ReAct-style agent that orchestrates existing tools."""

from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from .main import (
    planner,
    metadata_handler,
    generator,
    validator,
    decision_maker,
    code_generator_node,
    code_validator_node,
    code_executor_node,
    interpreter_node,
    graph_agent_node,
    Text2SQLState,
    get_metadata_context,
    get_llm,
)


class ReActAgent:
    """A lightweight ReAct loop that chooses which internal tool to invoke."""

    tools = {
        "planner": planner,
        "metadata_handler": metadata_handler,
        "generator": generator,
        "validator": validator,
        "decision_maker": decision_maker,
        "code_generator": code_generator_node,
        "code_validator": code_validator_node,
        "code_executor": code_executor_node,
        "interpreter": interpreter_node,
        "graph_agent": graph_agent_node,
    }

    def __init__(self, temperature: float = 0.0):
        self.llm = get_llm(temperature=temperature)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a coordinator agent. Decide which tool to use next from {tool_names}. Always output JSON with fields 'action' and 'finish', where 'finish' is true when the task is complete."""),
            ("human", "{state}\nUser question: {question}")
        ])
        self.parser = JsonOutputParser()

    def run(self, question: str, username: str = "guest") -> Dict:
        state: Text2SQLState = {
            "question": question,
            "username": username,
            "structured_intent": None,
            "sql_query": None,
            "query_result": None,
            "post_processing_required": None,
            "python_code": None,
            "final_result": None,
            "explanation": None,
        }
        while True:
            state_summary = {
                k: v for k, v in state.items() if k in {"structured_intent", "sql_query", "final_result", "explanation"}
            }
            decision = self.prompt | self.llm | self.parser
            decision_out = decision.invoke({
                "tool_names": list(self.tools.keys()),
                "state": state_summary,
                "question": state["question"],
            })
            action = decision_out.get("action")
            finish = decision_out.get("finish", False)
            if finish:
                break
            tool = self.tools.get(action)
            if tool is None:
                break
            result = tool(state)
            state.update(result)
        return state

