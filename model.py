import json
from typing import List, Optional, Dict, Any, Tuple, Callable, Iterable

import numpy as np
from langchain import LLMChain
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun, get_openai_callback,
)
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate, AIMessagePromptTemplate, )
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.schema import BaseOutputParser, BaseMemory, PromptValue, StrOutputParser
from pydantic import BaseModel, Field, validator, Extra

import config


class MealIdeaLLMOutput(BaseModel):
    explanation: str = Field(description=config.THEME_MEAL_EXP)
    name: str = Field(description=config.THEME_MEAL_NAME_DESC)
    idea: str = Field(description=config.THEME_MEAL_IDEA_DESC)

    @validator("name")
    def meal_name_valid(cls, field):
        field = field.lower().strip()
        if field not in config.THEME_MEAL_NAME_VALID:
            raise ValueError("Meal name {name} should be in {field}".format(name=field,
                                                                            field=config.THEME_MEAL_NAME_VALID))
        return field


class MealIdeaScheduleOutput(BaseModel):
    schedule: List[MealIdeaLLMOutput] = Field(description=config.THEME_MEAL_LIST,
                                              min_items=len(config.THEME_MEAL_NAME_VALID),
                                              max_items=len(config.THEME_MEAL_NAME_VALID))


class Feedback:

    def __init__(self, content, obj):
        self.content = content
        self.obj = obj


class MealIdea:

    def __init__(self, day, name, idea, explanation, feedback=""):
        self.name = name
        self.day = day
        self.idea = idea
        self.explanation = explanation
        self.feedback = feedback

    @classmethod
    def from_llm(cls, day, llm_output):
        return cls(day=day, name=llm_output.name, idea=llm_output.idea, explanation=llm_output.explanation)

    @property
    def all_feedbacks(self):
        if len(self.feedback.strip()) > 0:
            return [Feedback(content=self.feedback, obj=self)]
        return []


class MealPlan:

    def __init__(self, daily_plan):
        self.daily_plan = daily_plan

    @property
    def all_feedbacks(self):
        result = []
        for d in self.daily_plan.values():
            for m in d.values():
                result.extend(m.all_feedbacks)
        return result


class PromptAdoptingParser(BaseOutputParser):
    base_parser: BaseOutputParser
    prompt: PromptValue

    def parse_with_prompt(self, completion: str, prompt_value: PromptValue):
        return self.base_parser.parse_with_prompt(completion, prompt_value)

    def parse(self, completion: str):
        return self.parse_with_prompt(completion, self.prompt)

    def get_format_instructions(self):
        return self.base_parser.get_format_instructions()


class RandomExampleSelector(BaseExampleSelector):

    def __init__(self, examples: List[Dict[str, str]], k=3):
        self.examples = examples
        self.k = k

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        return np.random.choice(self.examples, size=self.k, replace=False)


class KeyValueMemory(BaseMemory, BaseModel):
    base_chat_mem: BaseChatMemory
    history: List[Tuple[Dict, Dict]] = []
    input_prompt: BasePromptTemplate
    output_prompt: BasePromptTemplate
    human_role: str = "Human"
    ai_role: str = "AI"
    memory_key: str = "history"

    def clear(self):
        self.history = []
        self.base_chat_mem.clear()

    @property
    def memory_variables(self) -> List[str]:
        """Define the variables we are providing to the prompt."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Load the memory variables, in this case the entity key."""
        return {self.memory_key: self.base_chat_mem.load_memory_variables({})["history"]}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        self.history.append((inputs, outputs))
        chat_input = {"input": self.input_prompt.format_prompt(
            **{k: v for k, v in inputs.items() if k in self.input_prompt.input_variables}).to_string()}
        chat_output = {"output": self.output_prompt.format_prompt(
            **{k: v for k, v in outputs.items() if k in self.output_prompt.input_variables}).to_string()}
        self.base_chat_mem.save_context(chat_input, chat_output)


class PermanentTempMemory(BaseMemory, BaseModel):
    perm_mem: BaseMemory
    temp_mem: BaseMemory
    merger: Callable[[str, Any, Any], Any]

    def clear(self):
        self.temp_mem.clear()

    @property
    def memory_variables(self) -> List[str]:
        """Define the variables we are providing to the prompt."""
        return list(set(self.perm_mem.memory_variables).union(self.temp_mem.memory_variables))

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Load the memory variables, in this case the entity key."""
        perm_mem_vars = self.perm_mem.load_memory_variables({})
        temp_mem_vars = self.temp_mem.load_memory_variables({})
        for k in set(perm_mem_vars.keys()).intersection(temp_mem_vars.keys()):
            temp_mem_vars[k] = self.merger(k, perm_mem_vars[k], temp_mem_vars[k])
        perm_mem_vars.update(temp_mem_vars)
        return perm_mem_vars

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        self.temp_mem.save_context(inputs, outputs)


def list_to_markdown(l: Iterable[str]):
    result = ""
    result_template = "- %s\n"
    for a in l:
        result += result_template % a
    return result.strip()


class Client(BaseModel):
    calories: float
    carbs: float
    fat: float
    proteins: float
    activity: str = ""
    dietary: str = ""


def format_ideas(ideas):
    result = ""
    for f, idea_dict in ideas.items():
        for name, idea in idea_dict.items():
            result += config.ADJUST_MEAL_PLAN_TEMP.format(day=idea.day, name=idea.name, idea=idea.idea,
                                                          explanation=idea.explanation)
    return result.strip()


def format_target(feedback):
    idea = feedback.obj
    return config.TARGET_IDEA_TEMP.format(name=idea.name, day=idea.day)
