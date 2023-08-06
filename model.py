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
    HumanMessagePromptTemplate, )
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.schema import BaseOutputParser, BaseMemory, PromptValue
from pydantic import BaseModel, Field, validator, Extra

import config

misc_llm = ChatOpenAI(openai_api_key=config.API_KEY,
                      temperature=config.MISC_TEMP,
                      model_name=config.MISC_MODEL_OPENAI)


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


class DieticianPlanChain(Chain):
    system_prompt: BasePromptTemplate
    command_prompt: BasePromptTemplate
    user_prompt: BasePromptTemplate
    format_prompt: BasePromptTemplate
    format_cmd: BasePromptTemplate
    output_plan: str = "plan"
    output_raw: str = "raw_plan"
    output_tokens: str = "tokens"
    user_profile: str = "profile"
    plan_llm: BaseChatModel
    parse_llm: BaseChatModel

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        input_set = set(self.system_prompt.input_variables)
        input_set = input_set.union(self.command_prompt.input_variables)
        input_set = input_set.union([self.user_profile])
        return list(input_set)

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_plan, self.output_tokens]

    def predict(self, callbacks=None, **inputs):
        result = self(inputs=inputs, return_only_outputs=True, callbacks=callbacks)
        return result[self.output_plan], result[self.output_tokens]

    def _format_idea(self, day_plan, run_manager: Optional[CallbackManagerForChainRun] = None):
        if not self.format_prompt.output_parser:
            raise NotImplementedError("Need output parser in parser prompt")
        parse_prompt = self._format_prompt
        parse_prompt_val = parse_prompt.format_prompt(raw_text=day_plan)
        adopted_parser = PromptAdoptingParser(base_parser=self.format_prompt.output_parser, prompt=parse_prompt_val)
        parse_chain = LLMChain(prompt=parse_prompt, llm=self.parse_llm, output_parser=adopted_parser,
                               verbose=self.verbose)
        return parse_chain.predict(callbacks=run_manager.get_child() if run_manager else None, raw_text=day_plan)

    @property
    def _format_prompt(self):
        parse_system_template = SystemMessagePromptTemplate(prompt=self.format_prompt.partial(
            format_instruction=self.format_prompt.output_parser.get_format_instructions()))
        parse_human_template = HumanMessagePromptTemplate(prompt=self.format_cmd)
        parse_chat_template = ChatPromptTemplate.from_messages([parse_system_template, parse_human_template])
        return parse_chat_template

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        messages = self._generate_plan_messages(inputs, run_manager)
        if self.verbose:
            if run_manager:
                run_manager.on_text("Prompt: " + messages.to_string() + "\n")
        messages_typed = messages.to_messages()
        with get_openai_callback() as cb:
            msg_out = self.plan_llm.generate([messages_typed],
                                             callbacks=run_manager.get_child() if run_manager else None)
        if self.verbose:
            if run_manager:
                run_manager.on_text("\nTotal Tokens: %d\n" % cb.total_tokens)
        return {self.output_plan: self._format_idea(msg_out.generations[0][0].text, run_manager),
                self.output_raw: msg_out.generations[0][0].text,
                self.output_tokens: cb.total_tokens}

    async def _acall(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        messages = self._generate_plan_messages(inputs, run_manager)
        if self.verbose:
            if run_manager:
                await run_manager.on_text("Prompt: " + messages.to_string() + "\n")
        messages_typed = messages.to_messages()
        with get_openai_callback() as cb:
            msg_out = await self.plan_llm.agenerate([messages_typed],
                                                    callbacks=run_manager.get_child() if run_manager else None)
        if self.verbose:
            if run_manager:
                await run_manager.on_text("\nTotal Tokens: %d\n" % cb.total_tokens)
        return {self.output_plan: self._format_idea(msg_out.generations[0][0].text, run_manager),
                self.output_raw: msg_out.generations[0][0].text,
                self.output_tokens: cb.total_tokens}

    def _generate_plan_messages(self, inputs, run_manager=None):
        if "history" in inputs:
            chat_hist = inputs["history"]
            if self.verbose:
                if run_manager:
                    run_manager.on_text("Current History: " + str(chat_hist) + "\n")
        else:
            chat_hist = []
        self._prep_input_dict(inputs)
        theme_system_template = SystemMessagePromptTemplate(prompt=self.system_prompt)
        theme_profile_template = SystemMessagePromptTemplate(prompt=self.user_prompt)
        theme_human_template = HumanMessagePromptTemplate(prompt=self.command_prompt)
        theme_chat_template = ChatPromptTemplate.from_messages([theme_system_template, theme_profile_template,
                                                                *chat_hist, theme_human_template])
        final_prompt = theme_chat_template.format_prompt(**inputs)
        return final_prompt

    def _prep_input_dict(self, inputs):
        inputs.update(inputs[self.user_profile])
        del inputs[self.user_profile]


class RandomExampleSelector(BaseExampleSelector):

    def __init__(self, examples: List[Dict[str, str]], k=3):
        self.examples = examples
        self.k = 3

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
    weight: float
    height: float
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


def create_planner_chain(plan_model, parse_model, memory_base):
    output_parser = RetryWithErrorOutputParser.from_llm(
        parser=PydanticOutputParser(pydantic_object=MealIdeaScheduleOutput),
        llm=parse_model)
    system_template = PromptTemplate.from_template(config.THEME_SYSTEM)
    commmand_template = PromptTemplate.from_template(config.THEME_COMMAND)
    client_template = PromptTemplate.from_template(config.CLIENT_PROFILE_SYSTEM)
    format_system = PromptTemplate.from_template(config.THEME_PARSE_SYSTEM, output_parser=output_parser)
    format_cmd = PromptTemplate.from_template(config.THEME_PARSE_USER)

    temp_memory = KeyValueMemory(base_chat_mem=memory_base, input_prompt=commmand_template,
                                 output_prompt=PromptTemplate.from_template("{raw_plan}"))
    perm_memory = ConversationBufferMemory(return_messages=True)
    memory = PermanentTempMemory(perm_mem=perm_memory, temp_mem=temp_memory, merger=lambda k, v1, v2: v1 + v2)
    plan_chain = DieticianPlanChain(system_prompt=system_template, command_prompt=commmand_template,
                                    user_prompt=client_template, plan_llm=plan_model, memory=memory,
                                    format_prompt=format_system, format_cmd=format_cmd, parse_llm=parse_model,
                                    verbose=True)
    return plan_chain


def create_adjuster_chain(plan_model, parse_model):
    output_parser = RetryWithErrorOutputParser.from_llm(
        parser=PydanticOutputParser(pydantic_object=MealIdeaLLMOutput),
        llm=parse_model)
    system_template = PromptTemplate.from_template(config.ADJUST_SYSTEM)
    commmand_template = PromptTemplate.from_template(config.ADJUST_COMMAND)
    client_template = PromptTemplate.from_template(config.CLIENT_PROFILE_SYSTEM)
    format_system = PromptTemplate.from_template(config.THEME_PARSE_SYSTEM, output_parser=output_parser)
    format_cmd = PromptTemplate.from_template(config.THEME_PARSE_USER)
    chain = DieticianPlanChain(system_prompt=system_template, command_prompt=commmand_template,
                               user_prompt=client_template, plan_llm=plan_model,
                               format_prompt=format_system, format_cmd=format_cmd, parse_llm=parse_model,
                               verbose=True)
    return chain


class Dietician:

    def __init__(self, planner_model: BaseChatModel,
                 plan_examples: BaseExampleSelector):
        self.planner_model = planner_model
        self.plan_examples = plan_examples

    def plan_meal(self, client_info: Client):
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        daily_plan = {}
        result = MealPlan(daily_plan=daily_plan)
        chain = self._prepare_plan_chain()

        for d in days:
            input_dict = {"day": d, chain.user_profile: client_info}
            output, tokens = chain.predict(**input_dict)
            daily_dict = {s.name: MealIdea.from_llm(d, s) for s in
                          output.schedule}
            daily_plan[d] = daily_dict

        return result

    def adjust_plan(self, client_info: Client, plan: MealPlan):
        chain = self._prepare_adjust_chain()
        result = MealPlan(daily_plan=dict(plan.daily_plan))
        for f in plan.all_feedbacks:
            input_dict = {"orig_plan": format_ideas(plan.daily_plan), chain.user_profile: client_info,
                          "target": format_target(f), "feedback": f.content}
            output, tokens = chain.predict(**input_dict)
            result.daily_plan[f.obj.day][output.name] = MealIdea.from_llm(f.obj.day, output)
        return result

    def _prepare_plan_chain(self):
        memory_base = ConversationBufferWindowMemory(k=config.THEME_HISTORY,
                                                     return_messages=True)
        # for e in self.plan_examples.select_examples({}):
        #     perm_memory.save_context(
        #         {"daily_instruction": e["input"]},
        #         {"plan": e["output"]}
        #     )
        plan_chain = create_planner_chain(self.planner_model, misc_llm, memory_base)
        return plan_chain

    def _prepare_adjust_chain(self):
        return create_adjuster_chain(self.planner_model, misc_llm)


if __name__ == "__main__":

    def test_chain():
        plan_llm = ChatOpenAI(openai_api_key=config.API_KEY, temperature=config.THEME_TEMP,
                              model_name=config.THEME_MODEL_OPENAI)
        memory_base = ConversationBufferWindowMemory(k=config.THEME_HISTORY,
                                                     return_messages=True)
        plan_chain = create_planner_chain(plan_llm, misc_llm, memory_base)
        profile = Client(weight=68, height=170)
        for d in ["Monday", "Tuesday", "Wednesday"]:
            print(plan_chain.predict(**{"day": d, plan_chain.user_profile: profile}))


    def test_dietician():
        plan_llm = ChatOpenAI(openai_api_key=config.API_KEY, temperature=config.THEME_TEMP,
                              model_name=config.THEME_MODEL_OPENAI)

        example_selector = RandomExampleSelector(examples=[])
        with open("data/themes.json", "r") as fp:
            example_dicts = json.load(fp)
            for d in example_dicts:
                example_selector.add_example(
                    {"input": "For a day as an example without considering client information and explanation.",
                     "output": json.dumps(d)})

        dietician = Dietician(planner_model=plan_llm, plan_examples=example_selector)
        client_info = Client(height=175, weight=62.6)
        client_info.dietary = list_to_markdown(["Vegan", "Gluten Intolerance"])
        plan = dietician.plan_meal(client_info)
        print(format_ideas(plan.daily_plan))
        plan.daily_plan["Monday"]["breakfast"].feedback = "I want some fusion food"
        adj_plan = dietician.adjust_plan(client_info, plan)
        print(format_ideas(adj_plan.daily_plan))


    test_dietician()
