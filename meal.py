import json
from itertools import combinations
from typing import Iterable, Dict, List, Any
from datetime import datetime
from string import Formatter
import random

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import RetryWithErrorOutputParser, PydanticOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.schema import OutputParserException, AIMessage
from model import RandomExampleSelector
from pydantic import BaseModel, Field

import config
from model import PromptAdoptingParser


class RecipeMealInput(BaseModel):
    name: str
    type: str
    id: int


class MealValidOutput(BaseModel):
    valid: bool = Field(description=config.MEAL_VALID_DESC)
    thought: str = Field(description=config.MEAL_VALID_REASON)


class ChromaMealExampleSelector(BaseExampleSelector):

    def __init__(self, collection, fpos=1, fneg=1, acc=1):
        BaseExampleSelector.__init__(self)
        self.collection = collection
        self.fpos = fpos
        self.fneg = fneg
        self.acc = acc

    def add_example(self, example: Dict[str, str]) -> Any:
        raise NotImplementedError("Read only")

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        self.collection.query(query_texts=["Name: fried lobsters"], where={"valid": False}, n_results=1)
        fp_result = self.collection.query(query_texts=[input_variables["recipes"]], where={"classified": "fpos"},
                                          n_results=self.fpos)
        fn_result = self.collection.query(query_texts=[input_variables["recipes"]], where={"classified": "fneg"},
                                          n_results=self.fneg)
        acc_result = self.collection.query(query_texts=[input_variables["recipes"]], where={"classified": "accurate"},
                                           n_results=self.fneg)
        combined = []
        combined.extend(self._convert_query_result(fp_result))
        combined.extend(self._convert_query_result(fn_result))
        combined.extend(self._convert_query_result(acc_result))
        return combined

    def _convert_query_result(self, result) -> List[dict]:
        results = []
        for combo, metadata in zip(result["documents"][0], result["metadatas"][0]):
            results.append({
                "recipes": combo,
                "valid": metadata["valid"],
                "thought": metadata["thought"]
            })
        return results


def format_recipes(recipes: Iterable[RecipeMealInput]):
    result = []
    fieldnames = {fname for _, fname, _, _ in Formatter().parse(config.MEAL_VALID_RECIPE) if fname}
    for r in recipes:
        result.append(config.MEAL_VALID_RECIPE.format(**{k: v for k, v in r.dict().items() if k in fieldnames}))
    return "\n\n".join(result)


def create_appropriate_meal_examples(examples: Iterable[Dict[str, str]]):
    result = []
    human_template = HumanMessagePromptTemplate.from_template(config.MEAL_VALID_USER)
    ai_format = "Thought process:\n{thought}\n\nFinal response:\n{valid_response}"
    valid_true = "The given combination is appropriate for a meal"
    valid_false = "The given combination is not appropriate for a meal"
    for e in examples:
        result.append(human_template.format(recipes=e["recipes"]))
        if e["valid"]:
            valid_response = valid_true
        else:
            valid_response = valid_false
        result.append(AIMessage(content=ai_format.format(thought=e["thought"], valid_response=valid_response)))
    return result


def appropriate_meal_raw(model: BaseChatModel, recipes: Iterable[RecipeMealInput],
                         example_selector: BaseExampleSelector = None):
    recipes_str = format_recipes(recipes)
    examples = []
    if example_selector:
        examples = create_appropriate_meal_examples(example_selector.select_examples({"recipes": recipes_str}))
    system_template = SystemMessagePromptTemplate.from_template(config.MEAL_VALID_SYSTEM)
    human_template = HumanMessagePromptTemplate.from_template(config.MEAL_VALID_USER)
    prompt = ChatPromptTemplate.from_messages([system_template, *examples,
                                               human_template])

    prompt_val = prompt.format_prompt(recipes=recipes_str)
    token_counts = model.get_num_tokens_from_messages(prompt_val.to_messages())
    if token_counts > config.MEAL_VALID_MAX_TOKENS:
        raise ValueError("Token count {count} > {max_token}".format(count=token_counts,
                                                                    max_token=config.MEAL_VALID_MAX_TOKENS))

    raw_chain = LLMChain(llm=model, prompt=prompt, verbose=config.VERBOSE)
    raw_text = raw_chain(inputs={"recipes": recipes_str})["text"]
    return raw_text


def create_format_examples(examples: Iterable[Dict[str, str]]):
    result = []
    human_prompt = HumanMessagePromptTemplate.from_template(config.MEAL_VALID_FORMAT_USER)

    for e in examples:
        valid_obj = MealValidOutput(valid=e["valid"], thought=e["thought"])
        result.append(human_prompt.format(raw=e["raw"]))
        result.append(AIMessage(content=json.dumps(valid_obj.dict())))

    return result


def format_appropriate_meal(model: BaseChatModel, raw_text: str, examples: Iterable[Dict[str, str]] = ()):
    format_prompt = ChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate.from_template(config.MEAL_VALID_FORMAT_SYSTEM),
         *create_format_examples(examples),
         HumanMessagePromptTemplate.from_template(config.MEAL_VALID_FORMAT_USER)])
    output_parser = RetryWithErrorOutputParser.from_llm(
        parser=PydanticOutputParser(pydantic_object=MealValidOutput),
        llm=model)
    try:
        format_prompt = format_prompt.partial(format_instructions=output_parser.get_format_instructions())
    except NotImplementedError:
        format_prompt = format_prompt.partial(format_instructions="")
    prompt_val = format_prompt.format_prompt(raw=raw_text)
    token_counts = model.get_num_tokens_from_messages(prompt_val.to_messages())
    if token_counts > config.MEAL_VALID_MAX_TOKENS:
        raise ValueError("Token count {count} > {max_token}".format(count=token_counts,
                                                                    max_token=config.MEAL_VALID_MAX_TOKENS))
    output_parser = PromptAdoptingParser(base_parser=output_parser, prompt=prompt_val)
    parse_chain = LLMChain(llm=model, prompt=format_prompt, output_parser=output_parser,
                           verbose=config.VERBOSE)
    result = parse_chain(inputs={"raw": raw_text})["text"].dict()
    result["raw"] = raw_text
    return result


def evaluate_recipes(model: BaseChatModel, recipes: Iterable[RecipeMealInput],
                     format_selector: BaseExampleSelector = None,
                     meal_selector: BaseExampleSelector = None):
    format_examples = []
    if format_selector:
        format_examples = format_selector.select_examples(input_variables={})

    raw_text = appropriate_meal_raw(model, recipes, meal_selector)
    formatted = format_appropriate_meal(model, raw_text, format_examples)
    return formatted


def evaluate_shuffle_recipes(model: BaseChatModel, recipes: Iterable[RecipeMealInput],
                             start: int, end: int, format_selector: BaseExampleSelector = None,
                             meal_selector: BaseExampleSelector = None):
    all_combo = []
    for i in range(start, end):
        combos = list(combinations(recipes, i))
        random.shuffle(combos)
        all_combo.append(combos)
    random.shuffle(all_combo)
    elt_remaining = len(all_combo)
    while elt_remaining != 0:
        next_list_idx = random.randint(0, len(all_combo) - 1)
        next_list = all_combo[next_list_idx]
        next_idx = random.randint(0, len(next_list) - 1)
        try:
            formatted = evaluate_recipes(model, next_list[next_idx], format_selector, meal_selector)
            yield next_list[next_idx], formatted
        except (ValueError, OutputParserException):
            continue
        finally:
            del next_list[next_idx]
            if len(next_list) == 0:
                del all_combo[next_list_idx]
            elt_remaining = len(all_combo)


if __name__ == "__main__":
    import random
    random.seed(93)

    data_src = "data/recipes.json"
    data_out = "data/meal-{date}.json".format(date=int(datetime.now().timestamp()))
    format_examples = "data/format_examples.json"
    with open(config.API_KEY_PATH, "r") as fp:
        API_KEY = fp.read().strip()

    recipes_list = []
    with open(data_src, "r") as fp:
        recipes_raw = json.load(fp)
        for idx, rec in enumerate(recipes_raw):
            recipes_list.append(RecipeMealInput(name=rec["recipe_name"],
                                                type=", ".join(rec["cuisine_path"]), id=idx))
    plan_llm = ChatOpenAI(openai_api_key=API_KEY, temperature=config.MEAL_VALID_TEMP,
                          model_name=config.MEAL_VALID_MODEL, model_kwargs=dict(timeout=config.TIMEOUT))

    with open(format_examples, "r") as fp:
        examples = json.load(fp)
        selector = RandomExampleSelector(examples, k=2)

    with open(data_out, "a") as fp:
        count = 0
        for combo, valid in evaluate_shuffle_recipes(plan_llm, recipes_list, start=1, end=3, format_selector=selector):
            if count % 200 == 0:
                print("Processed: " + str(count))
            meal_idea = {"recipes": [c.dict() for c in combo],
                         "appropriate": valid}
            out_str = json.dumps(meal_idea).strip()
            fp.write(out_str + "\n")
            count += 1
