import json

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

import config
from model import MealIdeaScheduleOutput, RandomExampleSelector, Dietician, Client, list_to_markdown


@st.cache_resource
def load_examples():
    example_selector = RandomExampleSelector(examples=[])
    with open("data/themes.json", "r") as fp:
        example_dicts = json.load(fp)
        for d in example_dicts:
            example_selector.add_example(
                {"input": "For a day as an example without considering client information and explanation.",
                 "output": json.dumps(d)})
    return example_selector


plan_llm = ChatOpenAI(openai_api_key=config.API_KEY, temperature=config.THEME_TEMP,
                      model_name=config.THEME_MODEL_OPENAI)
output_parser = PydanticOutputParser(pydantic_object=MealIdeaScheduleOutput)
dietician = Dietician(planner_model=plan_llm, plan_parser=output_parser, plan_examples=load_examples())

if "submitted" not in st.session_state:
    st.session_state["submitted"] = False
if "traits" not in st.session_state:
    st.session_state["traits"] = [""]
planned = "plan" in st.session_state
submitted = st.session_state["submitted"]

with st.sidebar:
    st.write("## Your information")
    height = st.number_input("Height (cm)", min_value=0, disabled=submitted)
    weight = st.number_input("Weight (kg)", min_value=0, disabled=submitted)
    trait_1, trait_2 = st.columns(2)
    with trait_1:
        st.write("### Requirements")
    with trait_2:
        add_btn = st.button("Add", disabled=submitted)
    traits = st.session_state.traits
    if add_btn:
        traits.append("")
    for i, t in enumerate(traits):
        req_1, req_2 = st.columns([2.5, 1])
        with req_1:
            traits[i] = st.text_input(label="Requirement", key="trait_%d" % i, label_visibility="collapsed",
                                      placeholder="Low carb", value=traits[i], disabled=submitted)
        with req_2:
            def del_trait():
                trait_idx = int(i)

                def del_trait_func():
                    new_traits = list(st.session_state.traits)
                    del new_traits[trait_idx]
                    st.session_state["traits"] = new_traits

                return del_trait_func


            remove_req = st.button("🗑", key="Remove_%d" % i, on_click=del_trait(), disabled=submitted)

    def submit_for_plan():
        st.session_state["submitted"] = True

    adjust_btn = False
    if not submitted:
        submit_btn = st.button("Submit", type="primary", on_click=submit_for_plan)
    else:
        reset_col, adjust_col = st.columns(2)

        with reset_col:

            def reset_plan():
                st.session_state["submitted"] = False
                del st.session_state["plan"]

            st.button("Reset", on_click=reset_plan, disabled=not planned)
        with adjust_col:
            st.button("Adjust", disabled=not planned)


def display_plan(plan):
    if not plan:
        return
    for d in plan.daily_plan:
        st.write("## " + d)
        meals = plan.daily_plan[d]
        for m_name, m_idea in meals.items():
            label = "### {meal}: {idea}".format(meal=m_name[0].upper() + m_name[1:], idea=m_idea.idea)
            st.write(label)
            st.write(m_idea.explanation)


st.write("# Meal Plan")
if submitted and not planned:
    client_info = Client(height=height, weight=weight)
    client_info.dietary = list_to_markdown(st.session_state.traits)

    with st.spinner("Planning..."):
        plan = dietician.plan_meal(client_info)
    st.session_state["plan"] = plan
    st.experimental_rerun()

if "plan" not in st.session_state:
    st.write("Ready whenever you are")
else:
    display_plan(st.session_state["plan"])
