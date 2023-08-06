import json

import streamlit as st
from langchain.chat_models import ChatOpenAI

import config
from model import RandomExampleSelector, Dietician, Client, list_to_markdown


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
dietician = Dietician(planner_model=plan_llm, plan_examples=load_examples())

if "submitted" not in st.session_state:
    st.session_state["submitted"] = False
if "adjusting" not in st.session_state:
    st.session_state["adjusting"] = False
if "traits" not in st.session_state:
    st.session_state["traits"] = [""]
if "feedback-texts" not in st.session_state:
    st.session_state["feedback-texts"] = {}
planned = "plan" in st.session_state
submitted = st.session_state["submitted"]
adjusting = st.session_state["adjusting"]


def submit_for_plan():
    st.session_state["submitted"] = True


def submit_feedback():
    st.session_state["adjusting"] = True


def disable_feedbacks():
    for k in st.session_state:
        if k.endswith("feedback"):
            st.session_state[k] = False


def create_input_section():
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

            st.button("🗑", key="Remove_%d" % i, on_click=del_trait(), disabled=submitted)
    return height, weight


with st.sidebar:
    height, weight = create_input_section()
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
            adjust_disabled = (not planned and not adjusting) or (planned and adjusting)
            st.button("Adjust", disabled=adjust_disabled, on_click=submit_feedback)


def display_plan(plan):
    if not plan:
        return
    for d in plan.daily_plan:
        st.write("## " + d)
        meals = plan.daily_plan[d]
        for m_name, m_idea in meals.items():
            label = "### {meal}: {idea}".format(meal=m_name[0].upper() + m_name[1:], idea=m_idea.idea)
            st.write(label)
            exp_col, feedback_col = st.columns([2, 1])
            with exp_col:
                st.write(m_idea.explanation)
            with feedback_col:
                which_feedback = "{day}-{name}-feedback".format(day=d, name=m_name)

                def toggle_feedback(which):
                    if which not in st.session_state:
                        st.session_state[which] = False
                    st.session_state[which] = not st.session_state[which]

                st.button("Feedback", key=which_feedback + "-btn-key", kwargs=dict(which=which_feedback),
                          on_click=toggle_feedback)
            if which_feedback in st.session_state and st.session_state[which_feedback]:
                feedbacks = st.session_state["feedback-texts"]
                feedback = st.text_area(label="Your feedback", key=which_feedback + "-text-key")
                if len(feedback.strip()) > 0:
                    feedbacks[(d, m_name)] = feedback


st.write("# Meal Plan")
client_info = Client(height=height, weight=weight)
client_info.dietary = list_to_markdown(st.session_state.traits)
if submitted and not planned:
    with st.spinner("Planning..."):
        plan = dietician.plan_meal(client_info)
    st.session_state["plan"] = plan
    st.experimental_rerun()

if adjusting:
    plan = st.session_state["plan"]
    feedbacks = st.session_state["feedback-texts"]
    if len(feedbacks) > 0:
        for (day, meal), text in feedbacks.items():
            plan.daily_plan[day][meal].feedback = text
        with st.spinner("Adjusting..."):
            new_plan = dietician.adjust_plan(client_info, plan)
        st.session_state["plan"] = new_plan
        st.session_state["adjusting"] = False
        disable_feedbacks()
        st.experimental_rerun()

if "plan" not in st.session_state:
    st.write("Ready whenever you are")
else:
    display_plan(st.session_state["plan"])
