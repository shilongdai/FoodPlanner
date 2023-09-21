from io import StringIO
from typing import Iterable

import streamlit as st
import json
import random
from meal import RecipeMealInput

SEED = 93
TRUE_SAMPLE = 100
FALSE_SAMPLE = 100
random.seed(SEED)


def format_recipes(recipes: Iterable[RecipeMealInput]):
    result = []
    format = "**Id**: {id}\n\n**Name**: {name}\n\n**Type**: {type}\n\n**Ingredients**: {ingredients}"
    for r in recipes:
        result.append(format.format(**r.dict()))
    return "\n\n".join(result)


@st.cache_data
def sample_prelabeled(fp):
    entries = json.load(fp)
    for i, entry in enumerate(entries):
        if entry["appropriate"]["valid"]:
            true_entries.append(i)
        else:
            false_entries.append(i)
    true_samples = random.sample(true_entries, min(TRUE_SAMPLE, len(true_entries)))
    false_samples = random.sample(false_entries, min(FALSE_SAMPLE, len(false_entries)))
    final_sample = sorted(true_samples + false_samples)
#    random.shuffle(final_sample)
    return [entries[i] for i in final_sample]


with st.sidebar:
    true_entries = []
    false_entries = []
    fpb = st.file_uploader("Upload GPT Labeled File")

if fpb:
    with StringIO(fpb.getvalue().decode("utf-8")) as fp:
        final_sample = sample_prelabeled(fp)
    if "init" not in st.session_state:
        st.session_state["init"] = [m["appropriate"]["valid"] for m in final_sample]

    for i, s in enumerate(final_sample):
        st.write("## Recipe %d" % i)
        st.write(format_recipes([RecipeMealInput(**j) for j in s["recipes"]]))
        update_changed = False
        if "changed" not in s:
            s["changed"] = False
            update_changed = True
        s["appropriate"]["thought"] = st.text_area("Explanation of Appropriateness",
                                                  value=s["appropriate"]["thought"], key="reason_" + str(i))
        s["appropriate"]["valid"] = st.checkbox("Appropriate", value=s["appropriate"]["valid"],
                                                key="check_" + str(i))
        if s["appropriate"]["valid"] != st.session_state.init[i] and update_changed:
            s["changed"] = True

    st.download_button("Download", data=json.dumps(final_sample), file_name="labeled_meals.json", mime='text/json')
