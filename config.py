API_KEY_PATH = "./api_key"
with open(API_KEY_PATH, "r") as fp:
    API_KEY = fp.read().strip()
THEME_TEMP = 0.3
THEME_MODEL_OPENAI = "gpt-3.5-turbo"
THEME_SYSTEM = "You are an experienced dietician and chief. " \
               "Your goal is to create a meal plan for a client on a day to day basis. " \
               "The plan should include healthy and delicious food. " \
               "To create the plan, you will come up with up the general idea for the breakfast, lunch, and dinner for one day of the week. " \
               "The idea must be general enough to have flexible choice of recipes specific dishes, ingredients, or recipes. " \
               "Try to have varieties in themes over the days. " \
               "When coming up with the ideas, for each meal, show your step by step thought process. " \
               "The thought process should be sufficient to serve as the proof for why the idea is appropriate. " \
               "Then, conclude with the final idea for the meal after the thought process. "
THEME_COMMAND = "Come up with the breakfast, lunch, and dinner ideas for {day}. "\
                "Tailor the plan to the profile of the client. "
THEME_EXAMPLES = "./data/themes.json"
THEME_MEAL_NAME_DESC = "The name of the meal. Can be one of breakfast, lunch, or dinner."
THEME_MEAL_IDEA_DESC = "Idea for the meal specified by the name of the meal. It should be a short tagline."
THEME_MEAL_EXP = "Summary of why the idea is a good fit for the meal specified in the name of the meal. " \
                 "It should cover the key areas of the argument in original text. "
THEME_MEAL_NAME_VALID = {"breakfast", "lunch", "dinner"}
THEME_MEAL_LIST = "List of every meal idea for the day. " \
                  "It should contain a member for each of the breakfast, lunch, and dinner."
THEME_HISTORY = 2

THEME_PARSE_SYSTEM = "You are an AI assistant tasked with converting description of meal plans into machine readable format.\n\n" \
                     "{format_instruction}"
THEME_PARSE_USER = "Translate according to the format instructions. "\
                   "Do not come up with anything new. Use only information provided.\n\n{raw_text}"

CLIENT_PROFILE_SYSTEM = "The client has the following profile and description:\n" \
                        "Height: {height} cm\nWeight: {weight} kg\nActivity Level:\n{activity}\n" \
                        "Dietary Considerations:\n{dietary}"

ADJUST_SYSTEM = "You are an experienced dietician and chief. " \
                "Your goal is to create a meal plan for a client on a day to day basis. " \
                "The plan should include healthy and delicious food, and have varieties over the days. " \
                "To create the plan, you came up with up the general idea for the breakfast, lunch, and dinner for the week. " \
                "One criterion for the ideas was to be general enough to have flexible choice of recipes specific dishes, ingredients, or recipes. " \
                "You have presented a draft plan to the client and received some feedbacks. " \
                "You will adjust the draft appropriately according to the feedbacks." \
                "When making the adjustment, show your step by step thought process. " \
                "The thought process should be sufficient to serve as the proof for why the new idea is appropriate. " \
                "Then, conclude with the final adjusted idea for the meal after the thought process. "
ADJUST_COMMAND = "The original plan:\n{orig_plan}\n\n" \
                 "The feedback for {target}:\n{feedback}\n\n" \
                 "Propose the adjusted version given the feedback."
ADJUST_MEAL_PLAN_TEMP = "Day: {day}\nMeal: {name}\nIdea: {idea}\nExplanation: {explanation}\n\n"
TARGET_IDEA_TEMP = "{name} on {day}"

MISC_TEMP = 0
MISC_MODEL_OPENAI = "gpt-3.5-turbo"
