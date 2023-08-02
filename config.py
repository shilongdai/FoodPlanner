API_KEY_PATH = "./api_key"
with open(API_KEY_PATH, "r") as fp:
    API_KEY = fp.read().strip()
THEME_TEMP = 0.3
THEME_MODEL_OPENAI = "gpt-3.5-turbo"
THEME_SYSTEM = "You are an experienced dietician and chief. " \
               "Your goal is to create a meal plan for a client on a day to day basis. " \
               "The plan should include healthy and delicious food. " \
               "To come up with the plan, you will ideate the general theme for the breakfast, lunch, and dinner for one day of the week. " \
               "The theme must be at a category/cuisine level without consideration of the specific dishes, ingredients, or recipes. " \
               "Try to have varieties in themes over the days." \
               "\n\n{format_instructions}"
THEME_COMMAND = "Come up with the themes for {day}. Tailor the plan to the profile of the client. Provide concise but adequate explanation."
THEME_EXAMPLES = "./data/themes.json"
THEME_MEAL_NAME_DESC = "The name of the meal. Can be one of breakfast, lunch, or dinner."
THEME_MEAL_IDEA_DESC = "Idea for the meal."
THEME_MEAL_EXP = "Brief summary of why this idea is a good fit."
THEME_MEAL_NAME_VALID = {"breakfast", "lunch", "dinner"}
THEME_HISTORY = 2

CLIENT_PROFILE_SYSTEM = "The client has the following profile and description:\n" \
                        "Height: {height} cm\nWeight: {weight} kg\nActivity Level:\n{activity}\n" \
                        "Dietary Considerations:\n{dietary}"

ADJUST_SYSTEM = "You are an experienced dietician and chief. " \
                "Your goal is to create a meal plan for a client on a day to day basis. " \
                "The plan should include healthy and delicious food, and have varieties over the days. " \
                "To come up with the plan, you decided to ideate the general theme for the breakfast, lunch, and dinner for one day of the week. " \
                "The theme must be at a category/cuisine level without consideration of the specific dishes, ingredients, or recipes. " \
                "You have presented a draft plan to the client and received some feedbacks. " \
                "You intend to adjust the draft appropriately according to the feedbacks." \
                "\n\n{format_instructions}"
ADJUST_COMMAND = "The original plan:\n{orig_plan}\n\n" \
                 "The feedback for {target}:\n{feedback}\n\n" \
                 "Propose the adjusted version given the feedback."
ADJUST_MEAL_PLAN_TEMP = "Day: {day}\nMeal: {name}\nIdea: {idea}\nExplanation: {explanation}\n\n"
TARGET_IDEA_TEMP = "{name} on {day}"

MISC_TEMP = 0
MISC_MODEL_OPENAI = "gpt-3.5-turbo"
