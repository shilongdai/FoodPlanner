API_KEY_PATH = "./api_key"
VERBOSE = False
TIMEOUT = 120
MEAL_VALID_SYSTEM = "You are an AI assistant that determines if a combination of recipes taken together would be appropriate. " \
                    "A combination of recipes is appropriate if and only if all recipes taken together can be a standalone breakfast, lunch, or dinner. " \
                    "Do not consider the nutritional values, but consider whether the recipes can complement each other in flavors or cuisines, or whether the recipes would make for a conventional combination. " \
                    "Only consider the combination given. Do not assume that anything besides the recipes listed would be a part of the meal. " \
                    "Thus, assume that a side dish such as salad or rice would not be provided unless explicitly mentioned. " \
                    "For each list of recipes, first give your thought process on determining whether it is appropriate, and then give your final response. "
MEAL_VALID_FORMAT_SYSTEM = "You are an AI assistant that will format user inputs into json.\n{format_instructions}"
MEAL_VALID_FORMAT_USER = "Format:\n{raw}"
MEAL_VALID_USER = "Start of combination\n\n{recipes}\n\nEnd of combination\n\n"\
                  "Is this combination an appropriate standalone breakfast, lunch or dinner?"
MEAL_VALID_RECIPE = "Name: {name}\nType: {type}\nIngredients: {ingredients}"
MEAL_VALID_DESC = "Whether the combination is appropriate for breakfast, lunch, or dinner"
MEAL_VALID_REASON = "The exact copy of the thought process given"
MEAL_VALID_REINFORCEMENT = "Excellent. "
MEAL_VALID_TEMP = 0
MEAL_VALID_MODEL = "gpt-3.5-turbo"
MEAL_VALID_MAX_TOKENS = 1024 * 3
MEAL_VALID_EXAMPLE_COL = "label_examples"
MEAL_VALID_EXAMPLE_CHROMBA = "./data/meal_db"

THEME_TEMP = 0.3
THEME_MODEL_OPENAI = "gpt-3.5-turbo"
THEME_SYSTEM = "You are an experienced dietician and meal planner. " \
               "Your goal is to create a meal plan for a client on a day to day basis. " \
               "The plan should include healthy and delicious food. " \
               "To create the plan, you will come up with up the general idea for the breakfast, lunch, and dinner for one day of the week. " \
               "The idea must be general enough to have flexible choice of recipes specific dishes, ingredients, or recipes. " \
               "Try to have varieties in themes over the days. " \
               "When coming up with the ideas, for each meal, show your step by step thought process. " \
               "The thought process should be sufficient to serve as the proof for why the idea is appropriate. " \
               "Then, conclude with the final idea for the meal after the thought process. "
THEME_COMMAND = "Come up with the breakfast, lunch, and dinner ideas for {day}. " \
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
THEME_PARSE_USER = "Translate according to the format instructions. " \
                   "Do not come up with anything new. Use only information provided.\n\n{raw_text}"

CLIENT_PROFILE_SYSTEM = "The client has the following profile and description:\n" \
                        "Daily caloric goal: {calories} cal\nCarbohydrate goal: {carbs}g\nProtein goal: {proteins}g\n" \
                        "Fat goal: {fat}g\n" \
                        "Dietary Considerations:\n{dietary}"

ADJUST_SYSTEM = "You are an experienced dietician and meal planner. " \
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

RECIPE_SUGGEST_SYSTEM = "You are an experienced dietician and meal planner. " \
                        "Your goal is to create a meal plan for a client on a day to day basis. " \
                        "To create the plan, you came up with up the general idea for the breakfast, lunch, and dinner for the week. " \
                        "Thus, for each of the ideas, you will think of at least 3 likely combinations of recipes that together would create the meal. " \
                        "The recipes should be easy to find in resources such as the internet, cookbooks etc. " \
                        "Thus, they should not be excessively obscure or specific so that the chance of finding the exact recipe is low. " \
                        "Show your step by step thought process for how you came up with the combinations and the recipes in each combinations. " \
                        "Then, after providing the thought process, provide the recipe combinations. "

RECIPE_BUDGET_SYSTEM = "The client has the following caloric and macro-nutrient budget left for the day:\n" \
                       "Calories: {calories} cal\nCarbohydrate: {carbs}g\nProteins: {proteins}g\n" \
                       "Fat: {fat}g\n\n" \
                       "The client has the following dietary considerations:\n{dietary}"
RECIPE_SUGGEST_COMMAND = "Come up with the expected recipes for making the dishes for the following meal idea:\nName: {meal_name}\n" \
                         "Explanation: {meal_explanation}"

RECIPE_SEARCH_SYSTEM = "You are an AI Assistant tasked with helping a human to find recipes. " \
                       "All of the available recipes are located in a database that can be searched via the name of the recipe. " \
                       "The database will return recipes with names semantically similar to the input query. "

MISC_TEMP = 0
MISC_MODEL_OPENAI = "gpt-3.5-turbo"
