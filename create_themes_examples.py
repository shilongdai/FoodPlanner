import sys
from model import MealIdeaLLMOutput
from model import MealIdeaScheduleOutput
import json
import config


if __name__ == "__main__":
    finished = False
    result = []
    while True:
        plan = MealIdeaScheduleOutput(schedule=[])
        for m in config.THEME_MEAL_NAME_VALID:
            print("Meal Name: " + m)
            idea = input("Meal Idea: ")
            if idea.strip() == ".":
                finished = True
                break
            plan.schedule.append(MealIdeaLLMOutput(meal_name=m, meal_idea=idea))
        if finished:
            break
        result.append(plan.dict())

    with open(sys.argv[1], "w") as fp:
        json.dump(result, fp)
