
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import shutil
import io
import unittest
from unittest.mock import patch


FILEPATH = "structured_restaurant_data.json"
BACKUP_PATH = "structured_restaurant_data.json.bak"

EXAMPLE_OUTPUT = """
    {{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier."
    "shortcomings": [],
    "itemId" : ""
    }}
"""

EXAMPLE_RESTAURANT_PARAGRAPH = (
    "Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, "
    "**casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** "
    "rating, it captures the salt-air energy of the coast through its signature "
    "beer-battered snapper tacos and zesty octopus ceviche, making it a premier "
    "spot for open-air dining near the pier. Price range: "
)

class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)
    itemId: int

def load_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except FileNotFoundError:
        return []

    except json.JSONDecodeError:
        print("Error: Invalid JSON file.")
        return []


# Update your restaurant_data_structure_prompt_generation
def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    base_system_msg = f"""
    You are an expert restaurant data extraction assistant.
    Extract restaurant information accurately.
    Return valid JSON only.
    Do not invent missing information.
    Follow the required JSON structure
    """
    
    base_user_prompt = f"""
    Task:
     Extract the restaurant information from the description
     and convert it into the JSON format shown in the example.

    Restaurant description:
    {restaurant_paragraph}

    Example:
    Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
    Output:
    {EXAMPLE_OUTPUT}
    
    """
    return base_system_msg, base_user_prompt


# Might need to explain why we are using granite here (cheap)
def vision_llm(system_msg, prompt_txt):
    #system_msg: input system message for the LLM
    #prompt_txt: input user prompt for the LLM
    #image_path: the file path of the input image

    ### Credentials of the model
    model_id = 'meta-llama/llama-4-maverick-17b-128e-instruct-fp8'
    project_id="skills-network"
    credentials = Credentials( url = "https://us-south.ml.cloud.ibm.com",)
    generate_params = {"max_tokens": 300}

    ### Step 2.1: Define the model by ModelInference
    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
        params=generate_params
    )
    ### Step 2.3: Define the messages for the model
    messages=[{"role": "system","content": system_msg},
              {"role": "user","content": prompt_txt}]
    ### Step 2.4: Get the response for the messages 
    response = model.chat(messages=messages)
    content = response["choices"][0]["message"]["content"]
    print(content)
    return content


def JSON_auto_repair_prompts(candidate_json_output, error_message):

    auto_repair_system_msg = """
    You are an expert JSON repair assistant.

    Your task is to repair invalid JSON and make it
    conform to the required restaurant data structure.

    Use the provided JSON candidate and validation error
    to identify and fix the problems.

    Do not invent missing information.
    Preserve all existing valid information.
    Return valid JSON only, without explanations or Markdown.
    """

    auto_repair_prompt = f"""
    Task:
    Repair the following JSON output based on the validation error.

    Candidate JSON:
    {candidate_json_output}

    Validation Error:
    {error_message}

    Expected JSON Structure:
    {EXAMPLE_OUTPUT}

    Instructions:
    - Fix JSON syntax errors.
    - Correct data types when the intended value is clear.
    - Follow the expected JSON structure.
    - Do not invent missing information.
    - Preserve existing valid values.
    - Return valid JSON only.
    """

    return auto_repair_system_msg, auto_repair_prompt



def new_data_entry_process(paragraph, itemId):
    # Step 1: Generate the system message and user prompt
    system_msg, prompt_txt = restaurant_data_structure_prompt_generation(
        paragraph
    )

    # Step 2: Generate the initial JSON output
    for attempt in range(3):
        candidate_json_output = vision_llm(system_msg,prompt_txt)
        # Remove Markdown code formatting if present
        candidate_json_output = candidate_json_output.strip()
        if candidate_json_output.startswith("```"):
            candidate_json_output = (
                candidate_json_output
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
        # Step 4: Validate the JSON output
        try:
            restaurant_data = json.loads(candidate_json_output)
            restaurant_data["itemId"] = itemId

            validated_restaurant = Restaurant(**restaurant_data)
            print("Restaurant:", validated_restaurant)
            # Return validated restaurant data
            return validated_restaurant.model_dump()

        except (json.JSONDecodeError, ValidationError) as error:

            # Step 5: Generate an auto-repair prompt
            repair_system_msg, repair_prompt = (
                JSON_auto_repair_prompts(
                    candidate_json_output,
                    str(error)
                )
            )
            system_msg = repair_system_msg
            prompt_txt = repair_prompt
    print("Error: Unable to generate valid restaurant data.")
    return None

def show_restaurant_card(restaurant):
    for key, value in restaurant.items():
        print(f"{key}: {value}")

def search_restaurant_by_index(data,index):
    for restaurant in data:
        if restaurant["itemId"] == index:
            show_restaurant_card(restaurant)
            return True

def find_restaurant_position(data, item_id):
    for position, restaurant in enumerate(data):
        if restaurant.get("itemId") == item_id:
            return position

    return None

def save_data(data, file_path, backup_path):
    # Create a backup before overwriting the original file
    if os.path.exists(file_path):
        shutil.copy2(file_path, backup_path)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------
# Restaurant Database Management UI
# ---------------------------------------------------------------------
def manage_restaurants(file_path, backup_path):

    while True:
        data = load_data(file_path)

        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ")

        # BROWSE ALL
        if choice == "1":
            print("\n--- Current Listings ---")

            for restaurant in data:
                print(restaurant.get("name", "N/A"))

        # VIEW RECORD
        elif choice == "2":
            try:
                index = int(input("Enter the value of ID: "))

                found = search_restaurant_by_index(data, index)

                if not found:
                    print("❌ Invalid ID.")

            except ValueError:
                print("❌ Please enter a valid numeric ID.")

        # WRITE OPERATIONS
        elif choice in ["3", "4", "5"]:

            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")

            confirm = input(
                "Are you sure? (type 'yes' to proceed): "
            ).lower()

            if confirm != "yes":
                print("Operation cancelled.")
                continue

            # ADD NEW RESTAURANT
            if choice == "3":

                try:
                    # Generate a unique ID
                    existing_ids = [
                        restaurant.get("itemId", 1000000)
                        for restaurant in data
                    ]

                    itemId = max(existing_ids, default=1000000) + 1

                    restaurant_data = input(
                        "Input a new restaurant description:\n"
                    )

                    restaurant_model = new_data_entry_process(
                        restaurant_data, itemId
                    )

                    if restaurant_model is None:
                        print("❌ Failed to generate valid restaurant data.")

                    else:
                        data.append(restaurant_model)

                        save_data(data, file_path, backup_path)

                        print("✅ Restaurant added successfully.")

                except (ValueError, TypeError, KeyError) as error:
                    print(f"❌ Data processing error: {error}")

                except Exception as error:
                    print(f"❌ Unexpected error: {error}")

            # EDIT RESTAURANT
            elif choice == "4":

                try:
                    index = int(input("Enter the ID of the restaurant to edit: "))

                    position = find_restaurant_position(data, index)

                    if position is None:
                        print("❌ Invalid ID.")
                        continue

                    # Display the current restaurant
                    show_restaurant_card(data[position])

                    restaurant_data = input(
                        "\nEnter the updated restaurant description:\n"
                    )

                    # Keep the same ID for the updated restaurant
                    restaurant_model = new_data_entry_process(
                        restaurant_data, index
                    )

                    if restaurant_model is None:
                        print("❌ Failed to update restaurant.")

                    else:
                        # Replace the existing record, do not append
                        data[position] = restaurant_model

                        save_data(data, file_path, backup_path)

                        print("✅ Record updated successfully.")

                except ValueError:
                    print("❌ Please enter a valid numeric ID.")

                except Exception as error:
                    print(f"❌ Unexpected error: {error}")

            # DELETE RESTAURANT
            elif choice == "5":

                try:
                    index = int(input("Enter the ID of the restaurant to delete: "))

                    position = find_restaurant_position(data, index)

                    if position is None:
                        print("❌ Invalid ID.")

                    else:
                        deleted_restaurant = data.pop(position)

                        save_data(data, file_path, backup_path)

                        print(
                            f"✅ {deleted_restaurant.get('name', 'Restaurant')} "
                            "deleted successfully."
                        )

                except ValueError:
                    print("❌ Please enter a valid numeric ID.")

                except Exception as error:
                    print(f"❌ Unexpected error: {error}")

        # EXIT
        elif choice == "6":
            print("Goodbye!")
            break

        else:
            print("Invalid input.")

if __name__ == "__main__":
    manage_restaurants(
                FILEPATH,
                BACKUP_PATH
            )

    # unittest.main()  # Unit Test

    # Actual UI Call (run separately)
    # manage_restaurants(FILEPATH, BACKUP_PATH)
