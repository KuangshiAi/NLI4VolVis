import socket
import json
import os
import torch
import numpy as np
from open_clip import tokenize
from openai import OpenAI
import collections

MAX_HISTORY_SIZE = 30  # Adjust this limit as needed

def manage_conversation_history(history, new_message):
    """
    Manage the conversation history by appending a new message and ensuring
    it does not exceed MAX_HISTORY_SIZE.
    """
    history.append(new_message)
    if len(history) > MAX_HISTORY_SIZE:
        history.pop(0)  # Remove the oldest message

def load_best_frames(source_dir, tf_name):
    file_path = os.path.join(source_dir, f"{tf_name}/best_frames.txt")

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Read the content of the file
    with open(file_path, "r") as file:
        content = file.readlines()

    # Process and return the content (remove newline characters and strip whitespace)
    best_frames = [line.strip() for line in content]

    return best_frames

def embed_text(description, model):
    """Embeds a textual description using CLIP."""
    tokenized_text = tokenize([description])
    with torch.no_grad():
        text_embedding = model.encode_text(tokenized_text).squeeze(0).numpy()
    return text_embedding

def find_best_tf(description_embedding, tf_embeddings):
    """Finds the TF with the highest similarity to the description embedding."""
    best_tf = None
    highest_similarity = -float("inf")

    for tf_name, tf_embedding in tf_embeddings.items():
        similarity = np.dot(description_embedding, tf_embedding) / (
            np.linalg.norm(description_embedding) * np.linalg.norm(tf_embedding)
        )
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_tf = tf_name

    return best_tf, highest_similarity

def find_best_tfs(description_embedding, description, tf_embeddings, threshold=0.01):
    """Finds the TFs with the highest similarity to the description embedding."""
    similarity_results = []

    for tf_name, tf_embedding in tf_embeddings.items():
        similarity = np.dot(description_embedding, tf_embedding) / (
            np.linalg.norm(description_embedding) * np.linalg.norm(tf_embedding)
        )
        similarity_results.append((tf_name, similarity))

    # Sort by similarity in descending order
    similarity_results.sort(key=lambda x: x[1], reverse=True)

    # Select top matches based on threshold
    top_tfs = [(similarity_results[0][0], similarity_results[0][1], description)]  # Always include the best match
    for i in range(1, len(similarity_results)):
        if similarity_results[i][1] >= similarity_results[0][1] - threshold:
            top_tfs.append((similarity_results[i][0], similarity_results[i][1], description))
        else:
            break

    return top_tfs

def send_command(command, host="127.0.0.1", port=65432):
    """
    Sends a single command string to the gui.py server.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(command.encode("utf-8"))
        print(f"Sent command: {command}")

def get_status(host="127.0.0.1", port=65432):
    """
    Sends 'get_status' command to the GUI and waits for the returned status JSON.
    Returns the parsed JSON as a Python dict, or None if failed.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(b"get_status")
        data = s.recv(4096).decode('utf-8')

    if "Current Status:" in data:
        json_str = data.split("Current Status:")[1].strip()
        try:
            status = json.loads(json_str)
            return status
        except json.JSONDecodeError:
            print("Failed to parse JSON status.")
            return None
    else:
        print("No status returned by GUI.")
        return None

def call_llm(conversation_history: list, client: OpenAI, system: str = None, model: str = "gpt-4o", response_format=None) -> str:
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an assistant specializing in understanding user requests and converting them into GUI control commands."},
            {"role": "system", "content": system}
        ] + list(conversation_history),
        "max_tokens": 500,
        "temperature": 0.1,
        "stream": False
    }
    if response_format:
        params["response_format"] = response_format
    response = client.chat.completions.create(**params)
    # In JSON mode, the content should be a valid serialized JSON string.
    return response.choices[0].message.content.strip()

def process_user_query(
    user_input: str, 
    conversation_history_parser: collections.deque,
    conversation_history_controller: collections.deque,
    step: int,
    iteration: int,
    client: OpenAI,
    clip_model,
    tf_embeddings: dict,
    model_name: str,
    source_dir: str,
    dataset_info: str,
    current_image: str  # base64-encoded current visualization image
):
    debug_text = f"Step {step}, Iteration {iteration}:\n"
    open_vocab_results = f"------------Iteration {iteration}------------\n"

    # 1) Record user input
    manage_conversation_history(conversation_history_parser, {"role": "user", "content": user_input})

    # 2) Get GUI status
    status = get_status()
    if not status:
        debug_text += "Failed to retrieve GUI status.\n"
        return ([], ["Sorry, I cannot connect to the GUI."], "NO", None, debug_text)

    debug_text += f"Current Status: {status}\n"

    # 3) Parse object descriptions using the LLM
    system_message_for_object_extraction = (
        "You are an assistant that extracts two kinds of descriptions from user input: a manipulation description and a stylization description, as well as a stylization prompt if applicable. "
        "Information about this dataset: " + dataset_info + "\n"
        f"Step {step}, Iteration{iteration}\n"
        "The manipulation description should capture the objects that the user intends to directly manipulate (e.g., changing opacity, color, asking to show the object or rotate to the best view, ). If the user does not intend to manipulate any object, return None. "
        "The stylization description should capture the objects the user intends to stylize. If the user intends to stylize the whole image, return the string \"whole\". Always return \"whole\" if the user does not specify any object to stylize. Otherwise, if there is no stylization, return None. "
        "If the user wants a guided tour of the dataset, you should put one object description into manipulation at an iteration until all objects are queried."
        "Additionally, if stylization is intended, provide a stylization prompt describing the desired style. If not, return None for the stylization prompt. "
        "Return your answer as a JSON object with keys: 'manipulation', 'stylization', and 'stylize_prompt'. "
        "For example:\n"
        " - If the user says: 'Change the color of the pencil to red', output: "
        "{\"manipulation\": [\"a pencil\"], \"stylization\": None, \"stylize_prompt\": None}.\n"
        " - If the user says: 'What is the yellow object', output: "
        "{\"manipulation\": [\"a yellow object\"], \"stylization\": None, \"stylize_prompt\": None}.\n"
        " - If the user says: 'show me the snake in an egg', output: "
        "{\"manipulation\": [\"an egg\", \"a snake\"], \"stylization\": None, \"stylize_prompt\": None}.\n"
        " - If the user says: 'Stylize the pencil in a cartoon style', output: "
        "{\"manipulation\": None, \"stylization\": [\"a pencil\"], \"stylize_prompt\": \"change it into cartoon style\"}.\n"
        " - If the user says: 'Stylize the whole scene and make it look like Van Goph's Starry night', output: "
        "{\"manipulation\": None, \"stylization\": \"whole\", \"stylize_prompt\": \"make it look like Van Goph's Starry night\"}."
        " - If the user says: 'I want to visualize the water container and the toothpaste', output: "
        "{\"manipulation\": [\"a water container\", \"a toothpaste\"], \"stylization\": None, \"stylize_prompt\": None}."
        " - If the user says: 'Change all the objects into green color', output: "
        "{\"manipulation\": None, \"stylization\": None, \"stylize_prompt\": None}."
        " - If the user says: 'Show me TF1' or 'Show me the object number 1' or 'Also show me another possible one', output: "
        "{\"manipulation\": None, \"stylization\": None, \"stylize_prompt\": None}."
        " - If the user says: 'Give me a guided tour of the dataset', the dataset has objects: pencil, desk, laptop, and currently it is iteration 0, output: "
        "{\"manipulation\": [\"a pencil\"], \"stylization\": None, \"stylize_prompt\": None}."
        " - If the user says: 'Give me a guided tour of the dataset', the dataset has objects: pencil, desk, laptop, and currently it is iteration 1, output: "
        "{\"manipulation\": [\"a desk\"], \"stylization\": None, \"stylize_prompt\": None}."
        "Respond with only the JSON object, do not explain."
    )

    extraction_response = call_llm(
        conversation_history=conversation_history_parser,
        client=client,
        system=system_message_for_object_extraction,
        model=model_name,
        response_format={"type": "json_object"}
    )
    manage_conversation_history(conversation_history_parser, {"role": "assistant", "content": extraction_response})
    debug_text += f"Extraction response: {extraction_response}\n"

    try:
        # Check if the response is already a dictionary (in case the client returns a parsed object)
        if isinstance(extraction_response, dict):
            extraction = extraction_response
        else:
            extraction = json.loads(extraction_response)
        manipulation_desc = extraction.get("manipulation")
        stylization_desc = extraction.get("stylization")
        stylize_prompt = extraction.get("stylize_prompt")
        debug_text += f"Extracted manipulation: {manipulation_desc}, stylization: {stylization_desc}, prompt: {stylize_prompt}\n"
        # Normalize values: if the string value is "none" (case-insensitive), set it to None.
        if manipulation_desc is None or str(manipulation_desc).lower() == "none":
            manipulation_desc = None       
        if stylization_desc is None or str(stylization_desc).lower() == "none":
            stylization_desc = None
        if stylize_prompt is None or str(stylize_prompt).lower() == "none":
            stylize_prompt = None
    except Exception as e:
        debug_text += "Failed to parse extraction response.\n"
        manipulation_desc, stylization_desc, stylize_prompt = None, None, None

    # Extract the best matching TFs based on the manipulation description
    if manipulation_desc is not None:
        best_tf = []

        debug_text += f"Extracted manipulation object descriptions: {manipulation_desc}\n"

        # 3. Embed each extracted object description and find matching TFs
        for description in manipulation_desc:
            description_embedding = embed_text(description, clip_model)

            # 4. Find the best matching TFs based on the embedding
            top_tfs = find_best_tfs(description_embedding, description, tf_embeddings, threshold=0.01)
            tf_to_best_frames = {}
            open_vocab_results += f"Query: {description}\n"
            for tf_name, similarity, description in top_tfs:
                best_frames = load_best_frames(source_dir, tf_name)
                tf_to_best_frames[tf_name] = best_frames
                open_vocab_results += f"{tf_name}, similarity: {similarity:.4f}\n"
            
            for tf_name, similarity, description in top_tfs:
                best_tf.append((description, tf_name, similarity))
    
    else:
        debug_text += "No object to manipulate.\n"
        best_tf = "no TF specified"

    # Extract the best matching TFs based on the stylization description
    if stylization_desc is not None and stylization_desc != "whole":
        best_stylization_tf = []

        debug_text += f"Extracted stylization object descriptions: {stylization_desc}\n"

        # 3. Embed each extracted object description and find matching TFs
        for description in stylization_desc:
            description_embedding = embed_text(description, clip_model)

            # 4. Find the best matching TFs based on the embedding
            top_tfs = find_best_tfs(description_embedding, description, tf_embeddings, threshold=0.01)
            try:
                tf_to_best_frames
            except NameError:
                tf_to_best_frames = {}
            open_vocab_results += f"Query: {description}\n"
            for tf_name, similarity, description in top_tfs:
                best_frames = load_best_frames(source_dir, tf_name)
                tf_to_best_frames[tf_name] = best_frames
                open_vocab_results += f"{tf_name}, similarity: {similarity:.4f}\n"
            
            for tf_name, similarity, description in top_tfs:
                best_stylization_tf.append((description, tf_name, similarity))
    
    elif stylize_prompt is not None:
        debug_text += f"Stylize the whole scene with {stylize_prompt}."
        best_stylization_tf = "no TF specified"

    else:
        debug_text += "No object to stylize.\n"
        best_stylization_tf = "no TF specified"

    # 4) Construct the system and user prompts for the LLM to generate commands
    system_message_for_commands = (
        "You are an assistant that converts user natural language requests into GUI control commands, meanwhile generate explanations to the user in natural language. Current visualization is also input as an image.\n"
        "Your answers should have three parts: Part1 is the commands that can be sent to the GUI, Part2 is a natural language explanation, final part is 'YES' or 'NO' to decide whether you want to process current user query for another iteration.\n"
        "Information about this dataset: " + dataset_info + "\n"
        "The dataset information has nothing to do with the label of the TFs.\n"

        "Part1: (Command). Available GUI commands are:\n"
        "- set_fov <value>: sets the field of view (1 to 120 degrees)\n"
        "- set_opacity <tf_index> <value>: sets the opacity factor of a given transfer function index\n"
        "- set_color <tf_index> <r> <g> <b>: sets the palette color of a TF in RGB (0-255)\n"
        "- set_light <param> <value>: sets a lighting parameter, param can be angle, elevation, ambient, diffuse, specular, shininess, headlight. And for param headlight, the value should be true or false, for all the other param, value should be a float number\n"
        "- set_mode <mode>: sets the rendering mode, mode can be phong, normal, diffuse_term, specular_term, ambient_term\n"
        "- set_background <r> <g> <b>: sets the background color\n"
        "- set_view <tf_index> <frame_number>: sets the view to a specific frame of a certain tf\n"
        "- legend add <label> <r> <g> <b>: adds a legend to the scene with the specified text label and color\n"
        "- legend delete <label>: deletes the legend with the specified text label\n"
        "- reset_view: resets the view to the default\n"
        "- reset_color_opacity: resets the color and opacity to the default\n"
        "- save_image: saves the current visualization to an image file\n"
        "- start_tour: a signal for staring a guided tour of a dataset\n"
        "- stylize <tf_index(s)> <prompt>: applies a text-driven stylization to the object corresponding to the given transfer function index. if there are multiple TFs to stylize, use '&' to connect these indices. e.g., stylize 1&2 \"make it cartoon\"; stylize 4 \"make it look like it just snowed\"\n"
        "- stylize whole <prompt>: applies a text-driven stylization to the entire scene\n\n"

        "Part2 (Explanation):\n"
        " - Provide a natural language explanation of what you did in Part1.\n"
        " - If the user has asked additional questions (e.g. 'What is the use of the pectoral fin of the carp?'), answer them here in natural language.\n"
        " - This part is for user-facing explanations and clarifications.\n\n"

        "Part3 (Iteration Decision):\n"
        "After Part1 and Part2, on a new line output either 'ITERATE: YES' or 'ITERATE: NO'.\n"
        "  - Output 'ITERATE: YES' if you believe that further refinement is necessary. For example, if the visualization still does not fully meet the user's request (e.g. user asks you to demostrate all objects in the dataset one by one) or if additional adjustments (such as changing colors, view angles, or lighting) could improve clarity.\n"
        "  - Output 'ITERATE: YES' if the user queried a specific object, but the open-vocabulary query results are not clear, or you changed the view and fov, as these modifications' effectiveness need to be further confirmed by checking the new visualization image\n"
        "  - Output 'ITERATE: NO' if you believe that your provided commands have fully addressed the user's request and no further refinement is needed.\n\n"

        "Your job is to figure out the necessary commands (if any) in Part1, and then provide explanations or answers to the user in Part2.\n"
        "Below are some examples of how you should structure your output.\n\n"

        "---------------------------------------------------------\n"
        "Example 1\n"
        "User request: \"Change the color of the pencil into red.\"\n"
        "For object manipulation, the user is referring: [('a pencil', 'TF00', 0.27)]\n"
        "The user just wants the color changed to red, so you might say in Part1:\n"
        "  Part1:\n"
        "  set_opacity 0 1\n"
        "  set_color 0 255 0 0\n"
        "  legend add \"pencil\" 255 0 0\n\n"
        "And then in Part2:\n"
        "  Part2:\n"
        "  \"I have changed the pencil (TF2) to red color, and set its opacity to 1.0 so that it's visible.\"\n"
        "  \"Let me know if you want any other modifications!\"\n"
        "  ITERATE: NO\n"
        "---------------------------------------------------------\n\n"

        "---------------------------------------------------------\n"
        "Example 2\n"
        "User request: \"Show me the swim bladder, and tell me how it works. \"\n"
        "For object manipulation, the user is referring: [('a swim bladder', 'TF00', 0.26), ('a swim bladder', 'TF03', 0.23)\n"
        "Best frames: {'TF00': [189, 186, 184, 192]}, freeze_view is false in current status\n"
        "You might produce:\n"
        "  Part1:\n"
        "  set_opacity 0 1\n"
        "  set_view 0 189\n\n"
        "  Part2:\n"
        "  \"I've set the swim bladder's opacity to 1.0 so it is visible, and moved the camera to frame 189 which provides one of the best views for that object.\"\n"
        "  \"Regarding your question about its function: the swim bladder is an internal gas-filled organ that helps fish control their buoyancy, so they can maintain or change depth in the water without expending much energy. It adjusts internal gas volume to balance overall density relative to the surrounding water.\"\n"
        "  ITERATE: YES\n"
        "---------------------------------------------------------\n\n"

        "---------------------------------------------------------\n"
        "Example 3\n"
        "User request: \"Make the scene brighter and more detailed, and reset everything else.\"\n"
        "No specific object reference.\n"
        "The user wants a brighter scene (decrease shininess or increase ambient, diffuse and specular) and more detailed (reduce FOV), and also to reset any color/opacity changes done so far.\n"
        "You might do:\n"
        "  Part1:\n"
        "  reset_view\n"
        "  reset_color_opacity\n"
        "  set_light shininess 2.0\n"
        "  set_light ambient 1.5\n"
        "  set_light diffuse 1.5\n"
        "  set_fov 40\n\n"
        "  Part2:\n"
        "  \"I've reset the camera view and all color/opacity settings.\"\n"
        "  \"To make the scene look brighter, the shininess is decreased to 2.0, and the ambient and diffuse are increased to 1.5\"\n"
        "  \"And I've set the field of view to 40 degrees so we can see details more closely.\"\n"
        "  ITERATE: YES\n"
        "---------------------------------------------------------\n\n"

        "---------------------------------------------------------\n"
        "Example 4:\n"
        "User request: 'Stylize the claws of the lobster in a cyborg style.'\n"
        "The stylization prompt is: make it cyborg.\n"
        "For stylization, the user is referring to [('claws of the lobster', 'TF02', 0.27), ('claws of the lobster', 'TF03', 0.265), ('claws of the lobster', 'TF07', 0.24)]\n"
        "Output:\n"
        "   Part1:\n"
        "   stylize 2&3 \"make it cyborg\"\n"
        "   Part2:\n"
        "   \"I have applied a cyborg stylization to the claws of the lobster (TF2 & TF3) as requested.\"\n"
        "  ITERATE: NO\n"
        "---------------------------------------------------------\n\n"

        "Remember:\n"
        "- Reset everything means to reset view, reset color/opacity, and set_fov 29.\n"
        "- For the mantle, supernova and hurricane datasets, do not trust the open query results, instead the mapping from description to TFxx is given in dataset info.\n"
        "- In your final output, always produce 'Part1:' followed by only the commands, then 'Part2:' followed by your explanations.\n"
        "- Do not insert extra text in Part1 besides the commands.\n"
        "- When you change the color of an object, make sure to add it to the legend. If you make it invisible, remove it from the legend.\n"
        "- 'reset everything' might mean calling reset_view and reset_color_opacity.\n"
        "- 'brighter visualization' can be achieved by using set_light to increase ambient, diffuse and specular, or to decrease shininess\n"
        "- If the user asks to change the direction of the lighting, remember to set headlight to false, and then change angle and elevation.\n"
        "- 'more details' might mean decreasing field_of_view.\n"
        "- do not change view if freeze_view is true in the status\n"
        "- do not make any change after applying stylization.\n"
        "- You do not know the relation between TFs and labels, so do not overconfident in adding legends. When users ask you to set a different color to each object, do not add legnds as you do not know!\n"
        "- When the user asks to only show an object, set other objects' opacity to 0, change to the best view of the target object, and meanwhile zoom in by decreasing field_of_view.\n"
        "- Whenever the user refers to objects, change objects' color or visualize anything, you should set the target objects' opacity to 1.0 so that the change is visible.\n"
        "- If the user wants the 'best view', call set_view <tf_index> <frame_number> with the best frame.\n"
        "- If the user only wants to visualize one object, change the camera to its best view and zoom in.\n"
        "- In a guided tour, always add new legend of an object each iteration!\n"
        "- If the user says \"undo previous action\", your commands might revert to the prior status. Then in Part2 you can say something like \"I've undone the last changes.\".\n"
        "- When the user only wants to stylize, do not change any object's opacity or color, just send the stylize command.\n"
        "- Legend added should be clear and concise, e.g. \"left claw\" instead of \"a left claw of a lobster\".\n"
        "- When the user wants a guided tour of the dataset, you should firstly set all objects' opacity to 1.0 at iteration 0. And at each iteration a different object in the dataset will be referred and give you the according TF, you should explain what this object is briefly and change its color to a different color for demo. Always use a new color for a new object, and do not stop iterating until all objects are shown.\n"
        "- When you starts a guided tour, at iteration 0 first send 'start_tour' command as a signal to the GUI, then start show the first object but do not change view.\n"
        "- Do not change objects' color in a guided tour for supernova, but remember to add legends.\n"
        "- Do not say things like 'let me know if you want to continue' during the guided tour, as it is not a conversation.\n"
        "- Here is a palette for you to choose color from: {'Red': [255, 0, 0], 'Green': [0, 255, 0], 'Blue': [0, 0, 255], 'Cyan': [0, 255, 255], 'Black': [0, 0, 0], 'Maroon': [128, 0, 0], 'Navy': [0, 0, 128], 'Teal': [0, 128, 128], 'Purple': [128, 0, 128], 'Lime': [0, 255, 0], 'Silver': [192, 192, 192], 'Brown': [165, 42, 42]}\n"
        "- Do not choose similar colors for different objects, e.g. do not use red and maroon for two different objects.\n"
        "- An example guide-mode part2 output: \"I have changed the pencil in blue, it is used for writing or drawing.\" Do not include things like setting opacity or view in guide words (although you should set them in the commands), keep it simple and naive.\n"
        "- Do not mix the commands with the explanations. Keep them strictly in Part1, while Part2 is free-form text.\n"
        "- When the user says show me object 1 in object 2 and keep both, you should lower object 2's opacity to 0.2 to make it transparent and keep object 1 visible.\n\n"
    
        "Now, you will receive:\n"
        "1) The current GUI status as a JSON object.\n"
        "2) The user's request.\n"
        "3) The identified transfer functions (TFs) and their best frames.\n\n"

        "Please use the following template in your output:\n"
        "Part1:\n"
        "<command1>\n"
        "<command2>\n"
        "... (more commands if needed)\n"
        "\n"
        "Part2:\n"
        "<text1>\n"
        "<text2>\n"
        "... (more natural language explanations for the user if needed)\n"
        "\n"
        "No other format is allowed.\n"
    )
    if stylize_prompt is None:
        if best_tf == "no TF specified":
            prompt_for_commands = (
                f"Step {step}, Iteration{iteration}\n\n"
                f"GUI status in step {step}: {json.dumps(status, indent=2)}\n\n"
                f"User request in step {step}: {user_input}\n\n"
                "For object manipulation, the user is not referring to any specific TF.\n\n"
                "For stylization, the user is not referring to any specific TF.\n\n"
                "Respond with commands (one per line) and explanations:"
            )
        else:
            prompt_for_commands = (
                f"Step {step}, Iteration{iteration}\n\n"
                f"GUI status in step {step}: {json.dumps(status, indent=2)}\n\n"
                f"User request in step {step}: {user_input}\n\n"
                f"For object manipulation, the user describes {manipulation_desc}, and it is referring {best_tf}.\n\n"
                f"The best frames (i.e. views, in descending order) for the TFs are: {tf_to_best_frames}\n\n"
                "For stylization, the user is not referring to any specific TF.\n\n"
                "Respond with commands (one per line) and explanations:"
            )
    else:
        if best_tf == "no TF specified":
            prompt_for_commands = (
                f"Step {step}, Iteration{iteration}\n\n"
                f"GUI status in step {step}: {json.dumps(status, indent=2)}\n\n"
                f"User request in step {step}: {user_input}\n\n"
                "For object manipulation, the user is not referring to any specific TF.\n\n"
                f"The stylization prompt is: {stylize_prompt}.\n\n"
                f"For stylization, the user is referring to {best_stylization_tf}.\n\n"
                "Respond with commands (one per line) and explanations:"
            )
        else:
            prompt_for_commands = (
                f"Step {step}, Iteration{iteration}\n\n"
                f"GUI status in step {step}: {json.dumps(status, indent=2)}\n\n"
                f"User request in step {step}: {user_input}\n\n"
                f"For object manipulation, the user describes {manipulation_desc}, and it is referring {best_tf}.\n\n"
                f"The best frames (i.e. views, in descending order) for the TFs are: {tf_to_best_frames}\n\n"
                f"The stylization prompt is: {stylize_prompt}.\n\n"
                f"For stylization, the user describes {stylization_desc}, and it is referring to {best_stylization_tf}.\n\n"
                "Respond with commands (one per line) and explanations:"
            )
    
    # conversation_history_controller.append({"role": "user", "content": prompt_for_commands})

    # NEW: Append a multi-part message that includes the image as a separate element.
    if model_name == "gpt-4o":
        manage_conversation_history(conversation_history_controller, {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_for_commands},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{current_image}"}}
            ]
        })
    else:
        # For models that do not support image inputs, just send the text prompt.
        manage_conversation_history(conversation_history_controller, {"role": "user", "content": prompt_for_commands})

    # 6. Call the LLM to generate commands
    llm_response = call_llm(conversation_history=conversation_history_controller, client=client, system=system_message_for_commands, model=model_name)

    manage_conversation_history(conversation_history_controller, {"role": "assistant", "content": llm_response})

    lines = llm_response.strip().split("\n")
    part1_commands = []
    part2_explanations = []
    iterate_decision = "NO"
    current_section = None
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.lower() == "part1:":
            current_section = "part1"
            continue
        elif line_stripped.lower() == "part2:":
            current_section = "part2"
            continue
        # Check if this line is the iterate decision.
        elif line_stripped.upper().startswith("ITERATE:"):
            iterate_decision = line_stripped.split("ITERATE:")[1].strip().upper()
            continue
        if not line_stripped:
            continue
        if current_section == "part1" and current_section !="```":
            part1_commands.append(line_stripped)
        elif current_section == "part2" and current_section !="```":
            cleaned_text = line_stripped.strip('"').strip()
            part2_explanations.append(cleaned_text)

    return (part1_commands, part2_explanations, iterate_decision, best_tf, debug_text, open_vocab_results, conversation_history_parser, conversation_history_controller)