import os
import json
import numpy as np
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)
import concurrent.futures
import argparse
from tqdm import tqdm
import math
from io import BytesIO
from PIL import Image
import base64
import io
from openai import OpenAI
import requests
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="qwen", help="Model name for result save")
parser.add_argument("--api_key", type=str, default="EMPTY", help="API key")
parser.add_argument("--api_url", type=str, default="http://127.0.0.1:61131/v1", help="API URL")
parser.add_argument(
    "--save_path",
    type=str,
    help="Path to save the results",
)
parser.add_argument("--eval_model_name", type=str, default="qwen", help="Model name for evaluation")
parser.add_argument("--test_dataset_dir", type=str, help="path to your test dataset")
parser.add_argument("--num_workers", type=int, default=8)
args = parser.parse_args()


openai_api_key = args.api_key
openai_api_base = args.api_url

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
if args.eval_model_name is None:
    response = requests.get(f"{openai_api_base}/models")
    models = response.json()
    eval_model_name = models["data"][0]["id"]
else:
    eval_model_name = args.eval_model_name

# vstar_bench_path = args.vstar_bench_path
save_path = args.save_path
save_path = os.path.join(save_path, args.model_name)
os.makedirs(save_path, exist_ok=True)
abc_map = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F"}

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# the following code is copied from qwen-vl-utils
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)  # type: ignore
        w_bar = floor_by_factor(width / beta, factor)  # type: ignore
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)  # type: ignore
        w_bar = ceil_by_factor(width * beta, factor)  # type: ignore
    return h_bar, w_bar


def process(img_arg: tuple[dict, str, bool]):
    json_data, test_path, if_HD = img_arg
    img_path = os.path.join(test_path, json_data["image_id"])

    if if_HD:
        double_question_prompt = f"Here are captions about the image inside <captions>...</captions> which may have some hallucinated informations :\n<captions>{json_data['system1_question']}\n{json_data['system1_answer']}</captions>\nIf you think there is NO HALLUCINATION, think twice, there is impossible for no hallucination. Remember ONLY ANSWER THE OPTION. Now, you need to answer the question below.\nQuestion: {json_data['system2_question']}"
    else:
        double_question_prompt = f"Here are captions about the image inside <captions>...</captions> which need you to judge the claim is factual :\n<captions>{json_data['context']}</captions>\nRemember ONLY ANSWER THE OPTION. Now, you need to check the claim below.\nClaim: {json_data['claim']}\n{json_data['question']}"

    options = f"{json_data['choices'][0]['id']}: {json_data['choices'][0]['choice']}\n{json_data['choices'][1]['id']}: {json_data['choices'][1]['choice']}\n{json_data['choices'][2]['id']}: {json_data['choices'][2]['choice']}\n{json_data['choices'][3]['id']}: {json_data['choices'][3]['choice']}\n"

    instruction_prompt_system = """You are a helpful assistant.

    # Tools
    You may call one or more functions to assist with the user query.
    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
    </tools>

    # How to call a tool
    Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call>

    **Example**:
    <tool_call>
    {"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}
    </tool_call>"""
    USER_PROMPT_V2 = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "

    instruction_prompt_before = (
        """{question}
    Options: {options}
    """
        + USER_PROMPT_V2
    )

    user_prompt = USER_PROMPT_V2

    start_token = "<tool_call>"
    end_token = "</tool_call>"

    prompt = instruction_prompt_before.format(question=double_question_prompt, options=options)
    pil_img = Image.open(img_path)
    base64_image = encode_image_to_base64(img_path)

    messages = [
        {
            "role": "system",
            "content": instruction_prompt_system,
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    print_messages = [
        {
            "role": "system",
            "content": instruction_prompt_system,
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpg;base64,"}},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    chat_message = messages

    response_message = ""

    status = "success"
    try_count = 0
    turn_idx = 0
    try:
        while "</answer>" not in response_message:
            if "</answer>" in response_message and "<answer>" in response_message:
                break

            if try_count > 10:
                break

            params = {
                "model": eval_model_name,
                "messages": chat_message,
                "temperature": 0.0,
                "max_tokens": 10240,
                # "stop": ["<|im_end|>\n".strip()],
                # "stop": ["<|im_end|>\n".strip(), "</tool_call>"],
                "stop": ["<|im_end|>\n".strip(), "\n\n\n\n\n\n"],
            }
            response = client.chat.completions.create(**params)
            response_message = response.choices[0].message.content

            if start_token in response_message:
                action_list = response_message.split(start_token)[1].split(end_token)[0].strip()
                action_list = eval(action_list)

                bbox_list = []
                cropped_pil_image_content_list = []

                bbox_str = action_list["arguments"]["bbox_2d"]
                bbox = bbox_str
                left, top, right, bottom = bbox
                cropped_image = pil_img.crop((left, top, right, bottom))
                new_w, new_h = smart_resize((right - left), (bottom - top), factor=IMAGE_FACTOR)
                cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)  # type: ignore
                cropped_pil_image = encode_pil_image_to_base64(cropped_image)
                bbox_list.append(bbox)
                cropped_pil_image_content = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{cropped_pil_image}"},
                }
                cropped_pil_image_content_list.append(cropped_pil_image_content)

                if len(bbox_list) == 1:
                    bbox_list = bbox_list[0]
                user_msg = user_prompt

                content_f = []
                content_f.append({"type": "text", "text": "<tool_response>"})
                for cropped_pil_image_content in cropped_pil_image_content_list:
                    content_f.append(cropped_pil_image_content)
                content_f.append({"type": "text", "text": user_msg})
                content_f.append({"type": "text", "text": "</tool_response>"})

                _message = [
                    {
                        "role": "assistant",
                        "content": response_message,
                    },
                    {
                        "role": "user",
                        "content": content_f,
                    },
                ]

                chat_message.extend(_message)

                p_message = [
                    {
                        "role": "assistant",
                        "content": response_message,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,"}},
                            {"type": "text", "text": user_msg},
                        ],
                    },
                ]
                print_messages.extend(p_message)
                turn_idx += 1
            else:
                p_message = [
                    {
                        "role": "assistant",
                        "content": response_message,
                    }
                ]
                print_messages.extend(p_message)

            try_count += 1
    except Exception as e:
        print(f"Error!!!!", e)
        status = "error"

    if "</answer>" in response_message and "<answer>" in response_message:
        output_text = response_message.split("<answer>")[1].split("</answer>")[0].strip()
    else:
        output_text = response_message

    save_info = {}
    save_info["image"] = json_data["image_id"]
    if if_HD:
        save_info["question"] = json_data["system2_question"]
    else:
        save_info["question"] = json_data["claim"]
    save_info["answer"] = json_data["correct_answer"]
    save_info["choice"] = json_data["correct_choice"]
    save_info["pred_ans"] = output_text
    save_info["pred_output"] = print_messages
    save_info["status"] = status
    return save_info


def process_normal_qwen(img_arg: tuple[dict, str, bool]):
    json_data, test_path, if_HD = img_arg
    img_path = os.path.join(test_path, json_data["image_id"])

    if if_HD:
        double_question_prompt = f"Here are captions about the image inside <captions>...</captions> which may have some hallucinated informations :\n<captions>{json_data['system1_question']}\n{json_data['system1_answer']}</captions>\nIf you think there is NO HALLUCINATION, think twice, there is impossible for no hallucination. Remember ONLY ANSWER THE OPTION in <answer></answer>, like <answer>A</answer>, if your answer is option A. Now, you need to answer the question below.\nQuestion: {json_data['system2_question']}"
    else:
        double_question_prompt = f"Here are captions about the image inside <captions>...</captions> which need you to judge the claim is factual :\n<captions>{json_data['context']}</captions>\nRemember ONLY ANSWER THE OPTION in <answer></answer>, like <answer>A</answer>, if your answer is option A. Now, you need to check the claim below.\nClaim: {json_data['claim']}\n{json_data['question']}."

    options = f"{json_data['choices'][0]['id']}: {json_data['choices'][0]['choice']}\n{json_data['choices'][1]['id']}: {json_data['choices'][1]['choice']}\n{json_data['choices'][2]['id']}: {json_data['choices'][2]['choice']}\n{json_data['choices'][3]['id']}: {json_data['choices'][3]['choice']}\n"

    instruction_prompt_system = """You are a helpful assistant."""
    USER_PROMPT_V2 = "\nThink first. Format strictly as:  <think>...</think>  <answer>...</answer> "

    instruction_prompt_before = (
        """{question}
    Options: {options}
    """
        + USER_PROMPT_V2
    )

    user_prompt = USER_PROMPT_V2

    start_token = "<tool_call>"
    end_token = "</tool_call>"

    prompt = instruction_prompt_before.format(question=double_question_prompt, options=options)
    pil_img = Image.open(img_path)
    base64_image = encode_image_to_base64(img_path)

    messages = [
        {
            "role": "system",
            "content": instruction_prompt_system,  # NOTE
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    print_messages = [
        {
            "role": "system",
            "content": instruction_prompt_system,  # NOTE
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpg;base64,"}},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    chat_message = messages

    response_message = ""

    status = "success"
    try_count = 0
    turn_idx = 0
    try:
        while "</answer>" not in response_message:
            if "</answer>" in response_message and "<answer>" in response_message:
                break

            if try_count > 10:
                break

            params = {
                "model": eval_model_name,
                "messages": chat_message,
                "temperature": 0.0,
                "max_tokens": 10240,
                # "stop": ["<|im_end|>\n".strip()],
                # "stop": ["<|im_end|>\n".strip(), "</tool_call>"],
                "stop": ["<|im_end|>\n".strip(), "\n\n\n\n\n\n"],
            }
            response = client.chat.completions.create(**params)
            response_message = response.choices[0].message.content

            if start_token in response_message:
                action_list = response_message.split(start_token)[1].split(end_token)[0].strip()
                action_list = eval(action_list)

                bbox_list = []
                cropped_pil_image_content_list = []

                bbox_str = action_list["arguments"]["bbox_2d"]
                bbox = bbox_str
                left, top, right, bottom = bbox
                cropped_image = pil_img.crop((left, top, right, bottom))
                new_w, new_h = smart_resize((right - left), (bottom - top), factor=IMAGE_FACTOR)
                cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)  # type: ignore
                cropped_pil_image = encode_pil_image_to_base64(cropped_image)
                bbox_list.append(bbox)
                cropped_pil_image_content = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{cropped_pil_image}"},
                }
                cropped_pil_image_content_list.append(cropped_pil_image_content)

                if len(bbox_list) == 1:
                    bbox_list = bbox_list[0]
                user_msg = user_prompt

                content_f = []
                content_f.append({"type": "text", "text": "<tool_response>"})
                for cropped_pil_image_content in cropped_pil_image_content_list:
                    content_f.append(cropped_pil_image_content)
                content_f.append({"type": "text", "text": user_msg})
                content_f.append({"type": "text", "text": "</tool_response>"})

                _message = [
                    {
                        "role": "assistant",
                        "content": response_message,
                    },
                    {
                        "role": "user",
                        "content": content_f,
                    },
                ]

                chat_message.extend(_message)

                p_message = [
                    {
                        "role": "assistant",
                        "content": response_message,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,"}},
                            {"type": "text", "text": user_msg},
                        ],
                    },
                ]
                print_messages.extend(p_message)
                turn_idx += 1
            else:
                p_message = [
                    {
                        "role": "assistant",
                        "content": response_message,
                    }
                ]
                print_messages.extend(p_message)

            try_count += 1
    except Exception as e:
        print(f"Error!!!!", e)
        status = "error"

    if "</answer>" in response_message and "<answer>" in response_message:
        output_text = response_message.split("<answer>")[1].split("</answer>")[0].strip()
    else:
        output_text = response_message

    save_info = {}
    save_info["image"] = json_data["image_id"]
    if if_HD:
        save_info["question"] = json_data["system2_question"]
    else:
        save_info["question"] = json_data["claim"]
    save_info["answer"] = json_data["correct_answer"]
    save_info["choice"] = json_data["correct_choice"]
    save_info["pred_ans"] = output_text
    save_info["pred_output"] = print_messages
    save_info["status"] = status
    return save_info


def save_results_on_exit(signum, frame):
    """
    Signal handler to save results when Ctrl+C is pressed.
    """
    global save_json, output_file_path
    if save_json:
        try:
            with open(output_file_path, "w") as f:
                for item in save_json:
                    f.write(json.dumps(item) + "\n")
            print(f"Successfully saved {len(save_json)} items to: {output_file_path}")
        except Exception as e:
            print(f"Error while saving results: {e}")
    else:
        print("No data processed yet to save.")
    sys.exit(0)


if __name__ == "__main__":

    dataset_dir = args.test_dataset_dir
    # dataset_name = "FactChecking"   # HallucinationDetection
    dataset_names = ["FactChecking", "HallucinationDetection"]
    for dataset_name in dataset_names:
        json_path = os.path.join(dataset_dir, f"{dataset_name}", "json", "test.json")
        image_path = os.path.join(dataset_dir, f"{dataset_name}", "images")
        with open(json_path, "r") as f:
            json_datas = json.load(f)

            save_json = []
            save_name = f"result_{dataset_name}_{args.model_name}.jsonl"
            output_file_path = os.path.join(save_path, save_name)

            # for i in tqdm(range(len(json_datas)), desc="Processing JSON data"):
            #     result = process((json_datas[i], image_path))
            #     save_json.append(result)
            #     # print(save_json)
            #     # print(f"finish {i}")
            #     # exit()

            task_args = [(data_item, image_path, dataset_name == "HallucinationDetection") for data_item in json_datas]

            # multi threads to speed up
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                with tqdm(total=len(task_args), desc=f"Processing JSON data with path: {json_path}") as pbar:
                    # for result in executor.map(process, task_args):
                    for result in executor.map(process_normal_qwen, task_args):
                        if result is not None:
                            save_json.append(result)
                        pbar.update(1)

            with open(output_file_path, "w") as f:
                for item in save_json:
                    f.write(json.dumps(item) + "\n")
