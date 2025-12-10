import base64
import io
import re
import os
from typing import List, Tuple, Optional

import json
import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw


def image2base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def build_messages(prompt: str, images: List[Image.Image]) -> List[dict]:
    """Builds the message structure for the GPT-4o API call."""
    content = [{"type": "text", "text": prompt}]
    for img in images:
        base64_image = image2base64(img)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high",
                },
            }
        )
    return [{"role": "user", "content": content}]


def call_gpt_api(
    prompt: str,
    images: List[Image.Image],
    model: str = "gpt-4o",
    api_key_path: str = "api_key.txt",
) -> Optional[str]:
    """Calls the OpenAI Chat Completions API and returns the message content."""
    try:
        with open(api_key_path, "r") as f:
            cfg=json.load(f)
        try:
            api_key = cfg["API_KEY"]
            base_url=cfg["API_BASE"]
        except KeyError:
            print(f"Error: 'API_KEY' or 'API_BASE' not found in {api_key_path}")
            return None
        client = OpenAI(api_key=api_key,base_url=base_url)

        messages = build_messages(prompt, images)

        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=50
        )
        return resp.choices[0].message.content
    except FileNotFoundError:
        print(f"Error: API key file not found at '{api_key_path}'")
        return None
    except Exception as e:
        print(f"Error calling GPT API: {e}")
        return None


class VisionBaseline:
    """
    A baseline model that uses a VLM to determine an operable point by
    loading prompts from external text files.
    """

    def __init__(
        self, prompts_dir: str, api_key_path: str = "api_key.json", model: str = "gpt-4o"
    ):
        if not os.path.isdir(prompts_dir):
            raise FileNotFoundError(f"Prompts directory not found at: {prompts_dir}")
        self.prompts_dir = prompts_dir
        self.api_key_path = api_key_path
        self.model = model
        
        # self._check_connectivity()
        if not self._check_connectivity():
            raise ConnectionError("Failed to connect to the OpenAI API. Check your API key and network connection.")

    def generate_prompt(self, task: str, object_name: str) -> str:
        try:
            with open(os.path.join(self.prompts_dir, "base_prompt.txt"), "r", encoding="utf-8") as f:
                base_prompt = f.read()
            with open(os.path.join(self.prompts_dir, f"{task}.txt"), "r") as f:
                task_specific_description = f.read()
        except FileNotFoundError as e:
            print(f"Error: Could not find a prompt file. {e}")
            raise
        task_specific_description = task_specific_description.format(
            object_name=object_name
        )
        full_prompt = base_prompt.format(
            object_name=object_name, task_specific_description=task_specific_description
        )
        return full_prompt

    def parse_point_from_response(self, response: str) -> Optional[Tuple[int, int]]:
        if not response:
            return None
        match = re.search(r"\((\s*(\d+)\s*),(\s*\d+\s*),(\s*\d+\s*)\)", response)
        if match:
            try:
                img_idx=int(match.group(1).strip())
                x = int(match.group(2).strip())
                y = int(match.group(3).strip())
                return (img_idx, x, y)
            except (ValueError, IndexError):
                return None
        else:
            return None

    
    
    def get_action_point(
        self, images: List[Image.Image], task: str, object_name: str,show=False,heuristic=False
    ) -> Optional[Tuple[int, int]]:
        if len(images) != 4:
            print(f"Error: Expected 4 images, but got {len(images)}.")
            return None
        try:
            prompt = self.generate_prompt(task, object_name)
        except FileNotFoundError:
            return None

        print("Calling Vision-Language Model API for action point...")
        if heuristic:
            list=["front", "back", "left", "right"]
            for i,img in enumerate(images):
                drawer = ImageDraw.Draw(img)
                img.show(title=list[i])
                
            # pyperclip.copy(prompt)
            print(f"\n\n\n{prompt}\n\n\n")
            i, x, y = map(int, input("请输入三个整数（图像索引 x y) , 用空格分隔：").split())
            print(f"Selected point: {i}, {x}, {y}")
            return (i, x, y)
            
            
            
                
        response_content = call_gpt_api(prompt, images, self.model, self.api_key_path)

        if response_content:
            point = self.parse_point_from_response(response_content)
            if point:
                print(f"Successfully parsed 2D action point: {point}")
                if show:
                    i,x,y=point
                    img=images[i]
                    drawer=ImageDraw.Draw(img)
                    r = max(3,min(img.size) // 100)
                    drawer.ellipse((x - r, y - r, x + r, y + r), outline="red", width=2)
                    img.show(title=f"{point}")
                return point

        print("Failed to get a valid response from the API.")
        return None

    def _check_connectivity(self)->bool:
        try:
            with open(self.api_key_path, "r") as f:
                cfg=json.load(f)
            
        except FileNotFoundError:
            print(f"API key file not found at: {self.api_key_path}")
            return False
        
        try:
            api_key = cfg["API_KEY"]
            base_url=cfg["API_BASE"]
        except KeyError:
            print(f"Error: 'API_KEY' or 'API_BASE' not found in {self.api_key_path}")
            return False
        try:
            client = OpenAI(api_key=api_key,base_url=base_url)
            _c=client.models.list()
            print(len(_c.data),"models found.")
        except Exception as e:
            print(f"Error connecting to OpenAI API: {e}")
            return False
        
        return True
        

if __name__ == "__main__":
    # Example usage
    prompts_dir = "/home/user/Dual-Assemble/workspace/GPT_Baseline/prompts"
    api_key_path = "/home/user/Dual-Assemble/workspace/GPT_Baseline/api_key.json"
    model = "qwen-vl-plus"

    baseline = VisionBaseline(prompts_dir, api_key_path, model)