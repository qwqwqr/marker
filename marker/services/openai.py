import json
import time
from typing import Annotated, List

import openai
import PIL
from marker.logger import get_logger
from openai import APITimeoutError, RateLimitError
from PIL import Image
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class OpenAIService(BaseService):
    openai_base_url: Annotated[
        str, "The base url to use for OpenAI-like models.  No trailing slash."
    ] = "https://api.openai.com/v1"
    openai_model: Annotated[str, "The model name to use for OpenAI-like model."] = (
        "gpt-4o-mini"
    )
    openai_api_key: Annotated[
        str, "The API key to use for the OpenAI-like service."
    ] = None
    openai_image_format: Annotated[
        str,
        "The image format to use for the OpenAI-like service. Use 'png' for better compatability",
    ] = "webp"

    def process_images(self, images: List[Image.Image]) -> List[dict]:
        """
        Generate the base-64 encoded message to send to an
        openAI-compatabile multimodal model.

        Args:
            images: Image or list of PIL images to include
            format: Format to use for the image; use "png" for better compatability.

        Returns:
            A list of OpenAI-compatbile multimodal messages containing the base64-encoded images.
        """
        if isinstance(images, Image.Image):
            images = [images]

        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/{};base64,{}".format(
                        self.openai_image_format, self.img_to_base64(img)
                    ),
                },
            }
            for img in images
        ]

    def __call__(
            self,
            prompt: str,
            image: PIL.Image.Image | List[PIL.Image.Image] | None,
            block: Block | None,
            response_schema: type[BaseModel],
            max_retries: int | None = None,
            timeout: int | None = None,
    ):
        if timeout is None:
            timeout = self.timeout

        client = self.get_client()
        image_data = self.format_image_for_llm(image)

        messages = [{
            "role": "user",
            "content": [*image_data, {"type": "text", "text": prompt}],
        }]

        tries = 0
        while True:
            tries += 1
            try:
                response = client.beta.chat.completions.parse(
                    extra_headers={
                        "X-Title": "Marker",
                        "HTTP-Referer": "https://github.com/datalab-to/marker",
                    },
                    model=self.openai_model,
                    messages=messages,
                    timeout=timeout,
                    response_format=response_schema,
                )
                response_text = response.choices[0].message.content
                total_tokens = response.usage.total_tokens
                if block:
                    block.update_metadata(llm_tokens_used=total_tokens, llm_request_count=1)
                return json.loads(response_text)

            except RateLimitError as e:
                reset_time = None
                is_token_limit = False
                remaining_tokens = 'unknown'
                limit_tokens = 'unknown'

                if hasattr(e, 'response') and e.response is not None:
                    headers = e.response.headers
                    logger.debug(f"RateLimit Headers: {dict(headers)}")

                    remaining_tokens = headers.get('x-ratelimit-remaining-tokens', 'unknown')
                    limit_tokens = headers.get('x-ratelimit-limit-tokens', 'unknown')
                    logger.info(f"[{self.openai_model}] Tokens: {remaining_tokens}/{limit_tokens}")

                    if headers.get('x-ratelimit-reset-tokens'):
                        try:
                            reset_header = headers['x-ratelimit-reset-tokens']
                            if 'h' in reset_header:
                                reset_time = int(reset_header.split('h')[0]) * 3600
                            elif 'm' in reset_header:
                                reset_time = int(reset_header.split('m')[0]) * 60
                            else:
                                reset_time = int(reset_header.replace('s', ''))
                            is_token_limit = True
                        except (ValueError, AttributeError):
                            reset_time = 360
                    else:
                        reset_time_str = headers.get('x-ratelimit-reset-requests') or \
                                         headers.get('x-ratelimit-reset') or \
                                         headers.get('retry-after')
                        try:
                            reset_time = int(float(reset_time_str)) if reset_time_str else None
                        except (TypeError, ValueError):
                            reset_time = None

                if reset_time:
                    try:
                        if is_token_limit:
                            sleep_time = reset_time
                            if remaining_tokens != 'unknown' and int(remaining_tokens) <= 0:
                                logger.warning(f"Token quota exhausted. Waiting {sleep_time}s...")
                        else:
                            current_time = time.time()
                            sleep_time = max(int(reset_time) - current_time, 1)

                        logger.warning(
                            f"Waiting {sleep_time:.1f}s for {'token' if is_token_limit else 'rate'} limit reset...")
                        time.sleep(sleep_time)
                        continue
                    except (ValueError, TypeError) as ve:
                        logger.warning(f"Error parsing reset time: {ve}")

                wait_time = min(tries * self.retry_wait_time, 360)
                logger.warning(f"Retrying in {wait_time}s... (Attempt {tries})")
                time.sleep(wait_time)

            except APITimeoutError as e:
                wait_time = min(tries * self.retry_wait_time, 360)
                logger.warning(f"Timeout error: {e}. Retrying in {wait_time}s... (Attempt {tries})")
                time.sleep(wait_time)

            except Exception as e:
                logger.error(f"Fatal error: {e}. Aborting after {tries} attempts.")
                break

        return {}

    def get_client(self) -> openai.OpenAI:
        return openai.OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
