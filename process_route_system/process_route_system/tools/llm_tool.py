"""
tools/llm_tool.py — General Tool (GPT-4o wrapper) shared by all four agents.

Provides a clean interface to the OpenAI Chat Completions API, supporting:
  - Single-turn completions (most agent calls)
  - Multi-turn conversation (agent self-reflection / iterative refinement)
  - JSON-mode responses (for structured outputs)

The paper uses GPT-4o as the "General Tool" for FEA, MPPA, SPPA, and POEA.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional, Union

import requests

logger = logging.getLogger(__name__)

OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"


class LLMTool:
    """
    Thin wrapper around the OpenAI Chat Completions API.
    """

    def __init__(self,
                 api_key: str,
                 model: str = "gpt-4o",
                 max_tokens: int = 4096,
                 temperature: float = 0.2,
                 max_retries: int = 3,
                 retry_delay: float = 5.0):
        self.api_key     = api_key
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }

    # ── Core call ──────────────────────────────────────────────────────────────

    def chat(self,
             messages: List[Dict[str, str]],
             json_mode: bool = False,
             temperature: Optional[float] = None) -> str:
        """
        Send a messages list to GPT-4o and return the assistant reply as a string.

        Args:
            messages   : OpenAI messages format [{"role": ..., "content": ...}]
            json_mode  : If True, instruct the model to return valid JSON.
            temperature: Override instance temperature for this call.

        Returns:
            Assistant message content string.
        """
        payload: Dict[str, Any] = {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(OPENAI_ENDPOINT, json=payload,
                                      headers=self._headers, timeout=120)
                resp.raise_for_status()
                data    = resp.json()
                content = data["choices"][0]["message"]["content"]
                logger.debug(f"LLM response ({len(content)} chars)")
                return content

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else "?"
                logger.warning(f"HTTP {status} on attempt {attempt}/{self.max_retries}")
                if status == 429:          # rate limit — back off
                    time.sleep(self.retry_delay * attempt)
                elif status >= 500:        # server error — retry
                    time.sleep(self.retry_delay)
                else:
                    raise                  # 4xx client error — don't retry
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error on attempt {attempt}: {e}")
                time.sleep(self.retry_delay)

        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts.")

    # ── Convenience helpers ────────────────────────────────────────────────────

    def complete(self,
                 system_prompt: str,
                 user_prompt:   str,
                 json_mode: bool = False) -> str:
        """Single-turn completion with system + user messages."""
        messages = [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ]
        return self.chat(messages, json_mode=json_mode)

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict:
        """
        Single-turn completion that guarantees a parsed JSON dict back.
        The system prompt should instruct the model to return only JSON.
        """
        raw = self.complete(system_prompt, user_prompt, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Strip markdown fences and retry
            cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
            return json.loads(cleaned)

    def refine(self,
               history: List[Dict[str, str]],
               feedback: str) -> str:
        """
        Append a user feedback message to the conversation and get a refined reply.
        Supports the paper's "self-reflection and iterative improvement" loop.
        """
        history.append({"role": "user", "content": feedback})
        reply = self.chat(history)
        history.append({"role": "assistant", "content": reply})
        return reply
