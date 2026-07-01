# context4096.py
# A proper fix for the original codephreak model's 4096-token crash.
#
# The original model (llama2-7b-chat-codeCherryPop GGML) has a hard 4096-token
# context. The old chunk4096.py tried to cope by slicing the text into 4096-
# *character* pieces and concatenating independent generations — but:
#   • the limit is 4096 *tokens* (~12–16k chars), so it chopped far too early, and
#   • concatenating per-chunk answers with no shared context is incoherent.
#
# The real crash is that format_to_llama_chat_style() can emit a prompt longer
# than 4096 tokens once the system prompt + conversation history grow. The correct
# fix is a token-aware sliding window: always keep the system prompt and the most
# recent turns that fit within (n_ctx − reserve_for_response), dropping the oldest
# turns first. This never crashes and preserves the most relevant context.
#
# Drop-in usage in uiux.py / hfUIUX.py:
#     from context4096 import ContextWindow
#     window = ContextWindow(tokenizer, n_ctx=4096, reserve=512)
#     memory = window.fit(memory)          # trim BEFORE formatting/generation
#     prompt = format_to_llama_chat_style(memory)

from typing import Callable, List, Optional


class ContextWindow:
    """Token-aware conversation trimmer for a fixed-context model."""

    def __init__(self, tokenizer=None, n_ctx: int = 4096, reserve: int = 512,
                 count_tokens: Optional[Callable[[str], int]] = None):
        """
        tokenizer     : a HF tokenizer (uses tokenizer.encode) — preferred.
        n_ctx         : the model's hard context window (4096 for Llama-2).
        reserve       : tokens held back for the model's response.
        count_tokens  : optional custom token counter; overrides tokenizer.
        """
        self.n_ctx = n_ctx
        self.reserve = reserve
        self.budget = max(1, n_ctx - reserve)
        if count_tokens is not None:
            self._count = count_tokens
        elif tokenizer is not None:
            self._count = lambda s: len(tokenizer.encode(s))
        else:
            # Conservative fallback: ~4 characters per token.
            self._count = lambda s: max(1, len(s) // 4)

    def count(self, text: str) -> int:
        return self._count(text or "")

    def turn_tokens(self, turn) -> int:
        """Tokens for one [instruction, response] memory turn."""
        instr = turn[0] or ""
        resp = turn[1] if len(turn) > 1 and turn[1] else ""
        # +8 accounts for the [INST]/[/INST]/<s></s> wrapper tokens.
        return self.count(instr) + self.count(resp) + 8

    def fit(self, memory: List, system_prompt: str = "") -> List:
        """
        Return the largest suffix of `memory` (most-recent turns) that fits the
        budget, always keeping the final (newest) turn and reserving room for the
        system prompt. Never drops the last user turn — if even that plus the
        system prompt overflows, the system prompt is what a caller should shorten.
        """
        if not memory:
            return memory
        sys_tokens = self.count(system_prompt) if system_prompt else 0
        available = max(1, self.budget - sys_tokens)

        kept: List = [memory[-1]]                 # always keep the newest turn
        used = self.turn_tokens(memory[-1])
        for turn in reversed(memory[:-1]):        # add older turns while they fit
            t = self.turn_tokens(turn)
            if used + t > available:
                break
            kept.insert(0, turn)
            used += t
        return kept

    def would_overflow(self, memory: List, system_prompt: str = "") -> bool:
        sys_tokens = self.count(system_prompt) if system_prompt else 0
        total = sys_tokens + sum(self.turn_tokens(t) for t in memory)
        return total > self.budget

    def stats(self, memory: List, system_prompt: str = "") -> dict:
        sys_tokens = self.count(system_prompt) if system_prompt else 0
        used = sys_tokens + sum(self.turn_tokens(t) for t in memory)
        return {
            "n_ctx": self.n_ctx, "reserve": self.reserve, "budget": self.budget,
            "system_tokens": sys_tokens, "used_tokens": used,
            "fits": used <= self.budget, "turns": len(memory),
        }


if __name__ == "__main__":
    # Demonstration with the ~4-chars-per-token fallback (no model needed).
    win = ContextWindow(n_ctx=4096, reserve=512)
    long_turn = ["x" * 4000, "y" * 4000]          # ~2000 tokens per turn
    memory = [list(long_turn) for _ in range(6)] + [["latest question?", None]]
    print("before:", win.stats(memory, "system prompt"))
    fitted = win.fit(memory, "system prompt")
    print("after :", win.stats(fitted, "system prompt"))
    print(f"kept {len(fitted)}/{len(memory)} turns (newest first), no crash.")
