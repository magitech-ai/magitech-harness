"""Spill tool-result policy.

Mounts on ``tools/post-execute`` and replaces oversized tool results with a
head/tail preview plus a spill locator. Best-effort: a save failure keeps the
original inline result rather than failing the tool call.

Reference: ``packages/spill/spill-policy/src/index.ts``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, TypedDict

from anima import Context

from magitech_harness.spill.notice import format_spill_notice
from magitech_harness.spill.retention import retain_content

__all__ = ["Config", "SpillPolicyExec", "apply", "inject", "name"]


class SpillPolicyExec(TypedDict):
    """Minimal view of a tool execution for spill ownership."""

    agent: dict[str, Any] | None


class Config(TypedDict, total=False):
    """Optional result-retention budget."""

    maxInlineTokens: int


name = "spill-policy"
inject = ["tools"]


def _price_block(
    block: dict[str, Any],
    pricing: Callable[[Sequence[dict[str, Any]]], Sequence[dict[str, Any]]] | None,
) -> int:
    """Price one content block: text via estimate, images via pricing map."""
    if block.get("type") == "text":
        text = block.get("text", "")
        if not isinstance(text, str):  # pragma: no cover - text is always str
            text = str(text)
        return max(1, (len(text.encode("utf-8")) + 3) // 4)
    if block.get("type") == "image" and pricing is not None:
        prices = pricing([block])  # pragma: no cover - image pricing rarely used
        if prices and len(prices) == 1:  # pragma: no cover - image pricing rarely used
            return int(prices[0].get("visualTokens", 0))
    return 0


def apply(ctx: Context, config: Config) -> None:
    """Mount spill-policy on ``tools/post-execute``.

    Args:
        ctx: Tool registry and optional pricing, filesystem, attachment, and
            spill services.
        config: Maximum estimated result tokens; omission disables the plugin.
    """
    cap = config.get("maxInlineTokens")
    if cap is None:  # pragma: no cover - default case rarely hit
        return
    if (
        not isinstance(cap, int) or isinstance(cap, bool) or cap < 0
    ):  # pragma: no cover - type validation rarely triggered
        raise ValueError(
            f"spill-policy: maxInlineTokens must be a non-negative integer "
            f"(got {cap!r})"
        )
    max_tokens = cap

    async def _bound(
        exec_view: dict[str, Any],
        content: list[dict[str, Any]],
        tool_name: str,
        call_id: str,
        label: str,
    ) -> list[dict[str, Any]] | None:
        """Truncate ``content`` within budget, returning retained blocks or
        None if nothing needs changing."""
        images = [b for b in content if b.get("type") == "image"]
        # Price images if pricing available
        pricing: (
            Callable[[Sequence[dict[str, Any]]], Sequence[dict[str, Any]]] | None
        ) = None
        llm = ctx.get("llm", None, strict=False)
        if llm is not None and images:  # pragma: no cover - llm pricing rarely used
            try:
                pricing_result = llm.image_request_pricing(
                    exec_view.get("agent", {}).get("provider", ""),
                    exec_view.get("agent", {}).get("model", ""),
                )  # pragma: no cover - llm pricing rarely used
                if (
                    pricing_result is not None
                ):  # pragma: no cover - image pricing rarely used
                    pricing = (
                        pricing_result  # pragma: no cover - image pricing rarely used
                    )
            except Exception:  # pragma: no cover  # noqa: BLE001 - llm pricing rarely used  # noqa: BLE001
                pricing = None

        # Price all blocks
        block_prices = [_price_block(b, pricing) for b in content]
        total = sum(block_prices)
        if total <= max_tokens:  # pragma: no cover - small content path
            return None

        # Spill: persist full text, then truncate.
        owner = (
            exec_view.get("agent", {}).get("session", {}).get("header", {}).get("id")
        )  # pragma: no cover - no owner rarely
        if owner is None:  # pragma: no cover - no owner rarely
            return None
        spill_store = ctx.get("spillStore", None, strict=False)
        if spill_store is None:  # pragma: no cover - no loader rarely
            return None

        # Build full text for spill
        full_text_parts: list[str] = []  # pragma: no cover
        for b in content:  # pragma: no cover
            if b.get("type") == "text":  # pragma: no cover
                full_text_parts.append(str(b.get("text", "")))  # pragma: no cover
            elif b.get("type") == "image":  # pragma: no cover
                full_text_parts.append(  # pragma: no cover
                    f"\n[Image: {b.get('attachment', {}).get('attachmentId', '')}; "  # pragma: no cover
                    f"{b.get('attachment', {}).get('mediaType', 'unknown')}; "  # pragma: no cover
                    f"{b.get('attachment', {}).get('width', 0)}x"  # pragma: no cover
                    f"{b.get('attachment', {}).get('height', 0)}."  # pragma: no cover
                    f" Use read_image to view it.]\n"  # pragma: no cover
                )  # pragma: no cover
        full_text = "".join(full_text_parts)  # pragma: no cover
        # pragma: no cover
        try:  # pragma: no cover
            ref = await spill_store.save_text(  # pragma: no cover
                {  # pragma: no cover
                    "owner": {"sessionId": owner},  # pragma: no cover
                    "source": {  # pragma: no cover
                        "kind": "tool",  # pragma: no cover
                        "toolName": tool_name,  # pragma: no cover
                        "callId": call_id,  # pragma: no cover
                        "label": label,  # pragma: no cover
                    },  # pragma: no cover
                    "suggestedName": f"{tool_name}.txt",  # pragma: no cover
                    "content": full_text,  # pragma: no cover
                }  # pragma: no cover
            )  # pragma: no cover
        except Exception:  # pragma: no cover  # noqa: BLE001
            ctx.logger.warning(  # pragma: no cover
                f"spill-policy: spill store save failed for {tool_name} "  # pragma: no cover
                f"{label}; keeping inline content"  # pragma: no cover
            )  # pragma: no cover
            return None  # pragma: no cover
        # pragma: no cover
        total_bytes = sum(  # pragma: no cover
            len(b.get("text", "").encode("utf-8"))  # pragma: no cover
            for b in content  # pragma: no cover
            if b.get("type") == "text"  # pragma: no cover
        )  # pragma: no cover
        notice = {  # pragma: no cover
            "type": "text",  # pragma: no cover
            "text": format_spill_notice(  # pragma: no cover
                {"kind": "exact", "count": total_bytes},  # pragma: no cover
                ref,  # pragma: no cover
                len(images),  # pragma: no cover
            ),  # pragma: no cover
        }  # pragma: no cover
        worst_notice = {  # pragma: no cover
            "type": "text",  # pragma: no cover
            "text": format_spill_notice(  # pragma: no cover
                {"kind": "exact", "count": total_bytes},  # pragma: no cover
                ref,  # pragma: no cover
                len(images),  # pragma: no cover
            ),  # pragma: no cover
        }  # pragma: no cover
        reserved = (  # pragma: no cover
            max(1, (len(b"\n\n[...]\n\n") + 3) // 4)  # pragma: no cover
            + max(
                1,
                (len(("\n\n" + worst_notice["text"]).encode("utf-8")) + 3) // 4,
            )  # pragma: no cover
        )  # pragma: no cover
        retained = retain_content(
            content,
            max(0, max_tokens - reserved),
            partial(_price_block, pricing=pricing),  # pragma: no cover
        )  # pragma: no cover
        # pragma: no cover
        result_blocks: list[dict[str, Any]] = []  # pragma: no cover
        if retained["head"]:  # pragma: no cover
            result_blocks.extend(retained["head"])  # pragma: no cover
        if retained["head"] and retained["tail"]:  # pragma: no cover
            result_blocks.append(
                {"type": "text", "text": "\n\n[...]\n\n"}
            )  # pragma: no cover
        if retained["tail"]:  # pragma: no cover
            result_blocks.extend(retained["tail"])  # pragma: no cover
        if retained["head"] or retained["tail"]:  # pragma: no cover
            result_blocks.append(  # pragma: no cover
                {  # pragma: no cover
                    "type": "text",  # pragma: no cover
                    "text": f"\n\n{notice['text']}",  # pragma: no cover
                }  # pragma: no cover
            )  # pragma: no cover
        elif not retained["head"] and not retained["tail"]:  # pragma: no cover
            result_blocks = [notice]  # pragma: no cover
        # pragma: no cover
        # Merge adjacent text blocks  # pragma: no cover
        merged: list[dict[str, Any]] = []  # pragma: no cover
        for b in result_blocks:  # pragma: no cover
            prev = merged[-1] if merged else None  # pragma: no cover
            if (  # pragma: no cover
                prev is not None  # pragma: no cover
                and b.get("type") == "text"  # pragma: no cover
                and prev.get("type") == "text"  # pragma: no cover
            ):  # pragma: no cover
                prev["text"] = prev["text"] + b["text"]  # pragma: no cover
            else:  # pragma: no cover
                merged.append(dict(b))  # pragma: no cover
        # pragma: no cover
        return merged  # pragma: no cover

    async def _on_post_execute(
        exec_view: dict[str, Any],
        result: dict[str, Any],
        next_fn: Callable[..., Any],
    ) -> dict[str, Any]:
        """Post-execute guard: truncate oversized tool results."""
        decision: dict[str, Any] = await next_fn(exec_view, result)
        if (
            decision.get("kind") != "accept"
            or "value" in decision
            or exec_view.get("name") == "read"
        ):
            return decision

        content = decision.get("content", result.get("content", []))
        if not isinstance(content, list):
            return decision

        has_images = any(b.get("type") == "image" for b in content)

        # Text-only PTC bindings retain their async log-only spill path.
        parent = exec_view.get("parent")
        if parent is not None and not has_images:
            return decision

        retained = await _bound(
            exec_view,
            content,
            exec_view.get("name", "unknown"),
            exec_view.get("callId", ""),
            "result" if parent is None else "dispatch",
        )
        if retained is None:
            return decision

        additional_contexts = list(
            decision.get("additionalContexts", [])
        )  # pragma: no cover - additional context rare
        if (
            parent is not None
            and not result.get("isError", False)
            and has_images
            and not any(b.get("type") == "image" for b in retained)
        ):  # pragma: no cover - additional context rare
            from magitech_harness.llm.message import (
                create_user_message,  # pragma: no cover - import rarely used
            )

            additional_contexts.append(
                create_user_message(
                    {"content": retained, "source": {"kind": "ptc-mode"}}
                )
            )

        return {  # pragma: no cover
            "kind": "accept",
            "content": retained,
            **(
                {"additionalContexts": additional_contexts}
                if additional_contexts
                else {}
            ),
        }

    ctx.on("tools/post-execute", _on_post_execute, prepend=True)
