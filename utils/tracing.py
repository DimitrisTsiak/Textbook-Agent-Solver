"""
Langfuse tracing utilities with graceful fallback and complete credit-saving switch control.

Supports:
- Global/environment switch via LANGFUSE_ENABLED ("true"/"false", "1"/"0")
- Safe fallback when credentials are not configured or package is not installed
- Decorator @observe with support for observation types (as_type="retriever", "tool", "generation")
- Safe helpers for observation updates, scoring, flushing, and trace URL retrieval
"""

import os
import functools
import inspect
from typing import Any, Callable, Optional, Dict
from utils.env import load_env

# Preload environment variables so that @observe evaluates with credentials at import time
load_env()

# Global toggle for runtime switching
_RUNTIME_ENABLED: Optional[bool] = None


def is_tracing_enabled() -> bool:
    """
    Determines if Langfuse tracing is active.
    Must have:
      1. Runtime switch enabled (if explicitly set)
      2. LANGFUSE_ENABLED != 'false' / '0' in environment
      3. Valid LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY configured
    """
    global _RUNTIME_ENABLED
    if _RUNTIME_ENABLED is False:
        return False

    env_switch = os.environ.get("LANGFUSE_ENABLED", "true").strip().lower()
    if env_switch in ("false", "0", "no", "off", "disable", "disabled"):
        return False

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()

    if not public_key or not secret_key:
        return False
    if public_key.startswith("YOUR_") or secret_key.startswith("YOUR_"):
        return False

    return True


def enable_tracing() -> None:
    """Manually enables Langfuse tracing at runtime."""
    global _RUNTIME_ENABLED
    _RUNTIME_ENABLED = True


def disable_tracing() -> None:
    """Manually disables Langfuse tracing at runtime (saves credits/telemetry)."""
    global _RUNTIME_ENABLED
    _RUNTIME_ENABLED = False


# Lazy import Langfuse only when enabled
_LANGFUSE_AVAILABLE = False
_lf_observe = None

def _init_langfuse():
    global _LANGFUSE_AVAILABLE, _lf_observe
    if not is_tracing_enabled():
        return False
    if _LANGFUSE_AVAILABLE:
        return True

    try:
        from langfuse import observe as lf_observe
        _lf_observe = lf_observe
        _LANGFUSE_AVAILABLE = True
        return True
    except ImportError:
        _LANGFUSE_AVAILABLE = False
        return False


def observe(*args, **kwargs):
    """
    Langfuse @observe decorator with complete on/off credit switch.
    If tracing is disabled or langfuse is missing, acts as a transparent pass-through.
    """
    def decorator(fn: Callable) -> Callable:
        # Check if tracing is enabled and available
        if is_tracing_enabled() and _init_langfuse() and _lf_observe is not None:
            return _lf_observe(*args, **kwargs)(fn)

        # No-op pass-through wrapper
        if inspect.isgeneratorfunction(fn):
            @functools.wraps(fn)
            def gen_wrapper(*fn_args, **fn_kwargs):
                return fn(*fn_args, **fn_kwargs)
            return gen_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*fn_args, **fn_kwargs):
                return fn(*fn_args, **fn_kwargs)
            return sync_wrapper

    # Handle case where @observe is used without parentheses: @observe
    if len(args) == 1 and callable(args[0]) and not kwargs:
        target_fn = args[0]
        args = ()
        return decorator(target_fn)

    return decorator


def update_current_observation(
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> None:
    """Safely updates the active observation/trace if tracing is enabled."""
    if not is_tracing_enabled() or not _init_langfuse():
        return

    try:
        from langfuse import get_client
        client = get_client()
        update_args = {}
        if input is not None:
            update_args["input"] = input
        if output is not None:
            update_args["output"] = output
        if metadata is not None:
            update_args["metadata"] = metadata
        if tags is not None:
            update_args["tags"] = tags
        if session_id is not None:
            update_args["session_id"] = session_id
        if user_id is not None:
            update_args["user_id"] = user_id

        if update_args:
            client.update_current_span(**update_args)
    except Exception as e:
        print(f"[Tracing Warning] Failed to update current observation: {e}")


def score_trace(
    name: str,
    value: float,
    comment: Optional[str] = None,
    id: Optional[str] = None
) -> None:
    """Safely scores the active observation/trace if tracing is enabled."""
    if not is_tracing_enabled() or not _init_langfuse():
        return

    try:
        from langfuse import get_client
        client = get_client()
        score_kwargs = {"name": name, "value": value}
        if comment:
            score_kwargs["comment"] = comment
        client.score_current_span(**score_kwargs)
    except Exception as e:
        print(f"[Tracing Warning] Failed to score current observation: {e}")


def flush_traces() -> None:
    """
    Safely flushes all pending traces.
    Should be called before process exit in script-based executions.
    """
    if not is_tracing_enabled() or not _init_langfuse():
        return
        
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception as e:
        print(f"[Tracing Warning] Failed to flush traces: {e}")


def get_current_trace_url() -> Optional[str]:
    """
    Safely retrieves the URL of the current trace if tracing is enabled.
    Returns "Tracing Disabled" or "Unknown URL" on failure/disabled.
    """
    if not is_tracing_enabled() or not _init_langfuse():
        return "Tracing Disabled"
        
    try:
        from langfuse import get_client
        return get_client().get_trace_url()
    except Exception:
        return "Unknown URL"
