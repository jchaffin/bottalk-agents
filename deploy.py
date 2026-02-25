#!/usr/bin/env python3
"""
Deploy/update bottalk agent definition on Pipecat Cloud.

Usage:
  python deploy.py
  python deploy.py --local

With --local, this targets a separate PCC agent name:
  bottalk-agent-local
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import tomllib
from dotenv import load_dotenv


PCC_API = "https://api.pipecat.daily.co/v1"
DEFAULT_AGENT_NAME = "bottalk-agent"
LOCAL_AGENT_NAME = "bottalk-agent-local"


def load_env() -> None:
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env.local", override=False)
    load_dotenv(Path(__file__).resolve().parent / ".env.local", override=False)


def read_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def pcc_request(method: str, url: str, api_key: str, body: dict | None = None) -> tuple[int, str]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            payload = resp.read().decode("utf-8")
            return resp.status, payload
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8") if e.fp else ""
        return e.code, payload


def build_update_body(cfg: dict) -> dict:
    scaling = cfg.get("scaling", {}) or {}
    body = {
        "image": cfg["image"],
        "imagePullSecretSet": cfg["image_credentials"],
        "secretSet": cfg["secret_set"],
        "agentProfile": cfg["agent_profile"],
        "autoScaling": {
            "minAgents": int(scaling.get("min_agents", 0)),
            "maxAgents": int(scaling.get("max_agents", 10)),
        },
    }
    return body


def build_create_body(cfg: dict, service_name: str) -> dict:
    body = build_update_body(cfg)
    body["serviceName"] = service_name
    body["region"] = cfg["region"]
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy bottalk Pipecat Cloud agent")
    parser.add_argument("--local", action="store_true", help="Use local PCC agent name")
    parser.add_argument(
        "--agent-name",
        default="",
        help="Explicit Pipecat Cloud agent name override",
    )
    parser.add_argument(
        "--config",
        default="pcc-deploy.toml",
        help="Path to deploy TOML config (default: pcc-deploy.toml)",
    )
    args = parser.parse_args()

    load_env()

    api_key = os.getenv("PCC_PRIVATE_KEY")
    if not api_key:
        print("Missing PCC_PRIVATE_KEY in environment", file=sys.stderr)
        return 1

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 1

    cfg = read_config(cfg_path)
    agent_name = (
        args.agent_name
        or (LOCAL_AGENT_NAME if args.local else "")
        or os.getenv("PCC_AGENT_NAME", "")
        or str(cfg.get("agent_name", DEFAULT_AGENT_NAME))
    )

    update_body = build_update_body(cfg)
    update_url = f"{PCC_API}/agents/{agent_name}"
    code, payload = pcc_request("POST", update_url, api_key, update_body)

    if code == 404:
        create_body = build_create_body(cfg, agent_name)
        create_url = f"{PCC_API}/agents"
        code, payload = pcc_request("POST", create_url, api_key, create_body)

    print(payload)
    if code in (200, 201):
        print(f"Deployed {agent_name} ({code})")
        return 0

    print(f"Deploy failed for {agent_name} ({code})", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

