import os
import re
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastmcp import FastMCP

mcp = FastMCP("openapi_tester")

RUNTIME_TOKEN: str = ""

# In-memory cache
SPEC: Dict[str, Any] = {}
SPEC_URL: str = os.getenv("OPENAPI_URL", "http://localhost:8080/v3/api-docs")
BASE_URL: str = os.getenv("BASE_URL", "http://localhost:8080")
AUTH_HEADER: str = os.getenv("AUTH_HEADER", "Authorization")
AUTH_TOKEN: str = os.getenv("AUTH_TOKEN", "")
DEFAULT_TIMEOUT_S: float = float(os.getenv("TIMEOUT_S", "15"))

def _hdrs(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    h = {"Accept": "application/json"}
    token = RUNTIME_TOKEN or AUTH_TOKEN
    if token:
        h[AUTH_HEADER] = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    if extra:
        h.update(extra)
    return h

def _get_by_ref(root: Dict[str, Any], ref: str) -> Dict[str, Any]:
    if not ref.startswith("#/"):
        return {}
    cur: Any = root
    for part in ref[2:].split("/"):
        cur = cur.get(part, {})
    return cur if isinstance(cur, dict) else {}

def _resolve_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    if "\$ref" in schema:
        return _resolve_schema(_get_by_ref(SPEC, schema["\$ref"]))
    if "allOf" in schema and isinstance(schema["allOf"], list):
        merged: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        for s in schema["allOf"]:
            rs = _resolve_schema(s)
            if rs.get("properties"):
                merged["properties"].update(rs["properties"])
            if rs.get("required"):
                for r in rs["required"]:
                    if r not in merged["required"]:
                        merged["required"].append(r)
        return merged
    return schema

def _example_for_schema(schema: Dict[str, Any], depth: int = 0) -> Any:
    if depth > 4:
        return None
    s = _resolve_schema(schema)
    if "example" in s:
        return s["example"]
    if "enum" in s and isinstance(s["enum"], list) and s["enum"]:
        return s["enum"][0]

    t = s.get("type")
    fmt = s.get("format", "")

    if t == "string":
        if fmt == "uuid":
            return "00000000-0000-0000-0000-000000000000"
        if fmt in ("date-time", "datetime"):
            return "2020-01-01T00:00:00Z"
        if fmt == "date":
            return "2020-01-01"
        return "test"
    if t == "integer":
        return 1
    if t == "number":
        return 1.0
    if t == "boolean":
        return True
    if t == "array":
        item = _example_for_schema(s.get("items", {}), depth + 1)
        return [] if item is None else [item]

    props = s.get("properties", {}) if isinstance(s.get("properties", {}), dict) else {}
    req = s.get("required", []) if isinstance(s.get("required", []), list) else []
    obj: Dict[str, Any] = {}
    for k in req:
        if k in props:
            obj[k] = _example_for_schema(props[k], depth + 1)
    for k, v in list(props.items())[:3]:
        if k not in obj:
            obj[k] = _example_for_schema(v, depth + 1)
    return obj

def _iter_operations(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    paths = spec.get("paths", {}) or {}
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method.lower() not in {"get","post","put","patch","delete","head","options"}:
                continue
            if not isinstance(op, dict):
                continue
            out.append({
                "method": method.upper(),
                "path": path,
                "operationId": op.get("operationId"),
                "tags": op.get("tags", []),
                "summary": op.get("summary") or op.get("description") or "",
                "parameters": op.get("parameters", []),
                "requestBody": op.get("requestBody"),
            })
    return out

def _apply_fixtures(value: str, fixtures: Dict[str, str]) -> str:
    if value in fixtures:
        return fixtures[value]
    for k, v in fixtures.items():
        try:
            if k.startswith("re:") and re.search(k[3:], value):
                return v
        except re.error:
            pass
    return value

def _build_request(op: Dict[str, Any], base_url: str, fixtures: Dict[str, str]) -> Tuple[str, Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    path = op["path"]

    def repl(m):
        name = m.group(1)
        return _apply_fixtures(name, fixtures) if name in fixtures else "1"
    resolved_path = re.sub(r"\{([^}]+)\}", repl, path)

    url = base_url.rstrip("/") + resolved_path

    query: Dict[str, Any] = {}
    for p in op.get("parameters", []) or []:
        if not isinstance(p, dict):
            continue
        if "\$ref" in p:
            p = _get_by_ref(SPEC, p["\$ref"]) or p
        if p.get("in") != "query":
            continue
        name = p.get("name")
        required = bool(p.get("required"))
        schema = p.get("schema", {})
        if required and name:
            query[name] = fixtures.get(name) or _example_for_schema(schema)

    body = None
    rb = op.get("requestBody")
    if isinstance(rb, dict):
        if "\$ref" in rb:
            rb = _get_by_ref(SPEC, rb["\$ref"]) or rb
        content = rb.get("content", {}) or {}
        app_json = content.get("application/json") or content.get("application/*+json")
        if isinstance(app_json, dict):
            schema = app_json.get("schema", {}) or {}
            body = _example_for_schema(schema)

    headers = _hdrs({"Content-Type": "application/json"})
    return url, query, headers, body

async def _load_openapi_impl(spec_url: Optional[str] = None) -> Dict[str, Any]:
    global SPEC, SPEC_URL
    if spec_url:
        SPEC_URL = spec_url

    async with httpx.AsyncClient() as client:
        r = await client.get(SPEC_URL, headers=_hdrs(), timeout=DEFAULT_TIMEOUT_S)
        r.raise_for_status()
        SPEC = r.json()

    ops = _iter_operations(SPEC)
    tag_counts: Dict[str, int] = {}
    for op in ops:
        for t in op.get("tags", []) or ["(untagged)"]:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    return {
        "spec_url": SPEC_URL,
        "base_url_default": BASE_URL,
        "endpoints_total": len(ops),
        "tags": dict(sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))),
    }

async def _select_endpoints_impl(
    tags: Optional[List[str]] = None,
    path_prefixes: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    limit: int = 5000,
) -> Dict[str, Any]:
    if not SPEC:
        await _load_openapi_impl()

    tags = tags or []
    path_prefixes = path_prefixes or []
    methods = [m.upper() for m in (methods or [])]

    selected = []
    for op in _iter_operations(SPEC):
        if methods and op["method"] not in methods:
            continue
        if tags:
            op_tags = set(op.get("tags", []) or [])
            if not op_tags.intersection(tags):
                continue
        if path_prefixes:
            if not any(op["path"].startswith(p) for p in path_prefixes):
                continue
        selected.append({
            "method": op["method"],
            "path": op["path"],
            "operationId": op.get("operationId"),
            "tags": op.get("tags", []),
            "summary": op.get("summary", ""),
        })
        if len(selected) >= limit:
            break

    return {"count": len(selected), "endpoints": selected}

async def _run_one(
    client: httpx.AsyncClient,
    op: Dict[str, Any],
    base_url: str,
    fixtures: Dict[str, str],
    timeout_s: float,
    ok_status: List[int],
) -> Dict[str, Any]:
    url, query, headers, body = _build_request(op, base_url, fixtures)
    method = op["method"]
    t0 = asyncio.get_event_loop().time()
    try:
        resp = await client.request(method, url, params=query, headers=headers, json=body, timeout=timeout_s)
        ms = int((asyncio.get_event_loop().time() - t0) * 1000)
        ok = (resp.status_code in ok_status) if ok_status else (200 <= resp.status_code < 300)
        return {
            "ok": ok,
            "status": resp.status_code,
            "ms": ms,
            "method": method,
            "path": op["path"],
            "url": str(resp.request.url),
            "operationId": op.get("operationId"),
        }
    except Exception as e:
        ms = int((asyncio.get_event_loop().time() - t0) * 1000)
        return {
            "ok": False,
            "status": None,
            "ms": ms,
            "method": method,
            "path": op["path"],
            "url": url,
            "operationId": op.get("operationId"),
            "error": str(e),
        }

# ---------- MCP TOOLS (wrappers) ----------

@mcp.tool
async def load_openapi(spec_url: str) -> Dict[str, Any]:
    return await _load_openapi_impl(spec_url)

@mcp.tool
async def select_endpoints(
    tags: Optional[List[str]] = None,
    path_prefixes: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    limit: int = 5000,
) -> Dict[str, Any]:
    return await _select_endpoints_impl(tags=tags, path_prefixes=path_prefixes, methods=methods, limit=limit)

@mcp.tool
async def ping() -> Dict[str, Any]:
    """Check connectivity to the BASE_URL and SPEC_URL."""
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(BASE_URL, timeout=5)
            base_ok = True
            base_status = r.status_code
        except Exception as e:
            base_ok = False
            base_status = str(e)
        
        try:
            r = await client.get(SPEC_URL, timeout=5)
            spec_ok = True
            spec_status = r.status_code
        except Exception as e:
            spec_ok = False
            spec_status = str(e)
            
    return {
        "BASE_URL": BASE_URL,
        "base_ok": base_ok,
        "base_status": base_status,
        "SPEC_URL": SPEC_URL,
        "spec_ok": spec_ok,
        "spec_status": spec_status,
        "RUNTIME_TOKEN_SET": bool(RUNTIME_TOKEN)
    }

@mcp.tool
async def test_endpoints(
    tags: Optional[List[str]] = None,
    path_prefixes: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    base_url: Optional[str] = None,
    concurrency: int = 10,
    timeout_s: float = 15,
    ok_status: Optional[List[int]] = None,
    stop_after_failures: int = 50,
    fixtures_json: Optional[str] = None,
    mode: str = "read-only",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    global RUNTIME_TOKEN
    if token:
        RUNTIME_TOKEN = token

    if not SPEC:
        await _load_openapi_impl()

    base_url = base_url or BASE_URL
    fixtures: Dict[str, str] = {}
    if fixtures_json:
        try:
            fixtures = json.loads(fixtures_json)
        except Exception:
            fixtures = {}

    if mode.lower() == "read-only":
        methods = ["GET", "HEAD", "OPTIONS"] if not methods else [m.upper() for m in methods if m.upper() in {"GET","HEAD","OPTIONS"}]

    selection = (await _select_endpoints_impl(tags=tags, path_prefixes=path_prefixes, methods=methods))["endpoints"]
    op_index = {(op["method"], op["path"]): op for op in _iter_operations(SPEC)}
    ops = [op_index[(e["method"], e["path"])] for e in selection if (e["method"], e["path"]) in op_index]

    sem = asyncio.Semaphore(max(1, concurrency))
    results: List[Dict[str, Any]] = []
    failures = 0

    async with httpx.AsyncClient() as client:
        async def run_guarded(op):
            nonlocal failures
            async with sem:
                r = await _run_one(client, op, base_url, fixtures, timeout_s, ok_status or [])
                results.append(r)
                if not r["ok"]:
                    failures += 1

        tasks = []
        for op in ops:
            if failures >= stop_after_failures:
                break
            tasks.append(asyncio.create_task(run_guarded(op)))

        if tasks:
            await asyncio.gather(*tasks)

    passed = sum(1 for r in results if r["ok"])
    failed = len(results) - passed
    failure_list = [r for r in results if not r["ok"]][:50]

    return {
        "base_url": base_url,
        "tested": len(results),
        "passed": passed,
        "failed": failed,
        "failures_sample": failure_list,
        "note": "If many 400/404, provide fixtures_json with real IDs/query params; if 401, set AUTH_TOKEN.",
    }

def _json_get(obj: Any, path: str) -> Optional[Any]:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

@mcp.tool
async def auth_login(
    user_name: str,
    password: str,
    login_path: str = "/api/auth/login",
    token_json_path: str = "data.access_token",
) -> Dict[str, Any]:
    """
    Logs in and stores Bearer token in memory for subsequent requests.
    Your API:
      POST /api/auth/login
      body: {"user_name": "...", "password": "..."}
      token: data.access_token
    """
    global RUNTIME_TOKEN

    url = BASE_URL.rstrip("/") + login_path
    payload = {"user_name": user_name, "password": password}

    async with httpx.AsyncClient() as client:
        r = await client.post(url, json=payload, headers={"Accept": "application/json"}, timeout=DEFAULT_TIMEOUT_S)
        r.raise_for_status()
        data = r.json()

    token = _json_get(data, token_json_path)
    if not token or not isinstance(token, str):
        return {
            "ok": False,
            "message": f"Login succeeded but token not found at '{token_json_path}'",
            "status": r.status_code
        }

    RUNTIME_TOKEN = token
    return {"ok": True, "message": "Logged in; token stored for this MCP session", "token_prefix": token[:12] + "..."}

if __name__ == "__main__":
    mcp.run(transport="stdio")
