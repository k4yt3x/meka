#!/usr/bin/env python3
"""A minimal MCP server over stdio, for tests that need a real peer.

Speaks both protocol eras, which it decides from the client's first request the way the SDKs do:
`initialize` makes the process legacy (2025-11-25), `server/discover` or a request carrying modern
`_meta` makes it modern (2026-07-28). One newline-delimited JSON-RPC message per line, which is
what the stdio transport uses.

Everything it advertises and does is read from a *state file* named by argv[1], re-read on every
request, so a test can change what the server does between two requests to it. The file is JSON:

    {
      "tools": ["search", "create_page"],   tools advertised; each takes no arguments
      "instructions": "call search first",
      "exit_after": 1,                      tools/list calls to serve before exiting
      "resources": ["file:///a"],           resources advertised, and the ones a subscription
                                            to updates is accepted for
      "legacy_only": true,                  answer server/discover with method-not-found
      "exit_on_probe": true,                exit on server/discover, as a server that dies on a
                                            method it does not know
      "ignore_probe": true,                 answer nothing to server/discover, or to anything
                                            after it, as a server that swallows the unknown
      "mute": true,                         answer nothing to anything, from now on
      "elicit": {"message": "Name?", "schema": {"type": "object", "properties": {}}},
                                            tools/call asks this once, as input_required, then
                                            answers with what came back; with "repeat": true it
                                            asks on every call and never answers
      "progress": true,                     tools/call sends one notifications/progress first
      "prompts": ["p"],                     prompts advertised; each renders one user message
      "elicit_reads": true,                 resources/read asks `elicit` once, as tools/call does
      "elicit_prompts": true,               prompts/get asks `elicit` once, as tools/call does
      "notify": {"seq": 1, "tools_list_changed": true, "resources_updated": ["file:///a"]},
                                            emitted once per new seq: on the listen stream
                                            whose filter asked for it (modern), unsolicited
                                            and for the URIs subscribed (legacy)
      "notify": {"seq": 2, "end_streams": true},
                                            ends every open listen stream with its final
                                            result instead (modern)
      "log": "/path/requests.jsonl"         every request appended as one JSON line
    }

`exit_after` is how a test closes the transport under meka without needing the child's pid.
"""

import json
import sys
import threading
import time

MODERN = "2026-07-28"
LEGACY = "2025-11-25"
META_PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"
META_SUBSCRIPTION_ID = "io.modelcontextprotocol/subscriptionId"
SERVER_INFO = {"name": "stub", "version": "0"}
CAPABILITIES = {
    "tools": {"listChanged": True},
    "prompts": {"listChanged": True},
    "resources": {"subscribe": True, "listChanged": True},
}

stdout_lock = threading.Lock()


def state(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def send(message):
    with stdout_lock:
        sys.stdout.write(json.dumps(message) + "\n")
        sys.stdout.flush()


def reply(message_id, result):
    send({"jsonrpc": "2.0", "id": message_id, "result": result})


def fail(message_id, code, message, data=None):
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    send({"jsonrpc": "2.0", "id": message_id, "error": error})


def notify(method, params):
    send({"jsonrpc": "2.0", "method": method, "params": params})


class Server:
    def __init__(self, path):
        self.path = path
        self.era = None
        self.served_lists = 0
        # Modern: listen request id -> the acknowledged filter. Legacy: the URIs subscribed to.
        self.streams = {}
        self.legacy_subscribed = set()
        self.notified_seq = 0
        self.lock = threading.Lock()

    def log(self, request):
        path = state(self.path).get("log")
        if path:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(request) + "\n")

    def result(self, body):
        """A result in the era's shape: modern ones carry the type and the server's identity."""
        if self.era == MODERN:
            body = dict(body)
            body.setdefault("resultType", "complete")
            meta = dict(body.get("_meta", {}))
            meta[META_SERVER_INFO] = SERVER_INFO
            body["_meta"] = meta
        return body

    def handle(self, request):
        method = request.get("method")
        params = request.get("params") or {}
        message_id = request.get("id")
        meta = params.get("_meta") or {}
        current = state(self.path)
        self.log(request)
        if self.era == "silent" or current.get("mute"):
            return

        if self.era is None:
            if method == "initialize":
                self.era = LEGACY
            elif current.get("legacy_only"):
                # A server from before the probe existed knows no era to settle on; the
                # `initialize` that follows the probe's rejection decides.
                pass
            elif method == "server/discover" or META_PROTOCOL_VERSION in meta:
                self.era = MODERN

        if message_id is None:
            self.handle_notification(method, params)
            return

        if self.era == MODERN:
            version = meta.get(META_PROTOCOL_VERSION)
            if version is None or META_CLIENT_CAPABILITIES not in meta:
                fail(message_id, -32602, "modern request without its _meta")
                return
            if version != MODERN:
                fail(
                    message_id,
                    -32022,
                    "Unsupported protocol version",
                    {"supported": [MODERN, LEGACY], "requested": version},
                )
                return

        if method == "server/discover":
            if current.get("ignore_probe"):
                # A server that swallows a method it does not know, and answers nothing after
                # it either: the request is logged and the process stays silent for good.
                self.era = "silent"
                return
            if current.get("legacy_only"):
                fail(message_id, -32601, "no such method: server/discover")
                return
            if current.get("exit_on_probe"):
                sys.exit(0)
            result = {
                "supportedVersions": [MODERN, LEGACY],
                "capabilities": CAPABILITIES,
                "ttlMs": 60000,
                "cacheScope": "private",
            }
            if current.get("instructions") is not None:
                result["instructions"] = current["instructions"]
            reply(message_id, self.result(result))
        elif method == "initialize":
            result = {
                "protocolVersion": LEGACY,
                "capabilities": CAPABILITIES,
                "serverInfo": SERVER_INFO,
            }
            if current.get("instructions") is not None:
                result["instructions"] = current["instructions"]
            reply(message_id, result)
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": name,
                        "description": f"stub {name}",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                    for name in current.get("tools", [])
                ]
            }
            if self.era == MODERN:
                result["ttlMs"] = 60000
                result["cacheScope"] = "private"
            reply(message_id, self.result(result))
            self.served_lists += 1
            limit = current.get("exit_after")
            if limit is not None and self.served_lists >= limit:
                sys.exit(0)
        elif method == "tools/call":
            self.call_tool(message_id, params, meta, current)
        elif method == "resources/list":
            result = {
                "resources": [
                    {"uri": uri, "name": uri.rsplit("/", 1)[-1]}
                    for uri in current.get("resources", [])
                ]
            }
            if self.era == MODERN:
                result["ttlMs"] = 60000
                result["cacheScope"] = "private"
            reply(message_id, self.result(result))
        elif method == "resources/read":
            uri = params.get("uri")
            if uri not in current.get("resources", []):
                fail(message_id, -32602, f"no such resource: {uri}")
                return
            answers = params.get("inputResponses")
            if current.get("elicit_reads") and answers is None:
                self.ask(message_id, current["elicit"])
                return
            text = f"content of {uri}"
            if answers is not None:
                text += f" with {json.dumps(answers, sort_keys=True)} after {params.get('requestState')}"
            result = {"contents": [{"uri": uri, "mimeType": "text/plain", "text": text}]}
            if self.era == MODERN:
                result["ttlMs"] = 60000
                result["cacheScope"] = "private"
            reply(message_id, self.result(result))
        elif method == "prompts/list":
            result = {"prompts": [{"name": name, "description": f"stub {name}"} for name in current.get("prompts", [])]}
            if self.era == MODERN:
                result["ttlMs"] = 60000
                result["cacheScope"] = "private"
            reply(message_id, self.result(result))
        elif method == "prompts/get":
            name = params.get("name")
            answers = params.get("inputResponses")
            if current.get("elicit_prompts") and answers is None:
                self.ask(message_id, current["elicit"])
                return
            text = f"prompt {name}"
            if answers is not None:
                text += f" with {json.dumps(answers, sort_keys=True)} after {params.get('requestState')}"
            reply(
                message_id,
                self.result({"messages": [{"role": "user", "content": {"type": "text", "text": text}}]}),
            )
        elif method == "resources/subscribe" and self.era == LEGACY:
            uri = params.get("uri")
            if uri in current.get("resources", []):
                self.legacy_subscribed.add(uri)
                reply(message_id, {})
            else:
                fail(message_id, -32602, f"no such resource: {uri}")
        elif method == "resources/unsubscribe" and self.era == LEGACY:
            self.legacy_subscribed.discard(params.get("uri"))
            reply(message_id, {})
        elif method == "subscriptions/listen" and self.era == MODERN:
            requested = params.get("notifications") or {}
            accepted = {}
            for kind in ("toolsListChanged", "promptsListChanged", "resourcesListChanged"):
                if requested.get(kind):
                    accepted[kind] = True
            offered = current.get("resources", [])
            uris = [uri for uri in requested.get("resourceSubscriptions", []) if uri in offered]
            if uris:
                accepted["resourceSubscriptions"] = uris
            with self.lock:
                self.streams[message_id] = accepted
            notify(
                "notifications/subscriptions/acknowledged",
                {"_meta": {META_SUBSCRIPTION_ID: message_id}, "notifications": accepted},
            )
        elif method == "ping":
            reply(message_id, self.result({}))
        else:
            fail(message_id, -32601, f"no such method: {method}")

    def handle_notification(self, method, params):
        if method != "notifications/cancelled":
            return
        request_id = params.get("requestId")
        with self.lock:
            ended = self.streams.pop(request_id, None)
        # A listen stream the client cancelled ends with the final result the spec defines.
        if ended is not None:
            reply(request_id, self.result({"_meta": {META_SUBSCRIPTION_ID: request_id}}))

    def call_tool(self, message_id, params, meta, current):
        name = params.get("name")
        if current.get("progress") and meta.get("progressToken") is not None:
            notify(
                "notifications/progress",
                {
                    "progressToken": meta["progressToken"],
                    "progress": 1,
                    "total": 2,
                    "message": "halfway",
                },
            )
        elicit = current.get("elicit")
        answers = params.get("inputResponses")
        if elicit and (answers is None or elicit.get("repeat")):
            self.ask(message_id, elicit)
            return
        text = f"{name} ran"
        if answers is not None:
            text = f"{name} ran with {json.dumps(answers, sort_keys=True)} after {params.get('requestState')}"
        reply(
            message_id,
            self.result({"content": [{"type": "text", "text": text}], "isError": False}),
        )

    def ask(self, message_id, elicit):
        """Answer a request with the one elicitation `elicit` describes, as input_required."""
        reply(
            message_id,
            self.result(
                {
                    "resultType": "input_required",
                    "inputRequests": {
                        "answer": {
                            "method": "elicitation/create",
                            "params": {
                                "mode": "form",
                                "message": elicit["message"],
                                "requestedSchema": elicit["schema"],
                            },
                        }
                    },
                    "requestState": "round-1",
                }
            ),
        )

    def emit_pending(self):
        """Send what the state file's `notify` asks for, once per new sequence number."""
        current = state(self.path)
        wanted = current.get("notify")
        if not wanted or wanted.get("seq", 0) <= self.notified_seq:
            return
        self.notified_seq = wanted["seq"]
        if self.era == MODERN:
            with self.lock:
                streams = list(self.streams.items())
            if wanted.get("end_streams"):
                # The server's own end of every open stream: the final result the spec defines,
                # as a server about to restart sends, with nothing on the stream first.
                with self.lock:
                    self.streams.clear()
                for stream_id, _accepted in streams:
                    reply(stream_id, self.result({"_meta": {META_SUBSCRIPTION_ID: stream_id}}))
                return
            for stream_id, accepted in streams:
                tagged = {"_meta": {META_SUBSCRIPTION_ID: stream_id}}
                if wanted.get("tools_list_changed") and accepted.get("toolsListChanged"):
                    notify("notifications/tools/list_changed", tagged)
                for uri in wanted.get("resources_updated", []):
                    if uri in accepted.get("resourceSubscriptions", []):
                        notify("notifications/resources/updated", {"uri": uri, **tagged})
        elif self.era == LEGACY:
            if wanted.get("tools_list_changed"):
                notify("notifications/tools/list_changed", {})
            for uri in wanted.get("resources_updated", []):
                if uri in self.legacy_subscribed:
                    notify("notifications/resources/updated", {"uri": uri})

    def watch(self):
        while True:
            time.sleep(0.05)
            try:
                self.emit_pending()
            except (OSError, ValueError):
                # The state file is being rewritten; the next tick reads it whole.
                continue


def main():
    server = Server(sys.argv[1])
    threading.Thread(target=server.watch, daemon=True).start()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        server.handle(request)


if __name__ == "__main__":
    main()
