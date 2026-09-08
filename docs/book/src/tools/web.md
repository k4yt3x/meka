# Web

## `fetch_url`

Fetch a web page and return its content as markdown text.

**Permission:** Read

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `url` | string | yes | The URL to fetch |
| `limit` | integer | no | Maximum characters to return (default: 30000, 0 for no limit) |
| `headers` | object | no | Custom HTTP headers (overrides defaults like User-Agent) |
| `regex` | string | no | If provided, return only matching content (matches joined by newlines) |
| `raw` | boolean | no | Return raw HTML instead of converting to markdown (default: false) |
| `scratchpad` | string | no | Save output to the scratchpad under this name |

### Behavior

- Fetches the page via HTTP GET.
- Converts HTML to Markdown using `fast_html2md` (unless `raw` is true). `<nav>` and `<footer>` containers are preserved (rewritten to `<div>` before conversion) so their links survive; `fast_html2md` would otherwise drop those subtrees as boilerplate. `<script>` / `<style>` / `<head>` are still stripped.
- Resolves root-relative links against the page's final (post-redirect) URL, so a `/docs` href renders as the absolute `https://host/docs` the model can follow directly.
- Truncates the output to `limit` characters (default: 30,000). When `regex` is given, the pattern runs against the whole document *before* this cap, so `limit` never decides which matches exist; the cap then applies to the joined match list.
- HTTP timeout: 30 seconds by default; [`[web].request_timeout`](../configuration/config-file.md#web) changes it, and `connect_timeout` / `read_timeout` add tighter caps on the handshake and on a stalled body.
- Reads at most 10 MiB of decompressed body, checked while streaming so a small compressed payload cannot expand past it.
- Returns the HTTP status code as an error if the request fails (e.g., 404, 500).
- `fetch_url` is not a network boundary. It reaches whatever the process can reach, including private and loopback addresses, and so does a sandboxed `execute_command`, whose network is open. Confine the network at the host, not per tool.

### Image URLs

If the response `Content-Type` is a supported raster image format, `fetch_url` returns a multimodal `Image` content block instead of markdown. No disk is touched; bytes are base64-encoded in memory.

**Provider-native formats** (passed through unchanged):
- `image/png`, `image/jpeg` (and `image/jpg`), `image/gif`, `image/webp`, `image/bmp` (and `image/x-ms-bmp`)

**Convertible formats** (decoded and re-encoded as PNG transparently):
- `image/tiff`, `image/vnd.microsoft.icon` / `image/x-icon`, `image/vnd.radiance` (HDR), `image/x-exr`, `image/x-targa`, `image/x-portable-*` (PNM), `image/qoi`, `image/vnd.ms-dds`, `image/x-farbfeld`

**Unsupported formats** (fall through to the text branch): `image/svg+xml`, `image/jxl`, `image/heic`, `image/avif`.

- The `limit`, `regex`, and `raw` options do **not** apply to image responses.
- Size cap of ~3.75 MB applies to the **output** bytes (after conversion). Conversion can enlarge an image, so a 1 MB TIFF may produce a larger PNG.
- Detection uses the response's actual `Content-Type` header, so redirect chains and extension-less URLs are handled correctly.

Only fetch image URLs when the current model supports vision input; text-only models will either error or silently drop the image block.
