# Web Tools

## `fetch_url`

Fetch a web page and return its content as markdown text.

**Permission:** Read

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `url` | string | yes | The URL to fetch |
| `max_length` | integer | no | Maximum characters to return (default: 30000, 0 for no limit) |
| `headers` | object | no | Custom HTTP headers (overrides defaults like User-Agent) |
| `regex` | string | no | If provided, return only matching content (matches joined by newlines) |
| `raw` | boolean | no | Return raw HTML instead of converting to markdown (default: false) |
| `scratchpad` | string | no | Save output to the scratchpad under this name |

### Behavior

- Fetches the page via HTTP GET.
- Converts HTML to Markdown using `fast_html2md` (unless `raw` is true).
- Truncates the output to `max_length` characters (default: 30,000).
- HTTP timeout: 30 seconds.
- Returns the HTTP status code as an error if the request fails (e.g., 404, 500).

### Image URLs

If the response `Content-Type` is a supported raster image format, `fetch_url` returns a multimodal `Image` content block instead of markdown. No disk is touched — bytes are base64-encoded in memory.

**Provider-native formats** (passed through unchanged):
- `image/png`, `image/jpeg` (and `image/jpg`), `image/gif`, `image/webp`, `image/bmp` (and `image/x-ms-bmp`)

**Convertible formats** (decoded and re-encoded as PNG transparently):
- `image/tiff`, `image/vnd.microsoft.icon` / `image/x-icon`, `image/vnd.radiance` (HDR), `image/x-exr`, `image/x-targa`, `image/x-portable-*` (PNM), `image/qoi`, `image/vnd.ms-dds`, `image/x-farbfeld`

**Unsupported formats** (fall through to the text branch): `image/svg+xml`, `image/jxl`, `image/heic`, `image/avif`.

- The `max_length`, `regex`, and `raw` options do **not** apply to image responses.
- Size cap of ~3.75 MB applies to the **output** bytes (after conversion). Conversion can enlarge an image, so a 1 MB TIFF may produce a larger PNG.
- Detection uses the response's actual `Content-Type` header, so redirect chains and extension-less URLs are handled correctly.

Only fetch image URLs when the current model supports vision input — text-only models will either error or silently drop the image block.

---

## `web_search`

Search the web and return results. Supports multiple search engines.

**Permission:** Read

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query` | string | yes | The search query |
| `engine` | string | no | Search engine to use (default: `duckduckgo`) |
| `headers` | object | no | Custom HTTP headers |
| `scratchpad` | string | no | Save output to the scratchpad under this name |

### Search Engines

| Value | Engine |
|-------|--------|
| `duckduckgo` | DuckDuckGo (default) |
| `google` | Google Search |
| `bing` | Bing Search |

### Behavior

- Returns up to 10 results per search.
- Each result includes the title, URL, and a snippet (when available).
- Uses HTML scraping (no API keys required for any search engine).
- HTTP timeout: 30 seconds.
