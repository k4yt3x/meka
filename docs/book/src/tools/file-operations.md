# File Operations

## `read_file`

Read the contents of a file at a given path. Supports text files and images.

**Permission:** Read

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | yes | The file path to read |
| `offset` | integer | no | Line number to start reading from (0-based) |
| `limit` | integer | no | Maximum number of lines to read |
| `scratchpad` | string | no | Save output to the scratchpad under this name |

### Behavior

- When `offset` and `limit` are both omitted, defaults to the first 2000 lines. If the file has more, a truncation notice is appended.
- Use `offset`/`limit` to page through large files.

### Image files

Recognized image extensions are returned as base64-encoded multimodal content:

- **Provider-native** (pass-through): `.png`, `.jpg`/`.jpeg`, `.gif`, `.webp`, `.bmp`
- **Convertible** (decoded and re-encoded as PNG transparently): `.tif`/`.tiff`, `.ico`, `.hdr`, `.exr`, `.tga`, `.pbm`/`.pgm`/`.ppm`/`.pnm`, `.qoi`, `.dds`, `.ff`/`.farbfeld`
- **Unsupported** (fall through to text read, which will fail on binary): `.svg`, `.jxl`, `.heic`, `.avif`

Images are rejected if the final payload exceeds 3.75 MB (~5 MB base64). Conversion can enlarge an image, so a small TIFF may produce a too-large PNG.

Only read image files when the current model supports vision input — text-only models will either error or silently drop the image block.

### Examples

Read an entire file:

```text
agsh [r] > show me the contents of src/main.rs
```

Read lines 10-20:

```text
agsh [r] > show me lines 10 through 20 of src/main.rs
```

---

## `edit_file`

Make a string replacement in a file. The file must have been read with `read_file` first (unless `force` is set).

**Permission:** Write

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | yes | The file path to edit |
| `old_string` | string | yes | The exact string to find and replace |
| `new_string` | string | yes | The replacement string |
| `replace_all` | boolean | no | Replace all occurrences (default: false) |
| `force` | boolean | no | Bypass read-before-edit requirement (default: false) |
| `scratchpad` | string | no | Save output to the scratchpad under this name |

### Behavior

- By default, only the **first** occurrence of `old_string` is replaced. Set `replace_all` to replace every occurrence.
- The file must have been previously read with `read_file` on the same path. This prevents blind edits. Set `force` to bypass this requirement.
- If `old_string` is not found, the tool returns an error (without modifying the file).

---

## `write_file`

Create or overwrite a file with the given content.

**Permission:** Write

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | yes | The file path to write |
| `content` | string | yes | The content to write to the file |
| `scratchpad` | string | no | Save output to the scratchpad under this name |

### Behavior

- Creates parent directories if they do not exist.
- Overwrites the file if it already exists.
