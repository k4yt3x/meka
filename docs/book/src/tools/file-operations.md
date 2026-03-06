# File Operations

## `read_file`

Read the contents of a file at a given path.

**Permission:** Read

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | yes | The file path to read |
| `offset` | integer | no | Line number to start reading from (0-based) |
| `limit` | integer | no | Maximum number of lines to read |

### Examples

Read an entire file:

```text
agsh [r] > show me the contents of src/main.rs
```

Read lines 10-20:

```text
agsh [r] > show me lines 10 through 20 of src/main.rs
```

The agent will call `read_file` with `offset: 10` and `limit: 10`.

Preview a [skill](../usage/skills.md) file's title and summary:

```text
agsh [r] > what does the deploy-app skill cover?
```

The agent will call `read_file` with `limit: 3` to read the title and summary without loading the full file.

---

## `edit_file`

Make a string replacement in a file. Replaces the first occurrence of `old_string` with `new_string`.

**Permission:** Write

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | yes | The file path to edit |
| `old_string` | string | yes | The exact string to find and replace |
| `new_string` | string | yes | The replacement string |

### Behavior

- Reads the file, performs the replacement, and writes it back.
- Only the **first** occurrence of `old_string` is replaced.
- If `old_string` is not found, the tool returns an error message (without modifying the file).

### Examples

```text
agsh [w] > change the function name "foo" to "bar" in src/lib.rs
```

```text
agsh [w] > fix the typo "recieve" to "receive" in README.md
```

---

## `write_file`

Create or overwrite a file with the given content.

**Permission:** Write

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | yes | The file path to write |
| `content` | string | yes | The content to write to the file |

### Behavior

- Creates parent directories if they do not exist.
- Overwrites the file if it already exists.

### Examples

```text
agsh [w] > create a new file called hello.py that prints "hello world"
```

```text
agsh [w] > write a .gitignore file for a Rust project
```
