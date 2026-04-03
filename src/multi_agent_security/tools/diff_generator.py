import difflib
import re


def generate_unified_diff(original: str, patched: str, file_path: str) -> str:
    """Generate a unified diff between original and patched code.

    Uses difflib.unified_diff with 3 context lines.
    Returns the diff as a string (empty string if no changes).
    """
    original_lines = original.splitlines(keepends=True)
    patched_lines = patched.splitlines(keepends=True)
    diff = difflib.unified_diff(
        original_lines,
        patched_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        n=3,
    )
    return "".join(diff)


def apply_patch_to_content(
    original: str, patched_code: str, line_start: int, line_end: int
) -> str:
    """Replace lines line_start..line_end (1-indexed, inclusive) in original with patched_code.

    Returns the full patched file content.
    Used when the LLM returns only a patched snippet, not the full file.
    """
    lines = original.splitlines(keepends=True)
    replacement_lines = patched_code.splitlines(keepends=True)

    # Normalise trailing newline on the last replacement line to match the
    # surrounding context (if the line being replaced ends with \n, the
    # replacement should too).
    if replacement_lines:
        last_idx = line_end - 1  # 0-indexed position of the last replaced line
        original_last_ends_with_newline = (
            last_idx < len(lines) and lines[last_idx].endswith("\n")
        )
        last_replacement = replacement_lines[-1]
        if original_last_ends_with_newline and not last_replacement.endswith("\n"):
            replacement_lines[-1] = last_replacement + "\n"

    new_lines = lines[: line_start - 1] + replacement_lines + lines[line_end:]
    return "".join(new_lines)


def validate_diff(diff: str) -> bool:
    """Basic validation of a unified diff.

    Returns False if:
    - The diff is empty / whitespace-only
    - There are no actual +/- change lines (excluding +++ / --- headers)
    - More than 50% of the original hunk lines are removed (sanity check against
      destructive patches that wipe most of the file)
    """
    if not diff.strip():
        return False

    lines = diff.splitlines()

    # Collect change lines, excluding the file header markers
    added = [ln for ln in lines if ln.startswith("+") and not ln.startswith("+++")]
    removed = [ln for ln in lines if ln.startswith("-") and not ln.startswith("---")]

    if not added and not removed:
        return False

    # Parse @@ -start,count +start,count @@ headers to get the original line
    # counts for each hunk.  A hunk header like "@@ -10,8 +10,6 @@" means the
    # original contained 8 lines in that region.
    original_hunk_lines = 0
    for line in lines:
        if not line.startswith("@@"):
            continue
        # Extract the "-start[,count]" part (first token after @@)
        m = re.search(r"@@\s+-(\d+)(?:,(\d+))?", line)
        if m:
            count = int(m.group(2)) if m.group(2) is not None else 1
            original_hunk_lines += count

    # If we could parse hunk headers, check the 50% removal threshold.
    # Use *net* removals (removed - added) to avoid false-positives on
    # equal-sized replacements (e.g. replace 1 line with 1 line).
    net_removed = len(removed) - len(added)
    if original_hunk_lines > 0 and net_removed > original_hunk_lines * 0.5:
        return False

    return True
