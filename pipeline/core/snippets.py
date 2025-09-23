"""Documentation linter implementation."""

import logging
import os
import re
import shutil
import subprocess
from functools import reduce
from pathlib import Path
from typing import TypedDict

import gitmatch
from tqdm import tqdm

from pipeline import constants

logger = logging.getLogger(__name__)


class CodeBlock(TypedDict):
    """Describes a code block, which is a single code block in a document."""

    line: int
    column: int
    language: str
    attributes: str
    content: str


class CodeSnippet(TypedDict):
    """Describes a group of code blocks.

    Code snippets are used to group consecutive code blocks that:
        1. Have the same language (or alias to the same language)
        2. Have a merge_before attribute in preceding code block frontmatter
    """

    language: str
    code_blocks: list[CodeBlock]


class DocumentationSnippetsParser:
    """Exports code blocks for documentation files in the source directory.

    This class handles the process of exporting code blocks for documentation files in
    the source directory to the snippets output directory for linting/testing by
    external tools.

    Attributes:
        src_dir: Path to the source directory containing documentation files.
        output_dir: Path to the snippets directory where files will be exported.
        copy_extensions: Set of file extensions that should be targeted for exporting.
        directory_whitelist: Set of directories that should be targeted for exporting.
    """

    def __init__(self, src_dir: Path, output_dir: Path) -> None:
        """Initialize the DocumentationLinter.

        Args:
            src_dir: Path to the source directory containing documentation files.
            output_dir: Path to the output directory where files will be written.
        """
        self.src_dir = src_dir
        self.output_dir = output_dir

        # File extensions to check
        self.file_extensions: set[str] = {
            ".mdx",
            ".md",
        }

        # Directories to check
        self.directory_whitelist: set[str] = {
            "oss",
        }

        # Language aliases - used to map language codes to full names for snippets
        self.language_aliases: dict[str, str] = {
            "py": "python",
            "python": "python",
            "js": "typescript",
            "javascript": "typescript",
            "ts": "typescript",
            "typescript": "typescript",
        }

        # Language file association - used to map languages to file extensions
        self.language_file_assoc: dict[str, str] = {
            "python": ".py",
            "typescript": ".ts",
        }

    def clean_all(self) -> None:
        """Clean all snippets in the output directory."""
        logger.info("Cleaning all snippets in %s", self.output_dir)
        for language_dir in self._get_language_paths():
            self._clean_ignored_files(language_dir)
            self._clean_empty_subdirs(language_dir)
            logger.info("Cleaned %s", language_dir)

    def export_all(self) -> None:
        """Export all snippets to the output directory."""
        logger.info("Exporting all snippets to %s/", self.output_dir)

        # Clean out per-language export dirs before exporting snippets
        self.clean_all()

        all_files: list[Path] = [
            path
            for top_dir in self.directory_whitelist
            for path in (self.src_dir / top_dir).rglob("*")
            if (self.src_dir / top_dir).exists()
            and (self.src_dir / top_dir).is_dir()
            and path.is_file()
            and path.suffix.lower() in self.file_extensions
        ]

        if not all_files:
            logger.info("No files to export in %s/", self.src_dir)
            return

        with tqdm(
            total=len(all_files),
            desc="Exporting files",
            unit="file",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for file_path in all_files:
                self._export_file(file_path, self.output_dir, pbar)
                pbar.update(1)

        logger.info("✅ Export complete")

    def lint_all(self) -> None:
        """Lint all snippets in the output directory.

        This will:
            1. Discover available languags in the output directory
            2. Ensure per-language export directories exist
            3. Export snippets (idempotent) so languages have input
            4. Run each language's lint command with progress output
        """
        logger.info("Linting all snippets in %s", self.output_dir)

        # Discover languages
        if not self.output_dir.exists() or not self.output_dir.is_dir():
            logger.info(
                "No lint commands found at %s, skipping linting", self.output_dir
            )
            return

        languages = [p.name for p in self._get_language_paths()]
        if not languages:
            logger.info("No supported languages found under %s", self.output_dir)
            return

        # Run lints per language with progress
        failures: list[tuple[str, int]] = []
        with tqdm(
            total=len(languages),
            desc="Linting snippets",
            unit="lang",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for language in languages:
                pbar.set_postfix_str(language)
                language_dir = self.output_dir / language
                return_code = self._run_language_lint(language, language_dir)
                if return_code != 0:
                    failures.append((language, return_code))
                pbar.update(1)

        if failures:
            failed_list = ", ".join(f"{lang} (rc={rc})" for lang, rc in failures)
            logger.error("❌ Linting failed for: %s", failed_list)
        else:
            logger.info("✅ Linting complete for all languages")

    def _get_language_paths(self) -> list[Path]:
        """Get all language paths in the output directory."""
        return [
            p
            for p in self.output_dir.iterdir()
            if p.is_dir() and p.name in self.language_file_assoc
        ]

    def _clean_ignored_files(self, target_dir: Path) -> None:
        """Clean a directory by removing all files excluded by .gitignore.

        If a .gitignore file exists in a directory, it will omit all files that match
        the patterns in the .gitignore file.
        """
        gi: gitmatch.Gitignore | None = None
        gitignore_path = target_dir / ".gitignore"
        if gitignore_path.exists():
            with gitignore_path.open("r") as f:
                gi = gitmatch.compile(line.strip() for line in f if line.strip())

        if not gi:
            return

        # Iterate over every file path (relative to the target_dir)
        for dirpath, _, filenames in os.walk(target_dir):
            for fname in filenames:
                file_path = Path(dirpath) / fname
                rel_path = file_path.relative_to(target_dir)
                # Remove the file if it matches the gitignore patterns
                if gi.match(rel_path):
                    try:
                        file_path.unlink()
                    except OSError as e:
                        logger.warning("Failed to remove %s: %s", file_path, e)

    def _clean_empty_subdirs(self, target_dir: Path) -> None:
        """Recursively clean empty subdirectories from the target directory."""
        if not target_dir.is_dir():
            return
        # Recurse into subdirectories first
        for subdir in target_dir.iterdir():
            if subdir.is_dir():
                self._clean_empty_subdirs(subdir)
        # After cleaning subdirectories, remove this directory if empty or a symlink
        if target_dir.is_symlink():
            target_dir.unlink()
        elif not any(target_dir.iterdir()):
            target_dir.rmdir()

    def _export_file(self, file_path: Path, output_dir: Path, pbar: tqdm) -> None:
        """Export a single file to a directory containing code snippets."""
        pbar.set_postfix_str(file_path.name)
        code_snippets = self._extract_code_snippets(file_path.read_text())
        for idx, code_snippet in enumerate(code_snippets):
            # Ensure the output directory exists
            language_path = output_dir / code_snippet["language"]
            if not language_path.exists():
                logger.info(
                    "Language path %s does not exist, skipping export", language_path
                )
                continue

            output_path = language_path / file_path.relative_to(self.src_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            # Write the code snippet to the output file
            ext = self.language_file_assoc.get(code_snippet["language"], ".txt")
            content = self._stringify_code_snippet(
                code_snippet, file_path=file_path.relative_to(self.src_dir).as_posix()
            )
            snippet_path = output_path / f"{code_snippet['language']}_{idx}{ext}"
            snippet_path.write_text(content)

    def _stringify_code_snippet(
        self,
        code_snippet: CodeSnippet,
        file_path: str | None = None,
        *,
        annotate_blocks: bool = True,
    ) -> str:
        """Return a string representation of the code snippet.

        Args:
            code_snippet: The code snippet to stringify.
            file_path: The optional file path to the file containing the code snippet.
                Including this will add a comment to the beginning of the code snippet
                that points to the file and line number of the code snippet.
            annotate_blocks: Whether to annotate the code blocks with their file path
                and line number. If True, each code block will be prefixed with a
                comment containing the line and column number of the code block.
        """
        lines = []
        comment_prefix = constants.LANGUAGE_COMMENT_SYNTAX[code_snippet["language"]]
        if file_path and comment_prefix:
            lines.append(f"{comment_prefix}File: {file_path}")
        for block in code_snippet["code_blocks"]:
            if annotate_blocks and comment_prefix:
                lines.append(
                    f"{comment_prefix}Line: {block['line']}, Column: {block['column']}"
                )
            lines.append(_dedent(block["content"]))
        return "\n".join(lines).strip("\n")

    def _extract_code_snippets(self, content: str) -> list[CodeSnippet]:
        """Extracts code snippets from the content.

        When extracting code snippets, we collapse consecutive code blocks that:
            1. Have the same language (or alias to the same language)
            2. Have a 'merge-before' attribute in the codeblock frontmatter

        and only keep code blocks that have a 'export' attribute.
        """
        code_blocks = self._extract_code_blocks(content)

        def reduce_code_blocks(
            acc: list[CodeSnippet], block: CodeBlock
        ) -> list[CodeSnippet]:
            prev_snippet = acc[-1] if acc else None
            if (
                prev_snippet
                and prev_snippet["language"] == block["language"]
                and "merge-before" in block["attributes"]
            ):
                prev_snippet["code_blocks"].append(block)
            else:
                acc.append(CodeSnippet(language=block["language"], code_blocks=[block]))
            return acc

        return reduce(reduce_code_blocks, code_blocks, [])

    def _extract_code_blocks(self, content: str) -> list[CodeBlock]:
        """Extract code blocks from the content."""
        code_blocks = []
        for match in list(constants.CODE_BLOCK_PATTERN.finditer(content)):
            attributes = match.group("attributes").rstrip()
            if "export" not in attributes and "ignore" not in attributes:
                continue
            # Compute line and column from character index
            char_index = match.start()
            # Find the line number (0-based)
            line = content.count("\n", 0, char_index)
            # Find the column (0-based, number of chars since last \n)
            last_newline = content.rfind("\n", 0, char_index)
            if last_newline == -1:
                column = char_index
            else:
                column = char_index - (last_newline + 1)
            # Normalize language using aliases
            language = match.group("language").strip()
            if language in self.language_aliases:
                language = self.language_aliases[language]
            # Add the code block to the list
            code_blocks.append(
                CodeBlock(
                    line=line,
                    column=column,
                    language=language,
                    attributes=attributes,
                    content=match.group("code"),
                )
            )
        return code_blocks

    def _run_language_lint(self, language: str, snippets_dir: Path) -> int:
        """Run the lint command for a specific snippets directory.

        Returns the subprocess return code.
        """
        env = os.environ.copy()
        env["SNIPPETS_DIR"] = str((self.output_dir / language).resolve())
        env.setdefault("CI", "1")

        makefile = snippets_dir / "Makefile"

        cmd: list[str]
        cwd: Path | None = None

        if makefile.exists():
            cmd = ["make", "-C", str(snippets_dir), "lint"]
            cwd = None  # -C handles working dir
        else:
            logger.warning(
                "No lint command for %s",
                snippets_dir,
            )
            return 0

        logger.info("Running lint command for %s: %s", language, " ".join(cmd))
        # Resolve absolute path for the executable to avoid PATH ambiguity.
        resolved_executable = shutil.which(cmd[0])
        if resolved_executable is None:
            logger.error(
                "Executable not found for '%s' while linting %s",
                cmd[0],
                language,
            )
            return 127
        cmd[0] = resolved_executable

        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            logger.exception("Failed to execute linter for %s", language)
            return 127

        if result.stdout:
            logger.debug("%s stdout:\n%s", language, result.stdout)
        if result.stderr:
            logger.error("%s stderr:\n%s", language, result.stderr)

        if result.returncode != 0:
            logger.error(
                "Linter for %s exited with code %s", language, result.returncode
            )
        else:
            logger.info("%s lint passed", language)
        return result.returncode


def _dedent(text: str) -> str:
    """Remove the leading indentation (tabs or spaces) from all non-empty lines."""
    lines = text.split("\n")
    if not lines:
        return text
    # find the first non empty line
    first_line = ""
    for line in lines:
        if line.strip() != "":
            first_line = line
            break
    m = re.match(r"^[ \t]+", first_line)
    first_indent = m.group(0) if m else ""
    if not first_indent:
        return text
    dedented_lines = [line.removeprefix(first_indent) for line in lines]
    return "\n".join(dedented_lines)
