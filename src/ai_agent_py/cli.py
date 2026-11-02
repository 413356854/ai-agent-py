"""Command line interface (typer + rich).

ai-agent-py ingest ./data/raw --recreate
ai-agent-py ask "报销流程是什么？"
ai-agent-py chat
ai-agent-py demo          # 无需 API Key 的离线全流程演示
ai-agent-py serve         # 启动 FastAPI
"""

from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn
from rich.table import Table

from ai_agent_py import __version__
from ai_agent_py.config import get_settings
from ai_agent_py.errors import AiAgentError
from ai_agent_py.graph.agent import ask_agent, build_agent_graph
from ai_agent_py.llm import get_chat_model
from ai_agent_py.logging_setup import configure_logging
from ai_agent_py.offline import LexicalHashEmbeddings, StaticChatModel
from ai_agent_py.rag.chain import NO_CONTEXT_ANSWER, generate_answer, stream_answer
from ai_agent_py.rag.ingest import IngestReport
from ai_agent_py.rag.ingest import ingest as run_ingest
from ai_agent_py.rag.retriever import get_retriever
from ai_agent_py.rag.store import build_vector_store, store_size

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="ai-agent-py：基于 LangChain + LangGraph 的 RAG 应用脚手架",
)
console = Console()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fail(message: str, code: int = 1) -> typer.Exit:
    console.print(Panel(message, title="出错", border_style="red"))
    return typer.Exit(code=code)


def _force_utf8_streams() -> None:
    """Keep Chinese output intact on Windows consoles and when piped."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            # pragma: no cover - depends on the stream implementation
            with contextlib.suppress(OSError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")


# Applied at import time: `--help` / `--version` are rendered before the
# command callback runs, so reconfiguring there would be too late.
_force_utf8_streams()


def _snippet(text: str, limit: int = 100) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[:limit]}…"


def _print_report(report: IngestReport) -> None:
    table = Table(title="索引完成", show_header=False, box=None, title_justify="left")
    table.add_column("字段", style="cyan", no_wrap=True)
    table.add_column("值")
    for key, value in report.as_dict().items():
        table.add_row(key, str(value))
    console.print(table)


def _print_sources(documents: list[Any]) -> None:
    table = Table(title="引用来源", title_justify="left")
    table.add_column("#", style="cyan", width=3, justify="right")
    table.add_column("来源", no_wrap=True)
    table.add_column("片段", overflow="fold")
    for index, document in enumerate(documents, start=1):
        source = str(document.metadata.get("source", "unknown"))
        page = document.metadata.get("page")
        if page is not None:
            source += f" (p.{page})"
        table.add_row(str(index), source, _snippet(document.page_content))
    console.print(table)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def _version_callback(value: bool) -> None:
    """Eager option callback: `--version` works without a sub-command."""
    if value:
        console.print(f"ai-agent-py {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="显示版本后退出",
            is_eager=True,
            callback=_version_callback,
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="输出调试日志")] = False,
) -> None:
    """全局选项。"""
    configure_logging("DEBUG" if verbose else None)


@app.command()
def info() -> None:
    """显示当前配置（密钥脱敏）与索引状态。"""
    cfg = get_settings()
    table = Table(title=f"ai-agent-py {__version__} · 配置", show_header=False, box=None)
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("value")
    for key, value in cfg.describe().items():
        table.add_row(key, str(value))
    console.print(table)

    try:
        chunks = store_size(build_vector_store(settings=cfg))
    except AiAgentError as exc:
        console.print(f"[yellow]无法打开向量库：{exc}[/]")
    else:
        console.print(f"已索引 chunk 数：[bold]{chunks if chunks is not None else '未知'}[/]")


@app.command()
def ingest(
    paths: Annotated[
        list[str] | None,
        typer.Argument(help="文件或目录（可多个）；默认使用 DATA_DIR"),
    ] = None,
    recreate: Annotated[bool, typer.Option("--recreate", "-r", help="清空集合后重建索引")] = False,
    chunk_size: Annotated[int | None, typer.Option("--chunk-size", help="分块大小（字符）")] = None,
    chunk_overlap: Annotated[
        int | None, typer.Option("--chunk-overlap", help="分块重叠（字符）")
    ] = None,
) -> None:
    """把文档切块、向量化并写入向量库。"""
    cfg = get_settings()
    targets: list[str | Path] = list(paths) if paths else [cfg.data_dir]
    console.print(f"文档来源：{[str(t) for t in targets]}")

    task_id: TaskID | None = None
    started = time.perf_counter()

    def on_progress(done: int, total: int) -> None:
        nonlocal task_id
        if task_id is None:
            task_id = progress.add_task("embedding", total=total)
        progress.update(task_id, completed=done)

    try:
        store = build_vector_store(settings=cfg)
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} chunks"),
            console=console,
            transient=True,
        ) as progress:
            report = run_ingest(
                targets,
                store=store,
                settings=cfg,
                recreate=recreate,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                on_progress=on_progress,
            )
    except (AiAgentError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    if report.chunks == 0:
        console.print(
            Panel(
                f"没有找到可索引的内容（{cfg.data_dir}）。\n"
                "请把 txt / md / pdf / docx 文档放进去，或显式指定路径。",
                border_style="yellow",
            )
        )
        return
    _print_report(report)
    console.print(f"总耗时 [bold]{time.perf_counter() - started:.2f}s[/]")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="要问的问题")],
    k: Annotated[int | None, typer.Option("-k", "--top-k", help="召回条数")] = None,
    stream: Annotated[bool, typer.Option("--stream", help="流式输出")] = False,
    sources: Annotated[bool, typer.Option("--sources/--no-sources", help="显示引用来源")] = True,
) -> None:
    """一次性问答（RAG chain：检索 → 提示词 → 回答）。"""
    cfg = get_settings()
    try:
        llm = get_chat_model(cfg, streaming=stream)
        retriever = get_retriever(settings=cfg, k=k)
    except AiAgentError as exc:
        raise _fail(str(exc)) from exc

    started = time.perf_counter()
    with console.status("[cyan]检索中…", spinner="dots"):
        documents = list(retriever.invoke(question))

    if not documents:
        console.print(Panel(NO_CONTEXT_ANSWER, title="未命中", border_style="yellow"))
        raise typer.Exit(code=0)

    try:
        if stream:
            console.print()
            for chunk in stream_answer(llm, question, documents):
                console.print(chunk, end="", markup=False, highlight=False, soft_wrap=True)
            console.print()
        else:
            with console.status("[cyan]生成回答…", spinner="dots"):
                answer = generate_answer(llm, question, documents)
            console.print(
                Panel(Markdown(answer), title="回答", border_style="green", padding=(1, 2))
            )
    except AiAgentError as exc:
        raise _fail(str(exc)) from exc
    except Exception as exc:  # upstream provider errors
        raise _fail(f"调用模型失败：{exc}") from exc

    if sources:
        _print_sources(documents)
    console.print(f"[dim]耗时 {time.perf_counter() - started:.2f}s[/]")


@app.command()
def chat(
    thread_id: Annotated[
        str, typer.Option("--thread-id", help="会话 ID（用于保留上下文）")
    ] = "cli",
    max_turns: Annotated[
        int | None, typer.Option("--max-turns", help="单轮最多工具调用次数")
    ] = None,
) -> None:
    """交互式对话（LangGraph agent，会调用检索工具）。"""
    from langgraph.checkpoint.memory import InMemorySaver

    cfg = get_settings()
    try:
        llm = get_chat_model(cfg)
        retriever = get_retriever(settings=cfg)
    except AiAgentError as exc:
        raise _fail(str(exc)) from exc

    saver = InMemorySaver()
    graph = build_agent_graph(llm=llm, retriever=retriever, checkpointer=saver)
    console.print(
        Panel(
            "输入问题开始对话；[bold]/reset[/] 清空上下文，[bold]/exit[/] 退出。",
            title=f"ai-agent-py chat · thread={thread_id}",
            border_style="blue",
        )
    )

    while True:
        try:
            text = console.input("[bold cyan]你 > [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not text:
            continue
        if text in {"/exit", "/quit", "exit", "quit"}:
            break
        if text == "/reset":
            delete = getattr(saver, "delete_thread", None)
            if callable(delete):
                delete(thread_id)
                console.print("[yellow]上下文已清空[/]")
            else:  # pragma: no cover - older langgraph
                console.print("[yellow]当前版本不支持清空，请重启会话[/]")
            continue

        try:
            with console.status("[cyan]思考中…", spinner="dots"):
                answer = ask_agent(graph, text, thread_id=thread_id, max_turns=max_turns)
        except Exception as exc:  # keep the REPL alive
            console.print(f"[red]出错：{exc}[/]")
            continue
        console.print(Panel(Markdown(answer or "（空回答）"), border_style="green", padding=(0, 1)))


@app.command()
def demo(
    question: Annotated[
        str, typer.Option("--question", "-q", help="演示问题")
    ] = "这个项目用什么管理依赖？",
) -> None:
    """离线演示：无需 API Key，跑通「加载 → 切块 → 向量化 → 检索 → 生成」。"""
    cfg = get_settings()
    offline_cfg = cfg.model_copy(
        update={"vector_store": "memory", "chunk_size": 500, "chunk_overlap": 80}
    )
    embeddings = LexicalHashEmbeddings()

    console.print(
        Panel("离线模式：词法哈希向量 + 占位模型，仅用于验证流程是否跑通。", border_style="blue")
    )

    try:
        store = build_vector_store(embeddings=embeddings, settings=offline_cfg)
        with console.status("[cyan]加载、切块、向量化…", spinner="dots"):
            report = run_ingest(
                [offline_cfg.data_dir], store=store, settings=offline_cfg, recreate=True
            )
    except (AiAgentError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    if report.chunks == 0:
        raise _fail(f"data 目录中没有可索引的文档：{offline_cfg.data_dir}")
    _print_report(report)

    retriever = get_retriever(store=store, settings=offline_cfg, k=3)
    documents = list(retriever.invoke(question))
    console.print(f"\n问题：[bold]{question}[/]")
    _print_sources(documents)

    answer = generate_answer(
        StaticChatModel(
            response="（离线演示）以上片段是检索到的最相关资料，真实模型会据此组织答案。"
        ),
        question,
        documents,
    )
    console.print(Panel(Markdown(answer), title="占位回答", border_style="green", padding=(1, 2)))
    console.print(
        "[dim]下一步：复制 .env.example 为 .env 并填入 API Key，然后运行 "
        '`ai-agent-py ingest` 与 `ai-agent-py ask "..."`。[/]'
    )


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="监听地址")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="监听端口")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="开发模式热重载")] = False,
) -> None:
    """启动 FastAPI 服务（交互式文档见 /docs）。"""
    import uvicorn

    console.print(f"API 文档：http://{host}:{port}/docs")
    uvicorn.run("ai_agent_py.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":  # pragma: no cover
    app()
