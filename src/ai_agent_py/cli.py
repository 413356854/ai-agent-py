"""Command line interface (typer + rich).

ai-agent-py ingest ./data/raw --recreate
ai-agent-py ask "报销流程是什么？"
ai-agent-py chat
ai-agent-py demo          # 无需 API Key 的离线全流程演示
ai-agent-py serve         # 启动 FastAPI
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn
from rich.table import Table

from ai_agent_py import __version__
from ai_agent_py.config import PROJECT_ROOT, get_settings
from ai_agent_py.db.health import DatabaseHealth, check_database
from ai_agent_py.errors import AiAgentError
from ai_agent_py.graph.agent import ask_agent, build_agent_graph
from ai_agent_py.llm import get_chat_model
from ai_agent_py.logging_setup import configure_logging
from ai_agent_py.offline import LexicalHashEmbeddings, OfflineChunkSearcher, StaticChatModel
from ai_agent_py.rag.chain import NO_CONTEXT_ANSWER, generate_answer, stream_answer
from ai_agent_py.rag.ingest import IngestReport
from ai_agent_py.rag.ingest import ingest as run_ingest
from ai_agent_py.rag.milvus_search import get_milvus_searcher
from ai_agent_py.rag.records import MilvusRagReader

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="ai-agent-py：基于 LangChain + LangGraph 的 RAG 应用脚手架",
)
console = Console()

db_app = typer.Typer(
    help="MySQL 连接检查与文档任务记录迁移（不自动建表）",
    no_args_is_help=True,
)
app.add_typer(db_app, name="db")


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
        chunks = MilvusRagReader(cfg).count()
    except AiAgentError as exc:
        console.print(f"[yellow]无法读取 Milvus 索引：{exc}[/]")
    else:
        console.print(f"已索引 chunk 数：[bold]{chunks}[/]")


def _print_database(health: DatabaseHealth, *, dsn: str | None) -> None:
    labels = {
        "ok": "[green]正常[/]",
        "down": "[red]连接失败[/]",
        "not_configured": "[yellow]未配置[/]",
    }
    table = Table(title="MySQL 连接检查", show_header=False, box=None)
    table.add_column("字段", style="cyan", no_wrap=True)
    table.add_column("值", overflow="fold")
    table.add_row("状态", labels.get(health.status, health.status))
    table.add_row("连接串", dsn or "（未配置）")
    if health.configured:
        table.add_row("服务端", health.server_version or "-")
        table.add_row("当前账号", health.user or "-")
        table.add_row("数据库", health.database or "-")
        table.add_row("表数量", str(health.table_count) if health.table_count is not None else "-")
        table.add_row("会话时区", health.session_time_zone or "-")
        table.add_row("耗时", f"{health.latency_ms} ms" if health.latency_ms is not None else "-")
        if health.pool:
            pool = ", ".join(f"{key}={value}" for key, value in health.pool.items())
            table.add_row("连接池", pool)
    if health.error:
        table.add_row("错误", f"[red]{escape(health.error)}[/]")
    console.print(table)


@db_app.command("check")
def db_check() -> None:
    """连接本地 MySQL，确认账号、库名、表数量与连接池可用。"""
    cfg = get_settings()
    health = check_database(cfg)
    _print_database(health, dsn=cfg.mysql_safe_dsn)
    if not health.ok:
        raise typer.Exit(code=1)
    console.print("[green]MySQL 连接正常。[/]")


@db_app.command("import-jobs")
def db_import_jobs(
    source: Annotated[
        Path | None,
        typer.Option(
            "--from",
            help="旧版 jobs.json 路径（默认 <MINERU_OUTPUT_DIR>/jobs.json）",
        ),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="只解析校验，不写数据库")] = False,
) -> None:
    """把旧版 data/parsed/jobs.json 的任务记录迁移到 tb_document_parse_job。"""
    from ai_agent_py.db.document_jobs import MySQLJobStore
    from ai_agent_py.mineru.models import ParseJob

    cfg = get_settings()
    path = source or (cfg.mineru_output_dir / "jobs.json")
    if not path.is_file():
        raise _fail(f"找不到任务记录文件：{path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _fail(f"读取 {path} 失败：{exc}") from exc

    entries = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise _fail(f"{path} 结构不正确：顶层应包含 jobs 数组")

    jobs: list[ParseJob] = []
    skipped = 0
    for entry in entries:
        try:
            jobs.append(ParseJob.model_validate(entry))
        except Exception as exc:  # 单条损坏不影响其它记录
            skipped += 1
            console.print(f"[yellow]跳过损坏的记录：{exc}[/]")

    table = Table(title="待迁移任务", title_justify="left")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("文件名", overflow="fold")
    table.add_column("状态")
    table.add_column("创建时间（UTC）")
    for job in jobs:
        table.add_row(
            job.id, job.file_name, job.state_label, job.created_at.isoformat(timespec="seconds")
        )
    console.print(table)
    if skipped:
        console.print(f"[yellow]共跳过 {skipped} 条损坏记录。[/]")

    if dry_run:
        console.print(f"[yellow]--dry-run：解析到 {len(jobs)} 条记录，未写入数据库。[/]")
        return
    if not jobs:
        console.print("[yellow]没有可迁移的任务记录。[/]")
        return

    try:
        store = MySQLJobStore.from_settings(cfg)
        for job in jobs:
            store.save(job)
    except AiAgentError as exc:
        raise _fail(str(exc)) from exc

    console.print(
        f"[green]已迁移 {len(jobs)} 条任务记录到 tb_document_parse_job"
        "（按主键 upsert，可重复执行）。[/]"
    )


@app.command()
def ingest(
    paths: Annotated[
        list[str] | None,
        typer.Argument(help="文件或目录（可多个）；默认使用 DATA_DIR"),
    ] = None,
    chunk_size: Annotated[int | None, typer.Option("--chunk-size", help="分块大小（字符）")] = None,
    chunk_overlap: Annotated[
        int | None, typer.Option("--chunk-overlap", help="分块重叠（字符）")
    ] = None,
) -> None:
    """把文档切块、向量化并写入 Milvus ``rag_chunk``（按源文件覆盖旧切片）。"""
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
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} chunks"),
            console=console,
            transient=True,
        ) as progress:
            report = run_ingest(
                targets,
                settings=cfg,
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
        searcher = get_milvus_searcher(cfg)
    except AiAgentError as exc:
        raise _fail(str(exc)) from exc

    started = time.perf_counter()
    with console.status("[cyan]检索中…", spinner="dots"):
        documents = searcher.search(question, k=k)

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
        int | None, typer.Option("--max-turns", help="单轮最多模型调用次数（覆盖 AGENT_MAX_TURNS）")
    ] = None,
) -> None:
    """交互式对话（LangGraph agent，会调用检索工具）。"""
    from langgraph.checkpoint.memory import InMemorySaver

    cfg = get_settings()
    if max_turns is not None:
        # 预算在构图时注入：只改 recursion_limit 会让预算与上限对不上（A-02）。
        cfg = cfg.model_copy(update={"agent_max_turns": max(1, max_turns)})
    try:
        llm = get_chat_model(cfg)
        searcher = get_milvus_searcher(cfg)
    except AiAgentError as exc:
        raise _fail(str(exc)) from exc

    saver = InMemorySaver()
    graph = build_agent_graph(llm=llm, searcher=searcher, checkpointer=saver, settings=cfg)
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
                answer = ask_agent(graph, text, thread_id=thread_id, settings=cfg)
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
    offline_cfg = cfg.model_copy(update={"chunk_size": 500, "chunk_overlap": 80})
    searcher = OfflineChunkSearcher(LexicalHashEmbeddings())

    console.print(
        Panel("离线模式：词法哈希向量 + 占位模型，仅用于验证流程是否跑通。", border_style="blue")
    )

    try:
        with console.status("[cyan]加载、切块、向量化…", spinner="dots"):
            report = run_ingest([offline_cfg.data_dir], writer=searcher, settings=offline_cfg)
    except (AiAgentError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    if report.chunks == 0:
        raise _fail(f"data 目录中没有可索引的文档：{offline_cfg.data_dir}")
    _print_report(report)

    documents = searcher.search(question, k=3)
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
def parse(
    path: Annotated[
        Path,
        typer.Argument(help="待解析文件（pdf / docx / pptx / 图片）", exists=True, dir_okay=False),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="把 Markdown 另存到指定路径"),
    ] = None,
    model_version: Annotated[
        str, typer.Option(help="MinerU 模型版本：vlm / pipeline / MinerU-HTML")
    ] = "vlm",
) -> None:
    """调用 MinerU 精准解析 API 把文档转成 Markdown。"""
    import asyncio
    import shutil

    from ai_agent_py.mineru.models import JobState
    from ai_agent_py.mineru.service import DocumentService

    cfg = get_settings()
    if not cfg.resolved_mineru_api_key:
        raise _fail("未配置 MINERU_API_KEY，请在 .env 中填写 Token 后重试。")

    async def _chunks() -> Any:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                yield block

    async def _run() -> tuple[DocumentService, Any]:
        service = DocumentService(cfg)
        options = service.default_options().model_copy(update={"model_version": model_version})
        job = await service.create(path.name, _chunks(), options)
        console.print(f"已提交解析任务 [cyan]{job.id}[/]，模型 [cyan]{model_version}[/]…")
        try:
            finished = await service.wait(job.id, timeout=cfg.mineru_timeout + 120)
        finally:
            await service.shutdown()
        return service, finished

    try:
        service, job = asyncio.run(_run())
    except AiAgentError as exc:
        raise _fail(str(exc)) from exc

    if job.state is not JobState.done:
        raise _fail(job.err_msg or "解析失败")

    markdown_path = service.job_dir(job.id) / (job.markdown_file or "full.md")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(markdown_path, output)
        console.print(f"Markdown 已保存：[bold]{output}[/]")
    console.print(
        f"页数 [cyan]{job.total_pages or '-'}[/]，字符 [cyan]{job.markdown_chars}[/]，"
        f"图片 [cyan]{job.asset_count}[/]，耗时 [cyan]{job.elapsed_seconds}s[/]"
    )
    console.print(f"结果目录：[dim]{service.job_dir(job.id)}[/]")
    return None


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="监听地址")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="监听端口")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="开发模式热重载")] = False,
) -> None:
    """启动 FastAPI 服务（管理后台与 /docs 接口文档）。"""
    import uvicorn

    console.print(f"管理后台：http://{host}:{port}/")
    console.print(f"API 文档：http://{host}:{port}/docs")
    if not (PROJECT_ROOT / "web" / "dist" / "index.html").is_file():
        console.print(
            "[yellow]提示：前端尚未构建，执行 `cd web && pnpm install && pnpm build` 后刷新页面。[/]"
        )
    uvicorn.run("ai_agent_py.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":  # pragma: no cover
    app()
