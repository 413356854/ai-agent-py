"""MySQL 访问层：连接池、健康检查，以及业务表的读写。

`health` 负责连接池与只读体检；`document_jobs` 是文档解析任务表
``tb_document_parse_job`` 的读写实现。建表脚本都在 ``docs/mysql/`` 下，
本包不做自动建表，避免应用去改生产库结构。
"""
