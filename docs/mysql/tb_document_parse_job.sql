
CREATE TABLE IF NOT EXISTS `tb_document_parse_job` (
  `id` VARCHAR(64) NOT NULL
    COMMENT '任务 ID（uuid4 hex），同时是 data/parsed/<id>/ 目录名',

  `file_name` VARCHAR(255) NOT NULL
    COMMENT '原始文件名（已清洗非法字符，最长 120 字符）',
  `file_size` BIGINT UNSIGNED NOT NULL DEFAULT 0
    COMMENT '源文件字节数',
  `suffix` VARCHAR(16) NOT NULL DEFAULT ''
    COMMENT '扩展名，不含点，如 pdf / docx / png',

  `state` VARCHAR(16) NOT NULL DEFAULT 'queued'
    COMMENT '任务状态：queued/uploading/parsing/downloading/done/failed',

  `model_version` VARCHAR(32) NOT NULL DEFAULT 'vlm'
    COMMENT 'MinerU 模型版本：vlm / pipeline / MinerU-HTML',
  `language` VARCHAR(32) NOT NULL DEFAULT 'ch'
    COMMENT 'OCR 语言，如 ch / en / chinese_cht',
  `enable_formula` TINYINT(1) NOT NULL DEFAULT 1
    COMMENT '是否识别公式：1 是，0 否',
  `enable_table` TINYINT(1) NOT NULL DEFAULT 1
    COMMENT '是否识别表格：1 是，0 否',
  `is_ocr` TINYINT(1) NOT NULL DEFAULT 0
    COMMENT '是否按扫描件走 OCR：1 是，0 否',
  `page_ranges` VARCHAR(64) NULL
    COMMENT '页码范围，如 "1-10,12"；NULL 表示全文',

  `batch_id` VARCHAR(64) NULL
    COMMENT 'MinerU 批量上传返回的 batch_id，断点续跑时复用',
  `err_msg` VARCHAR(500) NOT NULL DEFAULT ''
    COMMENT '失败原因（最多 500 字符），成功时为空串',

  `extracted_pages` INT UNSIGNED NOT NULL DEFAULT 0
    COMMENT '已解析页数',
  `total_pages` INT UNSIGNED NOT NULL DEFAULT 0
    COMMENT '总页数，0 表示未知',

  `source_file` VARCHAR(255) NULL
    COMMENT '上传的源文件名，如 source.pdf',
  `markdown_file` VARCHAR(255) NULL
    COMMENT '解析出的 Markdown 文件名，如 full.md',
  `zip_file` VARCHAR(255) NULL
    COMMENT 'MinerU 返回结果包文件名，如 mineru.zip',
  `markdown_chars` INT UNSIGNED NOT NULL DEFAULT 0
    COMMENT 'Markdown 字符数',
  `asset_count` INT UNSIGNED NOT NULL DEFAULT 0
    COMMENT '结果里的图片等附件数量',

  `ingest_chunks` INT UNSIGNED NULL
    COMMENT '写入 Chroma 的 chunk 数；NULL 表示未入库',
  `ingest_characters` INT UNSIGNED NULL
    COMMENT '入库字符数',
  `ingest_collection` VARCHAR(128) NULL
    COMMENT '向量库集合名',
  `ingest_store` VARCHAR(64) NULL
    COMMENT '向量库实现，如 Chroma / InMemoryVectorStore',
  `ingest_source` VARCHAR(512) NULL
    COMMENT '入库来源（Markdown 绝对路径）',
  `ingested_at` DATETIME(6) NULL
    COMMENT 'Chroma 入库时间（UTC）',

  `rag_ingest_chunks` INT UNSIGNED NULL
    COMMENT '写入 Milvus rag_chunk 的 chunk 数；NULL 表示未入库 RAG',
  `rag_ingest_characters` INT UNSIGNED NULL
    COMMENT 'RAG 入库字符数',
  `rag_ingest_collection` VARCHAR(128) NULL
    COMMENT 'RAG 目标集合，如 db_tax_rag.rag_chunk',
  `rag_ingested_at` DATETIME(6) NULL
    COMMENT 'RAG 入库时间（UTC）；非 NULL 即「已入库」',

  `created_at` DATETIME(6) NOT NULL
    COMMENT '任务创建时间（UTC）',
  `updated_at` DATETIME(6) NOT NULL
    COMMENT '最后一次状态更新时间（UTC），由应用写入',
  `started_at` DATETIME(6) NULL
    COMMENT '开始调用 MinerU 的时间（UTC）',
  `finished_at` DATETIME(6) NULL
    COMMENT '结束时间（UTC）',
  `elapsed_seconds` DECIMAL(10,2) NULL
    COMMENT '耗时（秒），保留两位小数',

  PRIMARY KEY (`id`),
  KEY `idx_created_at` (`created_at`),
  KEY `idx_state_created` (`state`, `created_at`),
  KEY `idx_batch_id` (`batch_id`),
  KEY `idx_file_name` (`file_name`),

  CONSTRAINT `chk_tb_document_parse_job_state`
    CHECK (`state` IN ('queued', 'uploading', 'parsing', 'downloading', 'done', 'failed'))
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_general_ci
  COMMENT = '文档解析任务记录';
