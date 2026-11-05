-- 通用 Agent 滚动摘要表（上下文压缩 L3）
-- 一行 = 一个会话的已压缩历史摘要；until_seq 表示摘要覆盖到消息表的哪一条（含）。
-- 原始消息仍完整保留在 tb_agent_chat_message，本表只影响发给模型的上下文。
-- 应用不会自动建表：首次使用前在 MySQL 客户端执行本脚本。

CREATE TABLE IF NOT EXISTS `tb_agent_chat_summary` (
  `thread_id` VARCHAR(64) NOT NULL
    COMMENT '会话 ID，对应 tb_agent_chat_session.id',
  `summary` MEDIUMTEXT NOT NULL
    COMMENT '结构化摘要（固定 8 个章节），会拼进系统提示词',
  `until_seq` INT UNSIGNED NOT NULL
    COMMENT '摘要覆盖到的消息序号（含）；seq <= 该值的 user/assistant 消息不再发送给模型',
  `source_messages` INT UNSIGNED NOT NULL
    COMMENT '生成摘要时压缩的历史消息条数（累计）',
  `tokens` INT UNSIGNED NULL
    COMMENT '摘要的估算 token 数',
  `model` VARCHAR(128) NULL
    COMMENT '生成摘要的模型名',
  `updated_at` DATETIME(6) NOT NULL
    COMMENT '摘要更新时间（UTC）',

  PRIMARY KEY (`thread_id`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_general_ci
  COMMENT = '通用 Agent 已压缩历史的摘要（滚动摘要）';
