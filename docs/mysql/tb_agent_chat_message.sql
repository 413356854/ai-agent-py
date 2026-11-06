-- 通用 Agent 会话消息表（每一轮对话的完整记录）
-- role = user      用户输入
--      = assistant 助手最终回答（进上下文）
--      = tool      一次工具调用记录（工具名 + 入参 + 结果，只用于界面回放，不进提示词）
-- (thread_id, seq) 唯一：seq 是会话内的消息序号，从 1 开始递增。
-- 应用不会自动建表：首次使用前在 MySQL 客户端执行本脚本。

CREATE TABLE IF NOT EXISTS `tb_agent_chat_message` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT
    COMMENT '自增主键（仅数据库内部使用）',
  `thread_id` VARCHAR(64) NOT NULL
    COMMENT '会话 ID，对应 tb_agent_chat_session.id',
  `seq` INT UNSIGNED NOT NULL
    COMMENT '会话内消息序号，从 1 开始递增',
  `role` VARCHAR(16) NOT NULL
    COMMENT '消息角色：user / assistant / tool',
  `content` MEDIUMTEXT NOT NULL
    COMMENT '消息正文；工具行是工具返回结果（超过 20000 字会截断）',

  `tool_name` VARCHAR(64) NULL
    COMMENT '工具名（role=tool 时有值），如 search_knowledge_base / read / shell',
  `tool_call_id` VARCHAR(128) NULL
    COMMENT '模型返回的工具调用 ID，用于把请求与结果对上',
  `tool_args` MEDIUMTEXT NULL
    COMMENT '工具入参 JSON（role=tool 时有值）',
  `tool_status` VARCHAR(16) NULL
    COMMENT '工具执行结果：ok / error / timeout / invalid_args / skipped（重复调用或超出预算被跳过）',
  `elapsed_ms` INT UNSIGNED NULL
    COMMENT '工具执行耗时（毫秒）；由工具节点写入 artifact.elapsed_ms',

  `created_at` DATETIME(6) NOT NULL
    COMMENT '写入时间（UTC）',

  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_thread_seq` (`thread_id`, `seq`),
  KEY `idx_thread_created` (`thread_id`, `created_at`),

  CONSTRAINT `chk_tb_agent_chat_message_role`
    CHECK (`role` IN ('user', 'assistant', 'tool'))
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_general_ci
  COMMENT = '通用 Agent 会话消息（用户 / 助手 / 工具调用）';
