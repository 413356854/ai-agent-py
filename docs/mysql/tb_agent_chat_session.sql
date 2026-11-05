-- 通用 Agent 会话表（「Agent 应用 → 通用 Agent」左侧会话列表）
-- 一行 = 一个会话；id 就是前端请求里的 thread_id（每轮对话都带它）。
-- 应用不会自动建表：首次使用前在 MySQL 客户端执行本脚本。
-- 配套的消息表见 docs/mysql/tb_agent_chat_message.sql。

CREATE TABLE IF NOT EXISTS `tb_agent_chat_session` (
  `id` VARCHAR(64) NOT NULL
    COMMENT '会话 ID（前端生成的 thread_id），与消息表 thread_id 对应',
  `agent_id` VARCHAR(32) NOT NULL DEFAULT 'general'
    COMMENT '应用标识：general 表示通用 Agent（其他 Agent 可复用同一套记忆表）',
  `title` VARCHAR(120) NOT NULL DEFAULT '新会话'
    COMMENT '会话标题：取第一句用户输入（最多 40 字，超出加省略号）',
  `message_count` INT UNSIGNED NOT NULL DEFAULT 0
    COMMENT '消息条数（含工具调用记录）',
  `created_at` DATETIME(6) NOT NULL
    COMMENT '创建时间（UTC）',
  `updated_at` DATETIME(6) NOT NULL
    COMMENT '最后一条消息时间（UTC），会话列表按它倒序',

  PRIMARY KEY (`id`),
  KEY `idx_agent_updated` (`agent_id`, `updated_at`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_general_ci
  COMMENT = '通用 Agent 会话（LangChain / LangGraph 多轮记忆）';
