-- Agent 运行记录表（人工审批 / HITL 状态机）
-- 一次「运行」= 一次可能停在审批点的图执行；报销单复核与破坏性工具审批共用本表。
--
-- status 取值（详见 src/ai_agent_py/runs/models.py）：
--   running    正在执行
--   pending    停在审批节点，等人拍板（图状态在 AGENT_RUN_CHECKPOINT_PATH 的 SQLite 检查点里）
--   escalated  超时未审批，已升级并延长了宽限期
--   approved   人工通过，已继续执行到结束
--   rejected   人工拒绝，单据退回
--   timeout    超时未审批，已自动拒绝
--   cancelled  人工取消
--   completed  没有审批点，直接跑完
--   failed     执行异常
--
-- payload 是 JSON：{"request": 审批请求, "events": [状态流转记录]}
-- result  是 JSON：跑完后的结论 / Finding / 节点轨迹（结构与 /run 的试跑结果一致）。
-- 应用不会自动建表：首次使用前在 MySQL 客户端执行本脚本。

CREATE TABLE IF NOT EXISTS `tb_agent_run` (
  `run_id` VARCHAR(64) NOT NULL
    COMMENT '运行 ID（UUID hex），也是 /runs/{id} 的路径参数',
  `agent_id` VARCHAR(32) NOT NULL
    COMMENT '应用 ID，如 reimburse',
  `thread_id` VARCHAR(64) NOT NULL
    COMMENT 'LangGraph 检查点线程 ID（默认等于 run_id）',
  `status` VARCHAR(16) NOT NULL
    COMMENT '运行状态：running / pending / escalated / approved / rejected / timeout / cancelled / completed / failed',
  `title` VARCHAR(200) NOT NULL DEFAULT ''
    COMMENT '列表展示标题（报销单号等）',
  `payload` MEDIUMTEXT NULL
    COMMENT '审批请求与状态流转记录（JSON：request / events）',
  `result` MEDIUMTEXT NULL
    COMMENT '运行结果（JSON：结论 / Finding / 节点轨迹 / 审批记录）',
  `approver` VARCHAR(64) NULL
    COMMENT '审批人（超时自动拒绝时为「系统（超时）」）',
  `comment` VARCHAR(500) NULL
    COMMENT '审批意见',
  `error` VARCHAR(500) NULL
    COMMENT '失败原因（status=failed 时有值）',
  `created_at` DATETIME(6) NOT NULL
    COMMENT '创建时间（UTC）',
  `updated_at` DATETIME(6) NOT NULL
    COMMENT '最近一次状态变化时间（UTC）',
  `expires_at` DATETIME(6) NULL
    COMMENT '审批截止时间（UTC）；NULL 表示不限时',
  `finished_at` DATETIME(6) NULL
    COMMENT '结束时间（UTC）；未结束时为 NULL',

  PRIMARY KEY (`run_id`),
  KEY `idx_agent_status_updated` (`agent_id`, `status`, `updated_at`),
  KEY `idx_status_expires` (`status`, `expires_at`),

  CONSTRAINT `chk_tb_agent_run_status`
    CHECK (`status` IN ('running', 'pending', 'escalated', 'approved', 'rejected',
                        'timeout', 'cancelled', 'completed', 'failed'))
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_general_ci
  COMMENT = 'Agent 运行与人工审批（HITL）状态机';
