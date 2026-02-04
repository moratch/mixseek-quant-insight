WITH RECURSIVE
-- パラメータ定義（trace_idはここで一箇所のみ指定）
params AS (
    SELECT '{{your_trace_id}}' AS trace_id
),

-- 1-4: leader_agent run を取得し、model_name ごとに通し番号を付ける
leader_agents AS (
    SELECT
        r.trace_id,
        r.span_id AS leader_span_id,
        r.attributes->>'model_name' AS leader_model_name,
        r.start_timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY r.attributes->>'model_name'
            ORDER BY r.start_timestamp
        ) AS round
    FROM records r
    CROSS JOIN params p
    WHERE r.trace_id = p.trace_id
      AND r.message = 'leader_agent run'
),

-- 5: leader_span_id 以下の全階層を再帰的に取得
descendant_spans AS (
    -- ベースケース: leader_span_id 直下
    SELECT
        r.trace_id,
        r.span_id,
        r.parent_span_id,
        r.message,
        r.attributes,
        r.start_timestamp,
        la.leader_span_id,
        la.leader_model_name,
        la.round
    FROM records r
    JOIN leader_agents la ON r.parent_span_id = la.leader_span_id
    CROSS JOIN params p
    WHERE r.trace_id = p.trace_id

    UNION ALL

    -- 再帰ケース: さらに下の階層
    SELECT
        r.trace_id,
        r.span_id,
        r.parent_span_id,
        r.message,
        r.attributes,
        r.start_timestamp,
        ds.leader_span_id,
        ds.leader_model_name,
        ds.round
    FROM records r
    JOIN descendant_spans ds ON r.parent_span_id = ds.span_id
    CROSS JOIN params p
    WHERE r.trace_id = p.trace_id
),

-- 6: running tool: で始まる span を取得
tool_spans AS (
    SELECT
        span_id AS tool_span_id,
        attributes->>'gen_ai.tool.call.id' AS tool_call_id,
        attributes->>'gen_ai.tool.name' AS tool_name,
        leader_span_id,
        leader_model_name,
        round
    FROM descendant_spans
    WHERE message LIKE 'running tool: %'
),

-- 7-8: tool_span_id を parent に持つ agent_run を取得
agent_runs AS (
    SELECT
        r.message,
        r.attributes->>'model_name' AS model_name,
        r.start_timestamp,
        r.attributes->>'pydantic_ai.all_messages' AS all_messages,
        ts.tool_span_id,
        ts.tool_call_id,
        ts.tool_name,
        ts.leader_span_id,
        ts.leader_model_name,
        ts.round
    FROM records r
    JOIN tool_spans ts ON r.parent_span_id = ts.tool_span_id
    CROSS JOIN params p
    WHERE r.trace_id = p.trace_id
      AND r.message = 'agent run'
)

-- 最終結果
SELECT
    round,
    leader_model_name,
    tool_name,
    tool_call_id,
    message,
    model_name,
    start_timestamp,
    all_messages
FROM agent_runs
ORDER BY leader_model_name, round, start_timestamp;
