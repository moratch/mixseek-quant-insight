WITH parent_spans AS (
  -- ステップ2&3: round_controller.run_roundのスパンを取得
  SELECT 
    trace_id,
    span_id,
    attributes->>'team_id' AS team_id
  FROM records
  WHERE trace_id = '{{your_trace_id}}'
    AND message = 'round_controller.run_round'
),

latest_agent_runs AS (
  -- ステップ4: 各parent_spanに対して、message='agent run'で最も遅い1行を抽出
  SELECT DISTINCT ON (p.span_id)
    p.trace_id,
    p.team_id,
    r.start_timestamp,
    r.attributes
  FROM records r
  INNER JOIN parent_spans p ON r.parent_span_id = p.span_id
  WHERE r.message = 'agent run'
  ORDER BY p.span_id, r.start_timestamp DESC
),

numbered_rounds AS (
  -- ステップ5: team_idごとにstart_timestamp順に通し番号（2始まり）を付ける
  SELECT 
    trace_id,
    team_id,
    (ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY start_timestamp) + 1) AS round,
    start_timestamp,
    attributes
  FROM latest_agent_runs
)

-- ステップ6: 最終出力
SELECT 
  trace_id,
  team_id,
  round,
  start_timestamp,
  attributes->'pydantic_ai.all_messages'->0->'parts'->0->>'content' AS content
FROM numbered_rounds
ORDER BY team_id, round;