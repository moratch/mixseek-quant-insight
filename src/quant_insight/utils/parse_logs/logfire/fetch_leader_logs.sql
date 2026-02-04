SELECT
    trace_id,
    span_id,
    ROW_NUMBER() OVER (
        PARTITION BY attributes->>'model_name' 
        ORDER BY start_timestamp
    ) AS round,
    message,
    attributes->>'model_name' AS model_name,
    attributes->>'gen_ai.system_instructions' AS system_instructions,
    start_timestamp,
    attributes->>'pydantic_ai.all_messages' AS all_messages
FROM records
WHERE 
    trace_id = '{{your_trace_id}}'
    AND message = 'leader_agent run'
ORDER BY attributes->>'model_name', start_timestamp;