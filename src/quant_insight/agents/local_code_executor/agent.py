"""LocalCodeExecutorAgent実装。

Pythonコード実行用カスタムMember Agent。
"""

import importlib
from typing import Any, cast

from mixseek.agents.member.base import BaseMemberAgent
from mixseek.models.member_agent import AgentType, MemberAgentConfig, MemberAgentResult, ResultStatus
from pydantic import BaseModel
from pydantic_ai import Agent

from quant_insight.agents.local_code_executor.models import ImplementationContext, LocalCodeExecutorConfig
from quant_insight.agents.local_code_executor.output_models import AnalyzerOutput, SubmitterOutput
from quant_insight.agents.local_code_executor.tools import local_code_executor_toolset
from quant_insight.storage import get_implementation_store


class LocalCodeExecutorAgent(BaseMemberAgent):
    """Pythonコード実行用カスタムMember Agent。"""

    def __init__(self, config: MemberAgentConfig) -> None:
        """Local Code Executorエージェントを初期化。

        Args:
            config: Member Agent設定。

        Raises:
            RuntimeError: DuckDBスキーマが初期化されていない場合。
        """
        super().__init__(config)

        # LocalCodeExecutorConfigを構築
        self.executor_config = self._build_executor_config(config)

        # DuckDBスキーマを検証
        self._verify_database_schema()

        # Pydantic AI Agentを初期化
        # Note: output_typeのデフォルトはstr、構造化出力が必要な場合のみ指定
        output_type = self._resolve_output_type()
        model_settings = self._create_model_settings()

        self.agent: Agent[LocalCodeExecutorConfig, Any] = Agent(
            model=self.config.model,
            deps_type=LocalCodeExecutorConfig,
            output_type=output_type,
            toolsets=[local_code_executor_toolset],  # type: ignore[list-item]
            instructions=self.config.system_instruction,
            model_settings=model_settings,
            retries=self.config.max_retries,
        )

    def _build_executor_config(self, config: MemberAgentConfig) -> LocalCodeExecutorConfig:
        """MemberAgentConfigからLocalCodeExecutorConfigを構築。

        Args:
            config: Member Agent設定。

        Returns:
            構築されたLocalCodeExecutorConfig。

        Raises:
            ValueError: 設定が不足している場合。
        """
        executor_settings = config.metadata.get("tool_settings", {}).get("local_code_executor", {})
        if not executor_settings:
            raise ValueError(
                "TOMLに[agent.metadata.tool_settings.local_code_executor]設定がありません。"
                "TOMLファイルにlocal code executor設定を追加してください。"
            )
        return LocalCodeExecutorConfig.model_validate(executor_settings)

    def _verify_database_schema(self) -> None:
        """DuckDBスキーマの存在を検証。

        Raises:
            RuntimeError: agent_implementationテーブルが存在しない場合。
        """
        store = get_implementation_store()
        if not store.table_exists():
            raise RuntimeError(
                "agent_implementationテーブルが存在しません。"
                "`quant-insight db init` を実行してスキーマを初期化してください。"
            )

    def _resolve_output_type(self) -> type[BaseModel] | type[str]:
        """構造化出力モデルを解決。

        Returns:
            構造化出力モデルクラス、または設定がない場合はstr。
        """
        output_model_config = self.executor_config.output_model
        if output_model_config:
            return self._load_output_model(
                output_model_config.module_path,
                output_model_config.class_name,
            )
        else:
            return str

    def _load_output_model(self, module_path: str, class_name: str) -> type[BaseModel]:
        """モジュールパスとクラス名から動的にモデルをロード。

        Args:
            module_path: モジュールパス（例: quant_insight.agents.local_code_executor.output_models）
            class_name: クラス名（例: AnalyzerOutput）

        Returns:
            ロードされたPydanticモデルクラス

        Raises:
            ImportError: モジュールのインポートに失敗
            AttributeError: クラスが見つからない
            TypeError: クラスがBaseModelのサブクラスでない
        """
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"モジュール '{module_path}' のインポートに失敗しました: {e}") from e

        try:
            model_class: Any = getattr(module, class_name)
        except AttributeError as e:
            raise AttributeError(f"モジュール '{module_path}' にクラス '{class_name}' が見つかりません: {e}") from e

        if not issubclass(model_class, BaseModel):
            raise TypeError(f"クラス '{class_name}' はBaseModelのサブクラスではありません")

        # 動的インポートなので、型チェッカーには型を伝えるためにcastを使用
        # 実行時にはissubclassで型チェック済み
        return cast(type[BaseModel], model_class)

    async def execute(self, task: str, context: dict[str, Any] | None = None, **kwargs: Any) -> MemberAgentResult:
        """エージェントタスクを実行。

        Args:
            task: Leader Agentからのタスク説明。
            context: オプションの実行コンテキスト。
                     execution_id, team_id, round_number が含まれている場合(leaderから呼ばれる場合)、
                     DuckDBへのスクリプト保存が有効になります。
            **kwargs: 追加の実行パラメータ。

        Returns:
            実行結果を含むMemberAgentResult。
        """
        # 基底クラスのシグネチャに合わせるため、未使用パラメータを明示的に無視
        _ = kwargs

        # contextからImplementationContextを構築
        if context is not None:
            self.executor_config.implementation_context = ImplementationContext(
                execution_id=context.get("execution_id", ""),
                team_id=context.get("team_id", ""),
                round_number=context.get("round_number", 0),
                member_agent_name=self.config.name,
            )

        try:
            # 既存スクリプト情報をtaskのフッタに追加
            enriched_task = await self._enrich_task_with_existing_scripts(task)

            # Pydantic AI Agentを実行
            result = await self.agent.run(enriched_task, deps=self.executor_config)

            # メッセージ履歴を取得（FR-034: Member Agent message history）
            all_messages = result.all_messages()

            # 構造化出力からコードを自動保存
            if self.executor_config.implementation_context is not None:
                await self._save_output_scripts(result.output)

            # 構造化出力をJSON文字列にシリアライズ
            if isinstance(result.output, BaseModel):
                content = result.output.model_dump_json(indent=2)
            else:
                # str出力の場合はそのまま使用
                content = result.output

            return MemberAgentResult(
                status=ResultStatus.SUCCESS,
                content=content,
                agent_name=self.config.name,
                agent_type=str(AgentType.CUSTOM),
                all_messages=all_messages,
            )

        except Exception as e:
            return MemberAgentResult(
                status=ResultStatus.ERROR,
                content=f"タスク実行エラー: {e!s}",
                agent_name=self.config.name,
                agent_type=str(AgentType.CUSTOM),
                error_message=str(e),
            )

    async def _enrich_task_with_existing_scripts(self, task: str) -> str:
        """既存スクリプト情報をtaskの末尾に追加。

        Args:
            task: 元のタスク文字列。

        Returns:
            既存スクリプト情報が追加されたタスク文字列。
        """
        impl_ctx = self.executor_config.implementation_context
        if impl_ctx is None:
            return task

        store = get_implementation_store()
        existing_scripts = await store.list_scripts(
            execution_id=impl_ctx.execution_id,
            team_id=impl_ctx.team_id,
            round_number=impl_ctx.round_number,
        )

        if not existing_scripts:
            return task

        file_names = [s["file_name"] for s in existing_scripts]
        footer = f"\n\n---\n既存スクリプト: {', '.join(file_names)}"
        return task + footer

    async def _save_output_scripts(self, output: Any) -> None:
        """構造化出力からスクリプトを抽出してDuckDBに保存。

        Args:
            output: エージェントの構造化出力。
        """
        impl_ctx = self.executor_config.implementation_context
        if impl_ctx is None:
            return

        store = get_implementation_store()

        if isinstance(output, AnalyzerOutput):
            for script in output.scripts:
                await store.save_script(
                    execution_id=impl_ctx.execution_id,
                    team_id=impl_ctx.team_id,
                    round_number=impl_ctx.round_number,
                    member_agent_name=impl_ctx.member_agent_name,
                    file_name=script.file_name,
                    code=script.code,
                )
        elif isinstance(output, SubmitterOutput):
            # submissionフィールドを保存
            await store.save_script(
                execution_id=impl_ctx.execution_id,
                team_id=impl_ctx.team_id,
                round_number=impl_ctx.round_number,
                member_agent_name=impl_ctx.member_agent_name,
                file_name="submission.py",
                code=output.submission,
            )
