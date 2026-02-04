"""プレロードモジュール設定のPydanticモデル。

Article 6（型安全性）に準拠。
"""

from pydantic import BaseModel, ConfigDict, Field


class PreloadedModule(BaseModel):
    """プレimportモジュール設定。

    Attributes:
        name: モジュール名（例: "polars", "sklearn.preprocessing"）
        alias: import時のエイリアス（省略可、例: "pl", "np"）
    """

    name: str = Field(description="モジュール名")
    alias: str | None = Field(default=None, description="import時のエイリアス（省略可）")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"name": "polars", "alias": "pl"},
                {"name": "numpy", "alias": "np"},
                {"name": "sklearn"},  # aliasなし
            ]
        }
    )


__all__ = ["PreloadedModule"]
