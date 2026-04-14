from pydantic import BaseModel, field_validator


class Sample(BaseModel):
    """One instruction-following sample in Alpaca format."""

    instruction: str
    input: str = ""
    output: str

    @field_validator("instruction")
    @classmethod
    def instruction_nonempty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("instruction must be non-empty")
        return v

    @field_validator("output")
    @classmethod
    def output_min_length(cls, v: str) -> str:
        v = v.strip()
        sentences = [s.strip() for s in v.replace("\n", " ").split(".") if s.strip()]
        if len(sentences) < 3:
            raise ValueError("output must contain at least 3 sentences")
        return v

    @field_validator("input")
    @classmethod
    def input_normalized(cls, v: str) -> str:
        return v.strip()
