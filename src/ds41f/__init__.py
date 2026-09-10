"""DS41F: a minimal serving engine for DeepSeek-V4.1-Flash.

Five responsibilities: api (codec/wire shapes), engine (single execution owner),
scheduler (static cohorts, pure metadata), backend (model adapters), state
(private dense rows). See README.md.
"""

__version__ = "0.1.0"
